from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from ..registry import Registry


VALID_STATUSES = {"native", "derived", "diagnostic", "artifact_required"}
VALID_EXPERIMENT_KEYS = {
    "id",
    "label",
    "status",
    "studies",
    "protocols",
    "outputs",
    "reason",
}


def load_paper_catalog(registry: Registry) -> Dict[str, Any]:
    path = registry.root / "configs" / "experiments" / "paper_catalog.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Invalid paper experiment catalog: {}".format(path))
    papers = payload.get("papers", {})
    for paper_id, paper in papers.items():
        method = str(paper.get("method", ""))
        if method not in registry.attacks:
            raise ValueError(
                "Paper catalog '{}' references unknown method '{}'".format(
                    paper_id, method
                )
            )
        for experiment in paper.get("experiments", []):
            unknown_keys = sorted(set(experiment) - VALID_EXPERIMENT_KEYS)
            if unknown_keys:
                raise ValueError(
                    "Paper catalog '{}.{}' has unknown fields: {}".format(
                        paper_id,
                        experiment.get("id"),
                        ", ".join(unknown_keys),
                    )
                )
            status = str(experiment.get("status", ""))
            if status not in VALID_STATUSES:
                raise ValueError(
                    "Paper catalog '{}.{}' has invalid status '{}'".format(
                        paper_id, experiment.get("id"), status
                    )
                )
            unknown = sorted(
                set(experiment.get("studies", []))
                - set(registry.ablation_studies)
            )
            if unknown:
                raise ValueError(
                    "Paper catalog '{}.{}' references unknown studies: {}".format(
                        paper_id,
                        experiment.get("id"),
                        ", ".join(unknown),
                    )
                )
            unknown_protocols = sorted(
                set(experiment.get("protocols", [])) - set(registry.protocols)
            )
            if unknown_protocols:
                raise ValueError(
                    "Paper catalog '{}.{}' references unknown protocols: {}".format(
                        paper_id,
                        experiment.get("id"),
                        ", ".join(unknown_protocols),
                    )
                )
    return payload


def _escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _rows(catalog: Mapping[str, Any]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for paper_id, paper in catalog["papers"].items():
        for experiment in paper.get("experiments", []):
            rows.append(
                {
                    "paper_id": str(paper_id),
                    "method": str(paper["method"]),
                    "paper": str(paper["title"]),
                    "experiment_id": str(experiment["id"]),
                    "experiment": str(experiment["label"]),
                    "status": str(experiment["status"]),
                    "studies": ",".join(experiment.get("studies", [])),
                    "protocols": ",".join(experiment.get("protocols", [])),
                    "outputs": ",".join(experiment.get("outputs", [])),
                    "reason": str(experiment.get("reason", "")),
                }
            )
    return rows


def latex_paper_catalog(catalog: Mapping[str, Any]) -> str:
    rows = _rows(catalog)
    status = {
        "native": "N",
        "derived": "D",
        "diagnostic": "DG",
        "artifact_required": "AR",
    }
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{Coverage of paper experiments beyond the main transfer table. "
        r"N is directly executable, D is derived from saved runs, DG is a disclosed "
        r"local diagnostic, and AR requires an external artifact or protocol.}",
        r"\label{tab:paper_experiment_coverage}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l c l l}",
        r"\toprule",
        r"Method & Experiment & Status & Runnable study/protocol & Limitation \\",
        r"\midrule",
    ]
    previous = None
    for row in rows:
        method = row["method"] if row["method"] != previous else ""
        runnable = row["studies"] or row["protocols"] or row["outputs"]
        lines.append(
            "{} & {} & {} & {} & {} \\\\".format(
                _escape(method),
                _escape(row["experiment"]),
                status[row["status"]],
                _escape(runnable or "external artifact"),
                _escape(row["reason"] or "N/A"),
            )
        )
        previous = row["method"]
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def write_paper_catalog(registry: Registry, output_dir: Path) -> Dict[str, Path]:
    catalog = load_paper_catalog(registry)
    rows = _rows(catalog)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_experiment_coverage.csv"
    tex_path = output_dir / "paper_experiment_coverage.tex"
    md_path = output_dir / "paper_experiment_coverage.md"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tex_path.write_text(latex_paper_catalog(catalog), encoding="utf-8")
    markdown = [
        "# Paper experiment coverage",
        "",
        "| Method | Experiment | Status | Runnable IDs | Reason |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        runnable = row["studies"] or row["protocols"] or row["outputs"]
        markdown.append(
            "| {} | {} | {} | {} | {} |".format(
                row["method"],
                row["experiment"].replace("|", r"\|"),
                row["status"],
                (runnable or "external artifact").replace("|", r"\|"),
                (row["reason"] or "N/A").replace("|", r"\|"),
            )
        )
    md_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"csv": csv_path, "tex": tex_path, "markdown": md_path}
