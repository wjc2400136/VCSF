from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .formatting import METRIC_DECIMAL_PLACES, format_metric
from .scope import SCOPE_FIELDS, scope_csv, scope_label, scoped_tex


COCO_BBOX_SUMMARY_SPECS = (
    ("Average Precision", "AP", "0.50:0.95", "all", 100, 0),
    ("Average Precision", "AP", "0.50", "all", 100, 1),
    ("Average Precision", "AP", "0.75", "all", 100, 2),
    ("Average Precision", "AP", "0.50:0.95", "small", 100, 3),
    ("Average Precision", "AP", "0.50:0.95", "medium", 100, 4),
    ("Average Precision", "AP", "0.50:0.95", "large", 100, 5),
    ("Average Recall", "AR", "0.50:0.95", "all", 1, 6),
    ("Average Recall", "AR", "0.50:0.95", "all", 10, 7),
    ("Average Recall", "AR", "0.50:0.95", "all", 100, 8),
    ("Average Recall", "AR", "0.50:0.95", "small", 100, 9),
    ("Average Recall", "AR", "0.50:0.95", "medium", 100, 10),
    ("Average Recall", "AR", "0.50:0.95", "large", 100, 11),
)


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


def coco_bbox_summary_rows(stats: Sequence[float]) -> List[Dict[str, Any]]:
    if len(stats) < len(COCO_BBOX_SUMMARY_SPECS):
        raise ValueError(
            "COCO bbox summary needs 12 statistics, received {}".format(
                len(stats)
            )
        )
    return [
        {
            "order": order,
            "measure": measure,
            "metric": metric,
            "iou": iou,
            "area": area,
            "max_dets": max_dets,
            "value": float(stats[index]),
        }
        for order, (
            measure,
            metric,
            iou,
            area,
            max_dets,
            index,
        ) in enumerate(COCO_BBOX_SUMMARY_SPECS, start=1)
    ]


def coco_bbox_summary_text(rows: Sequence[Dict[str, Any]], report_scope=None) -> str:
    lines = [
        "COCO-style bbox evaluation summary ({} decimal places)".format(
            METRIC_DECIMAL_PLACES
        )
    ]
    if scope_label(report_scope):
        lines.insert(0, scope_label(report_scope))
    for row in rows:
        lines.append(
            " {:<18} ({}) @[ IoU={:>9} | area={:>6} | maxDets={:>3} ] = {}".format(
                row["measure"],
                row["metric"],
                row["iou"],
                row["area"],
                row["max_dets"],
                format_metric(row["value"]),
            )
        )
    return "\n".join(lines)


def coco_bbox_summary_tex(
    rows: Sequence[Dict[str, Any]],
    *,
    dataset: str,
    target: str,
    source: str,
    attack: str,
    report_scope=None,
) -> str:
    label = "tab:{}_{}_{}_{}_coco_summary".format(
        dataset,
        target,
        source,
        attack,
    )
    label = "".join(
        character if character.isalnum() or character in ":._-" else "_"
        for character in label
    )
    lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        (
            r"\caption{COCO-style bounding-box evaluation summary for "
            + _escape(target)
            + r" on "
            + _escape(dataset.upper())
            + r" ("
            + _escape(source)
            + r"/"
            + _escape(attack)
            + r").}"
        ),
        r"\label{" + label + "}",
        r"\begin{tabular}{l c c c r c}",
        r"\toprule",
        r"Measure & Metric & IoU & Area & maxDets & Value \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\\\".format(
                _escape(str(row["measure"])),
                _escape(str(row["metric"])),
                _escape(str(row["iou"])),
                _escape(str(row["area"])),
                row["max_dets"],
                format_metric(row["value"]),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return scoped_tex("\n".join(lines), report_scope)


def write_coco_bbox_summary(
    stats: Sequence[float],
    output_dir: Path,
    *,
    dataset: str,
    target: str,
    source: str,
    attack: str,
    report_scope=None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = coco_bbox_summary_rows(stats)
    text = coco_bbox_summary_text(rows, report_scope)
    text_path = output_dir / "coco_summary.txt"
    csv_path = output_dir / "coco_summary.csv"
    tex_path = output_dir / "coco_summary.tex"
    text_path.write_text(text + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order",
                "measure",
                "metric",
                "iou",
                "area",
                "max_dets",
                "value",
            ] + (list(SCOPE_FIELDS) if scope_label(report_scope) else []),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "value": format_metric(row["value"]),
                    **(scope_csv(report_scope) if scope_label(report_scope) else {}),
                }
            )
    tex_path.write_text(
        coco_bbox_summary_tex(
            rows,
            dataset=dataset,
            target=target,
            source=source,
            attack=attack,
            report_scope=report_scope,
        ),
        encoding="utf-8",
    )
    return {
        "decimal_places": METRIC_DECIMAL_PLACES,
        "rows": rows,
        "text": text,
        "text_path": text_path,
        "csv_path": csv_path,
        "tex_path": tex_path,
    }
