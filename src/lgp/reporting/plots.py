from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..registry import Registry
from .scope import escape_scope, inferred_scope, scope_label


def _safe_label(value: str, fallback: int) -> str:
    label = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return label or "experiment_{}".format(fallback)


def _plain_model_code(registry: Registry, model_id: str) -> str:
    model = registry.model(model_id)
    suffix = ("†" if model.source else "") + ("‡" if model.held_out else "")
    return model.table_code + suffix


def _group_boundaries(registry: Registry, targets: Sequence[str]) -> List[int]:
    return [
        index
        for index in range(1, len(targets))
        if registry.model_group(targets[index - 1]).id
        != registry.model_group(targets[index]).id
    ]


def _metric_rows(
    records: Iterable[Mapping[str, Any]], dataset: str, metric: str
) -> List[Mapping[str, Any]]:
    rows = []
    for record in records:
        value = (record.get("metrics") or {}).get(metric)
        if record.get("dataset") != dataset or record.get("attack") == "clean":
            continue
        if value is not None and math.isfinite(float(value)):
            rows.append(record)
    return rows


def _save(figure: plt.Figure, stem: Path, report_scope=None) -> Tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    if scope_label(report_scope):
        figure.text(0.5, -0.04, scope_label(report_scope), ha="center", va="top", fontsize=8)
    figure.savefig(str(png), dpi=220, bbox_inches="tight")
    figure.savefig(str(pdf), bbox_inches="tight")
    plt.close(figure)
    return png, pdf


def _coverage_plot(
    records: Sequence[Mapping[str, Any]], output_dir: Path, report_scope=None
) -> Tuple[Path, Path]:
    counts = Counter(str(record.get("status", "unknown")) for record in records)
    labels = sorted(counts)
    figure, axis = plt.subplots(figsize=(max(5.5, len(labels) * 1.1), 3.6))
    bars = axis.bar(labels, [counts[label] for label in labels], color="#4472C4")
    axis.bar_label(bars, padding=3)
    axis.set_ylabel("Jobs / records")
    axis.set_title("Experiment coverage")
    axis.tick_params(axis="x", rotation=25)
    axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, output_dir / "coverage", report_scope)


def _method_plot(
    rows: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    metric: str,
    method_ids: Sequence[str],
    report_scope=None,
) -> Tuple[Path, Path]:
    values: Dict[str, List[float]] = defaultdict(list)
    for record in rows:
        values[str(record["attack"])].append(float(record["metrics"][metric]))
    methods = [method for method in method_ids if method in values]
    means = [float(np.mean(values[method])) for method in methods]
    errors = [
        float(np.std(values[method], ddof=1) / math.sqrt(len(values[method])))
        if len(values[method]) > 1
        else 0.0
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(max(6.0, len(methods) * 0.8), 4.0))
    names = [registry.attack(method).display_name for method in methods]
    axis.bar(names, means, yerr=errors, capsize=3, color="#5B9BD5")
    axis.set_ylabel(metric)
    axis.set_title("Mean attacked detection performance (lower is stronger)")
    axis.tick_params(axis="x", rotation=30)
    axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, output_dir / "method_mean_{}".format(metric), report_scope)


def _heatmap_plot(
    rows: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    metric: str,
    method_ids: Sequence[str],
    report_scope=None,
) -> Tuple[Path, Path]:
    methods = [
        method
        for method in method_ids
        if any(str(row.get("attack")) == method for row in rows)
    ]
    targets = [
        target
        for target in registry.target_ids()
        if any(str(row.get("target")) == target for row in rows)
    ]
    grouped: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["attack"]), str(row["target"]))].append(
            float(row["metrics"][metric])
        )
    matrix = np.full((len(methods), len(targets)), np.nan, dtype=np.float64)
    for method_index, method in enumerate(methods):
        for target_index, target in enumerate(targets):
            values = grouped.get((method, target), [])
            if values:
                matrix[method_index, target_index] = np.mean(values)
    figure, axis = plt.subplots(
        figsize=(max(7.0, len(targets) * 0.55), max(3.5, len(methods) * 0.5))
    )
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(
        range(len(targets)),
        [_plain_model_code(registry, value) for value in targets],
    )
    axis.set_yticks(range(len(methods)), [registry.attack(value).display_name for value in methods])
    axis.tick_params(axis="x", rotation=40)
    for boundary in _group_boundaries(registry, targets):
        axis.axvline(boundary - 0.5, color="white", linewidth=1.2)
    axis.set_title("Mean transfer {}".format(metric))
    figure.colorbar(image, ax=axis, shrink=0.8)
    return _save(figure, output_dir / "transfer_heatmap_{}".format(metric), report_scope)


def _object_size_plot(
    rows: Sequence[Mapping[str, Any]], registry: Registry, output_dir: Path,
    method_ids: Sequence[str],
    report_scope=None,
) -> Tuple[Path, Path] | None:
    keys = ("bbox_mAP_small", "bbox_mAP_medium", "bbox_mAP_large")
    grouped: Dict[str, List[List[float]]] = defaultdict(lambda: [[], [], []])
    for row in rows:
        for index, key in enumerate(keys):
            value = (row.get("metrics") or {}).get(key)
            if value is not None and math.isfinite(float(value)):
                grouped[str(row["attack"])][index].append(float(value))
    methods = [method for method in method_ids if method in grouped]
    if not methods:
        return None
    values = np.asarray(
        [
            [float(np.mean(grouped[method][index])) for index in range(3)]
            for method in methods
        ]
    )
    figure, axis = plt.subplots(figsize=(max(6.5, len(methods) * 0.8), 4.0))
    x = np.arange(len(methods))
    width = 0.25
    for index, label in enumerate(("small", "medium", "large")):
        axis.bar(x + (index - 1) * width, values[:, index], width, label=label)
    axis.set_xticks(x, [registry.attack(value).display_name for value in methods], rotation=30)
    axis.set_ylabel("COCO AP")
    axis.set_title("Object-size transfer breakdown")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, output_dir / "object_size", report_scope)


def _quality_efficiency_plot(
    rows: Sequence[Mapping[str, Any]], registry: Registry, output_dir: Path, report_scope=None
) -> Tuple[Path, Path] | None:
    points: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    for row in rows:
        quality = row.get("attack_quality_mean") or {}
        runtime = row.get("attack_runtime_seconds_mean")
        ap50 = (row.get("metrics") or {}).get("bbox_mAP_50")
        ssim = quality.get("ssim")
        if all(value is not None and math.isfinite(float(value)) for value in (runtime, ap50, ssim)):
            points[str(row["attack"])].append((float(runtime), float(ap50), float(ssim)))
    if not points:
        return None
    figure, axis = plt.subplots(figsize=(6.4, 4.5))
    for method, values in points.items():
        mean = np.mean(np.asarray(values), axis=0)
        axis.scatter(mean[0], mean[1], s=40 + 180 * mean[2], label=registry.attack(method).display_name)
    axis.set_xscale("log")
    axis.set_xlabel("Attack seconds / image (log scale)")
    axis.set_ylabel("Attacked AP@.50 (lower is stronger)")
    axis.set_title("Strength, quality and efficiency")
    axis.legend(frameon=False, fontsize=8)
    axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, output_dir / "quality_efficiency", report_scope)


def _diagnostic_plot(
    records: Sequence[Mapping[str, Any]], output_dir: Path, report_scope=None
) -> Tuple[Path, Path] | None:
    losses: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    linf: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    seen_runs = set()
    for record in records:
        run_value = record.get("adversarial_run")
        if not run_value or str(run_value) in seen_runs:
            continue
        seen_runs.add(str(run_value))
        attack = str(record.get("attack", ""))
        manifest = Path(str(run_value)) / "manifest.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            for diagnostic in item.get("diagnostics", []):
                step = int(diagnostic.get("step", 0))
                value = diagnostic.get("loss")
                if value is not None and math.isfinite(float(value)):
                    losses[attack][step].append(float(value))
                value = diagnostic.get("linf_pixel")
                if value is not None and math.isfinite(float(value)):
                    linf[attack][step].append(float(value))
    if not losses:
        return None
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for attack, by_step in losses.items():
        steps = sorted(by_step)
        axes[0].plot(steps, [np.mean(by_step[step]) for step in steps], label=attack)
    axes[0].set_xlabel("Outer iteration")
    axes[0].set_ylabel("Recorded objective")
    axes[0].set_title("Optimization convergence")
    for attack, by_step in linf.items():
        steps = sorted(by_step)
        axes[1].plot(steps, [np.mean(by_step[step]) for step in steps], label=attack)
    axes[1].set_xlabel("Outer iteration")
    axes[1].set_ylabel(r"$L_\infty$ (pixel units)")
    axes[1].set_title("Constraint trajectory")
    axes[0].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    return _save(figure, output_dir / "convergence", report_scope)


def _figures_tex(paths: Sequence[Path], output_dir: Path, report_scope=None) -> Path:
    tex = output_dir / "figures.tex"
    captions = {
        "coverage": "Experiment execution coverage by record status.",
        "method_mean_bbox_mAP_50": "Mean attacked AP@.50 by attack method.",
        "transfer_heatmap_bbox_mAP_50": "Transfer AP@.50 across attack methods and target detectors.",
        "object_size": "Attack performance by small, medium and large object size.",
        "quality_efficiency": "Attack strength, perceptual quality and runtime efficiency.",
        "convergence": "Optimization objective and perturbation-constraint trajectories.",
    }
    lines: List[str] = []
    for index, path in enumerate(paths, start=1):
        relative = path.relative_to(output_dir).as_posix()
        caption = captions.get(
            path.stem,
            "Experiment figure generated from the recorded benchmark artifacts.",
        )
        if scope_label(report_scope):
            caption = escape_scope(scope_label(report_scope)) + ". " + caption
        lines.extend(
            [
                r"\begin{figure*}[!htb]",
                r"\centering",
                r"\includegraphics[width=0.92\textwidth]{\detokenize{" + relative + "}}",
                r"\caption{" + caption + "}",
                r"\label{fig:" + _safe_label(path.stem, index) + "}",
                r"\end{figure*}",
                "",
            ]
        )
    tex.write_text("\n".join(lines), encoding="utf-8")
    return tex


def write_experiment_plots(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    dataset: str,
    metric: str = "bbox_mAP_50",
    method_ids: Sequence[str] | None = None,
    report_scope=None,
    include_saved_diagnostics: bool = True,
) -> Dict[str, Path]:
    """Write publication PNG/PDF figures and a LaTeX include file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    included_pngs: List[Path] = []
    report_scope = report_scope or inferred_scope(records)
    png, pdf = _coverage_plot(records, output_dir, report_scope)
    paths.update(coverage_png=png, coverage_pdf=pdf)
    included_pngs.append(png)
    rows = _metric_rows(records, dataset, metric)
    ordered_methods = list(method_ids or registry.attacks)
    if rows:
        for name, result in (
            (
                "method",
                _method_plot(
                    rows, registry, output_dir, metric, ordered_methods, report_scope
                ),
            ),
            (
                "heatmap",
                _heatmap_plot(
                    rows, registry, output_dir, metric, ordered_methods, report_scope
                ),
            ),
        ):
            png, pdf = result
            paths[name + "_png"] = png
            paths[name + "_pdf"] = pdf
            included_pngs.append(png)
        optional = {
            "object_size": _object_size_plot(
                rows, registry, output_dir, ordered_methods, report_scope
            ),
            "quality_efficiency": _quality_efficiency_plot(rows, registry, output_dir, report_scope),
            "convergence": (_diagnostic_plot(records, output_dir, report_scope)
                            if include_saved_diagnostics else None),
        }
        for name, result in optional.items():
            if result is None:
                continue
            png, pdf = result
            paths[name + "_png"] = png
            paths[name + "_pdf"] = pdf
            included_pngs.append(png)
    paths["tex"] = _figures_tex(included_pngs, output_dir, report_scope)
    return paths
