from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS
from ..paths import project_root
from ..registry import Registry
from .formatting import format_decimal, format_metric


MAIN_METRICS: Tuple[str, ...] = (
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_AR_100",
)

METRIC_LABELS = {
    "bbox_mAP": "AP",
    "bbox_mAP_50": "AP@.50",
    "bbox_mAP_75": "AP@.75",
    "bbox_mAP_small": "AP$_S$",
    "bbox_mAP_medium": "AP$_M$",
    "bbox_mAP_large": "AP$_L$",
    "bbox_AR_1": "AR@1",
    "bbox_AR_10": "AR@10",
    "bbox_AR_100": "AR@100",
    "bbox_AR_small": "AR$_S$",
    "bbox_AR_medium": "AR$_M$",
    "bbox_AR_large": "AR$_L$",
}


def runtime_provenance(root: Optional[Path] = None) -> Dict[str, Any]:
    repository = (root or project_root()).resolve()
    commit = "unavailable"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repository),
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip()
        if len(candidate) == 40 and all(
            character in "0123456789abcdef" for character in candidate.lower()
        ):
            commit = candidate.lower()
    except (OSError, subprocess.SubprocessError):
        pass

    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for package in (
        "torch",
        "torchvision",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmyolo",
        "pycocotools",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"git_commit": commit, "software_versions": versions}


def clean_command(
    dataset: str,
    *,
    split: str,
    targets: Sequence[str],
    canonical_targets: Sequence[str],
    execute: bool,
    max_images: Optional[int],
    device: str,
    save_visualizations: bool,
    devices: Optional[Sequence[str]] = None,
) -> str:
    arguments = [
        "conda",
        "run",
        "-n",
        "oda",
        "python",
        "experiments/clean_map.py",
        "--dataset",
        dataset,
    ]
    if split != "val":
        arguments.extend(["--split", split])
    if list(targets) != list(canonical_targets):
        arguments.extend(["--targets", ",".join(targets)])
    if not execute:
        arguments.append("--plan-only")
    if max_images is not None:
        arguments.extend(["--max-images", str(max_images)])
    if devices is not None and len(devices) > 1:
        arguments.extend(["--devices", ",".join(devices)])
    elif device != "cuda:0":
        arguments.extend(["--device", device])
    arguments.append(
        "--visualize-predictions"
        if save_visualizations
        else "--no-visualize-predictions"
    )
    return " ".join(arguments)


def group_boundaries(registry: Registry, targets: Sequence[str]) -> List[int]:
    return [
        index
        for index in range(1, len(targets))
        if registry.model_group(targets[index - 1]).id
        != registry.model_group(targets[index]).id
    ]


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}.".format(path.name),
            suffix=".part",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _atomic_save_figure(
    figure: Any,
    path: Path,
    *,
    file_format: str,
    dpi: Optional[int] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".{}".format(file_format),
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        metadata: Dict[str, Any] = {
            "Title": "Clean object-detection performance",
            "Creator": "LGP",
        }
        if file_format == "pdf":
            metadata.update({"CreationDate": None, "ModDate": None})
        figure.savefig(
            str(temporary),
            format=file_format,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            metadata=metadata,
        )
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _escape_latex(value: str) -> str:
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


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metric_value(record: Mapping[str, Any], metric: str) -> str:
    if record.get("status") == "failed":
        return "ERR"
    return format_metric((record.get("metrics") or {}).get(metric), missing="NR")


def _metric_latex(record: Mapping[str, Any], metric: str) -> str:
    value = _metric_value(record, metric)
    return (
        value
        if value not in {"NR", "ERR"}
        else r"\textit{" + value + "}"
    )


def _category_value(record: Mapping[str, Any], category: str) -> str:
    if record.get("status") == "failed":
        return "ERR"
    per_category = record.get("per_category_ap") or {}
    if (
        record.get("status") in {"complete", "complete_with_failures"}
        and category in per_category
        and per_category[category] is None
    ):
        return "NA"
    return format_metric(
        per_category.get(category),
        missing="NR",
    )


def _category_latex(record: Mapping[str, Any], category: str) -> str:
    value = _category_value(record, category)
    return (
        value
        if value not in {"NR", "ERR", "NA"}
        else r"\textit{" + value + "}"
    )


def _ordered_records(
    records: Sequence[Mapping[str, Any]], registry: Registry
) -> List[Mapping[str, Any]]:
    by_target = {str(record["target"]): record for record in records}
    return [
        by_target[target]
        for target in registry.target_ids()
        if target in by_target
    ]


def _model_name(registry: Registry, model_id: str) -> str:
    model = registry.model(model_id)
    return "{} ({})".format(model.display_name, model.table_code)


def _figure_model_name(registry: Registry, model_id: str) -> str:
    model = registry.model(model_id)
    return "{:02d} {} — {} ({})".format(
        registry.model_position(model_id),
        model.table_code,
        model.display_name,
        model.backbone,
    )


def _scope_label(registry: Registry, dataset: str, split: str) -> str:
    expected = registry.dataset(dataset).split(split).expected_images
    if dataset == "voc" and split == "val":
        return "VOC2007 test, {:,} images".format(expected)
    if dataset == "coco" and split == "val":
        return "COCO val2017, {:,} images".format(expected)
    if dataset == "bdd100k" and split == "val":
        return "BDD100K validation, {:,} images".format(expected)
    return "{} {}, {:,} images".format(dataset.upper(), split, expected)


def _scope_label_zh(registry: Registry, dataset: str, split: str) -> str:
    expected = registry.dataset(dataset).split(split).expected_images
    if dataset == "voc" and split == "val":
        return "VOC2007 test，共 {:,} 张图像".format(expected)
    if dataset == "coco" and split == "val":
        return "COCO val2017，共 {:,} 张图像".format(expected)
    if dataset == "bdd100k" and split == "val":
        return "BDD100K validation，共 {:,} 张图像".format(expected)
    return "{} {}，共 {:,} 张图像".format(dataset.upper(), split, expected)


def _clean_caption(registry: Registry, dataset: str, split: str) -> str:
    checkpoint_phrase = (
        "fixed final checkpoints"
        if dataset in {"voc", "bdd100k"}
        else "registered checkpoints"
    )
    selection_phrase = (
        "no test-set selection"
        if dataset == "voc" and split == "val"
        else "no formal-split selection"
    )
    return (
        "{}; {}; {}; all detectors evaluated identically as clean targets."
    ).format(
        _scope_label(registry, dataset, split),
        checkpoint_phrase,
        selection_phrase,
    )


def _clean_caption_zh(registry: Registry, dataset: str, split: str) -> str:
    checkpoint_phrase = (
        "固定最终 checkpoint"
        if dataset in {"voc", "bdd100k"}
        else "已登记 checkpoint"
    )
    selection_phrase = (
        "不使用测试集进行选择"
        if dataset == "voc" and split == "val"
        else "不使用正式 split 进行选择"
    )
    return "{}；{}；{}；所有检测器均以相同方式作为 clean 评估目标。".format(
        _scope_label_zh(registry, dataset, split),
        checkpoint_phrase,
        selection_phrase,
    )


def _is_complete_publication(
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    context: Mapping[str, Any],
) -> bool:
    if split != "val":
        return False
    if (context.get("execute") is not True or context.get("max_images") is not None
            or context.get("execution_failed")):
        return False
    if [str(record["target"]) for record in ordered] != registry.target_ids():
        return False
    expected_images = registry.dataset(dataset).split(split).expected_images
    categories = registry.dataset(dataset).classes
    for record in ordered:
        if record.get("dataset") != dataset or record.get("split") != split:
            return False
        if record.get("status") != "complete":
            return False
        if int(record.get("images") or -1) != expected_images:
            return False
        if record.get("failures"):
            return False
        if (
            not _finite(record.get("inference_seconds_total"))
            or float(record["inference_seconds_total"]) < 0.0
        ):
            return False
        if (
            not _finite(record.get("inference_seconds_mean"))
            or float(record["inference_seconds_mean"]) < 0.0
        ):
            return False
        try:
            if int(record.get("detections")) < 0:
                return False
        except (TypeError, ValueError):
            return False
        if not math.isclose(
            float(record["inference_seconds_mean"]),
            float(record["inference_seconds_total"]) / expected_images,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            return False
        metrics = record.get("metrics") or {}
        if any(not _finite(metrics.get(metric)) for metric in COCO_BBOX_METRICS):
            return False
        category_metrics = record.get("per_category_ap") or {}
        if any(
            category not in category_metrics
            or not _finite(category_metrics.get(category))
            for category in categories
        ):
            return False
        digest = str(record.get("checkpoint_sha256") or "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest.lower()
        ):
            return False
        predictions_digest = str(record.get("predictions_sha256") or "")
        if len(predictions_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in predictions_digest.lower()
        ):
            return False
    return True


def _write_main_csv(
    path: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
) -> None:
    fields = [
        "paper_order",
        "paper_group",
        "dataset",
        "split",
        "target",
        "model",
        "backbone",
        "source",
        "held_out",
        "status",
        *COCO_BBOX_METRICS,
        "inference_seconds_total",
        "inference_seconds_mean",
        "images",
        "detections",
        "failure_count",
        "checkpoint_sha256",
        "predictions_sha256",
        "reason",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".part",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in ordered:
                model = registry.model(str(record["target"]))
                writer.writerow(
                    {
                        "paper_order": registry.model_position(model.id),
                        "paper_group": registry.model_group(model.id).id,
                        "dataset": record.get("dataset", dataset),
                        "split": record.get("split", "val"),
                        "target": model.id,
                        "model": model.display_name,
                        "backbone": model.backbone,
                        "source": model.source,
                        "held_out": model.held_out,
                        "status": record.get("status", ""),
                        **{
                            metric: _metric_value(record, metric)
                            for metric in COCO_BBOX_METRICS
                        },
                        "inference_seconds_total": format_decimal(
                            record.get("inference_seconds_total"),
                            places=6,
                        ),
                        "inference_seconds_mean": format_decimal(
                            record.get("inference_seconds_mean"),
                            places=6,
                        ),
                        "images": record.get("images"),
                        "detections": record.get("detections"),
                        "failure_count": len(record.get("failures") or []),
                        "checkpoint_sha256": record.get("checkpoint_sha256"),
                        "predictions_sha256": record.get("predictions_sha256"),
                        "reason": record.get("reason", ""),
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _table_rows(
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    metrics: Sequence[str],
    *,
    include_speed: bool,
) -> List[str]:
    rows: List[str] = []
    previous_group: Optional[str] = None
    for record in ordered:
        model = registry.model(str(record["target"]))
        group = registry.model_group(model.id)
        if previous_group is not None and group.id != previous_group:
            rows.append(r"\midrule")
        previous_group = group.id
        values = [_metric_latex(record, metric) for metric in metrics]
        if include_speed:
            mean_seconds = record.get("inference_seconds_mean")
            milliseconds = (
                float(mean_seconds) * 1000.0 if _finite(mean_seconds) else None
            )
            values.append(
                format_decimal(
                    milliseconds,
                    missing=r"\textit{NR}",
                )
            )
        rows.append(
            "{} & {} & {} & {} \\\\".format(
                _escape_latex(group.display_name),
                _escape_latex(_model_name(registry, model.id)),
                _escape_latex(model.backbone),
                " & ".join(values),
            )
        )
    return rows


def _write_main_tex(
    path: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    *,
    complete: bool,
) -> None:
    caption = _clean_caption(registry, dataset, split)
    if not complete:
        caption = "PARTIAL diagnostic; not a formal result. " + caption
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:" + dataset + r"_clean_map}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l l *{7}{c} c}",
        r"\toprule",
        r"Group & Detector & Backbone & AP & AP@.50 & AP@.75 & AP$_S$ & AP$_M$ & AP$_L$ & AR@100 & ms/image \\",
        r"\midrule",
        *_table_rows(
            ordered,
            registry,
            MAIN_METRICS,
            include_speed=True,
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        r"\end{table*}",
        "",
    ]
    _atomic_write_text(path, "\n".join(lines))


def _write_all_metrics_tex(
    path: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    *,
    complete: bool,
) -> None:
    caption = (
        "All twelve standard COCO-style bbox metrics and inference speed; "
        + _clean_caption(registry, dataset, split)
    )
    if not complete:
        caption = "PARTIAL diagnostic; not a formal result. " + caption
    header = " & ".join(METRIC_LABELS[metric] for metric in COCO_BBOX_METRICS)
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:" + dataset + r"_clean_all_metrics}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l l *{12}{c} c}",
        r"\toprule",
        "Group & Detector & Backbone & {} & ms/image \\\\".format(header),
        r"\midrule",
        *_table_rows(
            ordered,
            registry,
            COCO_BBOX_METRICS,
            include_speed=True,
        ),
        r"\bottomrule",
        r"\end{tabular}",
        "}",
        r"\end{table*}",
        "",
    ]
    _atomic_write_text(path, "\n".join(lines))


def _write_per_category_csv(
    path: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
) -> None:
    categories = list(registry.dataset(dataset).classes)
    fields = [
        "paper_order",
        "paper_group",
        "dataset",
        "target",
        "model",
        "backbone",
        "status",
        *categories,
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(path.name),
        suffix=".part",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for record in ordered:
                model = registry.model(str(record["target"]))
                writer.writerow(
                    {
                        "paper_order": registry.model_position(model.id),
                        "paper_group": registry.model_group(model.id).id,
                        "dataset": dataset,
                        "target": model.id,
                        "model": model.display_name,
                        "backbone": model.backbone,
                        "status": record.get("status", ""),
                        **{
                            category: _category_value(record, category)
                            for category in categories
                        },
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_per_category_tex(
    path: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    *,
    complete: bool,
) -> None:
    categories = list(registry.dataset(dataset).classes)
    caption = (
        "Per-category bbox AP in canonical detector order; "
        + _clean_caption(registry, dataset, split)
    )
    if any(
        _category_value(record, category) == "NA"
        for record in ordered
        for category in categories
    ):
        caption += " NA denotes undefined category AP on the evaluated images."
    if not complete:
        caption = "PARTIAL diagnostic; not a formal result. " + caption
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\scriptsize",
        r"\caption{" + _escape_latex(caption) + "}",
        r"\label{tab:" + dataset + r"_clean_per_category}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l l *{" + str(len(categories)) + r"}{c}}",
        r"\toprule",
        "Group & Detector & {} \\\\".format(
            " & ".join(_escape_latex(category) for category in categories)
        ),
        r"\midrule",
    ]
    previous_group: Optional[str] = None
    for record in ordered:
        model = registry.model(str(record["target"]))
        group = registry.model_group(model.id)
        if previous_group is not None and group.id != previous_group:
            lines.append(r"\midrule")
        previous_group = group.id
        lines.append(
            "{} & {} & {} \\\\".format(
                _escape_latex(group.display_name),
                _escape_latex(_model_name(registry, model.id)),
                " & ".join(
                    _category_latex(record, category)
                    for category in categories
                ),
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            r"\end{table*}",
            "",
        ]
    )
    _atomic_write_text(path, "\n".join(lines))


def _group_ranges(
    registry: Registry, targets: Sequence[str]
) -> List[Tuple[int, int, str]]:
    ranges: List[Tuple[int, int, str]] = []
    if not targets:
        return ranges
    start = 0
    current = registry.model_group(targets[0])
    for index in range(1, len(targets)):
        group = registry.model_group(targets[index])
        if group.id != current.id:
            ranges.append((start, index - 1, current.display_name))
            start = index
            current = group
    ranges.append((start, len(targets) - 1, current.display_name))
    return ranges


def _write_figure(
    output_dir: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    *,
    complete: bool,
) -> Dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = "clean_map" if complete else "clean_map_PARTIAL"
    numeric = [
        record
        for record in ordered
        if all(
            _finite((record.get("metrics") or {}).get(metric))
            for metric in ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75")
        )
    ]
    caption = _clean_caption(registry, dataset, split)
    if not complete:
        caption = "PARTIAL diagnostic; not a formal result. " + caption

    if numeric:
        targets = [str(record["target"]) for record in numeric]
        y = np.arange(len(numeric), dtype=float)
        figure, axis = plt.subplots(
            figsize=(12.8, max(6.4, 0.49 * len(numeric) + 2.6)),
            facecolor="white",
        )
        for band_index, (start, end, group_name) in enumerate(
            _group_ranges(registry, targets)
        ):
            if band_index % 2:
                axis.axhspan(
                    start - 0.5,
                    end + 0.5,
                    color="#F4F5F6",
                    zorder=0,
                )
            axis.text(
                0.992,
                (start + end) / 2.0,
                group_name,
                ha="right",
                va="center",
                fontsize=7.3,
                color="#62666A",
                zorder=1,
            )
        values_by_metric = {
            metric: [
                float((record.get("metrics") or {})[metric])
                for record in numeric
            ]
            for metric in ("bbox_mAP", "bbox_mAP_50", "bbox_mAP_75")
        }
        low = [
            min(values_by_metric[metric][index] for metric in values_by_metric)
            for index in range(len(numeric))
        ]
        high = [
            max(values_by_metric[metric][index] for metric in values_by_metric)
            for index in range(len(numeric))
        ]
        axis.hlines(
            y,
            low,
            high,
            color="#C9CDD1",
            linewidth=1.0,
            zorder=1,
        )
        styles = (
            ("bbox_mAP", "AP@[.50:.95]", "#0072B2", "o"),
            ("bbox_mAP_50", "AP50", "#E69F00", "s"),
            ("bbox_mAP_75", "AP75", "#5C5C5C", "^"),
        )
        for metric, label, color, marker in styles:
            axis.scatter(
                values_by_metric[metric],
                y,
                s=38,
                marker=marker,
                color=color,
                edgecolors="#202124",
                linewidths=0.45,
                label=label,
                zorder=3,
            )
        for index, value in enumerate(values_by_metric["bbox_mAP"]):
            right_side = value > 0.90
            axis.annotate(
                "{:.4f}".format(value),
                (value, y[index]),
                xytext=(-7 if right_side else 7, 7),
                textcoords="offset points",
                ha="right" if right_side else "left",
                va="bottom",
                fontsize=7.5,
                color="#1F2933",
                zorder=4,
            )
        for boundary in group_boundaries(registry, targets):
            axis.axhline(
                boundary - 0.5,
                color="#7A7F85",
                linewidth=0.8,
                zorder=2,
            )
        axis.set_yticks(
            y,
            [_figure_model_name(registry, target) for target in targets],
            fontsize=8.2,
        )
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks(np.linspace(0.0, 1.0, 11))
        axis.set_xlabel("COCO-style bbox metric (0–1)", color="#30343B")
        axis.grid(axis="x", color="#E1E4E8", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            ncol=3,
            frameon=False,
        )
        axis.set_title(
            "Clean object-detection performance\n{}".format(
                _clean_caption(registry, dataset, split).rstrip(".")
            ),
            loc="left",
            fontsize=11.5,
            color="#202124",
            pad=30 if complete else 52,
        )
        if not complete:
            axis.annotate(
                "PARTIAL / DIAGNOSTIC — NOT A FORMAL RESULT",
                xy=(0.0, 1.0),
                xycoords="axes fraction",
                xytext=(0, 34),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#9C2F2F",
                annotation_clip=False,
            )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#8A8F96")
        axis.spines["bottom"].set_color("#8A8F96")
        axis.tick_params(colors="#3E434A")
        figure.subplots_adjust(left=0.35, right=0.97, top=0.84, bottom=0.09)
    else:
        counts = Counter(
            str(record.get("status", "unknown")) for record in ordered
        )
        labels = sorted(counts)
        figure, axis = plt.subplots(figsize=(8.0, 4.8), facecolor="white")
        bars = axis.barh(
            labels,
            [counts[label] for label in labels],
            color="#0072B2",
            edgecolor="#202124",
            linewidth=0.45,
        )
        axis.bar_label(bars, padding=4, fmt="%d")
        axis.set_xlabel("Detectors")
        axis.set_title(
            "PARTIAL clean-evaluation status\n{}".format(
                _scope_label(registry, dataset, split)
            ),
            loc="left",
            fontsize=13,
        )
        axis.text(
            0.0,
            1.01,
            "DIAGNOSTIC ONLY — NOT A FORMAL RESULT",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#9C2F2F",
        )
        axis.grid(axis="x", color="#E1E4E8", linewidth=0.65)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    png_path = figure_dir / "{}.png".format(stem)
    pdf_path = figure_dir / "{}.pdf".format(stem)
    _atomic_save_figure(figure, png_path, file_format="png", dpi=320)
    _atomic_save_figure(figure, pdf_path, file_format="pdf")
    plt.close(figure)

    figures_tex = figure_dir / (
        "figures.tex" if complete else "figures_PARTIAL.tex"
    )
    figure_label = "{}_clean_map{}".format(
        dataset, "" if complete else "_partial"
    )
    _atomic_write_text(
        figures_tex,
        "\n".join(
            [
                r"\begin{figure*}[!htb]",
                r"\centering",
                r"\includegraphics[width=0.96\textwidth]{\detokenize{"
                + "figures/"
                + stem
                + r".pdf}}",
                r"\caption{" + _escape_latex(caption) + "}",
                r"\label{fig:" + figure_label + "}",
                r"\end{figure*}",
                "",
            ]
        ),
    )
    obsolete_stem = "clean_map_PARTIAL" if complete else "clean_map"
    for obsolete in (
        figure_dir / "{}.png".format(obsolete_stem),
        figure_dir / "{}.pdf".format(obsolete_stem),
        figure_dir
        / ("figures_PARTIAL.tex" if complete else "figures.tex"),
    ):
        try:
            obsolete.unlink()
        except FileNotFoundError:
            pass
    return {
        "png": png_path,
        "pdf": pdf_path,
        "figures_tex": figures_tex,
    }


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(_escape_markdown(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(_escape_markdown(value) for value in row) + " |"
        for row in rows
    )
    return "\n".join(lines)


def _prediction_digest(
    record: Mapping[str, Any], run_root: Path, target: str
) -> str:
    recorded = str(record.get("predictions_sha256") or "")
    if len(recorded) == 64:
        return recorded
    predictions = run_root / "evaluations" / target / "predictions.json"
    return file_digest(predictions) if predictions.is_file() else "unavailable"


def _write_markdown_reports(
    output_dir: Path,
    ordered: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    split: str,
    context: Mapping[str, Any],
    *,
    complete: bool,
    figure_name: str,
) -> Dict[str, Path]:
    run_root = output_dir.parent
    scope = _scope_label(registry, dataset, split)
    scope_zh = _scope_label_zh(registry, dataset, split)
    caption = _clean_caption(registry, dataset, split)
    caption_zh = _clean_caption_zh(registry, dataset, split)
    status_en = "COMPLETE" if complete else "PARTIAL / DIAGNOSTIC"
    status_zh = "完整正式结果" if complete else "部分结果 / 诊断"
    command = str(context.get("command") or "unavailable")
    git_commit = str(context.get("git_commit") or "unavailable")
    versions = context.get("software_versions") or {}

    headline_rows: List[List[Any]] = []
    all_metric_rows: List[List[Any]] = []
    category_rows: List[List[Any]] = []
    provenance_rows: List[List[Any]] = []
    categories = list(registry.dataset(dataset).classes)
    for record in ordered:
        target = str(record["target"])
        model = registry.model(target)
        mean_seconds = record.get("inference_seconds_mean")
        mean_ms = float(mean_seconds) * 1000.0 if _finite(mean_seconds) else None
        headline_rows.append(
            [
                registry.model_position(target),
                registry.model_group(target).display_name,
                _model_name(registry, target),
                model.backbone,
                _metric_value(record, "bbox_mAP"),
                _metric_value(record, "bbox_mAP_50"),
                _metric_value(record, "bbox_mAP_75"),
                format_decimal(mean_ms, missing="NR"),
                format_decimal(
                    record.get("inference_seconds_total"),
                    missing="NR",
                ),
                record.get("images", ""),
                record.get("detections", ""),
                len(record.get("failures") or []),
            ]
        )
        all_metric_rows.append(
            [
                registry.model_position(target),
                model.table_code,
                *[
                    _metric_value(record, metric)
                    for metric in COCO_BBOX_METRICS
                ],
            ]
        )
        category_rows.append(
            [
                registry.model_position(target),
                model.table_code,
                *[
                    _category_value(record, category)
                    for category in categories
                ],
            ]
        )
        provenance_rows.append(
            [
                registry.model_position(target),
                model.table_code,
                record.get("checkpoint_sha256") or "unavailable",
                _prediction_digest(record, run_root, target),
            ]
        )

    metric_headers = [METRIC_LABELS[metric].replace("$", "") for metric in COCO_BBOX_METRICS]
    software_rows = [[name, versions[name]] for name in sorted(versions)]
    figure_path = "figures/{}".format(figure_name)
    inventory = [
        "clean_map.csv",
        "clean_map.tex",
        "clean_map_all_metrics.tex",
        "clean_per_category_ap.csv",
        "clean_per_category_ap.tex",
        "clean_map_complete.tex",
        "clean_map.md",
        "clean_map.zh-CN.md",
        figure_path,
        figure_path.replace(".png", ".pdf"),
        "figures/{}".format(
            "figures.tex" if complete else "figures_PARTIAL.tex"
        ),
        "artifact_manifest.json",
    ]

    missing_note_en = (
        "Every requested record passed the publication gates; the tables contain "
        "no missing or failed metric cells."
        if complete
        else "This run is diagnostic. Unrun values use `NR`; execution failures use `ERR`."
    )
    missing_note_zh = (
        "全部请求记录均通过发布闸门，表格中没有缺失或失败指标单元格。"
        if complete
        else "这是诊断运行。未运行值记为 `NR`，执行失败记为 `ERR`。"
    )
    if dataset == "voc" and split == "val":
        split_policy_en = (
            "The formal split is VOC2007 test registered locally as `val`; "
            "its 4,952 images are never used for training, early stopping, "
            "hyperparameter tuning, or checkpoint selection. The published "
            "model for each detector is its predeclared final epoch. VOC AP50 "
            "here is COCO-style bbox AP on the converted annotations, not "
            "native VOC2007 devkit AP. Difficult objects follow the frozen "
            "conversion to `iscrowd`/`ignore` and COCOeval matching."
        )
        split_policy_zh = (
            "正式 split 是在本项目中登记为 `val` 的 VOC2007 test，共 4,952 "
            "张图像；它不参与训练、早停、超参数调整或 checkpoint 选择。每个"
            "检测器只发布预先声明的最终 epoch。这里的 VOC AP50 是转换标注上"
            "的 COCO-style bbox AP，不是原生 VOC2007 devkit AP。difficult "
            "对象遵循冻结的 `iscrowd`/`ignore` 转换及 COCOeval 匹配规则。"
        )
    elif dataset == "coco" and split == "val":
        split_policy_en = (
            "The registered split is the 5,000-image COCO val2017 set. This "
            "clean report does not select or revise a detector checkpoint. "
            "Separately registered VCSF retrospective configuration selection "
            "on this split is not independent confirmation and does not "
            "change earlier frozen confirmatory protocols."
        )
        split_policy_zh = (
            "登记 split 是包含 5,000 张图像的 COCO val2017。本 clean 报告不"
            "选择或修改检测器 checkpoint。另行登记、在该 split 上开展的 "
            "VCSF 回顾性配置选择不是独立确认，也不改变较早冻结的确认协议。"
        )
    elif dataset == "bdd100k" and split == "val":
        split_policy_en = (
            "The formal split is the 10,000-image BDD100K validation set. It "
            "is not used for training, early stopping, hyperparameter tuning, "
            "or checkpoint selection; each detector uses its predeclared "
            "final epoch."
        )
        split_policy_zh = (
            "正式 split 是包含 10,000 张图像的 BDD100K validation；它不参与"
            "训练、早停、超参数调整或 checkpoint 选择，每个检测器使用预先声明"
            "的最终 epoch。"
        )
    else:
        split_policy_en = (
            "Interpret this report using the recorded split and evaluated "
            "image count. COCO dev is for development and parameter selection; "
            "a diagnostic or limited-image report is not a formal result. "
            "Checkpoint and configuration-selection boundaries remain those "
            "of the declared protocol."
        )
        split_policy_zh = (
            "请按记录中的 split 与实际评估图像数解释本报告。COCO dev 用于"
            "开发和参数选择；诊断或限图报告不是正式结果。checkpoint 与配置"
            "选择边界仍遵循声明的协议。"
        )

    english = "\n".join(
        [
            "[简体中文](clean_map.zh-CN.md)",
            "",
            "# Clean object-detection evaluation",
            "",
            "## Technical summary",
            "",
            "- **Status:** {}.".format(status_en),
            "- **Scope:** {}.".format(scope),
            "- **Protocol:** {}".format(caption),
            "- **Interpretation:** all 16 registered detectors are equal clean-evaluation targets. Attack-experiment role metadata remain only in machine-readable provenance and do not affect this evaluation, ordering, selection, or aggregation.",
            "",
            "## Clean results in canonical order",
            "",
            "The figure compares AP@[.50:.95], AP50, and AP75 on a fixed 0–1 scale. Marker shape as well as color distinguishes the three metrics; only the primary AP is directly annotated. Exact values are in the tables below.",
            "",
            "![Clean bbox summary]({})".format(figure_path),
            "",
            _markdown_table(
                [
                    "#",
                    "Architecture group",
                    "Detector",
                    "Backbone",
                    "AP",
                    "AP50",
                    "AP75",
                    "ms/image",
                    "Total inference s",
                    "Images",
                    "Detections",
                    "Failures",
                ],
                headline_rows,
            ),
            "",
            "## Scope, data, and metric definitions",
            "",
            "AP is COCO-style bbox AP averaged over IoU thresholds 0.50:0.05:0.95, using 101 recall thresholds from 0 to 1. AP50 and AP75 use the stated IoU thresholds; AP/AR size variants use the evaluator's standard area ranges. AP uses `maxDets=100`; AR@1, AR@10, and AR@100 use COCOeval `maxDets` settings of 1, 10, and 100. Default category-wise evaluation applies this limit within each image/category, not as a global cap across all categories. Native detector output limits are separate. Inference time covers detector inference and GPU synchronization, not dataset loading or report generation.",
            "",
            split_policy_en,
            "",
            "## All twelve COCO-style bbox metrics",
            "",
            _markdown_table(["#", "Code", *metric_headers], all_metric_rows),
            "",
            "## Per-category bbox AP",
            "",
            "Per-category AP averages IoU thresholds 0.50:0.05:0.95 with `area=all` and `maxDets=100`; these values are not per-category AP50. An explicit JSON `null` from an executed evaluation is shown as `NA`: AP is undefined for that category on the evaluated images, not zero. This definition also applies to `clean_per_category_ap.csv` and `clean_per_category_ap.tex`.",
            "",
            _markdown_table(["#", "Code", *categories], category_rows),
            "",
            "## Methodology and reproducibility",
            "",
            "Command:",
            "",
            "```text",
            command,
            "```",
            "",
            "Evaluation Git commit: `{}`".format(git_commit),
            "",
            _markdown_table(["Software", "Version"], software_rows),
            "",
            "`records.json` is the aggregate machine-readable source. Each `evaluations/<target>/metrics.json` preserves unrounded metrics and each `evaluations/<target>/predictions.json` preserves every `image_id`, `category_id`, `bbox`, and `score`. The prediction files are not duplicated in this document; their hashes below make the exact inputs to recomputation auditable.",
            "",
            "## Checkpoint and prediction provenance",
            "",
            _markdown_table(
                ["#", "Code", "Checkpoint SHA-256", "Predictions SHA-256"],
                provenance_rows,
            ),
            "",
            "## Limitations, failure semantics, and next step",
            "",
            missing_note_en,
            "",
            "These are descriptive clean bbox metrics for one frozen dataset split and one checkpoint per detector. They do not establish transfer robustness. The next experiment may consume the same checkpoint hashes and canonical target order for adversarial transfer evaluation.",
            "",
            "## Output inventory",
            "",
            *["- `{}`".format(item) for item in inventory],
            "",
        ]
    )

    chinese = "\n".join(
        [
            "[English](clean_map.md)",
            "",
            "# 目标检测器 Clean 评估",
            "",
            "## 技术摘要",
            "",
            "- **状态：**{}。".format(status_zh),
            "- **范围：**{}。".format(scope_zh),
            "- **协议：**{}".format(caption_zh),
            "- **解释：**16 个已登记检测器在 clean 评估中地位完全相同。攻击实验角色元数据仅保留在机器可读 provenance 中，不参与本评估、排序、选择或汇总。",
            "",
            "## 按固定顺序排列的 Clean 结果",
            "",
            "汇总图在固定 0–1 坐标范围内比较 AP@[.50:.95]、AP50 与 AP75，并同时使用颜色和 marker 形状区分指标；图中只直接标注主 AP。精确数值见下表。",
            "",
            "![Clean bbox 汇总]({})".format(figure_path),
            "",
            _markdown_table(
                [
                    "序号",
                    "架构组",
                    "检测器",
                    "Backbone",
                    "AP",
                    "AP50",
                    "AP75",
                    "毫秒/图",
                    "总推理秒数",
                    "图像数",
                    "检测数",
                    "失败数",
                ],
                headline_rows,
            ),
            "",
            "## 范围、数据与指标定义",
            "",
            "AP 是在 IoU=0.50:0.05:0.95 上平均的 COCO-style bbox AP，使用从 0 到 1 的 101 个 recall 阈值。AP50 与 AP75 使用对应 IoU，AP/AR 尺度指标采用评估器标准面积区间。AP 使用 `maxDets=100`；AR@1、AR@10、AR@100 分别使用 COCOeval 的 `maxDets` 设置 1、10、100。默认分类别评估将限制应用到每个图像/类别，而不是全部类别合计的全图上限；检测器自身的输出限制另行定义。推理时间包含检测器推理与 GPU 同步，不含数据加载和报告生成。",
            "",
            split_policy_zh,
            "",
            "## 全部 12 项 COCO-style bbox 指标",
            "",
            _markdown_table(["序号", "代码", *metric_headers], all_metric_rows),
            "",
            "## 逐类别 bbox AP",
            "",
            "逐类 AP 对 IoU=0.50:0.05:0.95 求平均，使用 `area=all` 与 `maxDets=100`，不是逐类 AP50。已执行评估中的显式 JSON `null` 显示为 `NA`，表示该类别在本次评估图像上没有可定义的 AP，不是零。`clean_per_category_ap.csv` 和 `clean_per_category_ap.tex` 也采用这一定义。",
            "",
            _markdown_table(["序号", "代码", *categories], category_rows),
            "",
            "## 方法与可复现信息",
            "",
            "命令：",
            "",
            "```text",
            command,
            "```",
            "",
            "评估 Git commit：`{}`".format(git_commit),
            "",
            _markdown_table(["软件", "版本"], software_rows),
            "",
            "`records.json` 是汇总机器可读源。每个 `evaluations/<target>/metrics.json` 保留未舍入指标，每个 `evaluations/<target>/predictions.json` 保留全部 `image_id`、`category_id`、`bbox` 与 `score`。本文档不复制海量预测框；下列哈希用于审计和重新计算。",
            "",
            "## Checkpoint 与预测 provenance",
            "",
            _markdown_table(
                ["序号", "代码", "Checkpoint SHA-256", "Predictions SHA-256"],
                provenance_rows,
            ),
            "",
            "## 局限、失败语义与下一步",
            "",
            missing_note_zh,
            "",
            "这些结果是在一个冻结数据 split 和每模型一个固定 checkpoint 上得到的描述性 clean bbox 指标，不能单独证明迁移鲁棒性。后续对抗迁移评估应复用相同 checkpoint 哈希和固定目标顺序。",
            "",
            "## 产物清单",
            "",
            *["- `{}`".format(item) for item in inventory],
            "",
        ]
    )

    english_path = output_dir / "clean_map.md"
    chinese_path = output_dir / "clean_map.zh-CN.md"
    _atomic_write_text(english_path, english)
    _atomic_write_text(chinese_path, chinese)
    return {"markdown_en": english_path, "markdown_zh": chinese_path}


def _write_artifact_manifest(
    path: Path,
    output_dir: Path,
    *,
    dataset: str,
    complete: bool,
) -> None:
    run_root = output_dir.parent
    candidates = [
        item
        for item in output_dir.rglob("*")
        if item.is_file()
        and item != path
        and ".part" not in item.name
    ]
    for relative in (
        "plan.json",
        "plan.csv",
        "records.json",
        "summary.json",
        "execution_assignments.json",
        "execution_state.json",
    ):
        candidate = run_root / relative
        if candidate.is_file():
            candidates.append(candidate)
    evaluations = run_root / "evaluations"
    if evaluations.is_dir():
        for pattern in (
            "*/metrics.json",
            "*/predictions.json",
            "*/coco_summary.txt",
            "*/coco_summary.csv",
            "*/coco_summary.tex",
        ):
            candidates.extend(evaluations.glob(pattern))
    unique = sorted({candidate.resolve() for candidate in candidates})
    artifacts = []
    for candidate in unique:
        try:
            relative = candidate.relative_to(run_root.resolve()).as_posix()
        except ValueError:
            relative = candidate.name
        artifacts.append(
            {
                "path": relative,
                "bytes": candidate.stat().st_size,
                "sha256": file_digest(candidate),
            }
        )
    payload = {
        "schema_version": 1,
        "dataset": dataset,
        "publication_status": "complete" if complete else "partial",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hash_algorithm": "sha256",
        "artifacts": artifacts,
    }
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def write_clean_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    output_dir: Path,
    *,
    context: Optional[Mapping[str, Any]] = None,
    include_manifest: bool = True,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = _ordered_records(records, registry)
    report_context: Mapping[str, Any] = context or {}
    split = str(
        report_context.get(
            "split",
            ordered[0].get("split", "val") if ordered else "val",
        )
    )
    complete = _is_complete_publication(
        ordered,
        registry,
        dataset,
        split,
        report_context,
    )

    csv_path = output_dir / "clean_map.csv"
    tex_path = output_dir / "clean_map.tex"
    all_metrics_path = output_dir / "clean_map_all_metrics.tex"
    category_csv_path = output_dir / "clean_per_category_ap.csv"
    category_tex_path = output_dir / "clean_per_category_ap.tex"
    complete_tex_path = output_dir / "clean_map_complete.tex"
    manifest_path = output_dir / "artifact_manifest.json"

    _write_main_csv(csv_path, ordered, registry, dataset)
    _write_main_tex(
        tex_path,
        ordered,
        registry,
        dataset,
        split,
        complete=complete,
    )
    _write_all_metrics_tex(
        all_metrics_path,
        ordered,
        registry,
        dataset,
        split,
        complete=complete,
    )
    _write_per_category_csv(
        category_csv_path,
        ordered,
        registry,
        dataset,
    )
    _write_per_category_tex(
        category_tex_path,
        ordered,
        registry,
        dataset,
        split,
        complete=complete,
    )
    figure_paths = _write_figure(
        output_dir,
        ordered,
        registry,
        dataset,
        split,
        complete=complete,
    )
    figure_input = (
        "figures/figures.tex"
        if complete
        else "figures/figures_PARTIAL.tex"
    )
    _atomic_write_text(
        complete_tex_path,
        "\n".join(
            [
                r"% Generated from records.json; do not hand-edit numeric values.",
                r"\input{clean_map.tex}",
                r"\input{clean_map_all_metrics.tex}",
                r"\input{clean_per_category_ap.tex}",
                r"\input{" + figure_input + "}",
                "",
            ]
        ),
    )
    markdown_paths = _write_markdown_reports(
        output_dir,
        ordered,
        registry,
        dataset,
        split,
        report_context,
        complete=complete,
        figure_name=figure_paths["png"].name,
    )
    if include_manifest:
        _write_artifact_manifest(
            manifest_path,
            output_dir,
            dataset=dataset,
            complete=complete,
        )

    return {
        "csv": csv_path,
        "tex": tex_path,
        "all_metrics_tex": all_metrics_path,
        "per_category_csv": category_csv_path,
        "per_category_tex": category_tex_path,
        "complete_tex": complete_tex_path,
        **figure_paths,
        **markdown_paths,
        "manifest": manifest_path,
    }
