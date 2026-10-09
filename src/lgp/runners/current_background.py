"""Fresh-user selected-A10 final-background attribution; no historical-result dependency."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from ..attacks.current_vcsf_background import PROTOCOL_ID, registered_parameters
from ..attacks.vcsf_public import _hash
from ..io import atomic_json
from ..metrics import COCO_BBOX_METRICS
from ..reporting import write_transfer_reports
from ..reporting.scope import SCOPE_FIELDS, scope_csv, scope_from_records, scope_label, scoped_tex
from .experiment import build_plan as build_experiment_plan, run_experiment
from .formal_parallel import normalize_execution_devices


def resolve_selection(registry, targets=None, devices=None, max_images=None):
    spec = registry.protocols[PROTOCOL_ID]
    canonical = registry.target_ids()
    requested = list(canonical) if targets is None else list(targets)
    if (not requested or any(type(target) is not str or target not in canonical for target in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("Select a unique canonical target subset")
    selected = [target for target in canonical if target in requested]
    execution = normalize_execution_devices("cuda:0", devices)
    if len(execution) not in spec["device_counts"] or any(
            not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device) for device in execution):
        raise ValueError("Final-background reproduction requires one or two distinct explicit CUDA devices")
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= spec["full_images"]):
        raise ValueError("max_images must be within the full COCO validation population")
    study = registry.ablation_studies[spec["canonical_variant_study"]]
    return dict(targets=selected, sources=list(spec["sources"]), variants=list(study["row_order"]),
        anchor=study["anchor"], devices=execution, max_images=max_images,
        diagnostic_only=max_images is not None or selected != canonical)


def build_plan(registry, **options):
    selected = resolve_selection(registry, **options)
    plan = build_experiment_plan(registry, PROTOCOL_ID, target_override=selected["targets"])
    for job in plan["jobs"]:
        values = registered_parameters(registry, job["variant"])
        job["resolved_parameters_sha256"] = _hash(values)
    return dict(plan, background_selection=selected, historical_result_inheritance=False,
        configuration_search=False, scientific_acceptance=False)


def _display(value, status="NR"):
    return status if value is None else "{:.4f}".format(value * 100.0)


def _status(rows):
    if rows and all(row.get("status") == "complete" for row in rows):
        return "complete"
    return "ERR" if any(str(row.get("status", "")).startswith("failed")
                        or row.get("status") == "complete_with_failures" for row in rows) else "NR"


def write_reports(registry, output, selected):
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    report_scope = scope_from_records(
        records, dataset="coco", split="val", max_images=selected.get("max_images"),
        diagnostic_only=selected.get("diagnostic_only"))
    panel = [row for row in records if row.get("attack") == "vcsf"]
    expected = {(variant, target) for variant in selected["variants"] for target in selected["targets"]}
    cells = {}
    for row in panel:
        key = (row.get("variant"), row.get("target"))
        if key not in expected or key in cells or row.get("source") != selected["sources"][0]:
            raise ValueError("Repeated or unregistered final-background result cell")
        metrics = row.get("metrics") or {}
        if row.get("status") == "complete" and (
                set(metrics) != set(COCO_BBOX_METRICS) or any(
                    isinstance(metrics[metric], bool) or not isinstance(metrics[metric], (int, float))
                    or not math.isfinite(metrics[metric]) for metric in COCO_BBOX_METRICS)):
            raise ValueError("Complete background cells require all twelve finite official metrics")
        cells[key] = row
    if set(cells) != expected:
        raise ValueError("Final-background records do not cover the entire selected ten-row panel")
    source = selected["sources"][0]
    blackbox = [target for target in selected["targets"] if target != source]
    means, contrasts = [], []
    clean = [row for row in records if row.get("attack") == "clean"]
    for variant in selected["variants"]:
        rows = [cells[variant, target] for target in blackbox]
        status = _status(rows)
        complete = status == "complete"
        means.append(dict(variant=variant, status=status, excluded_whitebox_target=source, targets=blackbox,
            metrics={metric: sum(row["metrics"][metric] for row in rows) / len(rows)
                     if complete else None for metric in COCO_BBOX_METRICS}))
        target_panel = [cells[variant, target] for target in selected["targets"]]
        projection = [dict(row, metrics=(row.get("metrics") or {})
                           if row.get("status") == "complete" else {}) for row in clean + target_panel]
        for metric in COCO_BBOX_METRICS:
            write_transfer_reports(projection, registry, output / "reports/background" / variant,
                dataset="coco", metric=metric, source_ids=selected["sources"],
                target_ids=selected["targets"], method_ids=["vcsf"], preserve_method_order=True,
                include_failure_markers=True, require_complete_panel=True, decimal_places=4,
                report_scope=scope_from_records(
                    projection, dataset="coco", split="val", max_images=selected.get("max_images"),
                    diagnostic_only=selected.get("diagnostic_only")),
                caption=("VCSF final-background control " + variant + "; " + metric
                    + ". BB Mean excludes the source-matched target; values are AP/AR percentage points. "
                    + ("Bounded diagnostic, not formal mAP." if selected["diagnostic_only"]
                       else "Fresh-input full-population reproduction; independent acceptance is separate.")),
                label="tab:current_background_" + variant + "_" + metric)
    by_variant = {row["variant"]: row for row in means}
    anchor = selected["anchor"]
    for variant in selected["variants"]:
        if variant == anchor:
            continue
        deltas = []
        for target in selected["targets"]:
            a, b = cells[anchor, target], cells[variant, target]
            complete = a.get("status") == b.get("status") == "complete"
            deltas.append(dict(target=target, status=_status([a, b]), metrics={metric:
                b["metrics"][metric] - a["metrics"][metric] if complete else None
                for metric in COCO_BBOX_METRICS}))
        mean_status = ("complete" if by_variant[variant]["status"] == by_variant[anchor]["status"] == "complete"
                       else "ERR" if "ERR" in (by_variant[variant]["status"], by_variant[anchor]["status"]) else "NR")
        contrasts.append(dict(variant=variant, anchor=anchor, status=mean_status, direction="control_minus_A10",
            per_target=deltas, BB_Mean={metric:
                by_variant[variant]["metrics"][metric] - by_variant[anchor]["metrics"][metric]
                if by_variant[variant]["metrics"][metric] is not None
                and by_variant[anchor]["metrics"][metric] is not None else None
                for metric in COCO_BBOX_METRICS}))
    atomic_json(output / "background_summary.json", dict(schema_version=1, protocol=PROTOCOL_ID,
        selection=selected, canonical_study=registry.protocols[PROTOCOL_ID]["canonical_variant_study"],
        cells=[cells[variant, target] for variant in selected["variants"] for target in selected["targets"]],
        BB_Mean=means, contrasts=contrasts, raw_metric_scale="official_unrounded_0_to_1",
        contrast_direction="control_minus_A10", expected_attack_cells=len(expected),
        historical_result_inheritance=False, configuration_search=False, scientific_acceptance=False,
        project_wide_device_release_accepted=False))
    directory = output / "reports/background"
    directory.mkdir(parents=True, exist_ok=True)
    delta_by_variant = {row["variant"]: row for row in contrasts}
    lines = [r"\begin{longtable}{llrr}", r"Variant & Metric & BB Mean & Control $-$ A10 \\", r"\hline"]
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["variant", "metric", "BB_Mean_AP_AR_points", "control_minus_A10_AP_AR_points"]
                        + (list(SCOPE_FIELDS) if scope_label(report_scope) else []))
        for row in means:
            for metric in COCO_BBOX_METRICS:
                value = row["metrics"][metric]
                contrast = delta_by_variant.get(row["variant"], {})
                delta = contrast.get("BB_Mean", {}).get(metric)
                shown_value = _display(value, row["status"])
                shown_delta = ("reference" if row["variant"] == anchor else
                               _display(delta, contrast.get("status", "NR")))
                row_scope = scope_from_records(
                    [cells[row["variant"], target] for target in blackbox], dataset="coco", split="val",
                    max_images=selected.get("max_images"), diagnostic_only=selected.get("diagnostic_only"))
                writer.writerow([row["variant"], metric, shown_value, shown_delta]
                                + (list(scope_csv(row_scope).values()) if scope_label(report_scope) else []))
                lines.append("{} & {} & {} & {} \\\\".format(
                    row["variant"], metric.replace("_", r"\_"), shown_value, shown_delta))
    lines.append(r"\end{longtable}")
    (directory / "summary.tex").write_text(scoped_tex("\n".join(lines) + "\n", report_scope),
                                           encoding="utf-8")


def run(registry, *, targets=None, devices=None, max_images=None, output=None,
        plan_only=False, save_visualizations=False):
    plan = build_plan(registry, targets=targets, devices=devices, max_images=max_images)
    selected = plan["background_selection"]
    if not plan_only:
        from ..data.coco import CocoIndex

        index = CocoIndex(registry.dataset("coco"), "val")
        ids = [image["id"] for image in index.images]
        if len(ids) != registry.protocols[PROTOCOL_ID]["full_images"] or ids != sorted(set(ids)):
            raise ValueError("Background reproduction requires the full canonical COCO validation index")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output or registry.root / "outputs/experiments" / PROTOCOL_ID / stamp).resolve()
    protected = [(registry.root / value).resolve() for value in ("src", "configs", "tools", "checkpoints")]
    protected.append(registry.dataset("coco").root.resolve())
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        raise ValueError("Background output overlaps protected code, data or checkpoints")
    result = run_experiment(registry, PROTOCOL_ID, execute=not plan_only, output_dir=directory,
        target_override=selected["targets"], max_images=max_images, devices=selected["devices"],
        seed=registry.protocols[PROTOCOL_ID]["seed"], download_weights=False, keep_going=False,
        payload_retention="keep_all", prediction_archive="gzip", save_visualizations=save_visualizations,
        visualization_score_threshold=0.50, visualization_max_images=3, visualization_max_detections=100)
    try:
        atomic_json(result / "background_plan.json", plan)
        write_reports(registry, result, selected)
    except Exception as exc:
        atomic_json(result / "background_failure.json", dict(status="failed", stage="background_report_projection",
            reason=str(exc), scientific_acceptance=False, project_wide_device_release_accepted=False))
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--devices", help="One or two explicit CUDA devices; default cuda:0")
    device.add_argument("--device", help="One explicit CUDA device")
    parser.add_argument("--targets", help="Canonical target subset; bounded diagnostic")
    parser.add_argument("--max-images", type=int, help="Bounded diagnostic image limit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--visualize-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.device is not None and "," in args.device:
        parser.error("--device accepts one device; use --devices for independent complete jobs")
    from ..registry import Registry
    from ..paths import project_root

    result = run(Registry(project_root()), targets=None if args.targets is None else
        [item.strip() for item in args.targets.split(",")], devices=
        [item.strip() for item in (args.devices or args.device or "cuda:0").split(",")],
        max_images=args.max_images, output=args.output, plan_only=args.plan_only,
        save_visualizations=args.visualize_predictions)
    print("[OUTPUT] " + str(result), flush=True)
    return 0