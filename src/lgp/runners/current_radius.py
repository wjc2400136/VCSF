"""Fresh-input selected-A10 COCO Common-2 five-radius reproduction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

from ..attacks.current_vcsf_radius import PROTOCOL_ID, validate_parameters
from ..attacks.vcsf_public import _hash
from ..io import atomic_json
from ..metrics import COCO_BBOX_METRICS
from ..reporting import write_transfer_reports
from ..reporting.scope import scope_from_records
from .experiment import build_plan as build_experiment_plan, run_experiment
from .formal_parallel import normalize_execution_devices


def selection(requested, canonical, name):
    values = list(canonical) if requested is None else list(requested)
    if (not values or any(type(value) is not str for value in values)
            or len(set(values)) != len(values) or any(value not in canonical for value in values)):
        raise ValueError("Select a unique registered " + name + " subset")
    return [value for value in canonical if value in values]


def resolve_selection(registry, sources=None, targets=None, epsilons=None, devices=None, max_images=None):
    spec = registry.protocols[PROTOCOL_ID]
    sources = selection(sources, spec["sources"], "source")
    targets = selection(targets, registry.target_ids(), "target")
    epsilons = selection(epsilons, spec["epsilon_order"], "epsilon")
    execution = normalize_execution_devices("cuda:0", devices)
    if len(execution) not in spec["device_counts"] or any(
            not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) for value in execution):
        raise ValueError("Current radius requires one or two distinct explicit CUDA devices")
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= spec["full_images"]):
        raise ValueError("max_images must be within the full COCO validation population")
    studies = [study for epsilon, study in zip(spec["epsilon_order"], spec["ablation_studies"])
               if epsilon in epsilons]
    return dict(sources=sources, targets=targets, epsilons=epsilons,
        studies=studies, devices=execution, max_images=max_images,
        diagnostic_only=(max_images is not None or sources != spec["sources"]
                         or targets != registry.target_ids() or epsilons != spec["epsilon_order"]))


def build_plan(registry, **options):
    selected = resolve_selection(registry, **options)
    plan = build_experiment_plan(registry, PROTOCOL_ID, source_override=selected["sources"],
        target_override=selected["targets"], method_override=["vcsf"], study_override=selected["studies"])
    for job in plan["jobs"]:
        parameters = dict(registry.attack("vcsf").parameters, **job["parameters"])
        values, epsilon, identity = validate_parameters(registry, parameters)
        if epsilon not in selected["epsilons"]:
            raise ValueError("Radius plan contains an undeclared epsilon")
        job["resolved_parameters_sha256"] = _hash(values)
    return dict(plan, radius_selection=selected)


def write_reports(registry, output, selected):
    spec = registry.protocols[PROTOCOL_ID]
    records = json.loads((output / "records.json").read_text(encoding="utf-8"))
    study_map = dict(zip(spec["epsilon_order"], spec["ablation_studies"]))
    families = []
    for epsilon in selected["epsilons"]:
        study = study_map[epsilon]
        panel = [row for row in records if row.get("study") == study and row.get("attack") == "vcsf"]
        by_key = {}
        for row in panel:
            key = (row["source"], row["target"])
            if key in by_key or key[0] not in selected["sources"] or key[1] not in selected["targets"]:
                raise ValueError("Repeated or unselected radius result cell")
            metrics = row.get("metrics") or {}
            if row.get("status") == "complete" and (
                    set(metrics) != set(COCO_BBOX_METRICS) or any(
                        isinstance(metrics[k], bool) or not isinstance(metrics[k], (int, float))
                        or not math.isfinite(metrics[k]) for k in COCO_BBOX_METRICS)):
                raise ValueError("Complete radius result must preserve all twelve finite official metrics")
            by_key[key] = row
        expected = {(source, target) for source in selected["sources"] for target in selected["targets"]}
        if set(by_key) != expected:
            raise ValueError("Radius result records do not cover the selected target panel")
        means = []
        for source in selected["sources"]:
            targets = [target for target in selected["targets"] if target != source]
            rows = [by_key[source, target] for target in targets]
            complete = bool(rows) and all(row.get("status") == "complete" for row in rows)
            means.append(dict(source=source, excluded_whitebox_target=source, targets=targets,
                metrics={key: sum(row["metrics"][key] for row in rows) / len(rows)
                         if complete and all(row["metrics"][key] >= 0 for row in rows)
                         else None for key in COCO_BBOX_METRICS}))
        report_dir = output / "reports/radius" / epsilon.replace("/", "_")
        clean = [row for row in records if row.get("attack") == "clean"]
        projection = [dict(row, metrics=row.get("metrics", {})
                           if row.get("status") == "complete" else {}) for row in clean + panel]
        report_scope = scope_from_records(clean + panel, dataset="coco", split="val",
            max_images=selected.get("max_images"), diagnostic_only=selected["diagnostic_only"])
        for metric in COCO_BBOX_METRICS:
            write_transfer_reports(projection, registry, report_dir, dataset="coco", metric=metric,
                source_ids=selected["sources"], target_ids=selected["targets"], method_ids=["vcsf"],
                preserve_method_order=True, include_failure_markers=True, require_complete_panel=True,
                decimal_places=4,
                report_scope=report_scope,
                caption=("VCSF selected A10 at epsilon " + epsilon + "; " + metric
                    + (". High-distortion stress point." if epsilon in spec["stress_epsilons"] else ".")
                    + " BB Mean excludes the source-matched target; values are AP/AR percentage points. "
                    + ("Bounded diagnostic, not formal mAP." if selected["diagnostic_only"]
                       else "Fresh-input full-population reproduction; independent acceptance is separate.")),
                label="tab:current_radius_" + epsilon.replace("/", "_") + "_" + metric)
        families.append(dict(epsilon=epsilon, stress_only=epsilon in spec["stress_epsilons"],
            study=study, report_scope=report_scope,
            cells=[by_key[source, target] for source in selected["sources"]
                               for target in selected["targets"]], BB_Mean=means))
    atomic_json(output / "radius_summary.json", dict(schema_version=1, protocol=PROTOCOL_ID,
        selection=selected, families=families, raw_metric_scale="official_unrounded_0_to_1",
        reference_epsilon=spec["reference_epsilon"], expected_attack_cells=sum(len(v["cells"]) for v in families),
        historical_result_inheritance=False, scientific_acceptance=False,
        project_wide_device_release_accepted=False))


def run(registry, *, sources=None, targets=None, epsilons=None, devices=None, max_images=None,
        output=None, plan_only=False, save_visualizations=False):
    plan = build_plan(registry, sources=sources, targets=targets, epsilons=epsilons,
                      devices=devices, max_images=max_images)
    selected = plan["radius_selection"]
    if not plan_only:
        from ..data.coco import CocoIndex

        index = CocoIndex(registry.dataset("coco"), "val")
        ids = [image["id"] for image in index.images]
        if len(ids) != registry.protocols[PROTOCOL_ID]["full_images"] or ids != sorted(set(ids)):
            raise ValueError("Current radius requires the complete canonical COCO validation index")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output or registry.root / "outputs/experiments" / PROTOCOL_ID / stamp).resolve()
    protected = [(registry.root / value).resolve() for value in ("src", "configs", "checkpoints")]
    protected.append(registry.dataset("coco").root.resolve())
    if any(directory == path or path in directory.parents or directory in path.parents for path in protected):
        raise ValueError("Radius output overlaps protected code, data or checkpoint inputs")
    result = run_experiment(registry, PROTOCOL_ID, execute=not plan_only, output_dir=directory,
        dataset_override=["coco"], source_override=selected["sources"], target_override=selected["targets"],
        method_override=["vcsf"], study_override=selected["studies"], max_images=max_images,
        seed=registry.protocols[PROTOCOL_ID]["seed"], devices=selected["devices"],
        download_weights=False, keep_going=False, payload_retention="keep_all", prediction_archive="gzip",
        save_visualizations=save_visualizations, visualization_score_threshold=0.50,
        visualization_max_images=3, visualization_max_detections=100,
        generic_report_projections=False)
    try:
        atomic_json(result / "radius_plan.json", plan)
        write_reports(registry, result, selected)
    except Exception as exc:
        atomic_json(result / "radius_failure.json", dict(status="failed", stage="radius_report_projection",
            reason=str(exc), scientific_acceptance=False, project_wide_device_release_accepted=False))
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    device = parser.add_mutually_exclusive_group()
    device.add_argument("--devices", help="One or two explicit CUDA devices; default cuda:0")
    device.add_argument("--device", help="One explicit CUDA device")
    parser.add_argument("--sources", help="Registered Common-2 source subset")
    parser.add_argument("--targets", help="Canonical target subset; bounded diagnostic")
    parser.add_argument("--epsilons", help="Subset of the five registered exact rational radii")
    parser.add_argument("--max-images", type=int, help="Bounded diagnostic image limit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--visualize-predictions", action="store_true")
    args = parser.parse_args(argv)
    if args.device is not None and "," in args.device:
        parser.error("--device accepts one device; use --devices for independent dual-GPU jobs")
    from ..registry import Registry
    from ..paths import project_root
    def csv(value):
        return None if value is None else [item.strip() for item in value.split(",")]
    result = run(Registry(project_root()), sources=csv(args.sources), targets=csv(args.targets),
        epsilons=csv(args.epsilons), devices=csv(args.devices or args.device or "cuda:0"),
        max_images=args.max_images, output=args.output, plan_only=args.plan_only,
        save_visualizations=args.visualize_predictions)
    print("[OUTPUT] " + str(result), flush=True)
    return 0
