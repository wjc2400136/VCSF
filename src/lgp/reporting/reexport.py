"""CPU-only presentation export from saved JSON; never load images or detectors."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ..io import atomic_json
from ..metrics import COCO_BBOX_METRICS
from .coco_summary import write_coco_bbox_summary
from .latex import metric_report_view, write_ablation_reports, write_analysis_reports, write_transfer_reports
from .plots import write_experiment_plots
from .scope import SCOPE_FIELDS, escape_scope, scope_csv, scope_for_record, scope_from_records, scope_label
from .preprocessing_cost import generation_cost_metadata, record_cost_metadata


NATIVE_COUNTERS = ("actual_logical_updates", "detector_input_backwards",
                   "feature_partial_input_backwards", "clean_reference_image_rows",
                   "adversarial_differentiable_views", "calibrated_whole_detector_BE")


def _read(path, bindings, snapshots=None):
    raw = path.read_bytes()
    bindings.append(dict(file=str(path), sha256=hashlib.sha256(raw).hexdigest()))
    if snapshots is not None:
        snapshots[path] = raw
    return json.loads(raw)


def _safe_component(value):
    value = str(value or "default")
    if value in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in value):
        raise ValueError("Unsafe saved record path component")
    return value


def _native_reports(records, run, output, parent, bindings, generation_refs=False, directory_key=None):
    observations = []
    seen = set()
    for record in records:
        path = (record.get("generation_run_file") if generation_refs else record.get("adversarial_run"))
        if not path or path in seen:
            continue
        seen.add(path)
        directory = Path(path).resolve().parent if generation_refs else Path(path).resolve()
        if not generation_refs and run not in directory.parents:
            raise ValueError("Native counters must belong to the selected saved run")
        source = directory / "native_cost.jsonl"
        if not source.is_file():
            continue
        raw = source.read_bytes()
        bindings.append(dict(file=str(source), sha256=hashlib.sha256(raw).hexdigest()))
        for line in raw.decode("utf-8").splitlines():
            item = json.loads(line)
            observations.append(dict(
                dataset=item.get("dataset"), split=item.get("split"),
                source=item.get("source"), variant=record.get("variant") or "default",
                generation_cost_scope=record.get("generation_cost_scope"),
                generation_sample_n=record.get("generation_sample_n"),
                image_id=item.get("image_id"), native_diagnostic_only=item.get("diagnostic_only"),
                **{key: item.get(key) for key in NATIVE_COUNTERS},
                report_scope=scope_for_record(record, parent)))
    if not observations:
        return
    directory = output / "reports" / _safe_component(directory_key or parent.get("dataset")) / "native_counts"
    directory.mkdir(parents=True)
    atomic_json(directory / "native_counts.json", observations)
    fields = ["dataset", "split", "source", "variant", "image_id", "native_diagnostic_only",
              *NATIVE_COUNTERS, "generation_cost_scope", "generation_sample_n", *SCOPE_FIELDS]
    with (directory / "native_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in observations:
            scope = item["report_scope"]
            row = {key: item.get(key) for key in fields if key not in SCOPE_FIELDS}
            row["native_diagnostic_only"] = (
                "unknown" if item["native_diagnostic_only"] is None else item["native_diagnostic_only"])
            for key in NATIVE_COUNTERS:
                if row[key] is None:
                    row[key] = "NR"
            writer.writerow(dict(row, **scope_csv(scope)))
    lines = [r"\noindent\textbf{" + escape_scope(scope_label(parent) or "Saved native counters") + r"}\par",
             r"\noindent Native per-image counters, not calibrated whole-detector BE. "
             r"Native diagnostic\_only is preserved independently of the report's global image cap.\par",
             r"\begin{longtable}{lllrrrrrrr}",
             r"Source & Variant & Image & Logical & Detector bwd. & Partial bwd. & Clean ref. & Adv. views & Calibrated BE & Native diagnostic \\",
             r"\hline"]
    for item in observations:
        values = [item["source"], item["variant"], item["image_id"],
                  *(item[key] for key in NATIVE_COUNTERS), item["native_diagnostic_only"]]
        lines.append(" & ".join(escape_scope("NR" if value is None else str(value)) for value in values) + r" \\")
    lines.append(r"\end{longtable}")
    (directory / "native_counts.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _preprocessing_records(run, records, plan, bindings):
    inputs = (_read(run / "input_binding.json", bindings) if any(
        row.get("attack") != "clean" and row.get("status") == "complete" for row in records) else {})
    enriched = []
    cache = {}
    entry_refs = {}
    def read_bytes(path):
        raw = path.read_bytes()
        bindings.append(dict(file=str(path), sha256=hashlib.sha256(raw).hexdigest()))
        return raw
    for record in records:
        row = dict(record)
        if row.get("attack") != "clean" and row.get("status") == "complete":
            group = plan["groups"][row["group_index"]]
            if group["index"] != row["group_index"]:
                raise ValueError("Invalid saved preprocessing group index")
            if any(row.get(key) != group[key] for key in (
                    "source", "attack", "defense", "parameters_sha256", "budget_profile")):
                raise ValueError("Saved preprocessing record differs from its declared group")
            entry_ref = row["prepared_view_binding"]
            if group["index"] in entry_refs and entry_ref != entry_refs[group["index"]]:
                raise ValueError("One preprocessing group has inconsistent prepared-view bindings")
            if group["index"] not in cache:
                entry_refs[group["index"]] = entry_ref
                entry_path = Path(entry_ref["file"]).resolve()
                if run not in entry_path.parents:
                    raise ValueError("Prepared view binding escapes the saved run")
                raw = read_bytes(entry_path)
                if hashlib.sha256(raw).hexdigest() != entry_ref["sha256"]:
                    raise ValueError("Prepared view binding changed")
                entry = json.loads(raw)
                if group.get("adaptive_generation"):
                    original = entry["generated_payload"]
                    if Path(original["root"]).resolve() != run / "adaptive_attacks" / "{:06d}".format(group["index"]):
                        raise ValueError("Adaptive accounting belongs to a foreign generation")
                else:
                    original = dict(inputs["payloads"][group["source"]][group["attack"]],
                                    headers=inputs["headers"])
                cache[group["index"]] = record_cost_metadata(
                    generation_cost_metadata(group, original, read_bytes))
            row.update(cache[group["index"]])
        row["max_images"] = plan.get("max_images")
        enriched.append(row)
    return enriched


def _cost_export(registry, run, output, plan, bindings, snapshots):
    from ..runners.current_cost import write_reports
    summary = _read(run / "summary.json", bindings, snapshots)
    bound = _read(run / "input_binding.json", bindings, snapshots) if (
        run / "input_binding.json").is_file() else plan
    for key in ("protocol", "dataset", "split", "max_images", "diagnostic_only", "methods"):
        if bound.get(key) != plan.get(key):
            raise ValueError("Saved cost binding differs from its original plan")
    expected = {(row["source"], row["method"]) for row in bound["groups"]}
    observed = [(row["source"], row["method"]) for row in summary["groups"]]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("Export cost runs separately; missing, duplicate or foreign groups")
    measurements = run / "measurements.json"
    if measurements.is_file():
        _read(measurements, bindings, snapshots)
    output.mkdir(parents=True)
    for path, raw in snapshots.items():
        (output / path.name).write_bytes(raw)
    write_reports(registry, output, bound, summary)
    for binding in bindings:
        if hashlib.sha256(Path(binding["file"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("Saved report input changed during export: " + binding["file"])
    atomic_json(output / "report_export.json", dict(
        schema="saved_paired_cost_label_export_v1",
        input_kind="paired_cost_plan_and_summary_no_records_required",
        source_bindings=bindings, immutable_input_copies=list(path.name for path in snapshots),
        records_invented=False, measurements_recomputed=False, missing_counts="NR",
        purpose="presentation scope and declared/observed count units only",
        evaluation_calls=0, model_calls=0, GPU_calls=0, predictions_read=False,
        scientific_acceptance=False))
    return output


def _radius_export(registry, output, radius_plan, bindings, snapshots):
    from ..runners.current_radius import PROTOCOL_ID, write_reports
    selected = radius_plan.get("radius_selection") if isinstance(radius_plan, dict) else None
    if (not isinstance(radius_plan, dict) or radius_plan.get("protocol") != PROTOCOL_ID
            or not isinstance(selected, dict)):
        raise ValueError("Saved current radius requires its original radius selection")
    output.mkdir(parents=True)
    for path, raw in snapshots.items():
        (output / path.name).write_bytes(raw)
    write_reports(registry, output, selected)
    for binding in bindings:
        if hashlib.sha256(Path(binding["file"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("Saved report input changed during export: " + binding["file"])
    atomic_json(output / "report_export.json", dict(
        schema="saved_current_radius_specialized_report_export_v1",
        input_kind="current_radius_records_and_original_selection",
        source_bindings=bindings, immutable_input_copies=list(path.name for path in snapshots),
        raw_records_sha256=hashlib.sha256((output / "records.json").read_bytes()).hexdigest(),
        route="lgp.runners.current_radius.write_reports",
        generic_ablation_projection=False, generic_scalar_plots=False,
        predictions_read=False, evaluation_calls=0, model_calls=0, GPU_calls=0,
        scientific_acceptance=False))
    return output


def reexport_saved_reports(registry, run, output, *, plots=False, native_counts=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Report export needs a fresh output directory")
    if run == output or run in output.parents or output in run.parents:
        raise ValueError("Report output must not overlap the immutable saved run")
    for name in ("src", "configs", "experiments", "tests", "data", "checkpoints"):
        protected = (registry.root / name).resolve()
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError("Report output overlaps a protected project tree")
    bindings, snapshots = [], {}
    plan = _read(run / "plan.json", bindings, snapshots) if (run / "plan.json").is_file() else {}
    if plan.get("protocol") == "current_paired_cost":
        if plots or native_counts:
            raise ValueError("Paired-cost input uses saved plan/summary, not scalar AP plots or native image archives")
        return _cost_export(registry, run, output, plan, bindings, snapshots)
    records = _read(run / "records.json", bindings, snapshots)
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError("Saved records must be a JSON list of objects")
    if (run / "radius_plan.json").is_file():
        if plots or native_counts:
            raise ValueError("Current radius export uses its specialized source-by-radius tables, not generic plots or native counts")
        radius_plan = _read(run / "radius_plan.json", bindings, snapshots)
        if plan and plan.get("protocol") != "current_vcsf_radius":
            raise ValueError("Saved current radius has a conflicting generic plan")
        return _radius_export(registry, output, radius_plan, bindings, snapshots)
    if plan.get("protocol") == "current_vcsf_radius":
        raise ValueError("Saved current radius export requires radius_plan.json; generic ablation fallback is unsafe")
    summary = _read(run / "summary.json", bindings) if (run / "summary.json").is_file() else {}
    background = _read(run / "background_plan.json", bindings) if (run / "background_plan.json").is_file() else None
    preprocessing = plan.get("protocol") in (
        "current_oblivious_preprocessing_transfer", "current_adaptive_preprocessing_transfer")
    if preprocessing:
        records = _preprocessing_records(run, records, plan, bindings)
    output.mkdir(parents=True)
    (output / "records.json").write_bytes(snapshots[run / "records.json"])
    if preprocessing:
        atomic_json(output / "presentation_records.json", records)
    if plan.get("protocol") == "current_training_state_transfer":
        from ..runners.current_training_state import _reports
        if plots or native_counts:
            raise ValueError("Training-state export supports its saved table records only")
        keys = [(row["training_state"], row["attack"], row["scope"]) for row in records]
        if len(set(keys)) != len(keys) or any(row["scope"] not in ("fullval", "retained500") for row in records):
            raise ValueError("Repeated or foreign saved training-state scope")
        _reports(registry, output, plan, records)
        for binding in bindings:
            if hashlib.sha256(Path(binding["file"]).read_bytes()).hexdigest() != binding["sha256"]:
                raise RuntimeError("Saved report input changed during export: " + binding["file"])
        atomic_json(output / "report_export.json", dict(
            schema="saved_training_state_label_export_v1", input_kind="training_state_records",
            source_bindings=bindings, raw_records_sha256=hashlib.sha256((output / "records.json").read_bytes()).hexdigest(),
            predictions_read=False, evaluation_calls=0, model_calls=0, GPU_calls=0,
            scientific_acceptance=False))
        return output
    keys = set()
    for record in records:
        key = tuple(_safe_component(record.get(name)) for name in
                    (("defense", "dataset", "source", "attack", "variant", "target") if preprocessing
                     else ("dataset", "source", "attack", "variant", "target")))
        if key in keys:
            raise ValueError("Repeated saved report cell; export runs separately")
        keys.add(key)
    sections = [(dataset, None) for dataset in dict.fromkeys(row.get("dataset") for row in records)]
    if preprocessing:
        sections = [(plan["dataset"], defense) for defense in plan["defenses"]]
        from ..runners.current_preprocessing import _reports
        preprocessing_views = _reports(registry, output, plan, records)
    for dataset, defense in sections:
        if dataset not in registry.datasets:
            raise ValueError("Unregistered saved dataset")
        rows = [row for row in records if row.get("dataset") == dataset
                and (defense is None or row.get("defense") == defense)]
        splits = {row.get("split") for row in rows if row.get("split") is not None}
        if len(splits) > 1:
            raise ValueError("Export saved splits separately; never merge split scopes")
        cap = summary.get("max_images")
        diagnostic = True if cap is not None else (
            False if summary.get("formal_eligible") is True else None)
        if background:
            selection = background["background_selection"]
            cap, diagnostic = selection.get("max_images"), selection.get("diagnostic_only")
        if preprocessing:
            cap, diagnostic = plan.get("max_images"), plan.get("diagnostic_only")
        scope = scope_from_records(rows, dataset=dataset, split=next(iter(splits)) if splits else None,
                                   max_images=cap, diagnostic_only=diagnostic)
        directory = output / "reports" / (defense or dataset)
        if preprocessing:
            pass  # The canonical preprocessing reporter has already written all twelve transfer tables.
        elif any(row.get("study") for row in rows):
            for metric in COCO_BBOX_METRICS:
                write_ablation_reports(rows, registry, directory, dataset=dataset, metric=metric,
                                       report_scope=scope)
        else:
            for metric in COCO_BBOX_METRICS:
                write_transfer_reports(rows, registry, directory, dataset=dataset, metric=metric,
                                       source_ids=plan.get("sources"), target_ids=plan.get("targets"),
                                       method_ids=plan.get("methods"), report_scope=scope)
        display_rows = preprocessing_views[defense] if preprocessing else metric_report_view(rows)
        write_analysis_reports(display_rows, registry, directory, dataset=dataset,
                               include_qualitative_payload_pairs=False, report_scope=scope,
                               include_saved_visualizations=False)
        if plots:
            write_experiment_plots(display_rows, registry, directory / "figures", dataset=dataset,
                                   method_ids=plan.get("methods"), report_scope=scope,
                                   include_saved_diagnostics=False)
        for record in rows:
            metrics = record.get("metrics") or {}
            if not all(type(metrics.get(key)) in (int, float) for key in COCO_BBOX_METRICS):
                continue
            parts = [_safe_component(record.get(name)) for name in ("dataset", "source", "attack", "variant", "target")]
            if record.get("attack") == "clean":
                parts = [parts[0], "clean", parts[-1]]
            if defense is not None:
                parts.insert(0, _safe_component(defense))
            write_coco_bbox_summary([metrics[key] for key in COCO_BBOX_METRICS],
                                    output.joinpath("evaluations", *parts),
                                    dataset=dataset, target=record.get("target"),
                                    source=record.get("source"), attack=record.get("attack"),
                                    report_scope=scope_for_record(record, scope))
        if native_counts:
            if preprocessing:
                _native_reports(rows, run, output, scope, bindings, True, defense)
            else:
                _native_reports(rows, run, output, scope, bindings)
    if background:
        from ..runners.current_background import write_reports
        write_reports(registry, output, background["background_selection"])
    for binding in bindings:
        if hashlib.sha256(Path(binding["file"]).read_bytes()).hexdigest() != binding["sha256"]:
            raise RuntimeError("Saved report input changed during export: " + binding["file"])
    atomic_json(output / "report_export.json", dict(
        schema="saved_report_label_export_v1", source_bindings=bindings,
        purpose="presentation scope and declared/observed cost units only",
        raw_records_sha256=hashlib.sha256((output / "records.json").read_bytes()).hexdigest(),
        input_kind="preprocessing_records_with_bound_generation_metadata" if preprocessing else "saved_records",
        original_records_unchanged=True, presentation_metadata_enriched=preprocessing,
        plots=plots, native_counts=native_counts, predictions_read=False,
        evaluation_calls=0, model_calls=0, GPU_calls=0, scientific_acceptance=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="Immutable saved records run, or current_paired_cost plan/summary run")
    parser.add_argument("--output", type=Path, required=True, help="New, non-overlapping report directory")
    parser.add_argument("--registry-root", type=Path, help="Read-only frozen registry/source root; defaults to this project")
    parser.add_argument("--plots", action="store_true", help="Render CPU PNG/PDF from saved scalar records only")
    parser.add_argument("--native-counts", action="store_true", help="Copy saved per-image native counters; no new measurement")
    args = parser.parse_args(argv)
    from ..registry import Registry
    result = reexport_saved_reports(Registry(args.registry_root), args.run, args.output,
                                   plots=args.plots, native_counts=args.native_counts)
    print("[OUTPUT] " + str(result), flush=True)
    return 0
