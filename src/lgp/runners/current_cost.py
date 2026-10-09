"""Current-input paired attack cost measurement without inherited AP results."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re
import traceback

from ..io import atomic_json, file_digest
from ..reporting.cost_units import COST_NOTE, count_unit
from ..reporting.scope import escape_scope, scope_csv, scope_from_records, scope_label
from . import cost_calibration as calibration
from .current_reproduction_inputs import _bind, _unchanged
from .formal_parallel import normalize_execution_devices, validate_available_cuda_devices

PROTOCOL_ID = "current_paired_cost"


def build_plan(registry, methods=None, devices=None, max_images=None):
    spec = registry.protocols[PROTOCOL_ID]
    main = registry.protocols[spec["method_source_protocol"]]
    defaults = list(main["methods"])
    requested = defaults if methods is None else list(methods)
    if (not requested or any(not isinstance(value, str) for value in requested)
            or len(set(requested)) != len(requested)
            or any(value not in defaults for value in requested)):
        raise ValueError("Select unique methods from the current main comparison")
    selected = [value for value in defaults if value in requested]
    if len(selected) % 2 or spec["reference_method"] not in selected:
        raise ValueError("Paired Williams measurement requires an even panel including VCSF")
    execution = normalize_execution_devices("cuda:0", devices)
    if (len(execution) not in spec["device_counts"]
            or any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) for value in execution)):
        raise ValueError("Select one or two distinct explicit CUDA devices")
    count = spec["retained_images"]
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= count):
        raise ValueError("max_images must be a positive integer within the retained population")
    from ..attacks.factory import ATTACK_TYPES
    from ..attacks.vcsf_public import verify_public_identity
    identity = verify_public_identity(registry.root, registry.attack("vcsf"))
    groups = []
    for source in spec["sources"]:
        for method in selected:
            if registry.compatibility_status(method, source)["status"] != "native":
                raise ValueError("Unsupported cost source-method group: " + source + "/" + method)
            config = ATTACK_TYPES[method][1].from_mapping(dict(registry.attack(method).parameters))
            config.validate()
            parameters = asdict(config)
            if parameters.get("eps") != 4.0 / 255.0:
                raise ValueError("Current cost measurement requires the registered 4/255 configuration")
            groups.append(dict(source=source, method=method, parameters=parameters,
                parameters_sha256=calibration.digest(parameters),
                budget_profile=main.get("method_budget_profiles", {}).get(method, main["budget_profile"])))
    selected_count = count if max_images is None else max_images
    definition = dict(spec, methods=selected)
    jobs = [dict(index=index, source=source, device=execution[index % len(execution)],
                 worker_slot=index % len(execution))
            for index, source in enumerate(spec["sources"])]
    return dict(schema_version=1, protocol=PROTOCOL_ID, purpose="physical_cost_only_no_AP",
        dataset=spec["datasets"][0], split=spec["split"], definition=definition,
        methods=selected, groups=groups, jobs=jobs, devices=execution,
        execution_schedule=spec["execution_schedule"], concurrent_measurements=1,
        selected_images=selected_count, retained_images=count, full_images=spec["full_images"],
        max_images=max_images, diagnostic_only=max_images is not None or selected != defaults,
        complete_method_selection=selected == defaults,
        expected_measured_calls=len(groups) * selected_count,
        expected_warmup_calls=len(groups) * min(spec["warmup_images"], selected_count),
        source_snapshot=calibration.source_snapshot(registry.root),
        public_identity=identity, evaluation_calls=0, payload_files=0,
        scientific_acceptance=False, historical_result_inheritance=False,
        project_wide_device_release_accepted=False)


def verify_inputs(registry, plan, output):
    from ..data.coco import CocoIndex
    from ..modeling import ensure_checkpoint
    from .retention import build_retained_image_selection
    selection = build_retained_image_selection(registry, plan["dataset"], plan["split"],
        plan["retained_images"], output / "selection.json")
    spec = plan["definition"]
    for key in ("all_image_ids_sha256", "retained_image_ids_sha256"):
        if selection[key] != spec[key]:
            raise ValueError("Canonical cost image selection changed: " + key)
    index = CocoIndex(registry.dataset(plan["dataset"]), plan["split"])
    ids = [image["id"] for image in index.images]
    if (len(ids) != plan["full_images"] or ids != sorted(set(ids))
            or any(type(value) is not int for value in ids)):
        raise ValueError("Current cost requires the complete canonical COCO validation index")
    selected = selection["retained"][:plan["selected_images"]]
    images = {image["id"]: image for image in index.images}
    image_hashes = {str(item["image_id"]): file_digest(index.image_path(images[item["image_id"]]))
                    for item in selected}
    headers = [_bind(index.annotation_path)]
    checkpoints = {}
    for job in plan["jobs"]:
        path = ensure_checkpoint(registry.model(job["source"]), registry.dataset(plan["dataset"]),
                                 download=False)
        binding = _bind(path)
        headers.append(binding)
        checkpoints[job["source"]] = binding
    bound = dict(plan, selected=selected, image_sha256=image_hashes,
        annotation_sha256=headers[0]["sha256"], input_headers=headers)
    bound["groups"] = [dict(group, checkpoint=checkpoints[group["source"]]["file"],
                           checkpoint_sha256=checkpoints[group["source"]]["sha256"])
                       for group in plan["groups"]]
    return index, bound


def check_inputs(registry, plan):
    for binding in plan["input_headers"]:
        _unchanged(binding)
    if calibration.source_snapshot(registry.root) != plan["source_snapshot"]:
        raise ValueError("Cost execution source or declaration changed")
    from ..attacks.vcsf_public import verify_public_identity
    if verify_public_identity(registry.root) != plan["public_identity"]:
        raise ValueError("Selected current A10 public identity changed")


def instantiate_current(registry, method, adapter, parameters):
    from ..attacks.factory import build_attack
    from .attack import _validate_budget_profile
    attack = build_attack(method, adapter, parameters)
    if calibration.digest(asdict(attack.config)) != calibration.digest(parameters):
        raise ValueError("Current cost builder changed the complete parameter configuration")
    main = registry.protocols[registry.protocols[PROTOCOL_ID]["method_source_protocol"]]
    profile = main.get("method_budget_profiles", {}).get(method, main["budget_profile"])
    _validate_budget_profile(registry, profile, attack)
    return attack


def write_reports(registry, output, plan, summary=None):
    values = summary["groups"] if summary is not None else [
        dict(source=group["source"], method=group["method"], images=None,
             mean_seconds=None, mean_paired_ratio_to_vcsf=None,
             maximum_peak_allocated_mib=None, mean_recorded_logical_gradients=None)
        for group in plan["groups"]]
    values = [dict(row) for row in values]
    group_lookup = {(group["source"], group["method"]): group for group in plan["groups"]}
    for row in values:
        group = group_lookup[(row["source"], row["method"])]
        profile = group.get("budget_profile")
        row["recorded_gradient_count_unit"] = count_unit(dict(
            budget_profile_definition=registry.budget_profiles.get(profile, {})))
        row["declared_budget_profile"] = profile or "unknown"
        origins = row.get("logical_gradient_count_sources") or []
        row["observed_count_evidence"] = ", ".join(origins) or "unknown"
        row["mean_observed_actual_count"] = (row.get("mean_recorded_logical_gradients")
            if origins and not any(value in ("declared_cap", "unavailable") for value in origins) else None)
        row["calibrated_whole_detector_BE"] = row.get("calibrated_whole_detector_BE")
        row.update(scope_csv(scope_from_records(
            dataset=plan.get("dataset"), split=plan.get("split"),
            sample_n=row.get("images"), max_images=plan.get("max_images"),
            diagnostic_only=plan.get("diagnostic_only"))))
    columns = list(dict.fromkeys(key for row in values for key in row))
    def display(value, field):
        if value is None:
            return "NR"
        if type(value) is float or (field.startswith("mean_") and type(value) is int):
            return format(value, ".4f")
        if isinstance(value, (list, dict)):
            return json.dumps(value, sort_keys=True, allow_nan=False)
        return value
    with (output / "cost_by_source_method.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in values:
            writer.writerow({key: display(row.get(key), key) for key in columns})
    def number(value, precision):
        return "NR" if value is None else format(value, "." + str(precision) + "f")
    def label(value):
        return value.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")
    population = scope_from_records(
        [dict(images=row.get("images"), failures=[]) for row in values],
        dataset=plan.get("dataset"), split=plan.get("split"), max_images=plan.get("max_images"),
        diagnostic_only=plan.get("diagnostic_only"))
    lines = [r"\noindent\textbf{" + escape_scope(scope_label(population) or
             "Paired cost; actual dataset={}; split={}; sample N={}".format(
                 plan.get("dataset") or "unknown", plan.get("split") or "unknown",
                 population.get("sample_n") if population.get("sample_n") is not None else "unknown")) + r"}\par",
             r"\noindent " + escape_scope(COST_NOTE) + r"\par",
             "\\begin{tabular}{llrrrrr}", "\\hline",
             "Source & Method & N & Time (s) & Ratio to VCSF & Peak (MiB) & Observed count \\\\", "\\hline"]
    for row in values:
        source = registry.model(row["source"]).display_name
        method = registry.attack(row["method"]).display_name
        lines.append(" & ".join([label(source), label(method),
            str(row["sample_n"]),
            number(row.get("mean_seconds"), 3), number(row.get("mean_paired_ratio_to_vcsf"), 3),
            number(row.get("maximum_peak_allocated_mib"), 2),
            number(row.get("mean_observed_actual_count"), 3)]) + " \\\\")
    lines += ["\\hline", r"\end{tabular}\par"]
    for row in values:
        lines.append(r"\noindent " + label(registry.model(row["source"]).display_name) + ", "
                     + label(registry.attack(row["method"]).display_name) + ": "
                     + label(row["recorded_gradient_count_unit"]) + r".\par")
    lines.append(r"\noindent A saved recorded count with declared\_cap or unavailable origin is not "
                 r"an observed actual count; its observation remains NR.\par")
    (output / "cost_by_source_method.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(registry, *, methods=None, devices=None, max_images=None, output=None, plan_only=False):
    plan = build_plan(registry, methods, devices, max_images)
    if not plan_only:
        validate_available_cuda_devices(plan["devices"], include_single=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = Path(output or registry.root / "outputs/experiments" / PROTOCOL_ID / stamp).resolve()
    protected = [(registry.root / value).resolve() for value in ("src", "configs", "checkpoints")]
    protected.append(registry.dataset(plan["dataset"]).root.resolve())
    if any(directory == path or path in directory.parents or directory in path.parents
           for path in protected):
        raise ValueError("Cost output overlaps protected code, data or checkpoint inputs")
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "plan.json", plan)
    write_reports(registry, directory, plan)
    if plan_only:
        atomic_json(directory / "terminal.json", dict(status="planned", model_calls=0,
            evaluation_calls=0, scientific_acceptance=False, project_wide_device_release_accepted=False))
        return directory
    try:
        environment = calibration.environment(server_only=False)
        index, bound = verify_inputs(registry, plan, directory)
        atomic_json(directory / "input_binding.json", bound)
        atomic_json(directory / "environment.json", environment)
        from .training_pair_processes import process_identity
        from .vcsf_structure_workers import physical_gpus, gpu_reservations
        owner = process_identity(os.getpid())
        atomic_json(directory / "coordinator.json", owner)
        rows = []
        devices = physical_gpus(plan["devices"])
        with gpu_reservations(devices):
            for job in plan["jobs"]:
                calibration.gpu_exclusive()
                check_inputs(registry, bound)
                branch = directory / "sources" / job["source"]
                branch.mkdir(parents=True, exist_ok=False)
                subplan = dict(bound, device=job["device"],
                    definition=dict(bound["definition"], sources=[job["source"]]),
                    groups=[group for group in bound["groups"] if group["source"] == job["source"]])
                subplan["expected_measured_calls"] = len(subplan["groups"]) * len(subplan["selected"])
                atomic_json(branch / "plan.json", subplan)
                atomic_json(directory / "execution_state.json", dict(status="running", owner=owner,
                    current_job=job, completed_source_jobs=job["index"],
                    updated_at=calibration.now(), concurrent_measurements=1))
                calibration.execute(registry, index, subplan, branch,
                    attack_builder=lambda method, adapter, parameters:
                        instantiate_current(registry, method, adapter, parameters))
                check_inputs(registry, bound)
                values = [json.loads(line) for line in (branch / "measurements.jsonl").read_text().splitlines()]
                if len(values) != subplan["expected_measured_calls"]:
                    raise ValueError("Incomplete source measurement prefix")
                rows.extend(values)
        summary = calibration.summarize(rows, bound)
        summary.update(protocol=PROTOCOL_ID, execution_schedule=plan["execution_schedule"],
            concurrent_measurements=1, devices=plan["devices"], owner=owner,
            sources_completed=len(plan["jobs"]), expected_warmup_calls=plan["expected_warmup_calls"],
            parameters_sha256_by_group=[dict(source=group["source"], method=group["method"],
                sha256=group["parameters_sha256"]) for group in plan["groups"]],
            A10_parameters_sha256=plan["public_identity"]["parameters_sha256"],
            scientific_acceptance=False, project_wide_device_release_accepted=False,
            historical_result_inheritance=False, payload_files=0, evaluation_calls=0)
        atomic_json(directory / "measurements.json", rows)
        atomic_json(directory / "summary.json", summary)
        write_reports(registry, directory, bound, summary)
        atomic_json(directory / "terminal.json", dict(status=summary["status"], owner=owner,
            measured_calls=len(rows), sources_completed=len(plan["jobs"]), failures=0,
            summary_sha256=file_digest(directory / "summary.json"),
            scientific_acceptance=False, project_wide_device_release_accepted=False))
        manifest = {path.relative_to(directory).as_posix(): file_digest(path)
                    for path in sorted(directory.rglob("*")) if path.is_file()
                    and path.name not in ("artifact_manifest.json", "execution_state.json")}
        atomic_json(directory / "artifact_manifest.json", manifest)
        atomic_json(directory / "execution_state.json", dict(status=summary["status"],
            updated_at=calibration.now(), owner=owner, measured_calls=len(rows)))
    except BaseException as exc:
        atomic_json(directory / "failure.json", dict(status="failed", reason=str(exc),
            scientific_acceptance=False, project_wide_device_release_accepted=False))
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", help="Even subset of current main methods including vcsf")
    parser.add_argument("--devices", default="cuda:0", help="One or two explicit CUDA devices")
    parser.add_argument("--max-images", type=int, help="Diagnostic retained-image limit; not publishable")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    from ..registry import Registry
    from ..paths import project_root
    methods = None if args.methods is None else [value.strip() for value in args.methods.split(",")]
    result = run(Registry(project_root()), methods=methods,
        devices=[value.strip() for value in args.devices.split(",")],
        max_images=args.max_images, output=args.output, plan_only=args.plan_only)
    print("[OUTPUT] " + str(result), flush=True)
    return 0
