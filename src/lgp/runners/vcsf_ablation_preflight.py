"""One-image source-only stage controls using the existing owned-worker scheduler."""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from ..io import atomic_json, file_digest
from ..reporting.vcsf_ablation_cost import compile_cost_contract, read_stage_plan, recount_group_traces
from .vcsf_ablation_stage_plan import PROTOCOL
from .vcsf_efficacy_contract import child, read_bound, require, verify_environment
from .vcsf_research_plan import canonical_hash


EXECUTION_PROTOCOL = "vcsf_single_source_ablation_preflight"
ENTRYPOINT = "experiments/vcsf_single_source_ablation_preflight.py"
ALIAS = "vcsf_ablation_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "single_source_ablation_preflight"
ALLOW_EMPTY_LANES = True
RUNTIME_FILES = (
    ENTRYPOINT, "src/lgp/runners/vcsf_ablation_preflight.py",
    "src/lgp/attacks/vcsf_ablation_preflight_isolation.py",
    "src/lgp/runners/vcsf_efficacy_runner.py", "src/lgp/runners/vcsf_efficacy_contract.py",
    "src/lgp/runners/vcsf_structure_workers.py", "src/lgp/runners/cost_calibration.py",
    "src/lgp/modeling.py", "src/lgp/runtime_config.py", "src/lgp/data/coco.py",
    "src/lgp/attacks/factory.py", "src/lgp/metrics.py", "configs/datasets/coco.yaml",
    "environment.yml", "requirements/locked-cu118.txt",
)


def verify_server():
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda" and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        "Ablation preflight requires the pinned server ODA interpreter")


def required_packages(root):
    from packaging.requirements import Requirement

    names = {"torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmyolo"}
    result = {}
    for line in (root / "requirements/locked-cu118.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        if requirement.name in names:
            specs = list(requirement.specifier)
            require(requirement.name not in result and len(specs) == 1 and specs[0].operator == "==",
                "Ambiguous pinned core package")
            result[requirement.name] = specs[0].version
    require(set(result) == names, "The locked environment is missing a core package")
    return result


def balanced_lanes(groups, sources, devices):
    require(isinstance(devices, (list, tuple)) and 1 <= len(devices) <= 2
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices), "Select one or two distinct explicit physical CUDA devices")
    require(sources == ["faster_rcnn_r50"] and groups
        and all(g["source"] == sources[0] for g in groups)
        and all(type(g["group_id"]) is int and g["group_id"] > 0 for g in groups)
        and len({g["group_id"] for g in groups}) == len(groups), "Invalid or duplicate source-only group")
    return [groups[i::len(devices)] for i in range(len(devices))]


def compile_diagnostic_plan(registry, stage_plan):
    """Resolve every stage control without reaccepting original stage evidence."""
    cost = compile_cost_contract(registry, stage_plan)
    definition = registry.protocols[EXECUTION_PROTOCOL]
    require(definition["parent_protocol"] == PROTOCOL and definition["images_per_group"] == 1
        and definition["target_AP_evaluations"] == 0 and definition["payload_retention"] == "keep_all",
        "Source-only diagnostic registration changed")
    groups = [dict(deepcopy(g), images=1, execution_status="diagnostic_not_formal",
        original_requested_images=g["images"]) for g in stage_plan["groups"]]
    sources = dict(stage_plan["source_sha256"], **cost["source_sha256"])
    sources.update({p: file_digest(child(registry.root, p)) for p in RUNTIME_FILES})
    result = dict(schema_version=1, record_type=EXECUTION_PROTOCOL,
        status="prepared_source_control_diagnostic_not_formal_admission", protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, stage=stage_plan["selected_stage"],
        definition=deepcopy(definition), definition_sha256=canonical_hash(definition),
        stage_preparation=deepcopy(stage_plan), prepared_plan_sha256=canonical_hash(stage_plan),
        stage_science_sha256=stage_plan["science_sha256"], cost_science_sha256=cost["science_sha256"],
        stage_evidence_reverified=False, groups=groups, new_group_ids=[g["group_id"] for g in groups],
        sources=list(stage_plan["sources"]), targets=[], canonical_report_targets=list(stage_plan["targets"]),
        source_sha256=dict(sorted(sources.items())), max_images=1, seed=42,
        model_calls_before_execution=0, AP_evaluations=0, diagnostic_only=True,
        original_generation_scope_unchanged=True, payload_retention="keep_all",
        deletion_authorized=False, formal_execution_admission=False, scientific_acceptance=False,
        independent_confirmation=False, automatic_promotion=False, historical_reuse_qualified=False,
        physical_cost_calibration_accepted=False,
        timing_scope="per_image_generation_integration_not_isolated_or_randomized_cost_measurement")
    result["science_sha256"] = canonical_hash({k: v for k, v in result.items()
        if k not in ("stage_preparation", "prepared_plan_sha256")})
    return result


def _bind_inputs(registry):
    from ..data.coco import CocoIndex
    from ..modeling import checkpoint_path

    index = CocoIndex(registry.dataset("coco"), "val")
    ids = [int(image["id"]) for image in index.images]
    require(len(ids) == len(set(ids)) == 5000 and ids == sorted(ids), "Expected canonical full COCO validation index")
    image = index.images[0]
    clean = index.image_path(image).resolve()
    checkpoint = checkpoint_path(registry.model("faster_rcnn_r50"), registry.dataset("coco")).resolve()
    return dict(image_id=ids[0], position=0, attack_seed=42, all_image_ids_sha256=canonical_hash(ids),
        image_file=str(clean), image_sha256=file_digest(clean),
        annotation_file=str(index.annotation_path.resolve()), annotation_sha256=file_digest(index.annotation_path),
        checkpoint_file=str(checkpoint), checkpoint_sha256=file_digest(checkpoint))


def bind_runtime(registry, diagnostic):
    from .cost_calibration import environment

    verify_server()
    actual = environment()
    inputs = _bind_inputs(registry)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=registry.root, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Missing original Git revision")
    result = dict(deepcopy(diagnostic), input_identity=inputs, input_identity_sha256=canonical_hash(inputs),
        packages=actual["versions"], base_commit=commit,
        runtime_sha256=deepcopy(diagnostic["source_sha256"]),
        runtime_snapshot_sha256=canonical_hash(diagnostic["source_sha256"]))
    result["runtime_binding_sha256"] = canonical_hash(result)
    return result


def verify_execution_plan(registry, plan):
    diagnostic = compile_diagnostic_plan(registry, plan["stage_preparation"])
    extras = {"input_identity", "input_identity_sha256", "packages", "base_commit",
        "runtime_sha256", "runtime_snapshot_sha256", "runtime_binding_sha256"}
    require(set(plan) == set(diagnostic) | extras
        and all(canonical_hash(plan[k]) == canonical_hash(v) for k, v in diagnostic.items()),
        "The source-control plan changed its full stage or runtime identity")
    require(plan["runtime_sha256"] == diagnostic["source_sha256"]
        and plan["runtime_snapshot_sha256"] == canonical_hash(plan["runtime_sha256"])
        and plan["runtime_binding_sha256"] == canonical_hash({k: v for k, v in plan.items()
            if k != "runtime_binding_sha256"}), "Diagnostic runtime binding changed")
    require(plan["packages"] == required_packages(registry.root)
        and isinstance(plan["base_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", plan["base_commit"]),
        "Diagnostic runtime omitted locked core versions or the Git revision")
    identity = plan["input_identity"]
    require(plan["input_identity_sha256"] == canonical_hash(identity)
        and type(identity.get("image_id")) is int and identity["image_id"] > 0
        and type(identity.get("position")) is int and identity["position"] == 0
        and type(identity.get("attack_seed")) is int and identity["attack_seed"] == 42,
        "Diagnostic input identity changed")
    for kind in ("image", "annotation", "checkpoint"):
        path = Path(identity[kind + "_file"])
        require(path.is_file() and not path.is_symlink()
            and file_digest(path) == identity[kind + "_sha256"], "Changed diagnostic input: " + kind)
    return plan


def execution_groups(plan, max_images):
    require(type(max_images) is int and max_images == plan["max_images"] == 1
        and plan["diagnostic_only"] is True and plan["targets"] == [],
        "This preflight cannot execute formal or multi-image work")
    require([g["group_id"] for g in plan["groups"]] == plan["new_group_ids"]
        and all(g["images"] == 1 and g["seed"] == 42 for g in plan["groups"]),
        "A complete selected-stage diagnostic group set is required")
    return plan["groups"]


def scheduled_stages(plan, selected):
    require([g["group_id"] for g in selected] == plan["new_group_ids"], "Partial diagnostic stage is forbidden")
    return [dict(name=plan["stage"], group_ids=list(plan["new_group_ids"]))]


def execution_metadata(plan, plan_sha256, group, max_images):
    execution_groups(plan, max_images)
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL,
        study=plan["stage_preparation"]["study"], stage=plan["stage"],
        variant=group["variant"], group_id=group["group_id"], seed=group["seed"],
        prepared_plan_sha256=plan["prepared_plan_sha256"], execution_plan_sha256=plan_sha256,
        runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_identity_sha256=plan["input_identity_sha256"], smoke_max_images=1,
        diagnostic_only=True, AP_evaluations=0, formal_metrics_eligible=False,
        independent_confirmation=False, stage_selection=False)


def diagnostic_request(plan, plan_hash):
    return dict(status="bounded_diagnostic_request_not_formal_admission",
        execution_plan_sha256=plan_hash, runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_identity_sha256=plan["input_identity_sha256"], stage_science_sha256=plan["stage_science_sha256"],
        group_ids=list(plan["new_group_ids"]), max_images=1, seed=42, source="faster_rcnn_r50",
        AP_evaluations=0, payload_retention="keep_all", deletion_authorized=False,
        independent_acceptance=False, formal_execution_admission=False)


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    execution_groups(plan, max_images)
    value = read_bound(path, digest)
    require(canonical_hash(value) == canonical_hash(diagnostic_request(plan, plan_hash)),
        "Missing the exact bounded one-image diagnostic request; formal admission is unsupported")
    return value


def verify_payload(registry, plan, plan_hash, group, directory):
    import numpy as np
    from PIL import Image, ImageOps

    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    manifest = directory / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    identity = plan["input_identity"]
    require(run.get("status") == "complete" and run.get("dataset") == "coco" and run.get("split") == "val"
        and run.get("source") == group["source"] and run.get("attack") == ALIAS
        and type(run.get("seed")) is int and run["seed"] == 42
        and len(rows) == 1 and all(type(run.get(k)) is int and run[k] == v for k, v in
            (("requested_images", 1), ("successful_images", 1), ("failed_images", 0))),
        "Incomplete or foreign source-control generation")
    require(canonical_hash(run.get("parameters")) == canonical_hash(group["parameters"]) == group["parameters_sha256"]
        and run.get("parameters_sha256") == group["parameters_sha256"]
        and run.get("checkpoint_sha256") == identity["checkpoint_sha256"]
        and run.get("code_commit") == plan["base_commit"]
        and run.get("budget_profile") == registry.protocols[PROTOCOL]["budget_profile"]
        and file_digest(manifest) == run.get("manifest_sha256"), "Diagnostic source/configuration/manifest differs")
    metadata = execution_metadata(plan, plan_hash, group, 1)
    require(run.get("run_metadata") == metadata and run.get("formal_AP_eligible") is False
        and run.get("research_execution", {}).get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"]
        and run.get("research_execution", {}).get("formal_AP_eligible") is False,
        "Diagnostic execution identity or eligibility differs")
    ids = [identity["image_id"]]
    require(run.get("image_ids_sha256") == run.get("requested_ordered_image_ids_sha256") == canonical_hash(ids)
        and run.get("seed_schedule") == "selected_position"
        and run.get("seed_offsets_sha256") == canonical_hash([[ids[0], 0]])
        and run.get("gradient_evaluations_per_image") == group["logical_gradients"]
        and run.get("actual_gradient_evaluations_total") == group["logical_gradients"],
        "Diagnostic image order, seed schedule or gradient count differs")
    row = rows[0]
    require(type(row.get("image_id")) is int and row["image_id"] == ids[0]
        and Path(row["source_file"]).resolve() == Path(identity["image_file"]).resolve(),
        "Diagnostic did not use the exact first canonical image")
    relative = "images/{:012d}.png".format(ids[0])
    png = child(directory, relative)
    require(row.get("output_file") == relative and png.is_file()
        and row.get("output_sha256") == file_digest(png) and type(row.get("output_bytes")) is int
        and row["output_bytes"] == png.stat().st_size
        and file_digest(Path(identity["image_file"])) == identity["image_sha256"], "Diagnostic payload binding differs")
    with Image.open(identity["image_file"]) as handle:
        clean = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"), dtype=np.int16)
    with Image.open(png) as handle:
        require(handle.format == "PNG", "Diagnostic payload must be PNG")
        adversarial = np.asarray(handle.convert("RGB"), dtype=np.int16)
    require(clean.shape == adversarial.shape, "Diagnostic payload shape differs")
    linf = int(np.abs(clean - adversarial).max())
    require(linf <= 4, "Decoded diagnostic payload exceeds epsilon")
    from ..data.coco import CocoIndex
    index = CocoIndex(registry.dataset("coco"), "val")
    require(len(index.images) == 5000 and int(index.images[0]["id"]) == ids[0]
        and canonical_hash([int(i["id"]) for i in index.images]) == identity["all_image_ids_sha256"]
        and file_digest(index.annotation_path) == identity["annotation_sha256"], "Canonical input scope differs")
    annotation = directory / "annotations.json"
    require(file_digest(annotation) == run.get("annotation_sha256")
        and json.loads(annotation.read_text(encoding="utf-8")) == index.adversarial_annotation({ids[0]: relative}),
        "Diagnostic annotation differs")
    counted = recount_group_traces(rows, group)
    peak = run.get("peak_cuda_memory_mb")
    require(type(peak) in (int, float) and math.isfinite(peak) and peak > 0, "Missing actual diagnostic group memory peak")
    return dict(group_id=group["group_id"], variant=group["variant"], images=1, decoded_png_max_linf=linf,
        parameters_sha256=group["parameters_sha256"], generation_run_sha256=file_digest(directory / "run.json"),
        manifest_sha256=file_digest(manifest), png_sha256=file_digest(png), trace_recount=counted,
        peak_cuda_memory_mb=peak, memory_scope="original_group_peak_including_resident_model",
        runtime_seconds=rows[0]["runtime_seconds"], cost_calibration_accepted=False,
        cost_disclosure_accepted=False, AP_evaluations=0, scientific_acceptance=False)


def run_diagnostic_group(registry, plan, descriptor, group, group_root, max_images,
        selection, progress, ownership, *, profile=None):
    from .attack import run_attack
    from . import vcsf_efficacy_runner as runner

    execution_groups(plan, max_images)
    require(not group_root.exists() and not group_root.is_symlink(), "Cannot resume or overwrite a diagnostic group")
    group_root.mkdir(parents=True, exist_ok=False)
    ownership()
    progress(phase="source_control_generation", images_completed=0, images_total=1,
        targets_completed=0, target=None)
    try:
        run_attack(registry, "coco", group["source"], ALIAS, split="val", output_dir=group_root / "attack",
            max_images=1, seed=42, device="cuda:0", download_weights=False, keep_going=False, strict=True,
            parameter_overrides=group["parameters"], budget_profile=registry.protocols[PROTOCOL]["budget_profile"],
            run_metadata=execution_metadata(plan, descriptor["execution_plan_sha256"], group, 1),
            isolated_research=descriptor)
    finally:
        runner._cleanup()
    ownership()
    counted = verify_payload(registry, plan, descriptor["execution_plan_sha256"], group, group_root / "attack")
    runner.write_new(group_root / "source_control_validation.json", counted)
    artifacts = {}
    for path in sorted(group_root.rglob("*")):
        require(not path.is_symlink(), "Symlink diagnostic artifact")
        if path.is_file():
            artifacts[path.relative_to(group_root).as_posix()] = file_digest(path)
    manifest_hash = runner.write_new(group_root / "artifact_manifest.json", dict(artifacts=artifacts))
    result = dict(status="complete_group_pending_independent_acceptance", errors=[], failed_records=0,
        group_id=group["group_id"], variant=group["variant"], source=group["source"], seed=42,
        parameters_sha256=group["parameters_sha256"], images=1, evaluations=0, targets=[],
        execution_plan_sha256=descriptor["execution_plan_sha256"], runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        artifact_manifest_sha256=manifest_hash, payload_state="diagnostic_keep_all", diagnostic_only=True,
        source_control_validation_sha256=file_digest(group_root / "source_control_validation.json"),
        scientific_acceptance=False, independent_confirmation=False, formal_metrics_eligible=False,
        completed_at=runner.utc_now())
    result["receipt_sha256"] = runner.write_new(group_root / "group_acceptance.json", result)
    progress(phase="source_control_verified", images_completed=1, targets_completed=0)
    return result


def write_reports(output, rows):
    columns = ("group_id", "variant", "runtime_seconds", "peak_cuda_memory_mb", "decoded_png_max_linf")
    with (output / "source_controls.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([format(row[k], ".4f") if k in ("runtime_seconds", "peak_cuda_memory_mb") else row[k]
                for k in columns])
    with (output / "source_controls.tex").open("x", encoding="utf-8") as handle:
        handle.write("% One-image integration diagnostics; not isolated cost or AP.\n")
        handle.write("\\begin{tabular}{lrr}\nVariant & Seconds & Peak MiB \\\\\n\\hline\n")
        for row in rows:
            handle.write("{} & {:.4f} & {:.4f} \\\\\n".format(
                row["variant"].replace("_", "\\_"), row["runtime_seconds"], row["peak_cuda_memory_mb"]))
        handle.write("\\end{tabular}\n")


def verify_stage_results(registry, plan, descriptor, output, results):
    """Reconcile authenticated group sidecars with the complete worker ownership."""
    from . import vcsf_efficacy_runner as runner

    stage = scheduled_stages(plan, execution_groups(plan, 1))[0]
    binding = read_bound(output / "plan_binding.json", descriptor["run_binding_sha256"])
    require(binding["execution_plan_sha256"] == descriptor["execution_plan_sha256"]
        and binding["admission_sha256"] == descriptor["admission_sha256"]
        and binding["group_ids"] == plan["new_group_ids"] and binding["stages"] == [stage],
        "Diagnostic coordinator binding differs")
    completed_path = output / "stages" / plan["stage"] / "completion.json"
    completed = read_bound(completed_path, file_digest(completed_path))
    require(completed["status"] == "complete_stage_mechanical" and completed["errors"] == []
        and canonical_hash(completed["stage"]) == canonical_hash(stage)
        and canonical_hash(completed["groups"]) == canonical_hash(results), "Diagnostic stage completion differs")
    assignment = read_bound(output / "stages" / plan["stage"] / "assignment.json", completed["assignment_sha256"])
    require(canonical_hash(assignment["stage"]) == canonical_hash(stage), "Diagnostic assignment stage differs")
    lanes = balanced_lanes(plan["groups"], plan["sources"], binding["devices"])
    entries = assignment["workers"]
    expected_slots = [i for i, lane in enumerate(lanes) if lane]
    require([e["slot"] for e in entries] == expected_slots, "Missing or duplicate diagnostic worker lane")
    returned = []
    for entry in entries:
        slot = entry["slot"]
        require(type(slot) is int and type(entry["pid"]) is int and entry["pid"] > 0
            and type(entry["pid_start_ticks"]) is int and entry["pid_start_ticks"] > 0,
            "Malformed diagnostic worker identity")
        directory = output / "workers" / plan["stage"] / str(slot)
        require(entry["directory"] == directory.relative_to(registry.root).as_posix()
            and entry["physical_device"] == binding["devices"][slot]
            and entry["group_ids"] == [g["group_id"] for g in lanes[slot]], "Diagnostic lane assignment differs")
        request = read_bound(directory / "request.json", entry["request_sha256"])
        expected_request = dict(coordinator_pid=os.getpid(), coordinator_pid_start_ticks=runner.process_start_ticks(os.getpid()),
            worker_slot=slot, stage=plan["stage"], physical_device=entry["physical_device"],
            devices=binding["devices"], gpu_uuid=entry["gpu_uuid"], group_ids=entry["group_ids"], descriptor=descriptor)
        require(all(canonical_hash(request.get(k)) == canonical_hash(v) for k, v in expected_request.items()),
            "Diagnostic request has foreign parent, device, stage or descriptor")
        ownership = dict(worker_pid=entry["pid"], worker_pid_start_ticks=entry["pid_start_ticks"],
            coordinator_pid=request["coordinator_pid"], coordinator_pid_start_ticks=request["coordinator_pid_start_ticks"],
            worker_slot=slot, physical_device=entry["physical_device"], gpu_uuid=entry["gpu_uuid"], stage=plan["stage"],
            request_sha256=entry["request_sha256"], execution_plan_sha256=descriptor["execution_plan_sha256"])
        worker_state = json.loads((directory / "execution_state.json").read_text())
        expected_state = dict(ownership, status="complete", failed_records=0, current=None,
            expected_groups=len(lanes[slot]), completed_groups=len(lanes[slot]), completed_evaluations=0)
        require(all(canonical_hash(worker_state.get(k)) == canonical_hash(v) for k, v in expected_state.items()),
            "Diagnostic worker terminal state has incomplete ownership or counts")
        worker_rows = json.loads((directory / "completed_groups.json").read_text())
        require([row["group_id"] for row in worker_rows] == entry["group_ids"], "Diagnostic worker row set differs")
        for group, row in zip(lanes[slot], worker_rows):
            group_root = output / "groups" / "{:06d}".format(group["group_id"])
            receipt = read_bound(group_root / "group_acceptance.json", row["receipt_sha256"])
            expected_group = dict(status="complete_group_pending_independent_acceptance", errors=[], failed_records=0,
                group_id=group["group_id"], variant=group["variant"], source=group["source"], seed=42,
                parameters_sha256=group["parameters_sha256"], images=1, evaluations=0, targets=[],
                execution_plan_sha256=descriptor["execution_plan_sha256"], runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
                payload_state="diagnostic_keep_all", diagnostic_only=True, scientific_acceptance=False,
                independent_confirmation=False, formal_metrics_eligible=False)
            require(all(canonical_hash(receipt.get(k)) == canonical_hash(v) for k, v in expected_group.items()),
                "Diagnostic group receipt has another configuration or scope")
            require(all(k not in receipt or canonical_hash(receipt[k]) == canonical_hash(v)
                for k, v in ownership.items()), "Diagnostic group receipt contradicts worker ownership")
            require(canonical_hash(row) == canonical_hash(dict(ownership, **receipt, receipt_sha256=row["receipt_sha256"])),
                "Worker row differs from authenticated group receipt and ownership")
            manifest = read_bound(group_root / "artifact_manifest.json", receipt["artifact_manifest_sha256"])["artifacts"]
            actual = set()
            for path in group_root.rglob("*"):
                require(not path.is_symlink(), "Symlink in completed diagnostic group")
                if path.is_file() and path not in (group_root / "artifact_manifest.json", group_root / "group_acceptance.json"):
                    actual.add(path.relative_to(group_root).as_posix())
            require(set(manifest) == actual, "Diagnostic group has unsealed or missing files")
            for rel, digest in manifest.items():
                require(file_digest(child(group_root, rel)) == digest, "Diagnostic artifact changed: " + rel)
            require(manifest["source_control_validation.json"] == receipt["source_control_validation_sha256"],
                "Group receipt and source-control sidecar disagree")
            sidecar = read_bound(group_root / "source_control_validation.json", receipt["source_control_validation_sha256"])
            recounted = verify_payload(registry, plan, descriptor["execution_plan_sha256"], group, group_root / "attack")
            require(canonical_hash(sidecar) == canonical_hash(recounted), "Source-control sidecar does not match saved evidence")
            returned.append((row, sidecar))
    returned.sort(key=lambda pair: pair[0]["group_id"])
    require(canonical_hash([row for row, _ in returned]) == canonical_hash(results), "Coordinator/worker group rows differ")
    return [sidecar for _, sidecar in returned]


def finalize_output(registry, plan, descriptor, output, results, state):
    from . import vcsf_efficacy_runner as runner

    state.update(status="finalizing_source_control_evidence", completed_groups=len(results), workers=[],
        updated_at=runner.utc_now())
    atomic_json(output / "execution_state.json", state)
    rows = verify_stage_results(registry, plan, descriptor, output, results)
    runner.write_new(output / "source_controls.json", rows)
    write_reports(output, rows)
    manifest = {}
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink(), "Symlink in diagnostic output")
        if path.is_file() and path != output / "execution_state.json":
            manifest[path.relative_to(output).as_posix()] = file_digest(path)
    manifest_hash = runner.write_new(output / "artifact_manifest.json", dict(artifacts=manifest))
    status = "complete_source_controls_pending_independent_acceptance"
    runner.write_new(output / "receipt.json", dict(status=status, groups=len(results),
        complete_attack_calls=len(results), images_per_group=1, AP_evaluations=0, failed_records=0,
        execution_plan_sha256=descriptor["execution_plan_sha256"], artifact_manifest_sha256=manifest_hash,
        source_controls_sha256=file_digest(output / "source_controls.json"),
        diagnostic_only=True, independent_acceptance=False, stage_evidence_reverified=False,
        formal_execution_admission=False, scientific_acceptance=False, physical_cost_calibration_accepted=False))
    state.update(status=status, completed_at=runner.utc_now())
    atomic_json(output / "execution_state.json", state)


def execute(registry, stage_path, stage_hash, devices, max_images=1, output=None):
    from . import vcsf_efficacy_runner as runner
    from .vcsf_structure_workers import gpu_reservations, physical_gpus, verify_device_owner

    verify_server()
    require(type(max_images) is int and max_images == 1, "Preflight requires exactly one image per full configuration")
    stage_path = Path(stage_path)
    stage = read_stage_plan(stage_path, stage_hash)
    stage_path = stage_path.resolve(strict=True)
    diagnostic = compile_diagnostic_plan(registry, stage)
    groups = diagnostic["groups"]
    lanes = balanced_lanes(groups, diagnostic["sources"], devices)
    root = registry.root.resolve()
    parent = root / "outputs" / "diagnostics" / EXECUTION_PROTOCOL
    output = Path(output).absolute() if output else parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    require(output.parent == parent and stage_path.parent not in (output, *output.parents),
        "Preflight requires a fresh direct diagnostic child outside the input artifact root")
    child(root, output.relative_to(root))
    require(not output.exists() and not output.is_symlink(), "Refusing to resume or overwrite a diagnostic root")
    require("CUDA_VISIBLE_DEVICES" not in os.environ, "Select physical devices without inherited CUDA_VISIBLE_DEVICES")
    uuids = physical_gpus(devices)
    require(shutil.disk_usage(root).free >= 24 * 1024**3, "Insufficient shared disk safety headroom")
    with gpu_reservations(uuids) as reservations:
        for uuid in uuids:
            verify_device_owner(uuid, os.getpid())
        plan = bind_runtime(registry, diagnostic)
        verify_environment(plan)
        output.mkdir(parents=True, exist_ok=False)
        plan_path = output / "plan.json"
        plan_hash = runner.write_new(plan_path, plan)
        request_path = output / "diagnostic_request.json"
        request_hash = runner.write_new(request_path, diagnostic_request(plan, plan_hash))
        selection_hash = runner.write_new(output / "retained_selection.json", dict(
            status="all_diagnostic_payloads_retained_no_selection_or_deletion", images_per_group=1))
        stages = scheduled_stages(plan, groups)
        binding = dict(execution_root=output.relative_to(root).as_posix(), execution_plan_sha256=plan_hash,
            admission_sha256=request_hash, max_images=1, group_ids=plan["new_group_ids"],
            devices=list(devices), stages=stages, diagnostic_only=True)
        binding_hash = runner.write_new(output / "plan_binding.json", binding)
        descriptor = dict(candidate_id=CANDIDATE_ID, execution_alias=ALIAS, mode=MODE,
            execution_plan=plan_path.relative_to(root).as_posix(), execution_plan_sha256=plan_hash,
            admission=request_path.relative_to(root).as_posix(), admission_sha256=request_hash,
            execution_root=binding["execution_root"], run_binding_sha256=binding_hash, max_images=1)
        state = dict(status="running", protocol=EXECUTION_PROTOCOL, coordinator_pid=os.getpid(),
            coordinator_pid_start_ticks=runner.process_start_ticks(os.getpid()), execution_plan_sha256=plan_hash,
            expected_groups=len(groups), expected_evaluations=0, completed_groups=0, completed_evaluations=0,
            failed_records=0, AP_evaluations=0, started_at=runner.utc_now(), diagnostic_only=True)
        atomic_json(output / "execution_state.json", state)
        try:
            results = runner.run_stage(registry, plan, descriptor, output, stages[0], list(devices), uuids,
                reservations, lanes, selection_hash, state, profile=sys.modules[__name__])
            verify_execution_plan(registry, plan)
            read_stage_plan(stage_path, stage_hash)
            require(_bind_inputs(registry) == plan["input_identity"], "Canonical inputs changed during preflight")
            finalize_output(registry, plan, descriptor, output, results, state)
        except BaseException as exc:
            state.update(status="failed", failed_records=1, reason=repr(exc)[:2000], updated_at=runner.utc_now())
            atomic_json(output / "execution_state.json", state)
            raise
    return output
