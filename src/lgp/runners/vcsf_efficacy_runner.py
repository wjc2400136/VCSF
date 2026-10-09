"""Complete-group, one/two-GPU execution for the registered layered efficacy study."""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..registry import Registry
from .vcsf_efficacy_contract import (
    ENTRYPOINT, EXECUTION_PROTOCOL, child, execution_groups, execution_metadata,
    read_bound, require, verify_admission, verify_environment, verify_execution_plan,
)
from .vcsf_research_plan import PROTOCOL, canonical_hash, compile_plan
from .vcsf_gpu_context_owner import RegisteredGpuOwner
from .vcsf_structure_workers import (
    balanced_lanes, gpu_reservations, install_parent_death_guard, physical_gpus,
    verify_coordinator, verify_device_owner, verify_reservation, worker_environment,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def process_start_ticks(pid):
    raw = Path("/proc") / str(pid) / "stat"
    return int(raw.read_text().rsplit(") ", 1)[1].split()[19])


def _execution_api(profile=None):
    if profile is not None:
        require(profile.EXECUTION_PROTOCOL in ("vcsf_single_source_scale_execution",
            "vcsf_single_source_ablation_preflight", "vcsf_single_source_ablation_executor_smoke",
            "vcsf_single_source_operator_factorial_execution", "vcsf_single_source_common_anchor_execution",
            "vcsf_final_attribution_execution", "vcsf_final_background_execution"),
            "Unsupported complete-group execution profile")
        return profile
    return SimpleNamespace(PROTOCOL=PROTOCOL, EXECUTION_PROTOCOL=EXECUTION_PROTOCOL,
        ENTRYPOINT=ENTRYPOINT, ALIAS="vcsf_research_isolated", CANDIDATE_ID="vcsf_layered_research",
        MODE="layered_efficacy", ALLOW_EMPTY_LANES=False, execution_groups=execution_groups,
        execution_metadata=execution_metadata, verify_execution_plan=verify_execution_plan,
        verify_admission=verify_admission, balanced_lanes=balanced_lanes, scheduled_stages=scheduled_stages)


def write_new(path, value):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Refusing to overwrite a published artifact")
    atomic_json(path, value)
    return file_digest(path)


def full_stage_keep_all(plan):
    mode = plan.get("execution_lifecycle")
    require(mode in (None, "full_stage_keep_all_v1"), "Unsupported complete-group lifecycle")
    return mode is not None


def keep_all_policy(plan, selected, max_images):
    require(full_stage_keep_all(plan), "A full-stage keep-all policy must be explicit")
    retention = plan["definition"]["payload_retention"]
    require(retention.get("mode") == "keep_all"
        and retention.get("deletion_authorized") is False
        and retention.get("prune_after_seal") is False
        and retention.get("preserve_failed_and_partial") is True,
        "Full-stage keep-all forbids selection, pruning and deletion")
    ids = [group["group_id"] for group in selected]
    require(isinstance(plan.get("new_group_ids"), list)
        and all(type(value) is int and value > 0 for value in plan["new_group_ids"])
        and ids and all(type(value) is int and value > 0 for value in ids)
        and len(set(ids)) == len(ids) and ids == plan["new_group_ids"],
        "Keep-all capacity must bind the complete, nonduplicated new job set")
    require(max_images is None or type(max_images) is int and 0 < max_images < 5000,
        "Invalid keep-all diagnostic scope")
    capacity = plan["capacity_contract"]
    budgets = capacity.get("group_budget_bytes")
    reserve = capacity.get("failure_reserve_bytes")
    required = capacity.get("required_free_bytes")
    require(capacity.get("policy") == "full_stage_keep_all_plus_failure_reserve"
        and isinstance(budgets, dict) and set(budgets) == {str(value) for value in ids}
        and all(isinstance(value, dict) and set(value) == {"payload", "prediction_and_metadata"}
            and all(type(part) is int and part > 0 for part in value.values())
            for value in budgets.values())
        and type(reserve) is int and reserve > 0 and type(required) is int
        and required == sum(sum(value.values()) for value in budgets.values()) + reserve,
        "Missing exact full-stage byte budgets and a positive failure reserve")
    return dict(schema_version=1, status="all_scheduled_payloads_retained_no_selection_or_deletion",
        execution_lifecycle="full_stage_keep_all_v1", group_ids=ids, max_images=max_images,
        payload_retention=deepcopy(retention), capacity_contract=deepcopy(capacity),
        capacity_contract_sha256=canonical_hash(capacity),
        runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        deletion_authorized=False, independent_acceptance=False)


def verify_capacity(registry, plan, group_ids=None):
    minimum = 24 * 1024**3
    if full_stage_keep_all(plan):
        keep_all_policy(plan, [group for group in plan["groups"]
            if group["group_id"] in plan["new_group_ids"]], None)
        capacity = plan["capacity_contract"]
        ids = plan["new_group_ids"] if group_ids is None else group_ids
        require(ids and len(ids) == len(set(ids))
            and all(type(value) is int and value in plan["new_group_ids"] for value in ids),
            "Capacity check names a foreign or duplicate job")
        required = sum(sum(capacity["group_budget_bytes"][str(value)].values()) for value in ids)
        minimum = max(minimum, required + capacity["failure_reserve_bytes"])
    require(shutil.disk_usage(registry.root).free >= minimum,
        "Insufficient disk headroom for complete jobs and preserved failure evidence")


def lifecycle_request(plan, policy_sha256):
    key = "payload_policy_sha256" if full_stage_keep_all(plan) else "retained_selection_sha256"
    return {key: policy_sha256}


def read_worker_lifecycle(plan, selected, max_images, output, request):
    if full_stage_keep_all(plan):
        require("retained_selection_sha256" not in request and "payload_policy_sha256" in request,
            "Keep-all worker cannot inherit a retained-subset request")
        require(not (output / "retained_selection.json").exists()
            and not (output / "retained_selection.json").is_symlink(),
            "Keep-all execution cannot contain a retained-subset selection")
        value = read_bound(output / "payload_policy.json", request["payload_policy_sha256"])
        require(canonical_hash(value) == canonical_hash(keep_all_policy(plan, selected, max_images)),
            "Worker keep-all policy or capacity changed")
        return value
    require("payload_policy_sha256" not in request,
        "A legacy worker cannot consume a full-stage keep-all policy")
    return read_bound(output / "retained_selection.json", request["retained_selection_sha256"])


def scheduled_stages(plan, selected):
    selected_ids = {g["group_id"] for g in selected}
    stages, seen = [], set()
    for stage in plan["prepared_plan"]["stages"]:
        pairs = {tuple(pair) for pair in stage["introduced_pairs"]}
        groups = [g for g in selected if (g["variant"], g["seed"]) in pairs]
        if groups:
            require(not (seen & {g["group_id"] for g in groups}), "Duplicate execution-stage group")
            stages.append({"name": stage["block"], "group_ids": [g["group_id"] for g in groups]})
            seen.update(g["group_id"] for g in groups)
    require(seen == selected_ids, "The registered stages do not cover the selected complete groups")
    return stages


def input_contract(registry, plan):
    from ..data.coco import CocoIndex
    from ..modeling import checkpoint_path

    index = CocoIndex(registry.dataset("coco"), "val")
    ids = [int(image["id"]) for image in index.images]
    require(len(ids) == plan["prepared_plan"]["groups"][0]["images"]
        and len(ids) == len(set(ids)) and ids == sorted(ids)
        and canonical_hash(ids) == plan["image_ids_sha256"]
        and file_digest(index.annotation_path) == plan["annotation_sha256"],
        "Canonical full-val annotations or image IDs changed")
    actual = {}
    for target in plan["targets"]:
        path = checkpoint_path(registry.model(target), registry.dataset("coco"))
        require(path.is_file(), "Missing registered target checkpoint: " + target)
        actual[target] = file_digest(path)
    require(actual == plan["checkpoint_sha256"], "Registered source/target checkpoint panel changed")
    return index


def validate_generation(registry, plan, plan_sha256, group, attack_dir, max_images, *, decode_pixels=True, profile=None):
    from ..data.coco import CocoIndex

    api = _execution_api(profile)
    run = json.loads((attack_dir / "run.json").read_text(encoding="utf-8"))
    index = CocoIndex(registry.dataset("coco"), "val")
    images = index.images if max_images is None else index.images[:max_images]
    ids = [int(image["id"]) for image in images]
    count = len(ids)
    require(run.get("status") == "complete" and run.get("dataset") == "coco"
        and run.get("split") == "val" and run.get("source") == group["source"]
        and run.get("attack") == api.ALIAS
        and type(run.get("seed")) is int and run["seed"] == group["seed"]
        and run.get("requested_images") == run.get("successful_images") == count
        and run.get("failed_images") == 0, "Generation is incomplete or belongs to another group")
    require(run.get("parameters") == group["parameters"]
        and run.get("parameters_sha256") == group["parameters_sha256"]
        and run.get("checkpoint_sha256") == plan["checkpoint_sha256"][group["source"]]
        and run.get("code_commit") == plan["base_commit"]
        and run.get("budget_profile") == registry.protocols[api.PROTOCOL]["budget_profile"],
        "Generation method, source or budget identity changed")
    require(run.get("run_metadata") == api.execution_metadata(plan, plan_sha256, group, max_images)
        and run.get("research_execution", {}).get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"],
        "Generation omitted the actual modified-runtime identity")
    require(run.get("image_ids_sha256") == canonical_hash(ids)
        and run.get("requested_ordered_image_ids_sha256") == canonical_hash(ids)
        and run.get("seed_schedule") == "selected_position"
        and run.get("seed_offsets_sha256") == canonical_hash([[image_id, i] for i, image_id in enumerate(ids)]),
        "Canonical image order or seed mapping changed")
    manifest_path = attack_dir / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    require(file_digest(manifest_path) == run.get("manifest_sha256")
        and len(rows) == count and [row.get("image_id") for row in rows] == ids,
        "Generation manifest bytes, order or count changed")
    require(run.get("gradient_evaluations_per_image") == group["logical_gradients"]
        and run.get("actual_gradient_evaluations_total") == count * group["logical_gradients"],
        "Generation did not execute the registered dominant-gradient schedule")
    max_linf = 0
    for position, (row, source_image) in enumerate(zip(rows, images)):
        relative = "images/{:012d}.png".format(ids[position])
        output = child(attack_dir, relative)
        require(row.get("status") == "ok" and row.get("position") == position
            and row.get("attack_seed") == group["seed"] + position
            and row.get("actual_gradient_evaluations") == group["logical_gradients"]
            and row.get("output_file") == relative and output.is_file()
            and file_digest(output) == row.get("output_sha256")
            and output.stat().st_size == row.get("output_bytes"),
            "Generation row identity, seed, budget or payload hash changed")
        require(Path(row["source_file"]).resolve() == index.image_path(source_image).resolve(),
            "Generation used a noncanonical clean image")
        require(type(row.get("linf_pixel")) in (int, float) and math.isfinite(row["linf_pixel"])
            and 0 <= row["linf_pixel"] <= group["parameters"]["eps"] * 255.0 + 1e-4,
            "Generation reports an invalid perturbation bound")
        if decode_pixels:
            import numpy as np
            from PIL import Image, ImageOps

            with Image.open(index.image_path(source_image)) as handle:
                clean = np.asarray(ImageOps.exif_transpose(handle).convert("RGB"), dtype=np.int16)
            with Image.open(output) as handle:
                adversarial = np.asarray(handle.convert("RGB"), dtype=np.int16)
            require(clean.shape == adversarial.shape, "Saved PNG dimensions changed")
            linf = int(np.max(np.abs(adversarial - clean)))
            require(linf <= group["parameters"]["eps"] * 255.0 + 1e-4,
                "Saved PNG violates the registered pixel bound")
            max_linf = max(max_linf, linf)
    annotation = json.loads((attack_dir / "annotations.json").read_text(encoding="utf-8"))
    expected_names = {image_id: "images/{:012d}.png".format(image_id) for image_id in ids}
    require(annotation == index.adversarial_annotation(expected_names)
        and file_digest(attack_dir / "annotations.json") == run.get("annotation_sha256"),
        "Generated annotation semantics or hash changed")
    return {"images": count, "image_ids_sha256": canonical_hash(ids),
        "generation_run_sha256_before_retention": file_digest(attack_dir / "run.json"),
        "manifest_sha256": file_digest(manifest_path), "decoded_png_max_linf": max_linf,
        "decoded_png_count": count if decode_pixels else 0}


def validate_target(plan, group, target, record, generation, *, profile=None):
    api = _execution_api(profile)
    require(record.get("status") == "complete" and record.get("failures") == []
        and record.get("dataset") == "coco" and record.get("split") == "val"
        and record.get("source") == group["source"] and record.get("target") == target
        and record.get("attack") == api.ALIAS
        and record.get("images") == generation["images"]
        and record.get("parameters_sha256") == group["parameters_sha256"]
        and record.get("checkpoint_sha256") == plan["checkpoint_sha256"][target]
        and record.get("code_commit") == plan["base_commit"],
        "Target result is incomplete or not the exact scheduled observation")
    require(record.get("evaluated_image_ids_sha256") == record.get("expected_image_ids_sha256")
        == generation["image_ids_sha256"] and record.get("evaluated_image_ids_match_expected") is True,
        "Target evaluated a different image set")
    values = record.get("metrics", {})
    require(all(type(values.get(key)) in (int, float) and math.isfinite(values[key])
        and (values[key] == -1 or 0 <= values[key] <= 1) for key in COCO_BBOX_METRICS)
        and values["bbox_mAP"] >= 0, "Target does not preserve all twelve unrounded COCO metrics")
    rows = record.get("coco_summary", {}).get("rows", [])
    require(len(rows) == len(COCO_BBOX_METRICS)
        and [row.get("order") for row in rows] == list(range(1, 13)),
        "Target COCO summary has missing or reordered rows")


def seal_group(registry, plan, plan_sha256, group, group_root, max_images, retained_selection, *, profile=None):
    from .retention import _validate_prediction_archive, validate_and_prune_attack_group

    api = _execution_api(profile)
    strict_keep_all = full_stage_keep_all(plan)
    if strict_keep_all:
        require(profile is not None, "Full-stage keep-all requires an explicit execution profile")
        expected_policy = keep_all_policy(plan, api.execution_groups(plan, max_images), max_images)
        require(canonical_hash(retained_selection) == canonical_hash(expected_policy),
            "Group sealing lost the bound keep-all policy")
        require(not (group_root / "artifact_manifest.json").exists()
            and not (group_root / "group_acceptance.json").exists(),
            "Refusing to reseal a published keep-all group")
    attack_dir = group_root / "attack"
    generation = validate_generation(registry, plan, plan_sha256, group, attack_dir, max_images, profile=profile)
    evaluations, records = {}, []
    for target in plan["targets"]:
        directory = group_root / "evaluations" / target
        evaluations[target] = directory
        record = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        validate_target(plan, group, target, record, generation, profile=profile)
        require(Path(record["adversarial_run"]).resolve() == attack_dir.resolve(),
            "Target evaluation points to another attack payload")
        _validate_prediction_archive(directory, record)
        records.append(dict(record, variant=group["variant"], seed=group["seed"],
            group_id=group["group_id"], execution_protocol=api.EXECUTION_PROTOCOL,
            execution_plan_sha256=plan_sha256,
            runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
            independent_confirmation=False, formal_metrics_eligible=False))
    # No irreversible lifecycle operation is reached until the entire panel passes.
    write_new(group_root / "panel_validation.json", {
        "status": "complete_panel_and_payload_verified", "errors": [],
        "group_id": group["group_id"], "generation": generation,
        "execution_plan_sha256": plan_sha256, "targets": plan["targets"],
        "target_metrics_sha256": {target: file_digest(evaluations[target] / "metrics.json")
            for target in plan["targets"]},
        "retention_started": False, "independent_confirmation": False,
    })
    retention = None
    keep_all = strict_keep_all or profile is not None and plan["definition"]["payload_retention"].get("mode") == "keep_all"
    if max_images is None and not keep_all:
        require(plan["payload_authorization"]["explicit_owner_confirmation"] is True
            and plan["payload_authorization"]["historical_roots_allowed"] is False
            and plan["payload_authorization"]["failed_or_partial_groups_allowed"] is False,
            "The exact new-root-only retention authorization is absent")
        group_root.relative_to(registry.root / "outputs" / "experiments" / api.EXECUTION_PROTOCOL)
        require(retained_selection["retained_image_ids_sha256"]
            == plan["definition"]["payload_retention"]["retained_image_ids_sha256"],
            "The predeclared retained image set changed")
        retention = validate_and_prune_attack_group(
            registry, "coco", "val", group["source"], api.ALIAS,
            attack_dir, evaluations, plan["targets"], retained_selection)
    records_hash = write_new(group_root / "records.json", records)
    artifacts = {}
    for path in sorted(group_root.rglob("*")):
        require(not path.is_symlink(), "A group artifact is a symlink")
        if path.is_file():
            artifacts[path.relative_to(group_root).as_posix()] = {
                "sha256": file_digest(path), "bytes": path.stat().st_size}
    manifest_hash = write_new(group_root / "artifact_manifest.json",
        {"schema_version": 1, "artifacts": artifacts, "group_id": group["group_id"]})
    result = {
        "schema_version": 1, "status": "complete_group_pending_independent_acceptance",
        "group_id": group["group_id"], "source": group["source"], "variant": group["variant"],
        "seed": group["seed"], "parameters_sha256": group["parameters_sha256"],
        "images": generation["images"], "targets": plan["targets"],
        "evaluations": len(records), "failed_records": 0, "errors": [],
        "execution_plan_sha256": plan_sha256, "runtime_snapshot_sha256": plan["runtime_snapshot_sha256"],
        "generation": generation, "records_sha256": records_hash, "artifact_manifest_sha256": manifest_hash,
        "retention": retention, "payload_state": "retained_500" if retention is not None else
            "formal_keep_all" if max_images is None and keep_all else "smoke_keep_all",
        "completed_at": utc_now(), "independent_confirmation": False, "scientific_acceptance": False,
        "formal_metrics_eligible": False,
    }
    if strict_keep_all:
        api.verify_execution_plan(registry, plan)
        require(canonical_hash(retained_selection) == canonical_hash(
            keep_all_policy(plan, api.execution_groups(plan, max_images), max_images)),
            "Keep-all policy changed during sealing")
        current_files = {}
        for path in group_root.rglob("*"):
            require(not path.is_symlink(), "A sealed keep-all artifact became a symlink")
            if path.is_file():
                current_files[path.relative_to(group_root).as_posix()] = path
        require(set(current_files) == set(artifacts) | {"artifact_manifest.json"},
            "Keep-all artifact set changed before terminal publication")
        for name, row in artifacts.items():
            path = current_files[name]
            require(path.stat().st_size == row["bytes"] and file_digest(path) == row["sha256"],
                "Keep-all artifact bytes changed before terminal publication")
        require(file_digest(group_root / "artifact_manifest.json") == manifest_hash,
            "Keep-all artifact manifest changed before terminal publication")
        result.update(execution_lifecycle="full_stage_keep_all_v1",
            payload_policy_content_sha256=canonical_hash(retained_selection),
            all_artifacts_rechecked_before_receipt=True, deletion_performed=False)
    result["receipt_sha256"] = write_new(group_root / "group_acceptance.json", result)
    return result


def verify_keep_all_groups(plan, plan_sha256, output, completed, expected_ids,
        policy_sha256, max_images, *, profile):
    api = _execution_api(profile)
    policy = read_worker_lifecycle(plan, api.execution_groups(plan, max_images), max_images,
        output, lifecycle_request(plan, policy_sha256))
    require([row["group_id"] for row in completed] == expected_ids
        and len(set(expected_ids)) == len(expected_ids),
        "Keep-all terminal reconciliation has missing or duplicate groups")
    for result in completed:
        root = child(output, "groups/{:06d}".format(result["group_id"]))
        receipt = read_bound(child(root, "group_acceptance.json"), result["receipt_sha256"])
        require(receipt.get("status") == "complete_group_pending_independent_acceptance"
            and receipt.get("group_id") == result["group_id"]
            and receipt.get("execution_plan_sha256") == plan_sha256
            and receipt.get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"]
            and receipt.get("execution_lifecycle") == "full_stage_keep_all_v1"
            and receipt.get("payload_policy_content_sha256") == canonical_hash(policy)
            and receipt.get("all_artifacts_rechecked_before_receipt") is True
            and receipt.get("deletion_performed") is False
            and receipt.get("retention") is None
            and receipt.get("payload_state") == ("formal_keep_all" if max_images is None else "smoke_keep_all")
            and receipt.get("targets") == plan["targets"]
            and receipt.get("evaluations") == len(plan["targets"])
            and receipt.get("failed_records") == 0 and receipt.get("errors") == [],
            "Keep-all group receipt changed its execution or payload identity")
        manifest = read_bound(child(root, "artifact_manifest.json"), receipt["artifact_manifest_sha256"])
        artifacts = manifest["artifacts"]
        require(manifest.get("group_id") == result["group_id"] and isinstance(artifacts, dict)
            and "records.json" in artifacts and "artifact_manifest.json" not in artifacts
            and "group_acceptance.json" not in artifacts,
            "Keep-all group artifact manifest has an invalid scope")
        files = set()
        for path in root.rglob("*"):
            require(not path.is_symlink(), "A completed keep-all artifact became a symlink")
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
        require(files == set(artifacts) | {"artifact_manifest.json", "group_acceptance.json"},
            "Completed keep-all artifact set changed")
        for name, row in artifacts.items():
            path = child(root, name)
            require(path.stat().st_size == row["bytes"] and file_digest(path) == row["sha256"],
                "Completed keep-all artifact bytes changed")
        require(file_digest(root / "records.json") == receipt["records_sha256"]
            and receipt["records_sha256"] == result["records_sha256"],
            "Completed keep-all records changed")
        read_bound(root / "artifact_manifest.json", receipt["artifact_manifest_sha256"])
        read_bound(root / "group_acceptance.json", result["receipt_sha256"])
    read_worker_lifecycle(plan, api.execution_groups(plan, max_images), max_images,
        output, lifecycle_request(plan, policy_sha256))


def _cleanup():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_group(registry, plan, descriptor, group, group_root, max_images, selection, progress, ownership, *, profile=None):
    from .attack import run_attack
    from .evaluate import run_evaluation

    api = _execution_api(profile)
    require(not group_root.exists(), "Refusing to resume or overwrite a group")
    if full_stage_keep_all(plan):
        selected = api.execution_groups(plan, max_images)
        require(canonical_hash(selection) == canonical_hash(keep_all_policy(plan, selected, max_images))
            and any(canonical_hash(group) == canonical_hash(scheduled) for scheduled in selected),
            "Generation requires its exact scheduled group and keep-all policy")
    verify_capacity(registry, plan, [group["group_id"]] if full_stage_keep_all(plan) else None)
    group_root.mkdir(parents=True, exist_ok=False)
    ownership()
    progress(phase="generation", group_id=group["group_id"], images_completed=0,
        images_total=max_images or group["images"], targets_completed=0, target=None)

    def generated(done, total, image_id, status):
        if done in (1, total) or done % 25 == 0:
            ownership()
            progress(phase="generation", images_completed=done, images_total=total,
                last_image_id=image_id, last_image_status=status)

    try:
        run_attack(registry, "coco", group["source"], api.ALIAS,
            split="val", output_dir=group_root / "attack", max_images=max_images,
            seed=group["seed"], device="cuda:0", download_weights=False, keep_going=False, strict=True,
            parameter_overrides=group["parameters"], budget_profile=registry.protocols[api.PROTOCOL]["budget_profile"],
            progress_callback=generated,
            run_metadata=api.execution_metadata(plan, descriptor["execution_plan_sha256"], group, max_images),
            isolated_research=descriptor)
    finally:
        _cleanup()
    for position, target in enumerate(plan["targets"]):
        ownership()
        progress(phase="target_evaluation", target=target, target_index=position + 1,
            targets_completed=position, images_completed=0, images_total=max_images or group["images"])

        def evaluated(done, total, image_id, status):
            if done in (1, total) or done % 100 == 0:
                ownership()
                progress(images_completed=done, images_total=total,
                    last_image_id=image_id, last_image_status=status)

        try:
            run_evaluation(registry, "coco", target, split="val",
                adversarial_run=group_root / "attack",
                output_dir=group_root / "evaluations" / target, max_images=max_images,
                device="cuda:0", download_weights=False, keep_going=False,
                save_visualizations=False, prediction_archive=plan["definition"]["prediction_archive"],
                progress_callback=evaluated)
        finally:
            _cleanup()
        progress(targets_completed=position + 1)
    ownership()
    progress(phase="complete_panel_validation_and_retention", target=None)
    result = seal_group(registry, plan, descriptor["execution_plan_sha256"],
        group, group_root, max_images, selection, profile=profile)
    ownership()
    return result


def run_worker(request_path, expected_sha256, *, profile=None):
    api = _execution_api(profile)
    request_path = Path(request_path).resolve(strict=True)
    request = read_bound(request_path, expected_sha256)
    install_parent_death_guard(request["coordinator_pid"])
    verify_reservation(request)
    registry = Registry()
    descriptor = request["descriptor"]
    plan = read_bound(child(registry.root, descriptor["execution_plan"]), descriptor["execution_plan_sha256"])
    api.verify_execution_plan(registry, plan)
    verify_environment(plan)
    admission = api.verify_admission(registry, plan, descriptor["execution_plan_sha256"],
        child(registry.root, descriptor["admission"]), descriptor["admission_sha256"], descriptor["max_images"])
    if api.EXECUTION_PROTOCOL in ("vcsf_single_source_operator_factorial_execution", "vcsf_single_source_common_anchor_execution", "vcsf_final_attribution_execution", "vcsf_final_background_execution"):
        api.verify_device_admission(admission, request["devices"])
    selected = api.execution_groups(plan, descriptor["max_images"])
    lanes = api.balanced_lanes(selected, plan["sources"], request["devices"])
    stages = api.scheduled_stages(plan, selected)
    stage = next((s for s in stages if s["name"] == request["stage"]), None)
    require(stage is not None, "Unregistered execution stage")
    slot = request["worker_slot"]
    require(type(slot) is int and 0 <= slot < len(lanes), "Invalid worker slot")
    expected = [g for g in lanes[slot] if g["group_id"] in stage["group_ids"]]
    require(request["group_ids"] == [g["group_id"] for g in expected]
        and request["physical_device"] == request["devices"][slot]
        and request["gpu_uuid"] == physical_gpus(request["devices"])[slot]
        and os.environ.get("CUDA_VISIBLE_DEVICES") == request["gpu_uuid"],
        "Worker ownership or canonical lane assignment changed")
    output = child(registry.root, descriptor["execution_root"])
    directory = output / "workers" / request["stage"] / str(slot)
    require(request_path == directory / "request.json", "Worker request is outside its bound root")
    selection = read_worker_lifecycle(plan, selected, descriptor["max_images"], output, request)
    binding = {"worker_pid": os.getpid(), "coordinator_pid": request["coordinator_pid"],
        "worker_pid_start_ticks": process_start_ticks(os.getpid()),
        "coordinator_pid_start_ticks": request["coordinator_pid_start_ticks"],
        "worker_slot": slot, "physical_device": request["physical_device"],
        "gpu_uuid": request["gpu_uuid"], "stage": request["stage"],
        "request_sha256": expected_sha256, "execution_plan_sha256": descriptor["execution_plan_sha256"]}
    state = dict(binding, status="running", completed_groups=0, completed_evaluations=0,
        expected_groups=len(expected), failed_records=0, current=None, started_at=utc_now())
    completed = []
    registered_owner = None

    def ownership():
        verify_coordinator(request["coordinator_pid"])
        require(process_start_ticks(request["coordinator_pid"]) == request["coordinator_pid_start_ticks"],
            "Coordinator process identity changed")
        if registered_owner is None:
            verify_device_owner(request["gpu_uuid"], os.getpid())
        else:
            registered_owner.verify()

    def progress(**fields):
        current = dict(state.get("current") or {})
        current.update(fields)
        state.update(current=current, updated_at=utc_now())
        atomic_json(directory / "execution_state.json", state)

    try:
        if api.EXECUTION_PROTOCOL in ("vcsf_single_source_operator_factorial_execution", "vcsf_single_source_common_anchor_execution", "vcsf_final_attribution_execution", "vcsf_final_background_execution"):
            registered_owner = RegisteredGpuOwner(request)
            state["gpu_context_registration"] = registered_owner.receipt
            atomic_json(directory / "execution_state.json", state)
        for group in expected:
            ownership()
            state["current"] = {key: group[key] for key in ("group_id", "source", "variant", "seed")}
            group_runner = api.run_diagnostic_group if api.EXECUTION_PROTOCOL == \
                "vcsf_single_source_ablation_preflight" else run_group
            result = group_runner(registry, plan, descriptor, group,
                output / "groups" / "{:06d}".format(group["group_id"]),
                descriptor["max_images"], selection, progress, ownership, profile=profile)
            completed.append(dict(binding, **result))
            atomic_json(directory / "completed_groups.json", completed)
            state.update(completed_groups=len(completed),
                completed_evaluations=sum(row["evaluations"] for row in completed), current=None, updated_at=utc_now())
            atomic_json(directory / "execution_state.json", state)
        api.verify_execution_plan(registry, plan)
        read_bound(request_path, expected_sha256)
        state.update(status="complete", current=None, completed_at=utc_now())
        atomic_json(directory / "execution_state.json", state)
        return 0
    except BaseException as exc:
        state.update(status="failed", failed_records=1, reason=repr(exc)[:2000], updated_at=utc_now())
        atomic_json(directory / "execution_state.json", state)
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def stop_owned_workers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_stage(registry, plan, descriptor, output, stage, devices, uuids, reservations,
    lanes, selection_sha256, state, *, profile=None):
    api = _execution_api(profile)
    if full_stage_keep_all(plan):
        selected = api.execution_groups(plan, descriptor["max_images"])
        read_worker_lifecycle(plan, selected, descriptor["max_images"], output,
            lifecycle_request(plan, selection_sha256))
        verify_capacity(registry, plan, stage["group_ids"])
    processes, handles, entries = [], [], []
    try:
        for slot, lane in enumerate(lanes):
            group_ids = [g["group_id"] for g in lane if g["group_id"] in stage["group_ids"]]
            if not group_ids and api.ALLOW_EMPTY_LANES:
                continue
            require(group_ids, "An execution stage must preserve a complete source pair")
            directory = output / "workers" / stage["name"] / str(slot)
            directory.mkdir(parents=True, exist_ok=False)
            reserved = os.fstat(reservations[uuids[slot]])
            request = {"coordinator_pid": os.getpid(), "worker_slot": slot, "stage": stage["name"],
                "coordinator_pid_start_ticks": process_start_ticks(os.getpid()),
                "physical_device": devices[slot], "devices": devices, "gpu_uuid": uuids[slot],
                "group_ids": group_ids, "descriptor": descriptor,
                "reservation_fd": reservations[uuids[slot]],
                "reservation_identity": [reserved.st_dev, reserved.st_ino],
                **lifecycle_request(plan, selection_sha256)}
            request_path = directory / "request.json"
            request_hash = write_new(request_path, request)
            stdout = (directory / "stdout.log").open("w", encoding="utf-8")
            stderr = (directory / "stderr.log").open("w", encoding="utf-8")
            handles.extend([stdout, stderr])
            process = subprocess.Popen([sys.executable, "-B", "-u", api.ENTRYPOINT,
                "--worker-request", str(request_path), "--worker-request-sha256", request_hash],
                cwd=registry.root, env=worker_environment(uuids[slot]), stdout=stdout, stderr=stderr,
                pass_fds=(reservations[uuids[slot]],))
            processes.append(process)
            entries.append({"slot": slot, "pid": process.pid, "group_ids": group_ids,
                "pid_start_ticks": process_start_ticks(process.pid),
                "request_sha256": request_hash, "directory": directory.relative_to(registry.root).as_posix(),
                "physical_device": devices[slot], "gpu_uuid": uuids[slot]})
        assignment_path = output / "stages" / stage["name"] / "assignment.json"
        assignment_hash = write_new(assignment_path, {"stage": stage, "workers": entries})
        while True:
            exits = [process.poll() for process in processes]
            snapshots = []
            for process_index, entry in enumerate(entries):
                path = registry.root / entry["directory"] / "execution_state.json"
                worker_state = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
                    "status": "starting", "completed_groups": 0, "completed_evaluations": 0, "current": None}
                snapshots.append(dict(entry, state=worker_state, exit_code=exits[process_index]))
            state.update(stage=stage["name"], workers=snapshots, updated_at=utc_now())
            atomic_json(output / "execution_state.json", state)
            require(all(code in (None, 0) for code in exits), "A worker failed; the new root remains partial")
            if all(code == 0 for code in exits):
                break
            time.sleep(5)
        read_bound(assignment_path, assignment_hash)
        results = []
        for entry in entries:
            directory = registry.root / entry["directory"]
            read_bound(directory / "request.json", entry["request_sha256"])
            worker_state = json.loads((directory / "execution_state.json").read_text(encoding="utf-8"))
            rows = json.loads((directory / "completed_groups.json").read_text(encoding="utf-8"))
            require(worker_state.get("status") == "complete" and worker_state.get("failed_records") == 0
                and worker_state.get("worker_pid") == entry["pid"]
                and worker_state.get("worker_pid_start_ticks") == entry["pid_start_ticks"]
                and worker_state.get("coordinator_pid") == os.getpid()
                and worker_state.get("request_sha256") == entry["request_sha256"]
                and worker_state.get("completed_groups") == len(entry["group_ids"])
                and [row["group_id"] for row in rows] == entry["group_ids"],
                "Completed worker evidence has wrong ownership or counts")
            for row in rows:
                require(row["worker_pid"] == entry["pid"] and row["worker_slot"] == entry["slot"]
                    and row["worker_pid_start_ticks"] == entry["pid_start_ticks"]
                    and row["coordinator_pid"] == os.getpid()
                    and row["request_sha256"] == entry["request_sha256"]
                    and row["gpu_uuid"] == entry["gpu_uuid"], "A group has another worker's ownership")
                receipt = read_bound(output / "groups" / "{:06d}".format(row["group_id"]) /
                    "group_acceptance.json", row["receipt_sha256"])
                require(receipt["status"] == "complete_group_pending_independent_acceptance"
                    and receipt["errors"] == [] and receipt["failed_records"] == 0
                    and receipt["execution_plan_sha256"] == descriptor["execution_plan_sha256"]
                    and receipt["evaluations"] == len(plan["targets"]), "A group receipt is incomplete")
            results.extend(rows)
        require(len(results) == len(stage["group_ids"])
            and {row["group_id"] for row in results} == set(stage["group_ids"]),
            "Execution-stage groups are missing or duplicated")
        results.sort(key=lambda row: row["group_id"])
        if full_stage_keep_all(plan):
            verify_keep_all_groups(plan, descriptor["execution_plan_sha256"], output, results,
                sorted(stage["group_ids"]), selection_sha256, descriptor["max_images"], profile=profile)
        write_new(output / "stages" / stage["name"] / "completion.json", {
            "status": "complete_stage_mechanical", "errors": [], "stage": stage,
            "assignment_sha256": assignment_hash, "groups": results,
            "scientific_acceptance": False, "independent_confirmation": False, "completed_at": utc_now()})
        return results
    except BaseException:
        stop_owned_workers(processes)
        write_new(output / "stages" / stage["name"] / "abort.json", {
            "status": "failed", "partial_evidence_preserved": True,
            "workers": [{"pid": p.pid, "exit_code": p.returncode} for p in processes]})
        raise
    finally:
        for handle in handles:
            handle.close()


def write_reports(output, records):
    columns = ["group_id", "variant", "seed", "source", "target"] + list(COCO_BBOX_METRICS)
    with (output / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(dict({key: row[key] for key in columns[:5]},
                **{key: "{:.4f}".format(row["metrics"][key]) for key in COCO_BBOX_METRICS}))
    lines = [r"\begin{longtable}{rlrllr}", r"Group & Variant & Seed & Source & Target & AP (\%) \\", r"\hline"]
    for row in records:
        values = [str(row["group_id"]), row["variant"], str(row["seed"]), row["source"], row["target"]]
        values = [value.replace("_", r"\_") for value in values]
        lines.append(" & ".join(values + ["{:.4f}".format(100 * row["metrics"]["bbox_mAP"])]) + r" \\")
    lines.append(r"\end{longtable}")
    (output / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(registry, plan_path, plan_sha256, admission_path, admission_sha256, devices, max_images, output=None, *, profile=None):
    from .retention import build_retained_image_selection

    api = _execution_api(profile)
    plan_path, admission_path = Path(plan_path).resolve(strict=True), Path(admission_path).resolve(strict=True)
    plan_path.relative_to(registry.root)
    admission_path.relative_to(registry.root)
    plan = read_bound(plan_path, plan_sha256)
    api.verify_execution_plan(registry, plan)
    verify_environment(plan)
    admission = api.verify_admission(registry, plan, plan_sha256, admission_path, admission_sha256, max_images)
    if api.EXECUTION_PROTOCOL in ("vcsf_single_source_operator_factorial_execution", "vcsf_single_source_common_anchor_execution", "vcsf_final_attribution_execution", "vcsf_final_background_execution"):
        api.verify_device_admission(admission, devices)
    require(len(devices) in plan["definition"]["device_counts"] and len(set(devices)) == len(devices),
        "Select a registered number of distinct explicitly available GPUs")
    selected = api.execution_groups(plan, max_images)
    payload_policy = keep_all_policy(plan, selected, max_images) if full_stage_keep_all(plan) else None
    require(payload_policy is None or profile is not None,
        "Full-stage keep-all requires an explicit execution profile")
    lanes = api.balanced_lanes(selected, plan["sources"], devices)
    stages = api.scheduled_stages(plan, selected)
    uuids = physical_gpus(devices)
    namespace = "experiments" if max_images is None else "diagnostics"
    parent = registry.root / "outputs" / namespace / api.EXECUTION_PROTOCOL
    output = Path(output).absolute() if output else parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.relative_to(parent)
    require(output.parent == parent, "Every execution root must be a new direct child of its protocol namespace")
    child(registry.root, output.relative_to(registry.root))
    require(not output.exists(), "No automatic resume, run merge or overwrite is supported")
    verify_capacity(registry, plan)
    with gpu_reservations(uuids) as reservations:
        for gpu_uuid in uuids:
            verify_device_owner(gpu_uuid, os.getpid())
        input_contract(registry, plan)
        output.mkdir(parents=True, exist_ok=False)
        if payload_policy is not None:
            selection_hash = write_new(output / "payload_policy.json", payload_policy)
        else:
            selection = build_retained_image_selection(registry, "coco", "val",
                plan["definition"]["payload_retention"]["retained_images"], output / "retained_selection.json")
            require(selection["retained_image_ids_sha256"] ==
                plan["definition"]["payload_retention"]["retained_image_ids_sha256"],
                "Predeclared retention selection mismatch")
            selection_hash = file_digest(output / "retained_selection.json")
        binding = {"execution_root": output.relative_to(registry.root).as_posix(),
            "execution_plan_sha256": plan_sha256, "admission_sha256": admission_sha256,
            "max_images": max_images, "group_ids": [g["group_id"] for g in selected],
            "devices": devices, "stages": stages, "created_at": utc_now(),
            "payload_authorization": plan["payload_authorization"]}
        if payload_policy is not None:
            binding.update(payload_policy_sha256=selection_hash,
                execution_lifecycle="full_stage_keep_all_v1")
        binding_hash = write_new(output / "plan_binding.json", binding)
        descriptor = {"candidate_id": api.CANDIDATE_ID, "execution_alias": api.ALIAS,
            "mode": api.MODE, "execution_plan": plan_path.relative_to(registry.root).as_posix(),
            "execution_plan_sha256": plan_sha256, "admission": admission_path.relative_to(registry.root).as_posix(),
            "admission_sha256": admission_sha256, "execution_root": binding["execution_root"],
            "run_binding_sha256": binding_hash, "max_images": max_images}
        state = {"status": "running", "protocol": api.EXECUTION_PROTOCOL, "coordinator_pid": os.getpid(),
            "coordinator_pid_start_ticks": process_start_ticks(os.getpid()),
            "execution_plan_sha256": plan_sha256, "expected_groups": len(selected),
            "expected_evaluations": len(selected) * len(plan["targets"]),
            "completed_groups": 0, "completed_evaluations": 0, "failed_records": 0,
            "started_at": utc_now(), "max_images": max_images, "scientific_acceptance": False}
        atomic_json(output / "execution_state.json", state)
        completed, records = [], []
        try:
            for stage in stages:
                verify_capacity(registry, plan, stage["group_ids"] if payload_policy is not None else None)
                results = run_stage(registry, plan, descriptor, output, stage, devices, uuids,
                    reservations, lanes, selection_hash, state, profile=profile)
                completed.extend(results)
                for result in results:
                    records.extend(read_bound(output / "groups" / "{:06d}".format(result["group_id"]) /
                        "records.json", result["records_sha256"]))
                records.sort(key=lambda row: (row["group_id"], plan["targets"].index(row["target"])))
                atomic_json(output / "records.json", records)
                state.update(completed_groups=len(completed), completed_evaluations=len(records), workers=[])
                atomic_json(output / "execution_state.json", state)
            require({r["group_id"] for r in completed} == {g["group_id"] for g in selected}
                and len(records) == len(selected) * len(plan["targets"]), "Incomplete final execution matrix")
            api.verify_execution_plan(registry, plan)
            input_contract(registry, plan)
            read_bound(output / "plan_binding.json", binding_hash)
            write_reports(output, records)
            artifact_manifest = {p.relative_to(output).as_posix(): file_digest(p)
                for p in sorted(output.rglob("*")) if p.is_file()
                and not p.name.endswith((".png", ".log")) and p != output / "execution_state.json"}
            manifest_hash = write_new(output / "artifact_manifest.json", {"artifacts": artifact_manifest})
            if payload_policy is not None:
                verify_keep_all_groups(plan, plan_sha256, output,
                    sorted(completed, key=lambda row: row["group_id"]), sorted(plan["new_group_ids"]),
                    selection_hash, max_images, profile=profile)
                current_names = set()
                for path in output.rglob("*"):
                    require(not path.is_symlink(), "A keep-all root artifact became a symlink")
                    if path.is_file() and not path.name.endswith((".png", ".log")) \
                            and path not in (output / "execution_state.json", output / "artifact_manifest.json"):
                        current_names.add(path.relative_to(output).as_posix())
                require(current_names == set(artifact_manifest),
                    "Keep-all root artifact set changed before completion")
                for name, digest in artifact_manifest.items():
                    require(file_digest(child(output, name)) == digest,
                        "Keep-all root artifact bytes changed before completion")
                read_bound(output / "artifact_manifest.json", manifest_hash)
            write_new(output / "completion.json", {"status": "complete_pending_independent_acceptance",
                "errors": [], "groups": len(completed), "evaluations": len(records),
                "generated_images": sum(r["images"] for r in completed),
                "execution_plan_sha256": plan_sha256, "artifact_manifest_sha256": manifest_hash,
                "records_sha256": file_digest(output / "records.json"), "completed_groups": completed,
                "max_images": max_images, "independent_confirmation": False, "scientific_acceptance": False,
                "formal_metrics_eligible": False, "completed_at": utc_now()})
            state.update(status="complete_pending_independent_acceptance", completed_at=utc_now(), workers=[])
            atomic_json(output / "execution_state.json", state)
        except BaseException as exc:
            state.update(status="failed", failed_records=1, reason=repr(exc)[:2000], updated_at=utc_now())
            atomic_json(output / "execution_state.json", state)
            raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execution-plan", type=Path)
    parser.add_argument("--execution-plan-sha256")
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--admission-sha256")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--max-images", type=int,
        help="Diagnostic only: limit images on the first pending complete source pair; keep all payloads.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_request:
        require(args.worker_request_sha256, "Worker request hash is required")
        return run_worker(args.worker_request, args.worker_request_sha256)
    registry = Registry()
    if args.plan_only:
        if args.execution_plan:
            plan = read_bound(args.execution_plan, args.execution_plan_sha256)
            selected = execution_groups(plan, args.max_images)
            lanes = balanced_lanes(selected, plan["sources"], args.devices)
            result = {"status": "inspection_only_no_model_calls", "registered_groups": len(plan["groups"]),
                "qualified_reused_groups": len(plan["reused_group_ids"]), "selected_new_groups": len(selected),
                "new_evaluations": sum(len(g["targets"]) for g in selected),
                "lane_group_ids": [[g["group_id"] for g in lane] for lane in lanes],
                "stages": scheduled_stages(plan, selected), "max_images": args.max_images}
        else:
            prepared = compile_plan(registry)
            result = {"status": "registered_scope_only_reuse_not_loaded",
                "registered_totals": prepared["totals_before_qualified_reuse"],
                "execution_requires_bound_plan_and_admission": True, "model_calls": 0}
        print(json.dumps(result, indent=2))
        return 0
    require(all((args.execution_plan, args.execution_plan_sha256, args.admission, args.admission_sha256)),
        "Execution requires a hash-bound plan and independent admission; use --plan-only for inspection")
    output = execute(registry, args.execution_plan, args.execution_plan_sha256,
        args.admission, args.admission_sha256, args.devices, args.max_images, args.output)
    print("[COMPLETE EXECUTION] " + str(output))
    print("Independent acceptance and complete-matrix scientific analysis remain required.")
    return 0
