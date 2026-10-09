"""Reconstruct original input equivalence and original saved-cost attribution."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.metadata
from pathlib import Path

from .vcsf_ablation_runtime_inputs import _normal, _read_json
from .vcsf_core_result_origins import _BoundMetadata
from .vcsf_research_plan import canonical_hash, require
from .vcsf_scale_execution_contract import normalized_model_config
from .vcsf_stage_evidence import _parse


def _raw(reference, checked):
    path = Path(reference["file"])
    require(path.is_absolute() and path == path.resolve()
        and not any(p.is_symlink() for p in (path, *path.parents)), "Original input path is not plain")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "Original input bytes changed")
    checked[str(path)] = reference["sha256"]
    return raw


def verify_original_inputs(binding, origin_row, read):
    """Binding and origin_row must first pass their original public reconstruction."""
    from mmengine.config import Config

    original, group, origins = origin_row["original"], origin_row["group"], origin_row["result_origins"]
    first, checked = origins[0], {}
    evidence = read.json(first["group_evidence"]["file"], first["group_evidence"]["sha256"])
    read.extend(evidence["checked_input_sha256"])
    generation = read.json(first["generation"]["file"], first["generation"]["sha256"])
    plan = read.json(first["execution_plan"]["file"], first["execution_plan"]["sha256"])
    ids = binding["ordered_image_ids"]
    require(len(ids) == len(set(ids)) == 5000 and binding["full_split_images"] == 5000
        and binding["dataset"] == generation["dataset"] == "coco"
        and binding["split"] == generation["split"] == "val"
        and generation["requested_images"] == generation["successful_images"] == 5000
        and generation["failed_images"] == 0 and generation["source"] == group["source"]
        and generation["image_ids_sha256"] == generation["requested_ordered_image_ids_sha256"]
            == plan["image_ids_sha256"] == binding["image_ids_sha256"] == canonical_hash(ids),
        "Original generation and current binding have different complete image panels")
    require(generation["seed"] == binding["seed"] == group["seed"] == 42
        and generation["seed_schedule"] == binding["seed_schedule"] == "selected_position"
        and generation["seed_offsets_sha256"] == canonical_hash([[value, i] for i, value in enumerate(ids)])
        and binding["seed_mapping_sha256"] == canonical_hash([[value, i, 42+i] for i, value in enumerate(ids)]),
        "Original and current selected-position seed schedules differ")
    require(first["canonical_annotation"]["sha256"] == binding["annotation"]["sha256"] == plan["annotation_sha256"],
        "Original canonical annotation differs from current input; exported annotation is not interchangeable")
    current_rows = _read_json(binding["clean_image_manifest"])
    raw = _raw(first["manifest"], checked)
    old_rows = [_parse(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    require(len(old_rows) == len(current_rows) == 5000, "Incomplete original or current image/seed manifest")
    full_byte_audit = first["pixel_audit_scope"]["clean_content_hashes"] == 5000
    for position, (old, current) in enumerate(zip(old_rows, current_rows)):
        require(old["status"] == "ok" and all(type(old[key]) is int and type(current[key]) is int
            and old[key] == current[key] == value for key, value in
            (("image_id", ids[position]), ("position", position), ("attack_seed", 42+position))),
            "Original image order, status or per-image attack seed differs")
        old_path = Path(old["source_file"]).resolve(strict=True)
        require(old_path == Path(current["input"]["file"]).resolve(strict=True),
            "Original generation read another canonical clean image")
        if full_byte_audit:
            require(evidence["checked_input_sha256"].get(str(old_path)) == current["input"]["sha256"],
                "Original independently audited clean bytes differ from current input")
    targets = [row["target"] for row in origins]
    require(targets == plan["targets"] and set(binding["checkpoints"]) == set(targets)
        and len(targets) == len(set(targets)) == 16
        and all(plan["checkpoint_sha256"][target] == binding["checkpoints"][target]["sha256"]
            == origin["checkpoint_sha256"] for target, origin in zip(targets, origins))
        and generation["checkpoint_sha256"] == binding["checkpoints"][group["source"]]["sha256"],
        "Original source or complete target checkpoint panel differs")
    packages = plan["packages"]
    require(packages and all(importlib.metadata.version(name) == version for name, version in packages.items())
        and all(packages.get(name) == version for name, version in binding["packages"].items()),
        "Recorded original package versions differ from the current pinned runtime")
    group_root = Path(first["group_receipt"]["file"]).parent
    receipt = read.json(first["group_receipt"]["file"], first["group_receipt"]["sha256"])
    inventory = read.json(group_root/"artifact_manifest.json", receipt["artifact_manifest_sha256"])["artifacts"]
    expected_roles = [("source", group["source"])] + [("target", target) for target in targets]
    require([(row["role"], row["model"]) for row in binding["model_configs"]] == expected_roles,
        "Current source/target config roles are incomplete or reordered")
    configs = []
    for current, (role, model) in zip(binding["model_configs"], expected_roles):
        relative = "attack/model/runtime_config.py" if role == "source" else "evaluations/"+model+"/model/runtime_config.py"
        path = group_root/relative
        reference = dict(file=str(path), sha256=inventory[relative]["sha256"])
        _raw(reference, checked)
        old_config = Config.fromfile(str(path)).to_dict()
        old_config.pop("work_dir", None)
        old_config = _normal(old_config)
        current_config = _read_json(current["effective_config"])
        old_projected = normalized_model_config(old_config)
        current_projected = normalized_model_config(current_config)
        require(canonical_hash(old_projected) == canonical_hash(current_projected),
            "Original resolved detector/preprocessing/evaluator config differs: "+role+":"+model)
        configs.append(dict(role=role, model=model, original_resolved_config=reference,
            current_effective_config=deepcopy(current["effective_config"]),
            original_raw_effective_sha256=canonical_hash(old_config),
            current_raw_effective_sha256=canonical_hash(current_config),
            compared_effective_sha256=canonical_hash(old_projected),
            normalization="exclude_work_dir_resolve_absolute_paths_json_sequences_preserve_other_fields"))
    for path, digest in list(checked.items()):
        _raw(dict(file=path, sha256=digest), checked)
    read.unchanged()
    result = dict(status="original_inputs_match_current_bound_single_group_conditions",
        original_identity_sha256=canonical_hash(original), current_binding_sha256=binding["binding_sha256"],
        image_ids_sha256=binding["image_ids_sha256"], images=5000,
        seed_mapping_sha256=binding["seed_mapping_sha256"], original_seed_offsets_sha256=generation["seed_offsets_sha256"],
        canonical_annotation_sha256=plan["annotation_sha256"], checkpoint_sha256=deepcopy(plan["checkpoint_sha256"]),
        original_recorded_packages=deepcopy(packages), model_configs=configs,
        clean_byte_comparison_count=5000 if full_byte_audit else None,
        original_clean_evidence_scope=deepcopy(first["pixel_audit_scope"]),
        original_protocol_input_chain_preserved=True, generation_instant_clean_content_hash_claim=False,
        complete_installed_package_byte_identity_claim=False, checked_auxiliary_sha256=checked,
        formal_reuse_qualified=False, physical_cost_inheritance=False, new_model_calls=0, new_AP_evaluations=0)
    result["input_mapping_sha256"] = canonical_hash(result)
    return result


def bind_original_saved_cost(stage):
    proof = stage["proof"]
    reference = stage["references"]["cost_acceptance"]
    read = _BoundMetadata(proof["checked_input_sha256"])
    read.extend({reference["file"]: reference["sha256"]})
    accepted = read.json(reference["file"], reference["sha256"])
    require(accepted["cost_report_sha256"] == proof["cost_report_sha256"], "Original cost report changed")
    report_ref = read.reference(accepted["cost_report_file"], accepted["cost_report_sha256"])
    report = read.json(report_ref["file"], report_ref["sha256"])
    rows = report["groups"]
    require(len(rows) == 17 and len({row["generation_sha256"] for row in rows}) == 17,
        "Original saved-cost generation inventory is incomplete or duplicated")
    lookup = {row["generation_sha256"]: (i, row) for i, row in enumerate(rows)}
    timer_rows = report["timer_source_binding"]
    require(len(timer_rows) == 17 and len({row["group_id"] for row in timer_rows}) == 17,
        "Original timer bindings are incomplete or duplicated")
    timers = {row["group_id"]: row for row in timer_rows}
    result = {}
    for original in proof["groups"]:
        group = original["group"]
        digest = original["original"]["generation_sha256"]
        require(digest in lookup and digest == original["generation_sha256"], "Original cost has another generation")
        position, row = lookup[digest]
        require(row["generation_file"] == original["generation_file"]
            and row["group_id"] == group["group_id"] and row["original_variant"] == group["variant"]
            and row["parameters_sha256"] == group["parameters_sha256"]
            and row["images"] == 5000 and row["image_ids_sha256"] == original["image_ids_sha256"]
            and all(row[key] is None for key in ("physical_full_detector_backward_equivalents",
                "measured_kernel_flops", "per_image_peak_memory")), "Original saved-cost row identity or unknown quantities changed")
        timer = timers[group["group_id"]]
        require(timer["execution_plan_sha256"] == original["original_execution_plan_sha256"]
            and timer["runtime_snapshot_sha256"] == original["runtime_snapshot_sha256"],
            "Original cost timer uses another execution plan")
        result[digest] = dict(status="verified_cost_attribution_to_original_generation_only",
            cost_acceptance=deepcopy(reference), report=report_ref, position=position,
            row_sha256=canonical_hash(row), row=deepcopy(row), timer_source_binding=deepcopy(timer),
            physical_cost_inheritance=False, current_run_time_or_memory_measured=False,
            physical_cost_calibration_accepted=False, scientific_acceptance=False)
    read.unchanged()
    return result
