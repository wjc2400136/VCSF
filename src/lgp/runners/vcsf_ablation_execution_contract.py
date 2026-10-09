"""Prepare a complete core execution contract without arming an executor.

Logical cells are not a new-job declaration: historical reuse, runtime inputs,
actual device qualification and independent admission remain separate bindings.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS
from ..reporting.vcsf_ablation_cost import compile_cost_contract, read_stage_plan
from ..registry import Registry
from .vcsf_ablation_stage_plan import PROTOCOL, compile_stage_plan
from .vcsf_efficacy_contract import child, require
from .vcsf_research_plan import canonical_hash


CONTRACT_PROTOCOL = "vcsf_single_source_ablation_execution_contract"
ENTRYPOINT = "experiments/prepare_vcsf_single_source_ablation_execution.py"
RECORD_TYPE = "vcsf_ablation_execution_contract_preparation"
SOURCE_FILES = (
    ENTRYPOINT, "src/lgp/runners/vcsf_ablation_execution_contract.py",
    "experiments/vcsf_single_source_ablation_reuse.py",
    "experiments/audit_vcsf_single_source_ablation_reuse.py",
    "src/lgp/runners/vcsf_efficacy_contract.py", "src/lgp/metrics.py",
    "src/lgp/data/coco.py", "src/lgp/paths.py", "environment.yml",
    "requirements/locked-cu118.txt",
)


def _process_source(root):
    names = {p.relative_to(root).as_posix() for directory, suffixes in
        (("src/lgp", (".py",)), ("configs", (".yaml", ".yml")))
        for p in (root / directory).rglob("*") if p.is_file() and p.suffix in suffixes}
    names.update(SOURCE_FILES)
    return {name: file_digest(child(root, name)) for name in sorted(names)}


_MODULE_ROOT = Path(__file__).resolve().parents[3]
_IMPORTED_SOURCE = _process_source(_MODULE_ROOT)


def _verify_process_binding(registry):
    require(_process_source(registry.root) == _IMPORTED_SOURCE,
        "Preparation source or configuration changed after module import; use a fresh process")
    fresh = Registry(registry.root)
    require(registry.__dict__ == fresh.__dict__, "In-memory registry differs from the current source snapshot")
    entry = sys.modules.get("__main__")
    entry_name = getattr(entry, "__file__", None)
    for public_entry in (ENTRYPOINT, "experiments/vcsf_single_source_ablation_reuse.py"):
        if entry_name == str(registry.root / public_entry):
            require(getattr(entry, "_ENTRY_SOURCE_SHA256", None) == _IMPORTED_SOURCE[public_entry],
                "Public preparation entry bytes differ from its source inventory")
    for name, module in list(sys.modules.items()):
        loader = getattr(module, "__loader__", None)
        if name == "lgp" or name.startswith("lgp."):
            require(getattr(loader, "source_only", None) is True,
                "Core preparation requires source-only project imports through its public entry")
            file = getattr(module, "__file__", None)
            require(isinstance(file, str), "A source-bound project module has no source filename")
            path = Path(file)
            require(_MODULE_ROOT in path.parents, "Project import escaped the compiler source root")
            relative = path.relative_to(_MODULE_ROOT).as_posix()
            require(getattr(loader, "source_sha256", None) == _IMPORTED_SOURCE.get(relative),
                "Executed project bytes differ from the preparation snapshot")


def _plain(path):
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Execution-contract paths must be plain and contain no parent traversal")
    return path


def _devices(devices):
    require(isinstance(devices, (list, tuple)) and 1 <= len(devices) <= 2
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices), "Select one or two distinct explicit CUDA ordinals")
    return list(devices)


def _source_inventory(registry, stage, cost):
    _verify_process_binding(registry)
    result = dict(_IMPORTED_SOURCE)
    for mapping in (stage["source_sha256"], cost["source_sha256"], stage["authority_sha256"]):
        for name, digest in mapping.items():
            require(name not in result or result[name] == digest, "Conflicting interpreted-source bindings")
            require(file_digest(child(registry.root, name)) == digest, "Prepared source changed: " + name)
            result[name] = digest
    for name in SOURCE_FILES:
        result[name] = file_digest(child(registry.root, name))
    return dict(sorted(result.items()))


def _stage_evidence(registry, stage):
    if stage["mode"] == "fixture":
        rebuilt = compile_stage_plan(registry, stage["selected_stage"], stage["working_baseline"],
            stage["nonzero_comparator"], mode="fixture", devices=stage["requested_devices"],
            max_images=stage["max_images"])
        require(canonical_hash(rebuilt) == canonical_hash(stage),
            "Fixture stage identity differs from complete reconstruction")
        return dict(status="synthetic_not_reverified", reverified=False, proof=None,
            independent_admission=False)
    original = stage["evidence_proof"]
    require(isinstance(original.get("references"), dict), "Bound stage omitted original evidence references")
    pointer = original["references"]["selection_decision"]
    decision = read_stage_plan(_plain(pointer["file"]), pointer["sha256"])
    rebuilt = compile_stage_plan(registry, stage["selected_stage"], decision["working_baseline"],
        decision["nonzero_comparator"], mode="bound_preparation", evidence=original["references"],
        devices=stage["requested_devices"], max_images=stage["max_images"])
    require(canonical_hash(rebuilt) == canonical_hash(stage),
        "Reconstructed original decision representation or complete bound stage differs")
    proof = rebuilt["evidence_proof"]
    return dict(status="preceding_scale_acceptance_and_selection_reverified_for_preparation",
        reverified=True, proof=proof, independent_admission=False)


def compile_execution_contract(registry, stage_path, stage_sha256, cost_path, cost_sha256,
        *, devices=None, max_images=None):
    """Reconstruct the original stage/cost pair; do not resolve pending run gates."""
    _verify_process_binding(registry)
    registry._validate()
    stage_path, cost_path = _plain(stage_path), _plain(cost_path)
    require(stage_path != cost_path, "Stage and prospective cost must be separate original files")
    stage = read_stage_plan(stage_path, stage_sha256)
    cost = read_stage_plan(cost_path, cost_sha256)
    definition = deepcopy(registry.protocols[CONTRACT_PROTOCOL])
    require(stage.get("selected_stage") in definition["supported_stages"],
        "Only the registered core execution-contract stage is implemented")
    require(max_images is None or type(max_images) is int and 0 < max_images < 5000,
        "Diagnostic image limit must be an integer from 1 through 4999")
    require(stage.get("max_images") == max_images and type(stage.get("max_images")) is type(max_images),
        "Execution-contract image limit must match the original stage preparation")
    current_cost = compile_cost_contract(registry, stage, devices=cost.get("requested_devices"))
    require(canonical_hash(current_cost) == canonical_hash(cost),
        "Saved prospective cost contract differs from exact reconstruction")
    devices = _devices(stage["requested_devices"] if devices is None else devices)
    proof = _stage_evidence(registry, stage)
    groups = deepcopy(stage["groups"])
    require(stage["sources"] == registry.source_ids()[:1] and stage["targets"] == registry.target_ids()
        and stage["dataset"] == "coco" and stage["split"] == "val" and stage["seeds"] == [42]
        and [g["group_id"] for g in groups] == list(range(1, 9))
        and len({g["parameters_sha256"] for g in groups}) == 8
        and all(g["images"] == (5000 if max_images is None else max_images)
            and g["reuse_status"] == "unassessed_do_not_subtract" and g["execution_status"] == "not_armed"
            for g in groups), "The complete eight-cell core design changed")
    analysis = deepcopy(stage["analysis"])
    require(analysis["family_size"] == len(analysis["contrasts"]) == 32
        and analysis["mode"] == "point_estimate_only"
        and analysis["uncertainty"]["bootstrap_required"] is False,
        "The exact core comparison family or point-estimate boundary changed")
    slots = [dict(group_id=g["group_id"], variant=g["variant"], source=g["source"], target=target,
        seed=g["seed"], images=g["images"], parameters_sha256=g["parameters_sha256"],
        source_matched=target == g["source"], included_in_BB_mean=target != g["source"],
        status="NR", result_file=None, result_sha256=None, prediction_archive_sha256=None,
        independent_acceptance_sha256=None, metrics={key: None for key in COCO_BBOX_METRICS})
        for g in groups for target in stage["targets"]]
    model_slots = [dict(role=role, model=model, checkpoint_file=None, checkpoint_sha256=None,
        resolved_config_file=None, resolved_config_sha256=None, effective_config_sha256=None,
        status="not_bound") for role, model in
        [("source", stage["sources"][0])] + [("target", t) for t in stage["targets"]]]
    gate_rows = [dict(name=name, status="not_bound", receipt_file=None, receipt_sha256=None)
        for name in definition["required_admission_evidence"]]
    if proof["reverified"]:
        gate_rows[0].update(status="reverified_preparation_only", proof_sha256=canonical_hash(proof["proof"]))
    source = _source_inventory(registry, stage, cost)
    scientific = dict(parent_protocol=PROTOCOL, selected_stage=stage["selected_stage"],
        stage_science_sha256=stage["science_sha256"], cost_science_sha256=cost["science_sha256"],
        groups=groups, analysis=analysis, sources=list(stage["sources"]), targets=list(stage["targets"]),
        dataset=stage["dataset"], split=stage["split"], seeds=list(stage["seeds"]), max_images=max_images,
        logical_group_ids=[g["group_id"] for g in groups],
        totals_before_qualified_reuse=deepcopy(stage["totals_before_qualified_reuse"]),
        original_source_inventories=deepcopy(stage["original_source_inventories"]),
        working_baseline=deepcopy(stage["working_baseline"]),
        nonzero_comparator=deepcopy(stage["nonzero_comparator"]),
        selected_working_variant=stage["selected_working_variant"],
        factorial_high_variant=stage["factorial_high_variant"],
        current_stage_identity_sha256=stage["current_stage_identity_sha256"],
        definition=definition, source_sha256=source)
    result = dict(schema_version=1, record_type=RECORD_TYPE, protocol=CONTRACT_PROTOCOL,
        status="prepared_fixture_core_execution_contract_not_executable" if stage["synthetic"] else
            "prepared_bound_core_execution_contract_not_executable",
        preparation_mode=stage["mode"], synthetic=stage["synthetic"],
        original_stage=dict(file=str(stage_path), sha256=stage_sha256),
        original_cost_contract=dict(file=str(cost_path), sha256=cost_sha256),
        embedded_stage=stage, embedded_cost_contract=cost, scientific_contract=scientific,
        science_sha256=canonical_hash(scientific), stage_evidence=proof,
        source_sha256=source, source_snapshot_sha256=canonical_hash(source),
        logical_group_ids=scientific["logical_group_ids"],
        new_group_ids=None, qualified_reused_group_ids=None, historical_reuse_qualified=False,
        new_generation_image_instances=None, new_target_evaluations=None,
        requested_devices=devices, device_availability_checked=False,
        lane_plan=[dict(device=d, logical_group_ids=[g["group_id"] for g in groups[i::len(devices)]])
            for i, d in enumerate(devices)],
        input_contract=dict(status="not_bound", full_split_images=5000,
            requested_images_per_group=5000 if max_images is None else max_images,
            ordered_image_ids=None, ordered_image_ids_sha256=None, annotation_file=None, annotation_sha256=None,
            clean_image_manifest_file=None, clean_image_manifest_sha256=None, seed_schedule="selected_position",
            seed_mapping=None, seed_mapping_sha256=None, model_slots=model_slots),
        runtime_binding=dict(status="not_bound", base_commit=None, packages=None,
            execution_runtime_sha256=None, runtime_snapshot_sha256=None, tested_executor_receipt_sha256=None),
        result_slots=slots, prediction_index_binding=dict(status="not_bound", file=None, sha256=None),
        analysis_contract=dict(status="prospective_exact_family_not_result_acceptance",
            contrasts=deepcopy(analysis["contrasts"]), contrast_family_sha256=canonical_hash(analysis["contrasts"]),
            count=32, metrics=list(COCO_BBOX_METRICS), target_order=list(stage["targets"]),
            blackbox_targets=[t for t in stage["targets"] if t != stage["sources"][0]],
            bootstrap_required=False, significance_claim=False, independent_confirmation=False),
        lifecycle_contract=dict(payload_retention=deepcopy(definition["payload_retention"]),
            failure_policy=definition["failure_policy"], output_policy=definition["output_policy"],
            required_seal_checks=["complete_ordered_generation_and_seed_mapping", "decoded_png_hashes_and_pixel_bound",
                "all_sixteen_targets_and_twelve_unrounded_metrics", "all_prediction_archives_and_input_identity",
                "source_trace_and_cost_reconciliation", "complete_group_artifact_manifest"],
            producer_seal_is_independent_acceptance=False, automatic_resume=False, run_merge=False,
            overwrite=False, actual_retention_performed=False, actual_deletion_performed=False),
        capacity_contract=dict(policy=definition["capacity_policy"], status="not_bound",
            complete_stage_payload_bytes=None, prediction_and_metadata_bytes=None, failure_reserve_bytes=None,
            required_free_bytes=None, observed_free_bytes=None, sufficient_capacity_verified=False),
        required_admission_evidence=gate_rows, preparation_only=True, runner_armed=False,
        execution_profile_installed=False, device_mode_qualified=False, formal_execution_admission=False,
        actual_source_controls_accepted=False, physical_cost_calibration_accepted=False,
        independent_result_acceptance=False, scientific_acceptance=False, independent_confirmation=False,
        automatic_promotion=False, model_calls=0, AP_evaluations=0)
    result["contract_sha256"] = canonical_hash(result)
    require(read_stage_plan(stage_path, stage_sha256) == stage
        and read_stage_plan(cost_path, cost_sha256) == cost, "Original preparations changed while being read")
    require(_source_inventory(registry, stage, cost) == source, "Contract source changed during preparation")
    return result


def verify_execution_contract(registry, contract):
    require(isinstance(contract, dict) and contract.get("record_type") == RECORD_TYPE,
        "Not a core execution-contract preparation")
    stage, cost = contract["original_stage"], contract["original_cost_contract"]
    current = compile_execution_contract(registry, stage["file"], stage["sha256"], cost["file"], cost["sha256"],
        devices=contract["requested_devices"], max_images=contract["scientific_contract"]["max_images"])
    require(canonical_hash(contract) == canonical_hash(current),
        "Execution-contract content differs from original evidence reconstruction")
    return current


def _write_bytes(path, raw):
    digest = hashlib.sha256(raw).hexdigest()
    with path.open("xb") as handle:
        handle.write(raw)
    return dict(sha256=digest, bytes=len(raw))


def _write_json(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return _write_bytes(path, raw)


def _csv(path, fields, rows):
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: row[key] for key in fields} for row in rows)
    return _write_bytes(path, handle.getvalue().encode("utf-8"))


def _tex(value):
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")


def write_execution_contract(registry, output, contract):
    """Publish reproducible preparation artifacts, never a run/admission receipt."""
    output = _plain(output)
    parent = _plain(registry.root) / "outputs/plans" / CONTRACT_PROTOCOL
    require(output.parent == parent and not output.exists(),
        "Use a fresh direct execution-contract preparation output leaf")
    current = verify_execution_contract(registry, contract)
    output.mkdir(parents=True, exist_ok=False)
    groups = current["scientific_contract"]["groups"]
    artifacts = {}
    artifacts["groups.csv"] = _csv(output / "groups.csv", ("group_id", "variant", "source", "seed", "images", "parameters_sha256",
        "logical_gradients", "risk_updates", "feature_updates", "feature_terms", "reuse_status"), groups)
    artifacts["result_slots.csv"] = _csv(output / "result_slots.csv", ("group_id", "variant", "source", "target", "seed", "images",
        "parameters_sha256", "source_matched", "included_in_BB_mean", "status"), current["result_slots"])
    artifacts["admission_requirements.csv"] = _csv(output / "admission_requirements.csv", ("name", "status"), current["required_admission_evidence"])
    artifacts["lifecycle.json"] = _write_json(output / "lifecycle.json", current["lifecycle_contract"])
    lines = [r"\begin{tabular}{rlrrrr}", r"\hline",
        r"Group & Planned variant & Images & Risk & Feature & Terms \\", r"\hline"]
    for group in groups:
        lines.append("{} & {} & {} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
            group["group_id"], _tex(group["variant"]), group["images"], group["risk_updates"],
            group["feature_updates"], group["feature_terms"]))
    lines.extend([r"\hline", r"\end{tabular}", r"\par\noindent\footnotesize "
        + ("Synthetic fixture. " if current["synthetic"] else "Bound preparation. ")
        + "Planned logical cells before qualified reuse; not execution, measured cost, AP or admission."])
    artifacts["planned_groups.tex"] = _write_bytes(output / "planned_groups.tex", ("\n".join(lines) + "\n").encode("utf-8"))
    artifacts["execution_contract.json"] = _write_json(output / "execution_contract.json", current)
    require(canonical_hash(verify_execution_contract(registry, current)) == canonical_hash(current),
        "Preparation changed before artifact publication")
    require({p.name for p in output.iterdir()} == set(artifacts)
        and all(p.is_file() and not p.is_symlink() for p in output.iterdir()),
        "Unexpected preparation artifact or alias")
    manifest = _write_json(output / "artifact_manifest.json", dict(artifacts=artifacts))
    for name, value in artifacts.items():
        require(file_digest(output / name) == value["sha256"] and (output / name).stat().st_size == value["bytes"],
            "Preparation artifact changed before terminal publication")
    require(file_digest(output / "artifact_manifest.json") == manifest["sha256"],
        "Preparation manifest changed before terminal publication")
    _verify_process_binding(registry)
    for pointer in (current["original_stage"], current["original_cost_contract"]):
        read_stage_plan(_plain(pointer["file"]), pointer["sha256"])
    if current["stage_evidence"]["reverified"]:
        for name in ("cost_acceptance", "point_acceptance", "selection_decision"):
            pointer = current["stage_evidence"]["proof"]["references"][name]
            read_stage_plan(_plain(pointer["file"]), pointer["sha256"])
    require({p.name for p in output.iterdir()} == set(artifacts) | {"artifact_manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in output.iterdir()),
        "Preparation artifact set changed before terminal publication")
    for name, value in dict(artifacts, **{"artifact_manifest.json": manifest}).items():
        require(file_digest(output / name) == value["sha256"] and (output / name).stat().st_size == value["bytes"],
            "Preparation artifact changed during final input checks")
    receipt = dict(status="prepared_contract_artifacts_not_execution_admission", completed_at=
        datetime.now(timezone.utc).isoformat(), contract_sha256=current["contract_sha256"],
        contract_file_sha256=artifacts["execution_contract.json"]["sha256"],
        artifact_manifest_sha256=manifest["sha256"],
        logical_groups=len(groups), logical_target_slots=len(current["result_slots"]), contrasts=32,
        original_stage_sha256=current["original_stage"]["sha256"],
        original_cost_contract_sha256=current["original_cost_contract"]["sha256"],
        preparation_consistency_checked=True, runner_armed=False, formal_execution_admission=False,
        independent_result_acceptance=False, scientific_acceptance=False, model_calls=0, AP_evaluations=0)
    _write_json(output / "preparation_receipt.json", receipt)
    return receipt
