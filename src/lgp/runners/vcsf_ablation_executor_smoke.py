"""Complete core executor diagnostic over full-bound inputs, never formal admission."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

from ..io import file_digest
from .vcsf_ablation_execution_contract import _plain, _write_json, verify_execution_contract
from .vcsf_ablation_preflight import balanced_lanes, verify_server
from .vcsf_ablation_runtime_inputs import _pinned_json, verify_published_runtime_inputs
from .vcsf_ablation_stage_plan import PROTOCOL
from .vcsf_efficacy_contract import child, read_bound, require
from .vcsf_research_plan import canonical_hash


EXECUTION_PROTOCOL = "vcsf_single_source_ablation_executor_smoke"
ENTRYPOINT = "experiments/vcsf_single_source_ablation_executor_smoke.py"
ALIAS = "vcsf_ablation_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "single_source_ablation_executor_smoke"
ALLOW_EMPTY_LANES = False


def _pointer(path, digest):
    path = _plain(path)
    _pinned_json(path, digest)
    return dict(file=str(path), sha256=digest)


def _entry_source(registry):
    path = child(registry.root, ENTRYPOINT)
    digest = file_digest(path)
    entry = sys.modules.get("__main__")
    if getattr(entry, "__file__", None) == str(path):
        require(getattr(entry, "_ENTRY_SOURCE_SHA256", None) == digest,
            "Executed smoke entry bytes differ from their snapshot")
    return digest


def compile_runtime_plan(registry, contract_path, contract_sha256, binding_path,
        binding_sha256, receipt_path, receipt_sha256, *, max_images=1):
    verify_server()
    require(type(max_images) is int and max_images == 1,
        "The actual executor diagnostic requires exactly one image per complete core job")
    references = dict(execution_contract=_pointer(contract_path, contract_sha256),
        runtime_inputs=_pointer(binding_path, binding_sha256), input_receipt=_pointer(receipt_path, receipt_sha256))
    original = _pinned_json(contract_path, contract_sha256)
    contract = verify_execution_contract(registry, original)
    science = contract["scientific_contract"]
    require(science["max_images"] is None and all(g["images"] == 5000 for g in science["groups"]),
        "Executor smoke consumes an unchanged full-5000 preparation, not a shortened source contract")
    inputs = verify_published_runtime_inputs(registry, binding_path, binding_sha256,
        receipt_path, receipt_sha256, contract)
    require(inputs["execution_contract_file"]["file"] == str(_plain(contract_path))
        and inputs["execution_contract_file"]["sha256"] == contract_sha256,
        "Runtime inputs were published for a different original contract file")
    definition = deepcopy(registry.protocols[EXECUTION_PROTOCOL])
    require(science["selected_stage"] in definition["supported_stages"], "Unsupported executor diagnostic stage")
    groups = deepcopy(science["groups"])
    ids = [g["group_id"] for g in groups]
    capacity = definition["capacity"]
    budgets = {str(g): dict(payload=capacity["payload_bytes_per_group"],
        prediction_and_metadata=capacity["prediction_and_metadata_bytes_per_group"]) for g in ids}
    capacity_contract = dict(policy="full_stage_keep_all_plus_failure_reserve", group_budget_bytes=budgets,
        failure_reserve_bytes=capacity["failure_reserve_bytes"],
        required_free_bytes=sum(sum(row.values()) for row in budgets.values()) + capacity["failure_reserve_bytes"],
        scope="registered_single_image_diagnostic_capacity_not_formal_capacity_or_measured_cost")
    runtime = dict(inputs["runtime_sha256"], **{ENTRYPOINT: _entry_source(registry)})
    plan = dict(schema_version=1, record_type=EXECUTION_PROTOCOL,
        status="bound_complete_core_executor_diagnostic_not_formal_admission", protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, stage=science["selected_stage"], references=references,
        execution_contract_sha256=contract["contract_sha256"], input_binding_sha256=inputs["binding_sha256"],
        prepared_plan=deepcopy(contract["embedded_stage"]), prepared_plan_sha256=canonical_hash(contract["embedded_stage"]),
        stage_science_sha256=science["stage_science_sha256"], synthetic=contract["synthetic"],
        definition=definition, groups=groups, sources=list(science["sources"]), targets=list(science["targets"]),
        max_images=1, new_group_ids=ids, reused_group_ids=[],
        new_job_scope="all_logical_cells_as_fresh_diagnostic_jobs_no_formal_reuse_decision",
        formal_new_group_ids=None, formal_reused_group_ids=None, historical_reuse_qualified=False,
        image_ids_sha256=inputs["image_ids_sha256"], annotation_sha256=inputs["annotation"]["sha256"],
        checkpoint_sha256={name: inputs["checkpoints"][name]["sha256"] for name in science["targets"]},
        base_commit=inputs["base_commit"], packages=deepcopy(inputs["packages"]),
        runtime_sha256=dict(sorted(runtime.items())), runtime_snapshot_sha256=canonical_hash(runtime),
        analysis_contract=deepcopy(contract["analysis_contract"]),
        result_slots=deepcopy(contract["result_slots"]), result_slots_scope="full_scientific_contract_not_smoke_metrics",
        capacity_contract=capacity_contract, execution_lifecycle="full_stage_keep_all_v1",
        payload_authorization=dict(mode="keep_all", deletion_authorized=False, preserve_failed_and_partial=True),
        expected_generated_images=len(ids), expected_target_evaluations=len(ids) * len(science["targets"]),
        diagnostic_only=True, model_calls_before_execution=0, AP_evaluations_before_execution=0,
        formal_execution_admission=False, formal_metrics_eligible=False, scientific_acceptance=False,
        independent_acceptance=False, independent_confirmation=False, device_mode_qualified=False,
        physical_cost_calibration_accepted=False, automatic_promotion=False)
    plan["plan_content_sha256"] = canonical_hash(plan)
    for pointer in references.values():
        _pinned_json(pointer["file"], pointer["sha256"])
    return plan


def verify_execution_plan(registry, plan):
    require(isinstance(plan, dict) and plan.get("record_type") == EXECUTION_PROTOCOL,
        "Not a core executor diagnostic plan")
    refs = plan["references"]
    args = [value for key in ("execution_contract", "runtime_inputs", "input_receipt")
        for value in (refs[key]["file"], refs[key]["sha256"])]
    current = compile_runtime_plan(registry, *args, max_images=plan["max_images"])
    require(canonical_hash(current) == canonical_hash(plan),
        "Core executor plan differs from its original full-input reconstruction")
    return current


def execution_groups(plan, max_images):
    require(type(max_images) is int and max_images == 1 and type(plan.get("max_images")) is int
        and plan["max_images"] == 1 and plan.get("diagnostic_only") is True
        and plan.get("formal_execution_admission") is False, "Smoke request cannot admit formal or enlarged work")
    groups = plan["groups"]
    require(canonical_hash([g["group_id"] for g in groups]) == canonical_hash(list(range(1, 9)))
        and canonical_hash(plan["new_group_ids"]) == canonical_hash(list(range(1, 9)))
        and plan["reused_group_ids"] == [] and plan["formal_new_group_ids"] is None
        and plan["formal_reused_group_ids"] is None and plan["historical_reuse_qualified"] is False
        and len(plan["targets"]) == 16, "Executor smoke must retain all eight fresh diagnostic jobs and all targets")
    return groups


def scheduled_stages(plan, selected):
    require(canonical_hash(selected) == canonical_hash(execution_groups(plan, 1)), "Partial smoke stage is forbidden")
    return [dict(name=plan["stage"], group_ids=list(plan["new_group_ids"]))]


def execution_metadata(plan, plan_sha256, group, max_images):
    require(group in execution_groups(plan, max_images), "Foreign diagnostic group")
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL, study="vcsf_single_source_ablation",
        stage=plan["stage"], variant=group["variant"], group_id=group["group_id"], seed=group["seed"],
        execution_plan_sha256=plan_sha256, prepared_plan_sha256=plan["prepared_plan_sha256"],
        runtime_snapshot_sha256=plan["runtime_snapshot_sha256"], input_binding_sha256=plan["input_binding_sha256"],
        smoke_max_images=max_images, diagnostic_only=True, formal_metrics_eligible=False,
        independent_confirmation=False, stage_selection=False)


def diagnostic_request(plan, plan_hash):
    execution_groups(plan, 1)
    return dict(status="bounded_complete_executor_request_not_formal_admission", protocol=EXECUTION_PROTOCOL,
        execution_plan_sha256=plan_hash, runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_binding_sha256=plan["input_binding_sha256"], new_group_ids=list(plan["new_group_ids"]),
        reused_group_ids=[], new_job_scope=plan["new_job_scope"], max_images=1,
        target_order=list(plan["targets"]), capacity_contract_sha256=canonical_hash(plan["capacity_contract"]),
        analysis_contract_sha256=canonical_hash(plan["analysis_contract"]),
        expected_generated_images=8, expected_target_evaluations=128,
        formal_execution_admission=False, independent_acceptance=False,
        scientific_acceptance=False, device_mode_qualified=False)


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    execution_groups(plan, max_images)
    path = _plain(path)
    require(path.name == "diagnostic_request.json"
        and path.parent.parent == _plain(registry.root) / "outputs/plans" / EXECUTION_PROTOCOL,
        "Executor diagnostic request escaped its preparation leaf")
    require({p.name for p in path.parent.iterdir()} == {"execution_plan.json", "diagnostic_request.json"}
        and all(p.is_file() and not p.is_symlink() for p in path.parent.iterdir()),
        "Executor diagnostic preparation is incomplete, failed or contains unexpected artifacts")
    original = _pinned_json(path.parent / "execution_plan.json", plan_hash)
    require(canonical_hash(original) == canonical_hash(plan), "Request and plan come from different preparations")
    request = _pinned_json(path, digest)
    require(canonical_hash(request) == canonical_hash(diagnostic_request(plan, plan_hash)),
        "Not the exact complete executor diagnostic request")
    return request


def publish_plan(registry, output, plan):
    output = _plain(output)
    require(output.parent == _plain(registry.root) / "outputs/plans" / EXECUTION_PROTOCOL
        and not output.exists(), "Use a fresh direct executor-smoke preparation leaf")
    verify_execution_plan(registry, plan)
    output.mkdir(parents=True, exist_ok=False)
    try:
        pin = _write_json(output / "execution_plan.json", plan)
        request = diagnostic_request(plan, pin["sha256"])
        verify_execution_plan(registry, plan)
        _pinned_json(output / "execution_plan.json", pin["sha256"])
        request_pin = _write_json(output / "diagnostic_request.json", request)
        verify_admission(registry, plan, pin["sha256"], output / "diagnostic_request.json", request_pin["sha256"], 1)
        return dict(execution_plan=pin, diagnostic_request=request_pin, formal_execution_admission=False)
    except BaseException as exc:
        _write_json(output / "preparation_failure.json", dict(status="failed_preserved", reason=type(exc).__name__))
        raise
