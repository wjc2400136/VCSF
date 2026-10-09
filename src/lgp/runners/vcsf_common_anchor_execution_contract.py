"""Runtime profile for independently admitted, fixed-common-anchor complete jobs."""
from copy import deepcopy
from pathlib import Path
import sys

from ..io import file_digest
from .vcsf_efficacy_contract import child, verify_environment
from .vcsf_operator_execution_contract import read_bound, balanced_lanes, verify_device_admission as verify_device_count
from .vcsf_common_anchor_execution_design import compile_execution_design
from .vcsf_research_plan import canonical_hash, require


PROTOCOL = "vcsf_single_source_common_anchor_research"
EXECUTION_PROTOCOL = "vcsf_single_source_common_anchor_execution"
ENTRYPOINT = "experiments/vcsf_common_anchor_experiments.py"
ALIAS = "vcsf_common_anchor_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "single_source_common_anchor_execution"
ALLOW_EMPTY_LANES = False


def verify_device_admission(receipt, devices):
    verify_device_count(receipt, devices)
    require(receipt.get("devices") == list(devices), "Common readiness binds different physical devices")


def read_reference(reference):
    require(isinstance(reference, dict) and set(reference) == {"file", "sha256"}
        and Path(reference["file"]).is_absolute(), "Require an absolute pinned reference")
    return read_bound(reference["file"], reference["sha256"])


def runtime_snapshot(root, *, entrypoint=ENTRYPOINT):
    root = Path(root)
    paths = {p for name in ("src", "tools", "tests", "configs") for p in (root / name).rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml")}
    paths.update(root / p for p in (entrypoint, "environment.yml", "requirements/locked-cu118.txt"))
    return {p.relative_to(root).as_posix(): file_digest(p) for p in sorted(paths)}


def verify_process_source(registry, snapshot):
    from .vcsf_ablation_execution_contract import _verify_process_binding

    _verify_process_binding(registry)
    entry = sys.modules.get("__main__")
    require(getattr(entry, "__file__", None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == snapshot[ENTRYPOINT],
        "Use the bound source-only common execution entry")


def compile_runtime_plan(registry, request, *, max_images=None):
    """Consume already qualified reuse/assets; never issue readiness or run a model."""
    from .vcsf_efficacy_runner import keep_all_policy

    require(isinstance(request, dict) and set(request) == {"anchor_binding", "reuse_partition",
        "reuse_audit", "runtime_inputs", "input_audit", "devices", "capacity_contract"},
        "Require the complete common runtime preparation request")
    require(max_images is None or type(max_images) is int and max_images == 1,
        "Only full5000 or explicit max1 diagnostics are supported")
    binding = read_reference(request["anchor_binding"])
    partition = read_reference(request["reuse_partition"])
    reuse_audit = read_reference(request["reuse_audit"])
    require(reuse_audit["status"] == "independently_verified_common_anchor_reuse_partition"
        and reuse_audit["partition_reference"] == request["reuse_partition"]
        and reuse_audit["anchor_binding_reference"] == request["anchor_binding"]
        and reuse_audit["independent_partition_acceptance"] is True,
        "Reuse audit is incomplete or belongs to another common anchor")
    design = compile_execution_design(registry, binding, partition, request["devices"])
    inputs = read_reference(request["runtime_inputs"])
    input_audit = read_reference(request["input_audit"])
    require(input_audit["status"] == "independently_verified_common_anchor_runtime_inputs"
        and input_audit["runtime_inputs_reference"] == request["runtime_inputs"]
        and input_audit["anchor_binding_reference"] == request["anchor_binding"]
        and input_audit["input_identity_verified"] is True,
        "Require independently qualified common-anchor input assets")
    require(inputs["status"] == "bound_common_anchor_full5000_runtime_inputs"
        and inputs["anchor_binding_sha256"] == request["anchor_binding"]["sha256"]
        and type(inputs["images"]) is int and inputs["images"] == 5000
        and type(inputs["seed"]) is int and inputs["seed"] == 42
        and inputs["sources"] == ["faster_rcnn_r50"] and inputs["targets"] == registry.target_ids()
        and set(inputs["checkpoints"]) == set(registry.target_ids())
        and {"torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmyolo"} <= set(inputs["packages"]),
        "Common input scientific scope or detector/environment coverage changed")
    roles = [("source", "faster_rcnn_r50")] + [("target", t) for t in registry.target_ids()]
    require(isinstance(inputs["model_configs"], list)
        and all(isinstance(row, dict) for row in inputs["model_configs"])
        and [(row.get("role"), row.get("model")) for row in inputs["model_configs"]] == roles,
        "Common inputs require the ordered source and sixteen target config roles")
    definition = deepcopy(registry.protocols[PROTOCOL])
    require(definition["execution_entrypoint"] == ENTRYPOINT and definition["device_counts"] == [1, 2, 4]
        and definition["execution_lifecycle"] == "full_stage_keep_all_v1"
        and definition["isolated_candidate"]["execution_alias"] == ALIAS,
        "Registered common execution interface differs")
    snapshot = runtime_snapshot(registry.root)
    verify_process_source(registry, snapshot)
    for proof in (reuse_audit, input_audit):
        require(proof["runtime_snapshot_sha256"] == canonical_hash(snapshot),
            "Qualification does not bind the current execution source snapshot")
    selected = [g for g in design["groups"] if g["group_id"] in design["new_group_ids"]]
    plan = dict(schema_version=1, record_type=EXECUTION_PROTOCOL, protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, status="prepared_common_anchor_execution_pending_readiness",
        preparation_request=deepcopy(request), definition=definition,
        definition_sha256=canonical_hash(definition), prepared_plan=deepcopy(binding["common_catalogue"]),
        execution_design=design, groups=deepcopy(design["groups"]), sources=["faster_rcnn_r50"],
        targets=list(registry.target_ids()), new_group_ids=design["new_group_ids"],
        reused_group_ids=design["reused_group_ids"], max_images=max_images,
        diagnostic_only=max_images is not None, new_groups=len(selected),
        new_images=len(selected) * (5000 if max_images is None else max_images),
        new_evaluations=len(selected) * len(registry.target_ids()),
        runtime_sha256=snapshot, runtime_snapshot_sha256=canonical_hash(snapshot),
        annotation_sha256=inputs["annotation"]["sha256"], image_ids_sha256=inputs["image_ids_sha256"],
        checkpoint_sha256={t: inputs["checkpoints"][t]["sha256"] for t in registry.target_ids()},
        model_config_bindings=deepcopy(inputs["model_configs"]), packages=deepcopy(inputs["packages"]),
        base_commit=inputs["base_commit"], clean_image_manifest=deepcopy(inputs["clean_image_manifest"]),
        input_binding_sha256=request["runtime_inputs"]["sha256"],
        reuse_assessment_sha256=request["reuse_partition"]["sha256"],
        anchor_binding_sha256=request["anchor_binding"]["sha256"],
        science_sha256=canonical_hash(binding["common_catalogue"]),
        execution_lifecycle="full_stage_keep_all_v1", capacity_contract=deepcopy(request["capacity_contract"]),
        payload_authorization=dict(explicit_owner_confirmation=False, deletion_authorized=False,
            historical_roots_allowed=False, failed_or_partial_groups_allowed=False),
        runner_armed=False, formal_execution_admission=False, independent_confirmation=False,
        scientific_acceptance=False, formal_metrics_eligible=False, automatic_promotion=False)
    plan["keep_all_policy_sha256"] = canonical_hash(keep_all_policy(plan, selected, max_images))
    require(runtime_snapshot(registry.root) == snapshot, "Common execution source changed during preparation")
    return plan


def verify_execution_plan(registry, plan):
    require(isinstance(plan, dict) and plan.get("record_type") == EXECUTION_PROTOCOL,
        "Not a common-anchor execution plan")
    expected = compile_runtime_plan(registry, plan["preparation_request"], max_images=plan["max_images"])
    require(canonical_hash(plan) == canonical_hash(expected), "Common runtime plan or qualification changed")
    return plan


def execution_groups(plan, max_images):
    require(plan.get("record_type") == EXECUTION_PROTOCOL
        and canonical_hash(plan["max_images"]) == canonical_hash(max_images), "Common execution scope changed")
    design = plan["execution_design"]
    require(canonical_hash(plan["groups"]) == canonical_hash(design["groups"])
        and plan["new_group_ids"] == design["new_group_ids"]
        and plan["reused_group_ids"] == design["reused_group_ids"], "Common job partition differs")
    return deepcopy([g for g in plan["groups"] if g["group_id"] in plan["new_group_ids"]])


def scheduled_stages(plan, selected):
    require(canonical_hash(selected) == canonical_hash(execution_groups(plan, plan["max_images"])),
        "Common scheduling requires the complete new partition")
    return deepcopy(plan["execution_design"]["execution_stages"])


def execution_metadata(plan, plan_sha256, group, max_images):
    require(any(canonical_hash(group) == canonical_hash(g) for g in execution_groups(plan, max_images)),
        "Foreign or reused common group cannot be generated")
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL,
        stage="common_anchor_exploration", group_id=group["group_id"], variant=group["variant"],
        seed=group["seed"], parameters_sha256=group["parameters_sha256"],
        execution_plan_sha256=plan_sha256, runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_binding_sha256=plan["input_binding_sha256"], anchor_binding_sha256=plan["anchor_binding_sha256"],
        reuse_assessment_sha256=plan["reuse_assessment_sha256"], science_sha256=plan["science_sha256"],
        max_images=max_images, diagnostic_only=max_images is not None,
        formal_metrics_eligible=False, independent_confirmation=False, automatic_promotion=False)


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    """Only an external, source-bound readiness audit can admit this exact plan."""
    verify_execution_plan(registry, plan)
    verify_environment(plan)
    receipt = read_bound(path, digest)
    require(receipt["status"] == "independently_verified_common_anchor_execution_readiness"
        and receipt["execution_plan_sha256"] == plan_hash
        and receipt["runtime_snapshot_sha256"] == plan["runtime_snapshot_sha256"]
        and canonical_hash(receipt["max_images"]) == canonical_hash(max_images)
        and receipt["new_group_ids"] == plan["new_group_ids"]
        and receipt["errors"] == [] and receipt["scientific_acceptance"] is False,
        "Common readiness does not cover this exact execution")
    checks = {"current_source_and_inputs", "common_control_and_budget_validation",
        "complete_job_ownership", "failure_containment", "keep_all_lifecycle"}
    require(set(receipt["checks"]) == checks and all(v is True for v in receipt["checks"].values()),
        "Missing independent common execution checks")
    require(receipt["real_model_smoke_verified"] is True if max_images is None else
        receipt["diagnostic_only"] is True, "Full execution requires verified real-model readiness")
    verify_device_admission(receipt, plan["execution_design"]["requested_devices"])
    return receipt
