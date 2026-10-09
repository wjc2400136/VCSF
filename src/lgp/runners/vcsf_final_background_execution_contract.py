"""Exact final-background jobs; historical reuse never admits a new executor."""
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import sys

from ..io import file_digest
from .vcsf_common_anchor_execution_contract import (
    read_reference, runtime_snapshot as common_snapshot, verify_device_admission,
)
from .vcsf_efficacy_contract import child, verify_environment
from .vcsf_final_background import PROTOCOL, prepare_final_background_design
from .vcsf_operator_execution_contract import balanced_lanes, read_bound
from .vcsf_research_plan import canonical_hash, require

EXECUTION_PROTOCOL = "vcsf_final_background_execution"
ENTRYPOINT = "experiments/vcsf_final_background_experiments.py"
ALIAS = "vcsf_final_background_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "final_background_execution"
ALLOW_EMPTY_LANES = False
DISPLAY_NAME = "VCSF final background: "
SCOPE_STATUS = "final_background_execution_scope_only"
FAILURE_STATUS = "final_background_execution_failed"
DESCRIPTION = "Prepare or execute independently admitted final-background complete jobs."
FIDELITY_STATUS = "registered_final_background_pending_result_acceptance"
SEMANTIC_CONTRACT = "working_A10_background_retrospective_not_final_method_or_independent_confirmation"


def runtime_snapshot(root):
    snapshot = common_snapshot(root, entrypoint=ENTRYPOINT)
    snapshot.update({p.relative_to(root).as_posix(): file_digest(p)
        for p in (Path(root) / "experiments").rglob("*.py")})
    for name in ("experiments/vcsf_single_source_ablation_executor_smoke.py",
            "experiments/prepare_vcsf_final_background.py",
            "docs/research/vcsf-final-background-request-20260929.json"):
        snapshot[name] = file_digest(child(root, name))
    return snapshot


def verify_process_source(registry, snapshot):
    from .vcsf_ablation_execution_contract import _verify_process_binding

    _verify_process_binding(registry)
    entry = sys.modules.get("__main__")
    require(getattr(entry, "__file__", None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == snapshot[ENTRYPOINT],
        "Use the source-bound final-background execution entry")
    bootstrap = "experiments/vcsf_single_source_ablation_executor_smoke.py"
    require(sha256(getattr(entry, "_bootstrap_raw", b"")).hexdigest() == snapshot[bootstrap],
        "Executed source bootstrap differs from its snapshot")
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            continue
        path = Path(filename).absolute()
        if registry.root / "tools" not in path.parents:
            continue
        loader = getattr(module, "__loader__", None)
        require(path == path.resolve() and getattr(loader, "source_only", None) is True
            and getattr(loader, "source_sha256", None) == snapshot.get(path.relative_to(registry.root).as_posix()),
            "Executed project tool differs from its source snapshot")


def _source_map(root, mapping):
    require(isinstance(mapping, dict) and mapping, "Missing original source identity")
    for name, digest in mapping.items():
        require(file_digest(child(root, name)) == digest, "Bound source changed: " + name)


def prepared_design(registry, reference):
    path = Path(reference["file"])
    require(path.is_absolute() and path.name == "completion.json"
        and path.parents[1].name == "vcsf_final_background"
        and path.parents[2].name == "plans" and path.parents[3].name == "outputs",
        "Require an original full-scope final-background preparation")
    publication = read_reference(reference)
    require(publication.get("record_type") == "vcsf_final_background_preparation_receipt"
        and publication.get("status") == "prepared_pending_input_reuse_runtime_and_independent_admission"
        and publication.get("max_images") is None and publication.get("diagnostic_only") is False
        and publication.get("runner_armed") is False and publication.get("reuse_accepted") is False
        and publication.get("formal_execution_admission") is False
        and publication.get("formal_result_cells_filled") == 0,
        "A diagnostic or armed publication cannot define the full scientific scope")
    _source_map(path.parents[4], publication["source_sha256"])
    require(set(publication["artifacts_sha256"]) == {"design.json"},
        "Incomplete final-background preparation publication")
    design = read_bound(path.parent / "design.json", publication["artifacts_sha256"]["design.json"])
    expected = prepare_final_background_design(registry, publication["requested_devices"])
    require(canonical_hash(design) == canonical_hash(expected)
        and publication["request_sha256"] == expected["request_sha256"],
        "Original preparation differs from the registered scientific design")
    return design


def comparison_reuse(registry, reference, preparation, design):
    path = Path(reference["file"])
    require(path.is_absolute() and path.name == "receipt.json"
        and path.parents[1].name == "vcsf_final_background_reuse"
        and path.parents[2].name == "audits" and path.parents[3].name == "outputs",
        "Require the original scoped A10 comparison-reuse receipt")
    proof = read_reference(reference)
    require(proof.get("status") == "qualified_original_A10_for_final_background_comparison"
        and proof.get("protocol") == PROTOCOL and proof.get("preparation_reference") == preparation
        and proof.get("selection_request_sha256") == design["request_sha256"]
        and proof.get("candidate_parameters_sha256") == design["candidate_parameters_sha256"]
        and proof.get("original_group_id") == 10 and proof.get("final_background_group_id") == 1
        and proof.get("reuse_qualified_for_final_background") is True
        and proof.get("reuse_partition_resolved") is True
        and proof.get("reused_group_ids") == [1]
        and proof.get("new_group_ids") == list(range(2, 11))
        and proof.get("new_image_instances") == 45000 and proof.get("new_target_evaluations") == 144,
        "A10 reuse identity or fixed new/reused partition differs")
    for flag in ("new_executor_qualified", "formal_execution_admission", "formal_metrics_eligible",
            "scientific_acceptance", "final_method_promoted", "independent_confirmation",
            "physical_cost_inheritance", "trajectory_identity_claim"):
        require(proof.get(flag) is False, "Historical reuse cannot confer new execution or scientific status")
    require(proof["runtime_snapshot_sha256"] == canonical_hash(proof["runtime_sha256"]),
        "Invalid original reuse runtime identity")
    _source_map(path.parents[4], proof["runtime_sha256"])
    cells = proof["original_cells"]
    require([cell["target"] for cell in cells] == registry.target_ids()
        and all(cell["group_id"] == 10 and cell["variant"] == "A10"
            and cell["source"] == "faster_rcnn_r50" and cell["seed"] == 42
            and cell["images"] == 5000 and cell["max_images"] is None
            and cell["diagnostic_only"] is False
            and cell["parameters_sha256"] == design["candidate_parameters_sha256"] for cell in cells),
        "Incomplete or foreign reused target panel")
    current = proof["current_inputs"]
    inputs = read_reference(current["original_runtime_inputs"])
    roles = [("source", "faster_rcnn_r50")] + [("target", target) for target in registry.target_ids()]
    require(current.get("current_input_identity_verified") is True
        and inputs["status"] == "bound_final_attribution_full5000_runtime_inputs"
        and inputs["binding_sha256"] == canonical_hash({k: v for k, v in inputs.items() if k != "binding_sha256"})
        and type(inputs["images"]) is int and inputs["images"] == inputs["full_split_images"] == 5000
        and inputs["dataset"] == "coco" and inputs["split"] == "val"
        and type(inputs["seed"]) is int and inputs["seed"] == 42 and inputs["seed_schedule"] == "selected_position"
        and inputs["sources"] == ["faster_rcnn_r50"] and inputs["targets"] == registry.target_ids()
        and set(inputs["checkpoints"]) == set(registry.target_ids())
        and [(row["role"], row["model"]) for row in inputs["model_configs"]] == roles
        and all(row["checkpoint_sha256"] == inputs["checkpoints"][row["model"]]["sha256"]
            for row in inputs["model_configs"])
        and {"torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmyolo"} <= set(inputs["packages"])
        and inputs["base_commit"] == current["original_base_commit"]
        and inputs["packages"] == current["packages"]
        and inputs["annotation"] == current["annotation"]
        and inputs["image_ids_sha256"] == current["image_ids_sha256"]
        and inputs["seed_mapping_sha256"] == current["seed_mapping_sha256"]
        and {key: value["sha256"] for key, value in inputs["checkpoints"].items()}
            == current["checkpoint_sha256"], "Historical input bindings disagree")
    ids = inputs["ordered_image_ids"]
    require(len(ids) == 5000 and all(type(value) is int for value in ids)
        and ids == sorted(set(ids)) and canonical_hash(ids) == inputs["image_ids_sha256"]
        and canonical_hash([[value, position, 42 + position] for position, value in enumerate(ids)])
            == inputs["seed_mapping_sha256"], "Full image order or selected-position seed mapping differs")
    return proof, inputs


def compile_runtime_plan(registry, request, *, max_images=None):
    from .attack import _git_commit
    from ..paths import project_root
    from .vcsf_common_anchor_catalogue import _resolved
    from .vcsf_efficacy_runner import keep_all_policy

    require(isinstance(request, dict) and set(request) == {
        "preparation", "reuse_audit", "devices", "capacity_contract"},
        "Require the exact preparation, original reuse, device and capacity request")
    snapshot = runtime_snapshot(registry.root)
    require(project_root() == registry.root, "Execution and registry roots differ")
    verify_process_source(registry, snapshot)
    design = prepare_final_background_design(registry, request["devices"], max_images)
    original = prepared_design(registry, request["preparation"])
    reuse, inputs = comparison_reuse(registry, request["reuse_audit"], request["preparation"], original)
    groups = deepcopy(prepare_final_background_design(registry, request["devices"])["groups"])
    definition = deepcopy(registry.protocols[PROTOCOL])
    require(definition["execution_entrypoint"] == ENTRYPOINT
        and definition["execution_protocol"] == EXECUTION_PROTOCOL
        and definition["execution_requires_independent_admission"] is True
        and definition["isolated_candidate"]["execution_alias"] == ALIAS,
        "Execution interface differs from the registry")
    for group in groups:
        group.update(_resolved(group["parameters"], registry.budget_profiles[definition["budget_profile"]]))
        group["parameters"]["feature_levels"] = list(group["parameters"]["feature_levels"])
        group.update(max_images=max_images, diagnostic_only=max_images is not None,
            potential_reuse=group["group_id"] == 1 and max_images is None,
            reuse_accepted=group["group_id"] == 1 and max_images is None)
    new = design["scheduled_group_ids"]
    reused = [1] if max_images is None else []
    require(new == (list(range(2, 11)) if max_images is None else list(range(1, 11))),
        "The complete formal/diagnostic job set changed")
    selected = [group for group in groups if group["group_id"] in new]
    lanes = balanced_lanes(selected, definition["sources"], request["devices"])
    require([[group["group_id"] for group in lane] for lane in lanes] == design["prospective_lanes"],
        "Complete-job lane assignment differs from preparation")
    require(definition["payload_retention"] == dict(mode="keep_all", cleanup_authorized=False),
        "The registered payload preservation scope changed")
    definition["payload_retention"].update(deletion_authorized=False, prune_after_seal=False,
        preserve_failed_and_partial=True)
    require(definition["prediction_archive"] == dict(format="gzip", retention="keep_all"),
        "Prediction archive format or preservation scope changed")
    definition["prediction_archive"] = definition["prediction_archive"]["format"]
    commit = _git_commit()
    execution = dict(groups=deepcopy(groups), new_group_ids=new, reused_group_ids=reused,
        requested_devices=list(request["devices"]), prospective_lanes=design["prospective_lanes"],
        execution_stages=[dict(name="final_background", group_ids=new)])
    plan = dict(schema_version=1, record_type=EXECUTION_PROTOCOL, protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, status="prepared_final_background_execution_pending_readiness",
        preparation_request=deepcopy(request), definition=definition,
        definition_sha256=canonical_hash(definition), prepared_plan=dict(groups=deepcopy(original["groups"])),
        execution_design=execution, groups=groups, sources=definition["sources"], targets=registry.target_ids(),
        new_group_ids=new, reused_group_ids=reused, max_images=max_images,
        diagnostic_only=max_images is not None, new_groups=len(new),
        new_images=len(new) * (5000 if max_images is None else max_images),
        new_evaluations=len(new) * len(registry.target_ids()), runtime_sha256=snapshot,
        runtime_snapshot_sha256=canonical_hash(snapshot), annotation_sha256=inputs["annotation"]["sha256"],
        image_ids_sha256=inputs["image_ids_sha256"], seed_mapping_sha256=inputs["seed_mapping_sha256"],
        checkpoint_sha256={key: row["sha256"] for key, row in inputs["checkpoints"].items()},
        model_config_bindings=deepcopy(inputs["model_configs"]), packages=deepcopy(inputs["packages"]),
        base_commit=commit, original_base_commit=inputs["base_commit"],
        checkout_kind="source_copy_without_git" if commit == "unavailable" else "git_checkout",
        original_reuse_runtime_sha256=reuse["runtime_snapshot_sha256"],
        new_executor_qualification_inherited=False, clean_image_manifest=deepcopy(inputs["clean_image_manifest"]),
        runtime_inputs_reference=deepcopy(reuse["current_inputs"]["original_runtime_inputs"]),
        input_binding_sha256=reuse["current_inputs"]["original_runtime_inputs"]["sha256"],
        reuse_assessment_sha256=request["reuse_audit"]["sha256"], science_sha256=design["request_sha256"],
        formal_result_slots=deepcopy(design["formal_result_slots"]),
        diagnostic_result_slots=deepcopy(design["diagnostic_result_slots"]),
        execution_lifecycle="full_stage_keep_all_v1", capacity_contract=deepcopy(request["capacity_contract"]),
        payload_authorization=dict(explicit_owner_confirmation=False, deletion_authorized=False,
            historical_roots_allowed=False, failed_or_partial_groups_allowed=False),
        runner_armed=False, formal_execution_admission=False, independent_confirmation=False,
        scientific_acceptance=False, formal_metrics_eligible=False, final_method_promoted=False,
        automatic_promotion=False, current_executor_inputs_verified=False)
    plan["keep_all_policy_sha256"] = canonical_hash(keep_all_policy(plan, selected, max_images))
    require(runtime_snapshot(registry.root) == snapshot, "Runtime changed while preparing the plan")
    return plan


def verify_execution_plan(registry, plan):
    require(plan.get("record_type") == EXECUTION_PROTOCOL, "Not a final-background execution plan")
    expected = compile_runtime_plan(registry, plan["preparation_request"], max_images=plan["max_images"])
    require(canonical_hash(expected) == canonical_hash(plan), "Execution plan or bound evidence changed")
    return plan


def execution_groups(plan, max_images):
    require(plan.get("record_type") == EXECUTION_PROTOCOL
        and canonical_hash(plan["max_images"]) == canonical_hash(max_images), "Execution scope differs")
    design = plan["execution_design"]
    require(canonical_hash(plan["groups"]) == canonical_hash(design["groups"])
        and plan["new_group_ids"] == design["new_group_ids"]
        and plan["reused_group_ids"] == design["reused_group_ids"], "Execution partition differs")
    return deepcopy([group for group in plan["groups"] if group["group_id"] in plan["new_group_ids"]])


def scheduled_stages(plan, selected):
    require(canonical_hash(selected) == canonical_hash(execution_groups(plan, plan["max_images"])),
        "Scheduling requires every new complete job exactly once")
    return deepcopy(plan["execution_design"]["execution_stages"])


def execution_metadata(plan, plan_sha256, group, max_images):
    require(any(canonical_hash(group) == canonical_hash(row) for row in execution_groups(plan, max_images)),
        "A foreign or reused group cannot be generated")
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL, stage="final_background",
        group_id=group["group_id"], variant=group["variant"], seed=group["seed"],
        parameters_sha256=group["parameters_sha256"], execution_plan_sha256=plan_sha256,
        runtime_snapshot_sha256=plan["runtime_snapshot_sha256"], input_binding_sha256=plan["input_binding_sha256"],
        reuse_assessment_sha256=plan["reuse_assessment_sha256"], science_sha256=plan["science_sha256"],
        max_images=max_images, diagnostic_only=max_images is not None, formal_metrics_eligible=False,
        independent_confirmation=False, automatic_promotion=False, final_method_promoted=False)


def qualified_reference(reference, namespace, status, snapshot):
    path = Path(reference["file"])
    require(path.is_absolute() and path.name == "receipt.json" and path.parents[1].name == namespace
        and path.parents[2].name == "audits" and path.parents[3].name == "outputs",
        "Qualification must be an original receipt in its own namespace")
    proof = read_reference(reference)
    require(proof.get("status") == status and proof.get("errors") == []
        and proof.get("runtime_snapshot_sha256") == canonical_hash(snapshot),
        "Qualification does not bind this exact execution runtime")
    _source_map(path.parents[4], proof["auditor_source_sha256"])
    return proof


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    verify_execution_plan(registry, plan)
    require(canonical_hash(plan["max_images"]) == canonical_hash(max_images), "Admission image scope differs")
    verify_environment(plan)
    proof = qualified_reference(dict(file=str(path), sha256=digest),
        "vcsf_final_background_readiness", "independently_verified_final_background_execution_readiness",
        plan["runtime_sha256"])
    require(proof["execution_plan_sha256"] == plan_hash
        and canonical_hash(proof["max_images"]) == canonical_hash(max_images)
        and proof["new_group_ids"] == plan["new_group_ids"]
        and proof["reused_group_ids"] == plan["reused_group_ids"]
        and proof["current_executor_inputs_verified"] is True
        and proof["base_commit"] == plan["base_commit"]
        and proof["checkout_kind"] == plan["checkout_kind"]
        and proof["reuse_assessment_sha256"] == plan["reuse_assessment_sha256"]
        and proof["input_binding_sha256"] == plan["input_binding_sha256"]
        and proof["science_sha256"] == plan["science_sha256"]
        and proof["scientific_acceptance"] is False and proof["final_method_promoted"] is False,
        "Readiness belongs to another execution or claims scientific promotion")
    checks = {"current_source_and_inputs", "final_background_controls_and_budget_validation",
        "complete_job_ownership", "failure_containment", "keep_all_lifecycle"}
    require(set(proof["checks"]) == checks and all(value is True for value in proof["checks"].values()),
        "Missing independent runtime readiness checks")
    require(proof["diagnostic_only"] is (max_images is not None), "Readiness diagnostic label differs")
    require(proof["formal_execution_admission"] is (max_images is None),
        "Readiness formal execution permission differs from its scope")
    if max_images is None:
        require(proof["real_model_smoke_verified"] is True
            and len(proof["accepted_executor_smokes"]) == 2, "Full execution needs both actual GPU modes")
        counts = []
        for reference in proof["accepted_executor_smokes"]:
            smoke = qualified_reference(reference, "vcsf_final_background_saved_result",
                "independently_verified_final_background_diagnostic_results", plan["runtime_sha256"])
            require(type(smoke["max_images"]) is int and smoke["max_images"] == 1
                and type(smoke["device_count"]) is int and smoke["diagnostic_only"] is True
                and smoke["group_ids"] == list(range(1, 11)) and smoke["target_cells"] == 160
                and smoke["targets"] == plan["targets"] and smoke["failed_records"] == 0
                and smoke["science_sha256"] == plan["science_sha256"]
                and smoke["input_binding_sha256"] == plan["input_binding_sha256"]
                and smoke["formal_metrics_eligible"] is False
                and smoke["scientific_acceptance"] is False, "Incomplete or foreign original GPU diagnostic")
            counts.append(smoke["device_count"])
        require(sorted(counts) == [1, 2], "Both distinct one/two-device modes must be accepted")
    else:
        require(proof["real_model_smoke_verified"] is False and proof["accepted_executor_smokes"] == []
            and proof["formal_execution_admission"] is False,
            "A diagnostic admission cannot inherit or assert unperformed GPU qualification")
    verify_device_admission(proof, plan["execution_design"]["requested_devices"])
    return proof
