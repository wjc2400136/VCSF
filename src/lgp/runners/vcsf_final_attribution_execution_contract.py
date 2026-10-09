"""Finite final-attribution runtime profile; qualification never follows from planning."""
from copy import deepcopy
from pathlib import Path
import sys

from ..io import file_digest
from .vcsf_common_anchor_execution_contract import (
    read_reference, runtime_snapshot as common_snapshot, verify_device_admission,
)
from .vcsf_efficacy_contract import child, verify_environment
from .vcsf_operator_execution_contract import read_bound, balanced_lanes
from .vcsf_research_plan import canonical_hash, require

PROTOCOL = "vcsf_final_attribution_research"
EXECUTION_PROTOCOL = "vcsf_final_attribution_execution"
ENTRYPOINT = "experiments/vcsf_final_attribution_experiments.py"
ALIAS = "vcsf_final_attribution_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "final_attribution_execution"
ALLOW_EMPTY_LANES = False
DISPLAY_NAME = "VCSF final attribution: "
FIDELITY_STATUS = "registered_final_attribution_pending_result_acceptance"
SEMANTIC_CONTRACT = "final_background_after_fullval_selection_not_independent_confirmation"


def runtime_snapshot(root):
    snapshot = common_snapshot(root, entrypoint=ENTRYPOINT)
    entry = Path(root) / 'experiments/prepare_vcsf_final_attribution_inputs.py'
    if entry.is_file():
        snapshot[entry.relative_to(root).as_posix()] = file_digest(entry)
    return snapshot


def verify_process_source(registry, snapshot):
    from .vcsf_ablation_execution_contract import _verify_process_binding

    _verify_process_binding(registry)
    entry = sys.modules.get("__main__")
    require(getattr(entry, "__file__", None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == snapshot[ENTRYPOINT],
        "Use the source-bound final-attribution execution entry")


def qualified_reference(reference, namespace, status, snapshot):
    path = Path(reference["file"])
    require(path.name == "receipt.json" and path.parents[1].name == namespace
        and path.parents[2].name == "audits" and path.parents[3].name == "outputs",
        "Final qualification receipt namespace differs")
    proof = read_reference(reference)
    require(proof.get("status") == status and proof.get("errors") == []
        and proof.get("runtime_snapshot_sha256") == canonical_hash(snapshot),
        "Final qualification does not bind this execution runtime")
    sources = proof.get("auditor_source_sha256")
    require(isinstance(sources, dict) and sources, "Missing qualification source binding")
    auditor_root = Path(proof.get("auditor_source_root", path.parents[4]))
    require(auditor_root.is_absolute(), "Qualification auditor root must be absolute")
    for name, digest in sources.items():
        require(file_digest(child(auditor_root, name)) == digest,
            "Qualification auditor source changed")
    return proof


def prepared_groups(registry, reference, devices):
    from .vcsf_common_anchor_catalogue import _resolved
    from .vcsf_ablation_runtime_inputs import _read_json

    path = Path(reference["file"])
    require(path.name == "completion.json" and path.parents[1].name == "vcsf_final_attribution"
        and path.parents[2].name == "plans" and path.parents[3].name == "outputs",
        "Require an original final preparation completion")
    publication = read_reference(reference)
    require(publication.get("status") == "prepared_final_attribution_pending_input_reuse_and_admission"
        and publication.get("reuse_accepted") is False and publication.get("runner_armed") is False
        and publication.get("requested_devices") == list(devices), "Final preparation scope changed")
    require(isinstance(publication.get("source_sha256"), dict) and publication["source_sha256"],
        "Final preparation must bind its source implementation")
    for name, digest in publication["source_sha256"].items():
        require(file_digest(child(path.parents[4], name)) == digest, "Preparation source changed")
    files = publication["artifacts_sha256"]
    require(set(files) == {"resolved.json", "factors.json", "ancestry.json", "jobs.json"},
        "Final execution needs complete parameter, factor, ancestry and job preparation")
    artifacts = {name: _read_json(dict(file=str(path.parent / name),
        bytes=(path.parent / name).stat().st_size, sha256=digest)) for name, digest in files.items()}
    definition = registry.protocols[PROTOCOL]
    request = read_bound(child(registry.root, definition["attribution_request"]),
        definition["attribution_request_sha256"])
    require(publication["request_sha256"] == definition["attribution_request_sha256"],
        "Preparation differs from the registered finite request")
    resolved = artifacts["resolved.json"]
    jobs = artifacts["jobs.json"]
    factors = artifacts["factors.json"]
    require(factors.get("status") == "independently_checked_final_parameter_factors_only"
        and factors["request_content_sha256"] == canonical_hash(request)
        and factors["resolved_content_sha256"] == canonical_hash(resolved)
        and factors["configurations"] == request["declared_count"]
        and factors["factorial_sizes"] == dict(core=8, objective=4, optimizer=4),
        "Final factor evidence differs from the finite parameter set")
    require(resolved["request_content_sha256"] == canonical_hash(request)
        and jobs["resolved_content_sha256"] == canonical_hash(resolved)
        and jobs["request_content_sha256"] == canonical_hash(request)
        and jobs["requested_devices"] == list(devices)
        and jobs["logical_configurations"] == request["declared_count"] <= definition["configuration_cap"],
        "Prepared request, parameters or devices differ")
    groups = deepcopy(jobs["groups"])
    require([g["attribution_id"] for g in groups] == [r["id"] for r in resolved["groups"]]
        == [r["id"] for r in request["configurations"]]
        and [g["group_id"] for g in groups] == list(range(1, len(groups) + 1)),
        "Final configuration identity/order changed")
    for group, row in zip(groups, resolved["groups"]):
        require(group["parameters_sha256"] == row["parameters_sha256"]
            == canonical_hash(group["parameters"]) and group["source"] == "faster_rcnn_r50"
            and group["seed"] == 42 and group["images"] == 5000
            and group["claims"] == row["claims"]
            and group["potential_reuse_group_ids"] == row["potential_reuse_group_ids"]
            and group["targets"] == registry.target_ids(), "Final group scope changed")
        group.update(_resolved(group["parameters"], registry.budget_profiles[definition["budget_profile"]]))
        group["variant"] = group["attribution_id"]
    ancestry = artifacts['ancestry.json']
    expected_ancestry = [(g['attribution_id'], origin, g) for g in groups
        for origin in g['potential_reuse_group_ids']]
    require(isinstance(ancestry, list) and len(ancestry) == len(expected_ancestry),
        'Incomplete original result ancestry')
    for evidence, (identity, origin, group) in zip(ancestry, expected_ancestry):
        cells = evidence['original_cells']
        require(evidence['final_id'] == identity and evidence['original_group_id'] == origin
            and evidence['reuse_accepted'] is False
            and [c['target'] for c in cells] == registry.target_ids()
            and all(c['group_id'] == origin and c['parameters_sha256'] == group['parameters_sha256']
                and c['images'] == 5000 and c['seed'] == 42 and c['source'] == 'faster_rcnn_r50'
                and c['max_images'] is None and c['diagnostic_only'] is False for c in cells),
            'Original result ancestry changed its identity or full target panel')
    return groups, jobs, publication


def compile_runtime_plan(registry, request, *, max_images=None):
    from .vcsf_efficacy_runner import keep_all_policy

    require(isinstance(request, dict) and set(request) == {"preparation", "reuse_audit",
        "runtime_inputs", "input_audit", "devices", "capacity_contract"},
        "Require the complete final runtime preparation request")
    require(max_images is None or type(max_images) is int and max_images == 1,
        "Only full5000 or explicit max1 diagnostics are supported")
    groups, jobs, publication = prepared_groups(registry, request["preparation"], request["devices"])
    snapshot = runtime_snapshot(registry.root)
    verify_process_source(registry, snapshot)
    reuse = qualified_reference(request["reuse_audit"], "vcsf_final_attribution_reuse",
        "independently_verified_final_attribution_reuse", snapshot)
    audit = qualified_reference(request["input_audit"], "vcsf_final_attribution_inputs",
        "independently_verified_final_attribution_inputs", snapshot)
    for proof in (reuse, audit):
        require(proof["preparation_reference"] == request["preparation"]
            and proof["runtime_inputs_reference"] == request["runtime_inputs"],
            "Qualification refers to another final preparation or input set")
    require(reuse.get("reuse_accepted") is True and audit.get("input_identity_verified") is True,
        "Final reuse or input qualification is not accepted")
    require([r["group_id"] for r in reuse["groups"]] == [g["group_id"] for g in groups],
        "Reuse qualification must cover every ordered final configuration")
    new, reused = [], []
    for group, row in zip(groups, reuse["groups"]):
        require(row["parameters_sha256"] == group["parameters_sha256"], "Reuse parameters changed")
        possible = group["potential_reuse_group_ids"]
        if possible:
            require(len(possible) == 1 and row["state"] == "qualified_reuse"
                and row["original_group_id"] == possible[0]
                and row.get("complete_input_and_implementation_identity_verified") is True
                and row.get("full_panel_accepted") is True, "Potential reuse remains unresolved")
            reused.append(group["group_id"])
        else:
            require(row["state"] == "requires_new" and row.get("original_group_id") is None,
                "Unmatched final group cannot inherit a historical result")
            new.append(group["group_id"])
    require(new == jobs["unmatched_group_ids"] and reused == jobs["unresolved_reuse_group_ids"],
        "Final partition changed; do not silently expand or reschedule")
    inputs = read_reference(request["runtime_inputs"])
    require(inputs["status"] == "bound_final_attribution_full5000_runtime_inputs"
        and inputs["preparation_reference"] == request["preparation"]
        and inputs["images"] == 5000 and inputs["seed"] == 42
        and inputs["sources"] == ["faster_rcnn_r50"] and inputs["targets"] == registry.target_ids()
        and set(inputs["checkpoints"]) == set(registry.target_ids())
        and {"torch", "torchvision", "mmcv", "mmengine", "mmdet", "mmyolo"} <= set(inputs["packages"]),
        "Final runtime inputs changed the scientific scope")
    roles = [("source", "faster_rcnn_r50")] + [("target", t) for t in registry.target_ids()]
    require([(r["role"], r["model"]) for r in inputs["model_configs"]] == roles,
        "Final inputs omit canonical source/target configuration roles")
    selected = [g for g in groups if g["group_id"] in new]
    lanes = balanced_lanes(selected, ["faster_rcnn_r50"], request["devices"])
    require([[g["group_id"] for g in lane] for lane in lanes] == jobs["prospective_lanes"],
        "Qualified execution changes the published complete-job lanes")
    definition = deepcopy(registry.protocols[PROTOCOL])
    design = dict(groups=groups, new_group_ids=new, reused_group_ids=reused,
        requested_devices=list(request["devices"]),
        execution_stages=[dict(name="final_attribution", group_ids=new)])
    plan = dict(schema_version=1, record_type=EXECUTION_PROTOCOL, protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, status="prepared_final_attribution_execution_pending_readiness",
        preparation_request=deepcopy(request), definition=definition,
        definition_sha256=canonical_hash(definition), prepared_plan=dict(groups=deepcopy(groups)),
        execution_design=design, groups=groups, sources=["faster_rcnn_r50"], targets=registry.target_ids(),
        new_group_ids=new, reused_group_ids=reused, max_images=max_images,
        diagnostic_only=max_images is not None, new_groups=len(new),
        new_images=len(new) * (5000 if max_images is None else max_images),
        new_evaluations=len(new) * len(registry.target_ids()), runtime_sha256=snapshot,
        runtime_snapshot_sha256=canonical_hash(snapshot), annotation_sha256=inputs["annotation"]["sha256"],
        image_ids_sha256=inputs["image_ids_sha256"],
        checkpoint_sha256={t: inputs["checkpoints"][t]["sha256"] for t in registry.target_ids()},
        model_config_bindings=deepcopy(inputs["model_configs"]), packages=deepcopy(inputs["packages"]),
        base_commit=inputs["base_commit"], clean_image_manifest=deepcopy(inputs["clean_image_manifest"]),
        input_binding_sha256=request["runtime_inputs"]["sha256"],
        reuse_assessment_sha256=request["reuse_audit"]["sha256"],
        science_sha256=publication["request_sha256"],
        execution_lifecycle="full_stage_keep_all_v1", capacity_contract=deepcopy(request["capacity_contract"]),
        payload_authorization=dict(explicit_owner_confirmation=False, deletion_authorized=False,
            historical_roots_allowed=False, failed_or_partial_groups_allowed=False),
        runner_armed=False, formal_execution_admission=False, independent_confirmation=False,
        scientific_acceptance=False, formal_metrics_eligible=False, automatic_promotion=False)
    plan["keep_all_policy_sha256"] = canonical_hash(keep_all_policy(plan, selected, max_images))
    require(runtime_snapshot(registry.root) == snapshot, "Final runtime changed during preparation")
    return plan


def verify_execution_plan(registry, plan):
    require(plan.get("record_type") == EXECUTION_PROTOCOL, "Not a final-attribution execution plan")
    expected = compile_runtime_plan(registry, plan["preparation_request"], max_images=plan["max_images"])
    require(canonical_hash(expected) == canonical_hash(plan), "Final runtime plan or qualification changed")
    return plan


def execution_groups(plan, max_images):
    require(plan.get("record_type") == EXECUTION_PROTOCOL
        and canonical_hash(plan["max_images"]) == canonical_hash(max_images), "Final execution scope changed")
    design = plan["execution_design"]
    require(canonical_hash(plan["groups"]) == canonical_hash(design["groups"])
        and plan["new_group_ids"] == design["new_group_ids"]
        and plan["reused_group_ids"] == design["reused_group_ids"], "Final execution partition changed")
    return deepcopy([g for g in plan["groups"] if g["group_id"] in plan["new_group_ids"]])


def scheduled_stages(plan, selected):
    require(canonical_hash(selected) == canonical_hash(execution_groups(plan, plan["max_images"])),
        "Final scheduling requires every new complete job")
    return deepcopy(plan["execution_design"]["execution_stages"])


def execution_metadata(plan, plan_sha256, group, max_images):
    require(any(canonical_hash(group) == canonical_hash(g) for g in execution_groups(plan, max_images)),
        "Foreign or reused final group cannot be generated")
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL, stage="final_attribution",
        group_id=group["group_id"], attribution_id=group["attribution_id"], variant=group["variant"],
        seed=group["seed"], parameters_sha256=group["parameters_sha256"],
        execution_plan_sha256=plan_sha256, runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_binding_sha256=plan["input_binding_sha256"], reuse_assessment_sha256=plan["reuse_assessment_sha256"],
        science_sha256=plan["science_sha256"], max_images=max_images,
        diagnostic_only=max_images is not None, formal_metrics_eligible=False,
        independent_confirmation=False, automatic_promotion=False)


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    verify_execution_plan(registry, plan)
    verify_environment(plan)
    proof = qualified_reference(dict(file=str(path), sha256=digest), "vcsf_final_attribution_readiness",
        "independently_verified_final_attribution_execution_readiness", plan["runtime_sha256"])
    require(proof["execution_plan_sha256"] == plan_hash
        and canonical_hash(proof["max_images"]) == canonical_hash(max_images)
        and proof["new_group_ids"] == plan["new_group_ids"]
        and proof["scientific_acceptance"] is False, "Final readiness belongs to another execution")
    checks = {"current_source_and_inputs", "final_controls_and_budget_validation",
        "complete_job_ownership", "failure_containment", "keep_all_lifecycle"}
    require(set(proof["checks"]) == checks and all(v is True for v in proof["checks"].values()),
        "Missing final runtime readiness checks")
    require(proof.get("real_model_smoke_verified") is True if max_images is None else
        proof.get("diagnostic_only") is True, "Full final execution requires real-model readiness")
    verify_device_admission(proof, plan["execution_design"]["requested_devices"])
    return proof
