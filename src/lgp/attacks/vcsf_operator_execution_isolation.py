"""Source-bound operator admission and the unchanged pure operator constructor."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from ..runners import vcsf_operator_execution_contract as contract
from ..runners.vcsf_research_plan import canonical_hash, require
from .factory import ATTACK_TYPES
from .vcsf_operator_factorial_isolated import VCSFOperatorFactorialCandidate, VCSFOperatorFactorialConfig


def build_operator_execution(adapter, parameters):
    return VCSFOperatorFactorialCandidate(adapter, VCSFOperatorFactorialConfig.from_mapping(dict(parameters)))


def resolve_operator_execution(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None):
    """Fail before model construction unless the complete admitted invocation matches."""
    from ..runners.vcsf_efficacy_runner import keep_all_policy, read_worker_lifecycle

    fields = {"candidate_id", "execution_alias", "mode", "execution_plan",
        "execution_plan_sha256", "admission", "admission_sha256", "execution_root",
        "run_binding_sha256", "max_images"}
    require(isinstance(descriptor, dict) and set(descriptor) == fields
        and descriptor["mode"] == contract.MODE
        and descriptor["candidate_id"] == contract.CANDIDATE_ID
        and descriptor["execution_alias"] == attack_id == contract.ALIAS,
        "Unsupported operator execution descriptor")
    require(contract.ALIAS not in registry.attacks and contract.ALIAS not in ATTACK_TYPES,
        "Operator execution must remain a process-local overlay")
    definition = registry.protocols[contract.PROTOCOL]
    require(dataset_id == definition["dataset"] and split == definition["split"]
        and source_id in definition["sources"] and type(seed) is int and seed == definition["seed"]
        and image_ids is None and seed_offsets is None and input_transform is None,
        "Operator execution requires canonical full inputs and selected-position seeds")
    require(canonical_hash(descriptor["max_images"]) == canonical_hash(max_images)
        and budget_profile == definition["budget_profile"],
        "Operator image limit or registered cost budget differs")
    plan = contract.read_bound(contract.child(registry.root, descriptor["execution_plan"]),
        descriptor["execution_plan_sha256"])
    contract.verify_execution_plan(registry, plan)
    contract.verify_admission(registry, plan, descriptor["execution_plan_sha256"],
        contract.child(registry.root, descriptor["admission"]), descriptor["admission_sha256"], max_images)
    scheduled = contract.execution_groups(plan, max_images)
    require(isinstance(run_metadata, dict) and isinstance(parameter_overrides, dict),
        "Operator parameters and metadata must be complete objects")
    matched = [g for g in scheduled if type(run_metadata.get("group_id")) is int
        and g["group_id"] == run_metadata["group_id"] and g["source"] == source_id and g["seed"] == seed]
    require(len(matched) == 1, "Reused or unscheduled operator group cannot be generated")
    group = matched[0]
    metadata = contract.execution_metadata(plan, descriptor["execution_plan_sha256"], group, max_images)
    require(canonical_hash(run_metadata) == canonical_hash(metadata)
        and canonical_hash(parameter_overrides) == group["parameters_sha256"],
        "Operator invocation changed exact group parameters or metadata")
    config = VCSFOperatorFactorialConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Operator parameter normalization changed the admitted method")
    output = contract.child(registry.root, descriptor["execution_root"])
    parent = registry.root / "outputs" / ("experiments" if max_images is None else "diagnostics") / contract.EXECUTION_PROTOCOL
    require(output.parent == parent, "Operator execution root escaped its fresh namespace")
    binding = contract.read_bound(output / "plan_binding.json", descriptor["run_binding_sha256"])
    require(set(binding) == {"execution_root", "execution_plan_sha256", "admission_sha256",
        "max_images", "group_ids", "devices", "stages", "created_at", "payload_authorization",
        "payload_policy_sha256", "execution_lifecycle"}
        and binding["execution_plan_sha256"] == descriptor["execution_plan_sha256"]
        and binding["admission_sha256"] == descriptor["admission_sha256"]
        and canonical_hash(binding["max_images"]) == canonical_hash(max_images)
        and canonical_hash(binding["group_ids"]) == canonical_hash(plan["new_group_ids"])
        and binding["execution_root"] == descriptor["execution_root"]
        and binding["execution_lifecycle"] == plan["execution_lifecycle"]
        and canonical_hash(binding["payload_authorization"]) == canonical_hash(plan["payload_authorization"])
        and canonical_hash(binding["stages"]) == canonical_hash(contract.scheduled_stages(plan, scheduled)),
        "Operator run binding changed its jobs, admission or keep-all lifecycle")
    contract.balanced_lanes(scheduled, plan["sources"], binding["devices"])
    policy = read_worker_lifecycle(plan, scheduled, max_images, output,
        dict(payload_policy_sha256=binding["payload_policy_sha256"]))
    require(canonical_hash(policy) == canonical_hash(keep_all_policy(plan, scheduled, max_images))
        == plan["keep_all_policy_sha256"], "Operator retained-payload policy changed")
    leaf = output / "groups" / "{:06d}".format(group["group_id"]) / "attack"
    require(output_dir is not None and Path(output_dir).absolute() == leaf,
        "Operator payload escaped its exact group leaf")
    contract.child(registry.root, leaf.relative_to(registry.root))
    return AttackSpec(id=contract.ALIAS, display_name="VCSF operator: " + group["variant"],
        executor="native", parameters=dict(group["parameters"]), metadata=dict(metadata,
            candidate_id=contract.CANDIDATE_ID,
            fidelity_status="registered_operator_pending_independent_result_acceptance",
            fidelity_basis="qualified_original_reuse_and_independent_execution_readiness",
            semantic_contract="retrospective_fullval_fixed_width_not_independent_confirmation",
            global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False))
