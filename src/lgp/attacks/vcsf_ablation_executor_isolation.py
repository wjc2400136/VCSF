"""Process-local full-panel diagnostic bridge for the existing ablation candidate."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from .factory import ATTACK_TYPES
from .vcsf_ablation_isolated import VCSFAblationConfig


def resolve_executor_smoke(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None):
    from ..runners import vcsf_ablation_executor_smoke as contract

    require, digest = contract.require, contract.canonical_hash
    require(descriptor.get("mode") == contract.MODE
        and descriptor.get("candidate_id") == contract.CANDIDATE_ID
        and descriptor.get("execution_alias") == attack_id == contract.ALIAS,
        "Unsupported complete core diagnostic descriptor")
    require(contract.ALIAS not in registry.attacks and contract.ALIAS not in ATTACK_TYPES,
        "Executor diagnostic must remain a process-local overlay")
    require(type(max_images) is int and max_images == 1
        and type(descriptor.get("max_images")) is int and descriptor["max_images"] == 1,
        "Executor smoke cannot admit formal or enlarged generation")
    require(dataset_id == "coco" and split == "val" and source_id == "faster_rcnn_r50"
        and type(seed) is int and seed == 42 and image_ids is None and seed_offsets is None
        and input_transform is None, "Executor smoke changed canonical input, seed or preprocessing")
    plan_hash = descriptor["execution_plan_sha256"]
    plan = contract.read_bound(contract.child(registry.root, descriptor["execution_plan"]), plan_hash)
    contract.verify_execution_plan(registry, plan)
    contract.verify_admission(registry, plan, plan_hash, contract.child(registry.root, descriptor["admission"]),
        descriptor["admission_sha256"], max_images)
    require(budget_profile == registry.protocols[contract.PROTOCOL]["budget_profile"], "Diagnostic budget changed")
    scheduled = contract.execution_groups(plan, max_images)
    matched = [g for g in scheduled if g["variant"] == run_metadata.get("variant")
        and g["source"] == source_id and g["seed"] == seed]
    require(len(matched) == 1, "Unscheduled or ambiguous executor diagnostic group")
    group = matched[0]
    metadata = contract.execution_metadata(plan, plan_hash, group, max_images)
    require(digest(dict(run_metadata)) == digest(metadata)
        and digest(dict(parameter_overrides)) == group["parameters_sha256"]
        and digest(group["parameters"]) == group["parameters_sha256"], "Diagnostic invocation changed its exact contract")
    config = VCSFAblationConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(digest(asdict(config)) == group["parameters_sha256"], "Diagnostic configuration normalization changed")
    output = contract.child(registry.root, descriptor["execution_root"])
    require(output.parent == registry.root.resolve() / "outputs/diagnostics" / contract.EXECUTION_PROTOCOL,
        "Executor smoke escaped its diagnostic namespace")
    binding = contract.read_bound(output / "plan_binding.json", descriptor["run_binding_sha256"])
    require(binding.get("execution_plan_sha256") == plan_hash
        and binding.get("admission_sha256") == descriptor["admission_sha256"]
        and type(binding.get("max_images")) is int and binding["max_images"] == 1
        and digest(binding.get("group_ids")) == digest(plan["new_group_ids"])
        and binding.get("execution_root") == descriptor["execution_root"]
        and binding.get("execution_lifecycle") == "full_stage_keep_all_v1",
        "Executor diagnostic run binding differs")
    from ..runners.vcsf_efficacy_runner import keep_all_policy

    policy = contract.read_bound(output / "payload_policy.json", binding["payload_policy_sha256"])
    require(digest(policy) == digest(keep_all_policy(plan, scheduled, 1)), "Diagnostic retention or capacity changed")
    payload = contract.child(output, Path("groups") / "{:06d}".format(group["group_id"]) / "attack")
    require(output_dir is not None and Path(output_dir).absolute() == payload and not payload.exists(),
        "Diagnostic payload must be the new exact bound group leaf")
    return AttackSpec(id=contract.ALIAS, display_name="VCSF executor diagnostic: " + group["variant"],
        executor="native", parameters=dict(group["parameters"]), metadata=dict(metadata,
            candidate_id=contract.CANDIDATE_ID, fidelity_status="complete_core_executor_diagnostic_not_selection",
            fidelity_basis="original_full_input_contract_and_bounded_complete_executor_request",
            semantic_contract="complete_core_executor_diagnostic_not_selection",
            global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False,
            efficacy_eligible=False, scientific_acceptance=False, independent_acceptance=False,
            historical_reuse_qualified=False, automatic_promotion=False))
