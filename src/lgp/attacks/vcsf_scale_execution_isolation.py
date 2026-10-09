"""Formal scale-stage resolver; the one-image structure resolver stays unchanged."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from ..runners import vcsf_scale_execution_contract as contract
from ..runners.vcsf_research_plan import canonical_hash, require
from .factory import ATTACK_TYPES
from .vcsf_scale_isolated import VCSFScaleConfig


def resolve_scale_efficacy(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None):
    require(descriptor.get("mode") == contract.MODE and descriptor.get("candidate_id") == contract.CANDIDATE_ID
        and descriptor.get("execution_alias") == attack_id == contract.ALIAS,
        "Unsupported single-source scale execution descriptor")
    require(contract.ALIAS not in registry.attacks and contract.ALIAS not in ATTACK_TYPES,
        "Scale efficacy must remain a process-local overlay")
    require(dataset_id == "coco" and split == "val" and image_ids is None and seed_offsets is None
        and input_transform is None and source_id == "faster_rcnn_r50" and type(seed) is int and seed == 42,
        "Scale efficacy requires the canonical source, images, seed and preprocessing")
    require(descriptor.get("max_images") == max_images and type(descriptor.get("max_images")) is type(max_images),
        "Execution image limit differs from its descriptor")
    plan = contract.read_bound(contract.child(registry.root, descriptor["execution_plan"]), descriptor["execution_plan_sha256"])
    contract.verify_execution_plan(registry, plan)
    contract.verify_admission(registry, plan, descriptor["execution_plan_sha256"],
        contract.child(registry.root, descriptor["admission"]), descriptor["admission_sha256"], max_images)
    require(budget_profile == registry.protocols[contract.PROTOCOL]["budget_profile"], "Scale cost regime changed")
    scheduled = contract.execution_groups(plan, max_images)
    matched = [g for g in scheduled if g["variant"] == run_metadata.get("variant")
        and g["source"] == source_id and g["seed"] == seed]
    require(len(matched) == 1, "A reused or unscheduled scale group cannot be generated")
    group = matched[0]
    metadata = contract.execution_metadata(plan, descriptor["execution_plan_sha256"], group, max_images)
    require(canonical_hash(dict(run_metadata)) == canonical_hash(metadata)
        and canonical_hash(dict(parameter_overrides)) == group["parameters_sha256"],
        "Scale invocation changed full parameters or runtime metadata")
    config = VCSFScaleConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"] and not config.is_reference(),
        "Scale normalization or reference shortcut differs")
    output = contract.child(registry.root, descriptor["execution_root"])
    parent = registry.root / "outputs" / ("experiments" if max_images is None else "diagnostics") / contract.EXECUTION_PROTOCOL
    require(output.parent == parent, "Scale execution root escaped its new namespace")
    binding = contract.read_bound(output / "plan_binding.json", descriptor["run_binding_sha256"])
    require(binding.get("execution_plan_sha256") == descriptor["execution_plan_sha256"]
        and binding.get("admission_sha256") == descriptor["admission_sha256"]
        and binding.get("max_images") == max_images and type(binding.get("max_images")) is type(max_images)
        and binding.get("group_ids") == [g["group_id"] for g in scheduled]
        and binding.get("execution_root") == descriptor["execution_root"], "Scale run binding differs")
    require(output_dir is not None and Path(output_dir).absolute() == output / "groups" / "{:06d}".format(group["group_id"]) / "attack",
        "Scale payload escaped its exact bound group")
    return AttackSpec(id=contract.ALIAS, display_name="VCSF scale: " + group["variant"], executor="native",
        parameters=dict(group["parameters"]), metadata=dict(metadata, candidate_id=contract.CANDIDATE_ID,
            fidelity_status="registered_single_source_scale_pending_group_acceptance",
            fidelity_basis="exact_anchor_mapping_and_bound_execution_admission",
            semantic_contract="retrospective_fullval_single_source_not_independent_confirmation",
            global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False))
