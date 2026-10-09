"""Explicit full-efficacy resolver; the existing structure guard stays unchanged."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from ..runners.vcsf_efficacy_contract import (
    EXECUTION_PROTOCOL, child, execution_groups, execution_metadata, read_bound,
    require, verify_admission, verify_execution_plan,
)
from ..runners.vcsf_research_plan import PROTOCOL, canonical_hash
from .factory import ATTACK_TYPES
from .vcsf_research_isolated import VCSFResearchConfig
from .vcsf_research_isolation import ALIAS, CANDIDATE_ID


def resolve_research_efficacy(
    registry, descriptor, *, dataset_id, split, source_id, attack_id, seed,
    max_images, parameter_overrides, run_metadata, budget_profile,
    image_ids=None, seed_offsets=None, input_transform=None, output_dir=None,
):
    require(descriptor.get("mode") == "layered_efficacy"
        and descriptor.get("candidate_id") == CANDIDATE_ID
        and descriptor.get("execution_alias") == attack_id == ALIAS,
        "Unsupported efficacy execution descriptor")
    require(ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        "Isolated efficacy must not mutate the global registry or factory")
    require(dataset_id == "coco" and split == "val" and image_ids is None
        and seed_offsets is None and input_transform is None,
        "Efficacy must preserve canonical images, seed offsets and preprocessing")
    require(descriptor.get("max_images") == max_images
        and type(descriptor.get("max_images")) is type(max_images),
        "The descriptor image limit differs from the invocation")
    plan_path = child(registry.root, descriptor["execution_plan"])
    plan = read_bound(plan_path, descriptor["execution_plan_sha256"])
    verify_execution_plan(registry, plan)
    verify_admission(registry, plan, descriptor["execution_plan_sha256"],
        child(registry.root, descriptor["admission"]), descriptor["admission_sha256"], max_images)
    require(budget_profile == registry.protocols[PROTOCOL]["budget_profile"],
        "The declared computational regime changed")
    scheduled = execution_groups(plan, max_images)
    matches = [group for group in scheduled if group["source"] == source_id
        and group["seed"] == seed and group["variant"] == run_metadata.get("variant")]
    require(type(seed) is int and len(matches) == 1, "Unscheduled or reused efficacy group")
    group = matches[0]
    expected_metadata = execution_metadata(plan, descriptor["execution_plan_sha256"], group, max_images)
    require(canonical_hash(dict(run_metadata)) == canonical_hash(expected_metadata),
        "Generation metadata is not the exact execution binding")
    require(canonical_hash(dict(parameter_overrides)) == group["parameters_sha256"],
        "Complete efficacy parameters must match the prepared group")
    config = VCSFResearchConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Configuration normalization changed the registered parameters")
    run_root = child(registry.root, descriptor["execution_root"])
    namespace = "experiments" if max_images is None else "diagnostics"
    run_root.relative_to(registry.root / "outputs" / namespace / EXECUTION_PROTOCOL)
    binding = read_bound(run_root / "plan_binding.json", descriptor["run_binding_sha256"])
    require(binding.get("execution_plan_sha256") == descriptor["execution_plan_sha256"]
        and binding.get("admission_sha256") == descriptor["admission_sha256"]
        and binding.get("max_images") == max_images
        and binding.get("group_ids") == [g["group_id"] for g in scheduled]
        and binding.get("execution_root") == descriptor["execution_root"],
        "New-root execution binding changed")
    expected_output = run_root / "groups" / "{:06d}".format(group["group_id"]) / "attack"
    require(output_dir is not None and Path(output_dir).absolute() == expected_output
        and not expected_output.is_symlink(), "Generation escaped its new bound group root")
    return AttackSpec(
        id=ALIAS, display_name="VCSF research: " + group["variant"], executor="native",
        parameters=dict(group["parameters"]),
        metadata=dict(expected_metadata, candidate_id=CANDIDATE_ID,
            fidelity_status="registered_retrospective_research_pending_group_acceptance",
            fidelity_basis="unchanged_candidate_and_hash_bound_execution_admission",
            semantic_contract="retrospective_fullval_not_independent_confirmation",
            global_registry_mutated=False, global_factory_mutated=False,
            formal_AP_eligible=False),
    )

