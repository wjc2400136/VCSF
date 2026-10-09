"""Plan-bound single-source scale diagnostics; this bridge cannot run efficacy."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from .factory import ATTACK_TYPES
from .vcsf_research_isolation import _read_bound_json
from .vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig


ALIAS = "vcsf_scale_isolated"
MODE = "scale_structure_preflight"


def resolve_scale_preflight(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata,
        budget_profile, image_ids=None, seed_offsets=None, input_transform=None):
    from ..runners.vcsf_scale_plan import PROTOCOL, compile_scale_plan
    from ..runners.vcsf_research_plan import canonical_hash, require

    protocol = registry.protocols[PROTOCOL]
    require(descriptor.get("mode") == MODE and descriptor.get("candidate_id") == PROTOCOL
        and descriptor.get("execution_alias") == ALIAS and attack_id == ALIAS,
        "Unsupported single-source scale diagnostic descriptor")
    require(ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        "Scale diagnostics must not change the global method registry or factory")
    require(type(max_images) is int and max_images == 1,
        "Scale structure preflight requires exactly one image, never formal AP")
    require(dataset_id == protocol["dataset"] and split == protocol["split"]
        and source_id == protocol["sources"][0] and type(seed) is int and seed == protocol["seed"]
        and image_ids is None and seed_offsets is None and input_transform is None,
        "Scale diagnostics require the canonical first COCO-val image/source/seed")
    require(budget_profile == protocol["budget_profile"]
        and run_metadata.get("protocol") == PROTOCOL
        and run_metadata.get("study") == protocol["ablation_study"],
        "Scale diagnostic protocol or cost regime changed")
    raw = Path(str(descriptor.get("plan", "")))
    require(not raw.is_absolute(), "Scale plan must use a project-relative path")
    path = (registry.root / raw).resolve()
    path.relative_to(registry.root.resolve())
    plan = _read_bound_json(path, descriptor.get("plan_sha256"))
    require(canonical_hash(plan) == canonical_hash(compile_scale_plan(registry)),
        "Scale plan or its implementation snapshot drifted")
    groups = [g for g in plan["groups"] if g["variant"] == run_metadata.get("variant")]
    require(len(groups) == 1, "Unscheduled scale diagnostic variant")
    group = groups[0]
    require(canonical_hash(dict(parameter_overrides)) == group["parameters_sha256"],
        "Scale parameters must exactly match the complete resolved group")
    config = VCSFScaleConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Scale parameter normalization changed the declared configuration")
    return AttackSpec(id=ALIAS, display_name="VCSF scale diagnostic: " + group["variant"],
        executor="native", parameters=dict(group["parameters"]), metadata={
            "candidate_id": PROTOCOL, "protocol": PROTOCOL,
            "study": protocol["ablation_study"], "variant": group["variant"],
            "configuration_seed_group_id": group["group_id"],
            "prepared_plan_sha256": descriptor["plan_sha256"],
            "implementation": protocol["isolated_candidate"]["implementation"],
            "fidelity_status": "isolated_structure_diagnostic_not_efficacy",
            "fidelity_basis": "registered_single_source_scale_plan",
            "semantic_contract": "four_level_scale_operator_structure_only",
            "global_registry_mutated": False, "global_factory_mutated": False,
            "formal_AP_eligible": False, "independent_confirmation": False,
            "historical_reuse_qualified": False, "automatic_promotion": False,
        })


def build_scale_preflight(adapter, parameters):
    return VCSFScaleCandidate(adapter, VCSFScaleConfig.from_mapping(dict(parameters)))
