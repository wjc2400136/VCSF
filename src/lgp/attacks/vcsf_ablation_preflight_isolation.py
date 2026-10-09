"""Process-local, one-image ablation integration bridge; never an AP admission."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..registry import AttackSpec
from .factory import ATTACK_TYPES
from .vcsf_ablation_isolated import VCSFAblationCandidate, VCSFAblationConfig


def resolve_ablation_preflight(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None):
    from ..runners import vcsf_ablation_preflight as contract

    require, canonical_hash = contract.require, contract.canonical_hash
    require(descriptor.get("mode") == contract.MODE
        and descriptor.get("candidate_id") == contract.CANDIDATE_ID
        and descriptor.get("execution_alias") == attack_id == contract.ALIAS,
        "Unsupported single-source ablation preflight descriptor")
    require(contract.ALIAS not in registry.attacks and contract.ALIAS not in ATTACK_TYPES,
        "Ablation preflight must remain a process-local overlay")
    require(type(max_images) is int and max_images == 1
        and type(descriptor.get("max_images")) is int and descriptor["max_images"] == 1,
        "Ablation integration preflight requires exactly one image, never formal AP")
    require(dataset_id == "coco" and split == "val" and source_id == "faster_rcnn_r50"
        and type(seed) is int and seed == 42 and image_ids is None and seed_offsets is None
        and input_transform is None,
        "Ablation preflight requires the canonical source, images, seed and preprocessing")
    plan_hash = descriptor["execution_plan_sha256"]
    plan = contract.read_bound(contract.child(registry.root, descriptor["execution_plan"]), plan_hash)
    contract.verify_execution_plan(registry, plan)
    # The shared runner's admission fields carry a scoped diagnostic request here.
    contract.verify_admission(registry, plan, plan_hash,
        contract.child(registry.root, descriptor["admission"]), descriptor["admission_sha256"], max_images)
    require(budget_profile == registry.protocols[contract.PROTOCOL]["budget_profile"],
        "Ablation preflight cost regime changed")
    scheduled = contract.execution_groups(plan, max_images)
    matched = [g for g in scheduled if g["variant"] == run_metadata.get("variant")
        and g["source"] == source_id and type(g["seed"]) is int and g["seed"] == seed]
    require(len(matched) == 1, "A reused or unscheduled ablation group cannot be generated")
    group = matched[0]
    metadata = contract.execution_metadata(plan, plan_hash, group, max_images)
    require(canonical_hash(dict(run_metadata)) == canonical_hash(metadata)
        and canonical_hash(dict(parameter_overrides)) == group["parameters_sha256"]
        and canonical_hash(group["parameters"]) == group["parameters_sha256"],
        "Ablation invocation changed full parameters or runtime metadata")
    config = VCSFAblationConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Ablation normalization changed the declared configuration")
    output = contract.child(registry.root, descriptor["execution_root"])
    parent = registry.root.resolve() / "outputs" / "diagnostics" / contract.EXECUTION_PROTOCOL
    require(output.parent == parent, "Ablation preflight root escaped its diagnostic namespace")
    binding = contract.read_bound(output / "plan_binding.json", descriptor["run_binding_sha256"])
    require(binding.get("execution_plan_sha256") == plan_hash
        and binding.get("admission_sha256") == descriptor["admission_sha256"]
        and type(binding.get("max_images")) is int and binding["max_images"] == 1
        and canonical_hash(binding.get("group_ids")) == canonical_hash([g["group_id"] for g in scheduled])
        and binding.get("execution_root") == descriptor["execution_root"],
        "Ablation preflight run binding differs")
    require(type(group["group_id"]) is int and group["group_id"] > 0,
        "Ablation preflight requires a positive integer group ID")
    relative = Path(descriptor["execution_root"]) / "groups" / "{:06d}".format(group["group_id"]) / "attack"
    payload = contract.child(registry.root, relative)
    require(output_dir is not None and Path(output_dir).absolute() == payload,
        "Ablation payload escaped its exact bound group")
    require(not payload.exists(), "Ablation preflight cannot resume or overwrite an existing payload")
    return AttackSpec(id=contract.ALIAS, display_name="VCSF ablation preflight: " + group["variant"],
        executor="native", parameters=dict(group["parameters"]), metadata=dict(metadata,
            candidate_id=contract.CANDIDATE_ID,
            fidelity_status="fixture_or_bound_stage_control_diagnostic_not_selection",
            fidelity_basis="bound_stage_plan_and_scoped_diagnostic_request",
            semantic_contract="fixture_or_bound_stage_control_diagnostic_not_selection",
            global_registry_mutated=False, global_factory_mutated=False,
            formal_AP_eligible=False, formal_metrics_eligible=False, efficacy_eligible=False,
            scientific_acceptance=False, independent_acceptance=False, independent_confirmation=False,
            historical_reuse_qualified=False, automatic_promotion=False))


def build_ablation_preflight(adapter, parameters):
    return VCSFAblationCandidate(adapter, VCSFAblationConfig.from_mapping(dict(parameters)))
