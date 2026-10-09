"""Process-local A10/F03 Swin full-val invocation; no numerical fork."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..data.coco import CocoIndex
from ..io import file_digest
from ..registry import AttackSpec
from ..runners.vcsf_efficacy_contract import child
from ..runners.vcsf_research_plan import canonical_hash, require
from .factory import ATTACK_TYPES
from .vcsf_common_anchor_isolated import VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig


MODE = "a10_f03_swin_fullval_execution"
ALIAS = "vcsf_a10_f03_swin_fullval_isolated"
PROTOCOL = "vcsf_a10_f03_swin_fullval_sensitivity"
EXECUTION_PROTOCOL = "vcsf_a10_f03_swin_fullval_execution"
STUDY = "vcsf_final_background_validation"
SOURCE = "mask_rcnn_swin_t"
VARIANTS = ("A10", "F03")
IMAGES = 5000
_SOURCE_FILES = {
    "src/lgp/runners/attack.py",
    "src/lgp/attacks/vcsf_a10_f03_swin_fullval_isolation.py",
    "src/lgp/attacks/vcsf_common_anchor_isolated.py",
    "src/lgp/attacks/vcsf_scale_isolated.py",
    "src/lgp/attacks/vcsf_research_isolated.py",
    "src/lgp/attacks/vcsf_final_candidate.py",
    "src/lgp/attacks/common.py",
}


def _sha256(value):
    return (type(value) is str and len(value) == 64
        and all(character in "0123456789abcdef" for character in value))


def _plan(registry, descriptor):
    require(type(descriptor) is dict and set(descriptor) == {
        "mode", "execution_alias", "execution_plan", "execution_plan_sha256", "group_id"}
        and descriptor["mode"] == MODE and descriptor["execution_alias"] == ALIAS
        and type(descriptor["group_id"]) is int and descriptor["group_id"] in (1, 2),
        "Unsupported A10/F03 Swin execution descriptor")
    require(_sha256(descriptor["execution_plan_sha256"])
        and type(descriptor["execution_plan"]) is str,
        "Require a SHA-256-bound execution plan")
    path = child(registry.root, descriptor["execution_plan"])
    require(path.suffix == ".json" and registry.root / "outputs" / "plans" in path.parents,
        "Execution plan escaped the plans namespace")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == descriptor["execution_plan_sha256"],
        "Execution plan bytes changed")
    return json.loads(raw.decode("utf-8"))


def _registered_groups(registry):
    protocol = registry.protocols[PROTOCOL]
    study = registry.ablation_studies[STUDY]
    targets = registry.target_ids()
    require(protocol["dataset"] == "coco" and protocol["split"] == "val"
        and protocol["new_sources"] == [SOURCE]
        and protocol["reused_sources"] == ["faster_rcnn_r50"]
        and protocol["variants"] == list(VARIANTS)
        and protocol["seed"] == 42
        and protocol["seed_schedule"] == "full_val_numeric_image_position"
        and type(protocol["images_per_group"]) is int and protocol["images_per_group"] == IMAGES
        and protocol["new_generation_groups"] == 2
        and protocol["new_target_evaluations"] == 32
        and protocol["targets"] == "all" and protocol["execution_requires_independent_admission"] is True
        and protocol["diagnostic_only"] is True
        and len(targets) == 16 and SOURCE in targets,
        "Registered A10/F03 Swin full-val scope changed")
    require(study["anchor"] == "A10" and study["row_order"][3] == "F03"
        and study["variants"]["A10"]["parameters"] == {}
        and study["variants"]["F03"]["parameters"] == {"surface": "backbone"},
        "Base A10/F03 study changed")
    expected = []
    for group_id, variant in enumerate(VARIANTS, 1):
        raw = dict(study["base_parameters"], **study["variants"][variant]["parameters"])
        config = VCSFCommonAnchorConfig.from_mapping(raw)
        config.validate()
        parameters = asdict(config)
        parameters["feature_levels"] = list(parameters["feature_levels"])
        digest = canonical_hash(parameters)
        require(protocol["variant_parameter_sha256"][variant] == digest,
            "Protocol A10/F03 parameter hash changed")
        if variant == "A10":
            require(study["anchor_parameters_sha256"] == digest,
                "Base A10 parameter hash changed")
        expected.append((group_id, variant, parameters, digest))
    require(set(protocol["variant_parameter_sha256"]) == set(VARIANTS),
        "Unexpected variant in protocol")
    return protocol, targets, expected


def _verify_plan(registry, plan, protocol, targets, expected, max_images):
    require(type(plan) is dict
        and plan.get("record_type") == "vcsf_a10_f03_swin_fullval_execution_plan"
        and plan.get("protocol") == PROTOCOL
        and plan.get("status") == "prepared_pending_independent_admission"
        and (plan.get("max_images") is None if max_images is None
            else type(plan.get("max_images")) is int and plan["max_images"] == 1)
        and plan.get("targets") == targets,
        "Execution plan is not the registered pending A10/F03 full-val plan")
    dataset = registry.dataset("coco")
    index = CocoIndex(dataset, "val")
    full_ids = [int(row["id"]) for row in index.images]
    require(len(full_ids) == IMAGES and full_ids == sorted(set(full_ids))
        and plan.get("full_image_ids_sha256") == canonical_hash(full_ids)
        and protocol["ordered_image_ids_sha256"] == plan["full_image_ids_sha256"]
        and plan.get("image_ids") == (full_ids if max_images is None else full_ids[:1])
        and _sha256(plan.get("annotation_sha256"))
        and file_digest(index.annotation_path) == plan["annotation_sha256"],
        "Plan image population or annotation differs from COCO val2017")
    require(protocol["annotation_sha256"] == plan["annotation_sha256"],
        "Protocol annotation hash differs")
    if "image_ids_sha256" in plan:
        require(plan["image_ids_sha256"] == canonical_hash(plan["image_ids"]),
            "Plan selected image IDs hash differs")
    checkpoints = plan.get("checkpoint_sha256")
    require(type(checkpoints) is dict and set(checkpoints) == set(targets)
        and all(_sha256(value) for value in checkpoints.values()),
        "Plan must bind all 16 checkpoint hashes")
    sources = plan.get("source_sha256")
    require(type(sources) is dict and _SOURCE_FILES <= set(sources)
        and all(type(name) is str and _sha256(digest) for name, digest in sources.items()),
        "Plan lacks numerical source hashes")
    for name, digest in sources.items():
        require(name.startswith(("src/", "configs/", "experiments/", "tools/", "tests/"))
            and file_digest(child(registry.root, name)) == digest,
            "Bound source changed: " + name)
    groups = plan.get("groups")
    require(type(groups) is list and len(groups) == 2,
        "A10/F03 Swin plan requires exactly two new groups")
    for row, (group_id, variant, parameters, digest) in zip(groups, expected):
        require(type(row) is dict and type(row.get("group_id")) is int
            and row["group_id"] == group_id and row.get("variant") == variant
            and row.get("source") == SOURCE and type(row.get("seed")) is int
            and row["seed"] == 42 and type(row.get("images")) is int
            and row["images"] == (IMAGES if max_images is None else 1)
            and row.get("parameters_sha256") == digest
            and type(row.get("parameters")) is dict
            and canonical_hash(row["parameters"]) == digest
            and row["parameters"] == parameters,
            "Plan group differs from the exact registered A10/F03 parameters")
    return full_ids, groups


def verify_source_checkpoint(checkpoint, expected_sha256):
    require(_sha256(expected_sha256) and file_digest(Path(checkpoint)) == expected_sha256,
        "Loaded Swin source checkpoint differs from the plan")


def build_swin_fullval(adapter, parameters):
    return VCSFCommonAnchorCandidate(adapter,
        VCSFCommonAnchorConfig.from_mapping(dict(parameters)))


def resolve_swin_fullval(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None):
    require(ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        "A10/F03 Swin alias must remain process-local")
    protocol, targets, expected = _registered_groups(registry)
    require(dataset_id == "coco" and split == "val" and source_id == SOURCE
        and attack_id == ALIAS and type(seed) is int and seed == 42
        and (max_images is None or type(max_images) is int and max_images == 1)
        and seed_offsets is None and input_transform is None
        and budget_profile == protocol["budget_profile"] and output_dir is not None,
        "Invocation differs from exact full-val Swin scope")
    plan = _plan(registry, descriptor)
    ids, groups = _verify_plan(registry, plan, protocol, targets, expected, max_images)
    require(type(image_ids) in (list, tuple) and list(image_ids) == ids
        and all(type(value) is int for value in image_ids),
        "Invocation requires explicit canonical sorted full-val image IDs")
    group = groups[descriptor["group_id"] - 1]
    output = Path(output_dir).absolute()
    category = "experiments" if max_images is None else "diagnostics"
    namespace = registry.root / "outputs" / category / EXECUTION_PROTOCOL
    require(output.name == "attack"
        and output.parent.name == "{:06d}".format(group["group_id"])
        and output.parent.parent.name == "groups"
        and output.parent.parent.parent.parent == namespace
        and registry.root in output.parents
        and child(registry.root, output.relative_to(registry.root)) == output,
        "Attack output escaped the exact Swin group leaf")
    require(type(parameter_overrides) is dict
        and parameter_overrides == group["parameters"]
        and canonical_hash(parameter_overrides) == group["parameters_sha256"],
        "Invocation parameters differ from the bound group")
    config = VCSFCommonAnchorConfig.from_mapping(parameter_overrides)
    config.validate()
    require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Configuration normalization changed")
    require(type(run_metadata) is dict and run_metadata.get("group_id") == group["group_id"]
        and run_metadata.get("variant") == group["variant"]
        and run_metadata.get("protocol") == PROTOCOL,
        "Invocation metadata differs from the bound group")
    return AttackSpec(id=ALIAS, display_name="VCSF Swin full-val " + group["variant"],
        executor="native", parameters=dict(group["parameters"]), metadata={
            "implementation": "src/lgp/attacks/vcsf_common_anchor_isolated.py",
            "protocol": PROTOCOL, "group_id": group["group_id"],
            "variant": group["variant"], "source": SOURCE,
            "execution_plan": descriptor["execution_plan"],
            "execution_plan_sha256": descriptor["execution_plan_sha256"],
            "parameters_sha256": group["parameters_sha256"],
            "image_ids_sha256": canonical_hash(plan["image_ids"]),
            "full_image_ids_sha256": plan["full_image_ids_sha256"],
            "annotation_sha256": plan["annotation_sha256"],
            "source_checkpoint_sha256": plan["checkpoint_sha256"][SOURCE],
            "diagnostic_only": max_images is not None,
            "fidelity_status": "registered_swin_fullval_pending_independent_result_audit",
            "semantic_contract": "retrospective_two_source_sensitivity_not_independent_confirmation",
            "global_registry_mutated": False, "global_factory_mutated": False,
            "formal_AP_eligible": False, "scientific_acceptance": False,
            "final_method_promoted": False})
