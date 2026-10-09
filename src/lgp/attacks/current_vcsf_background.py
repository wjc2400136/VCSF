"""Exact registered final-background controls over the unchanged numerical builder."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict

from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
from .vcsf_public import _hash, verify_public_identity

PROTOCOL_ID = "current_vcsf_final_background"


def registered_parameters(registry, variant):
    spec = registry.protocols[PROTOCOL_ID]
    study = registry.ablation_studies[spec["canonical_variant_study"]]
    if type(variant) is not str or variant not in study["row_order"]:
        raise ValueError("Select an exact registered final-background variant")
    raw = deepcopy(study["base_parameters"])
    raw.update(study["variants"][variant]["parameters"])
    config = VCSFCommonAnchorConfig.from_mapping(raw)
    config.validate()
    return asdict(config)


def validate_parameters(registry, parameters, variant):
    identity = verify_public_identity(registry.root, registry.attack("vcsf"))
    expected = registered_parameters(registry, variant)
    if type(parameters) is not dict or set(parameters) != set(expected):
        raise ValueError("Background controls require their complete registered parameter map")
    config = VCSFCommonAnchorConfig.from_mapping(parameters)
    config.validate()
    values = asdict(config)
    if _hash(values) != _hash(expected):
        raise ValueError("Final-background parameters differ from the exact registered control")
    if _hash(registered_parameters(registry, "A10")) != identity["parameters_sha256"]:
        raise RuntimeError("Final-background anchor differs from selected A10")
    return values, identity


def build_background(registry, adapter, parameters, variant):
    values, identity = validate_parameters(registry, dict(parameters), variant)
    from .vcsf_common_anchor_execution_isolation import build_common_anchor_execution

    attack = build_common_anchor_execution(adapter, dict(parameters))
    ancestry = [cls.__module__ + "." + cls.__name__ for cls in type(attack).__mro__]
    if ancestry != identity["actual_class_mro"] or _hash(asdict(attack.config)) != _hash(values):
        raise RuntimeError("Final-background reproduction changed the inherited numerical path")
    return attack


def validate_invocation(registry, invocation, parameter_overrides):
    spec = registry.protocols[PROTOCOL_ID]
    metadata = invocation.get("run_metadata") or {}
    if (type(invocation) is not dict or metadata.get("public_protocol") != PROTOCOL_ID
            or invocation.get("dataset_id") != "coco" or invocation.get("split") != "val"
            or invocation.get("source_id") not in spec["sources"]
            or invocation.get("attack_id") != "vcsf"
            or invocation.get("budget_profile") != spec["budget_profile"]
            or invocation.get("input_transform") is not None
            or invocation.get("image_ids") is not None or invocation.get("seed_offsets") is not None
            or invocation.get("strict") is not True or invocation.get("keep_going") is not False
            or invocation.get("download_weights") is not False):
        raise ValueError("Final-background reproduction requires its registered strict fresh-input invocation")
    if type(invocation.get("seed")) is not int or invocation["seed"] != spec["seed"]:
        raise ValueError("Final-background reproduction preserves the original full-index seed mapping")
    maximum = invocation.get("max_images")
    if maximum is not None and (type(maximum) is not int or not 0 < maximum <= spec["full_images"]):
        raise ValueError("Background max_images must be within the full validation population")
    if type(parameter_overrides) is not dict:
        raise ValueError("Final-background invocation requires the complete explicit control map")
    validate_parameters(registry, parameter_overrides, metadata.get("background_variant"))
    return deepcopy(parameter_overrides)