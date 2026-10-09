"""Registered epsilon-only validation over the unchanged selected A10 attack."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass
from fractions import Fraction

from .vcsf_common_anchor_isolated import VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig
from .vcsf_public import _hash, verify_public_identity

PROTOCOL_ID = "current_vcsf_radius"


def validate_parameters(registry, parameters):
    identity = verify_public_identity(registry.root, registry.attack("vcsf"))
    expected = identity["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != set(expected):
        raise ValueError("Radius execution requires the complete registered A10 parameter map")
    if isinstance(parameters.get("eps"), bool):
        raise ValueError("Boolean is not a registered radius")
    config = VCSFCommonAnchorConfig.from_mapping(parameters)
    values = asdict(config)
    spec = registry.protocols[PROTOCOL_ID]
    labels = [label for label in spec["epsilon_order"]
              if values["eps"] == float(Fraction(label))]
    if len(labels) != 1 or _hash(dict(values, eps=expected["eps"])) != _hash(expected):
        raise ValueError("Only the registered radius may differ from the complete A10 background")
    VCSFCommonAnchorConfig.from_mapping(expected).validate()
    return values, labels[0], identity


@dataclass(frozen=True)
class CurrentRadiusConfig(VCSFCommonAnchorConfig):
    registry_binding: InitVar[object] = None

    def __post_init__(self, registry_binding):
        object.__setattr__(self, "_registry", registry_binding)

    def validate(self):
        if self._registry is None:
            raise ValueError("Current radius requires its originating registry")
        validate_parameters(self._registry, asdict(self))


def make_config(registry, parameters):
    values, label, identity = validate_parameters(registry, dict(parameters))
    values["feature_levels"] = tuple(values["feature_levels"])
    config = CurrentRadiusConfig(**values, registry_binding=registry)
    config.validate()
    return config


def build_radius(registry, adapter, parameters):
    values, label, identity = validate_parameters(registry, dict(parameters))
    attack = VCSFCommonAnchorCandidate(adapter, make_config(registry, parameters))
    ancestry = [cls.__module__ + "." + cls.__name__ for cls in type(attack).__mro__]
    if ancestry != identity["actual_class_mro"] or _hash(asdict(attack.config)) != _hash(values):
        raise RuntimeError("Current radius changed the inherited A10 numerical path")
    return attack


def validate_invocation(registry, invocation, parameter_overrides):
    spec = registry.protocols[PROTOCOL_ID]
    if (type(invocation) is not dict or invocation.get("dataset_id") != "coco"
            or invocation.get("split") != "val" or invocation.get("source_id") not in spec["sources"]
            or invocation.get("attack_id") != "vcsf"
            or invocation.get("budget_profile") != spec["budget_profile"]
            or invocation.get("input_transform") is not None):
        raise ValueError("Current radius is restricted to its registered COCO Common-2 invocation")
    if not isinstance(parameter_overrides, dict) or set(parameter_overrides) != {"eps"}:
        raise ValueError("Current radius permits exactly one explicit epsilon override")
    parameters = deepcopy(dict(registry.attack("vcsf").parameters))
    parameters.update(parameter_overrides)
    validate_parameters(registry, parameters)
    if type(invocation.get("seed")) is not int or invocation["seed"] != spec["seed"]:
        raise ValueError("Current radius preserves its registered image seed schedule")
    maximum = invocation.get("max_images")
    if maximum is not None and (type(maximum) is not int or not 0 < maximum <= spec["full_images"]):
        raise ValueError("Radius max_images must be within the registered full population")
    return parameters
