"""Radius-only validation adapter; all attack numerical methods remain inherited."""
from copy import deepcopy
from dataclasses import InitVar, asdict, dataclass
from fractions import Fraction
import math
from typing import Any, Mapping, Optional, Sequence

from .vcsf_common_anchor_isolated import VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig
from ..registry import Registry
from ..runners.vcsf_research_plan import canonical_hash


@dataclass(frozen=True)
class VCSFFinalRadiusConfig(VCSFCommonAnchorConfig):
    frozen_background: InitVar[Optional[Mapping[str, Any]]] = None
    registered_radii: InitVar[Optional[Sequence[float]]] = None

    def __post_init__(self, frozen_background, registered_radii):
        object.__setattr__(self, "_background", deepcopy(frozen_background))
        object.__setattr__(self, "_radii", tuple(registered_radii or ()))

    def validate(self):
        if self._background is None or not self._radii:
            raise ValueError("Radius adapter requires the registered frozen background and radii")
        spec = Registry().protocols["vcsf_final_budget_refresh"]
        expected_radii = tuple(float(Fraction(value)) for value in spec["epsilon_order"])
        if canonical_hash(self._background) != spec["parameters_sha256"] or self._radii != expected_radii:
            raise ValueError("Radius adapter bindings do not match the registered frozen policy")
        if type(self.eps) not in (int, float) or not math.isfinite(self.eps) or self.eps not in self._radii:
            raise ValueError("Radius is outside the registered sensitivity axis")
        current = asdict(self)
        current["eps"] = self._background["eps"]
        if canonical_hash(current) != canonical_hash(self._background):
            raise ValueError("Radius sensitivity cannot change another frozen method parameter")
        # Validate the unchanged background, not a modified copy of the producer.
        VCSFCommonAnchorConfig.from_mapping(current).validate()


def make_radius_config(registry, frozen_parameters, epsilon):
    spec = registry.protocols["vcsf_final_budget_refresh"]
    if canonical_hash(frozen_parameters) != spec["parameters_sha256"]:
        raise ValueError("Radius adapter background does not match the frozen method")
    baseline = VCSFCommonAnchorConfig.from_mapping(frozen_parameters)
    baseline.validate()
    values = asdict(baseline)
    if canonical_hash(values) != spec["parameters_sha256"] or baseline.is_reference():
        raise ValueError("Frozen configuration changed while parsing")
    if isinstance(epsilon, bool):
        raise ValueError("Boolean is not a radius")
    if isinstance(epsilon, str):
        if epsilon not in spec["epsilon_order"]:
            raise ValueError("Unknown registered radius label")
        epsilon = float(Fraction(epsilon))
    radii = tuple(float(Fraction(value)) for value in spec["epsilon_order"])
    parameters = dict(values, eps=epsilon)
    config = VCSFFinalRadiusConfig(**parameters, frozen_background=values, registered_radii=radii)
    config.validate()
    return config


def build_radius_candidate(adapter, registry, frozen_parameters, epsilon):
    return VCSFCommonAnchorCandidate(adapter, make_radius_config(registry, frozen_parameters, epsilon))
