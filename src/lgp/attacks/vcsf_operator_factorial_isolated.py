"""Bounded operator factorial at half-width 1/3; no study-row registration."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .vcsf_research_isolated import VCSFResearchConfig
from .vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig


@dataclass(frozen=True)
class VCSFOperatorFactorialConfig(VCSFScaleConfig):
    """Distinct eight-operator namespace with the four-term background fixed."""

    def validate(self):
        # Reuse field validation, not the historical four-combination whitelist.
        VCSFResearchConfig.validate(self)
        if any(type(level) is not int for level in self.feature_levels):
            raise ValueError("Operator factorial requires integer feature_levels")
        reference = asdict(VCSFResearchConfig(levels_per_stage=2))
        current = asdict(self)
        for key, expected in reference.items():
            if current[key] != expected:
                raise ValueError("Operator factorial fixes the four-term background: " + key)
        domains = {
            "image_interpolation": ("bilinear", "nearest"),
            "placement": ("random", "center"),
            "image_padding": ("reflect", "zero"),
        }
        for key, allowed in domains.items():
            if current[key] not in allowed:
                raise ValueError("Unknown operator factorial setting: " + key)


class VCSFOperatorFactorialCandidate(VCSFScaleCandidate):
    """Reuse scale numerics without widening the historical scale producer."""

    implementation_path = "src/lgp/attacks/vcsf_operator_factorial_isolated.py"
    method_name = "VCSF isolated four-term operator factorial at half-width 1/3"

    def __init__(self, adapter, config: VCSFOperatorFactorialConfig):
        if not isinstance(config, VCSFOperatorFactorialConfig):
            raise ValueError("Operator factorial requires VCSFOperatorFactorialConfig")
        super().__init__(adapter, config)
