"""Isolated common-anchor controls; no catalogue or factory registration."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .vcsf_research_isolated import VCSFResearchConfig
from .vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig


@dataclass(frozen=True)
class VCSFCommonAnchorConfig(VCSFScaleConfig):
    """Eight operators with common-baseline ablations at two fixed scales."""

    def validate(self):
        if any(type(level) is not int for level in self.feature_levels):
            raise ValueError("Common anchor requires integer feature_levels")
        for name in ("scale_min", "scale_max", "momentum"):
            if isinstance(getattr(self, name), bool):
                raise ValueError(name + " must be numeric, not boolean")
        # Only the validation copy uses 20; the actual feature-only loop stays 19.
        checked = self
        if self.initialization == "none" and type(self.iterations) is int and self.iterations == 19:
            checked = replace(self, iterations=20)
        VCSFResearchConfig.validate(checked)
        domains = {
            "image_interpolation": ("bilinear", "nearest"),
            "placement": ("random", "center"),
            "image_padding": ("reflect", "zero"),
        }
        for name, allowed in domains.items():
            if getattr(self, name) not in allowed:
                raise ValueError("Unknown common-anchor setting: " + name)
        if (self.scale_min, self.scale_max) not in (
                (1.0, 1.0), (2.0 / 3.0, 4.0 / 3.0)):
            raise ValueError("Common anchor permits only identity or halfwidth 1/3 scale")
        if self.placement == "center" and self.correspondence != "shared":
            raise ValueError("Centered placement requires a prospective correspondence design; the inherited switch is inactive")
        if self.scale_min == self.scale_max == 1 and self.correspondence != "shared":
            raise ValueError("Identity geometry cannot represent a correspondence violation")


class VCSFCommonAnchorCandidate(VCSFScaleCandidate):
    """Inherit all numerical methods while enforcing the isolated config boundary."""

    implementation_path = "src/lgp/attacks/vcsf_common_anchor_isolated.py"
    method_name = "VCSF isolated common-anchor ablation"

    def __init__(self, adapter, config: VCSFCommonAnchorConfig):
        if not isinstance(config, VCSFCommonAnchorConfig):
            raise ValueError("Common anchor requires VCSFCommonAnchorConfig")
        super().__init__(adapter, config)
