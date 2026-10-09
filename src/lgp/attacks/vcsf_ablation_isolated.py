"""Whole-ablation controls without modifying frozen or retired implementations."""
from __future__ import annotations

from dataclasses import dataclass, replace

from .vcsf_research_isolated import VCSFResearchConfig
from .vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig


@dataclass(frozen=True)
class VCSFAblationConfig(VCSFScaleConfig):
    def validate(self):
        # The new owner-approved none+19 control changes only the loop bound.
        checked = self
        if self.initialization == "none" and type(self.iterations) is int and self.iterations == 19:
            checked = replace(self, iterations=20)
        VCSFResearchConfig.validate(checked)
        VCSFScaleConfig(
            scale_min=self.scale_min, scale_max=self.scale_max,
            image_interpolation=self.image_interpolation,
            placement=self.placement, image_padding=self.image_padding,
        ).validate()
        if self.placement == "center" and self.correspondence != "shared":
            raise ValueError("Centered placement requires a prospective correspondence design; the inherited switch is inactive")
        if self.scale_min == self.scale_max == 1 and self.correspondence != "shared":
            raise ValueError("Identity geometry cannot represent a correspondence violation")


class VCSFAblationCandidate(VCSFScaleCandidate):
    implementation_path = "src/lgp/attacks/vcsf_ablation_isolated.py"
    method_name = "VCSF isolated single-source whole ablation"
