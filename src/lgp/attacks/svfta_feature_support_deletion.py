from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from .base import AttackOutput
from .svfta import SVFTA
from .svfta_minimal_anchor import (
    SVFTAMinimalAnchor,
    SVFTAMinimalAnchorConfig,
)


@dataclass(frozen=True)
class SVFTAFeatureSupportDeletionConfig(SVFTAMinimalAnchorConfig):
    """Two-factor deletion of feature surface and spatial support."""

    feature_surface_mode: str = "cross_stage"
    spatial_support_mode: str = "object_energy"

    @property
    def include_backbone_surface(self) -> bool:
        return self.feature_surface_mode == "cross_stage"

    def validate(self) -> None:
        super().validate()
        if self.feature_surface_mode not in {"neck", "cross_stage"}:
            raise ValueError("feature_surface_mode must be neck or cross_stage")
        if self.spatial_support_mode not in {
            "global_energy",
            "object_energy",
        }:
            raise ValueError(
                "spatial_support_mode must be global_energy or object_energy"
            )


class SVFTAFeatureSupportDeletion(SVFTAMinimalAnchor):
    """Frozen complete-panel feature-surface by spatial-support factorial."""

    implementation_path = "src/lgp/attacks/svfta_feature_support_deletion.py"

    def _object_mask(
        self,
        clean: torch.Tensor,
        boxes: torch.Tensor,
        weighting: str,
    ) -> torch.Tensor:
        if self.config.spatial_support_mode == "global_energy":
            return torch.ones(
                (clean.shape[0], 1, clean.shape[-2], clean.shape[-1]),
                device=clean.device,
                dtype=clean.dtype,
            )
        return SVFTA._object_mask(clean, boxes, weighting)

    def _feature_loss(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        object_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        loss, stats = super()._feature_loss(
            clean,
            adversarial,
            object_mask,
            generator,
        )
        stats["feature_surface_mode"] = self.config.feature_surface_mode
        stats["spatial_support_mode"] = self.config.spatial_support_mode
        return loss, stats

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        return super().__call__(
            clean_bgr,
            boxes_xyxy,
            labels=labels,
            seed=seed,
        )

