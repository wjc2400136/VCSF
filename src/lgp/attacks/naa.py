from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    positive_proposal_classification_scores,
    project_linf,
    validate_linf_constraint,
)


@dataclass(frozen=True)
class NAAConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 12
    learning_rate: float = 0.5
    target_module: str = "layer3.2"
    integration_steps: int = 8
    keeping_rate: float = 0.40
    positive_iou_threshold: float = 0.50

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "NAAConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf_constraint(self.eps, self.iterations)
        if self.learning_rate <= 0.0:
            raise ValueError("NAA learning_rate must be positive")
        if not self.target_module.strip():
            raise ValueError("NAA target_module must not be empty")
        if self.integration_steps <= 0:
            raise ValueError("integration_steps must be positive")
        if not 0.0 < self.keeping_rate <= 1.0:
            raise ValueError("keeping_rate must lie in (0, 1]")
        if not 0.0 <= self.positive_iou_threshold <= 1.0:
            raise ValueError("positive_iou_threshold must lie in [0, 1]")


class NAA:
    """Neuron Attribution Attack adapted to detector proposal scores."""

    def __init__(self, adapter: OpenMMLabAdapter, config: NAAConfig) -> None:
        config.validate()
        if (
            getattr(adapter.model, "rpn_head", None) is None
            or getattr(adapter.model, "roi_head", None) is None
        ):
            raise ValueError(
                "NAA requires the paper's two-stage proposal classification "
                "pipeline; dense/query sources are structurally unsupported"
            )
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.integration_steps + config.iterations
        self.auxiliary_forward_passes_per_image = 2

    def _attribution(
        self,
        clean: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        baseline = torch.zeros_like(clean)
        with torch.no_grad():
            baseline_feature = self.adapter.extract_backbone_module_feature(
                baseline, self.config.target_module
            )
            clean_feature = self.adapter.extract_backbone_module_feature(
                clean, self.config.target_module
            )
        integrated: Optional[torch.Tensor] = None
        for index in range(1, self.config.integration_steps + 1):
            alpha = float(index) / self.config.integration_steps
            point = (baseline + alpha * (clean - baseline)).detach().requires_grad_(True)
            target, candidates = self.adapter.backbone_module_feature_and_candidates(
                point, self.config.target_module
            )
            candidate = candidates[0]
            positive_scores = positive_proposal_classification_scores(
                candidate,
                boxes,
                labels,
                self.config.positive_iou_threshold,
            )
            # NAA sums every positive proposal's true-class score.  If a
            # virtual black-baseline image has no positive proposal, retain a
            # zero graph rather than silently substituting top-k detections.
            decision = (
                positive_scores.sum()
                if positive_scores.numel()
                else candidate.scores.sum() * 0.0
            )
            gradient = torch.autograd.grad(decision, target)[0].detach()
            integrated = gradient if integrated is None else integrated + gradient
        integrated = integrated / float(self.config.integration_steps)
        attribution = (clean_feature - baseline_feature) * integrated
        flat = attribution.flatten()
        keep = max(1, int(round(flat.numel() * self.config.keeping_rate)))
        threshold = flat.topk(keep).values.min()
        mask = (attribution >= threshold).to(attribution.dtype)
        return integrated.detach(), mask.detach(), baseline_feature.detach()

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del seed
        clean = clean_bgr.to(self.adapter.device)
        gt_boxes = boxes.to(clean.device)
        gt_labels = labels.to(clean.device) if labels is not None else None
        integrated, mask, baseline_feature = self._attribution(
            clean, gt_boxes, gt_labels
        )
        adversarial = clean.detach().clone()
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            feature = self.adapter.extract_backbone_module_feature(
                adversarial, self.config.target_module
            )
            weighted = (feature - baseline_feature.detach()) * integrated * mask
            loss = torch.sqrt(weighted.square().sum() + 1.0e-12)
            gradient = torch.autograd.grad(loss, adversarial)[0]
            # Algorithm 1 normalizes the intermediate perturbation by its
            # L-infinity norm before accumulating it, not by an L2 norm.
            norm = gradient.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1)
            direction = gradient / norm.clamp(min=1.0e-12)
            adversarial = project_linf(
                clean,
                adversarial - self.config.learning_rate * 255.0 * direction,
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    target_module=self.config.target_module,
                    keeping_rate=self.config.keeping_rate,
                    positive_iou_threshold=self.config.positive_iou_threshold,
                    learning_rate=self.config.learning_rate,
                    total_gradient_image_equivalents=self.config.integration_steps + step + 1,
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
