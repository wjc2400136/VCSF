from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    normalized_gradient,
    pairwise_iou,
    positive_proposal_classification_scores,
    project_linf,
    validate_linf,
)


@dataclass(frozen=True)
class MLFAdvConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 9
    step_size: float = 1.0 / 255.0
    target_layer: int = -2
    auxiliary_target_layer: int = -3
    low_frequency_radius_pixels: int = 200
    low_frequency_weight: float = 1.0
    feature_weight: float = 10.0
    momentum: float = 1.0
    prediction_threshold: float = 1.0 / 2.0
    positive_iou_threshold: float = 1.0 / 2.0
    use_gradcam: bool = True

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "MLFAdvConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.low_frequency_radius_pixels <= 0:
            raise ValueError("low_frequency_radius_pixels must be positive")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must lie in [0, 1]")
        if not 0.0 <= self.positive_iou_threshold <= 1.0:
            raise ValueError("positive_iou_threshold must lie in [0, 1]")


class MLFAdv:
    """Feature-weight-masked low-frequency detector attack."""

    def __init__(self, adapter: OpenMMLabAdapter, config: MLFAdvConfig) -> None:
        config.validate()
        if (
            getattr(adapter.model, "rpn_head", None) is None
            or getattr(adapter.model, "roi_head", None) is None
        ):
            raise ValueError(
                "MLFAdv requires the paper's RPN proposal and ROI classification "
                "pipeline; dense/query sources are structurally unsupported"
            )
        self.adapter = adapter
        self.config = config
        # Each update differentiates a full-resolution and a low-frequency view.
        # Per-object Grad-CAM reuses one detector graph and stops at the target
        # feature; those head-only backwards are measured separately at runtime.
        self.gradient_evaluations_per_image = 2 * config.iterations
        self.auxiliary_forward_passes_per_image = int(config.use_gradcam)
        self.auxiliary_backward_passes_per_image = 0

    def _low_frequency(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        spectrum = torch.fft.fftshift(torch.fft.fft2(image), dim=(-2, -1))
        y = torch.arange(height, device=image.device, dtype=image.dtype) - height / 2
        x = torch.arange(width, device=image.device, dtype=image.dtype) - width / 2
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = float(self.config.low_frequency_radius_pixels)
        mask = (xx.square() + yy.square() <= radius * radius).to(image.dtype)
        filtered = torch.fft.ifftshift(spectrum * mask, dim=(-2, -1))
        return torch.fft.ifft2(filtered).real

    def _gradcam_mask(
        self,
        clean: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        point = clean.detach().requires_grad_(True)
        features, candidates = self.adapter.backbone_features_and_candidates(point)
        target = features[self.config.target_layer]
        candidate = candidates[0]
        heatmaps = []
        backward_count = 0
        for box, label in zip(boxes, labels):
            overlaps = pairwise_iou(box[None, :], candidate.bboxes)[0]
            eligible = overlaps >= self.config.positive_iou_threshold
            eligible = eligible & (candidate.labels == label)
            if not eligible.any():
                eligible = overlaps >= self.config.positive_iou_threshold
            if not eligible.any():
                continue
            local = torch.nonzero(eligible, as_tuple=False).flatten()
            index = local[overlaps[local].argmax()]
            if candidate.class_scores is not None:
                foreground_count = candidate.class_scores.shape[-1]
                if candidate.background_is_explicit:
                    foreground_count = max(foreground_count - 1, 1)
                class_index = label.clamp(min=0, max=foreground_count - 1)
                decision = candidate.class_scores[index, class_index]
            else:
                decision = candidate.scores[index]
            gradient = torch.autograd.grad(
                decision, target, retain_graph=True, allow_unused=True
            )[0]
            backward_count += 1
            if gradient is None:
                continue
            channel_weight = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = (channel_weight * target).sum(dim=1, keepdim=True).relu()
            heatmaps.append(
                F.interpolate(
                    heatmap,
                    size=clean.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        self._gradcam_object_count = len(heatmaps)
        self.auxiliary_backward_passes_per_image = backward_count
        if not heatmaps:
            # A detector with no clean semantic object has no paper-defined
            # Grad-CAM mask; keep the attack executable and record the fallback.
            return torch.ones_like(clean[:, :1])
        heatmap = torch.stack(heatmaps, dim=0).sum(dim=0)
        minimum = heatmap.amin(dim=(2, 3), keepdim=True)
        maximum = heatmap.amax(dim=(2, 3), keepdim=True)
        return ((heatmap - minimum) / (maximum - minimum).clamp(min=1.0e-8)).detach()

    def _view_loss(
        self,
        image: torch.Tensor,
        clean_features: tuple,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        features, candidates = self.adapter.backbone_features_and_candidates(image)
        candidate = candidates[0]
        selected = positive_proposal_classification_scores(
            candidate,
            boxes,
            labels,
            self.config.positive_iou_threshold,
        )
        if selected.numel():
            confidence = F.binary_cross_entropy(
                selected.clamp(min=1.0e-8, max=1.0 - 1.0e-8),
                torch.zeros_like(selected),
                reduction="sum",
            )
        else:
            confidence = candidate.scores.sum() * 0.0
        similarities = []
        for level in (self.config.auxiliary_target_layer, self.config.target_layer):
            if -len(features) <= level < len(features):
                target = features[level]
                clean_feature = clean_features[level]
                similarities.append(
                    F.cosine_similarity(
                        clean_feature.detach().flatten(1),
                        target.flatten(1),
                        dim=1,
                    ).mean()
                )
        if not similarities:
            raise RuntimeError("MLFAdv target feature layers are unavailable")
        feature_similarity = torch.stack(similarities).mean()
        return confidence + self.config.feature_weight * feature_similarity

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
        predicted_boxes, predicted_labels = self.adapter.pseudo_targets(
            clean, self.config.prediction_threshold
        )
        reference_boxes = predicted_boxes[0].to(clean.device)
        reference_labels = predicted_labels[0].to(clean.device)
        if reference_boxes.numel() == 0:
            reference_boxes = gt_boxes
            if gt_labels is None:
                raise RuntimeError("MLFAdv needs clean detections or ground-truth labels")
            reference_labels = gt_labels
        mask = (
            self._gradcam_mask(clean, reference_boxes, reference_labels)
            if self.config.use_gradcam
            else torch.ones_like(clean[:, :1])
        )
        with torch.no_grad():
            clean_features = self.adapter.extract_backbone_features(clean)
            clean_low = self._low_frequency(clean)
            clean_low_features = self.adapter.extract_backbone_features(clean_low)
        adversarial = clean.detach().clone()
        momentum: Optional[torch.Tensor] = None
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            full_loss = self._view_loss(
                adversarial,
                clean_features,
                reference_boxes,
                reference_labels,
            )
            low_adversarial = self._low_frequency(adversarial)
            low_loss = self._view_loss(
                low_adversarial,
                clean_low_features,
                reference_boxes,
                reference_labels,
            )
            loss = full_loss + self.config.low_frequency_weight * low_loss
            gradient = torch.autograd.grad(loss, adversarial)[0]
            normalized = normalized_gradient(gradient)
            momentum = (
                normalized
                if momentum is None
                else self.config.momentum * momentum + normalized
            )
            adversarial = project_linf(
                clean,
                adversarial
                - self.config.step_size * 255.0 * momentum.sign() * mask,
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    full_loss=full_loss,
                    low_frequency_loss=low_loss,
                    gradcam_coverage=(mask > 0.1).float().mean(),
                    gradcam_objects=getattr(self, "_gradcam_object_count", 0),
                    gradcam_head_backwards=self.auxiliary_backward_passes_per_image,
                    total_gradient_image_equivalents=2 * (step + 1),
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
