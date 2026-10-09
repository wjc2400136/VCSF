from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    make_generator,
    project_linf,
    validate_linf,
)


@dataclass(frozen=True)
class AugTransConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 10
    step_size: float = 1.0 / 255.0
    eot_samples: int = 2
    base_rotation_degrees: float = 8.0
    curriculum_scale: float = 0.5
    enable_rotation: bool = True
    enable_resize: bool = True
    enable_noise: bool = True
    dominant_objects: int = 3
    scale_modulation_min: float = -0.2
    scale_modulation_max: float = 0.5
    aspect_jitter: float = 0.1
    gaussian_sigma: float = 0.02
    salt_pepper_probability: float = 0.01
    roi_cls_weight: float = 1.0
    roi_box_weight: float = 1.0
    rpn_objectness_weight: float = 2.0
    rpn_box_weight: float = 1.0
    cls_exponent: float = 0.8
    objectness_exponent: float = 0.8
    target_source: str = "ground_truth"
    prediction_threshold: float = 1.0 / 2.0

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "AugTransConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.eot_samples <= 0:
            raise ValueError("eot_samples must be positive")
        if self.dominant_objects <= 0:
            raise ValueError("dominant_objects must be positive")
        if self.target_source not in {"ground_truth", "predicted"}:
            raise ValueError("target_source must be ground_truth or predicted")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must lie in [0, 1]")


class AugTrans:
    """Dynamic object-aware EOT attack with the full four-branch objective."""

    def __init__(self, adapter: OpenMMLabAdapter, config: AugTransConfig) -> None:
        config.validate()
        if (
            getattr(adapter.model, "rpn_head", None) is None
            or getattr(adapter.model, "roi_head", None) is None
        ):
            raise ValueError(
                "AugTrans requires separate RPN and ROI heads; this source is structurally unsupported"
            )
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations * config.eot_samples

    @staticmethod
    def _rotate_boxes(
        boxes: torch.Tensor,
        angle_degrees: float,
        center: Tuple[float, float],
        height: int,
        width: int,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        corners = torch.stack(
            (
                torch.stack((x1, y1), dim=1),
                torch.stack((x2, y1), dim=1),
                torch.stack((x2, y2), dim=1),
                torch.stack((x1, y2), dim=1),
            ),
            dim=1,
        )
        radians = math.radians(angle_degrees)
        cosine, sine = math.cos(radians), math.sin(radians)
        shifted_x = corners[..., 0] - center[0]
        shifted_y = corners[..., 1] - center[1]
        rotated_x = cosine * shifted_x + sine * shifted_y + center[0]
        rotated_y = -sine * shifted_x + cosine * shifted_y + center[1]
        output = torch.stack(
            (
                rotated_x.min(dim=1).values,
                rotated_y.min(dim=1).values,
                rotated_x.max(dim=1).values,
                rotated_y.max(dim=1).values,
            ),
            dim=1,
        )
        output[:, 0::2].clamp_(0, width)
        output[:, 1::2].clamp_(0, height)
        return output

    def _sample_center(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Tuple[float, float]:
        options = 3 if boxes.numel() else 2
        choice = int(
            torch.randint(0, options, (1,), device=boxes.device, generator=generator).item()
        )
        if choice == 0:
            return width * 0.5, height * 0.5
        if choice == 1:
            point = torch.rand((2,), device=boxes.device, generator=generator)
            return float(point[0].item() * width), float(point[1].item() * height)
        index = int(
            torch.randint(0, boxes.shape[0], (1,), device=boxes.device, generator=generator).item()
        )
        center = (boxes[index, :2] + boxes[index, 2:]) * 0.5
        return float(center[0].item()), float(center[1].item())

    def _content_scale(
        self,
        boxes: torch.Tensor,
        height: int,
        width: int,
        generator: torch.Generator,
    ) -> Tuple[float, float]:
        if boxes.numel():
            sizes = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1.0)
            areas = sizes.prod(dim=1)
            count = min(self.config.dominant_objects, boxes.shape[0])
            dominant = sizes[areas.topk(count).indices]
            relative_h = float(dominant[:, 1].mean().item() / height)
            relative_w = float(dominant[:, 0].mean().item() / width)
        else:
            relative_h = relative_w = 0.1
        modulation = torch.empty((2,), device=boxes.device).uniform_(
            self.config.scale_modulation_min,
            self.config.scale_modulation_max,
            generator=generator,
        )
        jitter = torch.empty((2,), device=boxes.device).uniform_(
            -self.config.aspect_jitter,
            self.config.aspect_jitter,
            generator=generator,
        )
        scale_h = (1.0 + relative_h * float(modulation[0].item())) * (
            1.0 + float(jitter[0].item())
        )
        scale_w = (1.0 + relative_w * float(modulation[1].item())) * (
            1.0 + float(jitter[1].item())
        )
        return max(scale_h, 0.5), max(scale_w, 0.5)

    def _resize_restore(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        scale_h: float,
        scale_w: float,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = image.shape
        new_height = max(2, int(round(height * scale_h)))
        new_width = max(2, int(round(width * scale_w)))
        transformed = F.interpolate(
            image, size=(new_height, new_width), mode="bilinear", align_corners=False
        )
        output_boxes = boxes * boxes.new_tensor((scale_w, scale_h, scale_w, scale_h))
        if new_height >= height:
            top = int(
                torch.randint(0, new_height - height + 1, (1,), device=image.device, generator=generator).item()
            )
            transformed = transformed[..., top : top + height, :]
            output_boxes[:, 1::2] -= top
        else:
            remaining = height - new_height
            top = int(
                torch.randint(0, remaining + 1, (1,), device=image.device, generator=generator).item()
            )
            pad_mode = "reflect" if max(top, remaining - top) < new_height else "replicate"
            transformed = F.pad(
                transformed, (0, 0, top, remaining - top), mode=pad_mode
            )
            output_boxes[:, 1::2] += top
        if new_width >= width:
            left = int(
                torch.randint(0, new_width - width + 1, (1,), device=image.device, generator=generator).item()
            )
            transformed = transformed[..., :, left : left + width]
            output_boxes[:, 0::2] -= left
        else:
            remaining = width - new_width
            left = int(
                torch.randint(0, remaining + 1, (1,), device=image.device, generator=generator).item()
            )
            pad_mode = "reflect" if max(left, remaining - left) < new_width else "replicate"
            transformed = F.pad(
                transformed, (left, remaining - left, 0, 0), mode=pad_mode
            )
            output_boxes[:, 0::2] += left
        output_boxes[:, 0::2].clamp_(0, width)
        output_boxes[:, 1::2].clamp_(0, height)
        valid = (output_boxes[:, 2] - output_boxes[:, 0] > 1) & (
            output_boxes[:, 3] - output_boxes[:, 1] > 1
        )
        return transformed, output_boxes[valid], valid

    def _augment(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        step: int,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = image.shape
        progress = float(step) / max(self.config.iterations, 1)
        transformed = image
        transformed_boxes = boxes
        if self.config.enable_rotation:
            maximum = self.config.base_rotation_degrees * (
                1.0 + self.config.curriculum_scale * progress
            )
            angle = float(
                torch.empty((), device=image.device).uniform_(
                    -maximum, maximum, generator=generator
                ).item()
            )
            center = self._sample_center(boxes, height, width, generator)
            transformed = rotate(
                transformed,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                center=[center[0], center[1]],
                fill=0.0,
            )
            transformed_boxes = self._rotate_boxes(
                transformed_boxes, angle, center, height, width
            )
        if self.config.enable_resize:
            scale_h, scale_w = self._content_scale(
                transformed_boxes, height, width, generator
            )
            transformed, transformed_boxes, valid = self._resize_restore(
                transformed, transformed_boxes, scale_h, scale_w, generator
            )
            labels = labels[valid]
        if self.config.enable_noise:
            gaussian = torch.randn(
                transformed.shape,
                device=transformed.device,
                dtype=transformed.dtype,
                generator=generator,
            ) * (self.config.gaussian_sigma * 255.0)
            noisy = transformed + gaussian
            salt_pepper = torch.rand(
                transformed.shape,
                device=transformed.device,
                dtype=transformed.dtype,
                generator=generator,
            )
            corrupted = torch.where(
                salt_pepper < self.config.salt_pepper_probability / 2,
                0.0,
                noisy,
            )
            corrupted = torch.where(
                salt_pepper > 1.0 - self.config.salt_pepper_probability / 2,
                255.0,
                corrupted,
            )
            # Straight-through estimator for the discrete salt-and-pepper assignment.
            transformed = noisy + (corrupted - noisy).detach()
        return transformed.clamp(0.0, 255.0), transformed_boxes, labels

    def _objective(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        zero = next(iter(losses.values())).new_zeros(())
        roi_cls = zero
        roi_box = zero
        rpn_objectness = zero
        rpn_box = zero
        for name, value in losses.items():
            lowered = name.lower()
            if "rpn" in lowered and ("bbox" in lowered or "box" in lowered):
                rpn_box = rpn_box + value
            elif "rpn" in lowered and ("cls" in lowered or "object" in lowered):
                rpn_objectness = rpn_objectness + value
            elif "bbox" in lowered or "box" in lowered:
                roi_box = roi_box + value
            elif "cls" in lowered:
                roi_cls = roi_cls + value
        objective = (
            self.config.roi_cls_weight
            * roi_cls.clamp(min=1.0e-12).pow(self.config.cls_exponent)
            + self.config.roi_box_weight * roi_box
            + self.config.rpn_objectness_weight
            * rpn_objectness.clamp(min=1.0e-12).pow(
                self.config.objectness_exponent
            )
            + self.config.rpn_box_weight * rpn_box
        )
        return objective, {
            "roi_cls": roi_cls,
            "roi_box": roi_box,
            "rpn_objectness": rpn_objectness,
            "rpn_box": rpn_box,
        }

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        clean = clean_bgr.to(self.adapter.device)
        if self.config.target_source == "predicted":
            predicted_boxes, predicted_labels = self.adapter.pseudo_targets(
                clean, self.config.prediction_threshold
            )
            gt_boxes = predicted_boxes[0]
            gt_labels = predicted_labels[0]
            if gt_boxes.numel() == 0:
                raise RuntimeError(
                    "AugTrans found no clean prediction above the configured threshold"
                )
        else:
            if labels is None:
                raise ValueError("AugTrans requires ground-truth labels")
            gt_boxes = boxes.to(clean.device)
            gt_labels = labels.to(clean.device)
        generator = make_generator(clean.device, seed)
        adversarial = clean.detach().clone()
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            views: List[torch.Tensor] = []
            view_boxes: List[torch.Tensor] = []
            view_labels: List[torch.Tensor] = []
            for _ in range(self.config.eot_samples):
                view, transformed_boxes, transformed_labels = self._augment(
                    adversarial, gt_boxes, gt_labels, step, generator
                )
                views.append(view)
                view_boxes.append(transformed_boxes)
                view_labels.append(transformed_labels)
            losses = self.adapter.detection_losses(
                torch.cat(views, dim=0), view_boxes, view_labels
            )
            objective, terms = self._objective(losses)
            gradient = torch.autograd.grad(objective, adversarial)[0]
            adversarial = project_linf(
                clean,
                adversarial + self.config.step_size * 255.0 * gradient.sign(),
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    objective,
                    **terms,
                    eot_samples=self.config.eot_samples,
                    target_source=self.config.target_source,
                    gradient_image_equivalents=(step + 1) * self.config.eot_samples,
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
