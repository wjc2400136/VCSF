# Contains the recorded OSFD random-rotation/Gaussian-blur adaptation.
# Source: wakuwu/OSFD@3744caf69e60b46012a6895c095f18c33db491a9.
# Original GPLv3 text: docs/third_party/licenses/GPL-3.0-upstream.txt.
# Modified for local source-detector/budget integration; notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# See THIRD_PARTY_NOTICES.md for covered scope and pending project licence.

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, rotate

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    feature_distance,
    make_generator,
    normalized_gradient,
    project_linf,
    random_linf_start,
    validate_linf,
)


@dataclass(frozen=True)
class OSFDConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 10
    step_size: float = 1.0 / 255.0
    views: int = 2
    amplification: float = 3.0
    feature_levels: Tuple[int, ...] = (0, 1, 2, 3)
    momentum: float = 1.0
    max_rotation_degrees: float = 7.0
    rotation_center_jitter: int = 10
    transform_probability: float = 1.0
    enable_rotation: bool = True
    enable_resize: bool = True
    enable_noise: bool = True
    resize_factor: float = 4.0 / 5.0
    max_scale: float = 11.0 / 10.0
    gaussian_sigma: float = 6.0
    random_start: bool = True
    initial_noise_pixels: int = 2

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "OSFDConfig":
        converted = dict(raw)
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(converted["feature_levels"])
        return config_from_mapping(cls, converted)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.views != 2:
            raise ValueError("Faithful OSFD-RRB uses exactly two views per update")
        if self.max_scale < 1.0:
            raise ValueError("max_scale must be at least one")
        if not 0.0 <= self.transform_probability <= 1.0:
            raise ValueError("transform_probability must lie in [0, 1]")
        if self.initial_noise_pixels < 0:
            raise ValueError("initial_noise_pixels must be non-negative")


class OSFD:
    """Object-aware Significant Feature Distortion with released RRB views."""

    def __init__(self, adapter: OpenMMLabAdapter, config: OSFDConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations * config.views

    def _rotation(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        if boxes.numel():
            object_centers = ((boxes[:, :2] + boxes[:, 2:]) * 0.5).detach()
            image_center = image.new_tensor([[width * 0.5, height * 0.5]])
            centers = torch.cat((object_centers, image_center), dim=0)
            choice = int(
                torch.randint(0, centers.shape[0], (1,), device=image.device, generator=generator).item()
            )
            center = centers[choice]
            jitter = self.config.rotation_center_jitter
            if jitter:
                center = center + torch.randint(
                    -jitter,
                    jitter,
                    (2,),
                    device=image.device,
                    generator=generator,
                )
            center_xy = [
                float(center[0].clamp(0, width - 1).item()),
                float(center[1].clamp(0, height - 1).item()),
            ]
        else:
            center_xy = [width * 0.5, height * 0.5]
        angle = float(
            torch.empty((), device=image.device).uniform_(
                -self.config.max_rotation_degrees,
                self.config.max_rotation_degrees,
                generator=generator,
            ).item()
        )
        return rotate(
            image,
            angle,
            interpolation=InterpolationMode.BILINEAR,
            center=center_xy,
            fill=0.0,
        )

    def _adaptive_resize(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        _, _, height, width = image.shape
        if boxes.numel():
            index = int(
                torch.randint(0, boxes.shape[0], (1,), device=image.device, generator=generator).item()
            )
            box = boxes[index]
            box_height = float((box[3] - box[1]).clamp(min=1).item())
            box_width = float((box[2] - box[0]).clamp(min=1).item())
        else:
            box_height, box_width = height * 0.25, width * 0.25
        max_h = min(1.0 + self.config.resize_factor * box_height / height, self.config.max_scale)
        max_w = min(1.0 + self.config.resize_factor * box_width / width, self.config.max_scale)
        new_height = int(
            torch.randint(
                height,
                max(height + 1, int(round(max_h * height)) + 1),
                (1,),
                device=image.device,
                generator=generator,
            ).item()
        )
        new_width = int(
            torch.randint(
                width,
                max(width + 1, int(round(max_w * width)) + 1),
                (1,),
                device=image.device,
                generator=generator,
            ).item()
        )
        resized = F.interpolate(
            image, size=(new_height, new_width), mode="bilinear", align_corners=True
        )
        canvas_height = max(new_height, int(round(max_h * height)))
        canvas_width = max(new_width, int(round(max_w * width)))
        remain_h = canvas_height - new_height
        remain_w = canvas_width - new_width
        top = int(
            torch.randint(0, remain_h + 1, (1,), device=image.device, generator=generator).item()
        )
        left = int(
            torch.randint(0, remain_w + 1, (1,), device=image.device, generator=generator).item()
        )
        padded = F.pad(
            resized,
            (left, remain_w - left, top, remain_h - top),
            mode="constant",
            value=0.0,
        )
        return F.interpolate(
            padded, size=(height, width), mode="bilinear", align_corners=True
        )

    def _views(
        self,
        adversarial: torch.Tensor,
        boxes: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        apply_rotation = self.config.enable_rotation and bool(
            torch.rand((), device=adversarial.device, generator=generator).item()
            < self.config.transform_probability
        )
        apply_resize = self.config.enable_resize and bool(
            torch.rand((), device=adversarial.device, generator=generator).item()
            < self.config.transform_probability
        )
        rotation = (
            self._rotation(adversarial, boxes, generator)
            if apply_rotation
            else adversarial
        )
        resized = (
            self._adaptive_resize(adversarial, boxes, generator)
            if apply_resize
            else adversarial
        )
        stacked = torch.cat((rotation, resized), dim=0)
        if self.config.enable_noise and self.config.gaussian_sigma > 0.0:
            noise = torch.randn(
                stacked.shape,
                device=stacked.device,
                dtype=stacked.dtype,
                generator=generator,
            ) * self.config.gaussian_sigma
            stacked = stacked + noise
        return stacked.clamp(0.0, 255.0)

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del labels
        clean = clean_bgr.to(self.adapter.device)
        gt_boxes = boxes.to(clean.device)
        generator = make_generator(clean.device, seed)
        if self.config.random_start and self.config.initial_noise_pixels:
            initial = torch.randint(
                -self.config.initial_noise_pixels,
                self.config.initial_noise_pixels + 1,
                clean.shape,
                device=clean.device,
                generator=generator,
            ).to(dtype=clean.dtype)
            adversarial = project_linf(clean, clean + initial, self.config.eps).detach()
        else:
            adversarial = clean.detach().clone()
        with torch.no_grad():
            clean_features = self.adapter.extract_backbone_features(clean)
        momentum: Optional[torch.Tensor] = None
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            transformed = self._views(adversarial, gt_boxes, generator)
            adversarial_features = self.adapter.extract_backbone_features(transformed)
            loss = feature_distance(
                clean_features,
                adversarial_features,
                self.config.feature_levels,
                amplify=self.config.amplification,
            )
            gradient = torch.autograd.grad(loss, adversarial)[0]
            normalized = normalized_gradient(gradient)
            momentum = (
                normalized
                if momentum is None
                else self.config.momentum * momentum + normalized
            )
            adversarial = project_linf(
                clean,
                adversarial + self.config.step_size * 255.0 * momentum.sign(),
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    transformed_views=self.config.views,
                    gradient_image_equivalents=(step + 1) * self.config.views,
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
