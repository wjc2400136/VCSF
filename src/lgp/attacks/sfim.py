from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    feature_distance,
    make_generator,
    project_linf,
    validate_linf_constraint,
)


def _dct_1d(value: torch.Tensor) -> torch.Tensor:
    length = value.shape[-1]
    reordered = torch.cat((value[..., ::2], value[..., 1::2].flip(-1)), dim=-1)
    spectrum = torch.fft.fft(reordered, dim=-1)
    index = torch.arange(length, device=value.device, dtype=value.dtype)
    angle = -math.pi * index / (2.0 * length)
    transformed = 2.0 * (
        spectrum.real * angle.cos() - spectrum.imag * angle.sin()
    )
    transformed[..., 0] /= math.sqrt(length) * 2.0
    if length > 1:
        transformed[..., 1:] /= math.sqrt(length / 2.0) * 2.0
    return transformed


def _idct_1d(value: torch.Tensor) -> torch.Tensor:
    length = value.shape[-1]
    scaled = value.clone()
    scaled[..., 0] *= math.sqrt(length) * 2.0
    if length > 1:
        scaled[..., 1:] *= math.sqrt(length / 2.0) * 2.0
    scaled = scaled / 2.0
    index = torch.arange(length, device=value.device, dtype=value.dtype)
    angle = math.pi * index / (2.0 * length)
    imaginary = torch.cat(
        (torch.zeros_like(scaled[..., :1]), -scaled.flip(-1)[..., :-1]), dim=-1
    )
    real = scaled * angle.cos() - imaginary * angle.sin()
    imag = scaled * angle.sin() + imaginary * angle.cos()
    reordered = torch.fft.ifft(torch.complex(real, imag), dim=-1).real
    output = torch.zeros_like(reordered)
    output[..., ::2] = reordered[..., : length - length // 2]
    output[..., 1::2] = reordered.flip(-1)[..., : length // 2]
    return output


def dct2(value: torch.Tensor) -> torch.Tensor:
    return _dct_1d(_dct_1d(value).transpose(-1, -2)).transpose(-1, -2)


def idct2(value: torch.Tensor) -> torch.Tensor:
    return _idct_1d(_idct_1d(value).transpose(-1, -2)).transpose(-1, -2)


@dataclass(frozen=True)
class SFIMConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 10
    learning_rate: float = 0.5
    feature_levels: Tuple[int, ...] = (0, 1, 2, 3)
    spatial_scale_upper: float = 0.30
    frequency_scale_upper: float = 0.80
    spatial_scale_lower: float = 0.10
    frequency_scale_lower: float = 0.30
    spatial_weight: float = 1.0
    frequency_weight: float = 1.0
    spatial_views: int = 1
    frequency_views: int = 1

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "SFIMConfig":
        converted = dict(raw)
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(converted["feature_levels"])
        return config_from_mapping(cls, converted)

    def validate(self) -> None:
        validate_linf_constraint(self.eps, self.iterations)
        if self.learning_rate <= 0.0:
            raise ValueError("SFIM learning_rate must be positive")
        if not 0.0 < self.spatial_scale_lower < self.spatial_scale_upper <= 1.0:
            raise ValueError("Invalid SFIM spatial masking interval")
        if not 0.0 < self.frequency_scale_lower < self.frequency_scale_upper <= 1.0:
            raise ValueError("Invalid SFIM frequency masking interval")
        if self.spatial_weight < 0.0 or self.frequency_weight < 0.0:
            raise ValueError("SFIM branch weights must be non-negative")
        if self.spatial_views < 0 or self.frequency_views < 0:
            raise ValueError("SFIM view counts must be non-negative")
        if self.spatial_views + self.frequency_views <= 0:
            raise ValueError("At least one SFIM view must remain enabled")
        if (
            self.spatial_views > 0
            and self.spatial_weight <= 0.0
            or self.frequency_views > 0
            and self.frequency_weight <= 0.0
        ):
            raise ValueError("Every enabled SFIM branch must have a positive weight")
        if (
            self.spatial_views == 0
            and self.frequency_views == 0
        ):
            raise ValueError("At least one SFIM branch must remain enabled")


class SFIM:
    """Backbone branch of Spatial-Frequency Information Masking (SFIM-B)."""

    def __init__(self, adapter: OpenMMLabAdapter, config: SFIMConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations * (
            config.spatial_views + config.frequency_views
        )

    def _spatial_mask(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        mask = torch.ones_like(image[:, :1])
        height, width = image.shape[-2:]
        for box in boxes.detach():
            box_width = max(int((box[2] - box[0]).item()), 1)
            box_height = max(int((box[3] - box[1]).item()), 1)
            scale = float(
                torch.empty((), device=image.device).uniform_(
                    self.config.spatial_scale_lower,
                    self.config.spatial_scale_upper,
                    generator=generator,
                ).item()
            )
            erase_width = max(1, int(round(box_width * scale)))
            erase_height = max(1, int(round(box_height * scale)))
            left_min = max(0, min(int(box[0].item()), width - 1))
            top_min = max(0, min(int(box[1].item()), height - 1))
            left_max = max(left_min, min(int(box[2].item()) - erase_width, width - erase_width))
            top_max = max(top_min, min(int(box[3].item()) - erase_height, height - erase_height))
            left = int(
                torch.randint(
                    left_min,
                    left_max + 1,
                    (1,),
                    device=image.device,
                    generator=generator,
                ).item()
            )
            top = int(
                torch.randint(
                    top_min,
                    top_max + 1,
                    (1,),
                    device=image.device,
                    generator=generator,
                ).item()
            )
            mask[..., top : top + erase_height, left : left + erase_width] = 0.0
        return image * mask

    def _frequency_mask(
        self, image: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        height, width = image.shape[-2:]
        scale = float(
            torch.empty((), device=image.device).uniform_(
                self.config.frequency_scale_lower,
                self.config.frequency_scale_upper,
                generator=generator,
            ).item()
        )
        keep_height = max(1, int(round(height * scale)))
        keep_width = max(1, int(round(width * scale)))
        coefficients = dct2(image)
        mask = torch.zeros_like(coefficients)
        mask[..., :keep_height, :keep_width] = 1.0
        return idct2(coefficients * mask)

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
        adversarial = clean.detach().clone()
        with torch.no_grad():
            clean_features = self.adapter.extract_backbone_features(clean)
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            views = [
                self._spatial_mask(adversarial, gt_boxes, generator)
                for _ in range(self.config.spatial_views)
            ]
            views.extend(
                self._frequency_mask(adversarial, generator)
                for _ in range(self.config.frequency_views)
            )
            joined = torch.cat(views, dim=0)
            features = self.adapter.extract_backbone_features(joined)
            offset = 0
            zero = joined.new_zeros(())
            spatial_loss = zero
            if self.config.spatial_views:
                spatial_losses = []
                for view_index in range(self.config.spatial_views):
                    branch_features = tuple(
                        value[offset + view_index : offset + view_index + 1]
                        for value in features
                    )
                    spatial_losses.append(
                        feature_distance(
                            clean_features,
                            branch_features,
                            self.config.feature_levels,
                        )
                    )
                spatial_loss = torch.stack(spatial_losses).mean()
                offset += self.config.spatial_views
            frequency_loss = zero
            if self.config.frequency_views:
                frequency_losses = []
                for view_index in range(self.config.frequency_views):
                    branch_features = tuple(
                        value[offset + view_index : offset + view_index + 1]
                        for value in features
                    )
                    frequency_losses.append(
                        feature_distance(
                            clean_features,
                            branch_features,
                            self.config.feature_levels,
                        )
                    )
                frequency_loss = torch.stack(frequency_losses).mean()
            loss = zero
            if self.config.spatial_views:
                loss = loss + self.config.spatial_weight * spatial_loss
            if self.config.frequency_views:
                loss = loss + self.config.frequency_weight * frequency_loss
            gradient = torch.autograd.grad(loss, adversarial)[0]
            flat_norm = gradient.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1)
            # Algorithm 1 uses g / ||g||_infinity followed by an ascent
            # update.  L2 normalization changes the relative branch balance
            # and is not an equivalent implementation.
            scaled = gradient / flat_norm.clamp(min=1.0e-12)
            adversarial = project_linf(
                clean,
                adversarial + self.config.learning_rate * 255.0 * scaled,
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    branches="SIMx{}+FIMx{}".format(
                        self.config.spatial_views,
                        self.config.frequency_views,
                    ),
                    spatial_loss=spatial_loss,
                    frequency_loss=frequency_loss,
                    gradient_image_equivalents=(step + 1)
                    * (self.config.spatial_views + self.config.frequency_views),
                    learning_rate=self.config.learning_rate,
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
