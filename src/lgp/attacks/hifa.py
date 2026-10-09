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
    make_generator,
    normalized_gradient,
    project_linf,
    validate_linf,
)


@dataclass(frozen=True)
class HIFAConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 12
    step_size: float = 1.0 / 255.0
    target_layer: int = -2
    integration_steps: int = 4
    augmented_copies: int = 2
    negative_balance: float = 1.0
    momentum: float = 1.0
    noise_sigma: float = 0.02
    transformation_probability: float = 1.0 / 2.0
    salt_pepper_probability: float = 0.01
    motion_blur_kernel_size: int = 5

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "HIFAConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.integration_steps <= 0 or self.augmented_copies <= 0:
            raise ValueError("HIFA integration_steps and augmented_copies must be positive")
        if not 0.0 <= self.transformation_probability <= 1.0:
            raise ValueError("transformation_probability must lie in [0, 1]")
        if self.noise_sigma < 0.0:
            raise ValueError("noise_sigma must be non-negative")
        if not 0.0 <= self.salt_pepper_probability <= 1.0:
            raise ValueError("salt_pepper_probability must lie in [0, 1]")
        if self.motion_blur_kernel_size < 3 or self.motion_blur_kernel_size % 2 == 0:
            raise ValueError("motion_blur_kernel_size must be an odd integer >= 3")


class HIFA:
    """High-level Interpretable Feature Attack with compute-counted DEIG."""

    def __init__(self, adapter: OpenMMLabAdapter, config: HIFAConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = (
            config.integration_steps * config.augmented_copies + config.iterations
        )

    def _transform(
        self, image: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        if (
            torch.rand((), device=image.device, generator=generator).item()
            >= self.config.transformation_probability
        ):
            return image
        transform_choice = int(
            torch.randint(0, 3, (1,), device=image.device, generator=generator).item()
        )
        if transform_choice == 0:
            kernel_size = self.config.motion_blur_kernel_size
            kernel = image.new_zeros((3, 1, kernel_size, kernel_size))
            kernel[:, 0, kernel_size // 2, :] = 1.0 / float(kernel_size)
            padded = F.pad(
                image,
                (kernel_size // 2,) * 4,
                mode="reflect",
            )
            return F.conv2d(padded, kernel, groups=3)
        if transform_choice == 1:
            noise = torch.randn(
                image.shape,
                device=image.device,
                dtype=image.dtype,
                generator=generator,
            ) * (self.config.noise_sigma * 255.0)
            return (image + noise).clamp(0.0, 255.0)
        random_values = torch.rand(
            image.shape,
            device=image.device,
            dtype=image.dtype,
            generator=generator,
        )
        salt = random_values > 1.0 - self.config.salt_pepper_probability / 2.0
        pepper = random_values < self.config.salt_pepper_probability / 2.0
        corrupted = torch.where(salt, image.new_tensor(255.0), image)
        corrupted = torch.where(pepper, image.new_zeros(()), corrupted)
        return image + (corrupted - image).detach()

    def _attribution(
        self,
        clean: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        accumulated: Optional[torch.Tensor] = None
        for _ in range(self.config.augmented_copies):
            transformed = self._transform(clean, generator).detach()
            copy_sum: Optional[torch.Tensor] = None
            for integration_index in range(1, self.config.integration_steps + 1):
                alpha = float(integration_index) / self.config.integration_steps
                point = (alpha * transformed).detach().requires_grad_(True)
                features, losses = self.adapter.backbone_features_and_losses(
                    point,
                    [boxes],
                    [labels],
                )
                target = features[self.config.target_layer]
                decision = self.adapter.detection_loss_total(losses)
                gradient = torch.autograd.grad(decision, target)[0]
                copy_sum = gradient.detach() if copy_sum is None else copy_sum + gradient.detach()
            copy_mean = copy_sum / float(self.config.integration_steps)
            accumulated = copy_mean if accumulated is None else accumulated + copy_mean
        attribution = accumulated / float(self.config.augmented_copies)
        positive = attribution.clamp(min=0.0)
        negative = attribution.clamp(max=0.0)
        return self.config.negative_balance * negative - positive

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        if labels is None:
            raise ValueError("HIFA requires ground-truth labels")
        clean = clean_bgr.to(self.adapter.device)
        gt_boxes = boxes.to(clean.device)
        gt_labels = labels.to(clean.device)
        generator = make_generator(clean.device, seed)
        weights = self._attribution(clean, gt_boxes, gt_labels, generator)
        adversarial = clean.detach().clone()
        momentum: Optional[torch.Tensor] = None
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            features = self.adapter.extract_backbone_features(adversarial)
            target = features[self.config.target_layer]
            if target.shape != weights.shape:
                weights_for_target = F.interpolate(
                    weights,
                    size=target.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                if weights_for_target.shape[1] != target.shape[1]:
                    weights_for_target = weights_for_target.mean(dim=1, keepdim=True)
            else:
                weights_for_target = weights
            loss = (weights_for_target.detach() * target).mean()
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
                    target_layer=self.config.target_layer,
                    attribution_gradient_evaluations=(
                        self.config.integration_steps * self.config.augmented_copies
                    ),
                    transform_probability=self.config.transformation_probability,
                    transform_family="motion_blur|gaussian|salt_pepper",
                    total_gradient_image_equivalents=(
                        self.config.integration_steps * self.config.augmented_copies + step + 1
                    ),
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
