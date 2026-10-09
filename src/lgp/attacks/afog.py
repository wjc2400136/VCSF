from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import config_from_mapping, diagnostics_record, make_generator, validate_linf


@dataclass(frozen=True)
class AFOGConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 20
    step_size: float = 1.0 / 255.0
    attention_step_size: float = 0.10
    learn_attention: bool = True
    mode: str = "baseline"
    random_start: bool = True
    normalization_eps: float = 1.0e-8
    prediction_threshold: float = 1.0 / 2.0

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "AFOGConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.attention_step_size <= 0.0:
            raise ValueError("attention_step_size must be positive")
        if self.mode not in {"baseline", "vanishing", "fabrication"}:
            raise ValueError("AFOG mode must be baseline, vanishing, or fabrication")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must lie in [0, 1]")


class AFOG:
    """AFOG Algorithm 1 with independent attention and perturbation states.

    The public implementation overwrites ``P`` with ``A * P`` after every
    iteration.  The paper instead updates ``A_k`` and ``P_k`` independently
    and applies their Hadamard product only when constructing ``x_adv``.  The
    latter is used here; otherwise the attention is unintentionally applied
    repeatedly and the implemented recurrence is no longer Algorithm 1.  If
    the configured clean-prediction threshold yields no object, the exact
    empty target is retained and the detector-native background loss is used,
    matching the executable release behavior without lowering the threshold
    or substituting ground-truth labels.
    """

    def __init__(self, adapter: OpenMMLabAdapter, config: AFOGConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del boxes, labels
        if self.config.mode == "fabrication":
            raise NotImplementedError(
                "AFOG-F requires all unthresholded predictions with confidence targets "
                "set to one; it is not approximated by a foreground mean"
            )
        clean = clean_bgr.to(self.adapter.device)
        generator = make_generator(clean.device, seed)
        radius = self.config.eps * 255.0
        perturbation = torch.zeros_like(clean)
        if self.config.random_start:
            perturbation.uniform_(-radius, radius, generator=generator)
        attention = torch.ones_like(clean)
        diagnostics = []
        if self.config.mode == "baseline":
            target_boxes, target_labels = self.adapter.pseudo_targets(
                clean, self.config.prediction_threshold
            )
            empty_clean_targets = not any(value.numel() for value in target_boxes)
            minimize = False
        else:
            target_boxes = [clean.new_zeros((0, 4))]
            target_labels = [
                torch.zeros((0,), dtype=torch.long, device=clean.device)
            ]
            empty_clean_targets = True
            minimize = True
        for step in range(self.config.iterations):
            effective = (attention * perturbation).clamp(-radius, radius)
            adversarial = (clean + effective).clamp(0.0, 255.0).detach()
            adversarial.requires_grad_(True)
            losses = self.adapter.detection_losses(
                adversarial, target_boxes, target_labels
            )
            objective = self.adapter.afog_loss_total(losses)
            input_gradient = torch.autograd.grad(objective, adversarial)[0]
            perturbation_gradient = input_gradient * attention
            attention_gradient = input_gradient * perturbation
            sign = -1.0 if minimize else 1.0
            perturbation = perturbation + (
                sign * self.config.step_size * 255.0 * perturbation_gradient.sign()
            )
            # Algorithm 1 projects x + A*P, not P itself.  Keeping P
            # independent is essential because its current value also defines
            # the attention gradient in the next iteration.
            perturbation = perturbation.detach()
            if self.config.learn_attention:
                centered = attention_gradient - attention_gradient.mean(
                    dim=(1, 2, 3), keepdim=True
                )
                scale = centered.square().mean(
                    dim=(1, 2, 3), keepdim=True
                ).sqrt().clamp(min=self.config.normalization_eps)
                attention = (
                    attention
                    + sign * self.config.attention_step_size * centered / scale
                ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    objective,
                    mode=self.config.mode,
                    learn_attention=self.config.learn_attention,
                    attention_mean=attention.mean(),
                    attention_std=attention.std(unbiased=False),
                    objective_surface="published_bbox_plus_class_loss",
                    pseudo_targets=sum(value.shape[0] for value in target_boxes),
                    empty_clean_targets=empty_clean_targets,
                    empty_target_behavior=(
                        "detector_native_background_loss"
                        if empty_clean_targets
                        else "not_applicable"
                    ),
                )
            )
        final_perturbation = (attention * perturbation).clamp(-radius, radius)
        adversarial = (clean + final_perturbation).clamp(0.0, 255.0).detach()
        final_perturbation = adversarial - clean
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=final_perturbation.detach(),
            diagnostics=diagnostics,
        )
