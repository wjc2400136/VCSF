from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    config_from_mapping,
    diagnostics_record,
    make_generator,
    random_linf_start,
    sign_step,
    validate_linf,
)


@dataclass(frozen=True)
class TOGConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 20
    step_size: float = 1.0 / 255.0
    mode: str = "vanishing"
    random_start: bool = True
    prediction_threshold: float = 1.0 / 5.0

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "TOGConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if self.mode not in {"vanishing", "untargeted", "fabrication"}:
            raise ValueError("TOG mode must be vanishing, untargeted, or fabrication")
        if not 0.0 <= self.prediction_threshold <= 1.0:
            raise ValueError("prediction_threshold must lie in [0, 1]")


class TOG:
    """TOG using each detector's native loss and the released update rule.

    Vanishing trains the frozen victim toward an empty target. Untargeted TOG
    freezes clean post-NMS detections and maximizes the native detection loss.
    These are distinct published objectives and must never be collapsed into a
    shared foreground-score surrogate.
    """

    def __init__(self, adapter: OpenMMLabAdapter, config: TOGConfig) -> None:
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
                "TOG fabrication needs detector-specific positive objectness targets; "
                "the unified adapter intentionally does not approximate that objective"
            )
        clean = clean_bgr.to(self.adapter.device)
        generator = make_generator(clean.device, seed)
        adversarial = (
            random_linf_start(clean, self.config.eps, generator)
            if self.config.random_start
            else clean.detach().clone()
        )
        diagnostics = []
        if self.config.mode == "untargeted":
            target_boxes, target_labels = self.adapter.pseudo_targets(
                clean, self.config.prediction_threshold
            )
            if not any(value.numel() for value in target_boxes):
                raise RuntimeError(
                    "TOG untargeted found no clean detection above the configured threshold"
                )
            minimize = False
        else:
            target_boxes = [clean.new_zeros((0, 4))]
            target_labels = [
                torch.zeros((0,), dtype=torch.long, device=clean.device)
            ]
            minimize = True
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            losses = self.adapter.detection_losses(
                adversarial, target_boxes, target_labels
            )
            objective = self.adapter.detection_loss_total(losses)
            gradient = torch.autograd.grad(objective, adversarial)[0]
            adversarial = sign_step(
                clean,
                adversarial,
                gradient,
                self.config.step_size,
                self.config.eps,
                minimize=minimize,
            )
            diagnostics.append(
                diagnostics_record(
                    step,
                    objective,
                    mode=self.config.mode,
                    objective_surface="native_detector_loss",
                    pseudo_targets=sum(value.shape[0] for value in target_boxes),
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial.detach(),
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
