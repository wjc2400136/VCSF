from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import AttackOutput
from .svfta_feature_support_deletion import (
    SVFTAFeatureSupportDeletion,
    SVFTAFeatureSupportDeletionConfig,
)


@dataclass(frozen=True)
class SVFTACrossStageClaimConfirmationConfig(
    SVFTAFeatureSupportDeletionConfig
):
    """Frozen final-condition test of the two cross-stage claim boundaries."""

    feature_surface_mode: str = "cross_stage"
    spatial_support_mode: str = "global_energy"
    scale_view_mode: str = "synchronized_scale"
    phase_boundary_mode: str = "reset"

    def validate(self) -> None:
        fixed = {
            "eps": math.isclose(self.eps, 4.0 / 255.0),
            "iterations": self.iterations == 20,
            "step_size": math.isclose(self.step_size, 1.0 / 255.0),
            "score_threshold": math.isclose(
                self.score_threshold,
                1.0 / 20.0,
            ),
            "feature_levels": tuple(self.feature_levels) == (0, 1, 2),
            "feature_eps": math.isclose(self.feature_eps, 1.0e-6),
            "feature_objective": self.feature_objective
            == "directional_cosine",
            "scale_min": math.isclose(self.scale_min, 2.0 / 3.0),
            "scale_max": math.isclose(self.scale_max, 4.0 / 3.0),
            "momentum": math.isclose(self.momentum, 7.0 / 10.0),
            "risk_warmup": self.risk_warmup == 1,
            "risk_to_feature_steps": self.risk_to_feature_steps == 0,
            "risk_objective": self.risk_objective == "smooth_log_odds",
            "object_mask_weighting": self.object_mask_weighting == "union",
            "spatial_support_mode": self.spatial_support_mode
            == "global_energy",
        }
        failed = [name for name, valid in fixed.items() if not valid]
        if failed:
            raise ValueError(
                "Cross-stage claim confirmation freezes: {}".format(
                    ", ".join(failed)
                )
            )
        if self.feature_surface_mode not in {"neck", "cross_stage"}:
            raise ValueError(
                "feature_surface_mode must be neck or cross_stage"
            )
        if self.scale_view_mode not in {
            "identity",
            "synchronized_scale",
        }:
            raise ValueError(
                "scale_view_mode must be identity or synchronized_scale"
            )
        if self.phase_boundary_mode not in {
            "feature_only",
            "reset",
            "carryover",
        }:
            raise ValueError(
                "phase_boundary_mode must be feature_only, reset or carryover"
            )
        expected_risk = self.phase_boundary_mode != "feature_only"
        if self.risk_enabled is not expected_risk:
            raise ValueError(
                "risk_enabled must match the selected phase_boundary_mode"
            )
        allowed = {
            ("neck", "identity", "reset"),
            ("neck", "synchronized_scale", "reset"),
            ("cross_stage", "identity", "reset"),
            ("cross_stage", "synchronized_scale", "reset"),
            ("cross_stage", "synchronized_scale", "feature_only"),
            ("cross_stage", "synchronized_scale", "carryover"),
        }
        resolved = (
            self.feature_surface_mode,
            self.scale_view_mode,
            self.phase_boundary_mode,
        )
        if resolved not in allowed:
            raise ValueError(
                "Only the six preregistered final-condition cells are valid"
            )


class SVFTACrossStageClaimConfirmation(SVFTAFeatureSupportDeletion):
    """Run only the six preregistered scale, surface and phase contrasts."""

    implementation_path = (
        "src/lgp/attacks/svfta_cross_stage_claim_confirmation.py"
    )

    def _same_scale_view(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.scale_view_mode == "identity":
            return clean, adversarial, mask
        return super()._same_scale_view(
            clean,
            adversarial,
            mask,
            generator,
        )

    @staticmethod
    def _empty_risk_stats(risk_objective: str) -> Dict[str, Any]:
        return {
            "score_risk": None,
            "score_risk_defined": False,
            "constraint_violation": 0.0,
            "risk_surface": "not_checked",
            "risk_objective": risk_objective,
            "risk_available_count": 0,
            "risk_aggregate": None,
        }

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del labels
        clean = clean_bgr.detach().to(
            self.adapter.device,
            dtype=torch.float32,
        ).clamp(0.0, 255.0)
        boxes = boxes_xyxy.detach().to(clean.device, dtype=clean.dtype)
        object_mask = self._object_mask(
            clean,
            boxes,
            self.config.object_mask_weighting,
        )
        epsilon = self.config.eps * 255.0
        step_size = self.config.step_size * 255.0
        noise = torch.zeros_like(clean, requires_grad=True)
        feature_momentum: Optional[torch.Tensor] = None
        diagnostics: List[Dict[str, Any]] = []
        generator = torch.Generator(device=clean.device)
        generator.manual_seed(int(seed))
        phase_mode = self.config.phase_boundary_mode
        risk_updates = 0

        for step in range(self.config.iterations):
            adversarial = (clean + noise).clamp(0.0, 255.0)
            use_risk = phase_mode != "feature_only" and step == 0
            if use_risk:
                total_loss, risk_stats = self._risk_loss(adversarial)
                if total_loss is None:
                    raise RuntimeError(
                        "Smooth log-odds initialization exposed no gradient"
                    )
                risk_updates = 1
                feature_stats = self._empty_feature_stats(
                    self.config.include_backbone_surface,
                    float(object_mask.mean().item()),
                )
                feature_stats.update(
                    {
                        "feature_surface_mode": (
                            self.config.feature_surface_mode
                        ),
                        "spatial_support_mode": (
                            self.config.spatial_support_mode
                        ),
                    }
                )
                phase = "risk_initialization"
            else:
                total_loss, feature_stats = self._feature_loss(
                    clean,
                    adversarial,
                    object_mask,
                    generator,
                )
                risk_stats = self._empty_risk_stats(
                    self.config.risk_objective
                )
                phase = "feature"

            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            total_loss.backward()
            if noise.grad is None:
                raise RuntimeError(
                    "Cross-stage claim objective did not produce an image gradient"
                )
            gradient = noise.grad.detach()
            if use_risk:
                direction, risk_state, update_stats = self._momentum_update(
                    gradient,
                    None,
                )
                feature_momentum = (
                    risk_state if phase_mode == "carryover" else None
                )
            else:
                direction, feature_momentum, update_stats = (
                    self._momentum_update(gradient, feature_momentum)
                )

            carryover_applied = (
                phase_mode == "carryover" and step > 0
            )
            update_stats.update(
                {
                    "uncertainty_gate_active": False,
                    "risk_direction_carryover_applied": carryover_applied,
                    "risk_direction_carryover_steps_remaining": (
                        self.config.iterations - step - 1
                        if carryover_applied
                        else 0
                    ),
                }
            )

            with torch.no_grad():
                noise -= step_size * direction
                noise.clamp_(-epsilon, epsilon)
                noise.copy_((clean + noise).clamp(0.0, 255.0) - clean)
                saturation = float(
                    (noise.abs() >= epsilon - 1.0e-6)
                    .float()
                    .mean()
                    .item()
                )
            if phase_mode == "feature_only":
                risk_state_label = "disabled"
            elif use_risk:
                risk_state_label = (
                    "initialized_carryover"
                    if phase_mode == "carryover"
                    else "initialized_reset"
                )
            else:
                risk_state_label = (
                    "carried" if phase_mode == "carryover" else "discarded"
                )
            diagnostics.append(
                {
                    "step": step,
                    "phase": phase,
                    "phase_boundary_mode": phase_mode,
                    "scale_view_mode": self.config.scale_view_mode,
                    "loss": float(total_loss.detach().item()),
                    "linf_pixel": float(noise.detach().abs().max().item()),
                    "budget_saturation": saturation,
                    **risk_stats,
                    **feature_stats,
                    **update_stats,
                    "risk_state": risk_state_label,
                    "risk_repair_updates": risk_updates,
                    "risk_repair_limit": (
                        0 if phase_mode == "feature_only" else 1
                    ),
                }
            )
            if noise.grad is not None:
                noise.grad.zero_()

        adversarial = (clean + noise.detach()).clamp(0.0, 255.0)
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=adversarial - clean,
            diagnostics=diagnostics,
        )
