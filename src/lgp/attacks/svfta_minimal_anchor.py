from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import AttackOutput
from .common import config_from_mapping, parse_fraction
from .svfta_evidence_bottleneck import SVFTAEvidenceBottleneck


@dataclass(frozen=True)
class SVFTAMinimalAnchorConfig:
    """Uncertainty-free configuration of the frozen cross-stage risk anchor."""

    eps: float = 4.0 / 255.0
    iterations: int = 20
    step_size: float = 1.0 / 255.0
    score_threshold: float = 1.0 / 20.0
    feature_levels: Tuple[int, ...] = (0, 1, 2)
    feature_eps: float = 1.0e-6
    feature_objective: str = "directional_cosine"
    scale_min: float = 2.0 / 3.0
    scale_max: float = 4.0 / 3.0
    momentum: float = 7.0 / 10.0
    risk_warmup: int = 1
    risk_to_feature_steps: int = 0
    risk_enabled: bool = True
    risk_objective: str = "smooth_log_odds"
    object_mask_weighting: str = "union"

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "SVFTAMinimalAnchorConfig":
        converted = dict(raw)
        compatibility_defaults = {
            "uncertainty_mode": "uncertainty",
            "uncertainty_decay": 7.0 / 10.0,
            "uncertainty_kappa": 1.0 / 2.0,
            "uncertainty_eps": 1.0e-12,
        }
        # The generic experiment runner merges the maintained SVFTA YAML before
        # applying study parameters. Validate and discard only those inherited
        # legacy defaults at this adapter boundary. They never become fields of
        # the minimal config and cannot affect execution.
        for key, expected in compatibility_defaults.items():
            if key not in converted:
                continue
            actual = parse_fraction(converted.pop(key))
            valid = (
                actual == expected
                if isinstance(expected, str)
                else isinstance(actual, (int, float))
                and math.isclose(float(actual), expected)
            )
            if not valid:
                raise ValueError(
                    "Removed uncertainty compatibility key {} must retain "
                    "the registered inherited default".format(key)
                )
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(
                int(value) for value in converted["feature_levels"]
            )
        return config_from_mapping(cls, converted)

    @property
    def uncertainty_mode(self) -> str:
        """Route the maintained executor through ordinary first-moment momentum."""

        return "none"

    @property
    def persistent_output_evidence(self) -> bool:
        return False

    @property
    def include_backbone_surface(self) -> bool:
        return True

    def validate(self) -> None:
        required = {
            "eps": math.isclose(self.eps, 4.0 / 255.0),
            "iterations": self.iterations == 20,
            "step_size": math.isclose(self.step_size, 1.0 / 255.0),
            "score_threshold": math.isclose(
                self.score_threshold, 1.0 / 20.0
            ),
            "feature_levels": tuple(self.feature_levels) == (0, 1, 2),
            "feature_eps": self.feature_eps > 0.0,
            "feature_objective": self.feature_objective == "directional_cosine",
            "scale_min": math.isclose(self.scale_min, 2.0 / 3.0),
            "scale_max": math.isclose(self.scale_max, 4.0 / 3.0),
            "momentum": math.isclose(self.momentum, 7.0 / 10.0),
            "risk_warmup": self.risk_warmup == 1,
            "risk_to_feature_steps": self.risk_to_feature_steps == 0,
            "risk_enabled": self.risk_enabled is True,
            "risk_objective": self.risk_objective == "smooth_log_odds",
            "object_mask_weighting": self.object_mask_weighting == "union",
        }
        failed = [name for name, valid in required.items() if not valid]
        if failed:
            raise ValueError(
                "Minimal cross-stage risk anchor freezes: {}".format(
                    ", ".join(failed)
                )
            )


class SVFTAMinimalAnchor(SVFTAEvidenceBottleneck):
    """One smooth-risk step plus nineteen cross-stage momentum updates."""

    implementation_path = "src/lgp/attacks/svfta_minimal_anchor.py"

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

        for step in range(self.config.iterations):
            adversarial = (clean + noise).clamp(0.0, 255.0)
            if step == 0:
                total_loss, risk_stats = self._risk_loss(adversarial)
                if total_loss is None:
                    raise RuntimeError(
                        "Smooth log-odds initialization exposed no gradient"
                    )
                feature_stats = {
                    "feature_objective": self.config.feature_objective,
                    "feature_mean_cos": None,
                    "feature_max_cos": None,
                    "feature_mask_coverage": float(object_mask.mean().item()),
                    "feature_route_clean_entropy": None,
                    "feature_route_adversarial_entropy": None,
                    "feature_route_target_cross_entropy": None,
                    "feature_retained_projection_mean": None,
                    "feature_retained_projection_max": None,
                    "feature_adversarial_clean_norm_ratio_mean": None,
                    "feature_retained_projection_energy_mean": None,
                    "feature_orthogonal_energy_ratio_mean": None,
                }
                phase = "risk_initialization"
            else:
                total_loss, feature_stats = self._feature_loss(
                    clean,
                    adversarial,
                    object_mask,
                    generator,
                )
                risk_stats = {
                    "score_risk": None,
                    "score_risk_defined": False,
                    "constraint_violation": 0.0,
                    "risk_surface": "not_checked",
                    "risk_objective": self.config.risk_objective,
                    "risk_available_count": 0,
                    "risk_aggregate": None,
                }
                phase = "feature"

            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            total_loss.backward()
            if noise.grad is None:
                raise RuntimeError(
                    "Minimal anchor objective did not produce an image gradient"
                )
            gradient = noise.grad.detach()
            if step == 0:
                direction, _risk_state, update_stats = self._momentum_update(
                    gradient,
                    None,
                )
            else:
                direction, feature_momentum, update_stats = (
                    self._momentum_update(gradient, feature_momentum)
                )
            update_stats["uncertainty_gate_active"] = False
            update_stats["risk_direction_carryover_applied"] = False
            update_stats["risk_direction_carryover_steps_remaining"] = 0

            with torch.no_grad():
                noise -= step_size * direction
                noise.clamp_(-epsilon, epsilon)
                noise.copy_((clean + noise).clamp(0.0, 255.0) - clean)
                saturation = float(
                    (noise.abs() >= epsilon - 1.0e-6).float().mean().item()
                )
            diagnostics.append(
                {
                    "step": step,
                    "phase": phase,
                    "loss": float(total_loss.detach().item()),
                    "linf_pixel": float(noise.detach().abs().max().item()),
                    "budget_saturation": saturation,
                    **risk_stats,
                    **feature_stats,
                    **update_stats,
                    "risk_state": (
                        "fixed_initialization" if step == 0 else "disabled"
                    ),
                    "risk_repair_updates": 1,
                    "risk_repair_limit": 1,
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
