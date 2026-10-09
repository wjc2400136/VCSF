from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .base import AttackOutput
from .common import config_from_mapping


@dataclass(frozen=True)
class VCSFFinalCandidateConfig:
    """Frozen configuration for the unregistered VCSF-Attack candidate."""

    eps: float = 4.0 / 255.0
    iterations: int = 20
    step_size: float = 1.0 / 255.0
    feature_levels: Tuple[int, ...] = (0, 1, 2)
    feature_eps: float = 1.0e-6
    scale_min: float = 2.0 / 3.0
    scale_max: float = 4.0 / 3.0
    momentum: float = 7.0 / 10.0

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "VCSFFinalCandidateConfig":
        converted = dict(raw)
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(
                int(value) for value in converted["feature_levels"]
            )
        return config_from_mapping(cls, converted)

    def validate(self) -> None:
        if not 0.0 < self.eps <= 1.0:
            raise ValueError("eps must use normalized [0, 1] units")
        if self.iterations < 2:
            raise ValueError("iterations must leave one feature update after initialization")
        if not 0.0 < self.step_size <= self.eps:
            raise ValueError("step_size must be positive and no larger than eps")
        frozen = {
            "feature_levels": tuple(self.feature_levels) == (0, 1, 2),
            "feature_eps": math.isclose(self.feature_eps, 1.0e-6),
            "scale_min": math.isclose(self.scale_min, 2.0 / 3.0),
            "scale_max": math.isclose(self.scale_max, 4.0 / 3.0),
            "momentum": math.isclose(self.momentum, 7.0 / 10.0),
        }
        invalid = [name for name, valid in frozen.items() if not valid]
        if invalid:
            raise ValueError(
                "VCSF-Attack freezes the method constant(s): {}".format(
                    ", ".join(invalid)
                )
            )


class VCSFFinalCandidate:
    """View-Consistent Cross-Stage Feature-Survival Attack.

    The first counted backward minimizes a smooth detector-odds objective.
    Every remaining backward compares clean and adversarial backbone/neck
    features under one shared sampled scale view. Output-gradient momentum is
    discarded before the feature phase.
    """

    implementation_path = "src/lgp/attacks/vcsf_final_candidate.py"
    method_name = "VCSF-Attack"
    _BACKBONE_LEVEL_COUNT = 3

    def __init__(self, adapter: Any, config: VCSFFinalCandidateConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations
        self.auxiliary_forward_passes_per_image = 0
        self.auxiliary_backward_passes_per_image = 0

    @staticmethod
    def _resize_crop_pad(
        tensor: torch.Tensor,
        new_height: int,
        new_width: int,
        crop_top: int,
        crop_left: int,
        pad_left: int,
        pad_right: int,
        pad_top: int,
        pad_bottom: int,
        *,
        is_mask: bool = False,
    ) -> torch.Tensor:
        if is_mask:
            output = F.interpolate(
                tensor,
                size=(new_height, new_width),
                mode="nearest",
            )
        else:
            output = F.interpolate(
                tensor,
                size=(new_height, new_width),
                mode="bilinear",
                align_corners=False,
            )
        if new_height >= tensor.shape[-2]:
            output = output[:, :, crop_top : crop_top + tensor.shape[-2], :]
        if new_width >= tensor.shape[-1]:
            output = output[:, :, :, crop_left : crop_left + tensor.shape[-1]]
        if pad_top or pad_bottom or pad_left or pad_right:
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            if is_mask:
                output = F.pad(
                    output,
                    padding,
                    mode="constant",
                    value=0.0,
                )
            else:
                output = F.pad(output, padding, mode="reflect")
        return output

    def _same_scale_view(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, _, height, width = clean.shape
        scale = torch.empty((), device=clean.device).uniform_(
            self.config.scale_min,
            self.config.scale_max,
            generator=generator,
        )
        new_height = max(1, int(round(float(scale.item()) * height)))
        new_width = max(1, int(round(float(scale.item()) * width)))
        max_top = max(new_height - height, 0)
        max_left = max(new_width - width, 0)
        crop_top = int(
            torch.randint(
                0,
                max_top + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        crop_left = int(
            torch.randint(
                0,
                max_left + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        pad_height = max(height - new_height, 0)
        pad_width = max(width - new_width, 0)
        pad_top = int(
            torch.randint(
                0,
                pad_height + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        pad_left = int(
            torch.randint(
                0,
                pad_width + 1,
                (1,),
                device=clean.device,
                generator=generator,
            ).item()
        )
        arguments = (
            new_height,
            new_width,
            crop_top,
            crop_left,
            pad_left,
            pad_width - pad_left,
            pad_top,
            pad_height - pad_top,
        )
        mask_view = self._resize_crop_pad(
            mask,
            *arguments,
            is_mask=True,
        ).clamp(0.0, 1.0)
        return (
            self._resize_crop_pad(clean, *arguments).clamp(0.0, 255.0),
            self._resize_crop_pad(adversarial, *arguments).clamp(0.0, 255.0),
            mask_view,
        )

    @classmethod
    def _captured_tensors(cls, output: Any) -> Tuple[torch.Tensor, ...]:
        if isinstance(output, torch.Tensor):
            return (output,) if output.ndim == 4 else ()
        if isinstance(output, Mapping):
            values: Sequence[Any] = tuple(output.values())
        elif isinstance(output, (list, tuple)):
            values = output
        else:
            return ()
        tensors: List[torch.Tensor] = []
        for value in values:
            tensors.extend(cls._captured_tensors(value))
        return tuple(tensors)

    @staticmethod
    def _shape_signature(
        features: Sequence[torch.Tensor],
    ) -> List[List[int]]:
        return [
            [int(value) for value in feature.shape]
            for feature in features
        ]

    def _feature_only_surfaces(
        self,
        clean_view: torch.Tensor,
        adversarial_view: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        backbone = getattr(self.adapter.model, "backbone", None)
        if backbone is None or not hasattr(backbone, "register_forward_hook"):
            raise RuntimeError("Detector exposes no hookable backbone")
        captured: List[Tuple[torch.Tensor, ...]] = []

        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensors = self._captured_tensors(output)
            if tensors:
                captured.append(tensors)

        handle = backbone.register_forward_hook(capture)
        try:
            neck_features = self.adapter.extract_features(
                torch.cat((clean_view.detach(), adversarial_view), dim=0)
            )
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "Expected one backbone capture, received {}".format(
                    len(captured)
                )
            )
        return tuple(neck_features), captured[0]

    def _select_neck(
        self,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        selected = tuple(
            features[index]
            for index in self.config.feature_levels
            if 0 <= index < len(features)
        )
        if not selected:
            raise RuntimeError("Configured neck feature levels are unavailable")
        return selected

    def _select_backbone(
        self,
        features: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        if len(features) < self._BACKBONE_LEVEL_COUNT:
            raise RuntimeError("VCSF-Attack requires three backbone feature maps")
        return tuple(features[-self._BACKBONE_LEVEL_COUNT :])

    def _surface_cosines(
        self,
        features: Sequence[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cosines: List[torch.Tensor] = []
        coverages: List[torch.Tensor] = []
        for feature in features:
            if int(feature.shape[0]) != 2:
                raise RuntimeError(
                    "VCSF feature surface did not preserve batch size two"
                )
            clean_feature = feature[:1].detach()
            adversarial_feature = feature[1:2]
            mask_converter = getattr(self.adapter, "model_space_mask", None)
            source_mask = (
                mask_converter(mask_view.to(adversarial_feature.dtype))
                if callable(mask_converter)
                else mask_view.to(adversarial_feature.dtype)
            )
            resized_mask = F.interpolate(
                source_mask,
                size=adversarial_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            clean_energy = clean_feature.square().mean(dim=1, keepdim=True)
            weight = (resized_mask * clean_energy).detach()
            denominator = weight.sum().clamp(min=self.config.feature_eps)
            clean_direction = F.normalize(
                clean_feature,
                p=2,
                dim=1,
                eps=self.config.feature_eps,
            )
            adversarial_direction = F.normalize(
                adversarial_feature,
                p=2,
                dim=1,
                eps=self.config.feature_eps,
            )
            local_cosine = (clean_direction * adversarial_direction).sum(
                dim=1,
                keepdim=True,
            )
            cosines.append((weight * local_cosine).sum() / denominator)
            coverages.append(resized_mask.mean())
        if not cosines:
            raise RuntimeError("VCSF-Attack exposed no usable feature map")
        return torch.stack(cosines), torch.stack(coverages)

    @staticmethod
    def _log_mean_exp(values: torch.Tensor) -> torch.Tensor:
        flattened = values.reshape(-1)
        if flattened.numel() == 0:
            raise RuntimeError("VCSF-Attack received an empty feature surface")
        return torch.logsumexp(flattened, dim=0) - math.log(
            int(flattened.numel())
        )

    def _feature_survival(
        self,
        neck_features: Sequence[torch.Tensor],
        backbone_features: Sequence[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        selected_neck = self._select_neck(neck_features)
        selected_backbone = self._select_backbone(backbone_features)
        neck_cosines, neck_coverages = self._surface_cosines(
            selected_neck,
            mask_view,
        )
        backbone_cosines, backbone_coverages = self._surface_cosines(
            selected_backbone,
            mask_view,
        )
        stacked = torch.cat((neck_cosines, backbone_cosines))
        coverage = torch.cat((neck_coverages, backbone_coverages))
        survival = self._log_mean_exp(stacked)
        return survival, {
            "feature_objective": "directional_cosine_evidence_bottleneck",
            "feature_mean_cos": float(stacked.detach().mean().item()),
            "feature_max_cos": float(stacked.detach().max().item()),
            "feature_mask_coverage": float(coverage.detach().mean().item()),
            "feature_surface": "neck_backbone",
            "feature_surface_level_count": int(stacked.numel()),
            "feature_surface_shapes": {
                "neck": self._shape_signature(selected_neck),
                "backbone": self._shape_signature(selected_backbone),
            },
            "feature_survival_evidence": float(survival.detach().item()),
            "feature_route_clean_entropy": None,
            "feature_route_adversarial_entropy": None,
            "feature_route_target_cross_entropy": None,
            "feature_retained_projection_mean": None,
            "feature_retained_projection_max": None,
            "feature_adversarial_clean_norm_ratio_mean": None,
            "feature_retained_projection_energy_mean": None,
            "feature_orthogonal_energy_ratio_mean": None,
            "feature_surface_mode": "cross_stage",
            "spatial_support_mode": "global_energy",
        }

    def _feature_loss(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        global_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        clean_view, adversarial_view, mask_view = self._same_scale_view(
            clean,
            adversarial,
            global_mask,
            generator,
        )
        neck_features, backbone_features = self._feature_only_surfaces(
            clean_view,
            adversarial_view,
        )
        return self._feature_survival(
            neck_features,
            backbone_features,
            mask_view,
        )

    def _risk_loss(
        self,
        adversarial: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        scores = [
            value
            for value in self.adapter.prediction_scores(adversarial)
            if value.numel()
        ]
        surface = "post_nms"
        if not scores or not any(value.requires_grad for value in scores):
            fallback = getattr(self.adapter, "pre_nms_prediction_scores", None)
            scores = [
                value
                for value in (
                    fallback(adversarial) if callable(fallback) else []
                )
                if value.numel()
            ]
            surface = "pre_nms"
        if not scores:
            return None, {
                "score_risk": None,
                "score_risk_defined": False,
                "constraint_violation": 0.0,
                "risk_surface": "unavailable",
                "risk_objective": "smooth_log_odds",
                "risk_available_count": 0,
                "risk_aggregate": None,
            }
        flattened = torch.cat([value.reshape(-1) for value in scores])
        dtype_eps = torch.finfo(flattened.dtype).eps
        probabilities = flattened.clamp(
            min=dtype_eps,
            max=1.0 - dtype_eps,
        )
        log_odds = torch.log(probabilities) - torch.log1p(-probabilities)
        risk = torch.logsumexp(log_odds, dim=0)
        aggregate = float(torch.sigmoid(risk.detach()).item())
        if not risk.requires_grad:
            return None, {
                "score_risk": float(risk.item()),
                "score_risk_defined": True,
                "constraint_violation": 0.0,
                "risk_surface": "unavailable",
                "risk_objective": "smooth_log_odds",
                "risk_available_count": int(flattened.numel()),
                "risk_aggregate": aggregate,
            }
        return risk, {
            "score_risk": float(risk.detach().item()),
            "score_risk_defined": True,
            "constraint_violation": 0.0,
            "risk_surface": surface,
            "risk_objective": "smooth_log_odds",
            "risk_available_count": int(flattened.numel()),
            "risk_aggregate": aggregate,
        }

    @staticmethod
    def _empty_risk_stats() -> Dict[str, Any]:
        return {
            "score_risk": None,
            "score_risk_defined": False,
            "constraint_violation": 0.0,
            "risk_surface": "not_checked",
            "risk_objective": "smooth_log_odds",
            "risk_available_count": 0,
            "risk_aggregate": None,
        }

    @staticmethod
    def _empty_feature_stats(coverage: float) -> Dict[str, Any]:
        return {
            "feature_objective": "none",
            "feature_mean_cos": None,
            "feature_max_cos": None,
            "feature_mask_coverage": coverage,
            "feature_surface": "neck_backbone",
            "feature_surface_level_count": 0,
            "feature_surface_shapes": None,
            "feature_survival_evidence": None,
            "feature_route_clean_entropy": None,
            "feature_route_adversarial_entropy": None,
            "feature_route_target_cross_entropy": None,
            "feature_retained_projection_mean": None,
            "feature_retained_projection_max": None,
            "feature_adversarial_clean_norm_ratio_mean": None,
            "feature_retained_projection_energy_mean": None,
            "feature_orthogonal_energy_ratio_mean": None,
            "joint_evidence_mode": "survival_bottleneck",
            "joint_survival_evidence": None,
            "feature_surface_mode": "cross_stage",
            "spatial_support_mode": "global_energy",
        }

    @staticmethod
    def _normalize_gradient(gradient: torch.Tensor) -> torch.Tensor:
        centered = gradient - gradient.mean(dim=(2, 3), keepdim=True)
        denominator = centered.abs().mean(
            dim=(1, 2, 3),
            keepdim=True,
        ).clamp(min=1.0e-12)
        return centered / denominator

    def _momentum_update(
        self,
        gradient: torch.Tensor,
        state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        normalized = self._normalize_gradient(gradient)
        state = (
            normalized
            if state is None
            else self.config.momentum * state + normalized
        )
        direction = state.sign()
        return direction, state, {
            "update_keep_ratio": float(
                (direction != 0).float().mean().item()
            ),
            "gradient_snr_mean": 0.0,
            "gradient_sign_flip_rate": 0.0,
        }

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del boxes_xyxy, labels
        clean = clean_bgr.detach().to(
            self.adapter.device,
            dtype=torch.float32,
        ).clamp(0.0, 255.0)
        if clean.shape[0] != 1:
            raise ValueError("VCSF-Attack currently expects batch size one")
        global_mask = torch.ones(
            (clean.shape[0], 1, clean.shape[-2], clean.shape[-1]),
            device=clean.device,
            dtype=clean.dtype,
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
                        "Smooth detector-odds initialization exposed no gradient"
                    )
                feature_stats = self._empty_feature_stats(
                    float(global_mask.mean().item())
                )
                phase = "risk_initialization"
            else:
                total_loss, feature_stats = self._feature_loss(
                    clean,
                    adversarial,
                    global_mask,
                    generator,
                )
                risk_stats = self._empty_risk_stats()
                phase = "feature"

            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            total_loss.backward()
            if noise.grad is None:
                raise RuntimeError("VCSF objective produced no image gradient")
            gradient = noise.grad.detach()
            if step == 0:
                direction, _discarded, update_stats = self._momentum_update(
                    gradient,
                    None,
                )
            else:
                direction, feature_momentum, update_stats = (
                    self._momentum_update(gradient, feature_momentum)
                )
            update_stats.update(
                {
                    "uncertainty_gate_active": False,
                    "risk_direction_carryover_applied": False,
                    "risk_direction_carryover_steps_remaining": 0,
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
            diagnostics.append(
                {
                    "step": step,
                    "phase": phase,
                    "phase_boundary_mode": "reset",
                    "scale_view_mode": "synchronized_scale",
                    "loss": float(total_loss.detach().item()),
                    "linf_pixel": float(noise.detach().abs().max().item()),
                    "budget_saturation": saturation,
                    **risk_stats,
                    **feature_stats,
                    **update_stats,
                    "risk_state": (
                        "initialized_reset" if step == 0 else "discarded"
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
