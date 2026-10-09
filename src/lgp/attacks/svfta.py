from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import config_from_mapping


@dataclass(frozen=True)
class SVFTAConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 20
    step_size: float = 1.0 / 255.0
    score_threshold: float = 1.0 / 20.0
    feature_levels: Tuple[int, ...] = (0, 1, 2)
    feature_eps: float = 1.0e-6
    feature_objective: str = "directional_cosine"
    scale_min: float = 2.0 / 3.0
    scale_max: float = 4.0 / 3.0
    momentum: float = 2.0 / 3.0
    risk_warmup: int = 2
    risk_to_feature_steps: int = 0
    risk_enabled: bool = True
    risk_objective: str = "max_hinge"
    object_mask_weighting: str = "union"
    uncertainty_mode: str = "uncertainty"
    uncertainty_decay: float = 7.0 / 10.0
    uncertainty_kappa: float = 1.0 / 2.0
    uncertainty_eps: float = 1.0e-12

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "SVFTAConfig":
        converted = dict(raw)
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(int(value) for value in converted["feature_levels"])
        return config_from_mapping(cls, converted)

    def validate(self) -> None:
        if not 0.0 < self.eps <= 1.0:
            raise ValueError("eps must be in normalized [0, 1] units")
        if not 0.0 < self.step_size <= self.eps:
            raise ValueError("step_size must be positive and no larger than eps")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not 0.0 < self.score_threshold <= 1.0:
            raise ValueError("score_threshold must lie in (0, 1]")
        if not 0.0 < self.scale_min <= self.scale_max:
            raise ValueError("scale range must be positive and ordered")
        if self.momentum < 0.0:
            raise ValueError("momentum must be non-negative")
        if self.risk_warmup < 0:
            raise ValueError("risk_warmup must be non-negative")
        if self.risk_to_feature_steps < 0:
            raise ValueError("risk_to_feature_steps must be non-negative")
        if self.risk_objective not in {
            "max_hinge",
            "cumulative_hazard",
            "smooth_log_odds",
        }:
            raise ValueError(
                "risk_objective must be max_hinge, cumulative_hazard or "
                "smooth_log_odds"
            )
        if self.object_mask_weighting not in {"union", "equal_instance"}:
            raise ValueError(
                "object_mask_weighting must be union or equal_instance"
            )
        if self.feature_objective not in {
            "directional_cosine",
            "retained_projection",
            "retained_projection_energy",
            "evidence_turnover",
            "evidence_turnover_signed",
            "evidence_turnover_signed_hard",
            "scale_route_uniform",
            "scale_route_reverse",
        }:
            raise ValueError(
                "feature_objective must be directional_cosine, "
                "retained_projection, retained_projection_energy, "
                "evidence_turnover, evidence_turnover_signed, "
                "evidence_turnover_signed_hard, "
                "scale_route_uniform or "
                "scale_route_reverse"
            )
        if (
            self.feature_objective.startswith("scale_route_")
            and len(set(self.feature_levels)) < 2
        ):
            raise ValueError(
                "scale-routing feature objectives require at least two "
                "distinct feature levels"
            )
        if self.uncertainty_mode not in {"none", "uncertainty"}:
            raise ValueError("uncertainty_mode must be none or uncertainty")
        if not 0.0 <= self.uncertainty_decay < 1.0:
            raise ValueError("uncertainty_decay must lie in [0, 1)")
        if self.uncertainty_kappa < 0.0:
            raise ValueError("uncertainty_kappa must be non-negative")


class SVFTA:
    """Scale-view feature transfer attack with uncertainty-gated updates.

    The maintained default keeps the two project objectives distinct: a
    bounded source score-risk repair prefix and object-masked feature
    decorrelation. Development-only risk objectives are explicit configuration
    choices and do not silently change that default.
    It uses only public MMDetection/MMYOLO 3.x detector APIs.
    """

    def __init__(self, adapter: OpenMMLabAdapter, config: SVFTAConfig) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations

    @staticmethod
    def _object_mask(
        clean: torch.Tensor,
        boxes: torch.Tensor,
        weighting: str = "union",
    ) -> torch.Tensor:
        batch, _, height, width = clean.shape
        if batch != 1:
            raise ValueError("SVFTA runner currently expects batch size 1")
        mask = clean.new_zeros((1, 1, height, width))
        if boxes.numel() == 0:
            mask.fill_(1.0)
            return mask
        for box in boxes:
            left = max(0, min(int(torch.floor(box[0]).item()), width - 1))
            top = max(0, min(int(torch.floor(box[1]).item()), height - 1))
            right = max(left + 1, min(int(torch.ceil(box[2]).item()), width))
            bottom = max(top + 1, min(int(torch.ceil(box[3]).item()), height))
            if weighting == "equal_instance":
                area = float((right - left) * (bottom - top))
                mask[:, :, top:bottom, left:right] += 1.0 / area
            else:
                mask[:, :, top:bottom, left:right] = 1.0
        return mask

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
        is_mask: bool = False,
    ) -> torch.Tensor:
        if is_mask:
            output = F.interpolate(tensor, size=(new_height, new_width), mode="nearest")
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
                output = F.pad(output, padding, mode="constant", value=0.0)
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
            self.config.scale_min, self.config.scale_max, generator=generator
        )
        new_height = max(1, int(round(float(scale.item()) * height)))
        new_width = max(1, int(round(float(scale.item()) * width)))
        max_top = max(new_height - height, 0)
        max_left = max(new_width - width, 0)
        crop_top = int(torch.randint(0, max_top + 1, (1,), device=clean.device, generator=generator).item())
        crop_left = int(torch.randint(0, max_left + 1, (1,), device=clean.device, generator=generator).item())
        pad_height = max(height - new_height, 0)
        pad_width = max(width - new_width, 0)
        pad_top = int(torch.randint(0, pad_height + 1, (1,), device=clean.device, generator=generator).item())
        pad_left = int(torch.randint(0, pad_width + 1, (1,), device=clean.device, generator=generator).item())
        args = (
            new_height,
            new_width,
            crop_top,
            crop_left,
            pad_left,
            pad_width - pad_left,
            pad_top,
            pad_height - pad_top,
        )
        mask_view = self._resize_crop_pad(mask, *args, is_mask=True)
        mask_view = (
            mask_view.clamp(min=0.0)
            if self.config.object_mask_weighting == "equal_instance"
            else mask_view.clamp(0.0, 1.0)
        )
        return (
            self._resize_crop_pad(clean, *args).clamp(0.0, 255.0),
            self._resize_crop_pad(adversarial, *args).clamp(0.0, 255.0),
            mask_view,
        )

    def _directional_cosine_feature_loss(
        self,
        selected: List[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Preserve the maintained per-level feature objective exactly."""
        cosines = []
        coverages = []
        for feature in selected:
            clean_feature = feature[:1].detach()
            adversarial_feature = feature[1:2]
            # The production MMDetection adapter pads the raw object mask to
            # the detector's model-space geometry. Lightweight unit-test
            # adapters may expose features in raw-image coordinates.
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
            )
            resized_mask = (
                resized_mask.clamp(min=0.0)
                if self.config.object_mask_weighting == "equal_instance"
                else resized_mask.clamp(0.0, 1.0)
            )
            energy = clean_feature.square().mean(dim=1, keepdim=True)
            weight = (resized_mask * energy).detach()
            denominator = weight.sum().clamp(min=self.config.feature_eps)
            clean_direction = F.normalize(
                clean_feature, p=2, dim=1, eps=self.config.feature_eps
            )
            adversarial_direction = F.normalize(
                adversarial_feature, p=2, dim=1, eps=self.config.feature_eps
            )
            cosine = (clean_direction * adversarial_direction).sum(
                dim=1, keepdim=True
            )
            cosines.append((weight * cosine).sum() / denominator)
            coverages.append(resized_mask.mean())
        stacked = torch.stack(cosines)
        loss = torch.logsumexp(stacked, dim=0)
        return loss, {
            "feature_objective": "directional_cosine",
            "feature_mean_cos": float(stacked.detach().mean().item()),
            "feature_max_cos": float(stacked.detach().max().item()),
            "feature_mask_coverage": float(
                torch.stack(coverages).detach().mean().item()
            ),
            "feature_route_clean_entropy": None,
            "feature_route_adversarial_entropy": None,
            "feature_route_target_cross_entropy": None,
            "feature_retained_projection_mean": None,
            "feature_retained_projection_max": None,
            "feature_adversarial_clean_norm_ratio_mean": None,
        }

    def _retained_projection_feature_loss(
        self,
        selected: List[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Remove clean-aligned object evidence without a scale parameter.

        The per-level coefficient is the weighted least-squares projection of
        the adversarial feature onto the fixed clean feature. It factors into
        an adversarial-to-clean norm ratio and a weighted cosine, so unlike a
        cosine-only objective it remains sensitive to retained magnitude.
        """
        retained = []
        norm_ratios = []
        coverages = []
        for feature in selected:
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
            )
            resized_mask = (
                resized_mask.clamp(min=0.0)
                if self.config.object_mask_weighting == "equal_instance"
                else resized_mask.clamp(0.0, 1.0)
            )
            clean_energy = (
                resized_mask * clean_feature.square().sum(dim=1, keepdim=True)
            ).sum()
            denominator = clean_energy.clamp(min=self.config.feature_eps)
            clean_aligned = (
                resized_mask
                * (clean_feature * adversarial_feature).sum(
                    dim=1, keepdim=True
                )
            ).sum()
            adversarial_energy = (
                resized_mask
                * adversarial_feature.square().sum(dim=1, keepdim=True)
            ).sum()
            retained.append(clean_aligned / denominator)
            norm_ratios.append(
                torch.sqrt(
                    adversarial_energy.clamp(min=self.config.feature_eps)
                    / denominator
                )
            )
            coverages.append(resized_mask.mean())

        retained_stack = torch.stack(retained)
        norm_stack = torch.stack(norm_ratios)
        loss = torch.logsumexp(retained_stack, dim=0)
        return loss, {
            "feature_objective": "retained_projection",
            "feature_mean_cos": None,
            "feature_max_cos": None,
            "feature_mask_coverage": float(
                torch.stack(coverages).detach().mean().item()
            ),
            "feature_route_clean_entropy": None,
            "feature_route_adversarial_entropy": None,
            "feature_route_target_cross_entropy": None,
            "feature_retained_projection_mean": float(
                retained_stack.detach().mean().item()
            ),
            "feature_retained_projection_max": float(
                retained_stack.detach().max().item()
            ),
            "feature_adversarial_clean_norm_ratio_mean": float(
                norm_stack.detach().mean().item()
            ),
            "feature_retained_projection_energy_mean": None,
            "feature_orthogonal_energy_ratio_mean": None,
        }

    def _projection_energy_feature_loss(
        self,
        selected: List[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decompose object evidence into clean-aligned and orthogonal energy."""
        aligned_ratios = []
        signed_aligned_ratios = []
        signed_hard_aligned_ratios = []
        orthogonal_ratios = []
        norm_ratios = []
        coverages = []
        for feature in selected:
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
            )
            resized_mask = (
                resized_mask.clamp(min=0.0)
                if self.config.object_mask_weighting == "equal_instance"
                else resized_mask.clamp(0.0, 1.0)
            )
            clean_energy = clean_feature.square().sum(dim=1, keepdim=True)
            adversarial_energy = adversarial_feature.square().sum(
                dim=1, keepdim=True
            )
            clean_adversarial_dot = (
                clean_feature * adversarial_feature
            ).sum(dim=1, keepdim=True)
            aligned_energy = clean_adversarial_dot.square() / clean_energy.clamp(
                min=self.config.feature_eps
            )
            orthogonal_energy = (
                adversarial_energy - aligned_energy
            ).clamp(min=0.0)
            denominator = (resized_mask * clean_energy).sum().clamp(
                min=self.config.feature_eps
            )
            aligned_ratio = (resized_mask * aligned_energy).sum() / denominator
            orthogonal_ratio = (
                resized_mask * orthogonal_energy
            ).sum() / denominator
            adversarial_ratio = (
                resized_mask * adversarial_energy
            ).sum() / denominator
            aligned_ratios.append(aligned_ratio)
            sign_term = torch.tanh(
                clean_adversarial_dot
                / clean_energy.sqrt().clamp(min=self.config.feature_eps)
            )
            signed_aligned_ratios.append(
                (resized_mask * aligned_energy * sign_term).sum() / denominator
            )
            hard_sign_term = torch.tanh(
                4.0
                * clean_adversarial_dot
                / (
                    clean_energy.sqrt()
                    * adversarial_energy.sqrt()
                    + self.config.feature_eps
                ).clamp(min=self.config.feature_eps)
            )
            signed_hard_aligned_ratios.append(
                (resized_mask * aligned_energy * hard_sign_term).sum() / denominator
            )
            orthogonal_ratios.append(orthogonal_ratio)
            norm_ratios.append(
                torch.sqrt(adversarial_ratio.clamp(min=self.config.feature_eps))
            )
            coverages.append(resized_mask.mean())

        aligned_stack = torch.stack(aligned_ratios)
        signed_aligned_stack = torch.stack(signed_aligned_ratios)
        signed_hard_aligned_stack = torch.stack(signed_hard_aligned_ratios)
        orthogonal_stack = torch.stack(orthogonal_ratios)
        norm_stack = torch.stack(norm_ratios)
        if self.config.feature_objective == "evidence_turnover":
            level_scores = aligned_stack - orthogonal_stack
        elif self.config.feature_objective == "evidence_turnover_signed":
            level_scores = signed_aligned_stack - orthogonal_stack
        elif self.config.feature_objective == "evidence_turnover_signed_hard":
            level_scores = signed_hard_aligned_stack - orthogonal_stack
        else:
            level_scores = aligned_stack
        loss = torch.logsumexp(level_scores, dim=0)
        return loss, {
            "feature_objective": self.config.feature_objective,
            "feature_mean_cos": None,
            "feature_max_cos": None,
            "feature_mask_coverage": float(
                torch.stack(coverages).detach().mean().item()
            ),
            "feature_route_clean_entropy": None,
            "feature_route_adversarial_entropy": None,
            "feature_route_target_cross_entropy": None,
            "feature_retained_projection_mean": None,
            "feature_retained_projection_max": None,
            "feature_adversarial_clean_norm_ratio_mean": float(
                norm_stack.detach().mean().item()
            ),
            "feature_retained_projection_energy_mean": float(
                aligned_stack.detach().mean().item()
            ),
            "feature_orthogonal_energy_ratio_mean": float(
                orthogonal_stack.detach().mean().item()
            ),
        }

    def _scale_routing_feature_loss(
        self,
        selected: List[torch.Tensor],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Disrupt the joint object-energy route across feature levels.

        Each level contributes a channel-count-normalized RMS evidence map.
        The maps are aligned to one spatial lattice and normalized across
        levels at every location. The fixed clean route defines either a
        uniform target or its normalized complement; only the adversarial
        route remains differentiable.
        """
        if len(selected) < 2:
            raise RuntimeError(
                "Scale-routing objective resolved fewer than two feature levels"
            )
        common_height = max(int(feature.shape[-2]) for feature in selected)
        common_width = max(int(feature.shape[-1]) for feature in selected)
        common_size = (common_height, common_width)
        clean_evidence = []
        adversarial_evidence = []
        for feature in selected:
            clean_feature = feature[:1].detach()
            adversarial_feature = feature[1:2]
            clean_rms = torch.sqrt(
                clean_feature.square().mean(dim=1, keepdim=True)
                + self.config.feature_eps
            )
            adversarial_rms = torch.sqrt(
                adversarial_feature.square().mean(dim=1, keepdim=True)
                + self.config.feature_eps
            )
            clean_evidence.append(
                F.interpolate(
                    clean_rms,
                    size=common_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            adversarial_evidence.append(
                F.interpolate(
                    adversarial_rms,
                    size=common_size,
                    mode="bilinear",
                    align_corners=False,
                )
            )

        clean_stack = torch.cat(clean_evidence, dim=1).detach()
        adversarial_stack = torch.cat(adversarial_evidence, dim=1)
        clean_route = clean_stack / clean_stack.sum(
            dim=1, keepdim=True
        ).clamp(min=self.config.feature_eps)
        adversarial_route = adversarial_stack / adversarial_stack.sum(
            dim=1, keepdim=True
        ).clamp(min=self.config.feature_eps)
        level_count = clean_route.shape[1]
        if self.config.feature_objective == "scale_route_uniform":
            target_route = torch.full_like(clean_route, 1.0 / level_count)
        else:
            target_route = (1.0 - clean_route) / float(level_count - 1)
        target_route = target_route.detach()

        mask_converter = getattr(self.adapter, "model_space_mask", None)
        source_mask = (
            mask_converter(mask_view.to(adversarial_stack.dtype))
            if callable(mask_converter)
            else mask_view.to(adversarial_stack.dtype)
        )
        aligned_mask = F.interpolate(
            source_mask,
            size=common_size,
            mode="bilinear",
            align_corners=False,
        )
        aligned_mask = (
            aligned_mask.clamp(min=0.0)
            if self.config.object_mask_weighting == "equal_instance"
            else aligned_mask.clamp(0.0, 1.0)
        )
        spatial_weight = (aligned_mask * clean_stack.sum(dim=1, keepdim=True)).detach()
        denominator = spatial_weight.sum().clamp(min=self.config.feature_eps)
        target_cross_entropy = -(
            target_route
            * adversarial_route.clamp(min=self.config.feature_eps).log()
        ).sum(dim=1, keepdim=True)
        loss = (spatial_weight * target_cross_entropy).sum() / denominator

        clean_entropy = -(
            clean_route * clean_route.clamp(min=self.config.feature_eps).log()
        ).sum(dim=1, keepdim=True)
        adversarial_entropy = -(
            adversarial_route
            * adversarial_route.clamp(min=self.config.feature_eps).log()
        ).sum(dim=1, keepdim=True)

        def weighted_mean(value: torch.Tensor) -> float:
            return float(
                ((spatial_weight * value).sum() / denominator).detach().item()
            )

        return loss, {
            "feature_objective": self.config.feature_objective,
            "feature_mean_cos": None,
            "feature_max_cos": None,
            "feature_mask_coverage": float(aligned_mask.detach().mean().item()),
            "feature_route_clean_entropy": weighted_mean(clean_entropy),
            "feature_route_adversarial_entropy": weighted_mean(
                adversarial_entropy
            ),
            "feature_route_target_cross_entropy": float(loss.detach().item()),
            "feature_retained_projection_mean": None,
            "feature_retained_projection_max": None,
            "feature_adversarial_clean_norm_ratio_mean": None,
            "feature_retained_projection_energy_mean": None,
            "feature_orthogonal_energy_ratio_mean": None,
        }

    def _feature_loss(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        object_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        clean_view, adversarial_view, mask_view = self._same_scale_view(
            clean, adversarial, object_mask, generator
        )
        features = self.adapter.extract_features(
            torch.cat((clean_view.detach(), adversarial_view), dim=0)
        )
        selected = [
            features[index]
            for index in self.config.feature_levels
            if 0 <= index < len(features)
        ]
        if not selected:
            raise RuntimeError("Configured feature levels are unavailable")
        if self.config.feature_objective == "directional_cosine":
            return self._directional_cosine_feature_loss(selected, mask_view)
        if self.config.feature_objective == "retained_projection":
            return self._retained_projection_feature_loss(selected, mask_view)
        if self.config.feature_objective in {
            "retained_projection_energy",
            "evidence_turnover",
            "evidence_turnover_signed",
            "evidence_turnover_signed_hard",
        }:
            return self._projection_energy_feature_loss(selected, mask_view)
        return self._scale_routing_feature_loss(selected, mask_view)

    def _risk_loss(
        self, adversarial: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        scores = [value for value in self.adapter.prediction_scores(adversarial) if value.numel()]
        surface = "post_nms"
        # An empty post-NMS result is just as unsuitable as a detached one:
        # both require the differentiable pre-NMS fallback. Skipping this
        # branch for the empty case silently changed the declared risk-prefix
        # state on images where NMS returned no detections.
        if not scores or not any(value.requires_grad for value in scores):
            fallback = getattr(self.adapter, "pre_nms_prediction_scores", None)
            scores = [
                value
                for value in (fallback(adversarial) if callable(fallback) else [])
                if value.numel()
            ]
            surface = "pre_nms"
        if not scores:
            return None, {
                # The maximum of an empty score surface is undefined.  Keep
                # that semantic distinction in strict JSON instead of using
                # a non-finite numeric sentinel that can poison downstream
                # manifests and aggregate validation.
                "score_risk": None,
                "score_risk_defined": False,
                "constraint_violation": 0.0,
                "risk_surface": "unavailable",
                "risk_objective": self.config.risk_objective,
                "risk_available_count": 0,
                "risk_aggregate": None,
            }
        flattened = torch.cat([value.reshape(-1) for value in scores])
        if self.config.risk_objective == "cumulative_hazard":
            dtype_eps = torch.finfo(flattened.dtype).eps
            probabilities = flattened.clamp(min=0.0, max=1.0 - dtype_eps)
            mean_hazard = (-torch.log1p(-probabilities)).mean()
            risk = mean_hazard.clamp(min=dtype_eps).log()
            aggregate = float(mean_hazard.detach().item())
        elif self.config.risk_objective == "smooth_log_odds":
            dtype_eps = torch.finfo(flattened.dtype).eps
            probabilities = flattened.clamp(
                min=dtype_eps, max=1.0 - dtype_eps
            )
            log_odds = torch.log(probabilities) - torch.log1p(-probabilities)
            risk = torch.logsumexp(log_odds, dim=0)
            aggregate = float(torch.sigmoid(risk.detach()).item())
        else:
            maxima = torch.stack(
                [value.clamp(min=1.0e-12).max().log() for value in scores]
            )
            risk = maxima.mean()
            aggregate = float(risk.detach().exp().item())
        if not risk.requires_grad:
            return None, {
                "score_risk": float(risk.item()),
                "score_risk_defined": True,
                "constraint_violation": 0.0,
                "risk_surface": "unavailable",
                "risk_objective": self.config.risk_objective,
                "risk_available_count": int(flattened.numel()),
                "risk_aggregate": aggregate,
            }
        common = {
            "score_risk": float(risk.detach().item()),
            "score_risk_defined": True,
            "risk_surface": surface,
            "risk_objective": self.config.risk_objective,
            "risk_available_count": int(flattened.numel()),
            "risk_aggregate": aggregate,
        }
        if self.config.risk_objective != "max_hinge":
            return risk, {**common, "constraint_violation": 0.0}
        threshold = math.log(self.config.score_threshold)
        violation = risk - threshold
        loss = F.relu(violation)
        return loss, {
            **common,
            "constraint_violation": float(F.relu(violation).detach().item()),
        }

    @staticmethod
    def _normalize_gradient(gradient: torch.Tensor) -> torch.Tensor:
        centered = gradient - gradient.mean(dim=(2, 3), keepdim=True)
        denominator = centered.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1.0e-12)
        return centered / denominator

    def _momentum_update(
        self, gradient: torch.Tensor, state: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        normalized = self._normalize_gradient(gradient)
        state = normalized if state is None else self.config.momentum * state + normalized
        direction = state.sign()
        return direction, state, {
            "update_keep_ratio": float((direction != 0).float().mean().item()),
            "gradient_snr_mean": 0.0,
            "gradient_sign_flip_rate": 0.0,
        }

    def _uncertainty_update(
        self,
        gradient: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor], int]],
    ) -> Tuple[
        torch.Tensor,
        Tuple[torch.Tensor, Optional[torch.Tensor], int],
        Dict[str, Any],
    ]:
        normalized = self._normalize_gradient(gradient)
        if state is None:
            first = torch.zeros_like(normalized)
            second: Optional[torch.Tensor] = None
            count = 0
        else:
            first, second, count = state
        if count:
            previous = first / max(
                1.0 - self.config.uncertainty_decay ** count,
                self.config.uncertainty_eps,
            )
            flip_rate = float((previous * normalized < 0).float().mean().item())
        else:
            flip_rate = 0.0
        decay = self.config.uncertainty_decay
        first = decay * first + (1.0 - decay) * normalized
        count += 1
        if self.config.uncertainty_kappa == 0.0:
            direction = first.sign()
            return direction, (first, None, count), {
                "update_keep_ratio": float(
                    (direction != 0).float().mean().item()
                ),
                "gradient_snr_mean": None,
                "gradient_sign_flip_rate": flip_rate,
                "uncertainty_gate_active": False,
            }
        if second is None:
            second = torch.zeros_like(normalized)
        second = decay * second + (1.0 - decay) * normalized.square()
        correction = max(1.0 - decay ** count, self.config.uncertainty_eps)
        mean = first / correction
        variance = (second / correction - mean.square()).clamp(min=0.0)
        uncertainty = torch.sqrt(variance + self.config.uncertainty_eps)
        snr = mean.abs() / uncertainty
        keep = snr > self.config.uncertainty_kappa
        direction = first.sign() * keep.to(first.dtype)
        return direction, (first, second, count), {
            "update_keep_ratio": float(keep.float().mean().item()),
            "gradient_snr_mean": float(snr.detach().clamp(max=1.0e6).mean().item()),
            "gradient_sign_flip_rate": flip_rate,
            "uncertainty_gate_active": True,
        }

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del labels
        clean = clean_bgr.detach().to(self.adapter.device, dtype=torch.float32).clamp(0.0, 255.0)
        boxes = boxes_xyxy.detach().to(clean.device, dtype=clean.dtype)
        object_mask = self._object_mask(
            clean, boxes, self.config.object_mask_weighting
        )
        epsilon = self.config.eps * 255.0
        step_size = self.config.step_size * 255.0
        noise = torch.zeros_like(clean, requires_grad=True)
        risk_momentum: Optional[torch.Tensor] = None
        feature_momentum: Optional[torch.Tensor] = None
        uncertainty_state: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], int]
        ] = None
        diagnostics: List[Dict[str, Any]] = []
        risk_updates = 0
        risk_stop_reason: Optional[str] = None
        feature_carry_steps = int(self.config.risk_to_feature_steps)
        if feature_carry_steps < 0:
            feature_carry_steps = 0
        max_feature_steps = max(self.config.iterations - self.config.risk_warmup, 0)
        if feature_carry_steps > max_feature_steps:
            feature_carry_steps = max_feature_steps
        feature_carry_total = feature_carry_steps
        if not self.config.risk_enabled:
            risk_stop_reason = "disabled"
        elif self.config.risk_warmup == 0:
            risk_stop_reason = "limit_reached"
        generator = torch.Generator(device=clean.device)
        generator.manual_seed(int(seed))

        for step in range(self.config.iterations):
            adversarial = (clean + noise).clamp(0.0, 255.0)
            use_risk = False
            risk_stats = {
                "score_risk": None,
                "score_risk_defined": False,
                "constraint_violation": 0.0,
                "risk_surface": "not_checked",
                "risk_objective": self.config.risk_objective,
                "risk_available_count": 0,
                "risk_aggregate": None,
            }
            risk_state = risk_stop_reason or "checking"
            total_loss: Optional[torch.Tensor] = None
            if risk_stop_reason is None:
                risk_loss, risk_stats = self._risk_loss(adversarial)
                if risk_loss is None:
                    risk_stop_reason = "unavailable"
                    risk_state = risk_stop_reason
                elif (
                    self.config.risk_objective == "max_hinge"
                    and risk_stats["constraint_violation"] <= 0.0
                ):
                    # The repair constraint is already satisfied. Permanently
                    # leave the one-time prefix so stale risk momentum cannot
                    # move a zero-hinge-loss update or re-enter after feature
                    # optimization begins.
                    risk_stop_reason = "satisfied"
                    risk_momentum = None
                    risk_state = risk_stop_reason
                else:
                    use_risk = True
                    total_loss = risk_loss
                    risk_updates += 1
                    if risk_updates >= self.config.risk_warmup:
                        risk_stop_reason = "limit_reached"
                    risk_state = risk_stop_reason or "repairing"
            feature_stats = {
                "feature_objective": self.config.feature_objective,
                "feature_mean_cos": 0.0,
                "feature_max_cos": 0.0,
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
            if total_loss is None:
                total_loss, feature_stats = self._feature_loss(
                    clean, adversarial, object_mask, generator
                )
            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            total_loss.backward()
            if noise.grad is None:
                raise RuntimeError("Attack objective did not produce an image gradient")
            gradient = noise.grad.detach()
            update_stats: Dict[str, Any] = {
                "update_keep_ratio": None,
                "gradient_snr_mean": None,
                "gradient_sign_flip_rate": 0.0,
                "uncertainty_gate_active": False,
                "risk_direction_carryover_applied": False,
                "risk_direction_carryover_steps_remaining": 0,
            }
            if use_risk:
                direction, risk_momentum, update_stats = self._momentum_update(
                    gradient, risk_momentum
                )
                if risk_momentum is not None:
                    feature_momentum = risk_momentum
                    feature_momentum = feature_momentum / (
                        feature_momentum.abs().mean().clamp(min=1.0e-12)
                    )
                    update_stats["risk_direction_carryover_steps_remaining"] = (
                        feature_carry_steps
                    )
            elif self.config.uncertainty_mode == "uncertainty":
                direction, uncertainty_state, update_stats = (
                    self._uncertainty_update(gradient, uncertainty_state)
                )
                carryover_active = (
                    feature_carry_steps > 0 and risk_momentum is not None
                )
                carryover_steps_remaining = 0
                if carryover_active:
                    if feature_momentum is None:
                        feature_momentum = risk_momentum
                    else:
                        feature_momentum = (
                            self.config.momentum * feature_momentum + risk_momentum
                        )
                        feature_momentum = feature_momentum / (
                            feature_momentum.abs().mean().clamp(min=1.0e-12)
                        )
                    carry_ratio = float(feature_carry_steps) / float(
                        max(feature_carry_total, 1)
                    )
                    direction = (
                        (1.0 - carry_ratio) * direction
                        + carry_ratio * feature_momentum.sign()
                    ).sign()
                    feature_carry_steps -= 1
                    if feature_carry_steps < 0:
                        feature_carry_steps = 0
                    carryover_steps_remaining = feature_carry_steps
                update_stats["risk_direction_carryover_applied"] = carryover_active
                update_stats["risk_direction_carryover_steps_remaining"] = (
                    carryover_steps_remaining
                    if carryover_active
                    else 0
                )
            else:
                carryover_active = (
                    feature_carry_steps > 0 and risk_momentum is not None
                )
                carryover_steps_remaining = 0
                if carryover_active:
                    if feature_momentum is None:
                        feature_momentum = risk_momentum
                    else:
                        feature_momentum = (
                            self.config.momentum * feature_momentum + risk_momentum
                        )
                        feature_momentum = feature_momentum / (
                            feature_momentum.abs().mean().clamp(min=1.0e-12)
                        )
                direction, feature_momentum, update_stats = self._momentum_update(
                    gradient, feature_momentum
                )
                update_stats["risk_direction_carryover_applied"] = carryover_active
                if carryover_active:
                    feature_carry_steps -= 1
                    if feature_carry_steps < 0:
                        feature_carry_steps = 0
                    carryover_steps_remaining = feature_carry_steps
                update_stats["risk_direction_carryover_steps_remaining"] = (
                    carryover_steps_remaining
                    if carryover_active
                    else 0
                )
            with torch.no_grad():
                noise -= step_size * direction
                noise.clamp_(-epsilon, epsilon)
                noise.copy_((clean + noise).clamp(0.0, 255.0) - clean)
                saturation = float((noise.abs() >= epsilon - 1.0e-6).float().mean().item())
            diagnostics.append(
                {
                    "step": step,
                    "phase": (
                        "risk_initialization"
                        if use_risk
                        and self.config.risk_objective != "max_hinge"
                        else "risk_repair" if use_risk else "feature"
                    ),
                    "loss": float(total_loss.detach().item()),
                    "linf_pixel": float(noise.detach().abs().max().item()),
                    "budget_saturation": saturation,
                    **risk_stats,
                    **feature_stats,
                    **update_stats,
                    "risk_state": risk_state,
                    "risk_repair_updates": risk_updates,
                    "risk_repair_limit": self.config.risk_warmup,
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
