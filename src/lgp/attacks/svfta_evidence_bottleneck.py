from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .base import AttackOutput
from .svfta import SVFTA, SVFTAConfig


@dataclass(frozen=True)
class SVFTAEvidenceBottleneckConfig(SVFTAConfig):
    """Frozen development-only factorial over detector evidence surfaces."""

    persistent_output_evidence: bool = False
    include_backbone_surface: bool = False

    def validate(self) -> None:
        super().validate()
        required = {
            "iterations": self.iterations == 20,
            "eps": math.isclose(self.eps, 4.0 / 255.0),
            "step_size": math.isclose(self.step_size, 1.0 / 255.0),
            "scale_min": math.isclose(self.scale_min, 2.0 / 3.0),
            "scale_max": math.isclose(self.scale_max, 4.0 / 3.0),
            "risk_enabled": self.risk_enabled is True,
            "risk_objective": self.risk_objective == "smooth_log_odds",
            "risk_warmup": self.risk_warmup == 1,
            "risk_to_feature_steps": self.risk_to_feature_steps == 0,
            "feature_objective": self.feature_objective == "directional_cosine",
            "feature_levels": tuple(self.feature_levels) == (0, 1, 2),
            "object_mask_weighting": self.object_mask_weighting == "union",
            "uncertainty_mode": self.uncertainty_mode == "uncertainty",
            "uncertainty_decay": math.isclose(
                self.uncertainty_decay, 7.0 / 10.0
            ),
            "uncertainty_kappa": math.isclose(
                self.uncertainty_kappa, 1.0 / 2.0
            ),
        }
        failed = [name for name, valid in required.items() if not valid]
        if failed:
            raise ValueError(
                "Evidence-bottleneck confirmation freezes: {}".format(
                    ", ".join(failed)
                )
            )
        if not isinstance(self.persistent_output_evidence, bool):
            raise ValueError("persistent_output_evidence must be boolean")
        if not isinstance(self.include_backbone_surface, bool):
            raise ValueError("include_backbone_surface must be boolean")


class SVFTAEvidenceBottleneck(SVFTA):
    """Isolated two-factor evidence-bottleneck confirmation.

    The exact bridge is the ``False, False`` reference. The other cells test
    whether the nineteen feature updates should also suppress differentiable
    candidate confidence and whether clean-aligned evidence should be removed
    from both neck and backbone surfaces. Joint cells obtain every surface
    from one detector forward and retain one backward per iteration.
    """

    implementation_path = "src/lgp/attacks/svfta_evidence_bottleneck.py"
    _BACKBONE_LEVEL_COUNT = 3

    def __init__(
        self,
        adapter: Any,
        config: SVFTAEvidenceBottleneckConfig,
    ) -> None:
        super().__init__(adapter, config)
        self.config = config
        self.auxiliary_forward_passes_per_image = 0
        self.auxiliary_backward_passes_per_image = 0

    @staticmethod
    def _log_mean_exp(values: torch.Tensor) -> torch.Tensor:
        flattened = values.reshape(-1)
        if flattened.numel() == 0:
            raise RuntimeError("Evidence bottleneck received an empty surface")
        return torch.logsumexp(flattened, dim=0) - math.log(
            int(flattened.numel())
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
        return [[int(value) for value in feature.shape] for feature in features]

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
            raise RuntimeError(
                "Evidence bottleneck requires three backbone feature maps"
            )
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
                    "Evidence surface did not preserve clean/adversarial batch two"
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
            raise RuntimeError("Evidence surface exposed no usable feature map")
        return torch.stack(cosines), torch.stack(coverages)

    def _feature_survival(
        self,
        neck_features: Sequence[torch.Tensor],
        backbone_features: Optional[Sequence[torch.Tensor]],
        mask_view: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        selected_neck = self._select_neck(neck_features)
        neck_cosines, neck_coverages = self._surface_cosines(
            selected_neck,
            mask_view,
        )
        all_cosines = [neck_cosines]
        all_coverages = [neck_coverages]
        selected_backbone: Tuple[torch.Tensor, ...] = ()
        if self.config.include_backbone_surface:
            if backbone_features is None:
                raise RuntimeError("Backbone evidence was requested but not captured")
            selected_backbone = self._select_backbone(backbone_features)
            backbone_cosines, backbone_coverages = self._surface_cosines(
                selected_backbone,
                mask_view,
            )
            all_cosines.append(backbone_cosines)
            all_coverages.append(backbone_coverages)

        stacked = torch.cat(all_cosines)
        coverage = torch.cat(all_coverages)
        survival = self._log_mean_exp(stacked)
        return survival, {
            "feature_objective": "directional_cosine_evidence_bottleneck",
            "feature_mean_cos": float(stacked.detach().mean().item()),
            "feature_max_cos": float(stacked.detach().max().item()),
            "feature_mask_coverage": float(coverage.detach().mean().item()),
            "feature_surface": (
                "neck_backbone" if selected_backbone else "neck"
            ),
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
        }

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

    def _joint_surfaces_and_candidates(
        self,
        clean_view: torch.Tensor,
        adversarial_view: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, ...],
        Optional[Tuple[torch.Tensor, ...]],
        Any,
    ]:
        model = self.adapter.model
        neck = getattr(model, "neck", None)
        if neck is None or not hasattr(neck, "register_forward_hook"):
            raise RuntimeError(
                "Persistent evidence requires a hookable detector neck"
            )
        backbone = getattr(model, "backbone", None)
        if self.config.include_backbone_surface and (
            backbone is None or not hasattr(backbone, "register_forward_hook")
        ):
            raise RuntimeError("Detector exposes no hookable backbone")

        captured_neck: List[Tuple[torch.Tensor, ...]] = []
        captured_backbone: List[Tuple[torch.Tensor, ...]] = []

        def capture_neck(_module: Any, _inputs: Any, output: Any) -> None:
            tensors = self._captured_tensors(output)
            if tensors:
                captured_neck.append(tensors)

        def capture_backbone(_module: Any, _inputs: Any, output: Any) -> None:
            tensors = self._captured_tensors(output)
            if tensors:
                captured_backbone.append(tensors)

        handles = [neck.register_forward_hook(capture_neck)]
        if self.config.include_backbone_surface:
            handles.append(backbone.register_forward_hook(capture_backbone))
        try:
            candidates = self.adapter.candidate_predictions(
                torch.cat((clean_view.detach(), adversarial_view), dim=0)
            )
        finally:
            for handle in handles:
                handle.remove()
        if len(captured_neck) != 1:
            raise RuntimeError(
                "Expected one neck capture, received {}".format(
                    len(captured_neck)
                )
            )
        if self.config.include_backbone_surface and len(captured_backbone) != 1:
            raise RuntimeError(
                "Expected one backbone capture, received {}".format(
                    len(captured_backbone)
                )
            )
        if len(candidates) != 2:
            raise RuntimeError("Candidate forward did not preserve batch two")
        return (
            captured_neck[0],
            captured_backbone[0] if captured_backbone else None,
            candidates[1],
        )

    def _output_survival(self, candidates: Any) -> Tuple[torch.Tensor, Dict[str, Any]]:
        scores = getattr(candidates, "scores", None)
        if not isinstance(scores, torch.Tensor) or scores.numel() == 0:
            raise RuntimeError("Candidate surface exposed no differentiable score")
        probabilities = scores.reshape(-1).clamp(0.0, 1.0)
        bounded_survival = 2.0 * probabilities - 1.0
        survival = self._log_mean_exp(bounded_survival)
        return survival, {
            "score_risk": float(survival.detach().item()),
            "score_risk_defined": True,
            "constraint_violation": 0.0,
            "risk_surface": "candidate_same_forward",
            "risk_objective": "bounded_candidate_survival",
            "risk_available_count": int(probabilities.numel()),
            "risk_aggregate": float(probabilities.detach().mean().item()),
            "output_survival_evidence": float(survival.detach().item()),
        }

    def _feature_loss(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        object_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if not self.config.include_backbone_surface:
            return super()._feature_loss(
                clean,
                adversarial,
                object_mask,
                generator,
            )
        clean_view, adversarial_view, mask_view = self._same_scale_view(
            clean,
            adversarial,
            object_mask,
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

    def _joint_loss(
        self,
        clean: torch.Tensor,
        adversarial: torch.Tensor,
        object_mask: torch.Tensor,
        generator: torch.Generator,
    ) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        clean_view, adversarial_view, mask_view = self._same_scale_view(
            clean,
            adversarial,
            object_mask,
            generator,
        )
        neck_features, backbone_features, candidates = (
            self._joint_surfaces_and_candidates(clean_view, adversarial_view)
        )
        output_survival, risk_stats = self._output_survival(candidates)
        feature_survival, feature_stats = self._feature_survival(
            neck_features,
            backbone_features,
            mask_view,
        )
        total_loss = self._log_mean_exp(
            torch.stack((output_survival, feature_survival))
        )
        feature_stats["joint_evidence_mode"] = "survival_bottleneck"
        feature_stats["joint_survival_evidence"] = float(
            total_loss.detach().item()
        )
        return total_loss, risk_stats, feature_stats

    @staticmethod
    def _empty_feature_stats(
        include_backbone: bool,
        coverage: float,
    ) -> Dict[str, Any]:
        return {
            "feature_objective": "none",
            "feature_mean_cos": None,
            "feature_max_cos": None,
            "feature_mask_coverage": coverage,
            "feature_surface": (
                "neck_backbone" if include_backbone else "neck"
            ),
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
        }

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes_xyxy: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        if not self.config.persistent_output_evidence:
            return super().__call__(
                clean_bgr,
                boxes_xyxy,
                labels=labels,
                seed=seed,
            )

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
        uncertainty_state: Optional[
            Tuple[torch.Tensor, Optional[torch.Tensor], int]
        ] = None
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
                feature_stats = self._empty_feature_stats(
                    self.config.include_backbone_surface,
                    float(object_mask.mean().item()),
                )
                phase = "risk_initialization"
            else:
                total_loss, risk_stats, feature_stats = self._joint_loss(
                    clean,
                    adversarial,
                    object_mask,
                    generator,
                )
                phase = "joint_evidence_bottleneck"

            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            total_loss.backward()
            if noise.grad is None:
                raise RuntimeError(
                    "Evidence bottleneck did not produce an image gradient"
                )
            gradient = noise.grad.detach()
            if step == 0:
                direction, _initial_state, update_stats = (
                    self._momentum_update(gradient, None)
                )
            else:
                direction, uncertainty_state, update_stats = (
                    self._uncertainty_update(gradient, uncertainty_state)
                )
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
                        "fixed_initialization" if step == 0 else "joint"
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
