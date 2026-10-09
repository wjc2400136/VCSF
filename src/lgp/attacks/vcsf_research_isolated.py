"""Private ablation switches; the public frozen VCSF module is never modified."""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .base import AttackOutput
from .common import config_from_mapping
from .vcsf_final_candidate import VCSFFinalCandidate, VCSFFinalCandidateConfig


@dataclass(frozen=True)
class VCSFResearchConfig(VCSFFinalCandidateConfig):
    initialization: str = "detector"
    surface: str = "cross_stage"
    levels_per_stage: int = 3
    correspondence: str = "shared"
    spatial_weight: str = "energy"
    aggregation: str = "log_mean_exp"
    gradient_centering: bool = True
    support: str = "valid"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]):
        converted = dict(raw)
        if "feature_levels" in converted:
            converted["feature_levels"] = tuple(converted["feature_levels"])
        return config_from_mapping(cls, converted)

    def reference_parameters(self):
        raw = asdict(self)
        return {field.name: raw[field.name] for field in fields(VCSFFinalCandidateConfig)}

    def is_reference(self):
        return asdict(self) == asdict(type(self)())

    def validate(self):
        options = {
            "initialization": {"detector", "none", "random_sign"},
            "surface": {"neck", "backbone", "cross_stage"},
            "correspondence": {"shared", "independent_offsets", "independent_geometry"},
            "spatial_weight": {"energy", "uniform"},
            "aggregation": {"log_mean_exp", "mean"},
            "support": {"valid", "all_model_locations"},
        }
        for name, allowed in options.items():
            if getattr(self, name) not in allowed:
                raise ValueError("Unknown research factor: " + name)
        if type(self.gradient_centering) is not bool:
            raise ValueError("gradient_centering must be boolean")
        if type(self.levels_per_stage) is not int or self.levels_per_stage not in (1, 2, 3):
            raise ValueError("Only nested original one/two/three-layer subsets are permitted")
        if type(self.iterations) is not int or (self.iterations != 20 and not
                (self.initialization == "random_sign" and self.iterations == 19)):
            raise ValueError("Only the explicit 19-feature random control may use fewer than 20 gradients")
        fixed = (self.eps == 4.0 / 255.0 and self.step_size == 1.0 / 255.0
            and self.feature_eps == 1.0e-6 and tuple(self.feature_levels) == (0, 1, 2))
        if not fixed:
            raise ValueError("The study fixes radius, step, numerical epsilon and parent layer set")
        if not all(math.isfinite(v) for v in (self.scale_min, self.scale_max, self.momentum)):
            raise ValueError("Nonfinite research setting")
        if not 1.0 / 3.0 <= self.scale_min <= self.scale_max <= 5.0 / 3.0:
            raise ValueError("Scale outside the authorized study envelope")
        if not 0.0 <= self.momentum <= 1.0:
            raise ValueError("Momentum outside the authorized study envelope")


def private_seed(seed, domain):
    value = hashlib.sha256((str(int(seed)) + ":" + domain).encode("ascii")).digest()
    return int.from_bytes(value[:8], "big") % (2 ** 63)


def reflect_indices(length, before, after, device):
    if length < 1 or min(before, after) < 0:
        raise ValueError("Invalid reflected extent")
    coordinates = torch.arange(-before, length + after, device=device)
    if length == 1:
        return torch.zeros_like(coordinates)
    period = 2 * (length - 1)
    folded = coordinates.remainder(period)
    return torch.minimum(folded, period - folded).long()


class VCSFResearchCandidate(VCSFFinalCandidate):
    implementation_path = "src/lgp/attacks/vcsf_research_isolated.py"
    method_name = "VCSF isolated retrospective ablation"

    @staticmethod
    def _resize_crop_pad(tensor, new_height, new_width, crop_top, crop_left,
            pad_left, pad_right, pad_top, pad_bottom, *, is_mask=False):
        args = (new_height, new_width, crop_top, crop_left, pad_left, pad_right, pad_top, pad_bottom)
        height = min(new_height, tensor.shape[-2])
        width = min(new_width, tensor.shape[-1])
        native_valid = max(pad_top, pad_bottom) < height and max(pad_left, pad_right) < width
        if is_mask or native_valid:
            return VCSFFinalCandidate._resize_crop_pad(tensor, *args, is_mask=is_mask)
        output = VCSFFinalCandidate._resize_crop_pad(tensor, new_height, new_width,
            crop_top, crop_left, 0, 0, 0, 0)
        rows = reflect_indices(height, pad_top, pad_bottom, output.device)
        cols = reflect_indices(width, pad_left, pad_right, output.device)
        return output.index_select(-2, rows).index_select(-1, cols)

    def _geometry(self, clean, generator, scale=None):
        height, width = clean.shape[-2:]
        if scale is None:
            scale = float(torch.empty((), device=clean.device).uniform_(
                self.config.scale_min, self.config.scale_max, generator=generator).item())
        new_height = max(1, int(round(scale * height)))
        new_width = max(1, int(round(scale * width)))

        def integer(maximum):
            return int(torch.randint(0, maximum + 1, (1,), device=clean.device, generator=generator).item())

        top = integer(max(new_height - height, 0))
        left = integer(max(new_width - width, 0))
        pad_height, pad_width = max(height - new_height, 0), max(width - new_width, 0)
        pad_top, pad_left = integer(pad_height), integer(pad_width)
        return scale, (new_height, new_width, top, left,
            pad_left, pad_width - pad_left, pad_top, pad_height - pad_top)

    def _same_scale_view(self, clean, adversarial, mask, generator, clean_generator=None):
        if self.config.correspondence == "shared":
            return super()._same_scale_view(clean, adversarial, mask, generator)
        scale, adversarial_geometry = self._geometry(clean, generator)
        if clean_generator is None:
            raise RuntimeError("Independent geometry requires an invocation-local clean generator")
        _, clean_geometry = self._geometry(clean, clean_generator,
            scale if self.config.correspondence == "independent_offsets" else None)
        clean_view = self._resize_crop_pad(clean, *clean_geometry).clamp(0., 255.)
        adversarial_view = self._resize_crop_pad(adversarial, *adversarial_geometry).clamp(0., 255.)
        clean_mask = self._resize_crop_pad(mask, *clean_geometry, is_mask=True)
        adversarial_mask = self._resize_crop_pad(mask, *adversarial_geometry, is_mask=True)
        return clean_view, adversarial_view, (clean_mask * adversarial_mask).clamp(0., 1.)

    def _feature_loss(self, clean, adversarial, global_mask, generator, clean_generator=None):
        clean_view, adversarial_view, mask_view = self._same_scale_view(
            clean, adversarial, global_mask, generator, clean_generator)
        neck, backbone = self._feature_only_surfaces(clean_view, adversarial_view)
        loss, stats = self._feature_survival(neck, backbone, mask_view)
        stats.update({"spatial_support_mode": self.config.support + "_" + self.config.spatial_weight,
            "feature_objective": "directional_cosine_" + self.config.aggregation,
            "view_valid_coverage": float(mask_view.detach().mean().item())})
        return loss, stats

    def _select_neck(self, features):
        parent = super()._select_neck(features)
        if len(parent) != 3:
            raise RuntimeError("The complete parent neck surface is required before nested deletion")
        return parent[:self.config.levels_per_stage]

    def _select_backbone(self, features):
        return super()._select_backbone(features)[:self.config.levels_per_stage]

    def _surface_cosines(self, features, mask_view):
        if self.config.spatial_weight == "energy" and self.config.support == "valid":
            return super()._surface_cosines(features, mask_view)
        cosines, coverages = [], []
        for feature in features:
            if feature.shape[0] != 2:
                raise RuntimeError("Research feature extraction must preserve paired batch size two")
            clean_feature, adversarial_feature = feature[:1].detach(), feature[1:2]
            convert = getattr(self.adapter, "model_space_mask", None)
            source_mask = convert(mask_view.to(feature.dtype)) if callable(convert) else mask_view.to(feature.dtype)
            resized = F.interpolate(source_mask, size=feature.shape[-2:], mode="bilinear", align_corners=False).clamp(0., 1.)
            if self.config.support == "all_model_locations":
                resized = torch.ones_like(resized)
            weight = resized
            if self.config.spatial_weight == "energy":
                weight = weight * clean_feature.square().mean(dim=1, keepdim=True)
            weight = weight.detach()
            cosine = (F.normalize(clean_feature, dim=1, eps=self.config.feature_eps) *
                F.normalize(adversarial_feature, dim=1, eps=self.config.feature_eps)).sum(dim=1, keepdim=True)
            cosines.append((weight * cosine).sum() / weight.sum().clamp(min=self.config.feature_eps))
            coverages.append(resized.mean())
        if not cosines:
            raise RuntimeError("Empty configured research surface")
        return torch.stack(cosines), torch.stack(coverages)

    def _feature_survival(self, neck_features, backbone_features, mask_view):
        if (self.config.surface == "cross_stage" and self.config.levels_per_stage == 3
                and self.config.aggregation == "log_mean_exp"):
            return super()._feature_survival(neck_features, backbone_features, mask_view)
        selected = {}
        if self.config.surface in ("neck", "cross_stage"):
            selected["neck"] = self._select_neck(neck_features)
        if self.config.surface in ("backbone", "cross_stage"):
            selected["backbone"] = self._select_backbone(backbone_features)
        scores = [self._surface_cosines(value, mask_view) for value in selected.values()]
        stacked = torch.cat([value[0] for value in scores])
        coverage = torch.cat([value[1] for value in scores])
        loss = self._log_mean_exp(stacked) if self.config.aggregation == "log_mean_exp" else stacked.mean()
        return loss, {"feature_objective": "directional_cosine", "feature_surface": self.config.surface,
            "feature_surface_shapes": {key: self._shape_signature(value) for key, value in selected.items()},
            "feature_surface_level_count": stacked.numel(), "feature_mask_coverage": float(coverage.mean().item()),
            "feature_mean_cos": float(stacked.detach().mean().item()),
            "feature_max_cos": float(stacked.detach().max().item()),
            "feature_survival_evidence": float(loss.detach().item())}

    def _normalize_gradient(self, gradient):
        if self.config.gradient_centering:
            return super()._normalize_gradient(gradient)
        return gradient / gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1.0e-12)

    def __call__(self, clean_bgr, boxes_xyxy, labels=None, seed=42):
        if self.config.is_reference():
            reference = VCSFFinalCandidate(self.adapter,
                VCSFFinalCandidateConfig.from_mapping(self.config.reference_parameters()))
            return reference(clean_bgr, boxes_xyxy, labels, seed)
        clean = clean_bgr.detach().to(self.adapter.device, dtype=torch.float32).clamp(0., 255.)
        if clean.ndim != 4 or clean.shape[0] != 1:
            raise ValueError("Research attack expects one NCHW image")
        epsilon, step_size = self.config.eps * 255., self.config.step_size * 255.
        generator = torch.Generator(device=clean.device).manual_seed(int(seed))
        clean_generator = torch.Generator(device=clean.device).manual_seed(private_seed(seed, "clean_view"))
        noise = torch.zeros_like(clean)
        if self.config.initialization == "random_sign":
            initial_generator = torch.Generator(device=clean.device).manual_seed(private_seed(seed, "initialization"))
            signs = torch.randint(0, 2, clean.shape, device=clean.device, generator=initial_generator).to(clean.dtype) * 2 - 1
            noise = (clean + step_size * signs).clamp(0., 255.) - clean
        noise.requires_grad_(True)
        mask = torch.ones_like(clean[:, :1])
        momentum = None
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial = (clean + noise).clamp(0., 255.)
            use_risk = self.config.initialization == "detector" and step == 0
            if use_risk:
                loss, stats = self._risk_loss(adversarial)
                if loss is None:
                    raise RuntimeError("Frozen detector initialization exposes no gradient")
                phase = "risk_initialization"
            else:
                loss, stats = self._feature_loss(clean, adversarial, mask, generator, clean_generator)
                phase = "feature"
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("Nonfinite research objective")
            self.adapter.model.zero_grad(set_to_none=True)
            if noise.grad is not None:
                noise.grad.zero_()
            loss.backward()
            if noise.grad is None or not bool(torch.isfinite(noise.grad).all()):
                raise RuntimeError("Unavailable or nonfinite image gradient")
            gradient = noise.grad.detach()
            direction, next_state, update_stats = self._momentum_update(gradient, None if use_risk else momentum)
            if not use_risk:
                momentum = next_state
            with torch.no_grad():
                noise -= step_size * direction
                noise.clamp_(-epsilon, epsilon)
                noise.copy_((clean + noise).clamp(0., 255.) - clean)
            diagnostics.append({"step": step, "phase": phase, "loss": float(loss.detach().item()),
                "linf_pixel": float(noise.detach().abs().max().item()),
                "initialization": self.config.initialization, "surface": self.config.surface,
                "spatial_weight": self.config.spatial_weight, "aggregation": self.config.aggregation,
                "support": self.config.support, "correspondence": self.config.correspondence,
                "gradient_centering": self.config.gradient_centering,
                "image_gradient_abs_mean": float(gradient.abs().mean().item()),
                "risk_direction_carryover_applied": False, "uncertainty_gate_active": False,
                **stats, **update_stats})
            noise.grad.zero_()
        adversarial = (clean + noise.detach()).clamp(0., 255.)
        return AttackOutput(adversarial_bgr=adversarial, perturbation=adversarial - clean, diagnostics=diagnostics)
