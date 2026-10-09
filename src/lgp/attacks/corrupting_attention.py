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
    project_linf,
    random_linf_start,
    validate_linf,
)


@dataclass(frozen=True)
class CorruptingAttentionConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 10
    step_size: float = 2.0 / 255.0
    confidence_threshold: float = 0.5
    corruption: str = "dispersion"
    random_start: bool = True
    empty_target_behavior: str = "identity_output"

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "CorruptingAttentionConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf(self.eps, self.iterations, self.step_size)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must lie in [0, 1]")
        if self.corruption not in {
            "dispersion",
            "reranking",
            "permutation",
            "peak_suppression",
        }:
            raise ValueError("Unsupported attention corruption target")
        if self.empty_target_behavior not in {
            "identity_output",
            "identity_budget_padding",
        }:
            raise ValueError("Unsupported empty-target behavior")


class CorruptingAttention:
    """Pre-softmax encoder-attention corruption for detection transformers."""

    def __init__(
        self, adapter: OpenMMLabAdapter, config: CorruptingAttentionConfig
    ) -> None:
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations
        self.auxiliary_forward_passes_per_image = 1
        self.auxiliary_backward_passes_per_image = 0

    @staticmethod
    def _reshape_logits(surface: Dict[str, Any]) -> torch.Tensor:
        logits = surface["logits"]
        heads = int(surface["num_heads"])
        levels = int(surface["num_levels"])
        points = int(surface["num_points"])
        expected = heads * levels * points
        if logits.ndim != 3 or logits.shape[-1] != expected:
            raise RuntimeError(
                "Attention-logit width {} differs from {} heads x levels x points".format(
                    logits.shape[-1] if logits.ndim else -1, expected
                )
            )
        return logits.view(logits.shape[0], logits.shape[1], heads, levels * points)

    def _object_query_mask(
        self,
        surface: Dict[str, Any],
        candidates,
        raw_shape,
    ) -> torch.Tensor:
        if len(candidates) != 1:
            raise ValueError("Corrupting Attention currently requires batch size one")
        candidate = candidates[0]
        keep = candidate.scores.detach() >= self.config.confidence_threshold
        boxes = candidate.bboxes.detach()[keep]
        query_count = int(surface["logits"].shape[1])
        if boxes.numel() == 0:
            return torch.zeros(
                query_count, dtype=torch.bool, device=surface["logits"].device
            )

        reference_points = surface["reference_points"].detach()
        spatial_shapes = surface["spatial_shapes"].detach().to(dtype=torch.long)
        level_start_index = surface["level_start_index"].detach().to(dtype=torch.long)
        if (
            reference_points.ndim != 4
            or reference_points.shape[0] != 1
            or reference_points.shape[1] != query_count
            or reference_points.shape[-1] != 2
        ):
            raise RuntimeError("Unexpected deformable-attention reference-point shape")
        if int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) != query_count:
            raise RuntimeError("Encoder query count differs from registered spatial shapes")

        query_indices = torch.arange(query_count, device=reference_points.device)
        own_level = torch.bucketize(
            query_indices,
            level_start_index[1:].to(device=query_indices.device),
            right=True,
        )
        coordinates = reference_points[
            0, query_indices, own_level, :
        ]
        geometry = self.adapter.input_geometry(*raw_shape)
        padded_height, padded_width = geometry["padded_shape"]
        scale = coordinates.new_tensor((padded_width, padded_height))
        coordinates = coordinates * scale
        boxes = self.adapter.model_space_boxes(boxes, raw_shape)
        inside = (
            (coordinates[:, None, 0] >= boxes[None, :, 0])
            & (coordinates[:, None, 0] <= boxes[None, :, 2])
            & (coordinates[:, None, 1] >= boxes[None, :, 1])
            & (coordinates[:, None, 1] <= boxes[None, :, 3])
        )
        return inside.any(dim=1)

    def _target(
        self, clean_rows: torch.Tensor, generator: torch.Generator
    ) -> torch.Tensor:
        if self.config.corruption == "dispersion":
            return torch.zeros_like(clean_rows)
        if self.config.corruption == "reranking":
            return clean_rows.amax(dim=-1, keepdim=True) - clean_rows
        if self.config.corruption == "permutation":
            random_order = torch.rand(
                clean_rows.shape,
                device=clean_rows.device,
                dtype=clean_rows.dtype,
                generator=generator,
            ).argsort(dim=-1)
            return clean_rows.gather(-1, random_order)
        minimum = clean_rows.amin(dim=-1, keepdim=True)
        maximum = clean_rows.amax(dim=-1, keepdim=True)
        minimum_index = clean_rows.argmin(dim=-1, keepdim=True)
        target = minimum.expand_as(clean_rows).clone()
        return target.scatter(-1, minimum_index, maximum)

    def __call__(
        self,
        clean_bgr: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        seed: int = 42,
    ) -> AttackOutput:
        del boxes, labels
        clean = clean_bgr.to(self.adapter.device)
        if clean.shape[0] != 1:
            raise ValueError("Corrupting Attention currently requires batch size one")
        generator = make_generator(clean.device, seed)
        with torch.no_grad():
            clean_surface, clean_candidates = (
                self.adapter.deformable_encoder_attention_surface_and_candidates(clean)
            )
            clean_logits = self._reshape_logits(clean_surface)
            object_queries = self._object_query_mask(
                clean_surface,
                clean_candidates,
                tuple(int(item) for item in clean.shape[-2:]),
            )
            clean_detection_count = int(
                (
                    clean_candidates[0].scores.detach()
                    >= self.config.confidence_threshold
                ).sum().item()
            )
            if object_queries.any():
                clean_rows = clean_logits[:, object_queries, :, :].detach()
                target_rows = self._target(clean_rows, generator).detach()
            else:
                clean_rows = clean_logits[:, :0, :, :].detach()
                target_rows = clean_rows

        if not object_queries.any():
            actual_gradient_evaluations = 0
            if self.config.empty_target_behavior == "identity_budget_padding":
                # The paper does not define its normalized objective when the
                # clean object-token set is empty. Compute matching therefore
                # pads the scheduled source backward passes with a connected
                # zero loss while preserving the exact identity output. This
                # spends budget but introduces no fallback target or signal.
                for _ in range(self.config.iterations):
                    point = clean.detach().clone().requires_grad_(True)
                    surface, _ = (
                        self.adapter.deformable_encoder_attention_surface_and_candidates(
                            point
                        )
                    )
                    zero_loss = self._reshape_logits(surface).sum() * 0.0
                    gradient = torch.autograd.grad(zero_loss, point)[0]
                    if not torch.isfinite(gradient).all() or torch.count_nonzero(
                        gradient
                    ):
                        raise RuntimeError("Empty-target budget padding must be zero")
                    actual_gradient_evaluations += 1
            diagnostic = {
                "step": 0,
                "loss": 0.0,
                "empty_target_behavior": "identity_output",
                "empty_target_accounting": self.config.empty_target_behavior,
                "actual_gradient_evaluations": actual_gradient_evaluations,
                "reason": "no_clean_detection_token_above_threshold",
                "confidence_threshold": self.config.confidence_threshold,
                "clean_detection_count": clean_detection_count,
                "encoder_layer": int(clean_surface["encoder_layer"]),
            }
            return AttackOutput(
                adversarial_bgr=clean.detach().clone(),
                perturbation=torch.zeros_like(clean),
                diagnostics=[diagnostic],
            )

        adversarial = (
            random_linf_start(clean, self.config.eps, generator)
            if self.config.random_start
            else clean.detach().clone()
        )
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial.requires_grad_(True)
            surface, _ = (
                self.adapter.deformable_encoder_attention_surface_and_candidates(
                    adversarial
                )
            )
            rows = self._reshape_logits(surface)[:, object_queries, :, :]
            if rows.shape != target_rows.shape:
                raise RuntimeError("Encoder attention geometry changed during attack")
            loss = (rows - target_rows).square().mean()
            gradient = torch.autograd.grad(loss, adversarial)[0]
            adversarial = project_linf(
                clean,
                adversarial - self.config.step_size * 255.0 * gradient.sign(),
                self.config.eps,
            ).detach()
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    corruption=self.config.corruption,
                    object_query_count=int(object_queries.sum().item()),
                    object_query_fraction=float(
                        object_queries.float().mean().item()
                    ),
                    clean_detection_count=clean_detection_count,
                    confidence_threshold=self.config.confidence_threshold,
                    encoder_layer=int(surface["encoder_layer"]),
                    optimization_space="pre_softmax_deformable_attention_logits",
                    gradient_image_equivalents=step + 1,
                )
            )
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
