# NumbOD adaptation: Copyright (c) 2024 CGCL-codes.
# MIT notice: docs/third_party/licenses/NumbOD-MIT.txt.
# Source: CGCL-codes/NumbOD@0b22cbfb020ea20730fafd23e8e4ebe33b3adda7.
# Modified: local Haar/detector integration; notice added 2026-10-09.
# Existing adaptations predate this notice; execution logic is unchanged.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..adapters import OpenMMLabAdapter
from .base import AttackOutput
from .common import (
    ADAMAX_RUNTIME_COMPATIBILITY,
    background_log_probability,
    build_adamax,
    config_from_mapping,
    diagnostics_record,
    haar_reconstructed_components,
    per_object_iou_assignments,
    per_object_matching_label_assignments,
    project_linf,
    validate_linf_constraint,
)


@dataclass(frozen=True)
class NumbODConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 20
    learning_rate: float = 0.03
    weight_decay: float = 0.02
    iou_candidates: int = 15
    score_candidates: int = 15
    location_weight: float = 1.0
    classification_weight: float = 100.0
    low_frequency_weight: float = 1.0
    high_frequency_weight: float = 1.0

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "NumbODConfig":
        return config_from_mapping(cls, raw)

    def validate(self) -> None:
        validate_linf_constraint(self.eps, self.iterations)
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")


class NumbOD:
    """Repaired spatial-frequency fusion implementation.

    The public release references undefined local variables in its loss and
    projection statements.  This version uses the corresponding instance
    fields, retains dual-track candidate selection and the released
    low-preserve/high-disrupt Haar objective, and records the repair in run
    metadata through the native implementation identifier.
    """

    runtime_compatibility = dict(ADAMAX_RUNTIME_COMPATIBILITY)

    def __init__(self, adapter: OpenMMLabAdapter, config: NumbODConfig) -> None:
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
        del seed
        if labels is None:
            raise ValueError("NumbOD requires ground-truth labels")
        clean = clean_bgr.to(self.adapter.device)
        gt_boxes = boxes.to(clean.device)
        gt_labels = labels.to(clean.device)
        if gt_boxes.numel() == 0:
            # Both released tracks are indexed by GT objects.  With no valid
            # object the two target sets, and hence their sums, are empty.
            # Preserve the input exactly; running only the frequency terms
            # would create a different, background-only attack objective.
            adversarial = clean.detach().clone()
            return AttackOutput(
                adversarial_bgr=adversarial,
                perturbation=torch.zeros_like(adversarial),
                diagnostics=[
                    {
                        "no_attackable_target": True,
                        "empty_target_behavior": "identity_output",
                        "ground_truth_objects": 0,
                        "actual_gradient_evaluations": 0,
                        "regression_track": "empty",
                        "classification_track": "empty",
                    }
                ],
            )
        clean_low, clean_high = haar_reconstructed_components(clean / 255.0)
        perturbation = torch.nn.Parameter(torch.zeros_like(clean))
        optimizer = build_adamax(
            [perturbation],
            learning_rate=self.config.learning_rate * 255.0,
            weight_decay=self.config.weight_decay,
        )
        diagnostics = []
        for step in range(self.config.iterations):
            adversarial = (clean + perturbation).clamp(0.0, 255.0)
            candidates = self.adapter.candidate_predictions(adversarial)[0]
            regression_indices, regression_gt = per_object_iou_assignments(
                candidates,
                gt_boxes,
                self.config.iou_candidates,
            )
            classification_indices, classification_gt = (
                per_object_matching_label_assignments(
                    candidates,
                    gt_boxes,
                    gt_labels,
                    self.config.score_candidates,
                )
            )
            if regression_indices.numel() == 0:
                raise RuntimeError("NumbOD found no regression-track candidate")
            classification_capacity = int(gt_boxes.shape[0]) * int(
                self.config.score_candidates
            )
            classification_padding = max(
                classification_capacity - int(classification_indices.numel()), 0
            )

            # Figure 2 joins the regression- and classification-selected boxes
            # into one 2k attack-target set.  The pinned release likewise
            # concatenates both tracks before applying both attack objectives. A
            # short matching-label track is padded with its GT box, matching
            # the release's fixed-capacity tensor; padded rows carry no direct
            # gradient but remain in the published mean denominator.
            actual_indices = torch.cat(
                (regression_indices, classification_indices), dim=0
            )
            actual_gt = torch.cat((regression_gt, classification_gt), dim=0)
            combined_boxes = [candidates.bboxes[actual_indices]]
            if classification_padding:
                padding_boxes = []
                for gt_index in range(gt_boxes.shape[0]):
                    selected = int((classification_gt == gt_index).sum().item())
                    missing = max(int(self.config.score_candidates) - selected, 0)
                    if missing:
                        padding_boxes.append(gt_boxes[gt_index].expand(missing, 4))
                if padding_boxes:
                    combined_boxes.append(torch.cat(padding_boxes, dim=0))
            combined_boxes_tensor = torch.cat(combined_boxes, dim=0)
            combined_capacity = int(combined_boxes_tensor.shape[0])
            location = F.smooth_l1_loss(
                combined_boxes_tensor,
                torch.zeros_like(combined_boxes_tensor),
            )

            true_scores = candidates.scores[actual_indices]
            if candidates.class_scores is not None:
                class_scores = candidates.class_scores[actual_indices]
                foreground_count = class_scores.shape[1]
                if candidates.background_is_explicit:
                    foreground_count = max(foreground_count - 1, 1)
                selected_labels = gt_labels[actual_gt].clamp(
                    min=0, max=foreground_count - 1
                )
                true_scores = class_scores[
                    torch.arange(actual_indices.numel(), device=clean.device),
                    selected_labels,
                ]
            log_background = background_log_probability(candidates, actual_indices)
            # Paper Eq. (5) uses log(c_gt) - log(c_bg).  The public code's
            # log(1+c_gt) is a material objective change, not a numerical
            # repair.  Missing classification-track slots contribute zero by
            # construction while retaining the release's fixed denominator.
            classification_numerator = (
                true_scores.clamp(min=1.0e-8).log() - log_background
            ).sum()
            classification = classification_numerator / max(combined_capacity, 1)
            adv_low, adv_high = haar_reconstructed_components(
                adversarial / 255.0
            )
            low_frequency = F.smooth_l1_loss(
                adv_low, clean_low.detach()
            )
            high_frequency = -F.smooth_l1_loss(
                adv_high, clean_high.detach()
            )
            loss = (
                self.config.location_weight * location
                + self.config.classification_weight * classification
                + self.config.low_frequency_weight * low_frequency
                + self.config.high_frequency_weight * high_frequency
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                # Preserve the method-prescribed Adamax update.  NumbOD-CM
                # adds only the disclosed common L-infinity projection.
                projected = project_linf(clean, clean + perturbation, self.config.eps)
                perturbation.copy_(projected - clean)
            diagnostics.append(
                diagnostics_record(
                    step,
                    loss,
                    location=location,
                    classification=classification,
                    low_frequency=low_frequency,
                    high_frequency=high_frequency,
                    regression_candidates=int(regression_indices.numel()),
                    classification_candidates=int(
                        classification_indices.numel()
                    ),
                    classification_capacity=classification_capacity,
                    classification_padding=classification_padding,
                    classification_empty_behavior="zero_padded_track",
                    combined_attack_targets=combined_capacity,
                    combined_actual_candidates=int(actual_indices.numel()),
                    loss_target_scope="union_of_both_tracks",
                    classification_equation="log_true_minus_log_background",
                    target_selection_timing="current_adversarial_output",
                    optimizer_step_rule="adamax_then_common_linf_projection",
                    regression_objects=int(regression_gt.unique().numel()),
                    candidate_surface=candidates.surface,
                )
            )
        adversarial = (clean + perturbation.detach()).clamp(0.0, 255.0)
        return AttackOutput(
            adversarial_bgr=adversarial,
            perturbation=(adversarial - clean).detach(),
            diagnostics=diagnostics,
        )
