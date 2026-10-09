# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""Paper-first LGP executor for the registered compute-matched benchmark."""
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_overlaps

from .base import AttackOutput
from .common import (ADAMAX_RUNTIME_COMPATIBILITY, build_adamax, config_from_mapping,
                     make_generator, project_linf, random_linf_start, validate_linf_constraint)
from .lgp_assignment import lgp_assign_slots, lgp_object_success
from .lgp_fidelity import lgp_shape_loss, lgp_perceptibility_loss
from .lgp_heatmap import lgp_heatmaps
from .lgp_source import LGPSource


@dataclass(frozen=True)
class LGPConfig:
    eps: float = 4.0 / 255.0
    iterations: int = 20
    learning_rate: float = 0.10
    weight_decay: float = 0.0
    iou_candidates: int = 5
    score_candidates: int = 5
    localization_weight: float = 1.0
    shape_weight: float = 1.0
    classification_weight: float = 1.0
    imperceptibility_weight: float = 0.1
    foreground_scale: float = 3.0 / 2.0
    shape_scale: float = 1.0 / 10.0
    target_source: str = 'hq'
    prediction_threshold: float = 1.0 / 20.0
    perceptibility_mode: str = 'adaptive_fbs'
    random_start: bool = False

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]):
        return config_from_mapping(cls, raw)

    def validate(self):
        validate_linf_constraint(self.eps, self.iterations)
        if self.learning_rate <= 0 or self.foreground_scale <= 0 or self.shape_scale <= 0:
            raise ValueError('LGP rates and scales must be positive')
        if self.iou_candidates < 0 or self.score_candidates < 0:
            raise ValueError('LGP candidate counts must be nonnegative')
        if self.target_source not in {'hq', 'pre_nms', 'post_nms'}:
            raise ValueError('Unknown LGP target source')
        if not 0 <= self.prediction_threshold <= 1:
            raise ValueError('Invalid LGP prediction threshold')
        if self.perceptibility_mode not in {'image', 'image_l2', 'static_fbs', 'adaptive_fbs'}:
            raise ValueError('Unknown LGP perceptibility mode')


class LGP:
    runtime_compatibility = dict(ADAMAX_RUNTIME_COMPATIBILITY)

    def __init__(self, adapter, config):
        config.validate()
        self.adapter = adapter
        self.config = config
        self.gradient_evaluations_per_image = config.iterations

    def gradient_evaluations_for_output(self, output):
        rows = output.diagnostics
        if len(rows) == 1 and rows[0].get('empty_target_behavior') == 'identity_output':
            if rows[0].get('actual_gradient_evaluations') != 0:
                raise RuntimeError('Invalid LGP identity accounting')
            return 0
        if not rows:
            raise RuntimeError('Invalid LGP iteration accounting')
        stopped = rows[-1].get('early_stop_reason') == 'empty_later_real_tracked_set'
        updates = rows[:-1] if stopped else rows
        if len(updates) > self.gradient_evaluations_per_image or (
                stopped and (len(updates) >= self.gradient_evaluations_per_image
                             or rows[-1].get('step') != len(updates)
                             or rows[-1].get('completed_updates') != len(updates)
                             or rows[-1].get('actual_gradient_evaluations') != len(updates)
                             or rows[-1].get('selected_candidates') != 0
                             or not rows[-1].get('ground_truth_objects'))):
            raise RuntimeError('Invalid LGP empty-track accounting')
        if [row.get('actual_gradient_evaluations') for row in updates] != list(range(1, len(updates) + 1)):
            raise RuntimeError('LGP iteration accounting is not contiguous')
        return len(updates)

    def source_row_forwards_for_output(self, output):
        gradients = self.gradient_evaluations_for_output(output)
        rows = output.diagnostics
        if len(rows) == 1 and rows[0].get('empty_target_behavior') == 'identity_output':
            expected = [0]
        else:
            expected = list(range(2, gradients + 2))
            if rows[-1].get('early_stop_reason') == 'empty_later_real_tracked_set':
                expected.append(gradients + 2)
        if [row.get('source_forwards') for row in rows] != expected:
            raise RuntimeError('LGP source forward accounting differs')
        return expected[-1]

    def auxiliary_forward_passes_for_output(self, output):
        return self.source_row_forwards_for_output(output) - self.gradient_evaluations_for_output(output)

    def __call__(self, clean_bgr, boxes, labels: Optional[torch.Tensor] = None, seed=42):
        del labels
        clean = clean_bgr.detach().to(self.adapter.device)
        gt_raw = boxes.detach().to(clean)
        if clean.ndim != 4 or clean.shape[:2] != (1, 3) or not clean.is_floating_point() or not torch.isfinite(clean).all():
            raise ValueError('LGP requires one finite BGR image')
        if bool(((clean < 0) | (clean > 255)).any()):
            raise ValueError('LGP clean pixels outside [0,255]')
        if gt_raw.ndim != 2 or gt_raw.shape[1] != 4:
            raise ValueError('LGP requires GT xyxy boxes')
        if not torch.isfinite(gt_raw).all() or bool((gt_raw[:, 2:] <= gt_raw[:, :2]).any()):
            raise ValueError('LGP requires finite nondegenerate GT boxes')
        if gt_raw.shape[0] == 0:
            return AttackOutput(clean.clone(), torch.zeros_like(clean), [{
                'no_attackable_target': True, 'empty_target_behavior': 'identity_output',
                'ground_truth_objects': 0, 'actual_gradient_evaluations': 0,
                'source_forwards': 0, 'target_source': self.config.target_source}])
        source = LGPSource(self.adapter, clean)
        gt = source.model_boxes(gt_raw)
        with torch.no_grad():
            clean_loss_image = source.loss_image(clean).detach()
            if self.config.target_source == 'post_nms':
                original = source.post_boxes(clean, self.config.prediction_threshold)
                original_owners = bbox_overlaps(original, gt).argmax(dim=1)
                missing_clean_slots = 0
            else:
                clean_rows = source.rows(clean)
                if self.config.target_source == 'hq':
                    rank, eligible = clean_rows.ranking()
                    slots = lgp_assign_slots(gt, clean_rows.boxes, rank, eligible,
                                             self.config.iou_candidates, self.config.score_candidates)
                    indices, original_owners = slots.real_rows()
                    original = clean_rows.boxes[indices]
                    missing_clean_slots = int((~slots.real).sum())
                else:
                    original = clean_rows.boxes
                    original_owners = bbox_overlaps(original, gt).argmax(dim=1)
                    missing_clean_slots = 0
            if original.shape[0] == 0:
                raise RuntimeError('LGP valid GT but no real clean target')
            original = original.detach().clone()
            original_owners = original_owners.detach().clone()
            source.bind_fixed(original)
        start = random_linf_start(clean, self.config.eps, make_generator(clean.device, seed)) if self.config.random_start else clean
        q = torch.nn.Parameter(source.coordinates.normalize(start).detach().clone())
        optimizer = build_adamax([q], learning_rate=self.config.learning_rate, weight_decay=self.config.weight_decay)
        failed = torch.ones(gt.shape[0], dtype=torch.bool, device=clean.device)
        diagnostics = []
        for step in range(self.config.iterations):
            if not bool(failed.any()):
                break
            raw = source.coordinates.pixels(q)
            rows = source.rows(raw)
            if source.fixed_roi:
                if rows.boxes.shape[0] != original.shape[0]:
                    raise RuntimeError('LGP fixed ROI row count changed')
                indices = torch.arange(original.shape[0], device=clean.device)
                owners = original_owners
                missing_slots = 0
            else:
                rank, eligible = rows.ranking()
                slots = lgp_assign_slots(original, rows.boxes, rank, eligible, 1, 1)
                indices, reference_ids = slots.real_rows()
                owners = original_owners[reference_ids]
                missing_slots = int((~slots.real).sum())
            if indices.numel() == 0:
                diagnostics.append(dict(
                    step=step, early_stop_reason='empty_later_real_tracked_set',
                    ground_truth_objects=int(gt.shape[0]),
                    original_targets=int(original.shape[0]), selected_candidates=0,
                    missing_clean_slots=missing_clean_slots,
                    missing_tracked_slots=missing_slots,
                    unresolved_objects=int(failed.sum()), completed_updates=step,
                    actual_gradient_evaluations=step, source_forwards=source.forwards,
                    target_source=self.config.target_source,
                    perceptibility_mode=self.config.perceptibility_mode))
                break
            selected = rows.boxes[indices]
            assigned = gt[owners]
            iou = bbox_overlaps(selected, assigned, is_aligned=True)
            centers = (selected[:, :2] + selected[:, 2:]) / 2
            gt_centers = (assigned[:, :2] + assigned[:, 2:]) / 2
            localization = (iou - F.smooth_l1_loss(centers, gt_centers, reduction='none').sum(dim=1)).mean()
            shape = lgp_shape_loss((selected[:, 2:] - selected[:, :2]).clamp(min=0), self.config.shape_scale)
            classification, row_success = rows.semantic(indices)
            success = lgp_object_success(row_success.detach(), owners,
                                         bbox_overlaps(gt, rows.boxes.detach()), gt.shape[0], warmup=step <= 1)
            failed = ~success
            heatmap_failed = torch.ones_like(failed) if self.config.perceptibility_mode == 'static_fbs' else failed
            weights = lgp_heatmaps(gt, heatmap_failed, *source.valid_shape, self.config.foreground_scale)
            delta = source.loss_image(raw) - clean_loss_image
            perceptibility = lgp_perceptibility_loss(delta, *weights, self.config.perceptibility_mode)
            loss = (self.config.localization_weight * localization + self.config.shape_weight * shape +
                    self.config.classification_weight * classification + self.config.imperceptibility_weight * perceptibility)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite LGP loss')
            optimizer.zero_grad()
            loss.backward()
            if q.grad is None or not torch.isfinite(q.grad).all():
                raise RuntimeError('Missing or nonfinite LGP image gradient')
            optimizer.step()
            with torch.no_grad():
                projected = project_linf(clean, source.coordinates.pixels(q), self.config.eps)
                q.copy_(source.coordinates.normalize(projected))
            diagnostics.append(dict(step=step, loss=float(loss.detach()),
                localization=float(localization.detach()), shape=float(shape.detach()),
                classification=float(classification.detach()), imperceptibility=float(perceptibility.detach()),
                selected_candidates=int(indices.numel()), original_targets=int(original.shape[0]),
                missing_clean_slots=missing_clean_slots, missing_tracked_slots=missing_slots,
                successful_objects=int(success.sum()), success_checked_before_update=True,
                actual_gradient_evaluations=step + 1, source_forwards=source.forwards,
                valid_loss_shape=list(source.valid_shape), target_source=self.config.target_source,
                perceptibility_mode=self.config.perceptibility_mode,
                optimizer_step_rule='adamax_normalized_q_then_common_raw_linf_projection',
                perceptibility_domain='pre_defense_valid_model_image'))
        adversarial = project_linf(clean, source.coordinates.pixels(q), self.config.eps).detach()
        return AttackOutput(adversarial, (adversarial - clean).detach(), diagnostics)
