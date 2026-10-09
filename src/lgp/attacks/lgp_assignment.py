# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""LGP dual-track slots with explicit missing rows, never fake logits."""
from typing import NamedTuple

import torch
from mmdet.structures.bbox import bbox_overlaps


class LGPSlots(NamedTuple):
    indices: torch.Tensor
    real: torch.Tensor
    iou_count: int
    score_count: int

    def real_rows(self):
        owners = torch.arange(self.indices.shape[0], device=self.indices.device)
        owners = owners[:, None].expand_as(self.indices)
        return self.indices[self.real], owners[self.real]


def lgp_assign_slots(reference_boxes, candidate_boxes, rank_scores, foreground_eligible,
                     iou_count, score_count):
    """Keep release dual-track ordering/quotas; -1 marks missing score slots."""
    for count in (iou_count, score_count):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError('LGP track counts must be nonnegative integers')
    if reference_boxes.ndim != 2 or reference_boxes.shape[1] != 4:
        raise ValueError('Expected reference xyxy boxes')
    if candidate_boxes.ndim != 2 or candidate_boxes.shape[1] != 4:
        raise ValueError('Expected candidate xyxy boxes')
    size = candidate_boxes.shape[0]
    if rank_scores.shape != (size,) or foreground_eligible.shape != (size,) or foreground_eligible.dtype != torch.bool:
        raise ValueError('Candidate ranking/eligibility shape mismatch')
    tensors = (reference_boxes, candidate_boxes, rank_scores, foreground_eligible)
    if any(value.device != candidate_boxes.device for value in tensors):
        raise ValueError('Assignment tensors must share device')
    if not all(torch.isfinite(value).all() for value in tensors[:3]):
        raise ValueError('Nonfinite assignment inputs')
    if any(bool((value[:, 2:] < value[:, :2]).any()) for value in tensors[:2]):
        raise ValueError('Inverted assignment boxes')
    owners = reference_boxes.shape[0]
    ni = min(iou_count, size)
    if not owners:
        empty = torch.empty((0, 0), dtype=torch.long, device=candidate_boxes.device)
        return LGPSlots(empty, empty.bool(), 0, 0)
    overlap = bbox_overlaps(reference_boxes.detach(), candidate_boxes.detach())
    iou_indices = overlap.sort(dim=1, descending=True).indices[:, :ni]
    eligible = foreground_eligible.nonzero(as_tuple=True)[0]
    order = rank_scores.detach()[eligible].sort(descending=True).indices
    eligible = eligible[order]
    score_owners = overlap[:, eligible].argmax(dim=0) if eligible.numel() else eligible
    capacities = torch.bincount(score_owners, minlength=owners)
    nonempty = capacities[capacities > 0]
    ns = min(score_count, int(nonempty.min())) if nonempty.numel() else 0
    score_indices = torch.full((owners, ns), -1, dtype=torch.long, device=candidate_boxes.device)
    for owner in range(owners):
        pool = eligible[score_owners == owner]
        if pool.numel():
            score_indices[owner] = pool[:ns]
    indices = torch.cat((iou_indices, score_indices), dim=1)
    return LGPSlots(indices, indices >= 0, ni, ns)


def lgp_ranking(class_logits, semantic_kind, objectness_logits=None):
    """Return score-track ranking/eligibility separately from attack success."""
    if class_logits.ndim != 2 or class_logits.shape[1] == 0 or not torch.isfinite(class_logits).all():
        raise ValueError('Require finite complete class logits')
    if semantic_kind == 'softmax_background':
        if class_logits.shape[1] < 2:
            raise ValueError('Real background requires foreground classes')
        probabilities = class_logits.softmax(dim=-1)
        scores, labels = probabilities.max(dim=-1)
        return scores, labels != class_logits.shape[1] - 1
    probabilities = class_logits.sigmoid()
    if semantic_kind == 'objectness':
        if objectness_logits is None or objectness_logits.shape != class_logits.shape[:1] or not torch.isfinite(objectness_logits).all():
            raise ValueError('Require aligned actual objectness logits')
        probabilities = probabilities * objectness_logits.sigmoid()[:, None]
    elif semantic_kind != 'sigmoid_classes':
        raise ValueError('Unknown source semantic kind')
    # Sigmoid sources have no real background column to exclude from selection.
    return probabilities.max(dim=-1).values, torch.ones(class_logits.shape[0], dtype=torch.bool, device=class_logits.device)


def lgp_object_success(row_success, owner_ids, overlaps, object_count, *, warmup):
    """Existing summed-IoU disappearance or all-real-associated-row no-object."""
    if row_success.dtype != torch.bool or row_success.shape != owner_ids.shape or row_success.ndim != 1:
        raise ValueError('Success row/owner mismatch')
    if owner_ids.dtype != torch.long or overlaps.ndim != 2 or overlaps.shape[0] != object_count:
        raise ValueError('Invalid success association dimensions')
    if object_count < 0 or (owner_ids.numel() and (int(owner_ids.min()) < 0 or int(owner_ids.max()) >= object_count)):
        raise ValueError('Invalid success owner')
    if not torch.isfinite(overlaps).all() or bool((overlaps < 0).any()):
        raise ValueError('Invalid success overlaps')
    result = torch.zeros(object_count, dtype=torch.bool, device=overlaps.device)
    if warmup or overlaps.shape[1] == 0:
        return result
    for owner in range(object_count):
        selected = owner_ids == owner
        semantic = bool(selected.any() and row_success[selected].all())
        spatial = bool(overlaps[owner].sum() < .1)
        result[owner] = semantic or spatial
    return result
