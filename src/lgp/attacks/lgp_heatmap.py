# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""Explicit paper-first Eq7 raster contract on the unpadded model-image domain."""
from typing import NamedTuple

import torch


class LGPHeatmaps(NamedTuple):
    background: torch.Tensor
    failed_foreground: torch.Tensor
    heatmap: torch.Tensor


def lgp_heatmaps(gt_boxes, failed, height, width, foreground_scale):
    """Return B0, M_failed*H and complete H as 1x1xHxW tensors (eta=1)."""
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (height, width)):
        raise ValueError('Require positive valid-image dimensions')
    if isinstance(foreground_scale, bool) or not isinstance(foreground_scale, (int, float)) or not 0 < foreground_scale < float('inf'):
        raise ValueError('Require finite positive foreground scale')
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] != 4 or not gt_boxes.is_floating_point():
        raise ValueError('Require floating GT xyxy boxes')
    if failed.shape != gt_boxes.shape[:1] or failed.dtype != torch.bool or failed.device != gt_boxes.device:
        raise ValueError('Failed-GT mask mismatch')
    if not torch.isfinite(gt_boxes).all() or bool((gt_boxes[:, 2:] <= gt_boxes[:, :2]).any()):
        raise ValueError('Require finite nondegenerate GT boxes')
    # GT and binary success are fixed metadata, not optimization variables.
    boxes = gt_boxes.detach()
    y = torch.arange(height, dtype=boxes.dtype, device=boxes.device)[:, None] + .5
    x = torch.arange(width, dtype=boxes.dtype, device=boxes.device)[None, :] + .5
    all_foreground = torch.zeros((height, width), dtype=torch.bool, device=boxes.device)
    failed_union = torch.zeros_like(all_foreground)
    best = torch.full((height, width), float('inf'), dtype=boxes.dtype, device=boxes.device)
    for index, box in enumerate(boxes):
        size = box[2:] - box[:2]
        center = (box[:2] + box[2:]) / 2
        half = size * (foreground_scale / 2)
        inside = (x >= center[0] - half[0]) & (x < center[0] + half[0]) & (y >= center[1] - half[1]) & (y < center[1] + half[1])
        all_foreground |= inside
        if bool(failed[index]):
            failed_union |= inside
            radius = ((x - center[0]).square() + (y - center[1]).square()).sqrt() / size.square().sum().sqrt()
            best = torch.minimum(best, torch.where(inside, radius, torch.full_like(radius, float('inf'))))
    heatmap = torch.where(failed_union, best, torch.ones_like(best))
    background = (~all_foreground).to(boxes.dtype)
    failed_weight = failed_union.to(boxes.dtype) * heatmap
    return LGPHeatmaps(background[None, None], failed_weight[None, None], heatmap[None, None])
