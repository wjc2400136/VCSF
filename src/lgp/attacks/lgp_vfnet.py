# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""LGP-private VFNet pre-NMS rows with complete differentiable logits."""
from typing import NamedTuple

import torch
from mmdet.models.utils import filter_scores_and_topk


class LGPVFNetRows(NamedTuple):
    boxes: torch.Tensor
    logits: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    point_ids: torch.Tensor
    level_ids: torch.Tensor


def lgp_vfnet_rows(head, features, samples):
    """Decode eval-mode VFNet outputs in model-image coordinates, without NMS."""
    if head.training or not head.use_sigmoid_cls:
        raise ValueError('LGP VFNet requires an eval-mode sigmoid head')
    count = head.test_cfg.get('nms_pre')
    threshold = head.test_cfg.get('score_thr')
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('Require explicit positive registered nms_pre')
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
        raise ValueError('Require explicit finite score_thr in [0, 1]')
    outputs = head(features)
    if len(outputs) != 2:
        raise ValueError('VFNet eval must expose logits and refined distances only')
    classifications, distances = outputs
    if not classifications or len(classifications) != len(distances) or not samples:
        raise ValueError('Empty or mismatched VFNet levels/batch')
    for logits, boxes in zip(classifications, distances):
        if logits.ndim != 4 or boxes.ndim != 4:
            raise ValueError('Expected BCHW VFNet maps')
        if logits.shape[:2] != (len(samples), head.num_classes) or boxes.shape[:2] != (len(samples), 4):
            raise ValueError('VFNet batch/class/distance channels mismatch')
        if logits.shape[2:] != boxes.shape[2:] or not logits.numel():
            raise ValueError('VFNet spatial maps mismatch or empty')
        if logits.device != boxes.device or logits.dtype != boxes.dtype:
            raise ValueError('VFNet maps must share device and dtype')
        if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
            raise ValueError('Nonfinite VFNet output')
    priors = head.prior_generator.grid_priors(
        [value.shape[-2:] for value in classifications],
        dtype=classifications[0].dtype, device=classifications[0].device)
    if len(priors) != len(classifications):
        raise ValueError('VFNet prior levels mismatch')
    result = []
    for image_index, sample in enumerate(samples):
        rows = []
        offset = 0
        shape = sample.metainfo['img_shape'][:2]
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError('Invalid model image shape')
        for level, (logit_map, distance_map, points) in enumerate(zip(classifications, distances, priors)):
            logits = logit_map[image_index].permute(1, 2, 0).reshape(-1, head.num_classes)
            boxes = distance_map[image_index].permute(1, 2, 0).reshape(-1, 4)
            if points.shape != (logits.shape[0], 2):
                raise ValueError('VFNet must expose one two-coordinate point per spatial row')
            scores, labels, keep, selected = filter_scores_and_topk(
                logits.sigmoid(), threshold, count, dict(distances=boxes, points=points))
            decoded = head.bbox_coder.decode(selected['points'], selected['distances'], max_shape=shape)
            rows.append(LGPVFNetRows(decoded, logits[keep], scores, labels,
                                      keep + offset, torch.full_like(keep, level)))
            offset += logits.shape[0]
        result.append(LGPVFNetRows(*(torch.cat([row[index] for row in rows]) for index in range(6))))
    return result
