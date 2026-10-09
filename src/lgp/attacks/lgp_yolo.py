# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""Disclosed LGP YOLOv3 extension retaining actual objectness logits."""
import copy

import torch
from mmengine.structures import InstanceData
from mmdet.models.utils import filter_scores_and_topk


def lgp_yolo_rows(head, features, samples, candidate_limit):
    """Preserve existing pre-NMS adapter selection, with aligned private fields."""
    if head.training:
        raise ValueError('LGP YOLO requires eval mode')
    if isinstance(candidate_limit, bool) or not isinstance(candidate_limit, int) or candidate_limit <= 0:
        raise ValueError('Require positive candidate_limit')
    if head.num_attrib != head.num_classes + 5 or not samples:
        raise ValueError('Unexpected YOLO attribute/batch contract')
    outputs = head(features)
    if len(outputs) != 1:
        raise ValueError('YOLO head must expose one sequence of prediction maps')
    maps = outputs[0]
    if len(maps) != head.num_levels or len(maps) != len(head.featmap_strides) or not maps:
        raise ValueError('YOLO prediction levels mismatch')
    batch = len(samples)
    for value, anchors_per_point in zip(maps, head.prior_generator.num_base_priors):
        if value.ndim != 4 or value.shape[0] != batch or value.shape[1] != anchors_per_point * head.num_attrib or not value.numel():
            raise ValueError('YOLO prediction map shape mismatch')
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite YOLO prediction map')
    anchors = head.prior_generator.grid_priors([value.shape[-2:] for value in maps], device=maps[0].device)
    if len(anchors) != len(maps):
        raise ValueError('YOLO anchor levels mismatch')
    flattened, strides, level_ids = [], [], []
    for level, (value, stride, prior) in enumerate(zip(maps, head.featmap_strides, anchors)):
        raw = value.permute(0, 2, 3, 1).reshape(batch, -1, head.num_attrib)
        if prior.shape != (raw.shape[1], 4):
            raise ValueError('YOLO anchor count mismatch')
        flattened.append(raw)
        strides.append(raw.new_tensor(stride).expand(raw.shape[1]))
        level_ids.append(torch.full((raw.shape[1],), level, dtype=torch.long, device=raw.device))
    raw = torch.cat(flattened, dim=1)
    # Never apply the upstream in-place xy activation to the source prediction maps.
    encoded = torch.cat((raw[..., :2].sigmoid(), raw[..., 2:4]), dim=-1)
    boxes = head.bbox_coder.decode(torch.cat(anchors), encoded, torch.cat(strides).unsqueeze(-1))
    if not torch.isfinite(boxes).all():
        raise ValueError('Nonfinite decoded YOLO boxes')
    levels = torch.cat(level_ids)
    cfg = copy.deepcopy(head.test_cfg)
    cfg.update(score_thr=0.0, conf_thr=-1.0, nms_pre=candidate_limit)
    result = []
    for index, sample in enumerate(samples):
        scores, labels, keep, _ = filter_scores_and_topk(raw[index, :, 5:].sigmoid(), 0.0, candidate_limit)
        value = InstanceData(
            bboxes=boxes[index, keep], scores=scores, labels=labels,
            score_factors=raw[index, keep, 4].sigmoid(),
            objectness_logits=raw[index, keep, 4], class_logits=raw[index, keep, 5:],
            anchor_ids=keep, level_ids=levels[keep])
        value = head._bbox_post_process(value, cfg, rescale=False, with_nms=False, img_meta=sample.metainfo)
        result.append(value)
    return result
