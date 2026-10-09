# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""LGP-private query decoding; boxes stay in valid model-image coordinates."""
from typing import NamedTuple

import torch
from mmdet.structures.bbox import bbox2roi, bbox_cxcywh_to_xyxy


class LGPQueryRows(NamedTuple):
    boxes: torch.Tensor
    logits: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    query_ids: torch.Tensor


def _select(logits, boxes, head, count, sorted_rows):
    if not head.loss_cls.use_sigmoid:
        raise ValueError('Registered Sparse/Deformable LGP sources require sigmoid heads')
    if logits.ndim != 2 or boxes.shape != (logits.shape[0], 4):
        raise ValueError('Expected aligned query logits and xyxy boxes')
    if logits.shape[1] != head.num_classes or not logits.numel():
        raise ValueError('Expected every real foreground class and nonempty queries')
    if boxes.device != logits.device or boxes.dtype != logits.dtype:
        raise ValueError('Boxes and logits must share dtype and device')
    if not torch.isfinite(logits).all() or not torch.isfinite(boxes).all():
        raise ValueError('Nonfinite query output')
    if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= logits.numel():
        raise ValueError('Invalid registered max_per_img')
    scores, flat = logits.sigmoid().flatten().topk(count, sorted=sorted_rows)
    query_ids = torch.div(flat, head.num_classes, rounding_mode='floor')
    labels = flat.remainder(head.num_classes)
    # The same query can appear once per selected class; never deduplicate it.
    return LGPQueryRows(boxes[query_ids], logits[query_ids], scores, labels, query_ids)


def lgp_sparse_query_rows(model, features, samples):
    """Run every refinement stage; select aligned rows from the final stage."""
    roi = model.roi_head
    if roi.num_stages <= 0:
        raise ValueError('Sparse source has no refinement stages')
    initial = model.rpn_head.predict(features, samples, rescale=False)
    if not samples or len(initial) != len(samples):
        raise ValueError('Sparse proposal batch mismatch')
    proposals = [item.bboxes for item in initial]
    objects = torch.stack([item.features for item in initial])
    metas = [sample.metainfo for sample in samples]
    for stage in range(roi.num_stages):
        last = roi._bbox_forward(stage, features, bbox2roi(proposals), objects, metas)
        objects = last['object_feats']
        proposals = last['detached_proposals']
    logits = last['cls_score']
    if logits.ndim != 3 or logits.shape[0] != len(samples):
        raise ValueError('Sparse final classification batch mismatch')
    boxes = last['decoded_bboxes']
    if boxes.numel() != logits.shape[0] * logits.shape[1] * 4:
        raise ValueError('Sparse final decoded box count mismatch')
    boxes = boxes.reshape(logits.shape[0], logits.shape[1], 4)
    count = roi.test_cfg.get('max_per_img', logits.shape[1])
    return [_select(logits[index], boxes[index], roi.bbox_head[-1], count, False)
            for index in range(len(samples))]


def lgp_deformable_query_rows(model, processed_inputs, samples):
    """Use final decoder logits and normalized boxes, without NMS or rescaling."""
    output = model._forward(processed_inputs, samples)
    if len(output) < 2:
        raise ValueError('Deformable source did not expose decoder outputs')
    logits, normalized_boxes = output[0][-1], output[1][-1]
    if logits.ndim != 3 or logits.shape[0] != len(samples) or not samples:
        raise ValueError('Deformable final classification batch mismatch')
    if normalized_boxes.shape != (*logits.shape[:2], 4):
        raise ValueError('Deformable final box batch mismatch')
    result = []
    for index, sample in enumerate(samples):
        height, width = sample.metainfo['img_shape'][:2]
        if height <= 0 or width <= 0:
            raise ValueError('Invalid model image dimensions')
        xyxy = bbox_cxcywh_to_xyxy(normalized_boxes[index])
        scale = xyxy.new_tensor((width, height, width, height))
        scaled = xyxy * scale
        boxes = torch.stack((scaled[:, 0].clamp(0, width),
                             scaled[:, 1].clamp(0, height),
                             scaled[:, 2].clamp(0, width),
                             scaled[:, 3].clamp(0, height)), dim=1)
        count = model.bbox_head.test_cfg.get('max_per_img', logits.shape[1])
        result.append(_select(logits[index], boxes, model.bbox_head, count, True))
    return result
