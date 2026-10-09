# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""Private bridge binding qualified LGP source paths to one attack caller."""
from dataclasses import dataclass

import torch

from .lgp_assignment import lgp_ranking
from .lgp_fidelity import LGPImageCoordinates
from .lgp_query import lgp_sparse_query_rows, lgp_deformable_query_rows
from .lgp_roi import LGPFixedROIProposals
from .lgp_semantic import lgp_no_object_terms
from .lgp_vfnet import lgp_vfnet_rows
from .lgp_yolo import lgp_yolo_rows


@dataclass
class LGPSourceRows:
    boxes: torch.Tensor
    classes: torch.Tensor
    kind: str
    objectness: object = None

    def ranking(self):
        if self.kind == 'softmax_background':
            scores, labels = self.classes.max(dim=1)
            return scores, labels != self.classes.shape[1] - 1
        return lgp_ranking(self.classes, self.kind, self.objectness)

    def semantic(self, indices):
        if indices.numel() == 0:
            raise RuntimeError('LGP semantic input has no real tracked row')
        if self.kind == 'softmax_background':
            values = self.classes[indices]
            return -values[:, -1].sum(), values.argmax(dim=1) == values.shape[1] - 1
        logits = self.objectness[indices] if self.kind == 'objectness' else self.classes[indices]
        return lgp_no_object_terms(logits, self.kind)


class LGPSource:
    def __init__(self, adapter, clean):
        self.adapter = adapter
        self.model = adapter.model
        self.name = type(self.model).__name__
        if self.name not in {'FasterRCNN', 'MaskRCNN', 'SparseRCNN', 'DeformableDETR', 'VFNet', 'YOLOV3'}:
            raise ValueError('Unsupported registered LGP source: ' + self.name)
        if self.model.training or any(parameter.requires_grad for parameter in self.model.parameters()):
            raise ValueError('LGP source must be eval with frozen model parameters')
        transform = getattr(adapter, '_input_transform', None)
        if transform is not None:
            from ..defenses.preprocessing import AdaptivePreprocessor
            if not isinstance(transform, AdaptivePreprocessor) or transform.spec.get('kind') not in {
                    'identity', 'jpeg', 'median', 'bit_depth', 'gaussian', 'resize_roundtrip'}:
                raise ValueError('Unbound LGP adaptive image/GT transform contract')
        self.coordinates = LGPImageCoordinates.from_preprocessor(self.model.data_preprocessor, clean)
        self.raw_shape = tuple(clean.shape[-2:])
        self.geometry = adapter.input_geometry(*self.raw_shape)
        self.valid_shape = tuple(self.geometry['resized_shape'])
        self.fixed_roi = self.name in {'FasterRCNN', 'MaskRCNN'}
        self.fixed = None
        self.forwards = 0

    def model_boxes(self, boxes):
        return self.adapter.model_space_boxes(boxes, self.raw_shape)

    def loss_image(self, raw):
        # Eq8 measures gamma before any adaptive defense; no padded pixels enter it.
        resized, geometry = self.adapter._resize_raw(raw)
        if tuple(geometry['resized_shape']) != self.valid_shape or tuple(resized.shape[-2:]) != self.valid_shape:
            raise RuntimeError('LGP valid image geometry changed')
        return self.coordinates.normalize(resized)

    def bind_fixed(self, model_boxes):
        if self.fixed_roi:
            sx, sy = self.geometry['scale_factor']
            raw_boxes = model_boxes / model_boxes.new_tensor((sx, sy, sx, sy))
            self.fixed = LGPFixedROIProposals(self.adapter, raw_boxes)

    def post_boxes(self, raw, threshold):
        self.forwards += 1
        boxes, _ = self.adapter.pseudo_targets(raw, threshold)
        if len(boxes) != 1:
            raise RuntimeError('LGP post-NMS batch mismatch')
        return self.model_boxes(boxes[0])

    def _roi_rows(self, candidate, original_coordinates):
        if not candidate.background_is_explicit or candidate.class_scores is None:
            raise RuntimeError('LGP ROI source lost real background probabilities')
        boxes = self.model_boxes(candidate.bboxes) if original_coordinates else candidate.bboxes
        return LGPSourceRows(boxes, candidate.class_scores, 'softmax_background')

    def rows(self, raw):
        self.forwards += 1
        if self.fixed is not None:
            return self._roi_rows(self.fixed(raw), True)
        processed = self.adapter.preprocess(raw)
        samples = processed['data_samples']
        if len(samples) != 1 or tuple(samples[0].metainfo['img_shape'][:2]) != self.valid_shape:
            raise RuntimeError('LGP source metadata changed valid image geometry')
        if self.name == 'DeformableDETR':
            value, = lgp_deformable_query_rows(self.model, processed['inputs'], samples)
            return LGPSourceRows(value.boxes, value.logits, 'sigmoid_classes')
        features = self.model.extract_feat(processed['inputs'])
        if not isinstance(features, (tuple, list)):
            features = (features,)
        if self.fixed_roi:
            proposals = self.model.rpn_head.predict(features, samples, rescale=False)
            value, = self.model.roi_head.predict_bbox(features, [samples[0].metainfo], proposals,
                                                     rcnn_test_cfg=None, rescale=False)
            return self._roi_rows(self.adapter._candidate_from_instances(value, 'lgp_clean_roi'), False)
        if self.name == 'SparseRCNN':
            value, = lgp_sparse_query_rows(self.model, features, samples)
            return LGPSourceRows(value.boxes, value.logits, 'sigmoid_classes')
        if self.name == 'VFNet':
            value, = lgp_vfnet_rows(self.model.bbox_head, features, samples)
            return LGPSourceRows(value.boxes, value.logits, 'sigmoid_classes')
        value, = lgp_yolo_rows(self.model.bbox_head, features, samples, self.adapter.candidate_limit)
        return LGPSourceRows(value.bboxes, value.class_logits, 'objectness', value.objectness_logits)
