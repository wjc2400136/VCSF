# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""LGP-only fixed ROI proposal path for standard softmax bbox heads."""
import torch
from mmengine.structures import InstanceData


class LGPFixedROIProposals:
    def __init__(self, adapter, original_boxes):
        if original_boxes.ndim != 2 or original_boxes.shape[1] != 4 or original_boxes.shape[0] == 0:
            raise ValueError("Fixed LGP proposals must be a nonempty Nx4 tensor")
        if not bool(torch.isfinite(original_boxes).all()):
            raise ValueError("Fixed LGP proposals must be finite")
        roi = adapter.model.roi_head
        if getattr(roi, 'num_stages', None) is not None:
            raise ValueError("The standard LGP ROI path cannot serve cascade or sparse heads")
        head = roi.bbox_head
        if bool(head.loss_cls.use_sigmoid) or bool(head.custom_cls_channels):
            raise ValueError("Fixed LGP ROI proposals require an explicit softmax background")
        self.adapter = adapter
        self.original_boxes = original_boxes.detach().clone()
        self.num_classes = int(head.num_classes)

    def __call__(self, raw_bgr):
        if raw_bgr.ndim != 4 or raw_bgr.shape[0] != 1 or raw_bgr.shape[1] != 3:
            raise ValueError("Fixed LGP ROI proposals require one raw BGR image")
        adapter = self.adapter
        processed = adapter.preprocess(raw_bgr)
        samples = processed['data_samples']
        if len(samples) != 1:
            raise ValueError("Fixed LGP ROI metadata must describe one image")
        features = adapter.model.extract_feat(processed['inputs'])
        model_boxes = adapter.model_space_boxes(
            self.original_boxes.to(raw_bgr), tuple(raw_bgr.shape[-2:])
        )
        predictions = adapter.model.roi_head.predict_bbox(
            features, [samples[0].metainfo], [InstanceData(bboxes=model_boxes)],
            rcnn_test_cfg=None, rescale=False,
        )
        if len(predictions) != 1:
            raise ValueError("Fixed LGP ROI predictions must describe one image")
        prediction = predictions[0]
        count = self.original_boxes.shape[0]
        if prediction.scores.shape != (count, self.num_classes + 1):
            raise ValueError("Fixed LGP ROI predictions lost rows or the full background column")
        if prediction.bboxes.ndim != 2 or prediction.bboxes.shape[0] != count or prediction.bboxes.shape[1] not in (4, 4 * self.num_classes):
            raise ValueError("Fixed LGP ROI boxes and scores are not aligned")
        candidate = adapter._candidate_from_instances(prediction, 'lgp_fixed_roi_pre_nms')
        if not candidate.background_is_explicit:
            raise ValueError("Fixed LGP ROI output did not retain real background scores")
        # No candidate top-k or new RPN pass: each original slot keeps its row.
        return adapter._restore_candidate(candidate, samples[0])
