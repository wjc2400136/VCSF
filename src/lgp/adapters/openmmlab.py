from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox2roi, bbox_cxcywh_to_xyxy


@dataclass
class CandidateSet:
    """Differentiable detector candidates before final NMS.

    ``scores`` always denotes foreground confidence for the candidate selected
    by ``labels``.  ``class_scores`` is retained when the detector exposes a
    complete class distribution.  Boxes and scores remain connected to the
    input graph; labels and top-k indices are intentionally discrete.
    """

    bboxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    class_scores: Optional[torch.Tensor] = None
    background_scores: Optional[torch.Tensor] = None
    background_is_explicit: bool = False
    objectness: Optional[torch.Tensor] = None
    surface: str = "unknown"

    def topk(self, limit: int) -> "CandidateSet":
        if limit <= 0 or self.scores.numel() <= limit:
            return self
        indices = self.scores.topk(limit, sorted=False).indices
        return CandidateSet(
            bboxes=self.bboxes[indices],
            scores=self.scores[indices],
            labels=self.labels[indices],
            class_scores=(
                self.class_scores[indices] if self.class_scores is not None else None
            ),
            background_scores=(
                self.background_scores[indices]
                if self.background_scores is not None
                else None
            ),
            background_is_explicit=self.background_is_explicit,
            objectness=self.objectness[indices] if self.objectness is not None else None,
            surface=self.surface,
        )


class OpenMMLabAdapter:
    """Differentiable MMDetection/MMYOLO 3.x surface used by all attacks.

    MMDetection's public dense-head prediction helper detaches feature maps in
    some heads.  This adapter follows the same decode/filter implementation but
    explicitly requests non-detached tensors.  It also keeps the native raw
    ROI, Sparse R-CNN and DETR query paths instead of inventing an NMS-based
    surrogate objective.
    """

    def __init__(
        self,
        model,
        candidate_limit: int = 1000,
        input_transform=None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.candidate_limit = int(candidate_limit)
        self._resize_scale, self._keep_ratio = self._find_test_resize(model)
        data_preprocessor = getattr(model, "data_preprocessor", None)
        self._pad_size_divisor = max(
            int(getattr(data_preprocessor, "pad_size_divisor", 1) or 1), 1
        )
        self._input_transform = input_transform
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _find_test_resize(model) -> Tuple[Optional[Tuple[int, int]], bool]:
        """Read the deterministic test Resize from the pinned model config."""
        cfg = getattr(model, "cfg", None)
        pipeline = cfg.get("test_pipeline") if hasattr(cfg, "get") else None

        def walk(value: Any) -> Optional[Mapping[str, Any]]:
            if isinstance(value, Mapping):
                type_name = str(value.get("type", ""))
                if type_name in {
                    "Resize",
                    "FixScaleResize",
                    "LetterResize",
                    "YOLOv5KeepRatioResize",
                }:
                    return value
                for child in value.values():
                    found = walk(child)
                    if found is not None:
                        return found
            elif isinstance(value, (list, tuple)):
                for child in value:
                    found = walk(child)
                    if found is not None:
                        return found
            return None

        resize = walk(pipeline)
        if resize is None or resize.get("scale") is None:
            return None, True
        scale = resize.get("scale")
        if isinstance(scale, (list, tuple)) and len(scale) == 2:
            if all(isinstance(item, (int, float)) for item in scale):
                return (int(scale[0]), int(scale[1])), bool(
                    resize.get("keep_ratio", True)
                )
        return None, True

    def input_geometry(self, height: int, width: int) -> Dict[str, Any]:
        """Return the source test pipeline's resize/pad geometry."""
        if height <= 0 or width <= 0:
            raise ValueError("Image dimensions must be positive")
        if self._resize_scale is None:
            resized_width, resized_height = int(width), int(height)
        elif self._keep_ratio:
            target_width, target_height = self._resize_scale
            ratio = min(
                float(target_width) / float(width),
                float(target_height) / float(height),
            )
            resized_width = max(1, int(round(float(width) * ratio)))
            resized_height = max(1, int(round(float(height) * ratio)))
        else:
            resized_width, resized_height = self._resize_scale
        scale_x = float(resized_width) / float(width)
        scale_y = float(resized_height) / float(height)
        padded_width = int(
            ((resized_width + self._pad_size_divisor - 1) // self._pad_size_divisor)
            * self._pad_size_divisor
        )
        padded_height = int(
            ((resized_height + self._pad_size_divisor - 1) // self._pad_size_divisor)
            * self._pad_size_divisor
        )
        return {
            "original_shape": (int(height), int(width)),
            "resized_shape": (resized_height, resized_width),
            "padded_shape": (padded_height, padded_width),
            "scale_factor": (scale_x, scale_y),
            "resize_scale": self._resize_scale,
            "keep_ratio": self._keep_ratio,
            "pad_size_divisor": self._pad_size_divisor,
        }

    def _resize_raw(self, raw_bgr: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        geometry = self.input_geometry(raw_bgr.shape[-2], raw_bgr.shape[-1])
        target_shape = geometry["resized_shape"]
        if tuple(raw_bgr.shape[-2:]) == tuple(target_shape):
            return raw_bgr, geometry
        return (
            F.interpolate(
                raw_bgr,
                size=target_shape,
                mode="bilinear",
                align_corners=False,
            ),
            geometry,
        )

    def model_space_boxes(
        self, boxes: torch.Tensor, raw_shape: Tuple[int, int]
    ) -> torch.Tensor:
        """Map original-image xyxy boxes into the differentiable source input."""
        scale_x, scale_y = self.input_geometry(*raw_shape)["scale_factor"]
        factor = boxes.new_tensor((scale_x, scale_y, scale_x, scale_y))
        return boxes * factor

    def model_space_mask(self, raw_mask: torch.Tensor) -> torch.Tensor:
        """Apply source-test Resize and bottom/right padding to a spatial mask."""
        geometry = self.input_geometry(raw_mask.shape[-2], raw_mask.shape[-1])
        resized = F.interpolate(
            raw_mask,
            size=geometry["resized_shape"],
            mode="nearest",
        )
        padded_height, padded_width = geometry["padded_shape"]
        return F.pad(
            resized,
            (0, padded_width - resized.shape[-1], 0, padded_height - resized.shape[-2]),
            mode="constant",
            value=0.0,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @staticmethod
    def _samples(
        batch: int,
        ori_shape: Tuple[int, int],
        img_shape: Tuple[int, int],
        scale_factor: Tuple[float, float],
        boxes: Optional[Sequence[torch.Tensor]] = None,
        labels: Optional[Sequence[torch.Tensor]] = None,
    ) -> List[DetDataSample]:
        samples: List[DetDataSample] = []
        ori_height, ori_width = ori_shape
        image_height, image_width = img_shape
        for index in range(batch):
            sample = DetDataSample()
            sample.set_metainfo(
                {
                    "ori_shape": (ori_height, ori_width),
                    "img_shape": (image_height, image_width),
                    "pad_shape": (image_height, image_width),
                    "scale_factor": scale_factor,
                }
            )
            if boxes is not None and labels is not None:
                sample.gt_instances = InstanceData(
                    bboxes=boxes[index].to(dtype=torch.float32),
                    labels=labels[index].to(dtype=torch.long),
                )
            samples.append(sample)
        return samples

    def preprocess(
        self,
        raw_bgr: torch.Tensor,
        boxes: Optional[Sequence[torch.Tensor]] = None,
        labels: Optional[Sequence[torch.Tensor]] = None,
        training: bool = False,
    ) -> Dict[str, Any]:
        if raw_bgr.ndim != 4 or raw_bgr.shape[1] != 3:
            raise ValueError("Expected Bx3xHxW raw BGR tensor")
        if self._input_transform is not None:
            transformed = self._input_transform(raw_bgr)
            if not isinstance(transformed, torch.Tensor):
                raise TypeError("The adapter input transform must return a tensor")
            if transformed.shape != raw_bgr.shape:
                raise ValueError(
                    "The adapter input transform must preserve Bx3xHxW geometry"
                )
            raw_bgr = transformed
        batch, _, height, width = raw_bgr.shape
        if (boxes is None) != (labels is None):
            raise ValueError("boxes and labels must be supplied together")
        if boxes is not None and (len(boxes) != batch or len(labels or ()) != batch):
            raise ValueError("Ground-truth batch length differs from image batch")
        resized, geometry = self._resize_raw(raw_bgr)
        scale_x, scale_y = geometry["scale_factor"]
        factor = raw_bgr.new_tensor((scale_x, scale_y, scale_x, scale_y))
        scaled_boxes = (
            [
                value.to(device=raw_bgr.device, dtype=torch.float32) * factor
                for value in boxes
            ]
            if boxes is not None
            else None
        )
        resized_height, resized_width = geometry["resized_shape"]
        samples = self._samples(
            batch,
            (height, width),
            (resized_height, resized_width),
            (scale_x, scale_y),
            boxes=scaled_boxes,
            labels=(
                [value.to(device=raw_bgr.device, dtype=torch.long) for value in labels]
                if labels is not None
                else None
            ),
        )
        if training and boxes is not None:
            roi_head = getattr(self.model, "roi_head", None)
            if bool(getattr(roi_head, "with_mask", False)):
                # MMDetection's Mask R-CNN loss API requires masks even when an
                # attack consumes only the RPN/ROI box branches.  Rectangle
                # masks satisfy the untouched detector API; mask losses are
                # explicitly excluded from TOG/AFOG objectives below.
                from mmdet.structures.mask import BitmapMasks

                for index, sample in enumerate(samples):
                    masks = np.zeros(
                        (int(boxes[index].shape[0]), resized_height, resized_width),
                        dtype=np.uint8,
                    )
                    for mask_index, box in enumerate(scaled_boxes[index].detach().cpu()):
                        left = max(0, min(resized_width, int(torch.floor(box[0]).item())))
                        top = max(0, min(resized_height, int(torch.floor(box[1]).item())))
                        right = max(0, min(resized_width, int(torch.ceil(box[2]).item())))
                        bottom = max(0, min(resized_height, int(torch.ceil(box[3]).item())))
                        if right > left and bottom > top:
                            masks[mask_index, top:bottom, left:right] = 1
                    sample.gt_instances.masks = BitmapMasks(
                        masks, resized_height, resized_width
                    )
        data = {
            "inputs": [resized[index] for index in range(batch)],
            "data_samples": samples,
        }
        return self.model.data_preprocessor(data, training=training)

    def extract_features(self, raw_bgr: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        processed = self.preprocess(raw_bgr)
        features = self.model.extract_feat(processed["inputs"])
        if isinstance(features, torch.Tensor):
            features = (features,)
        if not isinstance(features, (list, tuple)):
            raise RuntimeError("Detector extract_feat did not return tensor feature maps")
        selected = tuple(
            feature
            for feature in features
            if isinstance(feature, torch.Tensor) and feature.ndim == 4
        )
        if not selected:
            raise RuntimeError("Detector exposed no 4-D feature map")
        return selected

    def extract_backbone_features(
        self, raw_bgr: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        processed = self.preprocess(raw_bgr)
        features = self.model.backbone(processed["inputs"])
        values = features if isinstance(features, (list, tuple)) else (features,)
        selected = tuple(
            value
            for value in values
            if isinstance(value, torch.Tensor) and value.ndim == 4
        )
        if not selected:
            raise RuntimeError("Detector backbone exposed no 4-D feature map")
        return selected

    @staticmethod
    def _single_spatial_tensor(output: Any, description: str) -> torch.Tensor:
        values = output if isinstance(output, (list, tuple)) else (output,)
        tensors = tuple(
            value
            for value in values
            if isinstance(value, torch.Tensor) and value.ndim == 4
        )
        if len(tensors) != 1:
            raise RuntimeError(
                "{} must expose exactly one 4-D feature tensor; found {}".format(
                    description, len(tensors)
                )
            )
        return tensors[0]

    def _backbone_module(self, module_path: str):
        path = str(module_path).strip()
        if not path:
            raise ValueError("Backbone module path must not be empty")
        try:
            return self.model.backbone.get_submodule(path)
        except (AttributeError, KeyError) as exc:
            raise ValueError(
                "Backbone does not expose required module '{}': {}".format(
                    path, self.model.backbone.__class__.__name__
                )
            ) from exc

    def extract_backbone_module_feature(
        self, raw_bgr: torch.Tensor, module_path: str
    ) -> torch.Tensor:
        """Capture one named backbone activation from a backbone-only pass."""
        module = self._backbone_module(module_path)
        captured: List[torch.Tensor] = []

        def capture(_module, _inputs, output) -> None:
            captured.append(
                self._single_spatial_tensor(
                    output, "Backbone module '{}'".format(module_path)
                )
            )

        handle = module.register_forward_hook(capture)
        try:
            processed = self.preprocess(raw_bgr)
            self.model.backbone(processed["inputs"])
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                "Backbone module '{}' was not executed".format(module_path)
            )
        return captured[-1]

    def backbone_module_feature_and_candidates(
        self, raw_bgr: torch.Tensor, module_path: str
    ) -> Tuple[torch.Tensor, List[CandidateSet]]:
        """Capture one named backbone activation from the candidate forward."""
        module = self._backbone_module(module_path)
        captured: List[torch.Tensor] = []

        def capture(_module, _inputs, output) -> None:
            captured.append(
                self._single_spatial_tensor(
                    output, "Backbone module '{}'".format(module_path)
                )
            )

        handle = module.register_forward_hook(capture)
        try:
            candidates = self.candidate_predictions(raw_bgr)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError(
                "Backbone module '{}' was not executed by candidate prediction".format(
                    module_path
                )
            )
        return captured[-1], candidates

    def deformable_encoder_attention_surface_and_candidates(
        self, raw_bgr: torch.Tensor
    ) -> Tuple[Dict[str, Any], List[CandidateSet]]:
        """Expose final-encoder logits, geometry and final clean detections.

        Corrupting Attention defines its objective on the linear projection
        immediately before the deformable-attention softmax.  Capturing the
        output of ``attention_weights`` preserves that exact gradient surface;
        using the post-softmax weights would be a different, weaker objective.
        """
        encoder = getattr(self.model, "encoder", None)
        layers = getattr(encoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise ValueError("Detector has no transformer encoder layers")
        attention = getattr(layers[-1], "self_attn", None)
        projection = getattr(attention, "attention_weights", None)
        required = ("num_heads", "num_levels", "num_points")
        if projection is None or any(not hasattr(attention, item) for item in required):
            raise ValueError(
                "Detector does not expose final-layer deformable-attention logits"
            )

        captured: Dict[str, Any] = {}

        def capture_context(_module, _args, kwargs) -> None:
            for key in ("reference_points", "spatial_shapes", "level_start_index"):
                value = kwargs.get(key)
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(
                        "Deformable attention did not expose '{}'".format(key)
                    )
                captured[key] = value

        def capture_logits(_module, _inputs, output) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError(
                    "Deformable attention projection exposed an invalid logit tensor"
                )
            captured["logits"] = output

        context_handle = attention.register_forward_pre_hook(
            capture_context, with_kwargs=True
        )
        logit_handle = projection.register_forward_hook(capture_logits)
        try:
            predictions = self.predict(raw_bgr)
        finally:
            logit_handle.remove()
            context_handle.remove()
        candidates: List[CandidateSet] = []
        for sample in predictions:
            instances = sample.pred_instances
            candidates.append(
                CandidateSet(
                    bboxes=instances.bboxes,
                    scores=instances.scores,
                    labels=instances.labels,
                    surface="postprocessed_clean_detections",
                )
            )
        required_outputs = {
            "logits",
            "reference_points",
            "spatial_shapes",
            "level_start_index",
        }
        missing = sorted(required_outputs - set(captured))
        if missing:
            raise RuntimeError(
                "Final deformable-attention capture is incomplete: {}".format(
                    ", ".join(missing)
                )
            )
        captured.update(
            {
                "num_heads": int(attention.num_heads),
                "num_levels": int(attention.num_levels),
                "num_points": int(attention.num_points),
                "encoder_layer": int(len(layers) - 1),
            }
        )
        return captured, candidates

    def predict(self, raw_bgr: torch.Tensor):
        processed = self.preprocess(raw_bgr)
        return self.model.predict(
            processed["inputs"], processed["data_samples"], rescale=True
        )

    @staticmethod
    def _restore_candidate(
        candidate: CandidateSet, sample: DetDataSample
    ) -> CandidateSet:
        scale = sample.metainfo.get("scale_factor", (1.0, 1.0))
        factor = candidate.bboxes.new_tensor(
            (
                float(scale[0]),
                float(scale[1]),
                float(scale[0]),
                float(scale[1]),
            )
        )
        return CandidateSet(
            bboxes=candidate.bboxes / factor,
            scores=candidate.scores,
            labels=candidate.labels,
            class_scores=candidate.class_scores,
            background_scores=candidate.background_scores,
            background_is_explicit=candidate.background_is_explicit,
            objectness=candidate.objectness,
            surface=candidate.surface,
        )

    @torch.no_grad()
    def pseudo_targets(
        self,
        raw_bgr: torch.Tensor,
        score_threshold: float,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Freeze post-processed clean detections as attack pseudo targets.

        TOG-untargeted and baseline AFOG define their target set from the
        victim's clean detections.  Keeping this operation here avoids
        reimplementing detector-specific NMS or changing any detector head.
        Returned boxes and labels are detached and remain in input-image
        coordinates because ``predict(..., rescale=True)`` is used.
        """
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must lie in [0, 1]")
        outputs = self.predict(raw_bgr)
        boxes: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for sample in outputs:
            instances = getattr(sample, "pred_instances", None)
            sample_boxes = getattr(instances, "bboxes", None)
            sample_labels = getattr(instances, "labels", None)
            sample_scores = getattr(instances, "scores", None)
            if not all(
                isinstance(value, torch.Tensor)
                for value in (sample_boxes, sample_labels, sample_scores)
            ):
                boxes.append(raw_bgr.new_zeros((0, 4)))
                labels.append(
                    torch.zeros((0,), dtype=torch.long, device=raw_bgr.device)
                )
                continue
            keep = sample_scores >= float(score_threshold)
            boxes.append(sample_boxes[keep].detach().to(dtype=torch.float32).clone())
            labels.append(sample_labels[keep].detach().to(dtype=torch.long).clone())
        return boxes, labels

    def prediction_scores(self, raw_bgr: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.predict(raw_bgr)
        result = []
        for sample in outputs:
            instances = getattr(sample, "pred_instances", None)
            scores = getattr(instances, "scores", None) if instances is not None else None
            if isinstance(scores, torch.Tensor):
                result.append(scores)
            else:
                result.append(raw_bgr.new_zeros((0,)))
        return result

    @staticmethod
    def _candidate_from_instances(instances: InstanceData, surface: str) -> CandidateSet:
        bboxes = instances.bboxes
        scores = instances.scores
        labels = getattr(instances, "labels", None)
        class_scores: Optional[torch.Tensor] = None
        background_scores: Optional[torch.Tensor] = None
        background_is_explicit = False
        if scores.ndim == 2:
            class_scores = scores
            foreground = scores
            if scores.shape[1] > 1:
                # ROI heads expose the background class in the last column.
                foreground = scores[:, :-1]
                background_scores = scores[:, -1]
                background_is_explicit = True
            scores, labels = foreground.max(dim=1)
            if bboxes.ndim == 2 and bboxes.shape[1] > 4:
                reshaped = bboxes.reshape(bboxes.shape[0], -1, 4)
                chosen = labels.clamp(max=reshaped.shape[1] - 1)
                bboxes = reshaped[torch.arange(len(chosen), device=chosen.device), chosen]
        if labels is None:
            labels = torch.zeros_like(scores, dtype=torch.long)
        if background_scores is None:
            background_scores = (1.0 - scores).clamp(0.0, 1.0)
        objectness = getattr(instances, "objectness", None)
        return CandidateSet(
            bboxes=bboxes,
            scores=scores,
            labels=labels,
            class_scores=class_scores,
            background_scores=background_scores,
            background_is_explicit=background_is_explicit,
            objectness=objectness,
            surface=surface,
        )

    def _dense_candidates(
        self,
        features: Sequence[torch.Tensor],
        samples: Sequence[DetDataSample],
    ) -> List[CandidateSet]:
        from mmdet.models.utils import select_single_mlvl

        head = self.model.bbox_head
        outputs = head(features)
        metas = [sample.metainfo for sample in samples]
        cfg = copy.deepcopy(head.test_cfg)
        cfg["score_thr"] = 0.0
        cfg["conf_thr"] = -1.0
        cfg["nms_pre"] = self.candidate_limit

        # YOLOv3 has a dedicated decoder and, unlike BaseDenseHead 3.0.0,
        # does not detach its prediction maps.
        if head.__class__.__name__ == "YOLOV3Head":
            instances = head.predict_by_feat(
                *outputs,
                batch_img_metas=metas,
                cfg=cfg,
                rescale=False,
                with_nms=False,
            )
            result = []
            for index, value in enumerate(instances):
                candidate = self._candidate_from_instances(value, "dense_pre_nms")
                if hasattr(value, "score_factors"):
                    candidate.objectness = value.score_factors
                result.append(
                    self._restore_candidate(
                        candidate.topk(self.candidate_limit), samples[index]
                    )
                )
            return result

        if len(outputs) < 2:
            raise RuntimeError("Dense head did not expose classification and box outputs")
        cls_scores, bbox_preds = outputs[:2]
        score_factors = outputs[2] if len(outputs) > 2 else None
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[index].shape[-2:] for index in range(num_levels)]
        priors = head.prior_generator.grid_priors(
            featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device
        )
        result: List[CandidateSet] = []
        for image_index, meta in enumerate(metas):
            cls_list = select_single_mlvl(cls_scores, image_index, detach=False)
            bbox_list = select_single_mlvl(bbox_preds, image_index, detach=False)
            if score_factors is None:
                factor_list = [None for _ in range(num_levels)]
            else:
                factor_list = select_single_mlvl(
                    score_factors, image_index, detach=False
                )
            instances = head._predict_by_feat_single(
                cls_score_list=cls_list,
                bbox_pred_list=bbox_list,
                score_factor_list=factor_list,
                mlvl_priors=priors,
                img_meta=meta,
                cfg=cfg,
                rescale=False,
                with_nms=False,
            )
            candidate = self._candidate_from_instances(instances, "dense_pre_nms").topk(
                self.candidate_limit
            )
            result.append(self._restore_candidate(candidate, samples[image_index]))
        return result

    def _two_stage_candidates(
        self,
        features: Sequence[torch.Tensor],
        samples: Sequence[DetDataSample],
    ) -> List[CandidateSet]:
        model = self.model
        proposals = model.rpn_head.predict(features, samples, rescale=False)
        metas = [sample.metainfo for sample in samples]
        instances = model.roi_head.predict_bbox(
            features,
            metas,
            proposals,
            rcnn_test_cfg=None,
            rescale=False,
        )
        return [
            self._restore_candidate(
                self._candidate_from_instances(value, "roi_pre_nms").topk(
                    self.candidate_limit
                ),
                samples[index],
            )
            for index, value in enumerate(instances)
        ]

    def _sparse_candidates(
        self,
        features: Sequence[torch.Tensor],
        samples: Sequence[DetDataSample],
    ) -> List[CandidateSet]:
        model = self.model
        roi_head = model.roi_head
        metas = [sample.metainfo for sample in samples]
        initial = model.rpn_head.predict(features, samples, rescale=False)
        proposal_list = [value.bboxes for value in initial]
        object_features = torch.cat([value.features[None, ...] for value in initial])
        last: Optional[Dict[str, Any]] = None
        for stage in range(roi_head.num_stages):
            rois = bbox2roi(proposal_list)
            last = roi_head._bbox_forward(
                stage, features, rois, object_features, metas
            )
            object_features = last["object_feats"]
            proposal_list = last["detached_proposals"]
        if last is None:
            raise RuntimeError("Sparse R-CNN exposed no refinement stage")
        class_logits = last["cls_score"]
        boxes = last["decoded_bboxes"].reshape(class_logits.shape[0], -1, 4)
        head = roi_head.bbox_head[-1]
        if head.loss_cls.use_sigmoid:
            class_scores = class_logits.sigmoid()
            foreground = class_scores
            background = 1.0 - foreground.max(dim=-1).values
        else:
            class_scores = class_logits.softmax(dim=-1)
            foreground = class_scores[..., :-1]
            background = class_scores[..., -1]
        scores, labels = foreground.max(dim=-1)
        result = []
        for index in range(class_logits.shape[0]):
            result.append(
                self._restore_candidate(CandidateSet(
                    bboxes=boxes[index],
                    scores=scores[index],
                    labels=labels[index],
                    class_scores=class_scores[index],
                    background_scores=background[index],
                    background_is_explicit=not head.loss_cls.use_sigmoid,
                    surface="sparse_last_refinement",
                ).topk(self.candidate_limit), samples[index])
            )
        return result

    def _transformer_candidates(
        self,
        processed_inputs: torch.Tensor,
        samples: Sequence[DetDataSample],
    ) -> List[CandidateSet]:
        class_logits, box_predictions = self.model._forward(processed_inputs, samples)[:2]
        class_logits = class_logits[-1]
        box_predictions = box_predictions[-1]
        head = self.model.bbox_head
        if head.loss_cls.use_sigmoid:
            class_scores = class_logits.sigmoid()
            foreground = class_scores
            background = 1.0 - foreground.max(dim=-1).values
        else:
            class_scores = class_logits.softmax(dim=-1)
            foreground = class_scores[..., :-1]
            background = class_scores[..., -1]
        scores, labels = foreground.max(dim=-1)
        result: List[CandidateSet] = []
        for index, sample in enumerate(samples):
            boxes = bbox_cxcywh_to_xyxy(box_predictions[index])
            height, width = sample.metainfo["img_shape"][:2]
            scale = boxes.new_tensor((width, height, width, height))
            boxes = boxes * scale
            result.append(
                self._restore_candidate(CandidateSet(
                    bboxes=boxes,
                    scores=scores[index],
                    labels=labels[index],
                    class_scores=class_scores[index],
                    background_scores=background[index],
                    background_is_explicit=not head.loss_cls.use_sigmoid,
                    surface="transformer_queries",
                ).topk(self.candidate_limit), sample)
            )
        return result

    def candidate_predictions(self, raw_bgr: torch.Tensor) -> List[CandidateSet]:
        """Return differentiable pre-NMS/proposal-like candidates."""
        processed = self.preprocess(raw_bgr)
        inputs = processed["inputs"]
        samples = processed["data_samples"]
        model_name = self.model.__class__.__name__
        if "DETR" in model_name or hasattr(self.model, "forward_transformer"):
            return self._transformer_candidates(inputs, samples)
        features = self.model.extract_feat(inputs)
        if not isinstance(features, (list, tuple)):
            features = (features,)
        if model_name == "SparseRCNN":
            return self._sparse_candidates(features, samples)
        if hasattr(self.model, "roi_head") and hasattr(self.model, "rpn_head"):
            return self._two_stage_candidates(features, samples)
        if hasattr(self.model, "bbox_head"):
            return self._dense_candidates(features, samples)
        raise RuntimeError("No differentiable candidate adapter for {}".format(model_name))

    def backbone_features_and_candidates(
        self, raw_bgr: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], List[CandidateSet]]:
        """Capture backbone activations from the exact candidate forward pass."""
        captured: List[Tuple[torch.Tensor, ...]] = []

        def capture(_module, _inputs, output) -> None:
            values = output if isinstance(output, (list, tuple)) else (output,)
            tensors = tuple(
                value
                for value in values
                if isinstance(value, torch.Tensor) and value.ndim == 4
            )
            if tensors:
                captured.append(tensors)

        handle = self.model.backbone.register_forward_hook(capture)
        try:
            candidates = self.candidate_predictions(raw_bgr)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Backbone hook captured no spatial feature tensor")
        return captured[-1], candidates

    def pre_nms_prediction_scores(self, raw_bgr: torch.Tensor) -> List[torch.Tensor]:
        return [candidate.scores for candidate in self.candidate_predictions(raw_bgr)]

    def detection_losses(
        self,
        raw_bgr: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Native detector training losses for structure-specific objectives.

        This path is intentionally restricted to attacks, such as AugTrans,
        whose published objective explicitly requires detector training
        branches.  It is not used as a generic surrogate for output attacks.
        """
        processed = self.preprocess(
            raw_bgr, boxes=boxes, labels=labels, training=True
        )
        # MMDetection's loss APIs are training-mode APIs.  The detector is
        # kept in eval mode for inference/candidate attacks, but several
        # heads change their forward return signature in eval mode (VFNet
        # omits the initial regression branch; two-stage Deformable DETR omits
        # encoder outputs).  Toggle only the flags needed by the loss call and
        # restore every module's original state afterwards, so BatchNorm/dropout
        # behaviour of the inference path is never leaked into later steps.
        training_states = {
            module: bool(module.training) for module in self.model.modules()
        }
        self.model.training = True
        for name in ("bbox_head", "rpn_head", "roi_head"):
            module = getattr(self.model, name, None)
            if module is not None:
                module.training = True
        try:
            losses = self.model.loss(processed["inputs"], processed["data_samples"])
        finally:
            for module, state in training_states.items():
                module.training = state
        result: Dict[str, torch.Tensor] = {}
        for name, value in losses.items():
            values = value if isinstance(value, (list, tuple)) else [value]
            tensors = [item for item in values if isinstance(item, torch.Tensor)]
            if tensors:
                result[name] = torch.stack([item.mean() for item in tensors]).sum()
        return result

    def backbone_features_and_losses(
        self,
        raw_bgr: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Capture backbone features and native training losses in one pass."""
        captured: List[Tuple[torch.Tensor, ...]] = []

        def capture(_module, _inputs, output) -> None:
            values = output if isinstance(output, (list, tuple)) else (output,)
            tensors = tuple(
                value
                for value in values
                if isinstance(value, torch.Tensor) and value.ndim == 4
            )
            if tensors:
                captured.append(tensors)

        handle = self.model.backbone.register_forward_hook(capture)
        try:
            losses = self.detection_losses(raw_bgr, boxes, labels)
        finally:
            handle.remove()
        if not captured:
            raise RuntimeError("Backbone hook captured no spatial feature tensor")
        return captured[-1], losses

    @staticmethod
    def detection_loss_total(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum native box-detector losses, excluding metrics and mask heads."""
        values = [
            value
            for name, value in losses.items()
            if "loss" in name.lower() and isinstance(value, torch.Tensor)
            and "mask" not in name.lower()
            and "seg" not in name.lower()
        ]
        if not values:
            raise RuntimeError("Detector returned no differentiable training loss")
        return torch.stack([value.mean() for value in values]).sum()

    def afog_loss_total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Published AFOG bbox-plus-class objective for the victim family.

        The released DETR/Deformable-DETR wrappers use only the final decoder's
        classification and L1-box terms and apply their stated balancing
        expression.  Regression detectors use their native box, objectness and
        class losses, as the released Faster R-CNN and YOLOv3 wrappers do.  Mask
        losses are never part of AFOG Eq. (3).
        """
        model_name = self.model.__class__.__name__
        if "DETR" in model_name:
            classification = losses.get("loss_cls")
            boxes = losses.get("loss_bbox")
            if classification is None or boxes is None:
                raise RuntimeError(
                    "AFOG transformer objective requires final loss_cls and loss_bbox"
                )
            classification = classification.mean()
            boxes = boxes.mean()
            return classification + boxes / (
                0.01 + boxes / (0.01 + classification)
            )
        return self.detection_loss_total(losses)

    @staticmethod
    def foreground_objective(candidates: Sequence[CandidateSet]) -> torch.Tensor:
        values = [
            candidate.scores.clamp(min=1.0e-8).log().mean()
            for candidate in candidates
            if candidate.scores.numel()
        ]
        if not values:
            raise RuntimeError("Detector exposed no foreground candidates")
        return torch.stack(values).mean()
