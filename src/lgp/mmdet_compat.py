from __future__ import annotations

import hashlib
import inspect
from typing import Any, Dict, Optional

import mmdet
import torch
from mmengine.structures import InstanceData
from mmdet.models.dense_heads.reppoints_head import RepPointsHead
from mmdet.models.task_modules.assigners.assign_result import AssignResult
from mmdet.models.task_modules.assigners.point_assigner import PointAssigner
from mmdet.registry import TASK_UTILS


_SUPPORTED_MMDET_VERSION = "3.0.0"
_POINT_ASSIGNER_ASSIGN_SHA256 = (
    "5688f8b0302197bdd72f71dffae79fca3746fc876da1f17868f8c9295cba7338"
)
_REPPOINTS_LOSS_BY_FEAT_SHA256 = (
    "b0f40681f38d7fd7540756987d48dda959b577296123252fd7ab69212eb81151"
)
_REPPOINTS_POINT_ASSIGNER_DISPATCH = (
    "self.train_cfg['init']['assigner']['type'] == 'PointAssigner'"
)


def validate_point_assigner_compatibility() -> Dict[str, str]:
    """Fail closed unless the pinned upstream method is exactly the audited one."""
    version = str(mmdet.__version__)
    source = inspect.getsource(PointAssigner.assign).replace("\r\n", "\n")
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    head_source = inspect.getsource(RepPointsHead.loss_by_feat).replace(
        "\r\n", "\n"
    )
    head_source_sha256 = hashlib.sha256(
        head_source.encode("utf-8")
    ).hexdigest()
    if version != _SUPPORTED_MMDET_VERSION:
        raise RuntimeError(
            "The RepPoints device-safe PointAssigner is audited only for "
            "mmdet=={}; found {}".format(_SUPPORTED_MMDET_VERSION, version)
        )
    if source_sha256 != _POINT_ASSIGNER_ASSIGN_SHA256:
        raise RuntimeError(
            "The installed mmdet PointAssigner source does not match the "
            "audited mmdet=={} implementation: expected {}, found {}".format(
                _SUPPORTED_MMDET_VERSION,
                _POINT_ASSIGNER_ASSIGN_SHA256,
                source_sha256,
            )
        )
    if (
        head_source_sha256 != _REPPOINTS_LOSS_BY_FEAT_SHA256
        or head_source.count(_REPPOINTS_POINT_ASSIGNER_DISPATCH) != 1
    ):
        raise RuntimeError(
            "The installed RepPointsHead dispatch does not match the audited "
            "mmdet=={} implementation: expected {}, found {}".format(
                _SUPPORTED_MMDET_VERSION,
                _REPPOINTS_LOSS_BY_FEAT_SHA256,
                head_source_sha256,
            )
        )
    return {
        "mmdet_version": version,
        "upstream_assign_sha256": source_sha256,
        "reppoints_loss_by_feat_sha256": head_source_sha256,
    }


@TASK_UTILS.register_module(name="PointAssigner", force=True)
@TASK_UTILS.register_module()
class LGPDeviceSafePointAssigner(PointAssigner):
    """Pinned PointAssigner with its index tensor on the input device.

    MMDetection 3.0.0 creates ``points_range`` on CPU and then indexes it with
    a CUDA boolean mask during RepPoints training. This implementation is
    byte-for-byte equivalent in algorithm and hyperparameters except for the
    explicit ``device=points.device`` argument on that tensor allocation.

    The class is registered under both its explicit compatibility name and
    the upstream ``PointAssigner`` name.  New resolved RepPoints configs keep
    the upstream name because ``RepPointsHead.loss_by_feat`` uses that literal
    value to select point candidates rather than box candidates.
    """

    def __init__(self, scale: int = 4, pos_num: int = 3) -> None:
        validate_point_assigner_compatibility()
        super().__init__(scale=scale, pos_num=pos_num)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: Optional[InstanceData] = None,
        **kwargs: Any,
    ) -> AssignResult:
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        points = pred_instances.priors

        num_points = points.shape[0]
        num_gts = gt_bboxes.shape[0]

        if num_gts == 0 or num_points == 0:
            assigned_gt_inds = points.new_full(
                (num_points,), 0, dtype=torch.long
            )
            assigned_labels = points.new_full(
                (num_points,), -1, dtype=torch.long
            )
            return AssignResult(
                num_gts=num_gts,
                gt_inds=assigned_gt_inds,
                max_overlaps=None,
                labels=assigned_labels,
            )

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(
            min=1e-6
        )
        scale = self.scale
        gt_bboxes_lvl = (
            (
                torch.log2(gt_bboxes_wh[:, 0] / scale)
                + torch.log2(gt_bboxes_wh[:, 1] / scale)
            )
            / 2
        ).int()
        gt_bboxes_lvl = torch.clamp(
            gt_bboxes_lvl, min=lvl_min, max=lvl_max
        )

        assigned_gt_inds = points.new_zeros(
            (num_points,), dtype=torch.long
        )
        assigned_gt_dist = points.new_full((num_points,), float("inf"))
        points_range = torch.arange(
            points.shape[0], device=points.device
        )

        for idx in range(num_gts):
            gt_lvl = gt_bboxes_lvl[idx]
            lvl_idx = gt_lvl == points_lvl
            points_index = points_range[lvl_idx]
            lvl_points = points_xy[lvl_idx, :]
            gt_point = gt_bboxes_xy[[idx], :]
            gt_wh = gt_bboxes_wh[[idx], :]
            points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
            min_dist, min_dist_index = torch.topk(
                points_gt_dist, self.pos_num, largest=False
            )
            min_dist_points_index = points_index[min_dist_index]
            less_than_recorded_index = min_dist < assigned_gt_dist[
                min_dist_points_index
            ]
            min_dist_points_index = min_dist_points_index[
                less_than_recorded_index
            ]
            assigned_gt_inds[min_dist_points_index] = idx + 1
            assigned_gt_dist[min_dist_points_index] = min_dist[
                less_than_recorded_index
            ]

        assigned_labels = assigned_gt_inds.new_full((num_points,), -1)
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False
        ).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[
                assigned_gt_inds[pos_inds] - 1
            ]

        return AssignResult(
            num_gts=num_gts,
            gt_inds=assigned_gt_inds,
            max_overlaps=None,
            labels=assigned_labels,
        )
