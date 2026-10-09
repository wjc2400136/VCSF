from __future__ import annotations

from typing import Tuple


COCO_BBOX_METRICS: Tuple[str, ...] = (
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_AR_1",
    "bbox_AR_10",
    "bbox_AR_100",
    "bbox_AR_small",
    "bbox_AR_medium",
    "bbox_AR_large",
)

ATTACK_QUALITY_METRICS: Tuple[str, ...] = (
    "mse_pixel",
    "psnr",
    "ssim",
    "linf_normalized",
    "l2_rms_normalized",
    "l0_fraction_gt_half_pixel",
    "mean_abs_normalized",
    "nmse",
    "total_variation_normalized",
)
