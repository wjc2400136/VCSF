"""Multiplicity-preserving image resampling of an official COCO evaluation cache.

This kernel performs no inference and is not an accepted analysis entry point.
It delegates matching and AP accumulation to the installed COCO API.
"""
from copy import copy, deepcopy
from numbers import Integral

import numpy as np


def bootstrap_primary_ap(evaluator, image_positions):
    """Reaccumulate bbox AP50:95 after sampling complete images with replacement."""
    params = evaluator._paramsEval
    if not params or params.iouType != "bbox" or params.useCats != 1:
        raise ValueError("A completed category-aware COCO bbox evaluation is required")
    image_ids = list(params.imgIds)
    if not image_ids or image_ids != sorted(set(image_ids)):
        raise ValueError("The evaluation cache must use unique, sorted image IDs")
    n = len(image_ids)
    positions = list(image_positions)
    if len(positions) != n or any(isinstance(i, (bool, np.bool_))
        or not isinstance(i, Integral) or not 0 <= i < n for i in positions):
        raise ValueError("Resample exactly N valid integer image positions, with multiplicity")
    positions = sorted(int(i) for i in positions)
    if not np.array_equal(params.iouThrs, np.linspace(0.5, 0.95, 10)) or not np.array_equal(
        params.recThrs, np.linspace(0.0, 1.0, 101)) or list(params.maxDets) != [1, 10, 100]:
        raise ValueError("Unexpected primary COCO AP thresholds or maxDets")
    areas = [i for i, label in enumerate(params.areaRngLbl) if label == "all"]
    if len(areas) != 1 or list(params.areaRng[areas[0]]) != [0, 100000 ** 2]:
        raise ValueError("Missing canonical all-area COCO evaluation")
    cats, area_count = len(params.catIds), len(params.areaRng)
    if not cats or len(set(params.catIds)) != cats or len(evaluator.evalImgs) != cats * area_count * n:
        raise ValueError("Incomplete per-image/category/area evaluation cache")
    for category in range(cats):
        start = (category * area_count + areas[0]) * n
        for position in range(n):
            row = evaluator.evalImgs[start + position]
            if row is not None and (row["image_id"] != image_ids[position]
                or row["category_id"] != params.catIds[category]
                or row["maxDet"] != 100 or list(row["aRng"]) != list(params.areaRng[areas[0]])):
                raise ValueError("Evaluation-cache row identity mismatch")
    replica = copy(evaluator)
    replica.params = deepcopy(params)
    # Unique occurrence IDs prevent COCO's set-based indexing from losing draws.
    replica.params.imgIds = list(range(n))
    replica.params.areaRng = [deepcopy(params.areaRng[areas[0]])]
    replica.params.areaRngLbl = ["all"]
    replica.params.maxDets = [100]
    replica._paramsEval = deepcopy(replica.params)
    replica.evalImgs = [evaluator.evalImgs[(category * area_count + areas[0]) * n + position]
        for category in range(cats) for position in positions]
    replica.eval = {}
    replica.accumulate()
    precision = replica.eval["precision"]
    if (not isinstance(precision, np.ndarray) or precision.shape != (10, 101, cats, 1, 1)
        or not np.isfinite(precision).all()
        or not ((precision == -1) | ((precision >= 0) & (precision <= 1))).all()):
        raise ValueError("Invalid precision in resampled COCO evaluation")
    values = precision[precision != -1]
    if not values.size:
        raise ValueError("Resampled AP is undefined because no evaluable ground truth remains")
    return float(np.mean(values))
