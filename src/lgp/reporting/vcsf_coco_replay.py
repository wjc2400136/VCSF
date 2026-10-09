"""Official CPU-only bbox replay after separately accepted annotation/archive binding."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import math

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..metrics import COCO_BBOX_METRICS
from .vcsf_image_bootstrap import bootstrap_primary_ap


def replay_coco_bbox(dataset, predictions, expected_raw_metrics, image_ids):
    """Rebuild all twelve metrics and verify identity resampling before bootstrap.

    This function does not attest where its inputs came from. The outer pipeline
    must bind the original annotation and prediction bytes to accepted receipts.
    A returned evaluator is for one cell, not a complete-panel acceptance.
    """
    if (not isinstance(image_ids, (list, tuple)) or not image_ids
            or any(type(value) is not int for value in image_ids)
            or list(image_ids) != sorted(set(image_ids))):
        raise ValueError("Replay requires unique sorted accepted image IDs")
    for name in ("images", "categories", "annotations"):
        if not isinstance(dataset.get(name), list):
            raise ValueError("Incomplete bound COCO annotation structure")
        ids = [row["id"] for row in dataset[name]]
        if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Duplicate or invalid annotation identity: " + name)
    if sorted(row["id"] for row in dataset["images"]) != list(image_ids):
        raise ValueError("Annotation image set differs from the accepted replay scope")
    image_set = set(image_ids)
    category_set = {row["id"] for row in dataset["categories"]}
    if not category_set or any(row["image_id"] not in image_set or row["category_id"] not in category_set
            for row in dataset["annotations"] + predictions):
        raise ValueError("Annotation/prediction category or image identity is outside the replay scope")
    if not isinstance(expected_raw_metrics, dict) or set(expected_raw_metrics) != set(COCO_BBOX_METRICS):
        raise ValueError("Replay requires all twelve accepted raw metrics")
    for value in expected_raw_metrics.values():
        if type(value) not in (int, float) or not math.isfinite(value) or not (value == -1 or 0 <= value <= 1):
            raise ValueError("Replay expected metrics must retain raw units and undefined sentinels")
    if expected_raw_metrics["bbox_mAP"] < 0:
        raise ValueError("Undefined primary AP cannot enter paired inference")

    # COCO.loadRes/evaluate mutate dictionaries; do not mutate the caller's evidence.
    with redirect_stdout(StringIO()):
        truth = COCO()
        truth.dataset = deepcopy(dataset)
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(deepcopy(predictions))
        else:
            detected = COCO()
            detected.dataset = {"images": deepcopy(dataset["images"]),
                "categories": deepcopy(dataset["categories"]), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        replayed = dict(zip(COCO_BBOX_METRICS, map(float, evaluator.stats)))
        if len(evaluator.stats) != 12 or any(not math.isfinite(value) for value in replayed.values()):
            raise ValueError("Official replay did not produce twelve finite summary values")
        deltas = {key: abs(replayed[key] - expected_raw_metrics[key]) for key in COCO_BBOX_METRICS}
        if max(deltas.values()) > 1e-12:
            raise ValueError("Official replay disagrees with accepted raw metrics: " + repr(deltas))
        identity = bootstrap_primary_ap(evaluator, range(len(image_ids)))
    if abs(identity - expected_raw_metrics["bbox_mAP"]) > 1e-12:
        raise ValueError("Identity resample disagrees with accepted primary AP")
    return evaluator, {"status": "cell_replay_verified_pending_pipeline_acceptance",
        "metrics": replayed, "maximum_absolute_metric_delta": max(deltas.values()),
        "identity_resample_bbox_mAP": identity, "comparison_tolerance": 1e-12,
        "model_calls": 0, "root_acceptance": False, "scientific_acceptance": False}
