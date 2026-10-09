"""Re-evaluate a declared image subset from complete saved bbox predictions."""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import math

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..metrics import COCO_BBOX_METRICS


def _ordered_ids(values, label):
    if (not isinstance(values, (list, tuple)) or not values
            or any(type(value) is not int or value < 0 for value in values)
            or list(values) != sorted(set(values))):
        raise ValueError(label + " must contain unique sorted integer image IDs")
    return set(values)


def evaluate_bbox_subset(dataset, predictions, evaluated_image_ids, subset_image_ids):
    """Return official COCO metrics, not an average of per-image AP.

    The caller must independently authenticate annotation/prediction bytes and
    the complete inference receipt. Detection rows alone cannot prove that an
    image with no detections was evaluated. This helper grants no acceptance.
    """
    full_ids = _ordered_ids(evaluated_image_ids, "Evaluated scope")
    subset = _ordered_ids(subset_image_ids, "Requested subset")
    if not subset <= full_ids:
        raise ValueError("Requested subset is outside the completed inference scope")
    if not isinstance(dataset, dict) or not isinstance(predictions, list):
        raise ValueError("Expected a COCO dataset and prediction array")
    for name in ("images", "categories", "annotations"):
        rows = dataset.get(name)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Invalid COCO collection: " + name)
        ids = [row.get("id") for row in rows]
        if any(type(value) is not int for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Duplicate or invalid COCO identity: " + name)
        if name == "annotations" and any(value <= 0 for value in ids):
            raise ValueError("COCO annotation IDs must be positive; zero is an unmatched sentinel")
    annotation_ids = {row["id"] for row in dataset["images"]}
    categories = {row["id"] for row in dataset["categories"]}
    if not full_ids <= annotation_ids or not categories:
        raise ValueError("Annotation does not cover the complete inference scope")
    for row in dataset["annotations"]:
        if (type(row.get("image_id")) is not int
                or type(row.get("category_id")) is not int
                or row["image_id"] not in annotation_ids
                or row["category_id"] not in categories):
            raise ValueError("Annotation references an unknown image or category")
        box = row.get("bbox")
        area = row.get("area")
        crowd = row.get("iscrowd")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Annotation bbox must have four coordinates")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in box + [area]):
            raise ValueError("Annotation contains a non-finite or non-numeric geometry value")
        if area < 0 or box[2] < 0 or box[3] < 0:
            raise ValueError("Annotation area and bbox extents must be nonnegative")
        if type(crowd) is not int or crowd not in (0, 1):
            raise ValueError("Annotation iscrowd must be zero or one")
    for row in predictions:
        if (not isinstance(row, dict)
                or set(row) != {"image_id", "category_id", "bbox", "score"}
                or type(row["image_id"]) is not int
                or type(row["category_id"]) is not int
                or row["image_id"] not in full_ids
                or row["category_id"] not in categories):
            raise ValueError("Prediction is outside the completed inference scope")
        box = row["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Prediction bbox must have four coordinates")
        values = box + [row["score"]]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("Prediction contains a non-finite or non-numeric value")
        if box[2] < 0 or box[3] < 0 or not 0 <= row["score"] <= 1:
            raise ValueError("Prediction extent or score is invalid")

    # Filter before loadRes, while retaining image entries with zero detections.
    selected_dataset = deepcopy(dataset)
    selected_dataset["images"] = [row for row in selected_dataset["images"] if row["id"] in subset]
    selected_dataset["annotations"] = [row for row in selected_dataset["annotations"] if row["image_id"] in subset]
    selected_predictions = deepcopy([row for row in predictions if row["image_id"] in subset])
    with redirect_stdout(StringIO()):
        truth = COCO()
        truth.dataset = selected_dataset
        truth.createIndex()
        if selected_predictions:
            detected = truth.loadRes(selected_predictions)
        else:
            detected = COCO()
            detected.dataset = deepcopy(selected_dataset)
            detected.dataset["annotations"] = []
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.imgIds = list(subset_image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    metrics = dict(zip(COCO_BBOX_METRICS, map(float, evaluator.stats)))
    if len(metrics) != 12 or any(not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Official COCO evaluation did not produce twelve finite metrics")
    return {"metrics": metrics, "image_ids": list(subset_image_ids),
            "images": len(subset), "detections": len(selected_predictions),
            "model_calls": 0, "root_acceptance": False,
            "scientific_acceptance": False}
