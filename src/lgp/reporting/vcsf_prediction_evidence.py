"""Read one hash-bound prediction cell without modifying its evidence root.

This is input validation, not root acceptance or scientific inference. The caller
must obtain expected identities and hashes from an independently accepted receipt.
JSON materializes one cell at a time; the resource cap is not a constant-RAM claim.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def _no_constant(value):
    raise ValueError("Non-finite JSON constant: " + value)


def _same_json(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _plain_path(path, directory=False):
    path = Path(path).absolute()
    _require(not any(part.is_symlink() for part in (path, *path.parents)),
        "Symlink is not accepted in prediction evidence or scratch paths")
    _require(path.is_dir() if directory else path.is_file(), "Missing prediction evidence or scratch path")
    return path.resolve(strict=True)


def _inside(path, root):
    return path == root or root in path.parents


def _bound_json(path, expected_sha256):
    _require(_sha(expected_sha256), "Invalid bound JSON SHA-256")
    path = _plain_path(path)
    with path.open("rb") as handle:
        raw = handle.read((4 << 20) + 1)
    _require(len(raw) <= 4 << 20, "Prediction metadata exceeds the metadata resource cap")
    _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Bound JSON SHA-256 mismatch")
    value = json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    _require(isinstance(value, dict), "Prediction metadata must be an object")
    return value


def _ids(values, name, minimum):
    _require(isinstance(values, (list, tuple)) and values
        and all(type(value) is int and value >= minimum for value in values), "Invalid " + name)
    result = list(values)
    _require(result == sorted(set(result)), "Expected unique sorted " + name)
    return result


def _number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _validate_records(predictions, artifact, image_ids, category_ids):
    _require(isinstance(predictions, list) and len(predictions) == artifact["prediction_records"],
        "Prediction array/count does not match bound evidence")
    images, categories, detected_images = set(image_ids), set(category_ids), set()
    for record in predictions:
        _require(isinstance(record, dict) and set(record) == {"image_id", "category_id", "bbox", "score"},
            "Unexpected prediction fields")
        _require(type(record["image_id"]) is int and record["image_id"] in images
            and type(record["category_id"]) is int and record["category_id"] in categories,
            "Prediction image/category is outside the accepted evaluation scope")
        bbox, score = record["bbox"], record["score"]
        _require(isinstance(bbox, list) and len(bbox) == 4 and all(_number(x) for x in bbox)
            and bbox[2] >= 0 and bbox[3] >= 0 and _number(score) and 0 <= score <= 1,
            "Invalid prediction box or score")
        _require(all(_number(value) for value in
            (bbox[0] + bbox[2], bbox[1] + bbox[3], bbox[2] * bbox[3])),
            "Prediction derived geometry overflows")
        detected_images.add(record["image_id"])
    _require(len(detected_images) == artifact["image_ids_with_detections"],
        "Detected-image count disagrees with bound evidence")


def load_bound_predictions(evaluation_dir, *, immutable_root, metrics_sha256,
        artifact_sha256, expected_identity, image_ids, category_ids,
        scratch_dir, max_uncompressed_bytes):
    """Return original-order predictions and metadata after one-cell validation.

    Scratch must be outside the whole immutable run root. Only owned temporary
    files are written there; neither original JSON nor gzip is rewritten. An
    oversized, failed or inconsistent cell blocks analysis instead of being skipped.
    """
    root = _plain_path(immutable_root, directory=True)
    directory = _plain_path(evaluation_dir, directory=True)
    scratch = _plain_path(scratch_dir, directory=True)
    _require(_inside(directory, root) and not _inside(scratch, root) and not _inside(root, scratch),
        "Scratch and immutable evidence must be disjoint")
    _require(type(max_uncompressed_bytes) is int and max_uncompressed_bytes >= 2,
        "A positive explicit per-cell JSON resource cap is required")
    image_ids = _ids(image_ids, "image IDs", 0)
    category_ids = _ids(category_ids, "category IDs", 1)
    ids_sha256 = hashlib.sha256(json.dumps(image_ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    metrics_path, artifact_path = directory / "metrics.json", directory / "predictions_artifact.json"
    metrics = _bound_json(metrics_path, metrics_sha256)
    artifact = _bound_json(artifact_path, artifact_sha256)
    required_identity = {"dataset", "split", "source", "attack", "target",
        "parameters_sha256", "checkpoint_sha256", "code_commit"}
    _require(isinstance(expected_identity, dict) and required_identity <= set(expected_identity),
        "Incomplete accepted cell identity")
    _require(all(type(expected_identity[key]) is str and expected_identity[key].strip()
        for key in required_identity), "Accepted cell identity must contain nonempty strings")
    for key, expected in expected_identity.items():
        _require(key in metrics and _same_json(metrics[key], expected),
            "Prediction cell identity mismatch: " + key)
    _require(_sha(expected_identity["parameters_sha256"]) and _sha(expected_identity["checkpoint_sha256"]),
        "Missing accepted parameter/checkpoint identity")
    _require(metrics.get("status") == "complete" and metrics.get("failures") == []
        and type(metrics.get("images")) is int and metrics["images"] == len(image_ids),
        "Incomplete or failed prediction cell")
    _require(metrics.get("evaluated_image_ids_sha256") == ids_sha256
        and metrics.get("expected_image_ids_sha256") == ids_sha256
        and metrics.get("evaluated_image_ids_match_expected") is True,
        "Prediction cell used a different image set or order")
    raw_metrics = metrics.get("metrics")
    _require(isinstance(raw_metrics, dict) and set(COCO_BBOX_METRICS) <= set(raw_metrics)
        and all(_number(raw_metrics[key]) and (raw_metrics[key] == -1 or 0 <= raw_metrics[key] <= 1)
            for key in COCO_BBOX_METRICS) and 0 <= raw_metrics["bbox_mAP"] <= 1,
        "Missing, undefined-primary or wrongly scaled raw COCO metrics")
    _require(type(artifact.get("schema_version")) is int and artifact["schema_version"] == 1
        and artifact.get("status") == "verified_lossless_archive" and artifact.get("format") == "gzip"
        and artifact.get("archive_file") == "predictions.json.gz"
        and artifact.get("uncompressed_file") == "predictions.json"
        and artifact.get("fields") == ["image_id", "category_id", "bbox", "score"],
        "Unsupported prediction archive schema")
    for field in ("archive_bytes", "uncompressed_bytes", "prediction_records", "image_ids_with_detections"):
        _require(type(artifact.get(field)) is int and artifact[field] >= 0, "Invalid archive count: " + field)
    _require(2 <= artifact["uncompressed_bytes"] <= max_uncompressed_bytes
        and 0 < artifact["archive_bytes"] <= max_uncompressed_bytes + (1 << 20),
        "Prediction archive exceeds the per-cell resource cap")
    _require(_sha(artifact.get("archive_sha256")) and _sha(artifact.get("uncompressed_sha256"))
        and type(metrics.get("detections")) is int and metrics["detections"] == artifact["prediction_records"]
        and metrics.get("predictions_sha256") == artifact["uncompressed_sha256"]
        and _same_json(metrics.get("predictions_artifact"), artifact),
        "Prediction archive provenance disagrees with metrics")
    _require(not (directory / "predictions.json").exists()
        and not (directory / "predictions.json").is_symlink(), "Unexpected raw prediction side file")
    archive_path = _plain_path(directory / artifact["archive_file"])
    _require(archive_path.stat().st_size == artifact["archive_bytes"], "Compressed byte-count mismatch")

    # Freeze the bytes actually consumed, so later parsing cannot race a path rewrite.
    with tempfile.SpooledTemporaryFile(max_size=8 << 20, dir=str(scratch)) as compressed:
        compressed_digest, compressed_bytes = hashlib.sha256(), 0
        with archive_path.open("rb") as source:
            while True:
                block = source.read(8 << 20)
                if not block:
                    break
                compressed_bytes += len(block)
                _require(compressed_bytes <= artifact["archive_bytes"], "Compressed file grew during read")
                compressed_digest.update(block)
                compressed.write(block)
        _require(compressed_bytes == artifact["archive_bytes"]
            and compressed_digest.hexdigest() == artifact["archive_sha256"], "Compressed SHA-256 mismatch")
        compressed.seek(0)
        with tempfile.SpooledTemporaryFile(max_size=8 << 20, dir=str(scratch)) as raw:
            raw_digest, raw_bytes = hashlib.sha256(), 0
            with gzip.GzipFile(fileobj=compressed, mode="rb") as source:
                while True:
                    block = source.read(min(8 << 20, artifact["uncompressed_bytes"] + 1 - raw_bytes))
                    if not block:
                        break
                    raw_bytes += len(block)
                    _require(raw_bytes <= artifact["uncompressed_bytes"], "Uncompressed byte-count overflow")
                    raw_digest.update(block)
                    raw.write(block)
            _require(raw_bytes == artifact["uncompressed_bytes"]
                and raw_digest.hexdigest() == artifact["uncompressed_sha256"], "Uncompressed SHA-256 mismatch")
            raw.seek(0)
            predictions = json.load(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    _validate_records(predictions, artifact, image_ids, category_ids)
    _require(file_digest(_plain_path(archive_path)) == artifact["archive_sha256"],
        "Prediction archive changed during validation")
    _bound_json(metrics_path, metrics_sha256)
    _bound_json(artifact_path, artifact_sha256)
    return predictions, {"status": "cell_bytes_verified_pending_root_and_replay_acceptance",
        "metrics": metrics, "artifact": artifact, "image_ids_sha256": ids_sha256,
        "metrics_sha256": metrics_sha256, "artifact_sha256": artifact_sha256,
        "images_without_detections": len(image_ids) - artifact["image_ids_with_detections"],
        "scientific_acceptance": False, "root_acceptance": False}
