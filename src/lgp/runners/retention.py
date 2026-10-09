from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..metrics import ATTACK_QUALITY_METRICS, COCO_BBOX_METRICS
from ..modeling import checkpoint_path
from ..registry import Registry


RETENTION_MODE = "fixed_count_after_group_validation"
SELECTION_ALGORITHM = "canonical_equal_bins_center_v1"
PREDICTION_ARCHIVE_GZIP = "gzip"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _value_digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError("Missing required JSON artifact: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError("Missing required JSONL artifact: {}".format(path))
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Invalid JSONL at {} line {}".format(path, line_number)
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(
                    "JSONL row {} in {} is not an object".format(line_number, path)
                )
            rows.append(row)
    return rows


def _inside(path: Path, root: Path) -> Path:
    absolute = Path(os.path.abspath(str(path)))
    root_absolute = Path(os.path.abspath(str(root)))
    try:
        absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise RuntimeError(
            "Refusing to operate outside {}: {}".format(
                root_absolute, absolute
            )
        ) from exc
    current = absolute
    while True:
        if current.is_symlink():
            raise RuntimeError(
                "Refusing a symlink in a formal artifact path: {}".format(
                    current
                )
            )
        if current == root_absolute:
            break
        current = current.parent
    resolved = absolute.resolve()
    root_resolved = root_absolute.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            "Resolved artifact path escapes {}: {}".format(
                root_resolved, resolved
            )
        ) from exc
    return resolved


def _artifact_path(value: Any, root: Path) -> Path:
    candidate = Path(str(value or ""))
    if not candidate.is_absolute():
        candidate = root / candidate
    return _inside(candidate, root)


def build_retained_image_selection(
    registry: Registry,
    dataset_id: str,
    split: str,
    retained_count: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Freeze one dataset-only image subset before any attack is generated."""
    dataset = registry.dataset(dataset_id)
    index = CocoIndex(dataset, split)
    image_count = len(index.images)
    expected = dataset.split(split).expected_images
    _require(
        image_count == expected,
        "{} {} contains {} images, expected {}".format(
            dataset_id, split, image_count, expected
        ),
    )
    if retained_count <= 0 or retained_count >= image_count:
        raise ValueError(
            "retained_count must lie in [1, {}]".format(image_count - 1)
        )
    positions = [
        ((2 * index_value + 1) * image_count) // (2 * retained_count)
        for index_value in range(retained_count)
    ]
    _require(
        len(set(positions)) == retained_count,
        "The retention selection algorithm produced duplicate positions",
    )
    selected = []
    for position in positions:
        image = index.images[position]
        selected.append(
            {
                "position": position,
                "image_id": int(image["id"]),
                "file_name": str(image["file_name"]),
            }
        )
    all_ids = [int(image["id"]) for image in index.images]
    all_names = [str(image["file_name"]) for image in index.images]
    payload = {
        "schema_version": 1,
        "status": "frozen_before_generation",
        "dataset": dataset_id,
        "split": split,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_inputs": {
            "ordered_by": "numeric image_id",
            "dataset_image_count": image_count,
            "retained_count": retained_count,
            "uses_annotations_or_model_outputs": False,
        },
        "annotation_sha256": file_digest(index.annotation_path),
        "all_image_ids_sha256": _value_digest(all_ids),
        "all_file_names_sha256": _value_digest(all_names),
        "retained_image_ids_sha256": _value_digest(
            [item["image_id"] for item in selected]
        ),
        "retained_file_names_sha256": _value_digest(
            [item["file_name"] for item in selected]
        ),
        "retained": selected,
        "created_at": _utc_now(),
    }
    atomic_json(output_path, payload)
    reread = _read_json(output_path)
    _require(reread == payload, "Retention selection did not round-trip exactly")
    return payload


def archive_predictions(
    predictions_path: Path,
    archive_format: str,
) -> Dict[str, Any]:
    """Validate, losslessly archive, and then remove one raw predictions JSON."""
    if archive_format != PREDICTION_ARCHIVE_GZIP:
        raise ValueError("Unsupported prediction archive format: {}".format(archive_format))
    predictions_path = predictions_path.resolve()
    payload = _read_json(predictions_path)
    if not isinstance(payload, list):
        raise RuntimeError("Predictions must be a JSON array: {}".format(predictions_path))
    image_ids = set()
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise RuntimeError("Prediction {} is not an object".format(index))
        if set(record) != {"image_id", "category_id", "bbox", "score"}:
            raise RuntimeError(
                "Prediction {} has unexpected fields: {}".format(
                    index, sorted(record)
                )
            )
        image_id = int(record["image_id"])
        category_id = int(record["category_id"])
        if image_id < 0 or category_id <= 0:
            raise RuntimeError(
                "Prediction {} has an invalid image/category ID".format(index)
            )
        bbox = record["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError("Prediction {} bbox is not length four".format(index))
        numeric = [float(value) for value in bbox] + [float(record["score"])]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("Prediction {} contains a non-finite value".format(index))
        if numeric[2] < 0.0 or numeric[3] < 0.0:
            raise RuntimeError("Prediction {} has a negative bbox extent".format(index))
        if numeric[4] < 0.0 or numeric[4] > 1.0:
            raise RuntimeError("Prediction {} score is outside [0, 1]".format(index))
        image_ids.add(image_id)

    raw_sha256 = file_digest(predictions_path)
    raw_bytes = predictions_path.stat().st_size
    archive_path = predictions_path.with_suffix(predictions_path.suffix + ".gz")
    audit_path = predictions_path.with_name("predictions_artifact.json")
    if archive_path.exists() or audit_path.exists():
        raise FileExistsError(
            "Refusing to overwrite a prediction archive beside {}".format(
                predictions_path
            )
        )

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".{}.".format(archive_path.name),
            suffix=".part",
            dir=str(archive_path.parent),
            delete=False,
        ) as raw_output:
            temporary_name = raw_output.name
            with predictions_path.open("rb") as source:
                with gzip.GzipFile(
                    filename="predictions.json",
                    mode="wb",
                    compresslevel=9,
                    fileobj=raw_output,
                    mtime=0,
                ) as compressed:
                    while True:
                        chunk = source.read(8 << 20)
                        if not chunk:
                            break
                        compressed.write(chunk)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        temporary = Path(temporary_name)
        decompressed_digest = hashlib.sha256()
        decompressed_bytes = 0
        with gzip.open(str(temporary), "rb") as handle:
            while True:
                chunk = handle.read(8 << 20)
                if not chunk:
                    break
                decompressed_digest.update(chunk)
                decompressed_bytes += len(chunk)
        _require(
            decompressed_digest.hexdigest() == raw_sha256,
            "Prediction archive decompression SHA-256 mismatch",
        )
        _require(
            decompressed_bytes == raw_bytes,
            "Prediction archive decompression byte-count mismatch",
        )
        os.replace(str(temporary), str(archive_path))
        temporary_name = None
        archive_sha256 = file_digest(archive_path)
        audit = {
            "schema_version": 1,
            "status": "verified_lossless_archive",
            "format": PREDICTION_ARCHIVE_GZIP,
            "archive_file": archive_path.name,
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_path.stat().st_size,
            "uncompressed_file": predictions_path.name,
            "uncompressed_sha256": raw_sha256,
            "uncompressed_bytes": raw_bytes,
            "prediction_records": len(payload),
            "image_ids_with_detections": len(image_ids),
            "fields": ["image_id", "category_id", "bbox", "score"],
            "verified_at": _utc_now(),
        }
        atomic_json(audit_path, audit)
        _require(
            file_digest(archive_path) == archive_sha256,
            "Prediction archive changed after audit write",
        )
        predictions_path.unlink()
        _require(
            not predictions_path.exists() and archive_path.is_file(),
            "Prediction archive publication did not reach its final state",
        )
        return audit
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _write_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".{}.".format(path.name),
            suffix=".part",
            dir=str(path.parent),
            delete=False,
        ) as raw_output:
            temporary_name = raw_output.name
            with gzip.GzipFile(
                filename=path.stem,
                mode="wb",
                compresslevel=9,
                fileobj=raw_output,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as text:
                    for row in rows:
                        text.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                        text.write("\n")
            raw_output.flush()
            os.fsync(raw_output.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _validate_prediction_archive(
    evaluation_dir: Path,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    artifact = metrics.get("predictions_artifact")
    _require(isinstance(artifact, Mapping), "Evaluation has no prediction archive audit")
    _require(
        artifact.get("status") == "verified_lossless_archive",
        "Prediction archive is not verified",
    )
    _require(
        artifact.get("format") == PREDICTION_ARCHIVE_GZIP,
        "Formal all-method predictions must use the gzip archive",
    )
    archive = _artifact_path(artifact.get("archive_file"), evaluation_dir)
    _require(archive.is_file(), "Missing prediction archive: {}".format(archive))
    _require(not (evaluation_dir / "predictions.json").exists(), "Raw predictions were not removed")
    _require(_is_sha256(artifact.get("archive_sha256")), "Invalid archive SHA-256")
    _require(
        file_digest(archive) == artifact["archive_sha256"],
        "Prediction archive SHA-256 mismatch: {}".format(archive),
    )
    _require(
        int(artifact.get("prediction_records", -1)) == int(metrics.get("detections", -2)),
        "Prediction record count does not match metrics detections",
    )
    _require(
        artifact.get("uncompressed_sha256") == metrics.get("predictions_sha256"),
        "Prediction archive/raw SHA-256 provenance mismatch",
    )
    audit_path = evaluation_dir / "predictions_artifact.json"
    _require(audit_path.is_file(), "Prediction archive audit file is missing")
    _require(
        _read_json(audit_path) == dict(artifact),
        "Prediction archive audit file does not match metrics.json",
    )
    return dict(artifact)


def _quarantine_and_delete_verified(
    attack_dir: Path,
    images_dir: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Validate all candidates, atomically quarantine them, then delete.

    No image is moved until the complete candidate set has passed its path,
    symlink, size and SHA-256 checks. A move-phase failure rolls every moved
    file back before the error is propagated. The quarantine lives on the
    same filesystem as the image payload, so each individual move is atomic.
    """
    attack_dir = attack_dir.resolve()
    images_dir = _inside(images_dir, attack_dir)
    quarantine = _inside(attack_dir / ".prune-quarantine", attack_dir)
    _require(not quarantine.exists(), "A prune quarantine already exists")
    candidates: List[Tuple[Path, Path, str]] = []
    destinations = set()
    for row in rows:
        candidate = _inside(attack_dir / str(row["output_file"]), images_dir)
        _require(
            candidate.is_file() and not candidate.is_symlink(),
            "Unsafe prune candidate: {}".format(candidate),
        )
        expected_sha = str(row.get("output_sha256", ""))
        _require(_is_sha256(expected_sha), "Prune candidate has no valid SHA-256")
        _require(
            int(row.get("output_bytes", -1)) == candidate.stat().st_size,
            "Prune candidate byte-count mismatch",
        )
        _require(
            file_digest(candidate) == expected_sha,
            "Prune candidate changed after planning",
        )
        relative = candidate.relative_to(images_dir)
        destination = _inside(quarantine / relative, quarantine)
        _require(destination not in destinations, "Duplicate prune destination")
        destinations.add(destination)
        candidates.append((candidate, destination, expected_sha))

    quarantine.mkdir(parents=True, exist_ok=False)
    moved: List[Tuple[Path, Path, str]] = []
    try:
        for candidate, destination, expected_sha in candidates:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(candidate), str(destination))
            moved.append((candidate, destination, expected_sha))
        for candidate, destination, expected_sha in moved:
            _require(not candidate.exists(), "Prune source still exists after quarantine")
            _require(
                destination.is_file()
                and not destination.is_symlink()
                and file_digest(destination) == expected_sha,
                "Quarantined prune candidate failed verification",
            )
    except Exception:
        rollback_failures = []
        for candidate, destination, _ in reversed(moved):
            try:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and not candidate.exists():
                    os.replace(str(destination), str(candidate))
            except OSError as rollback_error:
                rollback_failures.append(str(rollback_error))
        if rollback_failures:
            raise RuntimeError(
                "Prune quarantine failed and rollback was incomplete: {}".format(
                    "; ".join(rollback_failures)
                )
            )
        try:
            quarantine.rmdir()
        except OSError:
            pass
        raise

    for _, destination, expected_sha in moved:
        _require(
            file_digest(destination) == expected_sha,
            "Quarantined file changed before deletion",
        )
        destination.unlink()
    for directory in sorted(
        (path for path in quarantine.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.rmdir()
    quarantine.rmdir()


def validate_and_prune_attack_group(
    registry: Registry,
    dataset_id: str,
    split: str,
    source_id: str,
    attack_id: str,
    attack_dir: Path,
    evaluation_dirs: Mapping[str, Path],
    target_ids: Sequence[str],
    retained_selection: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed, then retain only the pre-frozen image payload for one group."""
    attack_dir = attack_dir.resolve()
    images_dir = _inside(attack_dir / "images", attack_dir)
    run_path = attack_dir / "run.json"
    manifest_path = attack_dir / "manifest.jsonl"
    annotation_path = attack_dir / "annotations.json"
    run = _read_json(run_path)
    manifest = _read_jsonl(manifest_path)
    annotation = _read_json(annotation_path)
    dataset = registry.dataset(dataset_id)
    index = CocoIndex(dataset, split)
    expected_images = dataset.split(split).expected_images
    canonical_targets = list(registry.target_ids())
    _require(list(target_ids) == canonical_targets, "Group targets are not canonical 1-16 order")
    _require(set(evaluation_dirs) == set(canonical_targets), "Group is missing target evaluation directories")
    _require(run.get("status") == "complete", "Attack generation is not complete")
    _require(run.get("dataset") == dataset_id, "Attack dataset mismatch")
    _require(run.get("split") == split, "Attack split mismatch")
    _require(run.get("source") == source_id, "Attack source mismatch")
    _require(run.get("attack") == attack_id, "Attack method mismatch")
    _require(int(run.get("requested_images", -1)) == expected_images, "Attack requested-image count mismatch")
    _require(int(run.get("successful_images", -1)) == expected_images, "Attack success count mismatch")
    _require(int(run.get("failed_images", -1)) == 0, "Attack generation contains failures")

    source_checkpoint = checkpoint_path(registry.model(source_id), dataset)
    _require(source_checkpoint.is_file(), "Missing source checkpoint")
    _require(
        run.get("checkpoint_sha256") == file_digest(source_checkpoint),
        "Attack source checkpoint SHA-256 mismatch",
    )
    quality = run.get("quality_mean")
    _require(isinstance(quality, Mapping), "Attack quality summary is absent")
    quality_definition = run.get("quality_definition") or {}
    _require(
        float(quality_definition.get("psnr_identical_cap_db", -1.0)) == 100.0,
        "Attack quality summary does not declare the 100 dB identity PSNR cap",
    )
    finite_counts = run.get("quality_finite_count") or {}
    nonfinite_counts = run.get("quality_nonfinite_count") or {}
    for metric in ATTACK_QUALITY_METRICS:
        value = quality.get(metric)
        _require(
            value is not None and math.isfinite(float(value)),
            "Attack quality metric '{}' is missing or non-finite".format(metric),
        )
        _require(
            int(finite_counts.get(metric, -1)) == expected_images
            and int(nonfinite_counts.get(metric, -1)) == 0,
            "Attack quality metric '{}' does not have {}/{} finite values".format(
                metric, expected_images, expected_images
            ),
        )
    for field in (
        "runtime_seconds_total",
        "runtime_seconds_mean",
        "peak_cuda_memory_mb",
    ):
        _require(
            run.get(field) is not None and math.isfinite(float(run[field])),
            "Attack {} is missing or non-finite".format(field),
        )
    gradient_evaluations = int(run.get("gradient_evaluations_per_image", -1))
    _require(
        0 <= gradient_evaluations <= 20,
        "Attack gradient-equivalent accounting violates the formal budget",
    )

    _require(len(manifest) == expected_images, "Attack manifest row count mismatch")
    manifest_by_id: Dict[int, Dict[str, Any]] = {}
    manifest_pixel_identical = 0
    manifest_gradient_evaluations = 0
    for row in manifest:
        _require(row.get("status") == "ok", "Attack manifest contains a failed row")
        image_id = int(row["image_id"])
        _require(image_id not in manifest_by_id, "Duplicate image_id in attack manifest")
        expected_relative = "images/{:012d}.png".format(image_id)
        _require(row.get("output_file") == expected_relative, "Unexpected attack image path")
        image_path = _inside(attack_dir / expected_relative, images_dir)
        _require(image_path.is_file() and not image_path.is_symlink(), "Missing or unsafe attack image")
        _require(_is_sha256(row.get("output_sha256")), "Attack image has no valid SHA-256")
        _require(file_digest(image_path) == row["output_sha256"], "Attack image SHA-256 mismatch")
        _require(int(row.get("output_bytes", -1)) == image_path.stat().st_size, "Attack image byte-count mismatch")
        image_gradient_evaluations = int(
            row.get("actual_gradient_evaluations", -1)
        )
        _require(
            0 <= image_gradient_evaluations <= gradient_evaluations,
            "Attack manifest has invalid per-image gradient accounting",
        )
        manifest_gradient_evaluations += image_gradient_evaluations
        manifest_pixel_identical += int(bool(row.get("pixel_identical")))
        manifest_by_id[image_id] = row

    canonical_ids = [int(image["id"]) for image in index.images]
    _require(set(manifest_by_id) == set(canonical_ids), "Attack manifest image IDs do not match the formal split")
    _require(
        int(run.get("pixel_identical_images", -1)) == manifest_pixel_identical,
        "Attack pixel-identical image count does not match its manifest",
    )
    _require(
        int(run.get("actual_gradient_evaluations_total", -1))
        == manifest_gradient_evaluations,
        "Attack actual-gradient total does not match its manifest",
    )
    _require(
        math.isclose(
            float(run.get("actual_gradient_evaluations_mean", -1.0)),
            manifest_gradient_evaluations / expected_images,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "Attack actual-gradient mean does not match its manifest",
    )
    if attack_id in {"lgp", "numbod"}:
        empty_target_ids = {
            int(image["id"])
            for image in index.images
            if not index.instances(int(image["id"]))[0]
        }
        for image_id in empty_target_ids:
            row = manifest_by_id[image_id]
            diagnostics = row.get("diagnostics") or []
            _require(
                bool(row.get("pixel_identical"))
                and float((row.get("quality") or {}).get("mse_pixel", -1.0)) == 0.0
                and int(row.get("actual_gradient_evaluations", -1)) == 0
                and any(
                    item.get("empty_target_behavior") == "identity_output"
                    and int(item.get("ground_truth_objects", -1)) == 0
                    for item in diagnostics
                ),
                "Object-indexed empty-target image is not an audited identity output",
            )
    _require(
        retained_selection.get("annotation_sha256") == file_digest(index.annotation_path),
        "Retention selection annotation SHA-256 mismatch",
    )
    _require(
        retained_selection.get("all_image_ids_sha256") == _value_digest(canonical_ids),
        "Retention selection does not describe this formal image set",
    )
    retained_count = int(
        (retained_selection.get("selection_inputs") or {}).get(
            "retained_count", -1
        )
    )
    retained_ids = [
        int(item["image_id"])
        for item in retained_selection.get("retained", [])
    ]
    _require(
        0 < retained_count < expected_images
        and len(retained_ids) == retained_count
        and len(set(retained_ids)) == retained_count,
        "Formal retained set does not match its frozen retained count",
    )
    _require(set(retained_ids).issubset(manifest_by_id), "Retained set is not a subset of generated images")

    annotation_ids = {int(image["id"]) for image in annotation.get("images", [])}
    _require(annotation_ids == set(canonical_ids), "Attack annotation image IDs are incomplete")
    evaluation_audits = []
    canonical_id_digest = _value_digest(canonical_ids)
    for target_id in canonical_targets:
        evaluation_dir = Path(evaluation_dirs[target_id]).resolve()
        metrics_path = evaluation_dir / "metrics.json"
        metrics = _read_json(metrics_path)
        _require(metrics.get("status") == "complete", "{} evaluation is not complete".format(target_id))
        _require(metrics.get("dataset") == dataset_id, "{} evaluation dataset mismatch".format(target_id))
        _require(metrics.get("split") == split, "{} evaluation split mismatch".format(target_id))
        _require(metrics.get("source") == source_id, "{} evaluation source mismatch".format(target_id))
        _require(metrics.get("attack") == attack_id, "{} evaluation attack mismatch".format(target_id))
        _require(metrics.get("target") == target_id, "{} evaluation target mismatch".format(target_id))
        _require(int(metrics.get("images", -1)) == expected_images, "{} image count mismatch".format(target_id))
        _require(metrics.get("failures") == [], "{} has image failures".format(target_id))
        _require(
            int(metrics.get("detections", -1)) >= 0
            and metrics.get("inference_seconds_total") is not None
            and math.isfinite(float(metrics["inference_seconds_total"]))
            and float(metrics["inference_seconds_total"]) >= 0.0
            and metrics.get("inference_seconds_mean") is not None
            and math.isfinite(float(metrics["inference_seconds_mean"]))
            and float(metrics["inference_seconds_mean"]) >= 0.0,
            "{} has invalid detection/time accounting".format(target_id),
        )
        _require(
            metrics.get("evaluated_image_ids_sha256") == canonical_id_digest,
            "{} evaluated image-ID SHA-256 mismatch".format(target_id),
        )
        _require(
            metrics.get("expected_image_ids_sha256") == canonical_id_digest
            and metrics.get("evaluated_image_ids_match_expected") is True,
            "{} expected/evaluated image-ID gate failed".format(target_id),
        )
        metric_values = metrics.get("metrics")
        _require(isinstance(metric_values, Mapping), "{} metrics are absent".format(target_id))
        for metric in COCO_BBOX_METRICS:
            value = metric_values.get(metric)
            _require(
                value is not None and math.isfinite(float(value)),
                "{} {} is missing or non-finite".format(target_id, metric),
            )
        category_ap = metrics.get("per_category_ap")
        _require(isinstance(category_ap, Mapping), "{} per-category AP is absent".format(target_id))
        _require(
            list(category_ap) == list(dataset.classes),
            "{} per-category AP order/content mismatch".format(target_id),
        )
        for category, value in category_ap.items():
            _require(
                value is not None and math.isfinite(float(value)),
                "{} category '{}' AP is non-finite".format(target_id, category),
            )
        target_checkpoint = checkpoint_path(registry.model(target_id), dataset)
        _require(target_checkpoint.is_file(), "Missing target checkpoint: {}".format(target_id))
        _require(
            metrics.get("checkpoint_sha256") == file_digest(target_checkpoint),
            "{} checkpoint SHA-256 mismatch".format(target_id),
        )
        summary = metrics.get("coco_summary")
        _require(isinstance(summary, Mapping), "{} COCO summary is absent".format(target_id))
        summary_rows = summary.get("rows", [])
        _require(len(summary_rows) == 12, "{} COCO summary is not 12 rows".format(target_id))
        _require(
            [int(row.get("order", -1)) for row in summary_rows]
            == list(range(1, 13)),
            "{} COCO summary order is invalid".format(target_id),
        )
        _require(
            _is_sha256(metrics.get("checkpoint_sha256")),
            "{} checkpoint SHA-256 is malformed".format(target_id),
        )
        for row in summary_rows:
            _require(
                math.isfinite(float(row.get("value"))),
                "{} COCO summary contains a non-finite value".format(target_id),
            )
        for key in ("text", "csv", "tex"):
            summary_path = _artifact_path(summary.get(key), evaluation_dir)
            _require(summary_path.is_file(), "{} COCO summary {} is missing".format(target_id, key))
        visualization = metrics.get("prediction_visualizations") or {}
        _require(not bool(visualization.get("enabled")), "Formal group contains prediction overlays")
        visualization_root = evaluation_dir / "visualizations"
        _require(
            not visualization_root.exists()
            or not any(path.is_file() for path in visualization_root.rglob("*")),
            "Formal group published prediction-overlay files",
        )
        prediction_audit = _validate_prediction_archive(evaluation_dir, metrics)
        evaluation_audits.append(
            {
                "target": target_id,
                "metrics_sha256": file_digest(metrics_path),
                "checkpoint_sha256": metrics["checkpoint_sha256"],
                "predictions_archive_sha256": prediction_audit["archive_sha256"],
                "detections": int(metrics.get("detections", 0)),
            }
        )

    retained_set = set(retained_ids)
    all_rows = []
    retained_rows = []
    pruned_rows = []
    bytes_before = 0
    for image_id in canonical_ids:
        original = dict(manifest_by_id[image_id])
        disposition = "retained" if image_id in retained_set else "pruned_after_full_evaluation"
        original["payload_disposition"] = disposition
        all_rows.append(original)
        bytes_before += int(original["output_bytes"])
        compact = {
            "image_id": image_id,
            "output_file": original["output_file"],
            "output_sha256": original["output_sha256"],
            "output_bytes": int(original["output_bytes"]),
            "payload_disposition": disposition,
        }
        if image_id in retained_set:
            retained_rows.append(compact)
        else:
            pruned_rows.append(compact)

    all_manifest_path = attack_dir / "all_images_manifest.jsonl.gz"
    pruned_count = expected_images - retained_count
    retained_manifest_path = attack_dir / "retained_{}_manifest.json".format(
        retained_count
    )
    pruned_manifest_path = attack_dir / "pruned_{}_manifest.json".format(
        pruned_count
    )
    retention_plan_path = attack_dir / "retention_plan.json"
    retention_summary_path = attack_dir / "retention_summary.json"
    retained_annotation_path = attack_dir / "annotations.retained-{}.json".format(
        retained_count
    )
    for publication in (
        all_manifest_path,
        retained_manifest_path,
        pruned_manifest_path,
        retention_plan_path,
        retention_summary_path,
        retained_annotation_path,
    ):
        _require(not publication.exists(), "Refusing to overwrite retention artifact: {}".format(publication))

    _write_gzip_jsonl(all_manifest_path, all_rows)
    atomic_json(
        retained_manifest_path,
        {
            "schema_version": 1,
            "count": len(retained_rows),
            "selection_sha256": retained_selection["retained_image_ids_sha256"],
            "images": retained_rows,
        },
    )
    atomic_json(
        pruned_manifest_path,
        {
            "schema_version": 1,
            "count": len(pruned_rows),
            "images": pruned_rows,
        },
    )
    retained_file_names = {
        image_id: manifest_by_id[image_id]["output_file"] for image_id in retained_ids
    }
    atomic_json(retained_annotation_path, index.adversarial_annotation(retained_file_names))
    gate = {
        "attack_run_sha256": file_digest(run_path),
        "attack_manifest_sha256": file_digest(manifest_path),
        "attack_annotation_sha256": file_digest(annotation_path),
        "source_checkpoint_sha256": run["checkpoint_sha256"],
        "target_evaluations": evaluation_audits,
        "target_count": len(evaluation_audits),
        "metric_count_per_target": len(COCO_BBOX_METRICS),
        "category_count_per_target": len(dataset.classes),
        "image_failures": 0,
    }
    retention_plan = {
        "schema_version": 1,
        "status": "validated_ready_to_prune",
        "mode": RETENTION_MODE,
        "dataset": dataset_id,
        "split": split,
        "source": source_id,
        "attack": attack_id,
        "generated_images": expected_images,
        "retained_images": len(retained_rows),
        "images_to_prune": len(pruned_rows),
        "retained_image_ids_sha256": retained_selection["retained_image_ids_sha256"],
        "gate": gate,
        "published_manifests": {
            "all_images_manifest": all_manifest_path.name,
            "retained_manifest": retained_manifest_path.name,
            "pruned_manifest": pruned_manifest_path.name,
            "retained_annotation": retained_annotation_path.name,
        },
        "created_at": _utc_now(),
    }
    atomic_json(retention_plan_path, retention_plan)

    _quarantine_and_delete_verified(
        attack_dir, images_dir, pruned_rows
    )

    actual_files = sorted(path.resolve() for path in images_dir.glob("*.png"))
    expected_retained_files = sorted(
        _inside(attack_dir / manifest_by_id[image_id]["output_file"], images_dir)
        for image_id in retained_ids
    )
    _require(
        actual_files == expected_retained_files,
        "Retained attack payload is not the exact frozen evidence set",
    )
    bytes_after = 0
    for row in retained_rows:
        retained_path = _inside(attack_dir / str(row["output_file"]), images_dir)
        _require(file_digest(retained_path) == row["output_sha256"], "Retained attack image SHA-256 mismatch")
        bytes_after += retained_path.stat().st_size

    retention_summary = {
        **retention_plan,
        "status": "complete",
        "full_split_generation_and_evaluation": True,
        "full_5000_image_generation_and_evaluation": expected_images == 5000,
        "full_payload_re_evaluation_available": False,
        "retained_payload_available": True,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "bytes_pruned": bytes_before - bytes_after,
        "retention_plan_sha256": file_digest(retention_plan_path),
        "all_images_manifest_sha256": file_digest(all_manifest_path),
        "retained_manifest_sha256": file_digest(retained_manifest_path),
        "pruned_manifest_sha256": file_digest(pruned_manifest_path),
        "retained_annotation_sha256": file_digest(retained_annotation_path),
        "completed_at": _utc_now(),
    }
    atomic_json(retention_summary_path, retention_summary)
    run["payload_state"] = "retained_subset_after_full_evaluation"
    run["full_payload_available"] = False
    run["retained_payload_images"] = len(retained_rows)
    run["retention_summary"] = retention_summary_path.name
    atomic_json(run_path, run)
    _require(
        _read_json(retention_summary_path) == retention_summary,
        "Retention summary did not round-trip exactly",
    )
    return retention_summary
