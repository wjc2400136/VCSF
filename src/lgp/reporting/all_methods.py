from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..io import atomic_json, file_digest
from ..data.coco import CocoIndex
from ..metrics import ATTACK_QUALITY_METRICS, COCO_BBOX_METRICS
from ..modeling import checkpoint_path
from ..registry import Registry
from .formatting import format_metric


COMMON_TWO_STAGE_SOURCES = ["faster_rcnn_r50", "mask_rcnn_swin_t"]
PINNED_REPORT_VERSIONS = {
    "torch": "2.0.0+cu118",
    "torchvision": "0.15.1+cu118",
    "mmcv": "2.0.1",
    "mmengine": "0.7.4",
    "mmdet": "3.0.0",
    "mmyolo": "0.6.0",
}


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".{}.".format(path.name),
            suffix=".part",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(value)
            if not value.endswith("\n"):
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_full_split_generation_evidence(
    retention: Mapping[str, Any],
    dataset_id: str,
    expected_images: int,
) -> bool:
    """Accept current evidence or the pinned legacy COCO-5000 marker."""
    generic_key = "full_split_generation_and_evaluation"
    legacy_key = "full_5000_image_generation_and_evaluation"
    if generic_key in retention:
        generic = retention.get(generic_key)
        legacy = retention.get(legacy_key)
        return generic is True and (
            legacy_key not in retention or legacy is True
        )
    return (
        dataset_id == "coco"
        and int(expected_images) == 5000
        and retention.get(legacy_key) is True
    )


def _validate_execution_assignment_payload(
    assignment: Mapping[str, Any],
    scheduler: Mapping[str, Any],
    registry: Registry,
) -> None:
    scheduler_devices = list(scheduler.get("devices") or [])
    clean_assignments = list(assignment.get("clean") or [])
    group_assignments = list(assignment.get("attack_groups") or [])
    worker_count = int(scheduler.get("worker_count", 0))
    clean_slot_counts = [
        sum(int(row.get("worker_slot", -1)) == slot for row in clean_assignments)
        for slot in range(worker_count)
    ]
    group_slot_counts = [
        sum(int(row.get("worker_slot", -1)) == slot for row in group_assignments)
        for slot in range(worker_count)
    ]
    expected_clean_counts = [16] if worker_count == 1 else [8, 8]
    expected_group_counts = [53] if worker_count == 1 else [27, 26]
    if (
        worker_count not in {1, 2}
        or assignment.get("algorithm") != "canonical_round_robin_v1"
        or list(assignment.get("devices") or []) != scheduler_devices
        or assignment.get("global_record_writer") != "coordinator_only"
        or assignment.get("clean_barrier_before_attack_groups") is not True
        or [int(row.get("canonical_index", -1)) for row in clean_assignments]
        != list(range(1, 17))
            or [int(row.get("canonical_index", -1)) for row in group_assignments]
            != list(range(1, 54))
        or clean_slot_counts != expected_clean_counts
        or group_slot_counts != expected_group_counts
        or any(
            list(row.get("targets") or []) != registry.target_ids()
            for row in group_assignments
        )
    ):
        raise RuntimeError("Formal execution assignment manifest is invalid")


def _inside(path: Path, root: Path) -> Path:
    absolute = Path(os.path.abspath(str(path)))
    root_absolute = Path(os.path.abspath(str(root)))
    try:
        absolute.relative_to(root_absolute)
    except ValueError as exc:
        raise RuntimeError(
            "Formal report artifact escapes its immutable run: {}".format(
                absolute
            )
        ) from exc
    current = absolute
    while True:
        if current.is_symlink():
            raise RuntimeError(
                "Formal report artifact path contains a symlink: {}".format(
                    current
                )
            )
        if current == root_absolute:
            break
        current = current.parent
    resolved = absolute.resolve()
    root = root_absolute.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "Resolved formal report artifact escapes its run: {}".format(
                resolved
            )
        ) from exc
    return resolved


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError("Missing formal report artifact: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )


def _is_git_object_id(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) in {40, 64} and all(
        character in "0123456789abcdef" for character in text
    )


def _gzip_uncompressed_identity(path: Path) -> Tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with gzip.open(str(path), "rb") as handle:
        while True:
            chunk = handle.read(8 << 20)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _mean(values: Iterable[Any]) -> Optional[float]:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else None


def _escape_latex(value: Any) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _csv_write(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline="",
            prefix=".{}.".format(path.name),
            suffix=".part",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(fields))
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _evaluation_records(
    records: Sequence[Mapping[str, Any]], dataset: str
) -> List[Mapping[str, Any]]:
    return [
        record
        for record in records
        if record.get("dataset") == dataset
        and record.get("status") == "complete"
        and record.get("attack") not in {None, ""}
        and bool(record.get("metrics"))
    ]


def _validate_complete_records(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_dir: Path,
    *,
    protocol_id: str = "coco_all_methods",
) -> None:
    if summary.get("status") != "complete":
        return
    protocol = dict(registry.protocols.get(protocol_id, {}))
    registered_datasets = list(protocol.get("datasets") or [])
    if (
        protocol.get("formal_all_methods") is not True
        or len(registered_datasets) != 1
    ):
        raise RuntimeError(
            "All-method report protocol registry is incomplete"
        )
    dataset_id = str(registered_datasets[0])
    split = str(protocol.get("split", "val"))
    entrypoint = str(protocol.get("entrypoint", ""))
    retained_count = int(
        (protocol.get("payload_retention") or {}).get(
            "retained_images", -1
        )
    )
    dataset = registry.dataset(dataset_id)
    expected_images = int(dataset.split(split).expected_images)
    pruned_count = expected_images - retained_count
    if (
        retained_count <= 0
        or pruned_count < 0
        or not entrypoint.startswith("experiments/")
        or not entrypoint.endswith(".py")
    ):
        raise RuntimeError(
            "All-method report lifecycle registry is incomplete"
        )
    if summary.get("protocol") != protocol_id:
        raise RuntimeError("All-method report received a different protocol")
    expected = {
        "jobs": 1056,
        "ready": 848,
        "skipped": 208,
        "completed_ready": 848,
        "clean_evaluations": 16,
        "completed_clean": 16,
        "attack_groups": 53,
        "attack_groups_completed": 53,
        "records": 1072,
        "failed_records": 0,
        "image_failures": 0,
    }
    for key, value in expected.items():
        if int(summary.get(key, -1)) != value:
            raise RuntimeError(
                "Formal report summary {}={}, expected {}".format(
                    key, summary.get(key), value
                )
            )
    if summary.get("max_images") is not None:
        raise RuntimeError("Formal all-method report cannot use max_images")
    if summary.get("formal_eligible") is not True:
        raise RuntimeError("Formal all-method report is not eligible")
    expected_seed = int(protocol.get("seed", -1))
    if int(summary.get("seed", -2)) != expected_seed:
        raise RuntimeError(
            "Formal all-method report does not use seed {}".format(
                expected_seed
            )
        )
    if (
        plan.get("protocol") != protocol_id
        or list(plan.get("datasets", [])) != [dataset_id]
        or plan.get("split") != split
        or list(plan.get("methods", []))
        != list(protocol.get("methods", []))
    ):
        raise RuntimeError("Formal report plan identity is invalid")
    if list(plan.get("sources", [])) != registry.source_ids():
        raise RuntimeError("Formal report source order is not canonical")
    if list(plan.get("targets", [])) != registry.target_ids():
        raise RuntimeError("Formal report target order is not canonical")
    expected_record_order = [
        ("clean", "clean", target_id)
        for target_id in registry.target_ids()
    ] + [
        (str(job["source"]), str(job["attack"]), str(job["target"]))
        for job in plan.get("jobs", [])
    ]
    actual_record_order = [
        (
            str(record.get("source", "")),
            str(record.get("attack", "")),
            str(record.get("target", "")),
        )
        for record in records
    ]
    if (
        len(records) != 1072
        or actual_record_order != expected_record_order
        or len(set(actual_record_order)) != 1072
    ):
        raise RuntimeError("Formal records.json order or coverage is invalid")
    scheduler = provenance.get("execution_scheduler") or {}
    scheduler_devices = list(scheduler.get("devices") or [])
    expected_command = "conda run -n oda python {}".format(entrypoint)
    if len(scheduler_devices) > 1:
        expected_command += " --devices {}".format(
            ",".join(str(device) for device in scheduler_devices)
        )
    elif scheduler_devices and scheduler_devices != ["cuda:0"]:
        expected_command += " --device {}".format(scheduler_devices[0])
    expected_command += " --no-visualize-predictions"
    plan_scheduler = plan.get("execution_scheduler") or {}
    if (
        provenance.get("protocol") != protocol_id
        or provenance.get("git_worktree_clean_at_start") is not True
        or not _is_git_object_id(provenance.get("git_commit"))
        or provenance.get("canonical_command") != expected_command
        or not scheduler_devices
        or int(scheduler.get("worker_count", 0)) != len(scheduler_devices)
        or scheduler.get("mode")
        not in {"single_device_serial", "independent_group_workers"}
        or any(
            scheduler.get(key) != plan_scheduler.get(key)
            for key in (
                "mode",
                "devices",
                "worker_count",
                "assignment_algorithm",
                "clean_barrier_before_attack_groups",
                "group_atomicity",
                "global_record_writer",
            )
        )
    ):
        raise RuntimeError("Formal execution provenance is incomplete")
    if not str(provenance.get("python", "")).startswith("3.8."):
        raise RuntimeError("Formal report was not produced with Python 3.8")
    package_versions = provenance.get("packages") or {}
    for package, expected_version in PINNED_REPORT_VERSIONS.items():
        if str(package_versions.get(package)) != expected_version:
            raise RuntimeError(
                "Formal report package {}={}, expected {}".format(
                    package, package_versions.get(package), expected_version
                )
            )

    run_dir = run_dir.resolve()
    assignment_name = str(plan_scheduler.get("assignment_manifest") or "")
    if assignment_name != "execution_assignments.json":
        raise RuntimeError("Formal execution assignment manifest is not fixed")
    assignment_path = _inside(run_dir / assignment_name, run_dir)
    if file_digest(assignment_path) != str(
        plan_scheduler.get("assignment_manifest_sha256") or ""
    ):
        raise RuntimeError("Formal execution assignment manifest hash mismatch")
    assignment = _read_json(assignment_path)
    _validate_execution_assignment_payload(assignment, scheduler, registry)
    evaluations_root = _inside(run_dir / "evaluations", run_dir)
    attacks_root = _inside(run_dir / "attacks", run_dir)

    evaluations = _evaluation_records(records, dataset_id)
    clean = [record for record in evaluations if record.get("attack") == "clean"]
    attacked = [record for record in evaluations if record.get("attack") != "clean"]
    if len(clean) != 16 or len(attacked) != 848:
        raise RuntimeError("Formal evaluation record count is incomplete")
    if {str(record.get("target")) for record in clean} != set(
        registry.target_ids()
    ):
        raise RuntimeError("Formal clean target coverage is incomplete")
    expected_attacked_keys = {
        (str(job["source"]), str(job["attack"]), str(job["target"]))
        for job in plan.get("jobs", [])
        if job.get("status") == "ready"
    }
    actual_attacked_keys = {
        (
            str(record.get("source")),
            str(record.get("attack")),
            str(record.get("target")),
        )
        for record in attacked
    }
    if (
        len(actual_attacked_keys) != len(attacked)
        or actual_attacked_keys != expected_attacked_keys
    ):
        raise RuntimeError("Formal attacked source-method-target coverage is invalid")
    index = CocoIndex(dataset, split)
    canonical_image_digest = hashlib.sha256(
        json.dumps(
            [int(image["id"]) for image in index.images],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    checkpoint_hashes: Dict[str, set] = defaultdict(set)
    attack_runs: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for record in evaluations:
        if int(record.get("images", -1)) != expected_images:
            raise RuntimeError(
                "A formal evaluation did not cover all {} images".format(
                    expected_images
                )
            )
        if record.get("failures") != []:
            raise RuntimeError("A formal evaluation contains image failures")
        if (
            int(record.get("detections", -1)) < 0
            or not _finite(record.get("inference_seconds_total"))
            or not _finite(record.get("inference_seconds_mean"))
            or float(record.get("inference_seconds_total")) < 0.0
            or float(record.get("inference_seconds_mean")) < 0.0
        ):
            raise RuntimeError(
                "A formal evaluation has invalid detection/time accounting"
            )
        if (
            record.get("evaluated_image_ids_sha256") != canonical_image_digest
            or record.get("expected_image_ids_sha256") != canonical_image_digest
            or record.get("evaluated_image_ids_match_expected") is not True
        ):
            raise RuntimeError("A formal evaluation has an invalid image-ID set")
        metrics = record.get("metrics") or {}
        if any(not _finite(metrics.get(metric)) for metric in COCO_BBOX_METRICS):
            raise RuntimeError("A formal evaluation contains an invalid bbox metric")
        category_ap = record.get("per_category_ap") or {}
        if list(category_ap) != list(dataset.classes):
            raise RuntimeError("A formal evaluation has an invalid category order")
        if any(not _finite(category_ap.get(category)) for category in dataset.classes):
            raise RuntimeError("A formal evaluation contains an invalid category AP")
        checkpoint_hashes[str(record.get("target"))].add(
            str(record.get("checkpoint_sha256"))
        )
        target_id = str(record.get("target"))
        checkpoint = checkpoint_path(registry.model(target_id), dataset)
        if (
            not checkpoint.is_file()
            or record.get("checkpoint_sha256") != file_digest(checkpoint)
        ):
            raise RuntimeError("A formal evaluation checkpoint hash is invalid")
        coco_summary = record.get("coco_summary") or {}
        rows = coco_summary.get("rows") or []
        if (
            len(rows) != 12
            or [int(row.get("order", -1)) for row in rows]
            != list(range(1, 13))
            or any(not _finite(row.get("value")) for row in rows)
        ):
            raise RuntimeError("A formal evaluation COCO summary is invalid")
        summary_files = [
            Path(str(coco_summary.get(key, "")))
            for key in ("text", "csv", "tex")
        ]
        if any(not path.is_file() for path in summary_files):
            raise RuntimeError("A formal evaluation COCO summary file is missing")
        evaluation_dir = _inside(summary_files[0].parent, evaluations_root)
        if any(
            _inside(path, evaluations_root).parent != evaluation_dir
            for path in summary_files
        ):
            raise RuntimeError("COCO summary artifacts span evaluation directories")
        prediction = record.get("predictions_artifact") or {}
        archive = _inside(
            evaluation_dir / str(prediction.get("archive_file", "")),
            evaluation_dir,
        )
        prediction_audit = evaluation_dir / "predictions_artifact.json"
        archive_sha = str(prediction.get("archive_sha256", ""))
        uncompressed_sha, uncompressed_bytes = (
            _gzip_uncompressed_identity(archive)
            if archive.is_file()
            else (None, -1)
        )
        if (
            prediction.get("status") != "verified_lossless_archive"
            or prediction.get("format") != "gzip"
            or prediction.get("uncompressed_sha256")
            != record.get("predictions_sha256")
            or int(prediction.get("prediction_records", -1))
            != int(record.get("detections", -2))
            or len(archive_sha) != 64
            or not archive.is_file()
            or file_digest(archive) != archive_sha
            or uncompressed_sha != prediction.get("uncompressed_sha256")
            or uncompressed_bytes
            != int(prediction.get("uncompressed_bytes", -1))
            or not prediction_audit.is_file()
            or json.loads(prediction_audit.read_text(encoding="utf-8"))
            != prediction
            or (evaluation_dir / "predictions.json").exists()
        ):
            raise RuntimeError("A formal evaluation prediction archive is invalid")
        if bool((record.get("prediction_visualizations") or {}).get("enabled")):
            raise RuntimeError("A formal evaluation contains prediction overlays")
        visualization_root = evaluation_dir / "visualizations"
        if visualization_root.exists() and any(
            path.is_file() for path in visualization_root.rglob("*")
        ):
            raise RuntimeError("A formal evaluation published visualization files")
        if record.get("attack") != "clean":
            attack_id = str(record.get("attack"))
            attack_spec = registry.attack(attack_id)
            expected_fidelity = {
                "fidelity_status": attack_spec.metadata.get("fidelity_status"),
                "fidelity_basis": attack_spec.metadata.get("fidelity_basis"),
                "semantic_contract": attack_spec.metadata.get("semantic_contract"),
                "paper": attack_spec.metadata.get("paper"),
                "audited_repository": attack_spec.metadata.get("repository"),
                "audit_commit": attack_spec.metadata.get("audit_commit"),
            }
            if any(
                key not in record or record.get(key) != value
                for key, value in expected_fidelity.items()
            ) or not isinstance(
                record.get("implementation_adjudication_details"), Mapping
            ):
                raise RuntimeError(
                    "Attack fidelity provenance is incomplete for {}/{}".format(
                        record.get("source"), attack_id
                    )
                )
            attack_runs.setdefault(
                (str(record.get("source")), attack_id), record
            )
    if set(checkpoint_hashes) != set(registry.target_ids()):
        raise RuntimeError("The checkpoint-hash panel is incomplete")
    if any(len(values) != 1 for values in checkpoint_hashes.values()):
        raise RuntimeError("A target used more than one checkpoint SHA-256")
    if len(attack_runs) != 53:
        raise RuntimeError("The formal run does not contain 53 attack payloads")
    attacked_by_key = {
        (
            str(record.get("source")),
            str(record.get("attack")),
            str(record.get("target")),
        ): record
        for record in attacked
    }
    for (source_id, attack_id), record in attack_runs.items():
        quality = record.get("attack_quality_mean") or {}
        if any(not _finite(quality.get(metric)) for metric in ATTACK_QUALITY_METRICS):
            raise RuntimeError("An attack payload contains an invalid quality metric")
        if not _finite(record.get("attack_runtime_seconds_mean")):
            raise RuntimeError("An attack payload has no finite generation runtime")
        attack_dir = _inside(
            Path(str(record.get("adversarial_run", ""))), attacks_root
        )
        run_payload = _read_json(attack_dir / "run.json")
        attack_spec = registry.attack(attack_id)
        expected_fidelity = {
            "fidelity_status": attack_spec.metadata.get("fidelity_status"),
            "fidelity_basis": attack_spec.metadata.get("fidelity_basis"),
            "semantic_contract": attack_spec.metadata.get("semantic_contract"),
            "paper": attack_spec.metadata.get("paper"),
            "audited_repository": attack_spec.metadata.get("repository"),
            "audit_commit": attack_spec.metadata.get("audit_commit"),
        }
        if any(
            key not in run_payload or run_payload.get(key) != value
            for key, value in expected_fidelity.items()
        ) or not isinstance(
            run_payload.get("implementation_adjudication_details"), Mapping
        ):
            raise RuntimeError(
                "Attack-run fidelity provenance is incomplete for {}/{}".format(
                    source_id, attack_id
                )
            )
        retention = _read_json(attack_dir / "retention_summary.json")
        finite_counts = run_payload.get("quality_finite_count") or {}
        nonfinite_counts = run_payload.get("quality_nonfinite_count") or {}
        if (
            run_payload.get("payload_state")
            != "retained_subset_after_full_evaluation"
            or run_payload.get("full_payload_available") is not False
            or int(run_payload.get("retained_payload_images", -1))
            != retained_count
            or retention.get("status") != "complete"
            or retention.get("mode") != "fixed_count_after_group_validation"
            or not _has_full_split_generation_evidence(
                retention,
                dataset_id,
                expected_images,
            )
            or retention.get("full_payload_re_evaluation_available") is not False
            or retention.get("retained_payload_available") is not True
            or int(retention.get("generated_images", -1))
            != expected_images
            or int(retention.get("retained_images", -1))
            != retained_count
            or int(retention.get("images_to_prune", -1))
            != pruned_count
            or int((retention.get("gate") or {}).get("target_count", -1)) != 16
            or int((retention.get("gate") or {}).get("image_failures", -1)) != 0
            or retention.get("retained_image_ids_sha256")
            != (plan.get("artifact_lifecycle") or {}).get(
                "retained_image_ids_sha256"
            )
            or any(
                int(finite_counts.get(metric, -1)) != expected_images
                or int(nonfinite_counts.get(metric, -1)) != 0
                for metric in ATTACK_QUALITY_METRICS
            )
            or not _finite(run_payload.get("runtime_seconds_total"))
            or not _finite(run_payload.get("runtime_seconds_mean"))
            or not _finite(run_payload.get("peak_cuda_memory_mb"))
            or int(run_payload.get("gradient_evaluations_per_image", -1)) < 0
            or int(run_payload.get("gradient_evaluations_per_image", 21)) > 20
            or not _finite(
                (run_payload.get("auxiliary_passes") or {}).get("forward_mean")
            )
            or not _finite(
                (run_payload.get("auxiliary_passes") or {}).get("backward_mean")
            )
        ):
            raise RuntimeError(
                "Formal payload retention state is invalid for {}/{}".format(
                    source_id, attack_id
                )
            )
        retention_files = {
            "retention_plan_sha256": "retention_plan.json",
            "all_images_manifest_sha256": "all_images_manifest.jsonl.gz",
            "retained_manifest_sha256": "retained_{}_manifest.json".format(
                retained_count
            ),
            "pruned_manifest_sha256": "pruned_{}_manifest.json".format(
                pruned_count
            ),
            "retained_annotation_sha256": (
                "annotations.retained-{}.json".format(retained_count)
            ),
        }
        for hash_key, file_name in retention_files.items():
            path = _inside(attack_dir / file_name, attack_dir)
            if (
                not path.is_file()
                or not _is_sha256(retention.get(hash_key))
                or file_digest(path) != retention.get(hash_key)
            ):
                raise RuntimeError(
                    "Retention artifact {} is invalid for {}/{}".format(
                        file_name, source_id, attack_id
                    )
                )
        retained_manifest = _read_json(
            attack_dir
            / "retained_{}_manifest.json".format(retained_count)
        )
        pruned_manifest = _read_json(
            attack_dir / "pruned_{}_manifest.json".format(pruned_count)
        )
        retained_rows = retained_manifest.get("images") or []
        if (
            int(retained_manifest.get("count", -1)) != retained_count
            or len(retained_rows) != retained_count
            or int(pruned_manifest.get("count", -1)) != pruned_count
            or len(pruned_manifest.get("images") or []) != pruned_count
        ):
            raise RuntimeError("Retention disposition manifests are incomplete")
        actual_images = sorted((attack_dir / "images").glob("*.png"))
        if len(actual_images) != retained_count or any(
            path.is_symlink() for path in actual_images
        ):
            raise RuntimeError(
                "Retained payload is not exactly {} regular PNGs".format(
                    retained_count
                )
            )
        expected_retained = {}
        for retained_row in retained_rows:
            image_path = _inside(
                attack_dir / str(retained_row.get("output_file", "")),
                attack_dir / "images",
            )
            if (
                not image_path.is_file()
                or not _is_sha256(retained_row.get("output_sha256"))
                or int(retained_row.get("output_bytes", -1))
                != image_path.stat().st_size
                or file_digest(image_path) != retained_row.get("output_sha256")
            ):
                raise RuntimeError("A retained evidence PNG failed its hash gate")
            expected_retained[image_path] = retained_row
        if set(actual_images) != set(expected_retained):
            raise RuntimeError("Retained PNG set does not match its manifest")
        target_audits = (retention.get("gate") or {}).get(
            "target_evaluations"
        ) or []
        if [item.get("target") for item in target_audits] != registry.target_ids():
            raise RuntimeError("Retention target-audit order is invalid")
        for item in target_audits:
            target_id = str(item["target"])
            target_record = attacked_by_key[(source_id, attack_id, target_id)]
            artifact = target_record.get("predictions_artifact") or {}
            target_summary = target_record.get("coco_summary") or {}
            target_metrics_path = _inside(
                Path(str(target_summary.get("text", ""))).parent
                / "metrics.json",
                evaluations_root,
            )
            if (
                not target_metrics_path.is_file()
                or item.get("metrics_sha256") != file_digest(target_metrics_path)
                or item.get("checkpoint_sha256")
                != target_record.get("checkpoint_sha256")
                or item.get("predictions_archive_sha256")
                != artifact.get("archive_sha256")
                or int(item.get("detections", -1))
                != int(target_record.get("detections", -2))
            ):
                raise RuntimeError("Retention target audit does not match records")
        group_status = _inside(
            run_dir
            / "group_status"
            / dataset_id
            / source_id
            / attack_id
            / "default.json",
            run_dir / "group_status",
        )
        group_payload = _read_json(group_status)
        if (
            group_payload.get("status") != "complete_validated_and_pruned"
            or int(group_payload.get("targets_completed", -1)) != 16
            or group_payload.get("retention_summary_sha256")
            != file_digest(attack_dir / "retention_summary.json")
        ):
            raise RuntimeError("Formal group-status audit is incomplete")


def validate_formal_all_methods_bundle(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    *,
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_dir: Path,
    protocol_id: str,
) -> Dict[str, Any]:
    """Fail closed on a completed formal all-method artifact bundle."""
    if summary.get("status") != "complete":
        raise RuntimeError(
            "Formal all-method bundle validation requires complete status"
        )
    run_dir = run_dir.resolve()
    _validate_complete_records(
        records,
        registry,
        summary,
        plan,
        provenance,
        run_dir,
        protocol_id=protocol_id,
    )
    protocol = registry.protocols[protocol_id]
    dataset_id = str(list(protocol["datasets"])[0])
    split = str(protocol.get("split", "val"))
    expected_images = int(
        registry.dataset(dataset_id).split(split).expected_images
    )
    required_files = {
        name: _inside(run_dir / name, run_dir)
        for name in (
            "plan.json",
            "provenance.json",
            "records.json",
            "summary.json",
        )
    }
    if any(not path.is_file() for path in required_files.values()):
        raise RuntimeError("Formal all-method root evidence is incomplete")
    return {
        "schema_version": 1,
        "status": "accepted",
        "protocol": protocol_id,
        "dataset": dataset_id,
        "split": split,
        "expected_images": expected_images,
        "jobs": 1056,
        "completed_evaluations": 880,
        "structural_skips": 208,
        "attack_payloads": 53,
        "retained_images_per_payload": int(
            protocol["payload_retention"]["retained_images"]
        ),
        "root_evidence_sha256": {
            name: file_digest(path)
            for name, path in required_files.items()
        },
    }


def _aggregate_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    method_ids: Sequence[str],
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    values: Dict[Tuple[str, str, str, str], float] = {}
    for record in records:
        if (
            record.get("dataset") != dataset_id
            or record.get("attack") in {None, "", "clean"}
            or record.get("status") != "complete"
        ):
            continue
        for metric in COCO_BBOX_METRICS:
            value = (record.get("metrics") or {}).get(metric)
            if _finite(value):
                values[
                    (
                        str(record.get("source")),
                        str(record.get("attack")),
                        str(record.get("target")),
                        metric,
                    )
                ] = float(value)

    rows: List[Dict[str, Any]] = []
    targets = registry.target_ids()
    all_sources = registry.source_ids()
    for method in method_ids:
        supported_sources = [
            source
            for source in all_sources
            if registry.compatibility_status(method, source)["status"] == "native"
        ]
        for metric in COCO_BBOX_METRICS:
            def selected(source_ids: Sequence[str], blackbox_only: bool = False) -> List[float]:
                output = []
                for source in source_ids:
                    for target in targets:
                        if blackbox_only and source == target:
                            continue
                        value = values.get((source, method, target, metric))
                        if value is not None:
                            output.append(value)
                return output

            supported = selected(supported_sources)
            supported_bb = selected(supported_sources, blackbox_only=True)
            common_two = selected(COMMON_TWO_STAGE_SOURCES)
            common_two_bb = selected(COMMON_TWO_STAGE_SOURCES, blackbox_only=True)
            six_source = selected(all_sources) if supported_sources == all_sources else []
            six_source_bb = (
                selected(all_sources, blackbox_only=True)
                if supported_sources == all_sources
                else []
            )
            whitebox = [
                values[(source, method, source, metric)]
                for source in supported_sources
                if (source, method, source, metric) in values
            ]
            rows.append(
                {
                    "method": method,
                    "method_name": registry.attack(method).display_name,
                    "metric": metric,
                    "supported_sources": len(supported_sources),
                    "expected_supported_cells": len(supported_sources) * len(targets),
                    "observed_supported_cells": len(supported),
                    "common_two_stage_mean": _mean(common_two),
                    "common_two_stage_bb_mean": _mean(common_two_bb),
                    "six_source_mean": _mean(six_source),
                    "six_source_bb_mean": _mean(six_source_bb),
                    "supported_source_mean": _mean(supported),
                    "supported_source_bb_mean": _mean(supported_bb),
                    "supported_whitebox_mean": _mean(whitebox),
                }
            )
    return rows


def _all_metrics_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    protocol_id: str = "coco_all_methods",
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    source_rank = {value: index for index, value in enumerate(registry.source_ids())}
    target_rank = {value: index for index, value in enumerate(registry.target_ids())}
    method_rank = {
        value: index
        for index, value in enumerate(
            registry.protocols[protocol_id]["methods"]
        )
    }
    evaluations = _evaluation_records(records, dataset_id)
    evaluations.sort(
        key=lambda record: (
            0 if record.get("attack") == "clean" else 1,
            source_rank.get(str(record.get("source")), -1),
            method_rank.get(str(record.get("attack")), -1),
            target_rank.get(str(record.get("target")), -1),
        )
    )
    output = []
    for record in evaluations:
        row = {
            "source": record.get("source"),
            "attack": record.get("attack"),
            "target": record.get("target"),
            "status": record.get("status"),
            "images": record.get("images"),
            "detections": record.get("detections"),
            "inference_seconds_total": format_metric(
                record.get("inference_seconds_total")
            ),
            "inference_seconds_mean": format_metric(
                record.get("inference_seconds_mean")
            ),
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "predictions_sha256": record.get("predictions_sha256"),
        }
        row.update(
            {
                metric: format_metric((record.get("metrics") or {}).get(metric))
                for metric in COCO_BBOX_METRICS
            }
        )
        output.append(row)
    return output


def _per_category_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    protocol_id: str = "coco_all_methods",
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    output = []
    for record in _all_metrics_order(
        records, registry, protocol_id, dataset_id
    ):
        row = {
            "source": record.get("source"),
            "attack": record.get("attack"),
            "target": record.get("target"),
        }
        row.update(
            {
                category: format_metric(
                    (record.get("per_category_ap") or {}).get(category)
                )
                for category in registry.dataset(dataset_id).classes
            }
        )
        output.append(row)
    return output


def _all_metrics_order(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    protocol_id: str = "coco_all_methods",
    dataset_id: str = "coco",
) -> List[Mapping[str, Any]]:
    source_rank = {value: index for index, value in enumerate(registry.source_ids())}
    target_rank = {value: index for index, value in enumerate(registry.target_ids())}
    method_rank = {
        value: index
        for index, value in enumerate(
            registry.protocols[protocol_id]["methods"]
        )
    }
    evaluations = _evaluation_records(records, dataset_id)
    return sorted(
        evaluations,
        key=lambda record: (
            0 if record.get("attack") == "clean" else 1,
            source_rank.get(str(record.get("source")), -1),
            method_rank.get(str(record.get("attack")), -1),
            target_rank.get(str(record.get("target")), -1),
        ),
    )


def _quality_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    protocol_id: str = "coco_all_methods",
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    seen = set()
    rows = []
    for record in _all_metrics_order(
        records, registry, protocol_id, dataset_id
    ):
        if record.get("attack") == "clean":
            continue
        key = (str(record.get("source")), str(record.get("attack")))
        if key in seen:
            continue
        seen.add(key)
        quality = record.get("attack_quality_mean") or {}
        row = {
            "source": key[0],
            "attack": key[1],
            "gradient_evaluations_per_image": record.get(
                "gradient_evaluations_per_image"
            ),
            "actual_gradient_evaluations_total": record.get(
                "attack_actual_gradient_evaluations_total"
            ),
            "actual_gradient_evaluations_mean": format_metric(
                record.get("attack_actual_gradient_evaluations_mean")
            ),
            "pixel_identical_images": record.get(
                "attack_pixel_identical_images"
            ),
            "auxiliary_forward_mean": format_metric(
                (record.get("auxiliary_passes") or {}).get("forward_mean")
            ),
            "auxiliary_backward_mean": format_metric(
                (record.get("auxiliary_passes") or {}).get("backward_mean")
            ),
            "attack_runtime_seconds_mean": format_metric(
                record.get("attack_runtime_seconds_mean")
            ),
            "attack_peak_cuda_memory_mb": format_metric(
                record.get("attack_peak_cuda_memory_mb")
            ),
        }
        row.update(
            {
                metric: format_metric(quality.get(metric))
                for metric in ATTACK_QUALITY_METRICS
            }
        )
        rows.append(row)
    return rows


def _checkpoint_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    clean_by_target = {
        str(record.get("target")): record
        for record in _evaluation_records(records, dataset_id)
        if record.get("attack") == "clean"
    }
    rows = []
    for position, target_id in enumerate(registry.target_ids(), start=1):
        model = registry.model(target_id)
        record = clean_by_target[target_id]
        rows.append(
            {
                "position": position,
                "target": target_id,
                "display_name": model.display_name,
                "table_code": model.table_code,
                "backbone": model.backbone,
                "checkpoint_file": checkpoint_path(
                    model, registry.dataset(dataset_id)
                ).name,
                "checkpoint_sha256": record.get("checkpoint_sha256"),
            }
        )
    return rows


def _retention_rows(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    run_dir: Path,
    protocol_id: str = "coco_all_methods",
    dataset_id: str = "coco",
) -> List[Dict[str, Any]]:
    first_by_group: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for record in _all_metrics_order(
        records, registry, protocol_id, dataset_id
    ):
        if record.get("attack") != "clean":
            first_by_group.setdefault(
                (str(record.get("source")), str(record.get("attack"))), record
            )
    rows = []
    for source_id in registry.source_ids():
        for attack_id in registry.protocols[protocol_id]["methods"]:
            key = (source_id, attack_id)
            if key not in first_by_group:
                continue
            record = first_by_group[key]
            attack_dir = _inside(
                Path(str(record.get("adversarial_run", ""))),
                run_dir / "attacks",
            )
            retention_path = attack_dir / "retention_summary.json"
            retention = _read_json(retention_path)
            rows.append(
                {
                    "source": source_id,
                    "source_name": registry.model(source_id).display_name,
                    "attack": attack_id,
                    "attack_name": registry.attack(attack_id).display_name,
                    "generated_images": retention.get("generated_images"),
                    "evaluated_targets": (retention.get("gate") or {}).get(
                        "target_count"
                    ),
                    "retained_images": retention.get("retained_images"),
                    "pruned_images": retention.get("images_to_prune"),
                    "bytes_before": retention.get("bytes_before"),
                    "bytes_after": retention.get("bytes_after"),
                    "bytes_pruned": retention.get("bytes_pruned"),
                    "retained_image_ids_sha256": retention.get(
                        "retained_image_ids_sha256"
                    ),
                    "source_checkpoint_sha256": (
                        retention.get("gate") or {}
                    ).get("source_checkpoint_sha256"),
                    "all_images_manifest_sha256": retention.get(
                        "all_images_manifest_sha256"
                    ),
                    "retained_manifest_sha256": retention.get(
                        "retained_manifest_sha256"
                    ),
                    "pruned_manifest_sha256": retention.get(
                        "pruned_manifest_sha256"
                    ),
                    "retained_annotation_sha256": retention.get(
                        "retained_annotation_sha256"
                    ),
                    "retention_summary_sha256": file_digest(retention_path),
                }
            )
    if len(rows) != 53:
        raise RuntimeError("Retention audit report does not contain 53 groups")
    return rows


def _structural_skip_rows(
    plan: Mapping[str, Any],
    registry: Registry,
    protocol_id: str = "coco_all_methods",
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for job in plan.get("jobs", []):
        if job.get("status") != "skipped":
            continue
        key = (str(job["source"]), str(job["attack"]))
        row = grouped.setdefault(
            key,
            {
                "source": key[0],
                "source_name": registry.model(key[0]).display_name,
                "attack": key[1],
                "attack_name": registry.attack(key[1]).display_name,
                "compatibility": job.get("compatibility"),
                "reason": job.get("reason"),
                "target_cells": 0,
            },
        )
        if (
            row["reason"] != job.get("reason")
            or row["compatibility"] != job.get("compatibility")
        ):
            raise RuntimeError("A structural-skip group has inconsistent reasons")
        row["target_cells"] += 1
    source_rank = {value: index for index, value in enumerate(registry.source_ids())}
    method_rank = {
        value: index
        for index, value in enumerate(
            registry.protocols[protocol_id]["methods"]
        )
    }
    rows = sorted(
        grouped.values(),
        key=lambda row: (
            source_rank[row["source"]],
            method_rank[row["attack"]],
        ),
    )
    if len(rows) != 13 or any(int(row["target_cells"]) != 16 for row in rows):
        raise RuntimeError("Structural-skip audit is not 13 groups / 208 cells")
    return rows


def write_formal_all_methods_audits(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_dir: Path,
    protocol_id: str,
) -> Dict[str, Path]:
    """Validate and write dataset-generic formal all-method audit tables."""
    acceptance = validate_formal_all_methods_bundle(
        records,
        registry,
        summary=summary,
        plan=plan,
        provenance=provenance,
        run_dir=run_dir,
        protocol_id=protocol_id,
    )
    dataset_id = str(acceptance["dataset"])
    output_dir.mkdir(parents=True, exist_ok=True)
    method_ids = list(plan.get("methods", []))
    aggregate = _aggregate_rows(
        records, registry, method_ids, dataset_id
    )
    aggregate_fields = [
        "method",
        "method_name",
        "metric",
        "supported_sources",
        "expected_supported_cells",
        "observed_supported_cells",
        "common_two_stage_mean",
        "common_two_stage_bb_mean",
        "six_source_mean",
        "six_source_bb_mean",
        "supported_source_mean",
        "supported_source_bb_mean",
        "supported_whitebox_mean",
    ]
    paths = {
        "aggregate_csv": output_dir
        / "{}_all_methods_aggregate.csv".format(dataset_id),
        "all_metrics_csv": output_dir
        / "{}_all_metrics.csv".format(dataset_id),
        "per_category_csv": output_dir
        / "{}_per_category_ap.csv".format(dataset_id),
        "quality_runtime_csv": output_dir
        / "{}_attack_quality_runtime.csv".format(dataset_id),
        "checkpoint_manifest_csv": output_dir
        / "{}_checkpoint_manifest.csv".format(dataset_id),
        "retention_audit_csv": output_dir
        / "{}_retention_audit.csv".format(dataset_id),
        "structural_skips_csv": output_dir
        / "{}_structural_skips.csv".format(dataset_id),
        "acceptance_json": output_dir
        / "formal_all_methods_acceptance.json",
    }
    _csv_write(
        paths["aggregate_csv"],
        aggregate_fields,
        (
            {
                **row,
                **{
                    field: format_metric(row.get(field))
                    for field in aggregate_fields
                    if field.endswith("mean")
                },
            }
            for row in aggregate
        ),
    )
    _csv_write(
        paths["all_metrics_csv"],
        [
            "source",
            "attack",
            "target",
            "status",
            "images",
            "detections",
            *COCO_BBOX_METRICS,
            "inference_seconds_total",
            "inference_seconds_mean",
            "checkpoint_sha256",
            "predictions_sha256",
        ],
        _all_metrics_rows(
            records, registry, protocol_id, dataset_id
        ),
    )
    _csv_write(
        paths["per_category_csv"],
        [
            "source",
            "attack",
            "target",
            *registry.dataset(dataset_id).classes,
        ],
        _per_category_rows(
            records, registry, protocol_id, dataset_id
        ),
    )
    quality_fields = [
        "source",
        "attack",
        *ATTACK_QUALITY_METRICS,
        "gradient_evaluations_per_image",
        "actual_gradient_evaluations_total",
        "actual_gradient_evaluations_mean",
        "pixel_identical_images",
        "auxiliary_forward_mean",
        "auxiliary_backward_mean",
        "attack_runtime_seconds_mean",
        "attack_peak_cuda_memory_mb",
    ]
    _csv_write(
        paths["quality_runtime_csv"],
        quality_fields,
        _quality_rows(records, registry, protocol_id, dataset_id),
    )
    _csv_write(
        paths["checkpoint_manifest_csv"],
        [
            "position",
            "target",
            "display_name",
            "table_code",
            "backbone",
            "checkpoint_file",
            "checkpoint_sha256",
        ],
        _checkpoint_rows(records, registry, dataset_id),
    )
    _csv_write(
        paths["retention_audit_csv"],
        [
            "source",
            "source_name",
            "attack",
            "attack_name",
            "generated_images",
            "evaluated_targets",
            "retained_images",
            "pruned_images",
            "bytes_before",
            "bytes_after",
            "bytes_pruned",
            "retained_image_ids_sha256",
            "source_checkpoint_sha256",
            "all_images_manifest_sha256",
            "retained_manifest_sha256",
            "pruned_manifest_sha256",
            "retained_annotation_sha256",
            "retention_summary_sha256",
        ],
        _retention_rows(
            records,
            registry,
            run_dir,
            protocol_id,
            dataset_id,
        ),
    )
    _csv_write(
        paths["structural_skips_csv"],
        [
            "source",
            "source_name",
            "attack",
            "attack_name",
            "compatibility",
            "reason",
            "target_cells",
        ],
        _structural_skip_rows(plan, registry, protocol_id),
    )
    atomic_json(paths["acceptance_json"], acceptance)
    return paths


def _aggregate_tex(rows: Sequence[Mapping[str, Any]]) -> str:
    ap50_rows = [row for row in rows if row["metric"] == "bbox_mAP_50"]
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{Unified eleven-method COCO transfer summary at AP@.50. The Faster/Mask two-source mean is shown only when both sources are native; exact NAA has no such value after the fidelity revision. Six-source is reported only for methods defined on all six sources. Supported-source means include their denominator and are not like-for-like when denominators differ.}",
        r"\label{tab:coco_all_methods_aggregate}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l c c c c c c c}",
        r"\toprule",
        r"Method & Sources & Cells & Common-2 & Common-2 BB & Six-source & Six-source BB & Supported BB \\",
        r"\midrule",
    ]
    for row in ap50_rows:
        def cell(key: str) -> str:
            value = row.get(key)
            return format_metric(value, missing="--")

        lines.append(
            " & ".join(
                [
                    _escape_latex(row["method_name"]),
                    str(row["supported_sources"]),
                    str(row["observed_supported_cells"]),
                    cell("common_two_stage_mean"),
                    cell("common_two_stage_bb_mean"),
                    cell("six_source_mean"),
                    cell("six_source_bb_mean"),
                    cell("supported_source_bb_mean"),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def _complete_tex() -> str:
    lines = [
        "% Generated from records.json; do not enter values by hand.",
        r"\input{coco_all_methods_aggregate.tex}",
    ]
    lines.extend(
        r"\input{transfer_" + metric + ".tex}" for metric in COCO_BBOX_METRICS
    )
    lines.extend(
        [
            r"\input{analysis.tex}",
            r"\input{per_category_ap.tex}",
            r"\input{target_family.tex}",
            r"\input{perceptual_quality.tex}",
            r"\input{runtime_efficiency.tex}",
            r"\input{attack_success_transfer_rate.tex}",
            r"\input{figures/figures.tex}",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown(
    *,
    english: bool,
    registry: Registry,
    summary: Mapping[str, Any],
    provenance: Mapping[str, Any],
    aggregate: Sequence[Mapping[str, Any]],
    method_ids: Sequence[str],
) -> str:
    status = str(summary.get("status", "unknown"))
    scheduler = provenance.get("execution_scheduler") or {}
    versions = provenance.get("packages") or {}
    version_text = ", ".join(
        "{} {}".format(package, versions.get(package, "unknown"))
        for package in PINNED_REPORT_VERSIONS
    )
    ap50 = {row["method"]: row for row in aggregate if row["metric"] == "bbox_mAP_50"}
    if english:
        title = "# Formal COCO all-method transfer results"
        switch = "[简体中文](coco_all_methods.zh-CN.md) | **English**"
        intro = (
            "This report is generated from `records.json`. The formal protocol uses all "
            "5,000 COCO val2017 images, six fixed sources, eleven maintained methods and "
            "sixteen fixed targets. COCO val2017 is not used for tuning or method selection."
        )
        status_line = "Run status: `{}`.".format(status)
        provenance_title = "## Execution provenance"
        provenance_items = [
            "Git commit: `{}`; the worktree was clean when the immutable run started.".format(
                provenance.get("git_commit", "unknown")
            ),
            "Python `{}`; {}.".format(
                provenance.get("python", "unknown"), version_text
            ),
            "No hostname, private absolute path, hardware identifier or credential is published in this report.",
            "Execution scheduler: `{}` with {} independent worker(s), fixed canonical round-robin assignment, a clean-stage barrier, and complete source-method groups kept on one worker.".format(
                scheduler.get("mode", "unknown"),
                scheduler.get("worker_count", "unknown"),
            ),
        ]
        fidelity_title = "## Method fidelity provenance"
        fidelity_headings = "| Method | Fidelity class | Evidence |"
        fidelity_separator = "|---|---|---|"
        paper_label = "paper"
        repository_label = "repository"
        paper_only_note = "no verifiable official repository"
        project_note = "project specification"
        results_title = "## AP50 aggregates"
        headings = "| Method | Native sources | Evaluated cells | Common-2 AP50 | Common-2 BB AP50 | Six-source BB AP50 | Supported-source BB AP50 |"
        separator = "|---|---:|---:|---:|---:|---:|---:|"
        notes = [
            "The earlier `Common-2` label belongs to the pre-refresh evidence. Exact NAA is Faster-only, so no current two-source denominator is shared by all eleven methods.",
            "`Six-source` is shown only for the eight methods defined on every source. `--` is an audited structural skip, never zero and never an execution failure.",
            "`Supported-source` uses each method's native source set and therefore must not be treated as a same-denominator ranking when the source count differs.",
            "Each source-method payload was first generated and evaluated on all 5,000 images and all sixteen targets. Only after the 16/16 fail-closed gate passed was the pre-frozen 500-image evidence subset retained; the other 4,500 PNGs were removed with hashes and disposition manifests preserved.",
            "Prediction arrays are stored as verified lossless `predictions.json.gz` archives. The uncompressed SHA-256 and byte count permit exact reconstruction auditing.",
        ]
        artifacts_title = "## Reproducibility artifacts"
        artifacts = [
            "`../../plan.json` and `../../execution_state.json`: resolved protocol and live state",
            "`../../execution_assignments.json`: fixed worker assignment and clean-stage barrier",
            "`../../records.json` and `../../summary.json`: machine-readable source of truth and completion audit",
            "`coco_all_metrics.csv`: all twelve COCO-style bbox metrics for every clean/attacked evaluation",
            "`coco_per_category_ap.csv`: all 80 category AP values",
            "`coco_attack_quality_runtime.csv`: nine image-quality metrics and compute/runtime accounting",
            "`coco_retention_audit.csv`: the 53 post-16/16 validation and 500/4,500 disposition audits",
            "`coco_checkpoint_manifest.csv`: canonical 16-target checkpoint SHA-256 manifest",
            "`coco_structural_skips.csv`: the 13 audited source-method skip groups (208 target cells)",
            "`../../attacks/.../all_images_manifest.jsonl.gz`: all 5,000 per-image quality/runtime/diagnostic and disposition rows for each payload",
            "`coco_all_methods_aggregate.csv` and `.tex`: explicitly denominated method aggregates",
            "`coco_all_methods_complete.tex`: one-file LaTeX include bundle",
            "`../../artifact_manifest.json` and `.sha256`: final whole-run artifact integrity index",
        ]
        command_title = "## Canonical command"
    else:
        title = "# 正式 COCO 全方法迁移结果"
        switch = "**简体中文** | [English](coco_all_methods.md)"
        intro = (
            "本报告由 `records.json` 自动生成。正式协议覆盖全部 5,000 张 COCO "
            "val2017 图像、六个固定源、十一种维护方法和十六个固定目标；COCO "
            "val2017 不参与调参或方法选择。"
        )
        status_line = "运行状态：`{}`。".format(status)
        provenance_title = "## 执行溯源"
        provenance_items = [
            "Git 提交：`{}`；不可覆盖运行启动时工作树干净。".format(
                provenance.get("git_commit", "unknown")
            ),
            "Python `{}`；{}。".format(
                provenance.get("python", "unknown"), version_text
            ),
            "本报告不公开主机名、私有绝对路径、硬件标识或凭据。",
            "执行调度：`{}`，{} 个独立 worker，采用固定规范轮转、clean 阶段屏障，并保证每个完整源—方法组只归一名 worker。".format(
                scheduler.get("mode", "unknown"),
                scheduler.get("worker_count", "unknown"),
            ),
        ]
        fidelity_title = "## 方法忠实性溯源"
        fidelity_headings = "| 方法 | 忠实性类别 | 证据 |"
        fidelity_separator = "|---|---|---|"
        paper_label = "论文"
        repository_label = "代码仓库"
        paper_only_note = "无可核验的官方代码仓库"
        project_note = "项目规范"
        results_title = "## AP50 汇总"
        headings = "| 方法 | 原生源数 | 已评估单元 | Common-2 AP50 | Common-2 黑盒 AP50 | 六源黑盒 AP50 | 支持源黑盒 AP50 |"
        separator = "|---|---:|---:|---:|---:|---:|---:|"
        notes = [
            "早期 `Common-2` 只属于修正前证据；精确 NAA 仅支持 Faster，因此当前十一种方法不存在共同的两源分母。",
            "`六源` 只对在六个源上均有定义的八种方法报告；`--` 表示经审计的结构性不适用，既不是零值，也不是执行失败。",
            "`支持源` 使用各方法自身的原生源集合；源数不同时，不能把它当作同分母排名。",
            "每个源—方法组合先在全部 5,000 张图像上生成，并完成十六目标评估；仅在 16/16 失败关闭闸门通过后，才保留预先冻结的同一 500 张证据图像，其余 4,500 张 PNG 删除但完整保留哈希与处置清单。",
            "预测数组保存为已验证的无损 `predictions.json.gz`；未压缩 SHA-256 与字节数支持精确重建审计。",
        ]
        artifacts_title = "## 可复现产物"
        artifacts = [
            "`../../plan.json` 与 `../../execution_state.json`：解析后的协议和运行状态",
            "`../../execution_assignments.json`：冻结的 worker 分配与 clean 阶段屏障",
            "`../../records.json` 与 `../../summary.json`：机器可读真源和完成审计",
            "`coco_all_metrics.csv`：每次 clean/攻击评估的全部 12 项 COCO-style bbox 指标",
            "`coco_per_category_ap.csv`：全部 80 类 AP",
            "`coco_attack_quality_runtime.csv`：九项图像质量指标及计算量/耗时记录",
            "`coco_retention_audit.csv`：53 组完成 16/16 验收后的 500/4,500 处置审计",
            "`coco_checkpoint_manifest.csv`：规范顺序的 16 个目标权重 SHA-256 清单",
            "`coco_structural_skips.csv`：13 个经审计的源—方法跳过组（208 个目标单元）",
            "`../../attacks/.../all_images_manifest.jsonl.gz`：每份载荷全部 5,000 条逐图质量/耗时/诊断及处置记录",
            "`coco_all_methods_aggregate.csv` 与 `.tex`：显式注明分母的方法汇总",
            "`coco_all_methods_complete.tex`：一次性 LaTeX 输入文件",
            "`../../artifact_manifest.json` 与 `.sha256`：整次运行最终产物完整性索引",
        ]
        command_title = "## 标准命令"

    lines = [
        title,
        "",
        switch,
        "",
        intro,
        "",
        status_line,
        "",
        provenance_title,
        "",
        *("- " + item for item in provenance_items),
        "",
        fidelity_title,
        "",
        fidelity_headings,
        fidelity_separator,
    ]
    for method in method_ids:
        spec = registry.attack(method)
        paper = spec.metadata.get("paper")
        repository = spec.metadata.get("repository")
        audit_commit = spec.metadata.get("audit_commit")
        if paper and repository and audit_commit:
            evidence = "{}={}; {}={} @ {}".format(
                paper_label, paper, repository_label, repository, audit_commit
            )
        elif paper:
            evidence = "{}={}; {}".format(paper_label, paper, paper_only_note)
        else:
            evidence = project_note
        lines.append(
            "| {} | `{}` | {} |".format(
                spec.display_name,
                spec.metadata.get("fidelity_status", "unknown"),
                evidence,
            )
        )
    lines.extend(["", results_title, "", headings, separator])
    for method in method_ids:
        row = ap50.get(method, {})
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                registry.attack(method).display_name,
                row.get("supported_sources", 0),
                row.get("observed_supported_cells", 0),
                format_metric(row.get("common_two_stage_mean"), missing="NR"),
                format_metric(row.get("common_two_stage_bb_mean"), missing="NR"),
                format_metric(row.get("six_source_bb_mean"), missing="--"),
                format_metric(row.get("supported_source_bb_mean"), missing="NR"),
            )
        )
    lines.extend(["", *("- " + note for note in notes), "", command_title, "", "```powershell", str(provenance.get("canonical_command", "")), "```", "", artifacts_title, "", *("- " + item for item in artifacts), ""])
    return "\n".join(lines)


def write_all_methods_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    provenance: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Path]:
    """Validate and publish the unified COCO result bundle from records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = run_dir.resolve()
    _validate_complete_records(
        records,
        registry,
        summary,
        plan,
        provenance,
        run_dir,
        protocol_id="coco_all_methods",
    )
    method_ids = list(plan.get("methods", []))
    aggregate = _aggregate_rows(records, registry, method_ids)
    all_metrics = _all_metrics_rows(records, registry)
    categories = _per_category_rows(records, registry)
    quality = _quality_rows(records, registry)
    is_complete = summary.get("status") == "complete"
    checkpoints = _checkpoint_rows(records, registry) if is_complete else []
    retention = (
        _retention_rows(records, registry, run_dir) if is_complete else []
    )
    structural_skips = _structural_skip_rows(plan, registry)

    aggregate_path = output_dir / "coco_all_methods_aggregate.csv"
    aggregate_fields = [
        "method",
        "method_name",
        "metric",
        "supported_sources",
        "expected_supported_cells",
        "observed_supported_cells",
        "common_two_stage_mean",
        "common_two_stage_bb_mean",
        "six_source_mean",
        "six_source_bb_mean",
        "supported_source_mean",
        "supported_source_bb_mean",
        "supported_whitebox_mean",
    ]
    _csv_write(
        aggregate_path,
        aggregate_fields,
        (
            {
                **row,
                **{
                    field: format_metric(row.get(field))
                    for field in aggregate_fields
                    if field.endswith("mean")
                },
            }
            for row in aggregate
        ),
    )
    all_metrics_path = output_dir / "coco_all_metrics.csv"
    _csv_write(
        all_metrics_path,
        [
            "source",
            "attack",
            "target",
            "status",
            "images",
            "detections",
            *COCO_BBOX_METRICS,
            "inference_seconds_total",
            "inference_seconds_mean",
            "checkpoint_sha256",
            "predictions_sha256",
        ],
        all_metrics,
    )
    category_path = output_dir / "coco_per_category_ap.csv"
    _csv_write(
        category_path,
        ["source", "attack", "target", *registry.dataset("coco").classes],
        categories,
    )
    quality_path = output_dir / "coco_attack_quality_runtime.csv"
    _csv_write(
        quality_path,
        [
            "source",
            "attack",
            *ATTACK_QUALITY_METRICS,
            "gradient_evaluations_per_image",
            "actual_gradient_evaluations_total",
            "actual_gradient_evaluations_mean",
            "pixel_identical_images",
            "auxiliary_forward_mean",
            "auxiliary_backward_mean",
            "attack_runtime_seconds_mean",
            "attack_peak_cuda_memory_mb",
        ],
        quality,
    )
    checkpoint_path_output = output_dir / "coco_checkpoint_manifest.csv"
    checkpoint_fields = [
        "position",
        "target",
        "display_name",
        "table_code",
        "backbone",
        "checkpoint_file",
        "checkpoint_sha256",
    ]
    _csv_write(checkpoint_path_output, checkpoint_fields, checkpoints)
    retention_path = output_dir / "coco_retention_audit.csv"
    retention_fields = [
        "source",
        "source_name",
        "attack",
        "attack_name",
        "generated_images",
        "evaluated_targets",
        "retained_images",
        "pruned_images",
        "bytes_before",
        "bytes_after",
        "bytes_pruned",
        "retained_image_ids_sha256",
        "source_checkpoint_sha256",
        "all_images_manifest_sha256",
        "retained_manifest_sha256",
        "pruned_manifest_sha256",
        "retained_annotation_sha256",
        "retention_summary_sha256",
    ]
    _csv_write(retention_path, retention_fields, retention)
    skips_path = output_dir / "coco_structural_skips.csv"
    skip_fields = [
        "source",
        "source_name",
        "attack",
        "attack_name",
        "compatibility",
        "reason",
        "target_cells",
    ]
    _csv_write(skips_path, skip_fields, structural_skips)
    aggregate_tex = output_dir / "coco_all_methods_aggregate.tex"
    complete_tex = output_dir / "coco_all_methods_complete.tex"
    english = output_dir / "coco_all_methods.md"
    chinese = output_dir / "coco_all_methods.zh-CN.md"
    _atomic_text(aggregate_tex, _aggregate_tex(aggregate))
    _atomic_text(complete_tex, _complete_tex())
    _atomic_text(
        english,
        _markdown(
            english=True,
            registry=registry,
            summary=summary,
            provenance=provenance,
            aggregate=aggregate,
            method_ids=method_ids,
        ),
    )
    _atomic_text(
        chinese,
        _markdown(
            english=False,
            registry=registry,
            summary=summary,
            provenance=provenance,
            aggregate=aggregate,
            method_ids=method_ids,
        ),
    )
    return {
        "aggregate_csv": aggregate_path,
        "aggregate_tex": aggregate_tex,
        "all_metrics_csv": all_metrics_path,
        "per_category_csv": category_path,
        "quality_runtime_csv": quality_path,
        "checkpoint_manifest_csv": checkpoint_path_output,
        "retention_audit_csv": retention_path,
        "structural_skips_csv": skips_path,
        "complete_tex": complete_tex,
        "english_markdown": english,
        "chinese_markdown": chinese,
    }


def write_artifact_manifest(run_dir: Path) -> Tuple[Path, Path]:
    """Hash every final run artifact without creating a circular self-entry."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "artifact_manifest.json"
    sidecar_path = run_dir / "artifact_manifest.sha256"
    if manifest_path.exists() or sidecar_path.exists():
        raise FileExistsError("Refusing to overwrite the artifact manifest")
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("Formal run contains a symlink: {}".format(path))
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        if relative in {manifest_path.name, sidecar_path.name}:
            continue
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
        )
    payload = {
        "schema_version": 1,
        "algorithm": "sha256",
        "root": ".",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(int(item["bytes"]) for item in files),
    }
    atomic_json(manifest_path, payload)
    digest = file_digest(manifest_path)
    _atomic_text(sidecar_path, "{}  {}".format(digest, manifest_path.name))
    if file_digest(manifest_path) != digest:
        raise RuntimeError("Artifact manifest changed after sidecar publication")
    return manifest_path, sidecar_path
