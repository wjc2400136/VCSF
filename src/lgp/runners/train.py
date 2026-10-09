from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from filelock import FileLock
from mmdet.apis import inference_detector, init_detector
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..modeling import (
    checkpoint_path,
    ensure_checkpoint,
    finite_state_dict_audit,
)
from ..paths import project_root
from ..registry import DatasetSpec, ModelSpec, Registry
from ..runtime_config import (
    build_runtime_config,
    register_framework,
    resume_phase_switch_boundary,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relative(path: Path) -> str:
    root = project_root()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def _git_provenance() -> Dict[str, Any]:
    root = project_root()

    def run(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=str(root),
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
        "status": status,
    }


def _checkpoint_payload(path: Path) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Checkpoint root is not a mapping: {}".format(path))
    state_dict = payload.get("state_dict", payload.get("model"))
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise RuntimeError("Checkpoint has no recognized state dictionary: {}".format(path))
    if not all(isinstance(key, str) for key in state_dict):
        raise RuntimeError("Checkpoint state dictionary contains a non-string key")
    if not any(torch.is_tensor(value) for value in state_dict.values()):
        raise RuntimeError("Checkpoint state dictionary contains no tensors")
    meta = payload.get("meta", {})
    return payload, meta if isinstance(meta, Mapping) else {}


def _load_json_mapping(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("{} is required: {}".format(label, path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("{} is unreadable: {}".format(label, path)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("{} root must be a mapping: {}".format(label, path))
    return payload


def _backup_superseded_checkpoint(
    destination: Path,
    work_dir: Path,
    dataset_id: str,
    model_id: str,
) -> Dict[str, Any]:
    if not destination.is_file():
        raise FileNotFoundError(destination)
    old_digest = file_digest(destination)
    manifest_path = destination.parent / "manifest.json"
    manifest = _load_json_mapping(
        manifest_path, "{} checkpoint manifest".format(dataset_id)
    )
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise RuntimeError("Checkpoint manifest has no checkpoint list")
    matching = [
        record
        for record in records
        if isinstance(record, dict) and record.get("model") == model_id
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "Cannot preserve unique superseded manifest record for {}".format(
                model_id
            )
        )
    old_record = matching[0]
    if (
        Path(str(old_record.get("path", ""))).name != destination.name
        or int(old_record.get("bytes", -1)) != destination.stat().st_size
        or str(old_record.get("sha256", "")).lower() != old_digest.lower()
    ):
        raise RuntimeError(
            "Superseded checkpoint differs from its manifest record"
        )
    evidence_dir = work_dir / "superseded_checkpoint"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    backup = evidence_dir / destination.name
    try:
        os.link(str(destination), str(backup))
        method = "hardlink"
    except OSError:
        shutil.copy2(str(destination), str(backup))
        method = "copy"
    if (
        not backup.is_file()
        or backup.stat().st_size != destination.stat().st_size
        or file_digest(backup) != old_digest
    ):
        raise RuntimeError("Superseded checkpoint backup verification failed")

    manifest_evidence_path = evidence_dir / "manifest.json"
    shutil.copy2(str(manifest_path), str(manifest_evidence_path))
    if (
        not manifest_evidence_path.is_file()
        or file_digest(manifest_evidence_path) != file_digest(manifest_path)
    ):
        raise RuntimeError("Superseded checkpoint manifest backup verification failed")
    manifest_record_path = evidence_dir / "manifest_record.json"
    atomic_json(manifest_record_path, old_record)
    return {
        "status": "preserved",
        "path": _relative(backup),
        "bytes": backup.stat().st_size,
        "sha256": old_digest,
        "method": method,
        "manifest": _relative(manifest_evidence_path),
        "manifest_sha256": file_digest(manifest_evidence_path),
        "manifest_record": _relative(manifest_record_path),
        "manifest_record_sha256": file_digest(manifest_record_path),
    }


def _verified_source_checkpoint(
    registry: Registry, model: ModelSpec, download_weights: bool
) -> Tuple[Path, str]:
    coco = registry.dataset("coco")
    source = ensure_checkpoint(model, coco, download=download_weights).resolve()
    manifest_path = registry.root / "checkpoints" / "coco" / "manifest.json"
    manifest = _load_json_mapping(manifest_path, "COCO checkpoint manifest")
    if int(manifest.get("schema_version", -1)) != 1:
        raise RuntimeError("Unsupported COCO checkpoint manifest schema")
    records = manifest.get("checkpoints")
    if not isinstance(records, list):
        raise RuntimeError("COCO checkpoint manifest has no checkpoint list")
    matching = [
        record
        for record in records
        if isinstance(record, dict) and record.get("model") == model.id
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "COCO manifest must contain exactly one record for {}".format(model.id)
        )
    record = matching[0]
    if (
        Path(str(record.get("path", ""))).name != source.name
        or int(record.get("bytes", -1)) != source.stat().st_size
    ):
        raise RuntimeError(
            "COCO source checkpoint path/size differs from its manifest for {}".format(
                model.id
            )
        )
    digest = file_digest(source)
    if str(record.get("sha256", "")).lower() != digest.lower():
        raise RuntimeError(
            "COCO source checkpoint SHA-256 differs from its manifest for {}".format(
                model.id
            )
        )
    return source, digest


def _expected_warm_start_keys(model_id: str) -> Tuple[set, set]:
    pair = lambda stem: {stem + ".weight", stem + ".bias"}
    shape: set
    unexpected: set = set()
    if model_id in {"faster_rcnn_r50", "mask_rcnn_swin_t"}:
        shape = pair("roi_head.bbox_head.fc_cls") | pair(
            "roi_head.bbox_head.fc_reg"
        )
    elif model_id == "cascade_rcnn_r50":
        shape = set().union(
            *[
                pair("roi_head.bbox_head.{}.fc_cls".format(index))
                for index in range(3)
            ]
        )
    elif model_id == "yolov3_d53":
        shape = set().union(
            *[
                pair("bbox_head.convs_pred.{}".format(index))
                for index in range(3)
            ]
        )
    elif model_id == "yolov5_s":
        shape = set().union(
            *[
                pair("bbox_head.head_module.convs_pred.{}".format(index))
                for index in range(3)
            ]
        )
    elif model_id == "yolox_s":
        shape = set().union(
            *[
                pair(
                    "bbox_head.head_module.multi_level_conv_cls.{}".format(index)
                )
                for index in range(3)
            ]
        )
    elif model_id == "yolov8_s":
        shape = set().union(
            *[
                pair("bbox_head.head_module.cls_preds.{}.2".format(index))
                for index in range(3)
            ]
        )
    elif model_id == "retinanet_free_anchor_r50":
        shape = pair("bbox_head.retina_cls")
    elif model_id == "reppoints_r50":
        shape = pair("bbox_head.reppoints_cls_out")
    elif model_id == "vfnet_r50":
        shape = pair("bbox_head.vfnet_cls")
    elif model_id == "tood_r50":
        shape = pair("bbox_head.tood_cls")
    elif model_id == "rtmdet_s":
        shape = set().union(
            *[
                pair("bbox_head.head_module.rtm_cls.{}".format(index))
                for index in range(3)
            ]
        )
    elif model_id == "sparse_rcnn_r50":
        shape = set().union(
            *[
                pair("roi_head.bbox_head.{}.fc_cls".format(index))
                for index in range(6)
            ]
        )
    elif model_id == "detr_r50":
        shape = pair("bbox_head.fc_cls")
        # The old official release serialized normalization buffers that are
        # non-persistent in the pinned MMDetection 3.0 model definition.
        unexpected = {"data_preprocessor.mean", "data_preprocessor.std"}
    elif model_id == "deformable_detr_r50":
        shape = set().union(
            *[
                pair("bbox_head.cls_branches.{}".format(index))
                for index in range(7)
            ]
        )
        unexpected = {"data_preprocessor.mean", "data_preprocessor.std"}
    elif model_id == "dino_r50":
        shape = set().union(
            *[
                pair("bbox_head.cls_branches.{}".format(index))
                for index in range(7)
            ]
        )
        shape.add("dn_query_generator.label_embedding.weight")
    else:
        raise KeyError("No warm-start audit policy for {}".format(model_id))
    if model_id == "mask_rcnn_swin_t":
        unexpected = set().union(
            *[
                pair("roi_head.mask_head.convs.{}.conv".format(index))
                for index in range(4)
            ],
            pair("roi_head.mask_head.upsample"),
            pair("roi_head.mask_head.conv_logits"),
        )
    return shape, unexpected


def _head_initialization_record(
    model: ModelSpec, dataset_id: str
) -> Optional[Dict[str, Any]]:
    declared = model.finetune_head_init.get(dataset_id)
    if declared is None:
        return None
    return {
        **dict(declared),
        "target_to_source": dict(declared["target_to_source"]),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _load_head_initialization_audit(
    path: Path,
    model: ModelSpec,
    dataset: DatasetSpec,
    source_dataset: DatasetSpec,
) -> Optional[Dict[str, Any]]:
    declared = _head_initialization_record(model, dataset.id)
    if declared is None:
        if path.exists():
            raise RuntimeError(
                "Unexpected semantic head initialization audit: {}".format(path)
            )
        return None
    audit = _load_json_mapping(path, "semantic head initialization audit")
    expected_mapping = [
        {
            "target_index": target_index,
            "target_class": target_class,
            "source_index": source_dataset.classes.index(
                str(declared["target_to_source"][target_class])
            ),
            "source_class": str(declared["target_to_source"][target_class]),
        }
        for target_index, target_class in enumerate(dataset.classes)
    ]
    required_values = {
        "schema_version": 1,
        "status": "verified",
        "policy": declared["policy"],
        "source_dataset": source_dataset.id,
        "target_dataset": dataset.id,
        "parameter_stem": declared["parameter_stem"],
        "source_class_count": source_dataset.num_classes,
        "target_class_count": dataset.num_classes,
        "class_mapping": expected_mapping,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatch_keys": [],
        "state_dict_exact_match": True,
    }
    for key, expected in required_values.items():
        if audit.get(key) != expected:
            raise RuntimeError(
                "Semantic head initialization audit {} mismatch: expected {!r}, "
                "got {!r}".format(key, expected, audit.get(key))
            )
    if audit.get("load_mode") not in {
        "fresh_semantic_class_copy",
        "target_shape_resume",
    }:
        raise RuntimeError("Semantic head initialization audit load_mode is invalid")
    if (
        int(audit.get("compatible_tensor_count", 0)) <= 0
        or int(audit.get("compatible_value_count", 0)) <= 0
        or audit.get("compatible_value_count") != audit.get("loaded_value_count")
        or not _is_sha256(audit.get("expected_state_sha256"))
        or audit.get("expected_state_sha256") != audit.get("loaded_state_sha256")
    ):
        raise RuntimeError(
            "Semantic head initialization audit state fingerprint is invalid"
        )
    expected_parameter = audit.get("expected_parameter_sha256")
    loaded_parameter = audit.get("loaded_parameter_sha256")
    parameter_keys = {
        str(declared["parameter_stem"]) + ".weight",
        str(declared["parameter_stem"]) + ".bias",
    }
    if (
        not isinstance(expected_parameter, Mapping)
        or not isinstance(loaded_parameter, Mapping)
        or set(expected_parameter) != parameter_keys
        or dict(expected_parameter) != dict(loaded_parameter)
        or not all(_is_sha256(value) for value in expected_parameter.values())
    ):
        raise RuntimeError(
            "Semantic head initialization audit parameter fingerprint is invalid"
        )
    return audit


def _audit_warm_start(
    detector: torch.nn.Module,
    source_checkpoint: Path,
    model_id: str,
) -> Dict[str, Any]:
    payload, _ = _checkpoint_payload(source_checkpoint)
    source_raw = payload.get("state_dict", payload.get("model"))
    source_finite = finite_state_dict_audit(source_raw)
    source = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source_raw.items()
    }
    target = detector.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    shape_mismatches = sorted(
        key
        for key in set(source).intersection(target)
        if torch.is_tensor(source[key])
        and torch.is_tensor(target[key])
        and tuple(source[key].shape) != tuple(target[key].shape)
    )
    expected_shapes, expected_unexpected = _expected_warm_start_keys(model_id)
    del payload
    gc.collect()
    if (
        missing
        or set(shape_mismatches) != expected_shapes
        or set(unexpected) != expected_unexpected
    ):
        raise RuntimeError(
            "COCO warm-start incompatibility for {}: missing={!r}; "
            "shape_mismatches={!r} (expected {!r}); unexpected={!r} "
            "(expected {!r})".format(
                model_id,
                missing,
                shape_mismatches,
                sorted(expected_shapes),
                unexpected,
                sorted(expected_unexpected),
            )
        )
    result = {
        "status": "verified",
        "missing_keys": [],
        "class_space_shape_mismatch_keys": shape_mismatches,
        "allowed_unexpected_keys": unexpected,
        "source_state_keys": len(source),
        "target_state_keys": len(target),
        "source_state_dict_finite": source_finite,
    }
    del source
    del target
    gc.collect()
    return result


def _audit_exact_resume(
    detector: torch.nn.Module, checkpoint: Path
) -> Dict[str, Any]:
    payload, _ = _checkpoint_payload(checkpoint)
    raw = payload.get("state_dict", payload.get("model"))
    finite_audit = finite_state_dict_audit(raw)
    source = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in raw.items()
    }
    target = detector.state_dict()
    missing = sorted(set(target).difference(source))
    unexpected = sorted(set(source).difference(target))
    shape_mismatches = sorted(
        key
        for key in set(source).intersection(target)
        if torch.is_tensor(source[key])
        and torch.is_tensor(target[key])
        and tuple(source[key].shape) != tuple(target[key].shape)
    )
    source_keys = len(source)
    target_keys = len(target)
    del payload
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Resume checkpoint is not an exact model state: missing={!r}; "
            "unexpected={!r}; shape_mismatches={!r}".format(
                missing, unexpected, shape_mismatches
            )
        )
    del source
    del target
    gc.collect()
    return {
        "status": "verified",
        "strict_key_and_shape_match": True,
        "checkpoint": _relative(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "source_state_keys": source_keys,
        "target_state_keys": target_keys,
        "state_dict_finite": finite_audit,
    }


def _last_checkpoint(work_dir: Path, expected_epoch: int) -> Path:
    pointer = work_dir / "last_checkpoint"
    candidates = []
    if pointer.is_file():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = work_dir / candidate
            candidates.append(candidate.resolve())
    candidates.append((work_dir / "epoch_{}.pth".format(expected_epoch)).resolve())
    candidates.extend(
        sorted(
            work_dir.glob("epoch_*.pth"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    )
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            payload, meta = _checkpoint_payload(resolved)
            epoch = meta.get("epoch")
            del payload
            if epoch is not None and int(epoch) != expected_epoch:
                continue
            return resolved
    raise RuntimeError(
        "Training did not produce a final epoch-{} checkpoint in {}".format(
            expected_epoch, work_dir
        )
    )


def _prepare_published_checkpoint(source: Path, destination: Path) -> Path:
    """Write an independent model-only checkpoint beside its final path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload, meta = _checkpoint_payload(source)
    state_dict = payload.get("state_dict", payload.get("model"))
    finite_state_dict_audit(state_dict)
    temporary_name: Optional[str] = None
    prepared: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".{}.".format(destination.name),
            suffix=".part",
            dir=str(destination.parent),
            delete=False,
        ) as handle:
            temporary_name = handle.name
            torch.save(
                {"meta": dict(meta), "state_dict": state_dict},
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        prepared = temporary
        temporary_name = None
        exported, exported_meta = _checkpoint_payload(temporary)
        if (
            int(exported_meta.get("epoch", -1)) != int(meta.get("epoch", -2))
            or set(exported.get("state_dict", {})) != set(state_dict)
        ):
            raise RuntimeError("Model-only checkpoint export did not round-trip")
        del exported
        prepared = None
        return temporary
    finally:
        del payload
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        if prepared is not None:
            try:
                prepared.unlink()
            except FileNotFoundError:
                pass


def _commit_published_checkpoint(
    temporary: Path, destination: Path, overwrite: bool = False
) -> str:
    if overwrite:
        os.replace(str(temporary), str(destination))
    else:
        os.link(str(temporary), str(destination))
        temporary.unlink()
    if hasattr(os, "O_DIRECTORY"):
        directory_fd = os.open(
            str(destination.parent), os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return "model_only_export"


def _verify_checkpoint(
    model: ModelSpec,
    dataset: DatasetSpec,
    checkpoint: Path,
    work_dir: Path,
    expected_epoch: int,
    device: str,
    image_count: int,
) -> Dict[str, Any]:
    payload, meta = _checkpoint_payload(checkpoint)
    state_dict = payload.get("state_dict", payload.get("model"))
    finite_audit = finite_state_dict_audit(state_dict)
    epoch = meta.get("epoch")
    if epoch is None or int(epoch) != expected_epoch:
        raise RuntimeError(
            "Final checkpoint epoch mismatch: expected {}, got {}".format(
                expected_epoch, epoch
            )
        )
    del payload
    gc.collect()

    verify_dir = work_dir / "verification"
    cfg = build_runtime_config(
        model,
        dataset,
        verify_dir / "model",
        mode="test",
        test_split="train",
        dump_config=False,
    )
    cfg.dump(str(verify_dir / "runtime_config.py"))
    register_framework(model.framework)
    detector = init_detector(cfg, str(checkpoint), device=device)
    # init_detector intentionally loads non-strictly for broad checkpoint
    # compatibility. Reload strictly so a missing dataset-specific head cannot
    # pass merely because a randomly initialized module still runs.
    load_checkpoint(detector, str(checkpoint), map_location="cpu", strict=True)
    detector.eval()

    index = CocoIndex(dataset, "train")
    images = index.images[:image_count]
    labels_seen = 0
    detections_seen = 0
    checked = []
    for image in images:
        image_id = int(image["id"])
        sample = inference_detector(detector, str(index.image_path(image)))
        instances = sample.pred_instances.to("cpu")
        labels = instances.labels
        if labels.numel():
            minimum = int(labels.min().item())
            maximum = int(labels.max().item())
            if minimum < 0 or maximum >= dataset.num_classes:
                raise RuntimeError(
                    "Inference produced label range [{}, {}] for {} classes".format(
                        minimum, maximum, dataset.num_classes
                    )
                )
            labels_seen += int(labels.numel())
        detections_seen += int(instances.bboxes.shape[0])
        checked.append(image_id)

    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "strict_state_dict_load": True,
        "checkpoint_epoch": int(epoch),
        "split": "train",
        "image_ids": checked,
        "images": len(checked),
        "detections": detections_seen,
        "labels": labels_seen,
        "state_dict_finite": finite_audit,
    }


def _update_dataset_manifest(
    registry: Registry, dataset_id: str, record: Dict[str, Any]
) -> Path:
    manifest_path = project_root() / "checkpoints" / dataset_id / "manifest.json"
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    with FileLock(str(lock_path), timeout=60):
        existing: Dict[str, Dict[str, Any]] = {}
        if manifest_path.is_file():
            payload = _load_json_mapping(
                manifest_path, "{} checkpoint manifest".format(dataset_id)
            )
            if (
                int(payload.get("schema_version", -1)) != 1
                or payload.get("dataset") != dataset_id
                or payload.get("selection_policy") != "fixed_final_epoch"
                or payload.get("validation_policy") != "post_training_only"
                or not isinstance(payload.get("checkpoints"), list)
            ):
                raise RuntimeError(
                    "Existing {} checkpoint manifest has invalid metadata".format(
                        dataset_id
                    )
                )
            for item in payload["checkpoints"]:
                if not isinstance(item, dict) or not item.get("model"):
                    raise RuntimeError(
                        "Existing {} checkpoint manifest has an invalid record".format(
                            dataset_id
                        )
                    )
                model_id = str(item["model"])
                if model_id in existing:
                    raise RuntimeError(
                        "Existing {} checkpoint manifest has duplicate model {}".format(
                            dataset_id, model_id
                        )
                    )
                existing[model_id] = item
        existing[str(record["model"])] = record
        ordered = [
            existing[model_id]
            for model_id in registry.target_ids()
            if model_id in existing
        ]
        unknown = sorted(set(existing).difference(registry.target_ids()))
        if unknown:
            raise RuntimeError(
                "Existing {} checkpoint manifest contains unknown models: {}".format(
                    dataset_id, ", ".join(unknown)
                )
            )
        atomic_json(
            manifest_path,
            {
                "schema_version": 1,
                "dataset": dataset_id,
                "selection_policy": "fixed_final_epoch",
                "validation_policy": "post_training_only",
                "checkpoints": ordered,
            },
        )
    return manifest_path


def run_training(
    registry: Registry,
    dataset_id: str,
    model_id: str,
    work_dir: Optional[Path] = None,
    device: str = "cuda:0",
    download_weights: bool = False,
    overwrite_checkpoint: bool = False,
    resume: Optional[Union[str, Path]] = None,
    max_images: Optional[int] = None,
) -> Path:
    kwargs = dict(work_dir=work_dir, device=device, download_weights=download_weights,
                  overwrite_checkpoint=overwrite_checkpoint, resume=resume, max_images=max_images)
    if max_images is not None:
        preparation = {}
        try:
            return _run_training(registry, dataset_id, model_id,
                                 _preparation_state=preparation, **kwargs)
        except BaseException as exc:
            owned_dir = preparation.get("work_dir")
            if owned_dir is not None and not (owned_dir / "run.json").exists():
                failed_at = _utc_now()
                failure = "{}: {}".format(type(exc).__name__, exc)
                atomic_json(owned_dir / "run.json", {
                    "schema_version": 1, "status": "failed", "dataset": dataset_id,
                    "model": model_id, "work_dir": _relative(owned_dir),
                    "canonical_publication_allowed": False, "failure": failure,
                    "phase": "preparation", "updated_at": failed_at,
                    "attempts": [{"status": "failed", "reason": failure,
                                  "completed_at": failed_at}],
                })
            raise
    destination = checkpoint_path(registry.model(model_id), registry.dataset(dataset_id))
    lock_path = destination.with_suffix(destination.suffix + ".training.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path), timeout=0):
        return _run_training(registry, dataset_id, model_id, **kwargs)


def _run_training(
    registry: Registry,
    dataset_id: str,
    model_id: str,
    work_dir: Optional[Path] = None,
    device: str = "cuda:0",
    download_weights: bool = False,
    overwrite_checkpoint: bool = False,
    resume: Optional[Union[str, Path]] = None,
    max_images: Optional[int] = None,
    _preparation_state: Optional[Dict[str, Any]] = None,
) -> Path:
    diagnostic = max_images is not None
    if diagnostic:
        if dataset_id != "voc":
            raise ValueError("Training diagnostics are authorized for VOC only")
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0:
            raise ValueError("Diagnostic --max-images must be a positive training-image count")
        if resume is not None or overwrite_checkpoint:
            raise ValueError("Diagnostic training cannot resume or overwrite a canonical checkpoint")
    if dataset_id == "coco":
        raise ValueError("The benchmark uses official COCO checkpoints; fine-tune VOC or BDD100K")
    dataset = registry.dataset(dataset_id)
    model = registry.model(model_id)
    protocol = dict(registry.training_protocol)
    train_annotation = dataset.root / dataset.split("train").annotation
    if not train_annotation.is_file():
        raise FileNotFoundError(
            "Prepared COCO-format train annotation is required. Run "
            "`lgp data prepare --dataset {}`.".format(dataset_id)
        )
    destination = checkpoint_path(model, dataset)
    if not diagnostic and (destination.exists() or destination.is_symlink()) and not overwrite_checkpoint:
        raise FileExistsError(
            "Fine-tuned checkpoint already exists: {} (use --overwrite-checkpoint explicitly)".format(
                destination
            )
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    actual_work_dir = (
        work_dir
        or project_root() / "outputs" / "training" / dataset_id / model_id / stamp
    ).resolve()
    if diagnostic:
        actual_work_dir.relative_to(registry.root.resolve())
        try:
            actual_work_dir.relative_to((registry.root / "checkpoints").resolve())
        except ValueError:
            pass
        else:
            raise ValueError("Diagnostic training cannot write inside the checkpoint registry")
        if actual_work_dir.exists():
            raise FileExistsError("Diagnostic training requires a new work directory")
    existing_items = list(actual_work_dir.iterdir()) if actual_work_dir.is_dir() else []
    if resume is None and existing_items:
        raise FileExistsError(
            "Refusing to start a fresh training run in non-empty directory: {}".format(
                actual_work_dir
            )
        )
    if resume is not None and not actual_work_dir.is_dir():
        raise FileNotFoundError(
            "Resume requires an existing work directory: {}".format(actual_work_dir)
        )
    actual_work_dir.mkdir(parents=True, exist_ok=not diagnostic)
    if diagnostic and _preparation_state is not None:
        _preparation_state["work_dir"] = actual_work_dir
    superseded_checkpoint: Optional[Dict[str, Any]] = None
    if (
        resume is None
        and overwrite_checkpoint
        and destination.is_file()
    ):
        superseded_checkpoint = _backup_superseded_checkpoint(
            destination,
            actual_work_dir,
            dataset_id,
            model_id,
        )
    run_path = actual_work_dir / "run.json"
    previous_run: Dict[str, Any] = {}
    if resume is not None:
        if not run_path.is_file():
            raise FileNotFoundError(
                "Resume requires the original run.json: {}".format(run_path)
            )
        try:
            previous_run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            raise RuntimeError(
                "Resume run.json is unreadable: {}".format(run_path)
            ) from exc
        if not isinstance(previous_run, dict):
            raise RuntimeError("Resume run.json root must be a mapping")

    register_framework(model.framework)
    cfg = build_runtime_config(
        model,
        dataset,
        actual_work_dir,
        mode="train",
        training_protocol=protocol,
        training_safety=registry.training_safety,
        warm_start_source=registry.dataset("coco"),
        dump_config=False,
    )
    diagnostic_receipt = None
    if diagnostic:
        from ..training_diagnostic import prepare_diagnostic_dataset

        original_annotation = train_annotation
        dataset, diagnostic_receipt = prepare_diagnostic_dataset(
            cfg, dataset, actual_work_dir, max_images,
        )
        train_annotation = dataset.root / dataset.split("train").annotation
        diagnostic_receipt.update({
            "source_annotation": _relative(original_annotation),
            "subset_annotation": _relative(train_annotation),
            "canonical_publication_allowed": False,
        })
    protocol_config_path = actual_work_dir / "protocol_config.py"
    if resume is not None:
        if not protocol_config_path.is_file():
            raise FileNotFoundError(
                "Resume requires the original protocol_config.py: {}".format(
                    protocol_config_path
                )
            )
        previous_protocol_hash = file_digest(protocol_config_path)
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="lgp-protocol-",
                suffix=".py",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
            cfg.dump(temporary_name)
            protocol_config_hash = file_digest(Path(temporary_name))
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)
        if previous_protocol_hash != protocol_config_hash:
            raise RuntimeError(
                "Resolved protocol changed since the interrupted run; refusing unsafe resume"
            )
        protocol_config_hash = previous_protocol_hash
    else:
        cfg.dump(str(protocol_config_path))
        protocol_config_hash = file_digest(protocol_config_path)

    source_checkpoint, source_checkpoint_sha256 = _verified_source_checkpoint(
        registry, model, download_weights
    )
    train_annotation_sha256 = file_digest(train_annotation)
    git = _git_provenance()
    if not diagnostic and (not git.get("commit") or git.get("dirty") is not False):
        raise RuntimeError(
            "Formal fine-tuning requires a clean Git worktree with a recorded commit"
        )
    expected_epoch = int(model.finetune_epochs[dataset_id])
    if resume is not None:
        previous_git = previous_run.get("git", {})
        if (
            previous_run.get("dataset") != dataset_id
            or previous_run.get("model") != model_id
            or previous_run.get("status") not in {"running", "failed"}
            or previous_run.get("work_dir") != _relative(actual_work_dir)
            or previous_run.get("protocol_config_sha256") != protocol_config_hash
            or previous_run.get("train_annotation_sha256")
            != train_annotation_sha256
            or previous_run.get("source_checkpoint_sha256")
            != source_checkpoint_sha256
            or not isinstance(previous_git, dict)
            or previous_git.get("commit") != git.get("commit")
            or previous_git.get("dirty") is not False
        ):
            raise RuntimeError(
                "Resume provenance differs from the original dataset/model/config/data/source/Git state"
            )

    def checked_resume_checkpoint(path: Path) -> Path:
        resolved = path.expanduser().resolve()
        try:
            resolved.relative_to(actual_work_dir)
        except ValueError as exc:
            raise RuntimeError(
                "Resume checkpoint must belong to the original work directory: {}".format(
                    resolved
                )
            ) from exc
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        payload, meta = _checkpoint_payload(resolved)
        state_dict = payload.get("state_dict", payload.get("model"))
        finite_state_dict_audit(state_dict)
        epoch = meta.get("epoch")
        dataset_meta = meta.get("dataset_meta", {})
        classes = (
            dataset_meta.get("classes")
            if isinstance(dataset_meta, Mapping)
            else None
        )
        del payload
        if (
            epoch is None
            or int(epoch) < 0
            or int(epoch) > expected_epoch
            or classes is None
            or list(classes) != list(dataset.classes)
        ):
            raise RuntimeError(
                "Resume checkpoint epoch/class metadata does not match this run"
            )
        switch_boundary = resume_phase_switch_boundary(cfg)
        if switch_boundary is not None and int(epoch) >= switch_boundary:
            raise RuntimeError(
                "Checkpoint epoch {} is at/after non-serializable phase-switch "
                "boundary {}; start a new run instead of unsafe resume".format(
                    epoch, switch_boundary
                )
            )
        return resolved

    validated_resume_checkpoint: Optional[Path] = None
    if resume is None:
        cfg.load_from = str(source_checkpoint)
        cfg.resume = False
    else:
        resume_value = str(resume)
        cfg.resume = True
        if resume_value == "auto":
            pointer = actual_work_dir / "last_checkpoint"
            if not pointer.is_file():
                raise FileNotFoundError(
                    "Automatic resume found no last_checkpoint in {}".format(
                        actual_work_dir
                    )
                )
            pointer_value = pointer.read_text(encoding="utf-8").strip()
            if not pointer_value:
                raise RuntimeError("last_checkpoint is empty: {}".format(pointer))
            pointer_checkpoint = Path(pointer_value)
            if not pointer_checkpoint.is_absolute():
                pointer_checkpoint = actual_work_dir / pointer_checkpoint
            validated_resume_checkpoint = checked_resume_checkpoint(
                pointer_checkpoint
            )
            cfg.load_from = None
        else:
            resume_checkpoint = checked_resume_checkpoint(Path(resume_value))
            validated_resume_checkpoint = resume_checkpoint
            cfg.load_from = str(resume_checkpoint)
    cfg.launcher = "none"
    parsed_device = torch.device(device)
    if parsed_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        torch.cuda.set_device(parsed_device.index or 0)
        cfg.device = "cuda"
    else:
        cfg.device = parsed_device.type
    cfg.dump(str(actual_work_dir / "runtime_config.py"))

    attempts = list(previous_run.get("attempts", []))
    attempts.append({"started_at": _utc_now(), "resume": str(resume) if resume else None})
    run_record: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "dataset": dataset_id,
        "model": model_id,
        "work_dir": _relative(actual_work_dir),
        "started_at": previous_run.get("started_at", attempts[0]["started_at"]),
        "updated_at": _utc_now(),
        "attempts": attempts,
        "git": git,
        "protocol": {
            **protocol,
            "epochs": int(model.finetune_epochs[dataset_id]),
            "micro_batch_size": model.train_batch_size,
            "reference_batch_size": model.train_base_batch_size,
        },
        "protocol_config": _relative(protocol_config_path),
        "protocol_config_sha256": protocol_config_hash,
        "train_annotation": _relative(train_annotation),
        "train_annotation_sha256": train_annotation_sha256,
        "source_checkpoint": _relative(source_checkpoint),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "gradient_clipping": (
            dict(model.train_grad_clip)
            if model.train_grad_clip is not None
            else None
        ),
        "head_initialization": _head_initialization_record(model, dataset_id),
        "superseded_checkpoint": superseded_checkpoint,
    }
    if diagnostic:
        run_record["diagnostic"] = diagnostic_receipt
        run_record["canonical_publication_allowed"] = False
    atomic_json(run_path, run_record)

    publication_temporary: Optional[Path] = None
    head_initialization_audit: Optional[Dict[str, Any]] = None
    head_initialization_audit_path = (
        actual_work_dir / "warm_start_load_audit.json"
    )
    try:
        runner = Runner.from_cfg(cfg)
        if diagnostic:
            from ..training_diagnostic import verify_diagnostic_dataset

            diagnostic_receipt["runtime_dataset"] = verify_diagnostic_dataset(
                runner.train_dataloader.dataset, diagnostic_receipt["image_ids"],
            )
            if len(runner.train_dataloader) == 0:
                raise RuntimeError("Diagnostic image count yields no batches under the unchanged training recipe")
        warm_start_audit = _audit_warm_start(
            runner.model, source_checkpoint, model_id
        )
        run_record["warm_start_audit"] = warm_start_audit
        if validated_resume_checkpoint is not None:
            resume_audit = _audit_exact_resume(
                runner.model, validated_resume_checkpoint
            )
            run_record["attempts"][-1]["resume_checkpoint"] = resume_audit
        run_record["updated_at"] = _utc_now()
        atomic_json(run_path, run_record)
        runner.train()
        if diagnostic:
            diagnostic_receipt["runtime_dataset_after_training"] = verify_diagnostic_dataset(
                runner.train_dataloader.dataset, diagnostic_receipt["image_ids"],
            )
        head_initialization_audit = _load_head_initialization_audit(
            head_initialization_audit_path,
            model,
            dataset,
            registry.dataset("coco"),
        )
        if head_initialization_audit is not None:
            run_record["warm_start_load_audit"] = head_initialization_audit
            attempts[-1][
                "warm_start_load_audit"
            ] = head_initialization_audit
            run_record["updated_at"] = _utc_now()
            atomic_json(run_path, run_record)
        final_checkpoint = _last_checkpoint(actual_work_dir, expected_epoch)
        if diagnostic:
            try:
                final_checkpoint.resolve().relative_to(actual_work_dir)
            except ValueError as exc:
                raise RuntimeError("Diagnostic checkpoint is outside its own work directory") from exc
        del runner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if diagnostic:
            verification = _verify_checkpoint(
                model, dataset, final_checkpoint, actual_work_dir, expected_epoch,
                device, min(int(protocol["verify_train_images"]), max_images),
            )
            if (file_digest(train_annotation) != train_annotation_sha256
                    or file_digest(source_checkpoint) != source_checkpoint_sha256
                    or file_digest(protocol_config_path) != protocol_config_hash
                    or file_digest(original_annotation) != diagnostic_receipt["source_annotation_sha256"]):
                raise RuntimeError("Diagnostic training inputs changed during execution")
            completed_at = _utc_now()
            run_record.update({
                "status": "diagnostic_complete", "updated_at": completed_at,
                "completed_at": completed_at, "final_checkpoint": _relative(final_checkpoint),
                "final_checkpoint_sha256": file_digest(final_checkpoint),
                "verification": verification, "publish_method": None,
                "canonical_publication_allowed": False,
            })
            attempts[-1].update({"completed_at": completed_at, "status": "diagnostic_complete"})
            atomic_json(run_path, run_record)
            return final_checkpoint
        publication_temporary = _prepare_published_checkpoint(
            final_checkpoint, destination
        )
        verification = _verify_checkpoint(
            model,
            dataset,
            publication_temporary,
            actual_work_dir,
            expected_epoch,
            device,
            int(protocol["verify_train_images"]),
        )
        if (
            file_digest(train_annotation) != train_annotation_sha256
            or file_digest(source_checkpoint) != source_checkpoint_sha256
        ):
            raise RuntimeError(
                "Training annotation or COCO source checkpoint changed during training"
            )
        publish_git = _git_provenance()
        if (
            publish_git.get("commit") != git.get("commit")
            or publish_git.get("dirty") is not False
        ):
            raise RuntimeError(
                "Git state changed during training; refusing to publish the checkpoint"
            )
        publish_method = _commit_published_checkpoint(publication_temporary, destination,
                                                      overwrite=overwrite_checkpoint)
        publication_temporary = None
        if superseded_checkpoint is not None:
            preserved = project_root() / str(
                superseded_checkpoint["path"]
            )
            if (
                not preserved.is_file()
                or preserved.stat().st_size
                != int(superseded_checkpoint["bytes"])
                or file_digest(preserved)
                != superseded_checkpoint["sha256"]
            ):
                raise RuntimeError(
                    "Superseded checkpoint evidence changed during publication"
                )
        destination_sha256 = file_digest(destination)
        completed_at = _utc_now()
        expected_manifest_path = (
            project_root() / "checkpoints" / dataset_id / "manifest.json"
        )
        run_record.update(
            {
                "status": "complete",
                "updated_at": completed_at,
                "completed_at": completed_at,
                "final_checkpoint": _relative(destination),
                "final_checkpoint_sha256": destination_sha256,
                "checkpoint_manifest": _relative(expected_manifest_path),
                "verification": verification,
                "publish_method": publish_method,
            }
        )
        attempts[-1]["completed_at"] = completed_at
        attempts[-1]["status"] = "complete"
        atomic_json(run_path, run_record)
        run_sha256 = file_digest(run_path)
        manifest_record = {
            "model": model_id,
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "sha256": destination_sha256,
            "num_classes": dataset.num_classes,
            "source_checkpoint": source_checkpoint.name,
            "source_sha256": run_record["source_checkpoint_sha256"],
            "train_annotation": _relative(train_annotation),
            "train_annotation_sha256": run_record["train_annotation_sha256"],
            "protocol_config_sha256": protocol_config_hash,
            "selection_policy": protocol["selection_policy"],
            "validation_policy": protocol["validation_policy"],
            "epochs": expected_epoch,
            "seed": int(protocol["seed"]),
            "deterministic": bool(protocol["deterministic"]),
            "amp": bool(protocol["amp"]),
            "micro_batch_size": model.train_batch_size,
            "reference_batch_size": model.train_base_batch_size,
            "lr_scaling": protocol["lr_scaling"],
            "gradient_clipping": (
                dict(model.train_grad_clip)
                if model.train_grad_clip is not None
                else None
            ),
            "head_initialization": _head_initialization_record(
                model, dataset_id
            ),
            "git_commit": git.get("commit"),
            "completed_at": completed_at,
            "work_dir": _relative(actual_work_dir),
            "training_run": _relative(run_path),
            "training_run_sha256": run_sha256,
            "verification": verification,
            "warm_start_audit": warm_start_audit,
            "warm_start_load_audit": head_initialization_audit,
            "superseded_checkpoint": superseded_checkpoint,
            "publish_method": publish_method,
        }
        _update_dataset_manifest(registry, dataset_id, manifest_record)
        return destination
    except Exception as exc:
        failed_at = _utc_now()
        if (
            head_initialization_audit is None
            and head_initialization_audit_path.is_file()
        ):
            try:
                head_initialization_audit = _load_json_mapping(
                    head_initialization_audit_path,
                    "semantic head initialization audit",
                )
                run_record[
                    "warm_start_load_audit"
                ] = head_initialization_audit
                attempts[-1][
                    "warm_start_load_audit"
                ] = head_initialization_audit
            except Exception as audit_exc:
                attempts[-1][
                    "warm_start_load_audit_error"
                ] = "{}: {}".format(type(audit_exc).__name__, audit_exc)
        run_record.update(
            {
                "status": "failed",
                "updated_at": failed_at,
                "failure": "{}: {}".format(type(exc).__name__, exc),
            }
        )
        attempts[-1]["completed_at"] = failed_at
        attempts[-1]["status"] = "failed"
        attempts[-1]["reason"] = run_record["failure"]
        atomic_json(run_path, run_record)
        raise
    finally:
        if publication_temporary is not None:
            try:
                publication_temporary.unlink()
            except FileNotFoundError:
                pass
