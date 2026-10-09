from __future__ import annotations

import gc
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch

from .io import file_digest
from .modeling import checkpoint_path, finite_state_dict_audit
from .registry import Registry


def _load_mapping(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError("{} is missing: {}".format(label, path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("{} is unreadable: {}".format(label, path)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("{} root must be a mapping".format(label))
    return payload


def _project_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RuntimeError("{} path is missing".format(label))
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("{} path escapes the repository".format(label)) from exc
    return resolved


def _digest(path: Path, cache: Optional[MutableMapping[Path, str]]) -> str:
    resolved = path.resolve()
    if cache is not None and resolved in cache:
        return cache[resolved]
    value = file_digest(resolved)
    if cache is not None:
        cache[resolved] = value
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _is_utc_second_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return False
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ") == value


def _validate_superseded_checkpoint_evidence(
    root: Path,
    record: Mapping[str, Any],
    model_id: str,
    destination: Path,
    *,
    deep: bool,
    hash_cache: Optional[MutableMapping[Path, str]],
) -> None:
    superseded = record.get("superseded_checkpoint")
    disposition = record.get("superseded_checkpoint_disposition")
    if superseded is None:
        if disposition is not None:
            raise RuntimeError(
                "Superseded checkpoint disposition lacks source evidence"
            )
        return
    if (
        not isinstance(superseded, dict)
        or superseded.get("status") != "preserved"
        or superseded.get("method") not in {"hardlink", "copy"}
        or int(superseded.get("bytes", 0)) <= (1 << 20)
        or not _is_sha256(superseded.get("sha256"))
        or not _is_sha256(superseded.get("manifest_sha256"))
        or not _is_sha256(superseded.get("manifest_record_sha256"))
    ):
        raise RuntimeError(
            "Superseded checkpoint evidence metadata is invalid"
        )
    superseded_path = _project_path(
        root,
        superseded.get("path"),
        "superseded checkpoint",
    )
    superseded_record_path = _project_path(
        root,
        superseded.get("manifest_record"),
        "superseded manifest record",
    )
    superseded_manifest_path = _project_path(
        root,
        superseded.get("manifest"),
        "superseded manifest",
    )
    evidence_paths = (
        superseded_path,
        superseded_record_path,
        superseded_manifest_path,
    )
    if disposition is not None:
        expected_keys = {
            "status",
            "method",
            "reason",
            "authorization",
            "purged_at_utc",
        }
        if (
            not isinstance(disposition, dict)
            or set(disposition) != expected_keys
            or disposition.get("status") != "purged"
            or disposition.get("method") != "permanent_delete"
            or disposition.get("reason")
            != "explicit_post_acceptance_cleanup"
            or disposition.get("authorization") != "explicit_user_request"
            or not _is_utc_second_timestamp(disposition.get("purged_at_utc"))
        ):
            raise RuntimeError(
                "Superseded checkpoint disposition metadata is invalid"
            )
        if any(path.exists() for path in evidence_paths):
            raise RuntimeError(
                "Purged superseded checkpoint evidence files unexpectedly exist"
            )
        return
    if (
        not superseded_path.is_file()
        or superseded_path.stat().st_size != int(superseded["bytes"])
        or not superseded_record_path.is_file()
        or not superseded_manifest_path.is_file()
    ):
        raise RuntimeError("Superseded checkpoint evidence files are missing")
    if not deep:
        return
    if (
        str(superseded["sha256"]).lower()
        != _digest(superseded_path, hash_cache).lower()
        or str(superseded["manifest_record_sha256"]).lower()
        != _digest(superseded_record_path, hash_cache).lower()
        or str(superseded["manifest_sha256"]).lower()
        != _digest(superseded_manifest_path, hash_cache).lower()
    ):
        raise RuntimeError(
            "Superseded checkpoint evidence SHA-256 mismatch"
        )
    superseded_record = _load_mapping(
        superseded_record_path,
        "superseded manifest record",
    )
    if (
        superseded_record.get("model") != model_id
        or superseded_record.get("path") != destination.name
        or str(superseded_record.get("sha256", "")).lower()
        != str(superseded["sha256"]).lower()
    ):
        raise RuntimeError("Superseded manifest record is inconsistent")


def _unique_records(
    payload: Mapping[str, Any], label: str
) -> Dict[str, Dict[str, Any]]:
    records = payload.get("checkpoints")
    if not isinstance(records, list):
        raise RuntimeError("{} has no checkpoint list".format(label))
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("model"):
            raise RuntimeError("{} contains an invalid checkpoint record".format(label))
        model_id = str(record["model"])
        if model_id in indexed:
            raise RuntimeError(
                "{} contains duplicate model {}".format(label, model_id)
            )
        indexed[model_id] = record
    return indexed


def validate_finetuned_checkpoint(
    registry: Registry,
    dataset_id: str,
    model_id: str,
    *,
    deep: bool,
    expected_commit: Optional[str] = None,
    hash_cache: Optional[MutableMapping[Path, str]] = None,
) -> Tuple[bool, List[str]]:
    """Validate a published fine-tuned checkpoint and its complete evidence chain."""
    issues: List[str] = []
    if dataset_id not in {"voc", "bdd100k"}:
        return False, ["fine-tuned artifact validation only supports VOC and BDD100K"]
    try:
        root = registry.root.resolve()
        dataset = registry.dataset(dataset_id)
        model = registry.model(model_id)
        destination = checkpoint_path(model, dataset).resolve()
        if not destination.is_file():
            raise FileNotFoundError("checkpoint is missing: {}".format(destination))
        if destination.stat().st_size <= (1 << 20):
            raise RuntimeError("checkpoint is implausibly small")

        manifest_path = root / "checkpoints" / dataset_id / "manifest.json"
        manifest = _load_mapping(
            manifest_path, "{} checkpoint manifest".format(dataset_id)
        )
        if (
            int(manifest.get("schema_version", -1)) != 1
            or manifest.get("dataset") != dataset_id
            or manifest.get("selection_policy") != "fixed_final_epoch"
            or manifest.get("validation_policy") != "post_training_only"
        ):
            raise RuntimeError("checkpoint manifest metadata is invalid")
        records = _unique_records(manifest, "{} manifest".format(dataset_id))
        if set(records).difference(registry.target_ids()):
            raise RuntimeError("checkpoint manifest contains unknown detector ids")
        if model_id not in records:
            raise RuntimeError("manifest record is missing")
        record = records[model_id]

        expected_epochs = int(model.finetune_epochs[dataset_id])
        protocol = dict(registry.training_protocol)
        expected_gradient_clipping = (
            dict(model.train_grad_clip)
            if model.train_grad_clip is not None
            else None
        )
        declared_head_init = model.finetune_head_init.get(dataset_id)
        expected_head_init = (
            {
                **dict(declared_head_init),
                "target_to_source": dict(
                    declared_head_init["target_to_source"]
                ),
            }
            if declared_head_init is not None
            else None
        )
        expected_record = {
            "path": destination.name,
            "bytes": destination.stat().st_size,
            "num_classes": dataset.num_classes,
            "selection_policy": protocol["selection_policy"],
            "validation_policy": protocol["validation_policy"],
            "epochs": expected_epochs,
            "seed": int(protocol["seed"]),
            "deterministic": bool(protocol["deterministic"]),
            "amp": bool(protocol["amp"]),
            "micro_batch_size": model.train_batch_size,
            "reference_batch_size": model.train_base_batch_size,
            "lr_scaling": protocol["lr_scaling"],
            "gradient_clipping": expected_gradient_clipping,
            "head_initialization": expected_head_init,
            "publish_method": "model_only_export",
        }
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise RuntimeError(
                    "manifest {} mismatch: expected {!r}, got {!r}".format(
                        key, expected, record.get(key)
                    )
                )
        git_commit = record.get("git_commit")
        if not isinstance(git_commit, str) or not git_commit:
            raise RuntimeError("manifest Git commit is missing")
        if expected_commit is not None and git_commit != expected_commit:
            raise RuntimeError("manifest Git commit differs from this campaign")

        verification = record.get("verification")
        if not isinstance(verification, dict):
            raise RuntimeError("verification evidence is missing")
        expected_verification = {
            "strict_state_dict_load": True,
            "checkpoint_epoch": expected_epochs,
            "split": "train",
            "images": int(protocol["verify_train_images"]),
        }
        for key, expected in expected_verification.items():
            if verification.get(key) != expected:
                raise RuntimeError(
                    "verification {} mismatch: expected {!r}, got {!r}".format(
                        key, expected, verification.get(key)
                    )
                )

        warm_start = record.get("warm_start_audit")
        if (
            not isinstance(warm_start, dict)
            or warm_start.get("status") != "verified"
            or warm_start.get("missing_keys") != []
        ):
            raise RuntimeError("COCO warm-start audit evidence is missing")

        warm_start_load = record.get("warm_start_load_audit")
        if expected_head_init is None:
            if warm_start_load is not None:
                raise RuntimeError(
                    "Unexpected semantic head initialization evidence is present"
                )
        else:
            source_dataset = registry.dataset(
                str(expected_head_init["source_dataset"])
            )
            expected_mapping = [
                {
                    "target_index": target_index,
                    "target_class": target_class,
                    "source_index": source_dataset.classes.index(
                        str(
                            expected_head_init["target_to_source"][
                                target_class
                            ]
                        )
                    ),
                    "source_class": str(
                        expected_head_init["target_to_source"][target_class]
                    ),
                }
                for target_index, target_class in enumerate(dataset.classes)
            ]
            if (
                not isinstance(warm_start_load, dict)
                or warm_start_load.get("status") != "verified"
                or warm_start_load.get("policy")
                != expected_head_init["policy"]
                or warm_start_load.get("source_dataset")
                != source_dataset.id
                or warm_start_load.get("target_dataset") != dataset_id
                or warm_start_load.get("parameter_stem")
                != expected_head_init["parameter_stem"]
                or warm_start_load.get("class_mapping") != expected_mapping
                or warm_start_load.get("missing_keys") != []
                or warm_start_load.get("unexpected_keys") != []
                or warm_start_load.get("shape_mismatch_keys") != []
                or warm_start_load.get("state_dict_exact_match") is not True
                or warm_start_load.get("expected_state_sha256")
                != warm_start_load.get("loaded_state_sha256")
                or not _is_sha256(
                    warm_start_load.get("expected_state_sha256")
                )
            ):
                raise RuntimeError(
                    "Semantic head initialization evidence is invalid"
                )
            expected_parameters = warm_start_load.get(
                "expected_parameter_sha256"
            )
            loaded_parameters = warm_start_load.get(
                "loaded_parameter_sha256"
            )
            expected_parameter_keys = {
                str(expected_head_init["parameter_stem"]) + ".weight",
                str(expected_head_init["parameter_stem"]) + ".bias",
            }
            if (
                not isinstance(expected_parameters, dict)
                or not isinstance(loaded_parameters, dict)
                or set(expected_parameters) != expected_parameter_keys
                or expected_parameters != loaded_parameters
                or not all(
                    _is_sha256(value)
                    for value in expected_parameters.values()
                )
            ):
                raise RuntimeError(
                    "Semantic head parameter fingerprints are invalid"
                )

        superseded = record.get("superseded_checkpoint")
        _validate_superseded_checkpoint_evidence(
            root,
            record,
            model_id,
            destination,
            deep=deep,
            hash_cache=hash_cache,
        )

        training_run = _project_path(
            root, record.get("training_run"), "training run"
        )
        work_dir = _project_path(root, record.get("work_dir"), "training work dir")
        if training_run != work_dir / "run.json":
            raise RuntimeError("training run is not inside its declared work dir")
        if not training_run.is_file():
            raise FileNotFoundError("training run is missing")

        if deep:
            if str(record.get("sha256", "")).lower() != _digest(
                destination, hash_cache
            ).lower():
                raise RuntimeError("checkpoint SHA-256 mismatch")
            if str(record.get("training_run_sha256", "")).lower() != _digest(
                training_run, hash_cache
            ).lower():
                raise RuntimeError("training run SHA-256 mismatch")
            run = _load_mapping(training_run, "training run")
            run_git = run.get("git")
            expected_protocol = {
                **protocol,
                "epochs": expected_epochs,
                "micro_batch_size": model.train_batch_size,
                "reference_batch_size": model.train_base_batch_size,
            }
            if (
                run.get("status") != "complete"
                or run.get("dataset") != dataset_id
                or run.get("model") != model_id
                or run.get("work_dir") != record.get("work_dir")
                or run.get("protocol") != expected_protocol
                or run.get("final_checkpoint")
                != destination.relative_to(root).as_posix()
                or str(run.get("final_checkpoint_sha256", "")).lower()
                != str(record.get("sha256", "")).lower()
                or run.get("verification") != verification
                or run.get("warm_start_audit") != warm_start
                or run.get("warm_start_load_audit") != warm_start_load
                or run.get("publish_method") != "model_only_export"
                or run.get("gradient_clipping")
                != expected_gradient_clipping
                or run.get("head_initialization") != expected_head_init
                or run.get("superseded_checkpoint") != superseded
                or not isinstance(run_git, dict)
                or run_git.get("commit") != git_commit
                or run_git.get("dirty") is not False
            ):
                raise RuntimeError("training run provenance is inconsistent")
            attempts = run.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                raise RuntimeError("training attempt history is missing")
            for attempt in attempts:
                if not isinstance(attempt, dict):
                    raise RuntimeError("training attempt record is invalid")
                if attempt.get("resume") is not None:
                    resume_evidence = attempt.get("resume_checkpoint")
                    if (
                        not isinstance(resume_evidence, dict)
                        or resume_evidence.get("status") != "verified"
                        or not resume_evidence.get("strict_key_and_shape_match")
                        or not resume_evidence.get("checkpoint_sha256")
                    ):
                        raise RuntimeError(
                            "resumed attempt lacks exact checkpoint audit evidence"
                        )
            if expected_head_init is not None:
                verified_loads = [
                    attempt.get("warm_start_load_audit")
                    for attempt in attempts
                    if isinstance(
                        attempt.get("warm_start_load_audit"), dict
                    )
                    and attempt["warm_start_load_audit"].get("status")
                    == "verified"
                ]
                if not verified_loads or not any(
                    audit.get("load_mode")
                    == "fresh_semantic_class_copy"
                    for audit in verified_loads
                ):
                    raise RuntimeError(
                        "Training attempts do not preserve a verified fresh "
                        "semantic class copy"
                    )

            annotation = dataset.root / dataset.split("train").annotation
            if not annotation.is_file():
                raise FileNotFoundError("current train annotation is missing")
            annotation_digest = _digest(annotation, hash_cache)
            if (
                record.get("train_annotation")
                != annotation.resolve().relative_to(root).as_posix()
                or str(record.get("train_annotation_sha256", "")).lower()
                != annotation_digest.lower()
                or run.get("train_annotation") != record.get("train_annotation")
                or str(run.get("train_annotation_sha256", "")).lower()
                != annotation_digest.lower()
            ):
                raise RuntimeError("train annotation provenance mismatch")

            coco = registry.dataset("coco")
            source = checkpoint_path(model, coco).resolve()
            if not source.is_file():
                raise FileNotFoundError("current COCO source checkpoint is missing")
            source_digest = _digest(source, hash_cache)
            if (
                record.get("source_checkpoint") != source.name
                or str(record.get("source_sha256", "")).lower()
                != source_digest.lower()
                or run.get("source_checkpoint")
                != source.relative_to(root).as_posix()
                or str(run.get("source_checkpoint_sha256", "")).lower()
                != source_digest.lower()
            ):
                raise RuntimeError("COCO source checkpoint provenance mismatch")
            coco_manifest = _load_mapping(
                root / "checkpoints" / "coco" / "manifest.json",
                "COCO checkpoint manifest",
            )
            if int(coco_manifest.get("schema_version", -1)) != 1:
                raise RuntimeError("COCO manifest schema is invalid")
            coco_records = _unique_records(coco_manifest, "COCO manifest")
            source_record = coco_records.get(model_id)
            if (
                source_record is None
                or Path(str(source_record.get("path", ""))).name != source.name
                or int(source_record.get("bytes", -1)) != source.stat().st_size
                or str(source_record.get("sha256", "")).lower()
                != source_digest.lower()
            ):
                raise RuntimeError("COCO manifest provenance mismatch")

            protocol_config = _project_path(
                root, run.get("protocol_config"), "protocol config"
            )
            protocol_digest = _digest(protocol_config, hash_cache)
            if (
                str(run.get("protocol_config_sha256", "")).lower()
                != protocol_digest.lower()
                or str(record.get("protocol_config_sha256", "")).lower()
                != protocol_digest.lower()
            ):
                raise RuntimeError("protocol config provenance mismatch")

            payload = torch.load(str(destination), map_location="cpu")
            try:
                if not isinstance(payload, Mapping):
                    raise RuntimeError("checkpoint root is not a mapping")
                state_dict = payload.get("state_dict")
                if not isinstance(state_dict, Mapping) or not state_dict:
                    raise RuntimeError("checkpoint state dictionary is missing")
                if not any(torch.is_tensor(value) for value in state_dict.values()):
                    raise RuntimeError("checkpoint state dictionary has no tensors")
                finite_state_dict_audit(state_dict)
                if "optimizer" in payload or "optim_wrapper" in payload:
                    raise RuntimeError("published checkpoint is not model-only")
                meta = payload.get("meta")
                dataset_meta = (
                    meta.get("dataset_meta") if isinstance(meta, Mapping) else None
                )
                classes = (
                    dataset_meta.get("classes")
                    if isinstance(dataset_meta, Mapping)
                    else None
                )
                if (
                    not isinstance(meta, Mapping)
                    or int(meta.get("epoch", -1)) != expected_epochs
                    or classes is None
                    or list(classes) != list(dataset.classes)
                ):
                    raise RuntimeError("checkpoint epoch/class metadata mismatch")
            finally:
                del payload
                gc.collect()
    except Exception as exc:
        issues.append("{}: {}".format(type(exc).__name__, exc))
    return not issues, issues


def strict_load_finetuned_checkpoint(
    registry: Registry, dataset_id: str, model_id: str, device: str
) -> None:
    """Build the current detector and require an exact state-dict load."""
    from mmdet.apis import init_detector
    from mmengine.runner.checkpoint import load_checkpoint

    from .runtime_config import build_runtime_config, register_framework

    dataset = registry.dataset(dataset_id)
    model = registry.model(model_id)
    checkpoint = checkpoint_path(model, dataset)
    temporary_root = registry.root / "outputs" / "training" / "validation"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="{}-{}-".format(dataset_id, model_id),
        dir=str(temporary_root),
    ) as work:
        cfg = build_runtime_config(
            model,
            dataset,
            Path(work),
            mode="test",
            test_split="train",
            dump_config=False,
        )
        register_framework(model.framework)
        detector = init_detector(cfg, str(checkpoint), device=device)
        load_checkpoint(
            detector,
            str(checkpoint),
            map_location="cpu",
            strict=True,
        )
        detector.eval()
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
