from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import sys
import time
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
from mmdet.apis import inference_detector, init_detector
from mmengine.runner import Runner
from mmengine.runner.checkpoint import load_checkpoint

from .. import training_state as _training_state  # noqa: F401
from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..modeling import finite_state_dict_audit
from ..registry import Registry
from ..runtime_config import build_runtime_config, register_framework
from .formal_parallel import normalize_execution_devices, validate_available_cuda_devices
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
    rename_directory_no_replace,
)


PROTOCOL_ID = "coco_retained500_victim_training_state_transfer"
STATE_ORDER = ["standard_control", "adversarial_training"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve_inside(root: Path, value: Any) -> Path:
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise RuntimeError("Artifact path escapes the repository: {}".format(value))
    return resolved


def _git_provenance(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_head": head,
        "worktree_dirty": status is None or bool(status),
        "captured_at_utc": _utc_now(),
    }


def build_pair_training_plan(
    registry: Registry,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    spec = deepcopy(registry.training_state_transfer)
    training = dict(spec["pair_training"])
    dataset = registry.dataset("coco")
    selected_images = (
        int(max_images)
        if max_images is not None
        else int(dataset.split("train").expected_images)
    )
    expected_images = int(dataset.split("train").expected_images)
    if selected_images <= 0 or selected_images > expected_images:
        raise ValueError(
            "max_images must lie in [1, {}]".format(expected_images)
        )
    outer_batches = int(math.ceil(selected_images / int(training["batch_size"])))
    optimizer_steps = outer_batches * int(
        training["replay_steps_per_minibatch"]
    ) * int(training["outer_epochs"])
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "dataset": "coco",
        "train_split": "train",
        "victim_model": str(spec["victim_model"]),
        "states": list(STATE_ORDER),
        "initial_checkpoint": str(training["initial_checkpoint"]),
        "training": training,
        "selected_train_images": selected_images,
        "diagnostic_max_images": max_images,
        "expected_outer_minibatches_per_state": outer_batches,
        "expected_optimizer_steps_per_state": optimizer_steps,
        "formal_pair_requested": max_images is None,
    }


def _dataset_order_images(index: CocoIndex) -> Sequence[Mapping[str, Any]]:
    images = index.payload.get("images")
    if not isinstance(images, list) or len(images) != len(index.images):
        raise RuntimeError("COCO annotation data-list order is invalid")
    image_ids = [int(image["id"]) for image in images]
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("COCO annotation data-list contains duplicate image IDs")
    return images


def _required_empty_gt_audit(
    index: CocoIndex, pair_training: Mapping[str, Any]
) -> Dict[str, Any]:
    diagnostic = pair_training["diagnostic_smoke"]
    position = int(diagnostic["required_empty_gt_dataset_index"])
    required_id = int(diagnostic["required_empty_gt_image_id"])
    dataset_order = _dataset_order_images(index)
    if position < 0 or position >= len(dataset_order):
        raise RuntimeError("Frozen empty-GT dataset index is out of range")
    observed_id = int(dataset_order[position]["id"])
    annotations = index.annotations(observed_id)
    earlier_empty = [
        int(image["id"])
        for image in dataset_order[:position]
        if not index.annotations(int(image["id"]))
    ]
    if observed_id != required_id or annotations or earlier_empty:
        raise RuntimeError(
            "Frozen diagnostic empty-GT sample does not match the registered policy"
        )
    return {
        "dataset_image_index": position,
        "image_id": observed_id,
        "annotation_count": 0,
        "covered_by_recommended_smoke": int(diagnostic["max_images"])
        > position,
    }


def _dataset_leaf(node: Any) -> Optional[Dict[str, Any]]:
    current = node
    while isinstance(current, Mapping) and isinstance(current.get("dataset"), Mapping):
        current = current["dataset"]
    return current if isinstance(current, dict) else None


def _training_config(
    registry: Registry,
    state: str,
    branch_dir: Path,
    initial_checkpoint: Path,
    max_images: Optional[int],
) -> Any:
    spec = registry.training_state_transfer
    recipe = dict(spec["pair_training"])
    model = registry.model(str(spec["victim_model"]))
    epochs = int(recipe["outer_epochs"])
    model = replace(
        model,
        finetune_epochs={**dict(model.finetune_epochs), "coco": epochs},
        train_batch_size=int(recipe["batch_size"]),
        train_base_batch_size=int(recipe["batch_size"]),
    )
    cfg = build_runtime_config(
        model,
        registry.dataset("coco"),
        branch_dir,
        mode="train",
        test_split="val",
        training_protocol={
            **dict(registry.training_protocol),
            "seed": int(spec["seed"]),
            "amp": False,
            "checkpoint_interval_epochs": epochs,
            "max_keep_checkpoints": 1,
        },
        training_safety=registry.training_safety,
        dump_config=False,
    )
    cfg.load_from = str(initial_checkpoint)
    cfg.resume = False
    cfg.model["type"] = "LGPTrainingStateFasterRCNN"
    cfg.model["training_state_mode"] = state
    cfg.model["replay_steps_per_minibatch"] = int(
        recipe["replay_steps_per_minibatch"]
    )
    adversarial = dict(recipe["adversarial_branch"])
    cfg.model["epsilon_pixel_255"] = float(adversarial["epsilon_pixel_255"])
    cfg.model["step_size_pixel_255"] = float(
        adversarial["step_size_pixel_255"]
    )
    cfg.model["padding_policy"] = str(adversarial["padding_policy"])
    cfg.custom_imports = {
        "imports": ["lgp.training_hooks", "lgp.training_state"],
        "allow_failed_imports": False,
    }
    optimizer = dict(recipe["optimizer"])
    cfg.optim_wrapper = {
        "type": "OptimWrapper",
        "optimizer": {
            "type": str(optimizer["type"]),
            "lr": float(optimizer["lr"]),
            "betas": tuple(float(value) for value in optimizer["betas"]),
            "weight_decay": float(optimizer["weight_decay"]),
        },
        "paramwise_cfg": {
            "norm_decay_mult": float(optimizer["norm_decay_multiplier"]),
            "custom_keys": {
                "backbone": {
                    "lr_mult": float(optimizer["backbone_lr_multiplier"]),
                    "decay_mult": 1.0,
                }
            },
            "bypass_duplicate": True,
        },
        "clip_grad": {
            "max_norm": float(optimizer["gradient_clip_max_norm"]),
            "norm_type": int(optimizer["gradient_clip_norm_type"]),
        },
    }
    cfg.param_scheduler = []
    cfg.auto_scale_lr = {
        "enable": False,
        "base_batch_size": int(recipe["batch_size"]),
    }
    cfg.train_cfg["max_epochs"] = epochs
    cfg.train_cfg["val_interval"] = epochs + 1
    cfg.train_dataloader["batch_size"] = int(recipe["batch_size"])
    cfg.train_dataloader["num_workers"] = int(recipe["num_workers"])
    cfg.train_dataloader["persistent_workers"] = False
    leaf = _dataset_leaf(cfg.train_dataloader["dataset"])
    if leaf is None:
        raise RuntimeError("Could not locate the COCO training dataset node")
    # The formal denominator is every one of the 118,287 registered
    # train2017 images. MMDetection's stock COCO config filters empty-GT
    # images, so apply the declarative protocol policy rather than silently
    # training on a smaller, version-dependent subset.
    leaf["filter_cfg"] = deepcopy(recipe["dataset_filter"])
    if max_images is not None:
        leaf["indices"] = list(range(int(max_images)))
    cfg.default_hooks["checkpoint"].update(
        interval=epochs,
        max_keep_ckpts=1,
        save_last=True,
    )
    cfg.default_hooks["checkpoint"].pop("save_best", None)
    cfg.default_hooks["logger"]["interval"] = 50
    hooks = list(cfg.get("custom_hooks") or [])
    hooks.append(
        {
            "type": "LGPTrainingStateProgressHook",
            "output": str((branch_dir / "training_state.json").resolve()),
            "interval": 50,
        }
    )
    cfg.custom_hooks = hooks
    cfg.lgp_training_state_pair = {
        "protocol": PROTOCOL_ID,
        "state": state,
        "pair_recipe": deepcopy(recipe),
        "formal_val_during_training": False,
        "checkpoint_selection": "fixed_final_epoch",
    }
    cfg.dump(str(branch_dir / "runtime_config.py"))
    return cfg


def _checkpoint_payload(path: Path) -> Any:
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise RuntimeError("Checkpoint root is not a mapping")
    state = payload.get("state_dict", payload.get("model"))
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Checkpoint state_dict is missing")
    return payload, state


def _state_schema(state: Mapping[str, Any]) -> Dict[str, Any]:
    rows = []
    for key, value in sorted(state.items()):
        normalized = key[7:] if str(key).startswith("module.") else str(key)
        rows.append(
            {
                "key": normalized,
                "shape": list(value.shape) if torch.is_tensor(value) else None,
                "dtype": str(value.dtype) if torch.is_tensor(value) else type(value).__name__,
            }
        )
    return {
        "entries": len(rows),
        "sha256": _canonical_digest(rows),
    }


def _export_model_only(source: Path, destination: Path, metadata: Mapping[str, Any]) -> None:
    payload, state = _checkpoint_payload(source)
    finite_state_dict_audit(state)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError("Stale checkpoint export exists: {}".format(temporary))
    torch.save(
        {"meta": dict(metadata), "state_dict": dict(state)},
        str(temporary),
    )
    check, check_state = _checkpoint_payload(temporary)
    try:
        finite_state_dict_audit(check_state)
        if _state_schema(check_state) != _state_schema(state):
            raise RuntimeError("Model-only export changed the state schema")
    finally:
        del check
        del check_state
        del payload
        del state
        gc.collect()
    os.replace(str(temporary), str(destination))


def _strict_verify(
    registry: Registry,
    checkpoint: Path,
    branch_dir: Path,
    device: str,
) -> Dict[str, Any]:
    model = registry.model(str(registry.training_state_transfer["victim_model"]))
    dataset = registry.dataset("coco")
    cfg = build_runtime_config(
        model,
        dataset,
        branch_dir / "strict_verification_model",
        mode="test",
        test_split="train",
        dump_config=False,
    )
    register_framework(model.framework)
    detector = init_detector(cfg, str(checkpoint), device=device)
    load_checkpoint(detector, str(checkpoint), map_location="cpu", strict=True)
    detector.eval()
    index = CocoIndex(dataset, "train")
    checked = []
    detections = 0
    for image in index.images[:2]:
        sample = inference_detector(detector, str(index.image_path(image)))
        instances = sample.pred_instances.to("cpu")
        if instances.labels.numel():
            minimum = int(instances.labels.min().item())
            maximum = int(instances.labels.max().item())
            if minimum < 0 or maximum >= dataset.num_classes:
                raise RuntimeError("Training verification produced an invalid label")
        detections += int(instances.bboxes.shape[0])
        checked.append(int(image["id"]))
    del detector
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "strict_state_dict_load": True,
        "train_inference_verified": True,
        "train_inference_image_ids": checked,
        "train_inference_detections": detections,
    }


def _branch_worker(
    root_value: str,
    state: str,
    branch_dir_value: str,
    initial_checkpoint_value: str,
    device: str,
    max_images: Optional[int],
    coordinator: Mapping[str, Any],
) -> None:
    branch_dir = Path(branch_dir_value)
    binding = install_branch_scope(coordinator)
    try:
        if not str(device).startswith("cuda:"):
            raise ValueError("Pair training requires an explicit CUDA device")
        torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        registry = Registry(Path(root_value))
        initial = Path(initial_checkpoint_value)
        index = CocoIndex(registry.dataset("coco"), "train")
        empty_gt_audit = _required_empty_gt_audit(
            index, registry.training_state_transfer["pair_training"]
        )
        cfg = _training_config(
            registry,
            state,
            branch_dir,
            initial,
            max_images,
        )
        atomic_json(
            branch_dir / "branch_execution.json",
            {
                "schema_version": 2,
                "status": "running",
                "state": state,
                "device": device,
                "started_at_utc": _utc_now(),
                **binding,
            },
        )
        Runner.from_cfg(cfg).train()
        expected_epoch = int(
            registry.training_state_transfer["pair_training"]["outer_epochs"]
        )
        source = branch_dir / "epoch_{}.pth".format(expected_epoch)
        if not source.is_file():
            raise FileNotFoundError("Fixed final-epoch checkpoint is missing")
        source_sha256 = file_digest(source)
        source_bytes = source.stat().st_size
        candidate = branch_dir / "candidate_model_only.pth"
        _export_model_only(
            source,
            candidate,
            {
                "epoch": expected_epoch,
                "dataset_meta": {
                    "classes": tuple(registry.dataset("coco").classes)
                },
                "protocol": PROTOCOL_ID,
                "training_state": state,
            },
        )
        verification = _strict_verify(
            registry, candidate, branch_dir, device
        )
        progress = json.loads(
            (branch_dir / "training_state.json").read_text(encoding="utf-8")
        )
        plan = build_pair_training_plan(registry, max_images=max_images)
        dataset_order = _dataset_order_images(index)
        selected = (
            dataset_order[: int(max_images)]
            if max_images is not None
            else dataset_order
        )
        image_ids = [int(image["id"]) for image in selected]
        train_image_id_set_sha256 = _canonical_digest(sorted(image_ids))
        expected_progress = {
            "samples": int(plan["selected_train_images"]),
            "unique_samples": int(plan["selected_train_images"]),
            "duplicate_samples": 0,
            "outer_minibatches": int(
                plan["expected_outer_minibatches_per_state"]
            ),
            "optimizer_steps": int(plan["expected_optimizer_steps_per_state"]),
        }
        actual_progress = {
            key: int(progress[key]) for key in expected_progress
        }
        if actual_progress != expected_progress:
            raise RuntimeError(
                "Training denominator mismatch: expected {}, got {}".format(
                    expected_progress, actual_progress
                )
            )
        if progress.get("processed_unique_image_ids_sha256") != (
            train_image_id_set_sha256
        ):
            raise RuntimeError(
                "Processed unique image-ID set does not match the selected denominator"
            )
        expected_padding_policy = str(
            registry.training_state_transfer["pair_training"][
                "adversarial_branch"
            ]["padding_policy"]
        )
        if progress.get("padding_policy") != expected_padding_policy:
            raise RuntimeError("Training padding policy audit mismatch")
        payload, state_dict = _checkpoint_payload(candidate)
        finite = finite_state_dict_audit(state_dict)
        architecture = _state_schema(state_dict)
        del payload
        del state_dict
        gc.collect()
        recipe = dict(registry.training_state_transfer["pair_training"])
        manifest = {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "status": "complete",
            "training_state": state,
            "model": str(registry.training_state_transfer["victim_model"]),
            "dataset": "coco",
            "train_split": "train",
            "formal_val_during_training": False,
            "checkpoint_selection": "fixed_final_epoch",
            "final_epoch": expected_epoch,
            "diagnostic_max_images": max_images,
            "initial_checkpoint": _relative(initial, registry.root),
            "initial_checkpoint_sha256": file_digest(initial),
            "checkpoint": _relative(candidate, registry.root),
            "checkpoint_sha256": file_digest(candidate),
            "checkpoint_bytes": candidate.stat().st_size,
            "source_training_checkpoint_sha256": source_sha256,
            "source_training_checkpoint_bytes": source_bytes,
            "source_training_checkpoint_removed_after_export": True,
            "runtime_config": _relative(branch_dir / "runtime_config.py", registry.root),
            "runtime_config_sha256": file_digest(branch_dir / "runtime_config.py"),
            "train_annotation_sha256": file_digest(index.annotation_path),
            "train_image_ids_sha256": _canonical_digest(image_ids),
            "train_image_id_set_sha256": train_image_id_set_sha256,
            "required_empty_gt_audit": empty_gt_audit,
            "train_images": len(image_ids),
            "expected_processed_samples": expected_progress["samples"],
            "expected_unique_samples": expected_progress["unique_samples"],
            "expected_duplicate_samples": expected_progress["duplicate_samples"],
            "expected_outer_minibatches": expected_progress[
                "outer_minibatches"
            ],
            "expected_optimizer_steps": expected_progress["optimizer_steps"],
            "processed_sample_trace_sha256": progress[
                "processed_sample_trace_sha256"
            ],
            "processed_samples": int(progress["samples"]),
            "unique_samples": int(progress["unique_samples"]),
            "duplicate_samples": int(progress["duplicate_samples"]),
            "processed_unique_image_ids_sha256": progress[
                "processed_unique_image_ids_sha256"
            ],
            "outer_minibatches": int(progress["outer_minibatches"]),
            "optimizer_steps": int(progress["optimizer_steps"]),
            "padding_policy": expected_padding_policy,
            "model_architecture_sha256": architecture["sha256"],
            "model_state_entries": architecture["entries"],
            "weight_optimizer_sha256": _canonical_digest(recipe["optimizer"]),
            "schedule_sha256": _canonical_digest(
                {
                    "scheduler": recipe["scheduler"],
                    "outer_epochs": recipe["outer_epochs"],
                    "replay_steps_per_minibatch": recipe[
                        "replay_steps_per_minibatch"
                    ],
                }
            ),
            "pair_common_recipe_sha256": _canonical_digest(
                {
                    key: value
                    for key, value in recipe.items()
                    if key != "adversarial_branch"
                }
            ),
            "state_dict_finite": finite["all_finite"],
            **verification,
            "completed_at_utc": _utc_now(),
        }
        atomic_json(branch_dir / "branch_manifest.json", manifest)
        # The optimizer-bearing checkpoint has no remaining runtime or
        # evidentiary role after the model-only hash, strict load and manifest
        # are committed.  Remove only this exact verified file.
        source.unlink()
        atomic_json(
            branch_dir / "branch_execution.json",
            {
                "schema_version": 2,
                "status": "complete",
                "state": state,
                "device": device,
                "completed_at_utc": _utc_now(),
                "manifest_sha256": file_digest(branch_dir / "branch_manifest.json"),
                **binding,
            },
        )
    except BaseException as exc:
        atomic_json(
            branch_dir / "branch_execution.json",
            {
                "schema_version": 2,
                "status": "failed",
                "state": state,
                "device": device,
                "reason": "{}: {}".format(type(exc).__name__, exc),
                "failed_at_utc": _utc_now(),
                **binding,
            },
        )
        raise


def qualify_checkpoint_pair(
    registry: Registry,
    pair_manifest_path: Optional[Path] = None,
    minimum_checkpoint_bytes: int = 1 << 20,
) -> Dict[str, Any]:
    configured = registry.training_state_transfer["artifacts"]["pair_manifest"]
    path = (
        pair_manifest_path.resolve()
        if pair_manifest_path is not None
        else _resolve_inside(registry.root, configured)
    )
    report: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "pair_manifest": _relative(path, registry.root),
        "checked_at_utc": _utc_now(),
        "strict_training_controlled_pair_eligible": False,
        "issues": [],
    }
    if not path.is_file():
        report["issues"].append("pair manifest is missing")
        return report
    try:
        pair = json.loads(path.read_text(encoding="utf-8"))
        if pair.get("protocol") != PROTOCOL_ID:
            raise RuntimeError("pair manifest protocol mismatch")
        if list(pair.get("states", [])) != STATE_ORDER:
            raise RuntimeError("pair manifest state order mismatch")
        manifests = {}
        for state in STATE_ORDER:
            manifest_path = _resolve_inside(registry.root, pair["manifests"][state])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") != "complete":
                raise RuntimeError("{} training manifest is incomplete".format(state))
            if manifest.get("training_state") != state:
                raise RuntimeError("{} training-state label mismatch".format(state))
            checkpoint = _resolve_inside(registry.root, manifest["checkpoint"])
            if (
                not checkpoint.is_file()
                or checkpoint.stat().st_size <= int(minimum_checkpoint_bytes)
            ):
                raise RuntimeError("{} checkpoint is missing or undersized".format(state))
            if file_digest(checkpoint) != manifest.get("checkpoint_sha256"):
                raise RuntimeError("{} checkpoint hash mismatch".format(state))
            for key in (
                "strict_state_dict_load",
                "train_inference_verified",
                "state_dict_finite",
            ):
                if manifest.get(key) is not True:
                    raise RuntimeError("{} lacks {}".format(state, key))
            denominators = {
                "processed_samples": "expected_processed_samples",
                "unique_samples": "expected_unique_samples",
                "duplicate_samples": "expected_duplicate_samples",
                "outer_minibatches": "expected_outer_minibatches",
                "optimizer_steps": "expected_optimizer_steps",
            }
            for actual_key, expected_key in denominators.items():
                if int(manifest.get(actual_key, -1)) != int(
                    manifest.get(expected_key, -2)
                ):
                    raise RuntimeError(
                        "{} {} does not match {}".format(
                            state, actual_key, expected_key
                        )
                    )
            diagnostic_max_images = manifest.get("diagnostic_max_images")
            required_train_images = (
                118287
                if diagnostic_max_images is None
                else int(diagnostic_max_images)
            )
            if int(manifest.get("train_images", -1)) != required_train_images:
                raise RuntimeError(
                    "{} train-image denominator is invalid".format(state)
                )
            if int(manifest.get("expected_processed_samples", -1)) != int(
                manifest.get("train_images", -2)
            ) or int(manifest.get("expected_unique_samples", -1)) != int(
                manifest.get("train_images", -2)
            ):
                raise RuntimeError(
                    "{} expected sample coverage is inconsistent".format(state)
                )
            if int(manifest.get("expected_duplicate_samples", -1)) != 0:
                raise RuntimeError(
                    "{} expected duplicate count must be zero".format(state)
                )
            if manifest.get("processed_unique_image_ids_sha256") != manifest.get(
                "train_image_id_set_sha256"
            ):
                raise RuntimeError(
                    "{} processed unique image-ID set is invalid".format(state)
                )
            if manifest.get("padding_policy") != "zero_outside_img_shape":
                raise RuntimeError(
                    "{} adversarial padding policy is invalid".format(state)
                )
            if manifest.get("required_empty_gt_audit") != {
                "dataset_image_index": 46,
                "image_id": 262284,
                "annotation_count": 0,
                "covered_by_recommended_smoke": True,
            }:
                raise RuntimeError(
                    "{} frozen empty-GT audit is invalid".format(state)
                )
            if manifest.get("formal_val_during_training") is not False:
                raise RuntimeError("{} used formal val during training".format(state))
            if manifest.get("checkpoint_selection") != "fixed_final_epoch":
                raise RuntimeError("{} did not use fixed final epoch".format(state))
            manifests[state] = manifest
        standard = manifests["standard_control"]
        adversarial = manifests["adversarial_training"]
        shared = (
            "initial_checkpoint_sha256",
            "train_annotation_sha256",
            "train_image_ids_sha256",
            "train_image_id_set_sha256",
            "required_empty_gt_audit",
            "processed_sample_trace_sha256",
            "processed_unique_image_ids_sha256",
            "train_images",
            "expected_processed_samples",
            "expected_unique_samples",
            "expected_duplicate_samples",
            "expected_outer_minibatches",
            "expected_optimizer_steps",
            "processed_samples",
            "unique_samples",
            "duplicate_samples",
            "outer_minibatches",
            "optimizer_steps",
            "padding_policy",
            "model_architecture_sha256",
            "model_state_entries",
            "weight_optimizer_sha256",
            "schedule_sha256",
            "pair_common_recipe_sha256",
            "final_epoch",
            "diagnostic_max_images",
        )
        mismatches = [
            key for key in shared if standard.get(key) != adversarial.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "paired training invariants differ: {}".format(", ".join(mismatches))
            )
        if standard["checkpoint_sha256"] == adversarial["checkpoint_sha256"]:
            raise RuntimeError("training-state checkpoints are byte-identical")
        formal = standard.get("diagnostic_max_images") is None
        report.update(
            status="qualified" if formal else "diagnostic_qualified",
            strict_training_controlled_pair_eligible=bool(formal),
            diagnostic_pair_eligible=True,
            state_manifests={
                state: _relative(
                    _resolve_inside(registry.root, pair["manifests"][state]),
                    registry.root,
                )
                for state in STATE_ORDER
            },
            checkpoint_sha256={
                state: manifests[state]["checkpoint_sha256"] for state in STATE_ORDER
            },
            paired_invariants={key: standard.get(key) for key in shared},
        )
    except Exception as exc:
        report["issues"].append("{}: {}".format(type(exc).__name__, exc))
        report["status"] = "rejected"
    return report


def _copy_atomic(source: Path, destination: Path, expected_sha256: Optional[str] = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Refusing to overwrite checkpoint: {}".format(destination))
    expected_sha256 = expected_sha256 or file_digest(source)
    if file_digest(source) != expected_sha256:
        raise RuntimeError("Source artifact changed before copy")
    with tempfile.NamedTemporaryFile(
        prefix=".{}-".format(destination.name),
        suffix=".part",
        dir=str(destination.parent),
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(str(source), str(temporary))
        if file_digest(temporary) != expected_sha256 or file_digest(source) != expected_sha256:
            raise RuntimeError("Checkpoint copy hash mismatch")
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        os.link(str(temporary), str(destination))
    finally:
        if temporary.exists():
            temporary.unlink()


def _branch_command(registry, state, branch, initial, device, max_images, coordinator):
    payload = {
        "root_value": str(registry.root), "state": state, "branch_dir_value": str(branch),
        "initial_checkpoint_value": str(initial), "device": device,
        "max_images": max_images, "coordinator": coordinator,
    }
    return [sys.executable, "-B", "-c", (
        "import json,sys; from lgp.runners.training_state_pair import _branch_worker; "
        "_branch_worker(**json.loads(sys.argv[1]))"
    ), json.dumps(payload)]


def _verify_branch_terminal(registry, state, branch, device, entry, coordinator):
    terminal_path = branch / "branch_execution.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete", "state": state, "device": device,
        "worker_pid": entry["pid"], "worker_start_ticks": entry["start_ticks"],
        "process_group": entry["pid"], "coordinator_pid": coordinator["pid"],
        "coordinator_start_ticks": coordinator["start_ticks"],
    }
    if any(terminal.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Training branch terminal ownership or completion mismatch: " + state)
    manifest_path = branch / "branch_manifest.json"
    if file_digest(manifest_path) != terminal.get("manifest_sha256"):
        raise RuntimeError("Training branch manifest differs from its terminal receipt")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("training_state") != state:
        raise RuntimeError("Training branch manifest is incomplete or mislabelled")
    for key, filename in (("checkpoint", "candidate_model_only.pth"), ("runtime_config", "runtime_config.py")):
        artifact = _resolve_inside(registry.root, manifest[key])
        if artifact != (branch / filename).resolve() or file_digest(artifact) != manifest[key + "_sha256"]:
            raise RuntimeError("Training branch artifact binding mismatch: " + key)
    return {"terminal_sha256": file_digest(terminal_path), "manifest_sha256": file_digest(manifest_path)}


def _run_pair_branches(registry, run_dir, branches, initial, devices, max_images,
                       execution_state, publish_state):
    coordinator = process_identity(os.getpid())
    jobs = execution_state["workers"]
    active = {}
    next_index = 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(registry.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    try:
        while next_index < len(STATE_ORDER) or active:
            while next_index < len(STATE_ORDER) and len(active) < len(devices):
                state = STATE_ORDER[next_index]
                device = devices[next_index % len(devices)]
                next_index += 1
                with (branches[state] / "stdout.log").open("wb") as log:
                    process = subprocess.Popen(
                        _branch_command(registry, state, branches[state], initial, device, max_images, coordinator),
                        cwd=str(registry.root), env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                identity = {"pid": process.pid, "parent_pid": coordinator["pid"],
                            "process_group": process.pid, "session_id": process.pid,
                            "start_ticks": None}
                active[state] = (process, identity)
                identity.update(process_identity(process.pid))
                if identity["parent_pid"] != coordinator["pid"] or identity["process_group"] != identity["session_id"] or identity["session_id"] != process.pid:
                    raise RuntimeError("Training branch process ownership mismatch")
                jobs[state].update(status="running", process_state=identity["state"],
                                   **{key: value for key, value in identity.items() if key != "state"})
                atomic_json(branches[state] / "launch.json", {
                    **identity, "state": state, "device": device,
                    "coordinator_pid": coordinator["pid"],
                    "coordinator_start_ticks": coordinator["start_ticks"],
                })
                publish_state()
            finished = []
            for state, (process, identity) in list(active.items()):
                progress_path = branches[state] / "training_state.json"
                if progress_path.is_file():
                    jobs[state]["training_progress"] = json.loads(progress_path.read_text(encoding="utf-8"))
                exit_code = peek_exit_code(process)
                if exit_code is None:
                    terminal_path = branches[state] / "branch_execution.json"
                    if terminal_path.is_file() and json.loads(terminal_path.read_text(encoding="utf-8")).get("status") == "failed":
                        raise RuntimeError("Training branch reported failure: " + state)
                    continue
                jobs[state]["exit_code"] = exit_code
                if exit_code != 0:
                    raise RuntimeError("Training branch {} exited with {}".format(state, exit_code))
                jobs[state].update(_verify_branch_terminal(
                    registry, state, branches[state], jobs[state]["device"], identity, coordinator,
                ))
                jobs[state]["cleanup"] = close_branch_process(process, identity)
                jobs[state]["status"] = "complete"
                finished.append(state)
            for state in finished:
                del active[state]
            publish_state()
            if active:
                time.sleep(0.2)
    finally:
        cleanup_errors = []
        for state, (process, identity) in active.items():
            if process.returncode is not None and jobs[state]["status"] == "complete":
                continue
            try:
                jobs[state]["cleanup"] = close_branch_process(process, identity)
            except Exception as exc:
                cleanup_errors.append("{}: {}".format(state, exc))
            jobs[state]["status"] = "failed"
            jobs[state]["reason"] = "Training branch did not reach an accepted terminal state"
        publish_state()
        if cleanup_errors:
            raise RuntimeError("Training cleanup could not be verified: " + "; ".join(cleanup_errors))


def _default_output(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "outputs" / "training" / "coco_training_state_pair" / stamp


def _verify_bound_files(root, hashes):
    for name, expected in hashes.items():
        if file_digest(_resolve_inside(root, name)) != expected:
            raise RuntimeError("Publication input changed: " + name)


def verify_pair_publication(registry, qualification):
    commit = qualification.get("publication_commit")
    if commit is None:
        if int(qualification.get("schema_version", 1)) >= 2:
            raise RuntimeError("New training-pair qualification lacks its publication commit")
        return
    if commit.get("schema_version") != 1:
        raise RuntimeError("Unknown training-pair publication contract")
    try:
        index = _publication_path(registry.root, commit["index"])
    except FileExistsError as exc:
        raise RuntimeError("Training-pair publication index is missing or changed") from exc
    if not index.is_file() or file_digest(index) != commit["index_sha256"]:
        raise RuntimeError("Training-pair publication index is missing or changed")
    _verify_bound_files(registry.root, commit["branch_manifest_sha256"])


def _publication_committed(registry, run_dir):
    path = run_dir / "publication_transaction.json"
    if not path.is_file():
        return False
    transaction = json.loads(path.read_text(encoding="utf-8"))
    try:
        index = _publication_path(registry.root, transaction["index"])
    except (OSError, RuntimeError):
        return False
    return index.is_file() and file_digest(index) == transaction["index_sha256"]


def _publication_path(root, value):
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    if ".." in path.parts:
        raise RuntimeError("Publication paths cannot contain parent traversal")
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise FileExistsError("Publication paths cannot traverse existing symbolic links")
    return _resolve_inside(root, path)


def _publish_pair(registry, run_dir, branches, candidate_pair):
    artifacts = registry.training_state_transfer["artifacts"]
    index = _publication_path(registry.root, artifacts["pair_manifest"])
    destination = index.parent
    checkpoints = {"standard_control": _publication_path(registry.root, artifacts["standard_checkpoint"]),
                   "adversarial_training": _publication_path(registry.root, artifacts["adversarial_checkpoint"])}
    if any(path.parent != destination for path in checkpoints.values()) or len({index, *checkpoints.values()}) != 3:
        raise RuntimeError("Training-pair publication paths are not a unique single-directory layout")
    names = [index.name, "pair_qualification.json", *(path.name for path in checkpoints.values()),
             *(state + suffix for state in STATE_ORDER for suffix in (".runtime_config.py", ".training_manifest.json"))]
    if len(set(names)) != len(names):
        raise RuntimeError("Training-pair publication artifact names collide")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Canonical pair publication directory must not already exist")
    stage = run_dir / "publication_stage"
    if destination == stage or destination in stage.parents or stage in destination.parents:
        raise RuntimeError("Training-pair stage and destination overlap")
    stage.mkdir()
    original_hashes = {}
    stage_hashes = {}
    manifests = {}
    for state in STATE_ORDER:
        manifest_path = branches[state] / "branch_manifest.json"
        original_hashes[_relative(manifest_path, registry.root)] = file_digest(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests[state] = manifest
        for key, basename in (("checkpoint", checkpoints[state].name),
                              ("runtime_config", state + ".runtime_config.py")):
            source = _resolve_inside(registry.root, manifest[key])
            original_hashes[_relative(source, registry.root)] = manifest[key + "_sha256"]
            target = stage / basename
            _copy_atomic(source, target, manifest[key + "_sha256"])
            stage_hashes[_relative(target, registry.root)] = manifest[key + "_sha256"]
        staged_manifest = {**manifest, "checkpoint": _relative(stage / checkpoints[state].name, registry.root),
                           "runtime_config": _relative(stage / (state + ".runtime_config.py"), registry.root)}
        staged_manifest_path = stage / (state + ".training_manifest.json")
        atomic_json(staged_manifest_path, staged_manifest)
        stage_hashes[_relative(staged_manifest_path, registry.root)] = file_digest(staged_manifest_path)
    _verify_bound_files(registry.root, original_hashes)
    staged_pair_path = run_dir / "staged_pair_manifest.json"
    staged_pair = {**candidate_pair, "manifests": {
        state: _relative(stage / (state + ".training_manifest.json"), registry.root) for state in STATE_ORDER}}
    atomic_json(staged_pair_path, staged_pair)
    stage_hashes[_relative(staged_pair_path, registry.root)] = file_digest(staged_pair_path)
    staged_qualification = qualify_checkpoint_pair(registry, staged_pair_path)
    atomic_json(run_dir / "staged_pair_qualification.json", staged_qualification)
    if staged_qualification.get("strict_training_controlled_pair_eligible") is not True:
        raise RuntimeError("Staged checkpoint pair failed qualification")
    _verify_bound_files(registry.root, original_hashes)
    _verify_bound_files(registry.root, stage_hashes)
    final_manifest_hashes = {}
    for state in STATE_ORDER:
        published_manifest = destination / (state + ".training_manifest.json")
        payload = {**manifests[state], "checkpoint": _relative(checkpoints[state], registry.root),
                   "runtime_config": _relative(destination / (state + ".runtime_config.py"), registry.root)}
        atomic_json(stage / published_manifest.name, payload)
        final_manifest_hashes[_relative(published_manifest, registry.root)] = file_digest(stage / published_manifest.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rename_directory_no_replace(stage, destination)
    final_hashes = {**final_manifest_hashes}
    for state in STATE_ORDER:
        final_hashes[_relative(checkpoints[state], registry.root)] = manifests[state]["checkpoint_sha256"]
        final_hashes[_relative(destination / (state + ".runtime_config.py"), registry.root)] = manifests[state]["runtime_config_sha256"]
    _verify_bound_files(registry.root, final_hashes)
    final_pair = {**candidate_pair, "status": "qualified", "created_at_utc": _utc_now(),
                  "manifests": {state: _relative(destination / (state + ".training_manifest.json"), registry.root)
                                for state in STATE_ORDER}}
    draft = run_dir / "publication_final_pair.json"
    atomic_json(draft, final_pair)
    draft_sha256 = file_digest(draft)
    qualification = qualify_checkpoint_pair(registry, draft)
    if qualification.get("strict_training_controlled_pair_eligible") is not True:
        raise RuntimeError("Final copied checkpoint pair failed qualification")
    _verify_bound_files(registry.root, {**original_hashes, **final_hashes,
                                       _relative(draft, registry.root): draft_sha256})
    commit = {"schema_version": 1, "index": _relative(index, registry.root),
              "index_sha256": draft_sha256, "branch_manifest_sha256": final_manifest_hashes}
    qualification = {**qualification, "schema_version": 2, "publication_commit": commit}
    atomic_json(run_dir / "publication_transaction.json", commit)
    qualification_path = run_dir / "publication_qualification.json"
    atomic_json(qualification_path, qualification)
    _copy_atomic(qualification_path, destination / "pair_qualification.json", file_digest(qualification_path))
    _verify_bound_files(registry.root, {**original_hashes, **final_hashes,
                                       _relative(draft, registry.root): draft_sha256})
    # This no-replace, byte-identical index copy is the publication commit point.
    _copy_atomic(draft, index, draft_sha256)
    return qualification


def _execute_pair_training(registry, run_dir, plan, provenance, selected_devices,
                           max_images, execution_state, publish_state):
    if plan["formal_pair_requested"] and (
        provenance["worktree_dirty"] or provenance["git_head"] is None
    ):
        raise RuntimeError("Formal pair training requires a clean Git worktree")
    initial = _resolve_inside(registry.root, plan["initial_checkpoint"])
    if not initial.is_file() or initial.stat().st_size <= 1 << 20:
        raise FileNotFoundError("Initial detector checkpoint is missing")
    index = CocoIndex(registry.dataset("coco"), "train")
    _required_empty_gt_audit(
        index, registry.training_state_transfer["pair_training"]
    )
    if max_images is None and len(index.images) != 118287:
        raise RuntimeError("Formal COCO train split is not exactly 118,287 images")
    branches = {state: run_dir / "branches" / state for state in STATE_ORDER}
    for branch in branches.values():
        branch.mkdir(parents=True, exist_ok=False)
    _run_pair_branches(registry, run_dir, branches, initial, selected_devices,
                       max_images, execution_state, publish_state)

    candidate_pair = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "status": "candidate",
        "states": list(STATE_ORDER),
        "manifests": {
            state: _relative(branches[state] / "branch_manifest.json", registry.root)
            for state in STATE_ORDER
        },
    }
    candidate_path = run_dir / "candidate_pair_manifest.json"
    atomic_json(candidate_path, candidate_pair)
    candidate_qualification = qualify_checkpoint_pair(
        registry, candidate_path
    )
    atomic_json(run_dir / "candidate_pair_qualification.json", candidate_qualification)
    eligible_key = (
        "strict_training_controlled_pair_eligible"
        if max_images is None
        else "diagnostic_pair_eligible"
    )
    if candidate_qualification.get(eligible_key) is not True:
        raise RuntimeError("Candidate checkpoint pair failed qualification")

    return (_publish_pair(registry, run_dir, branches, candidate_pair)
            if max_images is None else candidate_qualification)


def run_pair_training(registry: Registry, execute: bool = True,
                      output_dir: Optional[Path] = None,
                      devices: Optional[Sequence[str]] = None,
                      max_images: Optional[int] = None) -> Path:
    selected_devices = normalize_execution_devices("cuda:0", ["cuda:0", "cuda:1"] if devices is None else devices)
    if len(selected_devices) not in {1, 2} or any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) for value in selected_devices):
        raise ValueError("Pair training requires one or two distinct explicit CUDA devices")
    plan = build_pair_training_plan(registry, max_images=max_images)
    if execute:
        if sys.platform != "linux":
            raise RuntimeError("Pair training executes only on the Linux ODA server")
        validate_available_cuda_devices(selected_devices, include_single=True)
    run_dir = (output_dir or _default_output(registry.root)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty training run")
    run_dir.mkdir(parents=True, exist_ok=True)
    assignments = [{"state": state, "device": selected_devices[index % len(selected_devices)],
                    "canonical_index": index, "worker_slot": index % len(selected_devices)}
                   for index, state in enumerate(STATE_ORDER)]
    assignment_path = run_dir / "execution_assignments.json"
    atomic_json(assignment_path, {"schema_version": 1, "jobs": assignments})
    scheduler = {"mode": "serial_branches" if len(selected_devices) == 1 else "independent_branches",
                 "devices": selected_devices, "worker_count": len(selected_devices),
                 "branch_count": len(STATE_ORDER), "assignment_manifest": assignment_path.name,
                 "assignment_manifest_sha256": file_digest(assignment_path)}
    atomic_json(run_dir / "plan.json", {**plan, "execution_scheduler": scheduler})
    provenance = _git_provenance(registry.root)
    atomic_json(run_dir / "provenance.json", provenance)
    state = {"schema_version": 2, "protocol": PROTOCOL_ID,
             "status": "running" if execute else "planned", "phase": "training_checkpoint_pair",
             "execution_scheduler": scheduler,
             "workers": {job["state"]: {**job, "status": "planned"} for job in assignments}}

    def publish_state():
        state["failed_records"] = sum(worker["status"] == "failed" for worker in state["workers"].values())
        state["completed_branches"] = sum(worker["status"] == "complete" for worker in state["workers"].values())
        state["updated_at_utc"] = _utc_now()
        atomic_json(run_dir / "execution_state.json", state)

    def finish(qualification=None, reason=None):
        qualification = qualification or {}
        state["status"] = "failed" if reason else "complete" if execute else "planned"
        state["phase"] = "terminal"
        if reason:
            state["reason"] = reason
        state["publication_committed"] = _publication_committed(registry, run_dir)
        publish_state()
        atomic_json(run_dir / "summary.json", {
            "schema_version": 2, "protocol": PROTOCOL_ID, "status": state["status"],
            "strict_training_controlled_pair_eligible": bool(qualification.get("strict_training_controlled_pair_eligible")),
            "diagnostic_pair_eligible": bool(qualification.get("diagnostic_pair_eligible")),
            "formal_pair_requested": plan["formal_pair_requested"],
            "selected_train_images": plan["selected_train_images"],
            "expected_optimizer_steps_per_state": plan["expected_optimizer_steps_per_state"],
            "execution_scheduler": scheduler, "failed_records": state["failed_records"],
            "completed_branches": state["completed_branches"],
            "publication_committed": state["publication_committed"],
            **({"reason": reason} if reason else {}),
        })

    try:
        publish_state()
        qualification = (_execute_pair_training(registry, run_dir, plan, provenance,
                          selected_devices, max_images, state, publish_state) if execute else {})
        finish(qualification)
    except (Exception, KeyboardInterrupt) as exc:
        finish(reason="{}: {}".format(type(exc).__name__, exc))
        raise
    return run_dir
