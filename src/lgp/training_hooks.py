from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner

from .io import atomic_json


def _normalized_state(
    state: MutableMapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    normalized: Dict[str, Any] = {}
    raw_keys: Dict[str, str] = {}
    for raw_key, value in state.items():
        if not isinstance(raw_key, str):
            raise RuntimeError("Checkpoint state dictionary contains a non-string key")
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key in normalized:
            raise RuntimeError(
                "Checkpoint state dictionary has duplicate normalized key {}".format(
                    key
                )
            )
        normalized[key] = value
        raw_keys[key] = raw_key
    return normalized, raw_keys


def _tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    try:
        raw = value.numpy().tobytes(order="C")
    except TypeError:
        raw = value.view(torch.uint8).numpy().tobytes(order="C")
    digest.update(raw)
    return digest.hexdigest()


def _state_fingerprint(
    state: Mapping[str, Any], keys: Sequence[str]
) -> Tuple[str, Dict[str, str], int]:
    per_key: Dict[str, str] = {}
    value_count = 0
    aggregate = hashlib.sha256()
    for key in keys:
        value = state[key]
        if not torch.is_tensor(value):
            raise RuntimeError(
                "State dictionary value is not a tensor: {}".format(key)
            )
        tensor_hash = _tensor_digest(value)
        per_key[key] = tensor_hash
        value_count += int(value.numel())
        aggregate.update(key.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(tensor_hash.encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), per_key, value_count


@HOOKS.register_module()
class LGPSemanticClassHeadInitHook(Hook):
    """Copy audited source-class rows and verify the actual loaded state.

    MMEngine normally skips class-head tensors whose source and target shapes
    differ. For a declared semantic transfer policy, this hook remaps the
    checkpoint in memory before MMEngine loads it, then hashes every compatible
    tensor from the live model immediately before the first training step.
    """

    def __init__(
        self,
        *,
        source_dataset: str,
        target_dataset: str,
        parameter_stem: str,
        source_classes: Sequence[str],
        target_classes: Sequence[str],
        target_to_source: Mapping[str, str],
        audit_path: str,
        policy: str = "semantic_class_copy",
    ) -> None:
        if policy != "semantic_class_copy":
            raise ValueError("Unsupported semantic head initialization policy")
        self.policy = policy
        self.source_dataset = str(source_dataset)
        self.target_dataset = str(target_dataset)
        self.parameter_stem = str(parameter_stem)
        self.source_classes = tuple(str(value) for value in source_classes)
        self.target_classes = tuple(str(value) for value in target_classes)
        self.target_to_source = {
            str(key): str(value) for key, value in target_to_source.items()
        }
        self.audit_path = Path(audit_path)
        if not self.source_classes or len(set(self.source_classes)) != len(
            self.source_classes
        ):
            raise ValueError("source_classes must be non-empty and unique")
        if not self.target_classes or len(set(self.target_classes)) != len(
            self.target_classes
        ):
            raise ValueError("target_classes must be non-empty and unique")
        if set(self.target_to_source) != set(self.target_classes):
            raise ValueError("target_to_source must map every target class exactly once")
        mapped_sources = [
            self.target_to_source[target] for target in self.target_classes
        ]
        if not set(mapped_sources).issubset(self.source_classes):
            raise ValueError("target_to_source contains an unknown source class")
        if len(set(mapped_sources)) != len(mapped_sources):
            raise ValueError("target_to_source must be one-to-one")
        if not self.parameter_stem:
            raise ValueError("parameter_stem must be non-empty")
        self._prepared: Optional[Dict[str, Any]] = None
        self._expected_hashes: Dict[str, str] = {}
        self._expected_shapes: Dict[str, Tuple[int, ...]] = {}

    def _mapping_records(self) -> list:
        source_index = {
            class_name: index
            for index, class_name in enumerate(self.source_classes)
        }
        return [
            {
                "target_index": target_index,
                "target_class": target_class,
                "source_index": source_index[self.target_to_source[target_class]],
                "source_class": self.target_to_source[target_class],
            }
            for target_index, target_class in enumerate(self.target_classes)
        ]

    def _write_failure(self, exc: Exception) -> None:
        atomic_json(
            self.audit_path,
            {
                "schema_version": 1,
                "status": "failed",
                "policy": self.policy,
                "source_dataset": self.source_dataset,
                "target_dataset": self.target_dataset,
                "parameter_stem": self.parameter_stem,
                "reason": "{}: {}".format(type(exc).__name__, exc),
            },
        )

    def after_load_checkpoint(
        self, runner: Runner, checkpoint: MutableMapping[str, Any]
    ) -> None:
        try:
            if self._prepared is not None:
                raise RuntimeError(
                    "Semantic head initialization received more than one checkpoint"
                )
            if not isinstance(checkpoint, MutableMapping):
                raise RuntimeError("Loaded checkpoint root must be mutable")
            raw_state = checkpoint.get("state_dict", checkpoint.get("model"))
            if not isinstance(raw_state, MutableMapping) or not raw_state:
                raise RuntimeError(
                    "Loaded checkpoint has no mutable state dictionary"
                )
            source, raw_keys = _normalized_state(raw_state)
            target = runner.model.state_dict()
            weight_key = self.parameter_stem + ".weight"
            bias_key = self.parameter_stem + ".bias"
            for key in (weight_key, bias_key):
                if key not in source or key not in target:
                    raise RuntimeError(
                        "Semantic head parameter is missing: {}".format(key)
                    )
                if not torch.is_tensor(source[key]) or not torch.is_tensor(target[key]):
                    raise RuntimeError(
                        "Semantic head parameter is not a tensor: {}".format(key)
                    )

            source_rows = int(source[weight_key].shape[0])
            target_rows = int(target[weight_key].shape[0])
            if target_rows != len(self.target_classes):
                raise RuntimeError(
                    "Target head row count {} differs from {} target classes".format(
                        target_rows, len(self.target_classes)
                    )
                )
            if int(target[bias_key].shape[0]) != target_rows:
                raise RuntimeError("Target head weight/bias row counts differ")

            mapping = self._mapping_records()
            if source_rows == len(self.source_classes):
                indices = torch.tensor(
                    [int(record["source_index"]) for record in mapping],
                    dtype=torch.long,
                    device=source[weight_key].device,
                )
                for key in (weight_key, bias_key):
                    source_value = source[key]
                    target_value = target[key]
                    if tuple(source_value.shape[1:]) != tuple(
                        target_value.shape[1:]
                    ):
                        raise RuntimeError(
                            "Semantic head trailing shape mismatch for {}".format(key)
                        )
                    remapped = source_value.index_select(0, indices).clone()
                    if tuple(remapped.shape) != tuple(target_value.shape):
                        raise RuntimeError(
                            "Remapped semantic head shape mismatch for {}".format(key)
                        )
                    raw_state[raw_keys[key]] = remapped
                load_mode = "fresh_semantic_class_copy"
                source, raw_keys = _normalized_state(raw_state)
            elif source_rows == target_rows:
                for key in (weight_key, bias_key):
                    if tuple(source[key].shape) != tuple(target[key].shape):
                        raise RuntimeError(
                            "Target-shape resume parameter mismatch for {}".format(key)
                        )
                load_mode = "target_shape_resume"
            else:
                raise RuntimeError(
                    "Semantic head source row count {} is neither source ({}) nor "
                    "target ({}) class count".format(
                        source_rows, len(self.source_classes), target_rows
                    )
                )

            missing = sorted(set(target).difference(source))
            unexpected = sorted(set(source).difference(target))
            shape_mismatches = sorted(
                key
                for key in set(source).intersection(target)
                if (
                    not torch.is_tensor(source[key])
                    or not torch.is_tensor(target[key])
                    or tuple(source[key].shape) != tuple(target[key].shape)
                )
            )
            if missing or unexpected or shape_mismatches:
                raise RuntimeError(
                    "Post-remap state incompatibility: missing={!r}; "
                    "unexpected={!r}; shape_mismatches={!r}".format(
                        missing, unexpected, shape_mismatches
                    )
                )

            keys = sorted(target)
            expected_digest, expected_hashes, value_count = _state_fingerprint(
                source, keys
            )
            self._expected_hashes = expected_hashes
            self._expected_shapes = {
                key: tuple(source[key].shape) for key in keys
            }
            self._prepared = {
                "schema_version": 1,
                "status": "prepared",
                "policy": self.policy,
                "load_mode": load_mode,
                "source_dataset": self.source_dataset,
                "target_dataset": self.target_dataset,
                "parameter_stem": self.parameter_stem,
                "source_class_count": len(self.source_classes),
                "target_class_count": len(self.target_classes),
                "class_mapping": mapping,
                "missing_keys": [],
                "unexpected_keys": [],
                "shape_mismatch_keys": [],
                "compatible_tensor_count": len(keys),
                "compatible_value_count": value_count,
                "expected_state_sha256": expected_digest,
                "expected_parameter_sha256": {
                    key: expected_hashes[key] for key in (weight_key, bias_key)
                },
            }
        except Exception as exc:
            self._write_failure(exc)
            raise

    def before_train(self, runner: Runner) -> None:
        try:
            if self._prepared is None:
                raise RuntimeError(
                    "Semantic head initialization did not observe the loaded checkpoint"
                )
            actual = runner.model.state_dict()
            expected_keys = sorted(self._expected_hashes)
            missing = sorted(set(expected_keys).difference(actual))
            unexpected = sorted(set(actual).difference(expected_keys))
            shape_mismatches = sorted(
                key
                for key in set(expected_keys).intersection(actual)
                if (
                    not torch.is_tensor(actual[key])
                    or tuple(actual[key].shape) != self._expected_shapes[key]
                )
            )
            if missing or unexpected or shape_mismatches:
                raise RuntimeError(
                    "Live post-load state incompatibility: missing={!r}; "
                    "unexpected={!r}; shape_mismatches={!r}".format(
                        missing, unexpected, shape_mismatches
                    )
                )
            actual_digest, actual_hashes, value_count = _state_fingerprint(
                actual, expected_keys
            )
            mismatched = [
                key
                for key in expected_keys
                if actual_hashes[key] != self._expected_hashes[key]
            ]
            if mismatched:
                raise RuntimeError(
                    "Live model tensors differ from the audited warm start: {}".format(
                        ", ".join(mismatched[:20])
                    )
                )
            weight_key = self.parameter_stem + ".weight"
            bias_key = self.parameter_stem + ".bias"
            audit = {
                **self._prepared,
                "status": "verified",
                "loaded_state_sha256": actual_digest,
                "loaded_parameter_sha256": {
                    key: actual_hashes[key] for key in (weight_key, bias_key)
                },
                "loaded_value_count": value_count,
                "state_dict_exact_match": (
                    actual_digest == self._prepared["expected_state_sha256"]
                ),
            }
            atomic_json(self.audit_path, audit)
        except Exception as exc:
            self._write_failure(exc)
            raise


@HOOKS.register_module()
class LGPCheckFiniteLossHook(Hook):
    """Fail a training attempt as soon as its reported loss is non-finite."""

    def __init__(self, interval: int = 1) -> None:
        if int(interval) <= 0:
            raise ValueError("interval must be positive")
        self.interval = int(interval)

    def after_train_iter(
        self,
        runner: Runner,
        batch_idx: int,
        data_batch: Optional[dict] = None,
        outputs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not self.every_n_train_iters(runner, self.interval):
            return
        if not isinstance(outputs, Mapping) or "loss" not in outputs:
            raise RuntimeError(
                "Training output is missing the loss required by the finite-loss gate"
            )
        loss = outputs["loss"]
        if torch.is_tensor(loss):
            finite = bool(torch.isfinite(loss).all().item())
        else:
            try:
                finite = bool(torch.isfinite(torch.as_tensor(loss)).all().item())
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Training output loss cannot be checked for finiteness"
                ) from exc
        if not finite:
            raise FloatingPointError(
                "Non-finite training loss at iteration {}".format(runner.iter + 1)
            )
