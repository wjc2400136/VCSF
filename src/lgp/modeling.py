from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from mmdet.apis import init_detector

from .io import iter_download
from .paths import project_root
from .registry import DatasetSpec, ModelSpec
from .runtime_config import build_runtime_config, register_framework


def finite_state_dict_audit(
    state_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    """Require every floating or complex checkpoint tensor to be finite."""
    tensor_count = 0
    numeric_tensor_count = 0
    numeric_value_count = 0
    nonfinite_keys = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        if not (value.is_floating_point() or value.is_complex()):
            continue
        numeric_tensor_count += 1
        numeric_value_count += int(value.numel())
        if not bool(torch.isfinite(value).all().item()):
            nonfinite_keys.append(str(key))
    if nonfinite_keys:
        preview = nonfinite_keys[:20]
        suffix = "" if len(nonfinite_keys) <= len(preview) else " ..."
        raise FloatingPointError(
            "Checkpoint state dictionary contains non-finite tensors: {}{}".format(
                ", ".join(preview), suffix
            )
        )
    return {
        "all_finite": True,
        "tensor_count": tensor_count,
        "numeric_tensor_count": numeric_tensor_count,
        "numeric_value_count": numeric_value_count,
    }


def checkpoint_path(model: ModelSpec, dataset: DatasetSpec) -> Path:
    root = project_root() / "checkpoints" / dataset.id
    if dataset.id == "coco":
        return root / model.checkpoint_filename
    return root / (model.id + ".pth")


def ensure_checkpoint(
    model: ModelSpec,
    dataset: DatasetSpec,
    download: bool = False,
) -> Path:
    destination = checkpoint_path(model, dataset)
    if destination.is_file() and destination.stat().st_size > 1 << 20:
        return destination
    if dataset.id != "coco":
        raise FileNotFoundError(
            "Missing fine-tuned checkpoint for {}/{}: {}. Run `lgp train --dataset {} "
            "--model {}` first; COCO weights are not silently used for a different label space.".format(
                dataset.id, model.id, destination, dataset.id, model.id
            )
        )
    if not download:
        raise FileNotFoundError(
            "Missing official checkpoint: {}. Run `lgp weights download --models {}`.".format(
                destination, model.id
            )
        )
    last = None
    for last in iter_download(model.checkpoint, destination):
        pass
    if not destination.is_file():
        raise RuntimeError("Checkpoint download did not produce {}: {}".format(destination, last))
    return destination


def load_detector(
    model: ModelSpec,
    dataset: DatasetSpec,
    work_dir: Path,
    device: str = "cuda:0",
    download: bool = False,
    split: str = "val",
    checkpoint_override: Optional[Path] = None,
):
    register_framework(model.framework)
    if checkpoint_override is None:
        checkpoint = ensure_checkpoint(model, dataset, download=download)
    else:
        checkpoint = checkpoint_override.resolve()
        if not checkpoint.is_file() or checkpoint.stat().st_size <= 1 << 20:
            raise FileNotFoundError(
                "Missing or undersized checkpoint override: {}".format(checkpoint)
            )
    cfg = build_runtime_config(
        model, dataset, work_dir, mode="test", test_split=split
    )
    detector = init_detector(cfg, str(checkpoint), device=device)
    detector.eval()
    return detector, cfg, checkpoint
