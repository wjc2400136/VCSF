from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from ..io import atomic_json, file_digest, iter_download
from ..modeling import checkpoint_path
from ..paths import project_root
from ..registry import Registry


def _valid_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 1 << 20:
        return False
    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception:
        return False
    return isinstance(payload, dict) and any(
        key in payload for key in ("state_dict", "model", "meta")
    )


def download_weights(registry: Registry, model_ids: Iterable[str]) -> List[Dict[str, Any]]:
    dataset = registry.dataset("coco")
    records = []
    for model_id in model_ids:
        model = registry.model(model_id)
        destination = checkpoint_path(model, dataset)
        if destination.exists() and not _valid_checkpoint(destination):
            destination.unlink()
        if not destination.is_file():
            for _ in iter_download(model.checkpoint, destination):
                pass
        if not _valid_checkpoint(destination):
            destination.unlink(missing_ok=True)
            raise RuntimeError("Downloaded checkpoint has no recognized state dictionary: {}".format(destination))
        records.append(
            {
                "model": model_id,
                "path": str(destination),
                "bytes": destination.stat().st_size,
                "sha256": file_digest(destination),
                "url": model.checkpoint,
            }
        )
    manifest_path = project_root() / "checkpoints" / "coco" / "manifest.json"
    existing: Dict[str, Dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            existing = {
                str(item["model"]): dict(item)
                for item in payload.get("checkpoints", [])
                if isinstance(item, dict) and item.get("model")
            }
        except (OSError, ValueError, TypeError):
            existing = {}
    existing.update({str(item["model"]): item for item in records})
    ordered = [existing[model_id] for model_id in registry.target_ids() if model_id in existing]
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "framework_versions": registry.framework_versions,
            "checkpoints": ordered,
        },
    )
    return records
