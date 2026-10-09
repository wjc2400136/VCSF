from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .data.manager import validate_dataset
from .io import file_digest
from .modeling import checkpoint_path
from .registry import Registry
from .runtime_config import resolve_upstream_config
from .training_artifacts import validate_finetuned_checkpoint


EXPECTED = {
    "torch": "2.0.0+cu118",
    "torchvision": "0.15.1+cu118",
    "mmcv": "2.0.1",
    "mmengine": "0.7.4",
    "mmdet": "3.0.0",
    "mmyolo": "0.6.0",
    "numpy": "1.24.4",
    "cv2": "4.8.1",
    "albumentations": "1.3.1",
    "qudida": "0.0.4",
    "sklearn": "1.3.2",
    "joblib": "1.4.2",
    "threadpoolctl": "3.5.0",
}


def _select_datasets(registry: Registry, datasets: Optional[str]) -> List[str]:
    if datasets is None:
        return list(registry.datasets)
    dataset_ids = [item.strip() for item in datasets.split(",")]
    if any(not item for item in dataset_ids):
        raise ValueError("doctor --datasets requires nonempty comma-separated dataset IDs")
    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("doctor --datasets contains duplicate dataset IDs")
    unknown = [item for item in dataset_ids if item not in registry.datasets]
    if unknown:
        raise ValueError("doctor --datasets contains unknown dataset IDs: {}".format(
            ", ".join(unknown)
        ))
    return [dataset_id for dataset_id in registry.datasets if dataset_id in dataset_ids]


def run_doctor(
    registry: Registry, deep_data: bool = False, datasets: Optional[str] = None
) -> Dict[str, Any]:
    dataset_ids = _select_datasets(registry, datasets)
    checks: List[Dict[str, Any]] = []

    def add(name: str, status: str, detail: Any, critical: bool = False) -> None:
        checks.append(
            {"name": name, "status": status, "detail": str(detail), "critical": critical}
        )

    add(
        "python",
        "ok" if sys.version_info[:2] == (3, 8) else "error",
        "{} ({})".format(platform.python_version(), sys.executable),
        critical=True,
    )
    add(
        "environment_name",
        "ok" if Path(sys.prefix).name.lower() == "oda" else "error",
        Path(sys.prefix).name,
        critical=True,
    )
    for package, expected in EXPECTED.items():
        try:
            module = importlib.import_module(package)
            actual_value = getattr(module, "__version__", None)
            if actual_value is None:
                actual_value = importlib.metadata.version(package)
            actual = str(actual_value)
            add(
                package,
                "ok" if actual == expected else "error",
                "{} (expected {})".format(actual, expected),
                critical=True,
            )
        except Exception as exc:
            add(package, "error", exc, critical=True)

    add(
        "cuda",
        "ok" if torch.cuda.is_available() else "error",
        (
            "{}; runtime {}".format(torch.cuda.get_device_name(0), torch.version.cuda)
            if torch.cuda.is_available()
            else "not available"
        ),
        critical=True,
    )
    try:
        from mmcv.ops import nms

        device = "cuda" if torch.cuda.is_available() else "cpu"
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 9.0, 9.0]], device=device)
        scores = torch.tensor([0.9, 0.8], device=device)
        _, keep = nms(boxes, scores, 0.5)
        add("mmcv_ops", "ok" if keep.numel() == 1 else "error", "CUDA NMS keep={}".format(keep.tolist()), critical=True)
    except Exception as exc:
        add("mmcv_ops", "error", exc, critical=True)

    try:
        import mmdet

        local_shadow = registry.root in Path(mmdet.__file__).resolve().parents
        add(
            "package_shadow",
            "error" if local_shadow else "ok",
            mmdet.__file__,
            critical=True,
        )
    except Exception as exc:
        add("package_shadow", "error", exc, critical=True)

    missing_configs = []
    for model_id in registry.target_ids():
        model = registry.model(model_id)
        try:
            resolve_upstream_config(model)
        except Exception as exc:
            missing_configs.append("{}: {}".format(model.id, exc))
    add(
        "upstream_configs",
        "ok" if not missing_configs else "error",
        "16/16 found" if not missing_configs else "; ".join(missing_configs),
        critical=True,
    )

    dataset_reports = {}
    for dataset_id in dataset_ids:
        dataset = registry.dataset(dataset_id)
        report = validate_dataset(dataset, deep=deep_data)
        dataset_reports[dataset.id] = report
        add(
            "dataset:{}".format(dataset.id),
            "ok" if report["status"].startswith("valid") else "warning",
            report["status"],
            critical=False,
        )

    artifact_hash_cache: Dict[Path, str] = {}
    for dataset_id in dataset_ids:
        if dataset_id == "coco":
            coco = registry.dataset("coco")
            manifest_path = registry.root / "checkpoints" / "coco" / "manifest.json"
            manifest_records: Dict[str, Dict[str, Any]] = {}
            if manifest_path.is_file():
                try:
                    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_records = {
                        str(item["model"]): item
                        for item in manifest_payload.get("checkpoints", [])
                        if isinstance(item, dict) and item.get("model")
                    }
                except (OSError, ValueError, TypeError):
                    manifest_records = {}
            present = 0
            verified = 0
            checkpoint_issues = []
            for model_id in registry.target_ids():
                model = registry.model(model_id)
                path = checkpoint_path(model, coco)
                if not path.is_file():
                    checkpoint_issues.append("{} missing".format(model.id))
                    continue
                present += 1
                record = manifest_records.get(model.id)
                if not record:
                    checkpoint_issues.append("{} absent from manifest".format(model.id))
                    continue
                if int(record.get("bytes", -1)) != path.stat().st_size:
                    checkpoint_issues.append("{} size mismatch".format(model.id))
                    continue
                if deep_data and str(record.get("sha256", "")).lower() != file_digest(path).lower():
                    checkpoint_issues.append("{} SHA-256 mismatch".format(model.id))
                    continue
                verified += 1
            checkpoint_ok = present == 16 and verified == 16 and not checkpoint_issues
            add(
                "coco_checkpoints",
                "ok" if checkpoint_ok else "warning",
                "{}/16 present; {}/16 manifest{} verified{}".format(
                    present,
                    verified,
                    "+SHA-256" if deep_data else "",
                    "" if not checkpoint_issues else "; " + "; ".join(checkpoint_issues[:5]),
                ),
            )
            continue
        dataset = registry.dataset(dataset_id)
        present = 0
        verified = 0
        checkpoint_issues = []
        for model_id in registry.target_ids():
            model = registry.model(model_id)
            path = checkpoint_path(model, dataset)
            if not path.is_file() or path.stat().st_size <= (1 << 20):
                checkpoint_issues.append("{} missing".format(model_id))
                continue
            present += 1
            artifact_ok, artifact_issues = validate_finetuned_checkpoint(
                registry,
                dataset_id,
                model_id,
                deep=deep_data,
                hash_cache=artifact_hash_cache,
            )
            if artifact_ok:
                verified += 1
            else:
                checkpoint_issues.append(
                    "{} {}".format(model_id, "; ".join(artifact_issues))
                )
        checkpoint_ok = (
            present == len(registry.target_ids())
            and verified == len(registry.target_ids())
            and not checkpoint_issues
        )
        add(
            "{}_checkpoints".format(dataset_id),
            "ok" if checkpoint_ok else "warning",
            "{}/16 present; {}/16 manifest{} verified{}".format(
                present,
                verified,
                "+provenance+SHA-256+load+class-space"
                if deep_data
                else "+provenance",
                ""
                if not checkpoint_issues
                else "; " + "; ".join(checkpoint_issues[:5]),
            ),
        )
    critical_errors = [
        check for check in checks if check["critical"] and check["status"] == "error"
    ]
    return {
        "status": "ok" if not critical_errors else "error",
        "environment_prefix": sys.prefix,
        "checks": checks,
        "datasets": dataset_reports,
    }
