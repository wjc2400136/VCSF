from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import math
import multiprocessing as mp
import queue
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter, __version__ as PILLOW_VERSION

from ..attacks.common import parse_fraction
from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..paths import project_root
from ..registry import Registry
from ..reporting.all_methods import write_artifact_manifest
from .evaluate import run_evaluation
from .formal_parallel import (
    normalize_execution_devices,
    round_robin_indices,
    validate_available_cuda_devices,
)


PROTOCOL_ID = "coco_retained500_common2_preprocessing_defense"
SOURCE_PROTOCOL = "coco_all_methods"
RETAINED_IMAGES = 500
METRICS = (
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_AR_1",
    "bbox_AR_10",
    "bbox_AR_100",
    "bbox_AR_small",
    "bbox_AR_medium",
    "bbox_AR_large",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise RuntimeError(
            "Path escapes the allowed root: {} not under {}".format(
                resolved, resolved_root
            )
        ) from exc
    return resolved


def _relative(path: Path, root: Path) -> str:
    return _inside(path, root).relative_to(root.resolve()).as_posix()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.part".format(path.name))
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _number(value: Any) -> float:
    parsed = parse_fraction(value)
    if not isinstance(parsed, (int, float)):
        raise ValueError("Expected a numeric defense parameter, got {!r}".format(value))
    result = float(parsed)
    if not math.isfinite(result):
        raise ValueError("Defense parameter must be finite")
    return result


def transform_image(image: Image.Image, spec: Mapping[str, Any]) -> Image.Image:
    """Apply one deterministic victim-side preprocessing transform."""
    image = image.convert("RGB")
    kind = str(spec["kind"])
    if kind == "identity":
        return image.copy()
    if kind == "jpeg":
        quality = int(spec["quality"])
        if quality <= 0 or quality > 100:
            raise ValueError("JPEG quality must lie in [1, 100]")
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB").copy()
    if kind == "bit_depth":
        bits = int(spec["bits"])
        if bits <= 0 or bits > 8:
            raise ValueError("Bit depth must lie in [1, 8]")
        levels = float((1 << bits) - 1)
        values = np.asarray(image, dtype=np.float32) / 255.0
        quantized = np.rint(values * levels) / levels
        return Image.fromarray(
            np.rint(quantized * 255.0).astype(np.uint8), mode="RGB"
        )
    if kind == "gaussian":
        radius = _number(spec["radius"])
        if radius <= 0.0:
            raise ValueError("Gaussian radius must be positive")
        return image.filter(ImageFilter.GaussianBlur(radius=radius))
    if kind == "median":
        size = int(spec["size"])
        if size < 3 or size % 2 == 0:
            raise ValueError("Median size must be an odd integer >= 3")
        return image.filter(ImageFilter.MedianFilter(size=size))
    if kind == "resize_roundtrip":
        scale = _number(spec["scale"])
        if scale <= 0.0:
            raise ValueError("Resize scale must be positive")
        width, height = image.size
        scaled = image.resize(
            (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            ),
            Image.Resampling.BILINEAR,
        )
        return scaled.resize((width, height), Image.Resampling.BILINEAR)
    raise ValueError("Unknown preprocessing-defense kind: {}".format(kind))


def selected_defenses(
    registry: Registry, defense_override: Optional[Sequence[str]] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    order = list(
        defense_override
        if defense_override is not None
        else registry.preprocessing_defense_order
    )
    unknown = sorted(set(order) - set(registry.preprocessing_defenses))
    if unknown:
        raise ValueError("Unknown defense variants: {}".format(", ".join(unknown)))
    if len(order) != len(set(order)):
        raise ValueError("Defense selection contains duplicates")
    return [
        (defense_id, dict(registry.preprocessing_defenses[defense_id]))
        for defense_id in order
    ]


def _protocol_selection(
    registry: Registry,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    protocol = dict(registry.protocols[PROTOCOL_ID])
    if str(protocol.get("status", "active")) != "active":
        raise RuntimeError(
            "The retained-500 preprocessing protocol is historical after the "
            "SFIM-B/NAA fidelity revision and cannot be regenerated"
        )
    sources = list(source_override or protocol["sources"])
    methods = list(method_override or protocol["methods"])
    targets = list(target_override or registry.paper_order)
    defenses = selected_defenses(registry, defense_override)
    for source in sources:
        if source not in protocol["sources"]:
            raise ValueError("Source '{}' is outside Common-2".format(source))
        registry.model(source)
    for method in methods:
        registry.attack(method)
    for target in targets:
        registry.model(target)
    if len(sources) != len(set(sources)):
        raise ValueError("Source selection contains duplicates")
    if len(methods) != len(set(methods)):
        raise ValueError("Method selection contains duplicates")
    if len(targets) != len(set(targets)):
        raise ValueError("Target selection contains duplicates")
    for source in sources:
        for method in methods:
            status = registry.compatibility_status(method, source)["status"]
            if status != "native":
                raise RuntimeError(
                    "Common-2 requires a native method/source pair: {}/{} is {}".format(
                        method, source, status
                    )
                )
    return {
        "protocol": protocol,
        "sources": sources,
        "methods": methods,
        "targets": targets,
        "defenses": defenses,
    }


def build_defense_plan(
    registry: Registry,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images must be positive")
    selection = _protocol_selection(
        registry,
        source_override=source_override,
        method_override=method_override,
        target_override=target_override,
        defense_override=defense_override,
    )
    sources = selection["sources"]
    methods = selection["methods"]
    targets = selection["targets"]
    defenses = selection["defenses"]
    clean_jobs = []
    attacked_jobs = []
    for defense_id, spec in defenses:
        for target in targets:
            clean_jobs.append(
                {
                    "record_type": "clean",
                    "dataset": "coco",
                    "split": "val",
                    "source": "clean",
                    "attack": "clean",
                    "defense": defense_id,
                    "defense_parameters": spec,
                    "target": target,
                    "status": "ready",
                }
            )
    for source in sources:
        for method in methods:
            for defense_id, spec in defenses:
                for target in targets:
                    attacked_jobs.append(
                        {
                            "record_type": "attacked",
                            "dataset": "coco",
                            "split": "val",
                            "source": source,
                            "attack": method,
                            "defense": defense_id,
                            "defense_parameters": spec,
                            "target": target,
                            "status": "ready",
                        }
                    )
    full_selection = (
        sources == list(selection["protocol"]["sources"])
        and methods == list(selection["protocol"]["methods"])
        and targets == list(registry.paper_order)
        and [item[0] for item in defenses]
        == list(registry.preprocessing_defense_order)
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "dataset": "coco",
        "split": "val",
        "scope": "retained-500 fixed evidence subset",
        "full_coco_val_claim": False,
        "threat_model": dict(registry.preprocessing_defense_threat_model),
        "common_2": {
            "sources": sources,
            "definition": selection["protocol"]["common_2_definition"],
            "paper_term": False,
        },
        "methods": methods,
        "targets": targets,
        "defenses": [
            {"id": defense_id, **dict(spec)} for defense_id, spec in defenses
        ],
        "retained_images": RETAINED_IMAGES,
        "max_images": max_images,
        "diagnostic_limited_images": max_images is not None,
        "full_registered_selection": full_selection,
        "counts": {
            "clean_evaluations": len(clean_jobs),
            "attack_evaluations": len(attacked_jobs),
            "records": len(clean_jobs) + len(attacked_jobs),
            "clean_variant_groups": len(defenses),
            "attack_variant_groups": len(sources) * len(methods) * len(defenses),
            "attack_groups": len(sources) * len(methods),
        },
        "jobs": clean_jobs + attacked_jobs,
    }


def _write_plan_csv(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "record_type",
        "dataset",
        "split",
        "source",
        "attack",
        "defense",
        "defense_parameters",
        "target",
        "status",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = dict(job)
            row["defense_parameters"] = json.dumps(
                row["defense_parameters"], sort_keys=True
            )
            writer.writerow({field: row.get(field) for field in fields})


def _subset_annotation(
    payload: Mapping[str, Any],
    image_ids: Sequence[int],
    file_names: Optional[Mapping[int, str]] = None,
) -> Dict[str, Any]:
    order = {int(image_id): index for index, image_id in enumerate(image_ids)}
    images = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        if image_id not in order:
            continue
        copied = dict(image)
        if file_names is not None:
            copied["file_name"] = str(file_names[image_id])
        images.append(copied)
    images.sort(key=lambda item: order[int(item["id"])])
    if [int(item["id"]) for item in images] != list(image_ids):
        raise RuntimeError("Annotation does not contain the exact retained image IDs")
    selected = set(order)
    annotations = [
        dict(annotation)
        for annotation in payload.get("annotations", [])
        if int(annotation["image_id"]) in selected
    ]
    return {
        "info": dict(payload.get("info", {})),
        "licenses": list(payload.get("licenses", [])),
        "images": images,
        "annotations": annotations,
        "categories": list(payload.get("categories", [])),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def verify_source_evidence(
    registry: Registry,
    acceptance_path: Optional[Path] = None,
    deep: bool = True,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Fail closed on the accepted COCO source and every Common-2 payload."""
    acceptance = (
        acceptance_path
        or registry.root
        / "outputs"
        / "reports"
        / "coco"
        / "formal_all_methods_acceptance.json"
    ).resolve()
    if not acceptance.is_file():
        raise FileNotFoundError("Missing accepted COCO evidence: {}".format(acceptance))
    accepted = _read_json(acceptance)
    if accepted.get("protocol") != SOURCE_PROTOCOL:
        raise RuntimeError("COCO acceptance has the wrong source protocol")
    if accepted.get("formal_eligible") is not True:
        raise RuntimeError("COCO source evidence is not formally eligible")
    if accepted.get("status") != "accepted_by_evidence_revalidation":
        raise RuntimeError("Unexpected COCO acceptance status")
    counts = dict(accepted.get("counts", {}))
    required_counts = {
        "records": 1072,
        "attack_evaluations": 864,
        "attack_groups": 54,
        "clean_evaluations": 16,
        "failed_records": 0,
        "formal_images": 5000,
        "retained_images_per_attack_payload": RETAINED_IMAGES,
    }
    for key, value in required_counts.items():
        if int(counts.get(key, -1)) != value:
            raise RuntimeError(
                "COCO acceptance count {} is {}, expected {}".format(
                    key, counts.get(key), value
                )
            )
    source_value = Path(str(accepted["source_run"]))
    source_run = (
        source_value if source_value.is_absolute() else registry.root / source_value
    ).resolve()
    if not source_run.is_dir():
        raise FileNotFoundError("Accepted COCO source run is missing: {}".format(source_run))
    expected_hashes = dict(accepted.get("evidence_sha256", {}))
    evidence_files = {
        "records.json": source_run / "records.json",
        "plan.json": source_run / "plan.json",
        "execution_state.json": source_run / "execution_state.json",
        "execution_assignments.json": source_run / "execution_assignments.json",
        "provenance.json": source_run / "provenance.json",
        "summary.json": source_run / "summary.json",
    }
    verified_evidence = {}
    for key, path in evidence_files.items():
        expected = expected_hashes.get(key)
        if expected is None:
            raise RuntimeError("Acceptance omits source hash for {}".format(key))
        actual = file_digest(path)
        if actual != expected:
            raise RuntimeError("Accepted source hash mismatch for {}".format(key))
        verified_evidence[key] = actual
    retained_path = source_run / "retained_image_set.json"
    retained = _read_json(retained_path)
    if retained.get("status") != "frozen_before_generation":
        raise RuntimeError("Retained image selection was not frozen before generation")
    if retained.get("selection_algorithm") != "canonical_equal_bins_center_v1":
        raise RuntimeError("Unexpected retained-image selection algorithm")
    retained_rows = list(retained.get("retained", []))
    if len(retained_rows) != RETAINED_IMAGES:
        raise RuntimeError("Retained-image selection is not exactly 500 images")
    retained_ids = [int(item["image_id"]) for item in retained_rows]
    if _canonical_json_digest(retained_ids) != retained["retained_image_ids_sha256"]:
        raise RuntimeError("Retained-image ID digest mismatch")
    verify_count = RETAINED_IMAGES if max_images is None else min(max_images, RETAINED_IMAGES)
    ids_to_verify = retained_ids[:verify_count]

    protocol = registry.protocols[PROTOCOL_ID]
    groups = []
    for source in protocol["sources"]:
        for method in protocol["methods"]:
            root = source_run / "attacks" / "coco" / source / method / "default"
            manifest_path = root / "retained_500_manifest.json"
            annotation_path = root / "annotations.retained-500.json"
            if not manifest_path.is_file() or not annotation_path.is_file():
                raise FileNotFoundError(
                    "Missing retained payload evidence for {}/{}".format(source, method)
                )
            manifest = _read_json(manifest_path)
            rows = list(manifest.get("images", []))
            ids = [int(item["image_id"]) for item in rows]
            if int(manifest.get("count", -1)) != RETAINED_IMAGES or ids != retained_ids:
                raise RuntimeError(
                    "Retained payload selection mismatch for {}/{}".format(source, method)
                )
            if manifest.get("selection_sha256") != retained["retained_image_ids_sha256"]:
                raise RuntimeError(
                    "Retained payload selection hash mismatch for {}/{}".format(
                        source, method
                    )
                )
            verified = 0
            if deep:
                for item in rows[:verify_count]:
                    image_path = _inside(root / str(item["output_file"]), root)
                    if not image_path.is_file():
                        raise FileNotFoundError(
                            "Missing retained image: {}".format(image_path)
                        )
                    if image_path.stat().st_size != int(item["output_bytes"]):
                        raise RuntimeError(
                            "Retained image byte count mismatch: {}".format(image_path)
                        )
                    if file_digest(image_path) != item["output_sha256"]:
                        raise RuntimeError(
                            "Retained image hash mismatch: {}".format(image_path)
                        )
                    verified += 1
            groups.append(
                {
                    "source": source,
                    "method": method,
                    "root": _relative(root, registry.root),
                    "manifest": _relative(manifest_path, registry.root),
                    "manifest_sha256": file_digest(manifest_path),
                    "annotation": _relative(annotation_path, registry.root),
                    "annotation_sha256": file_digest(annotation_path),
                    "retained_images": len(rows),
                    "payload_hashes_verified": verified,
                }
            )

    index = CocoIndex(registry.dataset("coco"), "val")
    images_by_id = {int(image["id"]): image for image in index.images}
    clean_files = []
    for image_id in ids_to_verify:
        if image_id not in images_by_id:
            raise RuntimeError("Retained clean image ID is absent from COCO val")
        path = index.image_path(images_by_id[image_id])
        if not path.is_file():
            raise FileNotFoundError("Missing retained clean image: {}".format(path))
        clean_files.append(
            {
                "image_id": image_id,
                "file_name": str(images_by_id[image_id]["file_name"]),
                "bytes": path.stat().st_size,
                "sha256": file_digest(path) if deep else None,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "verified" if deep else "structure_verified",
        "verified_at_utc": _utc_now(),
        "acceptance": _relative(acceptance, registry.root),
        "acceptance_sha256": file_digest(acceptance),
        "source_run": _relative(source_run, registry.root),
        "source_protocol": SOURCE_PROTOCOL,
        "accepted_counts": required_counts,
        "source_evidence_sha256": verified_evidence,
        "retained_selection": _relative(retained_path, registry.root),
        "retained_selection_sha256": file_digest(retained_path),
        "retained_image_ids_sha256": retained["retained_image_ids_sha256"],
        "retained_image_ids": retained_ids,
        "deep_payload_hash_verification": bool(deep),
        "verified_images_per_payload": verify_count if deep else 0,
        "groups": groups,
        "clean_files": clean_files,
    }
    payload["evidence_set_sha256"] = _canonical_json_digest(
        {
            "acceptance_sha256": payload["acceptance_sha256"],
            "retained_selection_sha256": payload["retained_selection_sha256"],
            "groups": [
                {
                    "source": item["source"],
                    "method": item["method"],
                    "manifest_sha256": item["manifest_sha256"],
                    "annotation_sha256": item["annotation_sha256"],
                }
                for item in groups
            ],
            "clean_files": clean_files,
        }
    )
    return payload


def _clean_input(
    registry: Registry, image_ids: Sequence[int]
) -> Tuple[Dict[str, Any], Path]:
    index = CocoIndex(registry.dataset("coco"), "val")
    image_by_id = {int(image["id"]): image for image in index.images}
    file_names = {image_id: str(image_by_id[image_id]["file_name"]) for image_id in image_ids}
    annotation = _subset_annotation(index.payload, image_ids, file_names)
    image_root = (
        registry.dataset("coco").root / index.split.image_prefix
    ).resolve()
    return annotation, image_root


def _attack_input(
    registry: Registry,
    source_run: Path,
    source: str,
    method: str,
    image_ids: Sequence[int],
) -> Tuple[Dict[str, Any], Path]:
    root = source_run / "attacks" / "coco" / source / method / "default"
    payload = _read_json(root / "annotations.retained-500.json")
    annotation = _subset_annotation(payload, image_ids)
    return annotation, root.resolve()


def _materialize_view(
    registry: Registry,
    run_dir: Path,
    scratch: Path,
    manifest_path: Path,
    annotation: Mapping[str, Any],
    input_root: Path,
    source: str,
    method: str,
    defense_id: str,
    defense_spec: Mapping[str, Any],
    protocol_id: str = PROTOCOL_ID,
    threat_model: Optional[Mapping[str, Any]] = None,
    transform_implementation: Optional[Mapping[str, Any]] = None,
) -> Path:
    scratch_root = run_dir / "scratch"
    scratch = _inside(scratch, scratch_root)
    if scratch.exists():
        raise FileExistsError("Refusing to reuse defense scratch: {}".format(scratch))
    scratch.mkdir(parents=True)
    images = list(annotation.get("images", []))
    kind = str(defense_spec["kind"])
    started = time.perf_counter()
    output_rows = []
    output_names: Dict[int, str] = {}
    if kind == "identity":
        derived_annotation = deepcopy(dict(annotation))
    else:
        for item in images:
            image_id = int(item["id"])
            source_path = _inside(input_root / str(item["file_name"]), input_root)
            relative = "images/{:012d}.png".format(image_id)
            target = scratch / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source_path) as image:
                transformed = transform_image(image, defense_spec)
                transformed.save(target, format="PNG", compress_level=1)
            output_names[image_id] = relative
            output_rows.append(
                {
                    "image_id": image_id,
                    "output_file": relative,
                    "output_bytes": target.stat().st_size,
                    "output_sha256": file_digest(target),
                }
            )
        derived_annotation = _subset_annotation(
            annotation,
            [int(item["id"]) for item in images],
            output_names,
        )
    atomic_json(scratch / "annotations.json", derived_annotation)
    atomic_json(
        scratch / "run.json",
        {
            "schema_version": 1,
            "status": "complete",
            "dataset": "coco",
            "split": "val",
            "source": source,
            "attack": method,
            "annotation": "annotations.json",
            "image_root": str(input_root) if kind == "identity" else ".",
            "full_payload_available": True,
            "retained_subset": True,
            "retained_images": len(images),
            "defense_protocol": protocol_id,
            "defense": defense_id,
            "defense_parameters": dict(defense_spec),
            "threat_model": dict(
                threat_model or registry.preprocessing_defense_threat_model
            ),
        },
    )
    manifest = {
        "schema_version": 1,
        "status": "materialized_and_pending_target_panel",
        "record_type": "clean" if source == "clean" else "attacked",
        "source": source,
        "attack": method,
        "defense": defense_id,
        "defense_parameters": dict(defense_spec),
        "implementation": dict(
            transform_implementation or registry.preprocessing_defense_policy
        ),
        "pillow_version": PILLOW_VERSION,
        "images": len(images),
        "image_ids_sha256": _canonical_json_digest(
            [int(item["id"]) for item in images]
        ),
        "materialization": "reference_only" if kind == "identity" else "png",
        "outputs": output_rows,
        "elapsed_seconds": time.perf_counter() - started,
        "scratch_removed": False,
    }
    atomic_json(manifest_path, manifest)
    return scratch


def _remove_verified_scratch(
    scratch: Path, run_dir: Path, manifest_path: Path
) -> None:
    scratch_root = run_dir / "scratch"
    scratch = _inside(scratch, scratch_root)
    if not scratch.is_dir():
        raise RuntimeError("Validated scratch payload is missing: {}".format(scratch))
    quarantine = scratch.with_name("{}.verified-delete".format(scratch.name))
    _inside(quarantine, scratch_root)
    if quarantine.exists():
        raise FileExistsError("Scratch quarantine already exists: {}".format(quarantine))
    scratch.rename(quarantine)
    shutil.rmtree(str(quarantine))
    manifest = _read_json(manifest_path)
    manifest.update(
        status="complete_target_panel_and_payload_removed",
        scratch_removed=True,
        scratch_removed_at_utc=_utc_now(),
    )
    atomic_json(manifest_path, manifest)


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _record_key(record: Mapping[str, Any]) -> tuple:
    return (
        str(record.get("record_type", "")),
        str(record.get("source", "")),
        str(record.get("attack", "")),
        str(record.get("defense", "")),
        str(record.get("target", "")),
    )


def _expected_record_keys(plan: Mapping[str, Any]) -> List[tuple]:
    return [_record_key(job) for job in plan["jobs"]]


def _canonical_records(
    records: Sequence[Mapping[str, Any]], expected_keys: Sequence[tuple]
) -> List[Dict[str, Any]]:
    positions = {key: index for index, key in enumerate(expected_keys)}
    if len(positions) != len(expected_keys):
        raise RuntimeError("Defense plan contains duplicate record keys")
    by_key = {}
    for record in records:
        key = _record_key(record)
        if key not in positions:
            raise RuntimeError("Unexpected defense record key: {}".format(key))
        if key in by_key:
            raise RuntimeError("Duplicate defense record key: {}".format(key))
        by_key[key] = dict(record)
    return sorted(by_key.values(), key=lambda item: positions[_record_key(item)])


def _evaluation_record(
    completed: Path,
    record_type: str,
    source: str,
    method: str,
    defense_id: str,
    defense_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    record = dict(_read_json(completed / "metrics.json"))
    record.update(
        {
            "record_type": record_type,
            "source": source,
            "attack": method,
            "defense": defense_id,
            "defense_parameters": dict(defense_spec),
            "threat_model": "oblivious_victim_input_preprocessing",
            "source_matched_target": (
                record_type == "attacked" and str(record["target"]) == source
            ),
        }
    )
    atomic_json(completed / "metrics.json", record)
    return record


def _evaluate_variant(
    registry: Registry,
    run_dir: Path,
    source_run: Path,
    device: str,
    slot: int,
    record_type: str,
    source: str,
    method: str,
    defense_id: str,
    defense_spec: Mapping[str, Any],
    targets: Sequence[str],
    image_ids: Sequence[int],
    download_weights: bool,
    events: Any,
    group_status_path: Path,
) -> bool:
    if record_type == "clean":
        annotation, input_root = _clean_input(registry, image_ids)
        scratch = run_dir / "scratch" / "worker-{}".format(slot) / "clean" / defense_id
        transform_manifest = run_dir / "transform_manifests" / "clean" / "{}.json".format(defense_id)
        evaluation_root = run_dir / "evaluations" / "clean" / defense_id
    else:
        annotation, input_root = _attack_input(
            registry, source_run, source, method, image_ids
        )
        scratch = (
            run_dir
            / "scratch"
            / "worker-{}".format(slot)
            / source
            / method
            / defense_id
        )
        transform_manifest = (
            run_dir
            / "transform_manifests"
            / "attacked"
            / source
            / method
            / "{}.json".format(defense_id)
        )
        evaluation_root = (
            run_dir / "evaluations" / "attacked" / source / method / defense_id
        )
    current = {
        "kind": "preprocess_materialization",
        "record_type": record_type,
        "source": source,
        "method": method,
        "defense": defense_id,
    }
    events.put({"type": "task_started", "slot": slot, "current": current})
    try:
        view = _materialize_view(
            registry,
            run_dir,
            scratch,
            transform_manifest,
            annotation,
            input_root,
            source,
            method,
            defense_id,
            defense_spec,
        )
    except BaseException as exc:
        reason = "preprocessing failed: {}: {}".format(type(exc).__name__, exc)
        status = dict(_read_json(group_status_path))
        status.update(status="failed", reason=reason, failed_at_utc=_utc_now())
        atomic_json(group_status_path, status)
        failure = {
            "record_type": record_type,
            "dataset": "coco",
            "split": "val",
            "source": source,
            "attack": method,
            "defense": defense_id,
            "defense_parameters": dict(defense_spec),
            "target": str(targets[0]),
            "status": "failed",
            "reason": reason,
            "metrics": {},
        }
        events.put(
            {
                "type": "fatal",
                "slot": slot,
                "current": current,
                "records": [failure],
                "reason": reason,
            }
        )
        return False
    for target_index, target in enumerate(targets, start=1):
        current = {
            "kind": "target_evaluation",
            "record_type": record_type,
            "source": source,
            "method": method,
            "defense": defense_id,
            "target": target,
            "target_index": target_index,
            "target_total": len(targets),
        }
        events.put({"type": "task_started", "slot": slot, "current": current})
        try:
            completed = run_evaluation(
                registry,
                "coco",
                target,
                split="val",
                adversarial_run=view,
                output_dir=evaluation_root / target,
                max_images=None,
                device=device,
                download_weights=download_weights,
                keep_going=False,
                save_visualizations=False,
                prediction_archive="gzip",
            )
            record = _evaluation_record(
                completed,
                record_type,
                source,
                method,
                defense_id,
                defense_spec,
            )
            if record.get("status") != "complete":
                raise RuntimeError("Evaluation did not finish without failures")
            if int(record.get("images", -1)) != len(image_ids):
                raise RuntimeError("Evaluation image count differs from the fixed subset")
            if record.get("evaluated_image_ids_match_expected") is not True:
                raise RuntimeError("Evaluation image IDs differ from the fixed subset")
        except BaseException as exc:
            reason = "evaluation failed: {}: {}".format(type(exc).__name__, exc)
            failure = {
                "record_type": record_type,
                "dataset": "coco",
                "split": "val",
                "source": source,
                "attack": method,
                "defense": defense_id,
                "defense_parameters": dict(defense_spec),
                "target": target,
                "status": "failed",
                "reason": reason,
                "metrics": {},
            }
            status = dict(_read_json(group_status_path))
            status.update(status="failed", reason=reason, failed_at_utc=_utc_now())
            atomic_json(group_status_path, status)
            events.put(
                {
                    "type": "fatal",
                    "slot": slot,
                    "current": current,
                    "records": [failure],
                    "reason": reason,
                }
            )
            return False
        finally:
            _cleanup_cuda()
        status = dict(_read_json(group_status_path))
        status.update(
            current_defense=defense_id,
            current_target=target,
            current_target_index=target_index,
            targets_completed=target_index,
        )
        atomic_json(group_status_path, status)
        events.put(
            {
                "type": "record_complete",
                "slot": slot,
                "current": current,
                "record": record,
            }
        )
    _remove_verified_scratch(view, run_dir, transform_manifest)
    events.put(
        {
            "type": "variant_complete",
            "slot": slot,
            "current": current,
            "record_type": record_type,
            "source": source,
            "method": method,
            "defense": defense_id,
        }
    )
    return True


def _worker_main(
    root: str,
    run_dir_value: str,
    source_run_value: str,
    device: str,
    slot: int,
    stage: str,
    tasks: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    defenses: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    download_weights: bool,
    events: Any,
) -> None:
    try:
        if not str(device).startswith("cuda:"):
            raise RuntimeError("Parallel defense workers require explicit CUDA devices")
        torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        registry = Registry(Path(root))
        run_dir = Path(run_dir_value)
        source_run = Path(source_run_value)
        for task in tasks:
            if stage == "clean":
                source = "clean"
                method = "clean"
                defense_ids = [str(task["defense"])]
                status_path = run_dir / "group_status" / "clean" / "{}.json".format(defense_ids[0])
                status = {
                    "schema_version": 1,
                    "status": "running",
                    "worker_slot": slot,
                    "device": device,
                    "record_type": "clean",
                    "defense": defense_ids[0],
                    "targets_total": len(targets),
                    "started_at_utc": _utc_now(),
                }
            elif stage == "attacked":
                source = str(task["source"])
                method = str(task["method"])
                defense_ids = [str(item["id"]) for item in defenses]
                status_path = run_dir / "group_status" / "attacked" / source / "{}.json".format(method)
                status = {
                    "schema_version": 1,
                    "status": "running",
                    "worker_slot": slot,
                    "device": device,
                    "record_type": "attacked",
                    "source": source,
                    "method": method,
                    "variants_total": len(defense_ids),
                    "variants_completed": 0,
                    "targets_per_variant": len(targets),
                    "started_at_utc": _utc_now(),
                }
            else:
                raise RuntimeError("Unknown defense stage: {}".format(stage))
            atomic_json(status_path, status)
            for defense_index, defense_id in enumerate(defense_ids, start=1):
                spec = dict(registry.preprocessing_defenses[defense_id])
                status = dict(_read_json(status_path))
                status.update(
                    current_defense=defense_id,
                    current_defense_index=defense_index,
                )
                atomic_json(status_path, status)
                ok = _evaluate_variant(
                    registry,
                    run_dir,
                    source_run,
                    device,
                    slot,
                    "clean" if stage == "clean" else "attacked",
                    source,
                    method,
                    defense_id,
                    spec,
                    targets,
                    image_ids,
                    download_weights,
                    events,
                    status_path,
                )
                if not ok:
                    return
                status = dict(_read_json(status_path))
                if stage == "attacked":
                    status["variants_completed"] = defense_index
                atomic_json(status_path, status)
            status = dict(_read_json(status_path))
            status.update(status="complete", completed_at_utc=_utc_now())
            status.pop("current_target", None)
            status.pop("current_target_index", None)
            status.pop("current_defense", None)
            status.pop("current_defense_index", None)
            atomic_json(status_path, status)
            if stage == "attacked":
                events.put(
                    {
                        "type": "group_complete",
                        "slot": slot,
                        "current": {
                            "record_type": "attacked",
                            "source": source,
                            "method": method,
                        },
                    }
                )
        events.put({"type": "worker_done", "slot": slot, "stage": stage})
    except BaseException as exc:
        events.put(
            {
                "type": "fatal",
                "slot": slot,
                "current": None,
                "records": [],
                "reason": "defense worker failed: {}: {}".format(
                    type(exc).__name__, exc
                ),
            }
        )


def _run_stage(
    registry: Registry,
    run_dir: Path,
    source_run: Path,
    devices: Sequence[str],
    stage: str,
    tasks: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    defenses: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    download_weights: bool,
    records: List[Dict[str, Any]],
    expected_keys: Sequence[tuple],
    execution_state: MutableMapping[str, Any],
) -> None:
    slots = round_robin_indices(len(tasks), len(devices))
    assignments = [[dict(tasks[index]) for index in indices] for indices in slots]
    context = mp.get_context("spawn")
    events = context.Queue()
    processes = []
    for slot, device in enumerate(devices):
        process = context.Process(
            target=_worker_main,
            args=(
                str(registry.root),
                str(run_dir),
                str(source_run),
                str(device),
                slot,
                stage,
                assignments[slot],
                list(targets),
                [dict(item) for item in defenses],
                list(image_ids),
                bool(download_weights),
                events,
            ),
            name="defense-{}-worker-{}".format(stage, slot),
        )
        process.start()
        processes.append(process)
    workers = [
        {
            "slot": slot,
            "device": str(device),
            "stage": stage,
            "status": "running",
            "current": None,
        }
        for slot, device in enumerate(devices)
    ]
    execution_state["workers"] = workers
    execution_state["phase"] = (
        "clean_preprocessing_and_evaluation"
        if stage == "clean"
        else "attacked_preprocessing_and_evaluation"
    )
    state_path = run_dir / "execution_state.json"
    records_path = run_dir / "records.json"

    def publish_state() -> None:
        active = [
            worker["current"]
            for worker in workers
            if worker["status"] == "running" and worker.get("current")
        ]
        execution_state["current"] = active[0] if active else None
        execution_state["updated_at_utc"] = _utc_now()
        atomic_json(state_path, execution_state)

    publish_state()
    completed_workers = 0
    failure = ""
    normal_completion = False
    try:
        while completed_workers < len(processes):
            try:
                event = events.get(timeout=2.0)
            except queue.Empty:
                unexpected = [
                    (slot, process.exitcode)
                    for slot, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                if unexpected:
                    failure = "Defense worker exited unexpectedly: {}".format(unexpected)
                    break
                continue
            slot = int(event["slot"])
            event_type = str(event["type"])
            if event.get("current") is not None:
                workers[slot]["current"] = dict(event["current"])
            if event_type == "task_started":
                publish_state()
                continue
            if event_type == "record_complete":
                records.append(dict(event["record"]))
                records[:] = _canonical_records(records, expected_keys)
                atomic_json(records_path, records)
                counter = (
                    "clean_evaluations_completed"
                    if stage == "clean"
                    else "attack_evaluations_completed"
                )
                execution_state[counter] = int(execution_state[counter]) + 1
                publish_state()
                continue
            if event_type == "variant_complete":
                counter = (
                    "clean_variant_groups_completed"
                    if stage == "clean"
                    else "attack_variant_groups_completed"
                )
                execution_state[counter] = int(execution_state[counter]) + 1
                publish_state()
                continue
            if event_type == "group_complete":
                execution_state["attack_groups_completed"] = int(
                    execution_state["attack_groups_completed"]
                ) + 1
                publish_state()
                continue
            if event_type == "worker_done":
                if workers[slot]["status"] != "complete":
                    workers[slot]["status"] = "complete"
                    workers[slot]["current"] = None
                    completed_workers += 1
                publish_state()
                continue
            if event_type == "fatal":
                for record in event.get("records", []):
                    records.append(dict(record))
                records[:] = _canonical_records(records, expected_keys)
                atomic_json(records_path, records)
                execution_state["failed_records"] = sum(
                    1 for record in records if record.get("status") == "failed"
                )
                workers[slot]["status"] = "failed"
                failure = str(event.get("reason") or "defense worker failed")
                publish_state()
                break
            failure = "Unknown defense worker event: {}".format(event_type)
            break
        if failure:
            raise RuntimeError(failure)
        normal_completion = True
    finally:
        if normal_completion:
            for process in processes:
                process.join(timeout=30.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=30.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=10.0)
        events.close()
        events.join_thread()


def enrich_defense_records(
    records: Sequence[Mapping[str, Any]], metrics: Sequence[str] = METRICS
) -> List[Dict[str, Any]]:
    result = [dict(record) for record in records]
    clean = {
        (str(record["defense"]), str(record["target"])): record
        for record in result
        if record.get("record_type") == "clean" and record.get("status") == "complete"
    }
    attacked = {
        (
            str(record["source"]),
            str(record["attack"]),
            str(record["defense"]),
            str(record["target"]),
        ): record
        for record in result
        if record.get("record_type") == "attacked" and record.get("status") == "complete"
    }
    for record in result:
        if record.get("status") != "complete":
            continue
        defense = str(record["defense"])
        target = str(record["target"])
        if record.get("record_type") == "clean":
            identity = clean.get(("identity", target))
            if identity is None:
                raise RuntimeError("Missing identity clean record for {}".format(target))
            record["clean_utility"] = {
                metric: float(identity["metrics"][metric])
                - float(record["metrics"][metric])
                for metric in metrics
            }
            continue
        source = str(record["source"])
        method = str(record["attack"])
        clean_defense = clean.get((defense, target))
        clean_identity = clean.get(("identity", target))
        attack_identity = attacked.get((source, method, "identity", target))
        if clean_defense is None or clean_identity is None or attack_identity is None:
            raise RuntimeError(
                "Missing matched clean/identity record for {}/{}/{}/{}".format(
                    source, method, defense, target
                )
            )
        derived = {}
        for metric in metrics:
            clean_value = float(clean_defense["metrics"][metric])
            clean_identity_value = float(clean_identity["metrics"][metric])
            attacked_value = float(record["metrics"][metric])
            attack_identity_value = float(attack_identity["metrics"][metric])
            residual = clean_value - attacked_value
            derived[metric] = {
                "clean_defended": clean_value,
                "attacked_defended": attacked_value,
                "clean_utility_drop": clean_identity_value - clean_value,
                "defense_recovery": attacked_value - attack_identity_value,
                "residual_attack_effect": residual,
                "normalized_residual_attack_effect": (
                    residual / clean_value if clean_value > 0.0 else None
                ),
            }
        record["defense_metrics"] = derived
    return result


def _finite_mean(values: Iterable[float]) -> Optional[float]:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return mean(selected) if selected else None


def defense_summary_rows(
    records: Sequence[Mapping[str, Any]],
    defense_order: Sequence[str],
    metric: str = "bbox_mAP_50",
) -> List[Dict[str, Any]]:
    rows = []
    for defense in defense_order:
        clean = [
            record
            for record in records
            if record.get("record_type") == "clean"
            and record.get("defense") == defense
            and record.get("status") == "complete"
        ]
        blackbox = [
            record
            for record in records
            if record.get("record_type") == "attacked"
            and record.get("defense") == defense
            and record.get("status") == "complete"
            and record.get("source") != record.get("target")
        ]
        rows.append(
            {
                "defense": defense,
                "clean_ap": _finite_mean(
                    record["metrics"][metric] for record in clean
                ),
                "clean_utility_drop": _finite_mean(
                    record["clean_utility"][metric] for record in clean
                ),
                "attacked_bb_ap": _finite_mean(
                    record["metrics"][metric] for record in blackbox
                ),
                "defense_recovery_bb": _finite_mean(
                    record["defense_metrics"][metric]["defense_recovery"]
                    for record in blackbox
                ),
                "residual_attack_effect_bb": _finite_mean(
                    record["defense_metrics"][metric]["residual_attack_effect"]
                    for record in blackbox
                ),
                "normalized_residual_attack_effect_bb": _finite_mean(
                    record["defense_metrics"][metric][
                        "normalized_residual_attack_effect"
                    ]
                    for record in blackbox
                    if record["defense_metrics"][metric][
                        "normalized_residual_attack_effect"
                    ]
                    is not None
                ),
                "clean_cells": len(clean),
                "blackbox_cells": len(blackbox),
            }
        )
    return rows


def method_defense_rows(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    defenses: Sequence[str],
    metric: str = "bbox_mAP_50",
) -> List[Dict[str, Any]]:
    rows = []
    for method in methods:
        for defense in defenses:
            selected = [
                record
                for record in records
                if record.get("record_type") == "attacked"
                and record.get("attack") == method
                and record.get("defense") == defense
                and record.get("status") == "complete"
                and record.get("source") != record.get("target")
            ]
            rows.append(
                {
                    "method": method,
                    "defense": defense,
                    "attacked_bb_ap": _finite_mean(
                        record["metrics"][metric] for record in selected
                    ),
                    "defense_recovery_bb": _finite_mean(
                        record["defense_metrics"][metric]["defense_recovery"]
                        for record in selected
                    ),
                    "residual_attack_effect_bb": _finite_mean(
                        record["defense_metrics"][metric]["residual_attack_effect"]
                        for record in selected
                    ),
                    "blackbox_cells": len(selected),
                }
            )
    return rows


def _format_metric(value: Optional[float]) -> str:
    return "--" if value is None else "{:.4f}".format(float(value))


def _ranked_cell(
    value: Optional[float], values: Sequence[Optional[float]], lower_is_better: bool
) -> str:
    text = _format_metric(value)
    if value is None:
        return text
    finite = sorted(
        {float(item) for item in values if item is not None},
        reverse=not lower_is_better,
    )
    if finite and math.isclose(float(value), finite[0], rel_tol=0.0, abs_tol=1e-15):
        return "\\textbf{{{}}}".format(text)
    if len(finite) > 1 and math.isclose(
        float(value), finite[1], rel_tol=0.0, abs_tol=1e-15
    ):
        return "\\underline{{{}}}".format(text)
    return text


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_text(path, "")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_tex(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> str:
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{COCO retained-500 clean utility and Common-2 black-box AP@.50 under deterministic oblivious victim-side input preprocessing. Values are computed from unrounded JSON; this is not a full-val2017 result.}",
        r"\label{tab:coco_retained500_preprocessing_defense_summary}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Defense & Clean AP@.50 & Clean drop & Adv. BB AP@.50 & Recovery & Residual effect & Norm. residual \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                labels[str(row["defense"])],
                _format_metric(row["clean_ap"]),
                _format_metric(row["clean_utility_drop"]),
                _format_metric(row["attacked_bb_ap"]),
                _format_metric(row["defense_recovery_bb"]),
                _format_metric(row["residual_attack_effect_bb"]),
                _format_metric(row["normalized_residual_attack_effect_bb"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _matrix_tex(
    rows: Sequence[Mapping[str, Any]],
    registry: Registry,
    methods: Sequence[str],
    defenses: Sequence[str],
    labels: Mapping[str, str],
    field: str,
    caption: str,
    label: str,
    lower_is_better: bool,
) -> str:
    lookup = {
        (str(row["method"]), str(row["defense"])): row.get(field)
        for row in rows
    }
    columns = "l" + "r" * len(defenses)
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        "\\caption{{{}}}".format(caption),
        "\\label{{{}}}".format(label),
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        "\\begin{{tabular}}{{{}}}".format(columns),
        r"\toprule",
        "Method & {} \\\\".format(" & ".join(labels[item] for item in defenses)),
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for defense in defenses:
            value = lookup.get((method, defense))
            peer_values = [lookup.get((peer, defense)) for peer in methods]
            cells.append(_ranked_cell(value, peer_values, lower_is_better))
        lines.append(
            "{} & {} \\\\".format(
                registry.attack(method).display_name, " & ".join(cells)
            )
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    )
    return "\n".join(lines)


def write_defense_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    report_dir: Path,
    methods: Sequence[str],
    defenses: Sequence[str],
) -> Dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        defense: str(registry.preprocessing_defenses[defense]["label"])
        for defense in defenses
    }
    summary = defense_summary_rows(records, defenses)
    matrix = method_defense_rows(records, methods, defenses)
    summary_csv = report_dir / "defense_summary_bbox_mAP_50.csv"
    matrix_csv = report_dir / "method_by_defense_bbox_mAP_50.csv"
    records_csv = report_dir / "records.csv"
    _write_csv(summary_csv, summary)
    _write_csv(matrix_csv, matrix)
    flat = []
    for record in records:
        row = {
            "record_type": record.get("record_type"),
            "source": record.get("source"),
            "method": record.get("attack"),
            "defense": record.get("defense"),
            "target": record.get("target"),
            "status": record.get("status"),
            "images": record.get("images"),
        }
        for metric in METRICS:
            row[metric] = record.get("metrics", {}).get(metric)
        derived = record.get("defense_metrics", {}).get("bbox_mAP_50", {})
        for key in (
            "clean_defended",
            "attacked_defended",
            "clean_utility_drop",
            "defense_recovery",
            "residual_attack_effect",
            "normalized_residual_attack_effect",
        ):
            row[key] = derived.get(key)
        flat.append(row)
    _write_csv(records_csv, flat)
    summary_tex = report_dir / "defense_summary_bbox_mAP_50.tex"
    attack_tex = report_dir / "method_by_defense_bb_mean_bbox_mAP_50.tex"
    recovery_tex = report_dir / "method_by_defense_recovery_bbox_mAP_50.tex"
    _atomic_text(summary_tex, _summary_tex(summary, labels))
    _atomic_text(
        attack_tex,
        _matrix_tex(
            matrix,
            registry,
            methods,
            defenses,
            labels,
            "attacked_bb_ap",
            "COCO retained-500 Common-2 black-box AP@.50 after oblivious victim-side preprocessing. Lower values indicate attacks that remain stronger after preprocessing; source-matched targets are excluded.",
            "tab:coco_retained500_method_defense_bb_ap50",
            True,
        ),
    )
    _atomic_text(
        recovery_tex,
        _matrix_tex(
            matrix,
            registry,
            methods,
            defenses,
            labels,
            "defense_recovery_bb",
            "COCO retained-500 Common-2 black-box AP@.50 recovery relative to identity preprocessing. Higher values indicate greater defensive recovery; source-matched targets are excluded.",
            "tab:coco_retained500_method_defense_recovery_ap50",
            False,
        ),
    )
    outputs = {
        "summary_csv": summary_csv,
        "matrix_csv": matrix_csv,
        "records_csv": records_csv,
        "summary_tex": summary_tex,
        "attack_tex": attack_tex,
        "recovery_tex": recovery_tex,
    }
    if matrix and any(row.get("attacked_bb_ap") is not None for row in matrix):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        lookup = {
            (str(row["method"]), str(row["defense"])): row["attacked_bb_ap"]
            for row in matrix
        }
        values = np.asarray(
            [[lookup[(method, defense)] for defense in defenses] for method in methods],
            dtype=float,
        )
        figure, axis = plt.subplots(
            figsize=(max(10.0, len(defenses) * 0.9), max(5.0, len(methods) * 0.45))
        )
        image = axis.imshow(values, aspect="auto", cmap="viridis_r")
        axis.set_xticks(range(len(defenses)))
        axis.set_xticklabels([labels[item] for item in defenses], rotation=40, ha="right")
        axis.set_yticks(range(len(methods)))
        axis.set_yticklabels([registry.attack(item).display_name for item in methods])
        axis.set_title("Retained-500 Common-2 black-box AP@.50 after preprocessing")
        figure.colorbar(image, ax=axis, label="AP@.50 (lower = stronger attack)")
        figure.tight_layout()
        figures = report_dir / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        png = figures / "method_by_defense_bb_mean_bbox_mAP_50.png"
        pdf = figures / "method_by_defense_bb_mean_bbox_mAP_50.pdf"
        figure.savefig(png, dpi=180)
        figure.savefig(pdf)
        plt.close(figure)
        outputs.update(heatmap_png=png, heatmap_pdf=pdf)
    return outputs


def _git_provenance(root: Path) -> Dict[str, Any]:
    def command(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    tracked_status = command("status", "--short")
    files = [
        root / "configs" / "experiments" / "protocols.yaml",
        root / "configs" / "experiments" / "preprocessing_defenses.yaml",
        root / "experiments" / "coco_preprocessing_defense.py",
        root / "src" / "lgp" / "runners" / "preprocessing_defense.py",
    ]
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "git_head": command("rev-parse", "HEAD") or None,
        "worktree_dirty": bool(tracked_status),
        "worktree_status": tracked_status.splitlines(),
        "implementation_sha256": {
            _relative(path, root): file_digest(path) for path in files
        },
        "runtime": {
            "python": "{}.{}.{}".format(*__import__("sys").version_info[:3]),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pillow": PILLOW_VERSION,
        },
    }


def _write_assignments(
    run_dir: Path,
    devices: Sequence[str],
    defenses: Sequence[str],
    sources: Sequence[str],
    methods: Sequence[str],
) -> Dict[str, Any]:
    clean_tasks = [{"defense": defense} for defense in defenses]
    attack_tasks = [
        {"source": source, "method": method}
        for source in sources
        for method in methods
    ]
    clean_slots = round_robin_indices(len(clean_tasks), len(devices))
    attack_slots = round_robin_indices(len(attack_tasks), len(devices))
    payload = {
        "schema_version": 1,
        "algorithm": "canonical_round_robin_v1",
        "worker_count": len(devices),
        "devices": list(devices),
        "clean_barrier_before_attacked": True,
        "global_record_writer": "coordinator_only",
        "clean": [
            {**clean_tasks[index], "worker_slot": slot, "device": devices[slot]}
            for slot, indices in enumerate(clean_slots)
            for index in indices
        ],
        "attacked": [
            {**attack_tasks[index], "worker_slot": slot, "device": devices[slot]}
            for slot, indices in enumerate(attack_slots)
            for index in indices
        ],
    }
    atomic_json(run_dir / "execution_assignments.json", payload)
    return payload


def run_preprocessing_defense(
    registry: Registry,
    execute: bool = True,
    output_dir: Optional[Path] = None,
    source_acceptance: Optional[Path] = None,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
    device: str = "cuda:0",
    devices: Optional[Sequence[str]] = None,
    download_weights: bool = False,
) -> Path:
    plan = build_defense_plan(
        registry,
        source_override=source_override,
        method_override=method_override,
        target_override=target_override,
        defense_override=defense_override,
        max_images=max_images,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_dir
        or project_root()
        / "outputs"
        / "experiments"
        / "coco_preprocessing_defense"
        / stamp
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty experiment: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "plan.json", plan)
    _write_plan_csv(run_dir / "plan.csv", plan["jobs"])
    atomic_json(run_dir / "records.json", [])
    selection = _protocol_selection(
        registry,
        source_override=source_override,
        method_override=method_override,
        target_override=target_override,
        defense_override=defense_override,
    )
    sources = selection["sources"]
    methods = selection["methods"]
    targets = selection["targets"]
    defense_records = [
        {"id": defense_id, **spec}
        for defense_id, spec in selection["defenses"]
    ]
    defense_ids = [item["id"] for item in defense_records]
    if not execute:
        write_defense_reports(
            [], registry, run_dir / "reports" / "coco", methods, defense_ids
        )
        atomic_json(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "planned",
                "protocol": PROTOCOL_ID,
                "jobs": plan["counts"]["records"],
                "ready": plan["counts"]["records"],
                "records": 0,
                "max_images": max_images,
                "retained500_benchmark_eligible": False,
                "full_coco_val_claim": False,
            },
        )
        return run_dir

    execution_devices = normalize_execution_devices(device, devices)
    validate_available_cuda_devices(execution_devices)
    assignments = _write_assignments(
        run_dir, execution_devices, defense_ids, sources, methods
    )
    counts = plan["counts"]
    execution_state: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "validating_source_evidence",
        "protocol": PROTOCOL_ID,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "clean_variant_groups_completed": 0,
        "clean_variant_groups_total": counts["clean_variant_groups"],
        "clean_evaluations_completed": 0,
        "clean_evaluations_total": counts["clean_evaluations"],
        "attack_groups_completed": 0,
        "attack_groups_total": counts["attack_groups"],
        "attack_variant_groups_completed": 0,
        "attack_variant_groups_total": counts["attack_variant_groups"],
        "attack_evaluations_completed": 0,
        "attack_evaluations_total": counts["attack_evaluations"],
        "failed_records": 0,
        "workers": [],
        "current": None,
        "execution_scheduler": assignments,
    }
    atomic_json(run_dir / "execution_state.json", execution_state)
    records: List[Dict[str, Any]] = []
    expected_keys = _expected_record_keys(plan)
    try:
        source_evidence = verify_source_evidence(
            registry,
            acceptance_path=source_acceptance,
            deep=True,
            max_images=max_images,
        )
        atomic_json(run_dir / "source_evidence.json", source_evidence)
        source_run = registry.root / source_evidence["source_run"]
        selected_count = RETAINED_IMAGES if max_images is None else min(max_images, RETAINED_IMAGES)
        image_ids = list(source_evidence["retained_image_ids"][:selected_count])
        clean_tasks = [{"defense": defense_id} for defense_id in defense_ids]
        attack_tasks = [
            {"source": source, "method": method}
            for source in sources
            for method in methods
        ]
        _run_stage(
            registry,
            run_dir,
            source_run,
            execution_devices,
            "clean",
            clean_tasks,
            targets,
            defense_records,
            image_ids,
            download_weights,
            records,
            expected_keys,
            execution_state,
        )
        if int(execution_state["clean_evaluations_completed"]) != counts["clean_evaluations"]:
            raise RuntimeError("Clean preprocessing barrier is incomplete")
        execution_state.update(
            phase="clean_barrier_complete",
            workers=[],
            current=None,
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", execution_state)
        _run_stage(
            registry,
            run_dir,
            source_run,
            execution_devices,
            "attacked",
            attack_tasks,
            targets,
            defense_records,
            image_ids,
            download_weights,
            records,
            expected_keys,
            execution_state,
        )
        records = enrich_defense_records(records)
        records = _canonical_records(records, expected_keys)
        atomic_json(run_dir / "records.json", records)
        failures = [record for record in records if record.get("status") != "complete"]
        if len(records) != counts["records"] or failures:
            raise RuntimeError(
                "Defense record gate failed: records={}, expected={}, failures={}".format(
                    len(records), counts["records"], len(failures)
                )
            )
        if (run_dir / "scratch").exists():
            remaining = [path for path in (run_dir / "scratch").rglob("*") if path.is_file()]
            if remaining:
                raise RuntimeError("Verified transformed scratch files remain after execution")
            shutil.rmtree(str(_inside(run_dir / "scratch", run_dir)))
        execution_state.update(
            phase="writing_reports_and_acceptance",
            workers=[],
            current=None,
            failed_records=0,
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", execution_state)
        reports = write_defense_reports(
            records, registry, run_dir / "reports" / "coco", methods, defense_ids
        )
        provenance = _git_provenance(registry.root)
        atomic_json(run_dir / "provenance.json", provenance)
        exact_formal_scope = (
            max_images is None
            and bool(plan["full_registered_selection"])
            and len(records) == 4048
            and int(execution_state["clean_evaluations_completed"]) == 176
            and int(execution_state["attack_evaluations_completed"]) == 3872
            and int(execution_state["attack_variant_groups_completed"]) == 242
            and int(execution_state["attack_groups_completed"]) == 22
            and source_evidence["deep_payload_hash_verification"] is True
            and int(source_evidence["verified_images_per_payload"]) == RETAINED_IMAGES
        )
        acceptance = {
            "schema_version": 1,
            "status": "accepted" if exact_formal_scope else "diagnostic_complete",
            "protocol": PROTOCOL_ID,
            "accepted_at_utc": _utc_now(),
            "retained500_benchmark_eligible": bool(exact_formal_scope),
            "full_coco_val_claim_eligible": False,
            "scope": "COCO retained-500 oblivious preprocessing defense only",
            "common_2_is_paper_term": False,
            "threat_model": dict(registry.preprocessing_defense_threat_model),
            "counts": {
                "records": len(records),
                "clean_evaluations": execution_state["clean_evaluations_completed"],
                "attack_evaluations": execution_state["attack_evaluations_completed"],
                "clean_variant_groups": execution_state["clean_variant_groups_completed"],
                "attack_variant_groups": execution_state["attack_variant_groups_completed"],
                "attack_groups": execution_state["attack_groups_completed"],
                "failed_records": 0,
                "retained_images": selected_count,
                "sources": len(sources),
                "methods": len(methods),
                "targets": len(targets),
                "defenses": len(defense_ids),
                "workers": len(execution_devices),
            },
            "evidence_sha256": {
                "plan.json": file_digest(run_dir / "plan.json"),
                "records.json": file_digest(run_dir / "records.json"),
                "source_evidence.json": file_digest(run_dir / "source_evidence.json"),
                "execution_assignments.json": file_digest(run_dir / "execution_assignments.json"),
                "provenance.json": file_digest(run_dir / "provenance.json"),
                **{
                    _relative(path, run_dir): file_digest(path)
                    for path in reports.values()
                },
            },
            "limitations": [
                "The attacker is oblivious to preprocessing; no BPDA, EOT or query adaptation is evaluated.",
                "Metrics cover the one pre-frozen retained 500-image subset and must not be reported as full-val2017 COCO AP.",
                "No defended checkpoint, pruning, fine-tuning or adversarial training is evaluated.",
            ],
        }
        atomic_json(
            run_dir / "formal_preprocessing_defense_acceptance.json", acceptance
        )
        summary = {
            "schema_version": 1,
            "status": "complete",
            "protocol": PROTOCOL_ID,
            "started_at_utc": execution_state["started_at_utc"],
            "completed_at_utc": _utc_now(),
            "records": len(records),
            "failed_records": 0,
            "max_images": max_images,
            "retained500_benchmark_eligible": bool(exact_formal_scope),
            "full_coco_val_claim": False,
            "counts": acceptance["counts"],
        }
        atomic_json(run_dir / "summary.json", summary)
        execution_state.update(
            status="complete",
            phase="complete",
            workers=[],
            current=None,
            failed_records=0,
            completed_at_utc=summary["completed_at_utc"],
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", execution_state)
        write_artifact_manifest(run_dir)
        return run_dir
    except BaseException as exc:
        reason = "{}: {}".format(type(exc).__name__, exc)
        execution_state.update(
            status="failed",
            phase="failed",
            workers=[],
            current=None,
            reason=reason,
            failed_records=sum(
                1 for record in records if record.get("status") == "failed"
            ),
            failed_at_utc=_utc_now(),
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", execution_state)
        atomic_json(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "failed",
                "protocol": PROTOCOL_ID,
                "reason": reason,
                "records": len(records),
                "failed_records": execution_state["failed_records"],
                "retained500_benchmark_eligible": False,
                "full_coco_val_claim": False,
                "failed_at_utc": execution_state["failed_at_utc"],
            },
        )
        raise
