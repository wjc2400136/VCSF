from __future__ import annotations

import csv
import gc
import json
import math
import multiprocessing as mp
import queue
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import __version__ as PILLOW_VERSION

from ..defenses import AdaptivePreprocessor
from ..io import atomic_json, file_digest
from ..paths import project_root
from ..registry import Registry
from ..reporting.all_methods import write_artifact_manifest
from .attack import run_attack
from .evaluate import run_evaluation
from .formal_parallel import (
    normalize_execution_devices,
    round_robin_indices,
    validate_available_cuda_devices,
)
from .preprocessing_defense import (
    METRICS,
    RETAINED_IMAGES,
    _atomic_text,
    _canonical_json_digest,
    _inside,
    _materialize_view,
    _read_json,
    _relative,
    _remove_verified_scratch,
    selected_defenses,
)


PROTOCOL_ID = "coco_retained500_common2_adaptive_preprocessing_defense"
PAIRED_PROTOCOL_ID = "coco_retained500_common2_preprocessing_defense"
FORMAL_RECORDS = 3872
FORMAL_GROUPS = 22
FORMAL_PAYLOAD_GROUPS = 242


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _selection(
    registry: Registry,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    protocol = dict(registry.protocols[PROTOCOL_ID])
    if str(protocol.get("status", "active")) != "active":
        raise RuntimeError(
            "The adaptive preprocessing protocol is historical after the "
            "SFIM-B/NAA fidelity revision and cannot be regenerated"
        )
    sources = list(source_override or protocol["sources"])
    methods = list(method_override or protocol["methods"])
    targets = list(target_override or registry.paper_order)
    defenses = selected_defenses(registry, defense_override)
    if len(sources) != len(set(sources)):
        raise ValueError("Source selection contains duplicates")
    if len(methods) != len(set(methods)):
        raise ValueError("Method selection contains duplicates")
    if len(targets) != len(set(targets)):
        raise ValueError("Target selection contains duplicates")
    for source in sources:
        if source not in protocol["sources"]:
            raise ValueError("Source '{}' is outside Common-2".format(source))
        registry.model(source)
    for method in methods:
        registry.attack(method)
    for target in targets:
        registry.model(target)
    for source in sources:
        for method in methods:
            status = registry.compatibility_status(method, source)["status"]
            if status != "native":
                raise RuntimeError(
                    "Adaptive Common-2 requires native compatibility: {}/{} is {}".format(
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


def build_adaptive_defense_plan(
    registry: Registry,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    if max_images is not None and max_images <= 0:
        raise ValueError("--max-images must be positive")
    selected = _selection(
        registry,
        source_override,
        method_override,
        target_override,
        defense_override,
    )
    sources = selected["sources"]
    methods = selected["methods"]
    targets = selected["targets"]
    defenses = selected["defenses"]
    jobs = []
    for source in sources:
        for method in methods:
            for defense_id, spec in defenses:
                for target in targets:
                    jobs.append(
                        {
                            "record_type": "adaptive_attacked",
                            "dataset": "coco",
                            "split": "val",
                            "source": source,
                            "attack": method,
                            "defense": defense_id,
                            "defense_parameters": dict(spec),
                            "target": target,
                            "status": "ready",
                        }
                    )
    full_selection = (
        sources == list(selected["protocol"]["sources"])
        and methods == list(selected["protocol"]["methods"])
        and targets == list(registry.paper_order)
        and [item[0] for item in defenses]
        == list(registry.preprocessing_defense_order)
    )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "dataset": "coco",
        "split": "val",
        "scope": "paired retained-500 adaptive preprocessing benchmark",
        "full_coco_val_claim": False,
        "certified_robustness_claim": False,
        "threat_model": dict(registry.adaptive_preprocessing_policy),
        "common_2": {
            "sources": sources,
            "definition": selected["protocol"]["common_2_definition"],
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
        "paired_clean_evaluations_reused": len(defenses) * len(targets),
        "counts": {
            "attack_evaluations": len(jobs),
            "records": len(jobs),
            "attack_payload_groups": len(sources) * len(methods) * len(defenses),
            "attack_groups": len(sources) * len(methods),
        },
        "jobs": jobs,
    }


def _write_plan_csv(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in jobs:
            row = dict(item)
            row["defense_parameters"] = json.dumps(
                row["defense_parameters"], sort_keys=True
            )
            writer.writerow({field: row.get(field) for field in fields})


def _verify_manifest_sidecar(run_dir: Path, root: Path) -> Dict[str, Any]:
    manifest = run_dir / "artifact_manifest.json"
    sidecar = run_dir / "artifact_manifest.sha256"
    if not manifest.is_file() or not sidecar.is_file():
        raise FileNotFoundError("Paired run has no artifact manifest and sidecar")
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != manifest.name:
        raise RuntimeError("Paired artifact-manifest sidecar is malformed")
    actual = file_digest(manifest)
    if parts[0] != actual:
        raise RuntimeError("Paired artifact-manifest sidecar hash mismatch")
    payload = _read_json(manifest)
    if payload.get("algorithm") != "sha256":
        raise RuntimeError("Paired artifact manifest uses an unexpected algorithm")
    return {
        "artifact_manifest": _relative(manifest, root),
        "artifact_manifest_sha256": actual,
        "artifact_manifest_file_count": int(payload.get("file_count", -1)),
    }


def verify_paired_oblivious_evidence(
    registry: Registry,
    acceptance_path: Optional[Path] = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    protocol = registry.protocols[PROTOCOL_ID]
    default_value = Path(str(protocol["paired_oblivious_acceptance"]))
    acceptance = (
        acceptance_path
        or (default_value if default_value.is_absolute() else registry.root / default_value)
    ).resolve()
    if not acceptance.is_file():
        raise FileNotFoundError("Missing paired oblivious acceptance: {}".format(acceptance))
    run_dir = acceptance.parent.resolve()
    accepted = _read_json(acceptance)
    if accepted.get("protocol") != PAIRED_PROTOCOL_ID:
        raise RuntimeError("Paired acceptance has the wrong protocol")
    if accepted.get("status") != "accepted":
        raise RuntimeError("Paired oblivious run is not accepted")
    if accepted.get("retained500_benchmark_eligible") is not True:
        raise RuntimeError("Paired oblivious run is not retained-500 eligible")
    counts = dict(accepted.get("counts", {}))
    required = {
        "records": 4048,
        "clean_evaluations": 176,
        "attack_evaluations": 3872,
        "clean_variant_groups": 11,
        "attack_variant_groups": 242,
        "attack_groups": 22,
        "failed_records": 0,
        "retained_images": 500,
        "sources": 2,
        "methods": 11,
        "targets": 16,
        "defenses": 11,
    }
    for key, expected in required.items():
        if int(counts.get(key, -1)) != expected:
            raise RuntimeError(
                "Paired acceptance count {} is {}, expected {}".format(
                    key, counts.get(key), expected
                )
            )
    evidence_hashes = dict(accepted.get("evidence_sha256", {}))
    for relative in ("plan.json", "records.json", "source_evidence.json", "provenance.json"):
        expected = evidence_hashes.get(relative)
        path = run_dir / relative
        if expected is None or file_digest(path) != expected:
            raise RuntimeError("Paired evidence hash mismatch for {}".format(relative))
    summary = _read_json(run_dir / "summary.json")
    if summary.get("status") != "complete" or summary.get(
        "retained500_benchmark_eligible"
    ) is not True:
        raise RuntimeError("Paired oblivious summary is not complete and eligible")
    source = _read_json(run_dir / "source_evidence.json")
    retained_ids = [int(value) for value in source.get("retained_image_ids", [])]
    if len(retained_ids) != RETAINED_IMAGES:
        raise RuntimeError("Paired evidence does not freeze exactly 500 image IDs")
    if _canonical_json_digest(retained_ids) != source.get(
        "retained_image_ids_sha256"
    ):
        raise RuntimeError("Paired retained-image digest mismatch")
    if source.get("deep_payload_hash_verification") is not True:
        raise RuntimeError("Paired source payloads were not deeply verified")
    if int(source.get("verified_images_per_payload", -1)) != RETAINED_IMAGES:
        raise RuntimeError("Paired source payload verification is incomplete")
    groups = list(source.get("groups", []))
    if len(groups) != FORMAL_GROUPS or any(
        int(item.get("payload_hashes_verified", -1)) != RETAINED_IMAGES
        for item in groups
    ):
        raise RuntimeError("Paired source evidence is not 22 x 500 verified payloads")
    records = list(_read_json(run_dir / "records.json"))
    clean = [
        dict(record)
        for record in records
        if record.get("record_type") == "clean" and record.get("status") == "complete"
    ]
    oblivious = [
        dict(record)
        for record in records
        if record.get("record_type") == "attacked" and record.get("status") == "complete"
    ]
    if len(clean) != 176 or len(oblivious) != 3872:
        raise RuntimeError("Paired records have unexpected clean/attacked counts")
    selected = RETAINED_IMAGES if max_images is None else min(max_images, RETAINED_IMAGES)
    manifest = _verify_manifest_sidecar(run_dir, registry.root)
    return {
        "schema_version": 1,
        "status": "verified",
        "verified_at_utc": _utc_now(),
        "acceptance": _relative(acceptance, registry.root),
        "acceptance_sha256": file_digest(acceptance),
        "paired_run": _relative(run_dir, registry.root),
        "paired_protocol": PAIRED_PROTOCOL_ID,
        "paired_counts": required,
        "paired_records_sha256": file_digest(run_dir / "records.json"),
        "paired_source_evidence_sha256": file_digest(run_dir / "source_evidence.json"),
        "retained_image_ids": retained_ids,
        "retained_image_ids_sha256": source["retained_image_ids_sha256"],
        "selected_images": selected,
        "source_payload_groups": len(groups),
        "source_payload_hashes_verified_per_group": RETAINED_IMAGES,
        "clean_reference_records": len(clean),
        "oblivious_reference_records": len(oblivious),
        **manifest,
    }


def _record_key(record: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(record.get("source", "")),
        str(record.get("attack", "")),
        str(record.get("defense", "")),
        str(record.get("target", "")),
    )


def _canonical_records(
    records: Sequence[Mapping[str, Any]], jobs: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    order = {_record_key(job): index for index, job in enumerate(jobs)}
    if len(order) != len(jobs):
        raise RuntimeError("Adaptive plan contains duplicate record keys")
    found: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for record in records:
        key = _record_key(record)
        if key not in order:
            raise RuntimeError("Unexpected adaptive record key: {}".format(key))
        if key in found:
            raise RuntimeError("Duplicate adaptive record key: {}".format(key))
        found[key] = dict(record)
    return sorted(found.values(), key=lambda item: order[_record_key(item)])


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _payload_manifest(
    attack_dir: Path,
    destination: Path,
    source: str,
    method: str,
    defense_id: str,
    defense_spec: Mapping[str, Any],
    expected_ids: Sequence[int],
) -> Dict[str, Any]:
    run = _read_json(attack_dir / "run.json")
    rows = []
    with (attack_dir / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("status") != "ok":
                    raise RuntimeError("Adaptive attack manifest contains a failed image")
                rows.append(
                    {
                        "image_id": int(row["image_id"]),
                        "output_file": str(row["output_file"]),
                        "output_bytes": int(row["output_bytes"]),
                        "output_sha256": str(row["output_sha256"]),
                        "linf_pixel": float(row["linf_pixel"]),
                    }
                )
    ids = [int(item["image_id"]) for item in rows]
    if ids != list(expected_ids):
        raise RuntimeError("Adaptive payload image IDs differ from the frozen order")
    annotation_ids = [
        int(item["id"])
        for item in _read_json(attack_dir / "annotations.json").get("images", [])
    ]
    if annotation_ids != sorted(expected_ids):
        raise RuntimeError("Adaptive annotation image IDs differ from canonical order")
    if run.get("status") != "complete" or int(run.get("successful_images", -1)) != len(ids):
        raise RuntimeError("Adaptive attack run is incomplete")
    if int(run.get("failed_images", -1)) != 0:
        raise RuntimeError("Adaptive attack run reports failed images")
    metadata = dict(run.get("run_metadata", {}))
    if metadata.get("protocol") != PROTOCOL_ID or metadata.get("defense") != defense_id:
        raise RuntimeError("Adaptive attack run metadata differs from its task")
    payload = {
        "schema_version": 1,
        "status": "generated_pending_target_panel",
        "source": source,
        "method": method,
        "defense": defense_id,
        "defense_parameters": dict(defense_spec),
        "images": len(rows),
        "requested_image_ids_sha256": _canonical_json_digest(ids),
        "evaluation_image_ids_sha256": _canonical_json_digest(annotation_ids),
        "attack_run": str(attack_dir),
        "attack_run_json_sha256_before_pruning": file_digest(attack_dir / "run.json"),
        "attack_manifest_sha256": file_digest(attack_dir / "manifest.jsonl"),
        "annotation_sha256": file_digest(attack_dir / "annotations.json"),
        "max_linf_pixel": max(float(item["linf_pixel"]) for item in rows),
        "outputs": rows,
        "payload_removed": False,
    }
    atomic_json(destination, payload)
    return payload


def _prune_payload(attack_dir: Path, run_dir: Path, manifest_path: Path) -> None:
    attack_root = _inside(run_dir / "adaptive_attacks", run_dir)
    attack_dir = _inside(attack_dir, attack_root)
    images = _inside(attack_dir / "images", attack_dir)
    if not images.is_dir():
        raise RuntimeError("Adaptive generated payload is missing before pruning")
    manifest = _read_json(manifest_path)
    for item in manifest["outputs"]:
        path = _inside(attack_dir / str(item["output_file"]), attack_dir)
        if not path.is_file():
            raise FileNotFoundError("Adaptive payload disappeared: {}".format(path))
        if path.stat().st_size != int(item["output_bytes"]):
            raise RuntimeError("Adaptive payload byte count changed before pruning")
        if file_digest(path) != item["output_sha256"]:
            raise RuntimeError("Adaptive payload hash changed before pruning")
    quarantine = images.with_name("images.verified-delete")
    if quarantine.exists():
        raise FileExistsError("Adaptive payload quarantine already exists")
    images.rename(quarantine)
    shutil.rmtree(str(quarantine))
    run = dict(_read_json(attack_dir / "run.json"))
    run.update(
        full_payload_available=False,
        payload_state="removed_after_complete_target_panel",
        payload_removed_at_utc=_utc_now(),
        retained_payload_manifest=_relative(manifest_path, run_dir),
    )
    atomic_json(attack_dir / "run.json", run)
    run_after_hash = file_digest(attack_dir / "run.json")
    manifest.update(
        status="complete_target_panel_and_payload_removed",
        payload_removed=True,
        payload_removed_at_utc=_utc_now(),
        attack_run_json_sha256_after_pruning=run_after_hash,
    )
    atomic_json(manifest_path, manifest)


def audit_adaptive_payload_evidence(
    run_dir: Path,
    sources: Sequence[str],
    methods: Sequence[str],
    defenses: Sequence[str],
    image_ids: Sequence[int],
) -> Dict[str, Any]:
    """Revalidate every preserved manifest after its image payload is removed."""
    expected_ids = [int(value) for value in image_ids]
    expected_evaluation_ids = sorted(expected_ids)
    requested_digest = _canonical_json_digest(expected_ids)
    evaluation_digest = _canonical_json_digest(expected_evaluation_ids)
    assignment_rows = list(_read_json(run_dir / "execution_assignments.json")["tasks"])
    worker_for = {
        (str(item["source"]), str(item["method"])): int(item["worker_slot"])
        for item in assignment_rows
    }
    groups = []
    for source in sources:
        for method in methods:
            for defense in defenses:
                manifest_path = (
                    run_dir
                    / "payload_manifests"
                    / source
                    / method
                    / "{}.json".format(defense)
                )
                manifest = _read_json(manifest_path)
                if manifest.get("status") != "complete_target_panel_and_payload_removed":
                    raise RuntimeError(
                        "Adaptive payload was not safely finalized: {}/{}/{}".format(
                            source, method, defense
                        )
                    )
                if manifest.get("payload_removed") is not True:
                    raise RuntimeError("Adaptive payload manifest is not marked removed")
                rows = list(manifest.get("outputs", []))
                ids = [int(item["image_id"]) for item in rows]
                if int(manifest.get("images", -1)) != len(expected_ids):
                    raise RuntimeError("Adaptive payload manifest has the wrong image count")
                if ids != expected_ids:
                    raise RuntimeError("Adaptive payload manifest changed the retained IDs")
                if manifest.get("requested_image_ids_sha256") != requested_digest:
                    raise RuntimeError("Adaptive payload requested-ID digest mismatch")
                if manifest.get("evaluation_image_ids_sha256") != evaluation_digest:
                    raise RuntimeError("Adaptive payload evaluation-ID digest mismatch")
                if len({str(item["output_sha256"]) for item in rows}) != len(rows):
                    raise RuntimeError("Adaptive payload output hashes contain duplicates")
                if any(
                    len(str(item.get("output_sha256", ""))) != 64
                    or int(item.get("output_bytes", 0)) <= 0
                    for item in rows
                ):
                    raise RuntimeError("Adaptive payload contains an invalid hash or byte count")
                maximum_linf = float(manifest.get("max_linf_pixel", math.inf))
                if not math.isfinite(maximum_linf) or maximum_linf > 4.0001:
                    raise RuntimeError("Adaptive payload violates the 4/255 L-infinity gate")
                attack_dir = (
                    run_dir
                    / "adaptive_attacks"
                    / "worker-{}".format(worker_for[(source, method)])
                    / source
                    / method
                    / defense
                )
                if (attack_dir / "images").exists():
                    raise RuntimeError("Verified adaptive image payload still exists")
                run_path = attack_dir / "run.json"
                run = _read_json(run_path)
                if run.get("full_payload_available") is not False:
                    raise RuntimeError("Pruned adaptive run still claims a full payload")
                if run.get("payload_state") != "removed_after_complete_target_panel":
                    raise RuntimeError("Adaptive run has the wrong post-panel payload state")
                if file_digest(run_path) != manifest.get(
                    "attack_run_json_sha256_after_pruning"
                ):
                    raise RuntimeError("Pruned adaptive run.json hash mismatch")
                if file_digest(attack_dir / "manifest.jsonl") != manifest.get(
                    "attack_manifest_sha256"
                ):
                    raise RuntimeError("Adaptive attack manifest hash mismatch")
                if file_digest(attack_dir / "annotations.json") != manifest.get(
                    "annotation_sha256"
                ):
                    raise RuntimeError("Adaptive attack annotation hash mismatch")
                groups.append(
                    {
                        "source": source,
                        "method": method,
                        "defense": defense,
                        "images": len(rows),
                        "payload_hashes_preserved": len(rows),
                        "maximum_linf_pixel": maximum_linf,
                        "manifest": _relative(manifest_path, run_dir),
                        "manifest_sha256": file_digest(manifest_path),
                        "attack_run": _relative(attack_dir, run_dir),
                        "attack_run_json_sha256": file_digest(run_path),
                    }
                )
    payload = {
        "schema_version": 1,
        "status": "verified",
        "verified_at_utc": _utc_now(),
        "groups": len(groups),
        "images_per_group": len(expected_ids),
        "payload_hashes_preserved": sum(
            int(item["payload_hashes_preserved"]) for item in groups
        ),
        "requested_image_ids_sha256": requested_digest,
        "evaluation_image_ids_sha256": evaluation_digest,
        "linf_violations": 0,
        "remaining_image_payload_directories": 0,
        "details": groups,
    }
    payload["evidence_set_sha256"] = _canonical_json_digest(
        {
            "requested_image_ids_sha256": requested_digest,
            "evaluation_image_ids_sha256": evaluation_digest,
            "groups": groups,
        }
    )
    return payload


def _evaluation_record(
    completed: Path,
    source: str,
    method: str,
    defense_id: str,
    defense_spec: Mapping[str, Any],
    paired_lookup: Mapping[Tuple[str, str, str, str], Mapping[str, Any]],
    paired_comparable: bool,
) -> Dict[str, Any]:
    record = dict(_read_json(completed / "metrics.json"))
    key = (source, method, defense_id, str(record["target"]))
    paired = paired_lookup.get(key)
    if paired is None:
        raise RuntimeError("Missing paired oblivious record for {}".format(key))
    paired_metrics = {
        metric: (
            float(paired["metrics"][metric]) if paired_comparable else None
        )
        for metric in METRICS
    }
    deltas = {
        metric: (
            float(record["metrics"][metric]) - float(paired["metrics"][metric])
            if paired_comparable
            else None
        )
        for metric in METRICS
    }
    record.update(
        {
            "record_type": "adaptive_attacked",
            "source": source,
            "attack": method,
            "defense": defense_id,
            "defense_parameters": dict(defense_spec),
            "threat_model": "adaptive_source_pipeline_bpda",
            "source_matched_target": str(record["target"]) == source,
            "paired_oblivious_metrics": paired_metrics,
            "adaptive_minus_oblivious": deltas,
            "paired_metric_comparable": paired_comparable,
        }
    )
    atomic_json(completed / "metrics.json", record)
    return record


def _worker_main(
    root: str,
    run_dir_value: str,
    device: str,
    slot: int,
    tasks: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    defenses: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    seed_offsets: Mapping[int, int],
    paired_records_value: Sequence[Mapping[str, Any]],
    max_images: Optional[int],
    download_weights: bool,
    isolated_candidate: Optional[Mapping[str, Any]],
    events: Any,
) -> None:
    try:
        if not str(device).startswith("cuda:"):
            raise RuntimeError("Adaptive parallel workers require explicit CUDA devices")
        torch.cuda.set_device(int(str(device).split(":", 1)[1]))
        registry = Registry(Path(root))
        run_dir = Path(run_dir_value)
        paired_lookup = {
            _record_key(record): dict(record) for record in paired_records_value
        }
        for task in tasks:
            source = str(task["source"])
            method = str(task["method"])
            status_path = run_dir / "group_status" / source / "{}.json".format(method)
            status: Dict[str, Any] = {
                "schema_version": 1,
                "status": "running",
                "worker_slot": slot,
                "device": device,
                "source": source,
                "method": method,
                "defenses_total": len(defenses),
                "defenses_completed": 0,
                "targets_per_defense": len(targets),
                "images_per_payload": len(image_ids),
                "started_at_utc": _utc_now(),
            }
            atomic_json(status_path, status)
            for defense_index, defense in enumerate(defenses, start=1):
                defense_id = str(defense["id"])
                spec = {key: value for key, value in defense.items() if key != "id"}
                attack_dir = (
                    run_dir
                    / "adaptive_attacks"
                    / "worker-{}".format(slot)
                    / source
                    / method
                    / defense_id
                )
                payload_manifest = (
                    run_dir
                    / "payload_manifests"
                    / source
                    / method
                    / "{}.json".format(defense_id)
                )
                current = {
                    "kind": "adaptive_generation",
                    "source": source,
                    "method": method,
                    "defense": defense_id,
                    "defense_index": defense_index,
                    "defense_total": len(defenses),
                    "image_completed": 0,
                    "image_total": len(image_ids),
                }
                status.update(
                    current_defense=defense_id,
                    current_defense_index=defense_index,
                    current_phase="adaptive_generation",
                    current_image_completed=0,
                )
                atomic_json(status_path, status)
                events.put({"type": "task_started", "slot": slot, "current": current})

                def progress(done: int, total: int, image_id: int, state: str) -> None:
                    if done == 1 or done == total or done % 10 == 0 or state != "ok":
                        update = dict(current)
                        update.update(
                            image_completed=done,
                            image_total=total,
                            image_id=image_id,
                            image_status=state,
                        )
                        events.put(
                            {"type": "generation_progress", "slot": slot, "current": update}
                        )

                try:
                    run_attack(
                        registry,
                        "coco",
                        source,
                        method,
                        split="val",
                        output_dir=attack_dir,
                        max_images=None,
                        image_ids=image_ids,
                        seed_offsets=seed_offsets,
                        seed=42,
                        device=device,
                        download_weights=download_weights,
                        keep_going=False,
                        strict=True,
                        budget_profile="compute_matched",
                        input_transform=AdaptivePreprocessor(spec),
                        run_metadata={
                            "protocol": PROTOCOL_ID,
                            "threat_model": dict(registry.adaptive_preprocessing_policy),
                            "defense": defense_id,
                            "defense_parameters": dict(spec),
                            "retained_image_ids_sha256": _canonical_json_digest(image_ids),
                            "diagnostic_limited_images": max_images is not None,
                        },
                        progress_callback=progress,
                        isolated_candidate=isolated_candidate,
                    )
                    _payload_manifest(
                        attack_dir,
                        payload_manifest,
                        source,
                        method,
                        defense_id,
                        spec,
                        image_ids,
                    )
                except BaseException as exc:
                    reason = "adaptive generation failed: {}: {}".format(
                        type(exc).__name__, exc
                    )
                    status.update(status="failed", reason=reason, failed_at_utc=_utc_now())
                    atomic_json(status_path, status)
                    events.put(
                        {
                            "type": "fatal",
                            "slot": slot,
                            "current": current,
                            "records": [],
                            "reason": reason,
                        }
                    )
                    return
                finally:
                    _cleanup_cuda()
                annotation = _read_json(attack_dir / "annotations.json")
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
                    / source
                    / method
                    / "{}.json".format(defense_id)
                )
                try:
                    view = _materialize_view(
                        registry,
                        run_dir,
                        scratch,
                        transform_manifest,
                        annotation,
                        attack_dir,
                        source,
                        method,
                        defense_id,
                        spec,
                        protocol_id=PROTOCOL_ID,
                        threat_model=registry.adaptive_preprocessing_policy,
                        transform_implementation=registry.preprocessing_defense_policy,
                    )
                except BaseException as exc:
                    reason = "adaptive victim preprocessing failed: {}: {}".format(
                        type(exc).__name__, exc
                    )
                    status.update(status="failed", reason=reason, failed_at_utc=_utc_now())
                    atomic_json(status_path, status)
                    events.put(
                        {
                            "type": "fatal",
                            "slot": slot,
                            "current": current,
                            "records": [],
                            "reason": reason,
                        }
                    )
                    return
                evaluation_root = (
                    run_dir / "evaluations" / source / method / defense_id
                )
                for target_index, target in enumerate(targets, start=1):
                    current = {
                        "kind": "target_evaluation",
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
                            source,
                            method,
                            defense_id,
                            spec,
                            paired_lookup,
                            max_images is None,
                        )
                        if record.get("status") != "complete":
                            raise RuntimeError("Target evaluation is not complete")
                        if int(record.get("images", -1)) != len(image_ids):
                            raise RuntimeError("Target evaluation image count changed")
                        if record.get("evaluated_image_ids_match_expected") is not True:
                            raise RuntimeError("Target evaluation image IDs changed")
                    except BaseException as exc:
                        reason = "adaptive evaluation failed: {}: {}".format(
                            type(exc).__name__, exc
                        )
                        failure = {
                            "record_type": "adaptive_attacked",
                            "dataset": "coco",
                            "split": "val",
                            "source": source,
                            "attack": method,
                            "defense": defense_id,
                            "defense_parameters": dict(spec),
                            "target": target,
                            "status": "failed",
                            "reason": reason,
                            "metrics": {},
                        }
                        status.update(status="failed", reason=reason, failed_at_utc=_utc_now())
                        atomic_json(status_path, status)
                        events.put(
                            {
                                "type": "fatal",
                                "slot": slot,
                                "current": current,
                                "records": [failure],
                                "reason": reason,
                            }
                        )
                        return
                    finally:
                        _cleanup_cuda()
                    status.update(
                        current_phase="target_evaluation",
                        current_target=target,
                        current_target_index=target_index,
                    )
                    atomic_json(status_path, status)
                    events.put(
                        {
                            "type": "record_complete",
                            "slot": slot,
                            "current": current,
                            "record": record,
                        }
                    )
                _remove_verified_scratch(view, run_dir, transform_manifest)
                _prune_payload(attack_dir, run_dir, payload_manifest)
                status.update(
                    defenses_completed=defense_index,
                    current_phase="payload_removed_after_complete_target_panel",
                )
                atomic_json(status_path, status)
                events.put(
                    {
                        "type": "payload_complete",
                        "slot": slot,
                        "current": current,
                    }
                )
            status.update(status="complete", completed_at_utc=_utc_now())
            for key in (
                "current_defense",
                "current_defense_index",
                "current_phase",
                "current_image_completed",
                "current_target",
                "current_target_index",
            ):
                status.pop(key, None)
            atomic_json(status_path, status)
            events.put(
                {
                    "type": "group_complete",
                    "slot": slot,
                    "current": {"source": source, "method": method},
                }
            )
        events.put({"type": "worker_done", "slot": slot})
    except BaseException as exc:
        events.put(
            {
                "type": "fatal",
                "slot": slot,
                "current": None,
                "records": [],
                "reason": "adaptive worker failed: {}: {}".format(
                    type(exc).__name__, exc
                ),
            }
        )


def _run_workers(
    registry: Registry,
    run_dir: Path,
    devices: Sequence[str],
    tasks: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
    defenses: Sequence[Mapping[str, Any]],
    image_ids: Sequence[int],
    seed_offsets: Mapping[int, int],
    paired_records: Sequence[Mapping[str, Any]],
    max_images: Optional[int],
    download_weights: bool,
    records: List[Dict[str, Any]],
    plan: Mapping[str, Any],
    state: MutableMapping[str, Any],
    isolated_candidate: Optional[Mapping[str, Any]] = None,
) -> None:
    slot_indices = round_robin_indices(len(tasks), len(devices))
    assignments = [[dict(tasks[index]) for index in indices] for indices in slot_indices]
    context = mp.get_context("spawn")
    events = context.Queue()
    processes = []
    for slot, device in enumerate(devices):
        process = context.Process(
            target=_worker_main,
            args=(
                str(registry.root),
                str(run_dir),
                str(device),
                slot,
                assignments[slot],
                list(targets),
                [dict(item) for item in defenses],
                list(image_ids),
                dict(seed_offsets),
                [dict(item) for item in paired_records],
                max_images,
                bool(download_weights),
                dict(isolated_candidate) if isolated_candidate else None,
                events,
            ),
            name="adaptive-defense-worker-{}".format(slot),
        )
        process.start()
        processes.append(process)
    workers = [
        {
            "slot": slot,
            "device": str(device),
            "status": "running",
            "current": None,
        }
        for slot, device in enumerate(devices)
    ]
    state["workers"] = workers
    state["phase"] = "adaptive_generation_and_evaluation"
    state_path = run_dir / "execution_state.json"
    records_path = run_dir / "records.json"

    def publish() -> None:
        current = [
            item["current"]
            for item in workers
            if item["status"] == "running" and item.get("current")
        ]
        state["current"] = current[0] if current else None
        state["updated_at_utc"] = _utc_now()
        atomic_json(state_path, state)

    publish()
    complete_workers = 0
    failure = ""
    normal = False
    try:
        while complete_workers < len(processes):
            try:
                event = events.get(timeout=2.0)
            except queue.Empty:
                unexpected = [
                    (slot, process.exitcode)
                    for slot, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                if unexpected:
                    failure = "Adaptive worker exited unexpectedly: {}".format(unexpected)
                    break
                continue
            slot = int(event["slot"])
            kind = str(event["type"])
            if event.get("current") is not None:
                workers[slot]["current"] = dict(event["current"])
            if kind in {"task_started", "generation_progress"}:
                publish()
                continue
            if kind == "record_complete":
                records.append(dict(event["record"]))
                records[:] = _canonical_records(records, plan["jobs"])
                atomic_json(records_path, records)
                state["attack_evaluations_completed"] = int(
                    state["attack_evaluations_completed"]
                ) + 1
                publish()
                continue
            if kind == "payload_complete":
                state["attack_payload_groups_completed"] = int(
                    state["attack_payload_groups_completed"]
                ) + 1
                publish()
                continue
            if kind == "group_complete":
                state["attack_groups_completed"] = int(
                    state["attack_groups_completed"]
                ) + 1
                publish()
                continue
            if kind == "worker_done":
                if workers[slot]["status"] != "complete":
                    workers[slot].update(status="complete", current=None)
                    complete_workers += 1
                publish()
                continue
            if kind == "fatal":
                for record in event.get("records", []):
                    records.append(dict(record))
                records[:] = _canonical_records(records, plan["jobs"])
                atomic_json(records_path, records)
                state["failed_records"] = sum(
                    1 for record in records if record.get("status") == "failed"
                )
                workers[slot]["status"] = "failed"
                failure = str(event.get("reason") or "adaptive worker failed")
                publish()
                break
            failure = "Unknown adaptive worker event: {}".format(kind)
            break
        if failure:
            raise RuntimeError(failure)
        normal = True
    finally:
        if normal:
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


def _finite_mean(values: Iterable[float]) -> Optional[float]:
    chosen = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return mean(chosen) if chosen else None


def adaptive_summary_rows(
    records: Sequence[Mapping[str, Any]],
    paired_records: Sequence[Mapping[str, Any]],
    defenses: Sequence[str],
    metric: str = "bbox_mAP_50",
) -> List[Dict[str, Any]]:
    clean_lookup = {
        (str(record["defense"]), str(record["target"])): record
        for record in paired_records
        if record.get("record_type") == "clean" and record.get("status") == "complete"
    }
    clean_identity = {
        target: record for (defense, target), record in clean_lookup.items() if defense == "identity"
    }
    rows = []
    for defense in defenses:
        selected = [
            record
            for record in records
            if record.get("status") == "complete"
            and record.get("defense") == defense
            and record.get("source") != record.get("target")
        ]
        clean = [record for (item, _), record in clean_lookup.items() if item == defense]
        rows.append(
            {
                "defense": defense,
                "clean_ap": _finite_mean(record["metrics"][metric] for record in clean),
                "clean_utility_drop": _finite_mean(
                    float(clean_identity[str(record["target"])]["metrics"][metric])
                    - float(record["metrics"][metric])
                    for record in clean
                ),
                "adaptive_bb_ap": _finite_mean(
                    record["metrics"][metric] for record in selected
                ),
                "oblivious_bb_ap": _finite_mean(
                    record["paired_oblivious_metrics"][metric] for record in selected
                ),
                "adaptive_minus_oblivious": _finite_mean(
                    record["adaptive_minus_oblivious"][metric] for record in selected
                ),
                "residual_attack_effect": _finite_mean(
                    float(clean_lookup[(defense, str(record["target"]))]["metrics"][metric])
                    - float(record["metrics"][metric])
                    for record in selected
                    if (defense, str(record["target"])) in clean_lookup
                ),
                "clean_cells": len(clean),
                "blackbox_cells": len(selected),
            }
        )
    return rows


def adaptive_method_rows(
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
                if record.get("status") == "complete"
                and record.get("attack") == method
                and record.get("defense") == defense
                and record.get("source") != record.get("target")
            ]
            rows.append(
                {
                    "method": method,
                    "defense": defense,
                    "adaptive_bb_ap": _finite_mean(
                        record["metrics"][metric] for record in selected
                    ),
                    "oblivious_bb_ap": _finite_mean(
                        record["paired_oblivious_metrics"][metric]
                        for record in selected
                    ),
                    "adaptive_minus_oblivious": _finite_mean(
                        record["adaptive_minus_oblivious"][metric]
                        for record in selected
                    ),
                    "blackbox_cells": len(selected),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        _atomic_text(path, "")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric(value: Optional[float]) -> str:
    return "--" if value is None else "{:.4f}".format(float(value))


def write_adaptive_reports(
    records: Sequence[Mapping[str, Any]],
    paired_records: Sequence[Mapping[str, Any]],
    registry: Registry,
    report_dir: Path,
    methods: Sequence[str],
    defenses: Sequence[str],
) -> Dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = adaptive_summary_rows(records, paired_records, defenses)
    matrix = adaptive_method_rows(records, methods, defenses)
    summary_csv = report_dir / "adaptive_defense_summary_bbox_mAP_50.csv"
    matrix_csv = report_dir / "adaptive_method_by_defense_bbox_mAP_50.csv"
    records_csv = report_dir / "adaptive_records.csv"
    _write_csv(summary_csv, summary)
    _write_csv(matrix_csv, matrix)
    flat = []
    for record in records:
        row = {
            "source": record.get("source"),
            "method": record.get("attack"),
            "defense": record.get("defense"),
            "target": record.get("target"),
            "status": record.get("status"),
            "images": record.get("images"),
        }
        for metric in METRICS:
            row[metric] = record.get("metrics", {}).get(metric)
            row["oblivious_{}".format(metric)] = record.get(
                "paired_oblivious_metrics", {}
            ).get(metric)
            row["adaptive_minus_oblivious_{}".format(metric)] = record.get(
                "adaptive_minus_oblivious", {}
            ).get(metric)
        flat.append(row)
    _write_csv(records_csv, flat)
    labels = {
        defense: str(registry.preprocessing_defenses[defense]["label"])
        for defense in defenses
    }
    summary_tex = report_dir / "adaptive_defense_summary_bbox_mAP_50.tex"
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{COCO retained-500 Common-2 black-box AP@.50 under paired oblivious and source-pipeline-adaptive input preprocessing. Source-matched targets are excluded; values come from unrounded JSON and are not full-val2017 or certified-robustness results.}",
        r"\label{tab:coco_retained500_adaptive_defense_summary}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Defense & Clean AP@.50 & Clean drop & Adaptive BB & Oblivious BB & $\Delta$ adaptive & Residual \\",
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {} & {} & {} & {} & {} \\\\".format(
                labels[str(row["defense"])],
                _metric(row["clean_ap"]),
                _metric(row["clean_utility_drop"]),
                _metric(row["adaptive_bb_ap"]),
                _metric(row["oblivious_bb_ap"]),
                _metric(row["adaptive_minus_oblivious"]),
                _metric(row["residual_attack_effect"]),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    _atomic_text(summary_tex, "\n".join(lines))
    matrix_tex = report_dir / "adaptive_method_by_defense_bb_mean_bbox_mAP_50.tex"
    lookup = {
        (str(row["method"]), str(row["defense"])): row["adaptive_bb_ap"]
        for row in matrix
    }
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{COCO retained-500 Common-2 adaptive black-box AP@.50 by method and known input preprocessing. Source-matched targets are excluded; lower values indicate stronger defense-aware transfer.}",
        r"\label{tab:coco_retained500_adaptive_method_defense}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\resizebox{\textwidth}{!}{%",
        "\\begin{{tabular}}{{l{}}}".format("r" * len(defenses)),
        r"\toprule",
        "Method & {} \\\\".format(" & ".join(labels[item] for item in defenses)),
        r"\midrule",
    ]
    for method in methods:
        lines.append(
            "{} & {} \\\\".format(
                registry.attack(method).display_name,
                " & ".join(_metric(lookup.get((method, defense))) for defense in defenses),
            )
        )
    lines.extend(
        [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
    )
    _atomic_text(matrix_tex, "\n".join(lines))
    return {
        "summary_csv": summary_csv,
        "matrix_csv": matrix_csv,
        "records_csv": records_csv,
        "summary_tex": summary_tex,
        "matrix_tex": matrix_tex,
    }


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

    tracked = command("status", "--short")
    files = [
        root / "configs" / "experiments" / "protocols.yaml",
        root / "configs" / "experiments" / "preprocessing_defenses.yaml",
        root / "experiments" / "coco_adaptive_preprocessing_defense.py",
        root / "src" / "lgp" / "defenses" / "preprocessing.py",
        root / "src" / "lgp" / "runners" / "attack.py",
        root / "src" / "lgp" / "runners" / "adaptive_preprocessing_defense.py",
    ]
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "git_head": command("rev-parse", "HEAD") or None,
        "worktree_dirty": bool(tracked),
        "worktree_status": tracked.splitlines(),
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


def _assignments(
    run_dir: Path,
    devices: Sequence[str],
    sources: Sequence[str],
    methods: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    tasks = [
        {"source": source, "method": method}
        for source in sources
        for method in methods
    ]
    slots = round_robin_indices(len(tasks), len(devices))
    payload = {
        "schema_version": 1,
        "algorithm": "canonical_round_robin_source_method_groups_v1",
        "worker_count": len(devices),
        "devices": list(devices),
        "global_record_writer": "coordinator_only",
        "tasks": [
            {**tasks[index], "worker_slot": slot, "device": devices[slot]}
            for slot, indices in enumerate(slots)
            for index in indices
        ],
    }
    atomic_json(run_dir / "execution_assignments.json", payload)
    return payload, tasks


def run_adaptive_preprocessing_defense(
    registry: Registry,
    execute: bool = True,
    output_dir: Optional[Path] = None,
    paired_acceptance: Optional[Path] = None,
    source_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    defense_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
    device: str = "cuda:0",
    devices: Optional[Sequence[str]] = None,
    download_weights: bool = False,
) -> Path:
    plan = build_adaptive_defense_plan(
        registry,
        source_override,
        method_override,
        target_override,
        defense_override,
        max_images,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_dir
        or project_root()
        / "outputs"
        / "experiments"
        / "coco_adaptive_preprocessing_defense"
        / stamp
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty experiment: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "plan.json", plan)
    _write_plan_csv(run_dir / "plan.csv", plan["jobs"])
    atomic_json(run_dir / "records.json", [])
    selected = _selection(
        registry,
        source_override,
        method_override,
        target_override,
        defense_override,
    )
    sources = selected["sources"]
    methods = selected["methods"]
    targets = selected["targets"]
    defense_records = [
        {"id": defense_id, **dict(spec)}
        for defense_id, spec in selected["defenses"]
    ]
    defense_ids = [str(item["id"]) for item in defense_records]
    if not execute:
        write_adaptive_reports(
            [], [], registry, run_dir / "reports" / "coco", methods, defense_ids
        )
        atomic_json(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "planned",
                "protocol": PROTOCOL_ID,
                "jobs": plan["counts"]["records"],
                "records": 0,
                "max_images": max_images,
                "adaptive_retained500_benchmark_eligible": False,
                "full_coco_val_claim": False,
                "certified_robustness_claim": False,
            },
        )
        return run_dir

    execution_devices = normalize_execution_devices(device, devices)
    validate_available_cuda_devices(execution_devices)
    assignments, tasks = _assignments(run_dir, execution_devices, sources, methods)
    counts = plan["counts"]
    state: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "validating_paired_oblivious_evidence",
        "protocol": PROTOCOL_ID,
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "attack_groups_completed": 0,
        "attack_groups_total": counts["attack_groups"],
        "attack_payload_groups_completed": 0,
        "attack_payload_groups_total": counts["attack_payload_groups"],
        "attack_evaluations_completed": 0,
        "attack_evaluations_total": counts["attack_evaluations"],
        "failed_records": 0,
        "workers": [],
        "current": None,
        "execution_scheduler": assignments,
    }
    atomic_json(run_dir / "execution_state.json", state)
    records: List[Dict[str, Any]] = []
    try:
        paired = verify_paired_oblivious_evidence(
            registry, paired_acceptance, max_images=max_images
        )
        atomic_json(run_dir / "paired_oblivious_evidence.json", paired)
        paired_run = registry.root / str(paired["paired_run"])
        paired_records_all = list(_read_json(paired_run / "records.json"))
        paired_records = [
            dict(record)
            for record in paired_records_all
            if (
                record.get("record_type") == "attacked"
                and record.get("source") in sources
                and record.get("attack") in methods
                and record.get("defense") in defense_ids
                and record.get("target") in targets
            )
        ]
        paired_clean = [
            dict(record)
            for record in paired_records_all
            if (
                record.get("record_type") == "clean"
                and record.get("defense") in defense_ids
                and record.get("target") in targets
            )
        ]
        if len(paired_records) != counts["records"]:
            raise RuntimeError("Paired oblivious attacked-record selection is incomplete")
        if len(paired_clean) != len(defense_ids) * len(targets):
            raise RuntimeError("Paired clean-reference selection is incomplete")
        selected_count = (
            RETAINED_IMAGES if max_images is None else min(max_images, RETAINED_IMAGES)
        )
        image_ids = list(paired["retained_image_ids"][:selected_count])
        seed_offsets = {
            int(image_id): index
            for index, image_id in enumerate(sorted(paired["retained_image_ids"]))
        }
        _run_workers(
            registry,
            run_dir,
            execution_devices,
            tasks,
            targets,
            defense_records,
            image_ids,
            seed_offsets,
            paired_records,
            max_images,
            download_weights,
            records,
            plan,
            state,
        )
        records = _canonical_records(records, plan["jobs"])
        atomic_json(run_dir / "records.json", records)
        failures = [record for record in records if record.get("status") != "complete"]
        if len(records) != counts["records"] or failures:
            raise RuntimeError(
                "Adaptive record gate failed: records={}, expected={}, failures={}".format(
                    len(records), counts["records"], len(failures)
                )
            )
        if (run_dir / "scratch").exists():
            remaining = [
                path for path in (run_dir / "scratch").rglob("*") if path.is_file()
            ]
            if remaining:
                raise RuntimeError("Verified adaptive transform scratch remains")
            shutil.rmtree(str(_inside(run_dir / "scratch", run_dir)))
        remaining_payloads = [
            path
            for path in (run_dir / "adaptive_attacks").rglob("images")
            if path.is_dir()
        ]
        if remaining_payloads:
            raise RuntimeError("Generated adaptive payload directories remain after evaluation")
        payload_evidence = audit_adaptive_payload_evidence(
            run_dir, sources, methods, defense_ids, image_ids
        )
        atomic_json(run_dir / "adaptive_payload_evidence.json", payload_evidence)
        state.update(
            phase="writing_reports_and_acceptance",
            workers=[],
            current=None,
            failed_records=0,
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", state)
        reports = write_adaptive_reports(
            records,
            paired_clean if max_images is None else [],
            registry,
            run_dir / "reports" / "coco",
            methods,
            defense_ids,
        )
        provenance = _git_provenance(registry.root)
        atomic_json(run_dir / "provenance.json", provenance)
        formal_selection_requested = max_images is None and bool(
            plan["full_registered_selection"]
        )
        if formal_selection_requested and provenance["worktree_dirty"]:
            raise RuntimeError(
                "Formal adaptive preprocessing requires a clean Git worktree; "
                "commit the reviewed code and documentation before execution"
            )
        exact_formal = (
            max_images is None
            and bool(plan["full_registered_selection"])
            and len(records) == FORMAL_RECORDS
            and int(state["attack_evaluations_completed"]) == FORMAL_RECORDS
            and int(state["attack_payload_groups_completed"]) == FORMAL_PAYLOAD_GROUPS
            and int(state["attack_groups_completed"]) == FORMAL_GROUPS
            and int(paired["selected_images"]) == RETAINED_IMAGES
            and int(paired["source_payload_groups"]) == FORMAL_GROUPS
            and int(paired["source_payload_hashes_verified_per_group"])
            == RETAINED_IMAGES
            and int(payload_evidence["groups"]) == FORMAL_PAYLOAD_GROUPS
            and int(payload_evidence["images_per_group"]) == RETAINED_IMAGES
            and int(payload_evidence["payload_hashes_preserved"])
            == FORMAL_PAYLOAD_GROUPS * RETAINED_IMAGES
            and int(payload_evidence["linf_violations"]) == 0
            and int(payload_evidence["remaining_image_payload_directories"]) == 0
            and provenance["worktree_dirty"] is False
            and provenance["git_head"] is not None
        )
        acceptance = {
            "schema_version": 1,
            "status": "accepted" if exact_formal else "diagnostic_complete",
            "protocol": PROTOCOL_ID,
            "accepted_at_utc": _utc_now(),
            "adaptive_retained500_benchmark_eligible": bool(exact_formal),
            "full_coco_val_claim_eligible": False,
            "certified_robustness_claim_eligible": False,
            "scope": "COCO retained-500 Common-2 source-pipeline-adaptive preprocessing only",
            "common_2_is_paper_term": False,
            "threat_model": dict(registry.adaptive_preprocessing_policy),
            "counts": {
                "records": len(records),
                "attack_evaluations": state["attack_evaluations_completed"],
                "attack_payload_groups": state["attack_payload_groups_completed"],
                "attack_groups": state["attack_groups_completed"],
                "failed_records": 0,
                "retained_images": selected_count,
                "paired_clean_references": len(paired_clean),
                "sources": len(sources),
                "methods": len(methods),
                "targets": len(targets),
                "defenses": len(defense_ids),
                "workers": len(execution_devices),
            },
            "paired_oblivious_acceptance": paired["acceptance"],
            "retained_image_ids_sha256": paired["retained_image_ids_sha256"],
            "evidence_sha256": {
                "plan.json": file_digest(run_dir / "plan.json"),
                "records.json": file_digest(run_dir / "records.json"),
                "paired_oblivious_evidence.json": file_digest(
                    run_dir / "paired_oblivious_evidence.json"
                ),
                "adaptive_payload_evidence.json": file_digest(
                    run_dir / "adaptive_payload_evidence.json"
                ),
                "execution_assignments.json": file_digest(
                    run_dir / "execution_assignments.json"
                ),
                "provenance.json": file_digest(run_dir / "provenance.json"),
                **{
                    _relative(path, run_dir): file_digest(path)
                    for path in reports.values()
                },
            },
            "limitations": [
                "The attacker adapts to the known preprocessing-to-source pipeline but has no target-model gradients or target queries.",
                "All preprocessing variants are deterministic, so EOT is fixed to one and no stochastic-defense claim is made.",
                "Metrics cover the pre-frozen retained 500-image subset and are not full-val2017 COCO AP.",
                "This is empirical robustness under the declared BPDA rules, not certified robustness.",
            ],
        }
        atomic_json(
            run_dir / "formal_adaptive_preprocessing_defense_acceptance.json",
            acceptance,
        )
        summary = {
            "schema_version": 1,
            "status": "complete",
            "protocol": PROTOCOL_ID,
            "started_at_utc": state["started_at_utc"],
            "completed_at_utc": _utc_now(),
            "records": len(records),
            "failed_records": 0,
            "max_images": max_images,
            "adaptive_retained500_benchmark_eligible": bool(exact_formal),
            "full_coco_val_claim": False,
            "certified_robustness_claim": False,
            "counts": acceptance["counts"],
        }
        atomic_json(run_dir / "summary.json", summary)
        state.update(
            status="complete",
            phase="complete",
            workers=[],
            current=None,
            failed_records=0,
            completed_at_utc=summary["completed_at_utc"],
            updated_at_utc=_utc_now(),
        )
        atomic_json(run_dir / "execution_state.json", state)
        write_artifact_manifest(run_dir)
        return run_dir
    except BaseException as exc:
        reason = "{}: {}".format(type(exc).__name__, exc)
        state.update(
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
        atomic_json(run_dir / "execution_state.json", state)
        atomic_json(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "status": "failed",
                "protocol": PROTOCOL_ID,
                "reason": reason,
                "records": len(records),
                "failed_records": state["failed_records"],
                "adaptive_retained500_benchmark_eligible": False,
                "full_coco_val_claim": False,
                "certified_robustness_claim": False,
                "failed_at_utc": state["failed_at_utc"],
            },
        )
        raise
