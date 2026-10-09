from __future__ import annotations

import gc
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch

from ..io import atomic_json, file_digest
from ..registry import Registry
from .attack import run_attack
from .evaluate import run_evaluation
from .retention import RETENTION_MODE, validate_and_prune_attack_group


ASSIGNMENT_ALGORITHM = "canonical_round_robin_v1"
_CUDA_DEVICE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")


def normalize_execution_devices(
    device: str,
    devices: Optional[Sequence[str]] = None,
) -> List[str]:
    """Resolve one serial device or an explicit independent-worker list."""
    selected = [device] if devices is None else devices
    values = [str(value).strip() for value in selected]
    if not values or any(not value for value in values):
        raise ValueError("At least one non-empty execution device is required")
    if len(set(values)) != len(values):
        raise ValueError("Execution devices must be unique")
    if len(values) > 1 and any(not _CUDA_DEVICE.fullmatch(value) for value in values):
        raise ValueError(
            "Multi-device execution requires explicit CUDA devices such as "
            "cuda:0,cuda:1"
        )
    return values


def validate_available_cuda_devices(
    devices: Sequence[str], *, include_single: bool = False
) -> None:
    """Fail before creating a formal run if a requested logical GPU is absent."""
    if len(devices) <= 1 and not include_single:
        return
    if list(devices) == ["cpu"]:
        return
    if any(value != "cuda" and not _CUDA_DEVICE.fullmatch(value) for value in devices):
        raise ValueError("Execution devices must be cpu, cuda, or explicit CUDA ordinals")
    if not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA execution requires CUDA availability")
    available = int(torch.cuda.device_count())
    requested = [
        int(torch.cuda.current_device()) if device == "cuda"
        else int(str(device).split(":", 1)[1]) for device in devices
    ]
    missing = [index for index in requested if index >= available]
    if missing:
        raise RuntimeError(
            "Requested CUDA device ordinal(s) are unavailable: {} (count={})".format(
                ", ".join(str(index) for index in missing), available
            )
        )


def round_robin_indices(item_count: int, worker_count: int) -> List[List[int]]:
    if item_count < 0:
        raise ValueError("item_count must be non-negative")
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    assignments: List[List[int]] = [[] for _ in range(worker_count)]
    for index in range(item_count):
        assignments[index % worker_count].append(index)
    return assignments


def build_execution_assignment(
    clean_jobs: Sequence[Tuple[str, str, str]],
    group_items: Sequence[Tuple[tuple, Sequence[Mapping[str, Any]]]],
    devices: Sequence[str],
) -> Dict[str, Any]:
    clean_slots = round_robin_indices(len(clean_jobs), len(devices))
    group_slots = round_robin_indices(len(group_items), len(devices))
    clean = []
    groups = []
    for slot, indices in enumerate(clean_slots):
        for index in indices:
            dataset, split, target = clean_jobs[index]
            clean.append(
                {
                    "canonical_index": index + 1,
                    "worker_slot": slot,
                    "device": devices[slot],
                    "dataset": dataset,
                    "split": split,
                    "target": target,
                }
            )
    for slot, indices in enumerate(group_slots):
        for index in indices:
            key, jobs = group_items[index]
            dataset, split, source, attack, study, variant = key
            groups.append(
                {
                    "canonical_index": index + 1,
                    "worker_slot": slot,
                    "device": devices[slot],
                    "dataset": dataset,
                    "split": split,
                    "source": source,
                    "attack": attack,
                    "study": study,
                    "variant": variant,
                    "targets": [str(job["target"]) for job in jobs],
                }
            )
    clean.sort(key=lambda item: int(item["canonical_index"]))
    groups.sort(key=lambda item: int(item["canonical_index"]))
    return {
        "schema_version": 1,
        "algorithm": ASSIGNMENT_ALGORITHM,
        "worker_count": len(devices),
        "devices": list(devices),
        "clean_barrier_before_attack_groups": True,
        "group_atomicity": (
            "one worker generates the selected images and evaluates every declared "
            "target for one complete source-method-variant group; the registered "
            "retention policy is unchanged"
        ),
        "global_record_writer": "coordinator_only",
        "clean": clean,
        "attack_groups": groups,
    }


def formal_expected_record_keys(
    plan: Mapping[str, Any],
    target_ids: Sequence[str],
    *,
    variant_aware: bool = False,
) -> List[tuple]:
    if variant_aware:
        return [
            ("clean", "clean", "", "clean", str(target_id))
            for target_id in target_ids
        ] + [
            (
                str(job["source"]),
                str(job["attack"]),
                str(job.get("study", "")),
                str(job.get("variant", "default")),
                str(job["target"]),
            )
            for job in plan.get("jobs", [])
        ]
    return [
        ("clean", "clean", str(target_id)) for target_id in target_ids
    ] + [
        (str(job["source"]), str(job["attack"]), str(job["target"]))
        for job in plan.get("jobs", [])
    ]


def canonical_partial_records(
    records: Sequence[Mapping[str, Any]], expected_keys: Sequence[tuple]
) -> List[Dict[str, Any]]:
    """Keep live records deterministic even when workers finish out of order."""
    key_lengths = {len(key) for key in expected_keys}
    if len(key_lengths) > 1 or (key_lengths and next(iter(key_lengths)) not in {3, 5, 7}):
        raise RuntimeError("Expected formal record keys must use one supported schema")
    width = next(iter(key_lengths)) if key_lengths else 3

    def record_key(record: Mapping[str, Any]) -> tuple:
        base = (
            str(record.get("source", "")),
            str(record.get("attack", "")),
        )
        if width in {5, 7}:
            prefix = (
                (str(record.get("dataset", "")), str(record.get("split", "")))
                if width == 7 else ()
            )
            return prefix + base + (
                str(record.get("study", "")),
                str(record.get("variant", "clean" if base == ("clean", "clean") else "default")),
                str(record.get("target", "")),
            )
        return base + (str(record.get("target", "")),)

    positions = {key: index for index, key in enumerate(expected_keys)}
    if len(positions) != len(expected_keys):
        raise RuntimeError("Expected formal record keys are not unique")
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key not in positions:
            raise RuntimeError("Unexpected formal record key: {}".format(key))
        if key in by_key:
            raise RuntimeError("Duplicate formal record key: {}".format(key))
        by_key[key] = dict(record)
    return sorted(by_key.values(), key=lambda record: positions[record_key(record)])


def experiment_expected_record_keys(
    plan: Mapping[str, Any], clean_jobs: Sequence[Tuple[str, str, str]]
) -> List[tuple]:
    keys = [
        (dataset, split, "clean", "clean", "", "clean", target)
        for dataset, split, target in clean_jobs
    ] + [
        tuple(str(job.get(field, default)) for field, default in (
            ("dataset", ""), ("split", ""), ("source", ""), ("attack", ""),
            ("study", ""), ("variant", "default"), ("target", ""),
        ))
        for job in plan.get("jobs", [])
    ]
    canonical_partial_records([], keys)
    return keys


class _WorkerEvents:
    """Attach process and assignment identity to every worker message."""

    def __init__(self, queue_: Any, stage: str, slot: int) -> None:
        self.queue = queue_
        self.stage = stage
        self.slot = slot
        self.task_index: Optional[int] = None
        self.journal = None

    def put(self, event: Mapping[str, Any]) -> None:
        registration = {}
        if self.journal is not None:
            try:
                registration = self.journal.record(event, self.task_index)
            except BaseException as exc:
                if event.get("type") != "fatal":
                    raise
                registration = dict(worker_start_ticks=self.journal.worker["start_ticks"],
                                    journal_error=repr(exc))
        self.queue.put({
            **event, **registration, "pid": os.getpid(), "stage": self.stage,
            "slot": self.slot, "task_index": self.task_index,
        })


def _cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _clean_task(
    registry: Registry,
    run_dir: Path,
    device: str,
    slot: int,
    task: Mapping[str, Any],
    settings: Mapping[str, Any],
    events: Any,
) -> bool:
    dataset = str(task["dataset"])
    split = str(task["split"])
    target = str(task["target"])
    current = {
        "kind": "clean",
        "canonical_index": int(task["canonical_index"]),
        "dataset": dataset,
        "target": target,
    }
    events.put({"type": "task_started", "slot": slot, "current": current})
    layout = settings.get("clean_output_layout", "dataset_grouped")
    if layout not in {"dataset_grouped", "flat_targets"}:
        raise ValueError("Unknown clean output layout: {}".format(layout))
    clean_dir = (
        run_dir / "evaluations" / target if layout == "flat_targets"
        else run_dir / "evaluations" / dataset / "clean" / target
    )
    try:
        completed = run_evaluation(
            registry,
            dataset,
            target,
            split=split,
            output_dir=clean_dir,
            max_images=settings["max_images"],
            device=device,
            download_weights=bool(settings["download_weights"]),
            keep_going=bool(settings.get("keep_going", False)),
            save_visualizations=bool(settings["save_visualizations"]),
            visualization_score_threshold=float(
                settings["visualization_score_threshold"]
            ),
            visualization_max_images=int(settings["visualization_max_images"]),
            visualization_max_detections=int(
                settings["visualization_max_detections"]
            ),
            prediction_archive=settings["prediction_archive"],
            image_ids=settings.get("image_ids_by_dataset", {}).get(dataset),
        )
        record = _read_metrics(completed / "metrics.json")
        if layout == "flat_targets":
            record["predictions_sha256"] = file_digest(completed / "predictions.json")
    except Exception as exc:
        record = {
            "dataset": dataset,
            "split": split,
            "source": "clean",
            "attack": "clean",
            "target": target,
            "status": "failed",
            "reason": "clean evaluation failed: {}: {}".format(
                type(exc).__name__, exc
            ),
            "metrics": {},
        }
        if not settings.get("keep_going", False):
            events.put(
                {
                    "type": "fatal", "slot": slot, "current": current,
                    "records": [record], "reason": record["reason"],
                }
            )
            return False
    finally:
        _cleanup_cuda()
    events.put(
        {
            "type": "clean_complete",
            "slot": slot,
            "current": current,
            "record": record,
        }
    )
    return True


def _read_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("Metrics payload is not an object: {}".format(path))
    return payload


def validate_generated_group(
    generated: Path, dataset: str, split: str, source: str, attack: str,
    image_ids: Optional[Sequence[int]], max_images: Optional[int],
) -> None:
    """Reject partial generation before an evaluator can silently shrink its input."""
    run = _read_metrics(generated / "run.json")
    if run.get("status") != "complete" or int(run.get("failed_images", -1)) != 0:
        raise RuntimeError("Attack generation is incomplete or contains image failures")
    for field, expected in (("dataset", dataset), ("split", split), ("source", source), ("attack", attack)):
        if run.get(field) != expected:
            raise RuntimeError("Attack generation identity mismatch: " + field)
    requested = int(run.get("requested_images", -1))
    if requested <= 0 or int(run.get("successful_images", -1)) != requested:
        raise RuntimeError("Attack generation success/requested counts do not match")
    annotation_path = generated / "annotations.json"
    if run.get("annotation_sha256") != file_digest(annotation_path):
        raise RuntimeError("Attack generation annotation hash mismatch")
    annotation = _read_metrics(annotation_path)
    actual_ids = sorted(int(item["id"]) for item in annotation.get("images", []))
    if len(actual_ids) != requested or len(set(actual_ids)) != requested:
        raise RuntimeError("Attack generation annotation image coverage is incomplete")
    expected_ids = list(image_ids) if image_ids is not None else actual_ids
    if max_images is not None:
        expected_ids = expected_ids[:max_images]
    if sorted(expected_ids) != actual_ids:
        raise RuntimeError("Attack generation does not match the requested image IDs")
    digest = lambda values: hashlib.sha256(json.dumps(values, separators=(",", ":")).encode("utf-8")).hexdigest()
    if run.get("image_ids_sha256") != digest(actual_ids) or run.get("requested_ordered_image_ids_sha256") != digest(expected_ids):
        raise RuntimeError("Attack generation image-ID hashes do not match its request")


def _group_task(
    registry: Registry,
    run_dir: Path,
    device: str,
    slot: int,
    task: Mapping[str, Any],
    settings: Mapping[str, Any],
    retained_selection: Mapping[str, Any],
    events: Any,
) -> bool:
    dataset = str(task["dataset"])
    split = str(task["split"])
    source = str(task["source"])
    attack = str(task["attack"])
    study = str(task["study"])
    variant = str(task["variant"])
    group_jobs = [dict(job) for job in task["jobs"]]
    group_index = int(task["canonical_index"])
    suffix = Path(study) / variant if study else Path(variant)
    attack_dir = run_dir / "attacks" / dataset / source / attack / suffix
    group_status_path = (
        run_dir
        / "group_status"
        / dataset
        / source
        / attack
        / suffix.with_suffix(".json")
    )
    group_status: Dict[str, Any] = {
        "schema_version": 1,
        "status": "generating",
        "index": group_index,
        "total": int(task["group_total"]),
        "worker_slot": slot,
        "dataset": dataset,
        "split": split,
        "source": source,
        "attack": attack,
        "study": study,
        "variant": variant,
        "targets_total": len(group_jobs),
        "targets_completed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(group_status_path, group_status)
    current = {
        "kind": "attack_generation",
        "group_index": group_index,
        "dataset": dataset,
        "source": source,
        "attack": attack,
        "variant": variant,
    }
    events.put({"type": "task_started", "slot": slot, "current": current})
    job_seed = group_jobs[0].get("seed")
    progress_interval = max(int(settings.get("progress_interval_images", 25)), 1)

    def publish_generation_progress(
        completed: int, total: int, image_id: int, image_status: str
    ) -> None:
        if completed not in {1, total} and completed % progress_interval != 0:
            return
        group_status.update(
            generation_images_completed=int(completed),
            generation_images_total=int(total),
            last_image_id=int(image_id),
            last_image_status=str(image_status),
        )
        atomic_json(group_status_path, group_status)
        progress = {
            **current,
            "image_progress": int(completed),
            "image_total": int(total),
            "last_image_id": int(image_id),
            "last_image_status": str(image_status),
        }
        events.put({"type": "task_progress", "slot": slot, "current": progress})

    try:
        generated = run_attack(
            registry,
            dataset,
            source,
            attack,
            split=split,
            output_dir=attack_dir,
            max_images=settings["max_images"],
            seed=int(settings["seed"] if job_seed is None else job_seed),
            device=device,
            download_weights=bool(settings["download_weights"]),
            keep_going=bool(settings.get("keep_going", False)),
            strict=True,
            parameter_overrides=group_jobs[0].get("parameters", {}),
            budget_profile=group_jobs[0]["budget_profile"],
            progress_callback=publish_generation_progress,
            run_metadata=group_jobs[0].get("run_metadata", {}),
            isolated_candidate=settings.get("isolated_candidate"),
            image_ids=settings.get("image_ids_by_dataset", {}).get(dataset),
        )
        validate_generated_group(
            generated, dataset, split, source, attack,
            settings.get("image_ids_by_dataset", {}).get(dataset), settings["max_images"],
        )
    except Exception as exc:
        reason = "attack generation failed: {}: {}".format(
            type(exc).__name__, exc
        )
        group_status.update(
            status="failed_generation",
            reason=reason,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(group_status_path, group_status)
        failure_records = [
            {**job, "status": "failed", "reason": reason, "metrics": {}}
            for job in group_jobs
        ]
        events.put(
            {
                "type": "group_failed" if settings.get("keep_going", False) else "fatal",
                "slot": slot,
                "current": current,
                "records": failure_records,
                "reason": reason,
            }
        )
        return bool(settings.get("keep_going", False))
    finally:
        _cleanup_cuda()

    evaluation_dirs: Dict[str, Path] = {}
    group_failed = False
    for target_index, job in enumerate(group_jobs, start=1):
        target = str(job["target"])
        evaluation_dir = (
            run_dir
            / "evaluations"
            / dataset
            / source
            / attack
            / suffix
            / target
        )
        evaluation_dirs[target] = evaluation_dir
        group_status.update(
            status="evaluating_targets",
            current_target=target,
            current_target_index=target_index,
        )
        atomic_json(group_status_path, group_status)
        current = {
            "kind": "attack_target_evaluation",
            "group_index": group_index,
            "dataset": dataset,
            "source": source,
            "attack": attack,
            "variant": variant,
            "target": target,
            "target_index": target_index,
        }
        events.put({"type": "task_started", "slot": slot, "current": current})
        try:
            completed = run_evaluation(
                registry,
                dataset,
                target,
                split=split,
                adversarial_run=generated,
                output_dir=evaluation_dir,
                max_images=settings["max_images"],
                device=device,
                download_weights=bool(settings["download_weights"]),
                keep_going=bool(settings.get("keep_going", False)),
                save_visualizations=bool(settings["save_visualizations"]),
                visualization_score_threshold=float(
                    settings["visualization_score_threshold"]
                ),
                visualization_max_images=int(
                    settings["visualization_max_images"]
                ),
                visualization_max_detections=int(
                    settings["visualization_max_detections"]
                ),
                prediction_archive=settings["prediction_archive"],
                image_ids=settings.get("image_ids_by_dataset", {}).get(dataset),
            )
            record = _read_metrics(completed / "metrics.json")
            record["study"] = study
            record["variant"] = variant
            parameter_key = "parameters" if settings.get("generic_experiment", False) else "parameter_overrides"
            record[parameter_key] = group_jobs[0].get("parameters", {})
            record["budget_profile"] = job["budget_profile"]
            record["budget"] = job["budget"]
        except Exception as exc:
            reason = "evaluation failed: {}: {}".format(
                type(exc).__name__, exc
            )
            record = {**job, "status": "failed", "reason": reason, "metrics": {}}
            group_status.update(
                status="failed_target_panel",
                reason=reason,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            atomic_json(group_status_path, group_status)
            group_failed = True
            if not settings.get("keep_going", False):
                events.put(
                    {
                        "type": "fatal", "slot": slot, "current": current,
                        "records": [record], "reason": reason,
                    }
                )
                return False
        finally:
            _cleanup_cuda()
        group_failed = group_failed or record.get("status") != "complete"
        group_status["targets_completed"] = target_index
        atomic_json(group_status_path, group_status)
        events.put(
            {
                "type": "attack_evaluation_complete",
                "slot": slot,
                "current": current,
                "record": record,
            }
        )

    if group_failed:
        group_status.update(
            status="failed_target_panel",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(group_status_path, group_status)
        events.put({
            "type": "group_failed", "slot": slot, "current": current,
            "records": [], "reason": "One or more target evaluations failed",
        })
        return True

    if settings["retention_mode"] != RETENTION_MODE:
        group_status.update(
            status="complete_full_payload_retained",
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        group_status.pop("current_target", None)
        group_status.pop("current_target_index", None)
        atomic_json(group_status_path, group_status)
        events.put(
            {
                "type": "group_complete",
                "slot": slot,
                "current": current,
                "group_summary": group_status,
            }
        )
        return True
    group_status["status"] = "validating_before_prune"
    group_status.pop("current_target", None)
    group_status.pop("current_target_index", None)
    atomic_json(group_status_path, group_status)
    current = {
        "kind": "group_validation_and_prune",
        "group_index": group_index,
        "dataset": dataset,
        "source": source,
        "attack": attack,
        "variant": variant,
    }
    events.put({"type": "task_started", "slot": slot, "current": current})
    try:
        retention_summary = validate_and_prune_attack_group(
            registry,
            dataset,
            split,
            source,
            attack,
            generated,
            evaluation_dirs,
            [str(job["target"]) for job in group_jobs],
            retained_selection,
        )
    except BaseException as exc:
        reason = "group validation/prune failed: {}: {}".format(
            type(exc).__name__, exc
        )
        group_status.update(
            status="failed_validation_or_prune",
            reason=reason,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(group_status_path, group_status)
        events.put(
            {
                "type": "fatal",
                "slot": slot,
                "current": current,
                "records": [],
                "reason": reason,
            }
        )
        return False
    retention_path = generated / "retention_summary.json"
    group_status.update(
        retention_summary=str(retention_path.relative_to(run_dir)),
        retention_summary_sha256=file_digest(retention_path),
        retained_images=int(retention_summary["retained_images"]),
        pruned_images=int(retention_summary["images_to_prune"]),
        status="complete_validated_and_pruned",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_json(group_status_path, group_status)
    events.put(
        {
            "type": "group_complete",
            "slot": slot,
            "current": current,
            "group_summary": group_status,
        }
    )
    return True


def _worker_main(
    root: str,
    run_dir: str,
    device: str,
    slot: int,
    stage: str,
    tasks: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    retained_selection: Mapping[str, Any],
    events: Any,
) -> None:
    events = _WorkerEvents(events, stage, slot)
    owner = None
    try:
        if sys.platform == "linux":
            from .vcsf_structure_workers import install_parent_death_guard

            install_parent_death_guard(int(settings["coordinator_pid"]))
        if settings.get("dispatch_journal_enabled"):
            from .dispatch_journal import WorkerJournal

            events.journal = WorkerJournal(run_dir, stage, slot, device, tasks,
                                          int(settings["coordinator_pid"]), torch)
        if not _CUDA_DEVICE.fullmatch(device):
            raise RuntimeError("Parallel worker requires an explicit CUDA device")
        if (stage == "groups" and settings.get("selected_a23_owner_request")
                and tasks):
            from .vcsf_selected_a23_owner import SelectedA23ContextOwner

            owner = SelectedA23ContextOwner(
                settings["selected_a23_owner_request"], Path(run_dir), device,
                slot, [int(task["canonical_index"]) for task in tasks], torch,
            )
            owner.register()
            start_event = settings.get("selected_a23_start_event")
            if start_event is None or not start_event.wait(timeout=120):
                raise RuntimeError("Selected A23 coordinator sampling was not ready")
        elif not (stage == "groups" and settings.get("selected_a23_owner_request")):
            torch.cuda.set_device(int(device.split(":", 1)[1]))
        registry = Registry(Path(root))
        destination = Path(run_dir)
        for task in tasks:
            events.task_index = int(task["canonical_index"])
            if owner is not None:
                owner.start_task(events.task_index)
            if stage == "clean":
                proceed = _clean_task(
                    registry, destination, device, slot, task, settings, events
                )
            elif stage == "groups":
                proceed = _group_task(
                    registry,
                    destination,
                    device,
                    slot,
                    task,
                    settings,
                    retained_selection,
                    events,
                )
            else:
                raise RuntimeError("Unknown parallel stage: {}".format(stage))
            if not proceed:
                if owner is not None:
                    owner.record_failure(RuntimeError("source group did not complete"))
                return
            if owner is not None:
                owner.finish_task(events.task_index)
        events.task_index = None
        if owner is not None:
            owner.close()
        events.put({"type": "worker_done", "slot": slot, "stage": stage})
    except BaseException as exc:
        if owner is not None:
            try:
                owner.record_failure(exc)
            except Exception:
                pass
        events.put(
            {
                "type": "fatal",
                "slot": slot,
                "current": None,
                "records": [],
                "reason": "parallel worker failed: {}: {}".format(
                    type(exc).__name__, exc
                ),
            }
        )


def _run_stage(
    *,
    stage: str,
    registry: Registry,
    run_dir: Path,
    devices: Sequence[str],
    assignments: Sequence[Sequence[Mapping[str, Any]]],
    settings: Mapping[str, Any],
    retained_selection: Mapping[str, Any],
    records: List[Dict[str, Any]],
    expected_keys: Sequence[tuple],
    records_path: Path,
    execution_state: MutableMapping[str, Any],
    state_path: Path,
    group_summaries: List[Dict[str, Any]],
) -> None:
    context = mp.get_context("spawn")
    processes = []
    if len(assignments) != len(devices):
        raise ValueError("Each execution device requires one assignment list")
    task_maps = [
        {int(task["canonical_index"]): task for task in tasks}
        for tasks in assignments
    ]
    all_indices = [int(task["canonical_index"]) for tasks in assignments for task in tasks]
    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError("Parallel assignments contain duplicate task identities")
    closed_tasks = [set() for _ in devices]
    task_record_keys: Dict[tuple, List[tuple]] = {}
    seen_task_records: Dict[tuple, List[Dict[str, Any]]] = {}
    for slot, tasks in enumerate(assignments):
        for task in tasks:
            index = int(task["canonical_index"])
            task_record_keys[(slot, index)] = experiment_expected_record_keys(
                {"jobs": task["jobs"]} if stage == "groups" else {"jobs": []},
                [(str(task["dataset"]), str(task["split"]), str(task["target"]))]
                if stage == "clean" else [],
            )
            seen_task_records[(slot, index)] = []
    events = context.Queue()
    workers = [
        {
            "slot": slot,
            "device": str(device),
            "stage": stage,
            "status": "running",
            "pid": None,
            "current": None,
        }
        for slot, device in enumerate(devices)
    ]
    execution_state["workers"] = workers
    execution_state["phase"] = (
        "clean_evaluations_parallel"
        if stage == "clean"
        else "attack_groups_parallel"
    )

    def publish_state() -> None:
        active = [
            worker["current"]
            for worker in workers
            if worker.get("status") == "running" and worker.get("current")
        ]
        execution_state["current"] = active[0] if active else None
        execution_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, execution_state)

    completed_workers = 0
    failure: str = ""
    normal_completion = False
    owner_monitor = None
    journal = None
    try:
        worker_settings = {**settings, "coordinator_pid": os.getpid()}
        if (sys.platform == "linux" and settings.get("generic_experiment")
                and not settings.get("selected_a23_owner_request")):
            from .dispatch_journal import AppendOnlyJournal, process_birth

            journal = AppendOnlyJournal(run_dir / "dispatch_journal" / stage / "coordinator.jsonl",
                dict(schema="generic_dispatch_coordinator_v1", stage=stage,
                     coordinator=process_birth(os.getpid()), devices=list(devices),
                     assignments=[list(tasks) for tasks in assignments]))
            worker_settings["dispatch_journal_enabled"] = True
        if stage == "groups" and settings.get("selected_a23_owner_request"):
            from .vcsf_selected_a23_owner import process_identity

            worker_settings["selected_a23_owner_request"] = {
                **settings["selected_a23_owner_request"],
                "coordinator_pid": os.getpid(),
                "coordinator_start_ticks": process_identity(os.getpid())["start_ticks"],
            }
            worker_settings["selected_a23_start_event"] = context.Event()
        for slot, device in enumerate(devices):
            process = context.Process(
                target=_worker_main,
                args=(
                    str(registry.root), str(run_dir), str(device), slot, stage,
                    list(assignments[slot]), worker_settings,
                    dict(retained_selection), events,
                ),
                name="formal-{}-worker-{}".format(stage, slot),
            )
            processes.append(process)
            if journal is not None:
                journal.append("worker_launch_requested", slot=slot, device=str(device))
            process.start()
            workers[slot]["pid"] = process.pid
            if journal is not None:
                birth = process_birth(process.pid)
                if birth["parent_pid"] != os.getpid():
                    raise RuntimeError("Spawned dispatch worker has another parent")
                workers[slot]["process"] = birth
                journal.append("worker_spawned", slot=slot, device=str(device), process=birth,
                    worker_journal="dispatch_journal/{}/worker-{}.jsonl".format(stage, slot))
        publish_state()
        if stage == "groups" and worker_settings.get("selected_a23_owner_request"):
            from .vcsf_selected_a23_owner import SelectedA23RuntimeMonitor

            owner_monitor = SelectedA23RuntimeMonitor(
                worker_settings["selected_a23_owner_request"], run_dir,
                [process.pid for process in processes],
                [slot for slot, tasks in enumerate(assignments) if tasks],
            )
            owner_monitor.sample(workers, "workers_started")
            worker_settings["selected_a23_start_event"].set()
            next_owner_sample_at = time.monotonic() + 2.0
        while completed_workers < len(processes):
            if owner_monitor is not None and time.monotonic() >= next_owner_sample_at:
                owner_monitor.sample(workers, "interval")
                next_owner_sample_at = time.monotonic() + 2.0
            try:
                event = events.get(timeout=2.0)
            except queue.Empty:
                unexpected = [
                    (slot, process.exitcode)
                    for slot, process in enumerate(processes)
                    if process.exitcode is not None and (
                        process.exitcode != 0 or workers[slot]["status"] != "complete"
                    )
                ]
                if unexpected:
                    failure = "Parallel worker exited unexpectedly: {}".format(
                        unexpected
                    )
                    break
                continue
            slot = int(event["slot"])
            event_type = str(event["type"])
            if not 0 <= slot < len(workers):
                raise RuntimeError("Parallel event has an unassigned worker slot")
            if event.get("pid") != processes[slot].pid or event.get("stage") != stage:
                raise RuntimeError("Parallel event process or stage ownership mismatch")
            if journal is not None:
                if event.get("worker_start_ticks") != workers[slot]["process"]["start_ticks"]:
                    raise RuntimeError("Parallel event process birth ownership mismatch")
                registration = event.get("gpu_context_registration")
                if registration is not None:
                    if (registration.get("device") != devices[slot]
                            or registration.get("worker_pid") != processes[slot].pid
                            or registration.get("nvml_pid") != processes[slot].pid
                            or registration.get("worker_start_ticks") != event["worker_start_ticks"]
                            or not isinstance(registration.get("gpu_uuid"), str)
                            or not registration["gpu_uuid"].startswith("GPU-")
                            or not isinstance(registration.get("observed_context_pids"), list)
                            or processes[slot].pid not in registration["observed_context_pids"]):
                        raise RuntimeError("Parallel event GPU context ownership mismatch")
                    prior = workers[slot].get("gpu_context_registration")
                    if prior is not None and prior != registration:
                        raise RuntimeError("Parallel worker GPU context registration changed")
                    workers[slot]["gpu_context_registration"] = registration
            if workers[slot]["status"] != "running":
                raise RuntimeError("Parallel event arrived after worker completion")
            index = event.get("task_index")
            if event_type == "worker_done":
                if index is not None or closed_tasks[slot] != set(task_maps[slot]):
                    raise RuntimeError("Parallel worker completed before its assigned tasks")
            elif event_type != "fatal" or index is not None:
                if index not in task_maps[slot] or index in closed_tasks[slot]:
                    raise RuntimeError("Parallel event has an unassigned or closed task")
                task = task_maps[slot][index]
                current = event.get("current") or {}
                for field in ("dataset", "source", "attack", "study", "variant"):
                    if field in current and str(current[field]) != str(task.get(field, "")):
                        raise RuntimeError("Parallel event task identity mismatch: " + field)
                allowed_targets = (
                    [str(task["target"])] if stage == "clean"
                    else [str(job["target"]) for job in task["jobs"]]
                )
                if "target" in current and str(current["target"]) not in allowed_targets:
                    raise RuntimeError("Parallel event target is not assigned to its task")
                incoming = (
                    [event["record"]] if "record" in event else list(event.get("records", []))
                )
                seen_task_records[(slot, index)] = canonical_partial_records(
                    seen_task_records[(slot, index)] + incoming,
                    task_record_keys[(slot, index)],
                )
            elif event.get("records"):
                raise RuntimeError("Unassigned worker failure cannot publish records")
            if journal is not None:
                if (event_type == "worker_done" and assignments[slot]
                        and workers[slot].get("gpu_context_registration") is None):
                    raise RuntimeError("Completed parallel worker lacks context registration")
                journal.append("validated_worker_event", slot=slot, pid=event["pid"],
                    start_ticks=event["worker_start_ticks"], task_index=index,
                    event_type=event_type, current=event.get("current"),
                    gpu_context_registration=event.get("gpu_context_registration"),
                    journal_error=event.get("journal_error"))
            if event.get("current") is not None:
                workers[slot]["current"] = dict(event["current"])
            if event_type in {"task_started", "task_progress"}:
                publish_state()
                if (event_type == "task_started" and owner_monitor is not None
                        and current.get("kind") == "attack_generation"):
                    owner_monitor.sample(
                        workers, "task_started", task_slot=slot, task_id=index
                    )
                    next_owner_sample_at = time.monotonic() + 2.0
                continue
            if event_type in {"clean_complete", "attack_evaluation_complete"}:
                if (event_type == "clean_complete") != (stage == "clean"):
                    raise RuntimeError("Parallel completion event belongs to another stage")
                records.append(dict(event["record"]))
                records[:] = canonical_partial_records(records, expected_keys)
                atomic_json(records_path, records)
                counter = (
                    "clean_evaluations_completed"
                    if event_type == "clean_complete"
                    else "attack_evaluations_completed"
                )
                execution_state[counter] = int(execution_state[counter]) + 1
                execution_state["failed_records"] = sum(
                    record.get("status") in {"failed", "complete_with_failures"}
                    for record in records
                )
                if event_type == "clean_complete":
                    closed_tasks[slot].add(index)
                publish_state()
                continue
            if event_type == "group_complete":
                if stage != "groups" or len(seen_task_records[(slot, index)]) != len(
                    task_record_keys[(slot, index)]
                ) or any(record.get("status") != "complete"
                         for record in seen_task_records[(slot, index)]):
                    raise RuntimeError("Parallel group closed without its full target panel")
                group_summary = dict(event["group_summary"])
                task = task_maps[slot][index]
                if group_summary.get("index") != index or any(
                    str(group_summary.get(field, "")) != str(task.get(field, ""))
                    for field in ("dataset", "split", "source", "attack", "study", "variant")
                ):
                    raise RuntimeError("Parallel group summary identity mismatch")
                if group_summary.get("status") not in {
                    "complete_full_payload_retained", "complete_validated_and_pruned"
                }:
                    raise RuntimeError("Parallel group summary is not complete")
                closed_tasks[slot].add(index)
                group_summaries.append(group_summary)
                execution_state["attack_groups_completed"] = int(
                    execution_state["attack_groups_completed"]
                ) + 1
                if "new_payload_images_completed_group_lower_bound" in execution_state:
                    execution_state[
                        "new_payload_images_completed_group_lower_bound"
                    ] = int(
                        execution_state[
                            "new_payload_images_completed_group_lower_bound"
                        ]
                    ) + int(group_summary.get("generation_images_total", 0))
                publish_state()
                continue
            if event_type == "group_failed":
                if stage != "groups" or not settings.get("keep_going", False):
                    raise RuntimeError("Recoverable group failure is not enabled")
                if len(seen_task_records[(slot, index)]) != len(task_record_keys[(slot, index)]):
                    raise RuntimeError("Failed group is missing explicit target records")
                records[:] = canonical_partial_records(
                    records + list(event.get("records", [])), expected_keys
                )
                atomic_json(records_path, records)
                execution_state["failed_records"] = sum(
                    record.get("status") in {"failed", "complete_with_failures"}
                    for record in records
                )
                closed_tasks[slot].add(index)
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
                records[:] = canonical_partial_records(records, expected_keys)
                atomic_json(records_path, records)
                workers[slot]["status"] = "failed"
                execution_state["failed_records"] = sum(
                    record.get("status") in {"failed", "complete_with_failures"}
                    for record in records
                )
                failure = str(event.get("reason") or "parallel worker failed")
                publish_state()
                break
            failure = "Unknown parallel worker event: {}".format(event_type)
            break
        if failure:
            raise RuntimeError(failure)
        for slot, process in enumerate(processes):
            process.join(timeout=30.0)
            if process.exitcode != 0:
                raise RuntimeError("Parallel worker did not exit cleanly after completion")
            if owner_monitor is not None:
                workers[slot]["exitcode"] = process.exitcode
        try:
            events.get_nowait()
        except queue.Empty:
            pass
        else:
            raise RuntimeError("Parallel event arrived after all workers completed")
        if owner_monitor is not None:
            owner_monitor.sample(workers, "stage_complete")
            owner_monitor.require_seen()
        normal_completion = True
    finally:
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is None:
                continue
            process.join(timeout=30.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=10.0)
        events.close()
        events.join_thread()
        if not normal_completion:
            for worker in workers:
                if worker["status"] == "running":
                    worker["status"] = "terminated_after_failure"
            publish_state()
        if journal is not None:
            journal.append("stage_cleanup", normal_completion=normal_completion,
                workers=[dict(worker, exitcode=process.exitcode, alive=process.is_alive())
                         for worker, process in zip(workers, processes)])
        if owner_monitor is not None:
            owner_monitor.finish(normal_completion)


def run_formal_parallel(
    *,
    registry: Registry,
    run_dir: Path,
    devices: Sequence[str],
    clean_jobs: Sequence[Tuple[str, str, str]],
    group_items: Sequence[Tuple[tuple, Sequence[Mapping[str, Any]]]],
    settings: Mapping[str, Any],
    retained_selection: Mapping[str, Any],
    records: List[Dict[str, Any]],
    expected_keys: Sequence[tuple],
    records_path: Path,
    execution_state: MutableMapping[str, Any],
    state_path: Path,
    group_summaries: List[Dict[str, Any]],
) -> None:
    """Run clean and complete source-method groups on independent GPUs."""
    clean_indices = round_robin_indices(len(clean_jobs), len(devices))
    group_indices = round_robin_indices(len(group_items), len(devices))
    clean_assignments: List[List[Dict[str, Any]]] = []
    for indices in clean_indices:
        clean_assignments.append(
            [
                {
                    "canonical_index": index + 1,
                    "dataset": clean_jobs[index][0],
                    "split": clean_jobs[index][1],
                    "target": clean_jobs[index][2],
                }
                for index in indices
            ]
        )
    group_assignments: List[List[Dict[str, Any]]] = []
    for indices in group_indices:
        assigned = []
        for index in indices:
            key, jobs = group_items[index]
            dataset, split, source, attack, study, variant = key
            assigned.append(
                {
                    "canonical_index": index + 1,
                    "group_total": len(group_items),
                    "dataset": dataset,
                    "split": split,
                    "source": source,
                    "attack": attack,
                    "study": study,
                    "variant": variant,
                    "jobs": [dict(job) for job in jobs],
                }
            )
        group_assignments.append(assigned)
    clean_completed_before = int(execution_state["clean_evaluations_completed"])
    _run_stage(
        stage="clean",
        registry=registry,
        run_dir=run_dir,
        devices=devices,
        assignments=clean_assignments,
        settings=settings,
        retained_selection=retained_selection,
        records=records,
        expected_keys=expected_keys,
        records_path=records_path,
        execution_state=execution_state,
        state_path=state_path,
        group_summaries=group_summaries,
    )
    # Strict barrier: no attack generation starts until all clean workers exit cleanly.
    if (
        int(execution_state["clean_evaluations_completed"])
        - clean_completed_before
        != len(clean_jobs)
    ):
        raise RuntimeError("Parallel clean-evaluation barrier is incomplete")
    execution_state["workers"] = []
    execution_state["current"] = None
    execution_state["phase"] = "clean_barrier_complete"
    execution_state["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, execution_state)
    if not group_items:
        return
    _run_stage(
        stage="groups",
        registry=registry,
        run_dir=run_dir,
        devices=devices,
        assignments=group_assignments,
        settings=settings,
        retained_selection=retained_selection,
        records=records,
        expected_keys=expected_keys,
        records_path=records_path,
        execution_state=execution_state,
        state_path=state_path,
        group_summaries=group_summaries,
    )


def run_parallel_scheduler_diagnostic(
    *,
    registry: Registry,
    run_dir: Path,
    devices: Sequence[str],
    clean_jobs: Sequence[Tuple[str, str, str]],
    group_items: Sequence[Tuple[tuple, Sequence[Mapping[str, Any]]]],
    seed: int = 42,
) -> Path:
    """Exercise the real spawn/coordinator path on disjoint COCO dev images."""
    devices = normalize_execution_devices("cuda:0", devices)
    if len(devices) != 2:
        raise ValueError("The scheduler diagnostic requires exactly two devices")
    validate_available_cuda_devices(devices)
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite non-empty scheduler diagnostic: {}".format(
                run_dir
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    if len(clean_jobs) != len(devices) or len(group_items) != len(devices):
        raise ValueError(
            "The scheduler diagnostic requires one clean and one group per device"
        )
    expected_keys = [
        ("clean", "clean", str(target)) for _, _, target in clean_jobs
    ] + [
        (str(job["source"]), str(job["attack"]), str(job["target"]))
        for _, jobs in group_items
        for job in jobs
    ]
    records: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    records_path = run_dir / "records.json"
    state_path = run_dir / "execution_state.json"
    atomic_json(records_path, records)
    started_at = datetime.now(timezone.utc).isoformat()
    execution_state: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "protocol": "parallel_scheduler_diagnostic",
        "phase": "initializing",
        "current": None,
        "clean_evaluations_total": len(clean_jobs),
        "clean_evaluations_completed": 0,
        "attack_groups_total": len(group_items),
        "attack_groups_completed": 0,
        "attack_evaluations_total": sum(len(jobs) for _, jobs in group_items),
        "attack_evaluations_completed": 0,
        "structural_skips": 0,
        "failed_records": 0,
        "workers": [],
        "started_at": started_at,
        "updated_at": started_at,
    }
    atomic_json(state_path, execution_state)
    settings = {
        "max_images": 1,
        "seed": int(seed),
        "download_weights": False,
        "save_visualizations": False,
        "visualization_score_threshold": 0.3,
        "visualization_max_images": 1,
        "visualization_max_detections": 1,
        "prediction_archive": None,
        "retention_mode": "keep_all",
    }
    clean_assignments = [
        [
            {
                "canonical_index": index + 1,
                "dataset": clean_jobs[index][0],
                "split": clean_jobs[index][1],
                "target": clean_jobs[index][2],
            }
        ]
        for index in range(len(devices))
    ]
    group_assignments: List[List[Dict[str, Any]]] = []
    for index, (key, jobs) in enumerate(group_items):
        dataset, split, source, attack, study, variant = key
        group_assignments.append(
            [
                {
                    "canonical_index": index + 1,
                    "group_total": len(group_items),
                    "dataset": dataset,
                    "split": split,
                    "source": source,
                    "attack": attack,
                    "study": study,
                    "variant": variant,
                    "jobs": [dict(job) for job in jobs],
                }
            ]
        )
    try:
        _run_stage(
            stage="clean",
            registry=registry,
            run_dir=run_dir,
            devices=devices,
            assignments=clean_assignments,
            settings=settings,
            retained_selection={},
            records=records,
            expected_keys=expected_keys,
            records_path=records_path,
            execution_state=execution_state,
            state_path=state_path,
            group_summaries=group_summaries,
        )
        if int(execution_state["clean_evaluations_completed"]) != len(clean_jobs):
            raise RuntimeError("Diagnostic clean barrier did not complete")
        execution_state["workers"] = []
        execution_state["phase"] = "clean_barrier_complete"
        execution_state["current"] = None
        execution_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, execution_state)
        _run_stage(
            stage="groups",
            registry=registry,
            run_dir=run_dir,
            devices=devices,
            assignments=group_assignments,
            settings=settings,
            retained_selection={},
            records=records,
            expected_keys=expected_keys,
            records_path=records_path,
            execution_state=execution_state,
            state_path=state_path,
            group_summaries=group_summaries,
        )
        if (
            len(records) != len(expected_keys)
            or len(group_summaries) != len(group_items)
            or any(record.get("status") != "complete" for record in records)
            or any(record.get("images") != 1 for record in records)
        ):
            raise RuntimeError("Parallel scheduler diagnostic audit failed")
        execution_state.update(
            status="complete",
            phase="complete",
            current=None,
            workers=[],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, execution_state)
        atomic_json(
            run_dir / "summary.json",
            {
                "status": "complete",
                "formal_eligible": False,
                "diagnostic_only": True,
                "devices": list(devices),
                "clean_evaluations": len(clean_jobs),
                "attack_groups": len(group_items),
                "attack_evaluations": sum(
                    len(jobs) for _, jobs in group_items
                ),
                "images_per_evaluation": 1,
                "records": len(records),
                "records_sha256": file_digest(records_path),
                "started_at": started_at,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        return run_dir
    except (Exception, KeyboardInterrupt) as exc:
        execution_state.update(
            status="failed",
            phase="failed",
            current=None,
            workers=[],
            reason="{}: {}".format(type(exc).__name__, exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, execution_state)
        atomic_json(
            run_dir / "summary.json",
            {
                "status": "failed",
                "formal_eligible": False,
                "diagnostic_only": True,
                "reason": execution_state["reason"],
                "records": len(records),
                "started_at": started_at,
                "failed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise
