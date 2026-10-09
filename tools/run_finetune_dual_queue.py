from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence

import torch
from filelock import FileLock, Timeout


_STOP_SIGNAL: Optional[int] = None
_STOP_EVENT = threading.Event()
_CHILDREN_LOCK = threading.RLock()
_ACTIVE_CHILDREN: Dict[str, subprocess.Popen] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_device_list(value: str) -> List[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise ValueError("--devices requires at least two CUDA devices")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicates")
    parsed = [torch.device(item) for item in devices]
    if any(device.type != "cuda" or device.index is None for device in parsed):
        raise ValueError("--devices entries must be explicit CUDA indices")
    return ["cuda:{}".format(device.index) for device in parsed]


def _round_robin_assignments(
    model_ids: Sequence[str], devices: Sequence[str]
) -> Dict[str, List[str]]:
    if not devices:
        raise ValueError("At least one device is required")
    assignments = {device: [] for device in devices}
    for index, model_id in enumerate(model_ids):
        assignments[devices[index % len(devices)]].append(model_id)
    return assignments


def _parse_model_list(
    value: str, canonical_model_ids: Sequence[str]
) -> List[str]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if len(set(requested)) != len(requested):
        raise ValueError("--frozen-existing must not contain duplicates")
    unknown = sorted(set(requested).difference(canonical_model_ids))
    if unknown:
        raise ValueError(
            "--frozen-existing contains unknown detector ids: {}".format(
                ", ".join(unknown)
            )
        )
    selected = set(requested)
    return [
        model_id
        for model_id in canonical_model_ids
        if model_id in selected
    ]


def _json_digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _capture_frozen_existing(
    serial: ModuleType,
    registry: Any,
    dataset_id: str,
    model_ids: Sequence[str],
    device: str,
    hash_cache: MutableMapping[Path, str],
) -> Dict[str, Dict[str, Any]]:
    from lgp.modeling import checkpoint_path
    from lgp.training_artifacts import (
        strict_load_finetuned_checkpoint,
        validate_finetuned_checkpoint,
    )

    if not model_ids:
        return {}
    manifest_path = (
        registry.root / "checkpoints" / dataset_id / "manifest.json"
    )
    records = serial._unique_manifest_records(manifest_path)
    frozen: Dict[str, Dict[str, Any]] = {}
    for model_id in model_ids:
        ready, issues = validate_finetuned_checkpoint(
            registry,
            dataset_id,
            model_id,
            deep=True,
            expected_commit=None,
            hash_cache=hash_cache,
        )
        if not ready:
            raise RuntimeError(
                "{}/{} cannot be frozen: {}".format(
                    dataset_id, model_id, "; ".join(issues)
                )
            )
        record = records.get(model_id)
        if not isinstance(record, dict):
            raise RuntimeError(
                "{}/{} has no manifest record to freeze".format(
                    dataset_id, model_id
                )
            )
        git_commit = record.get("git_commit")
        if not isinstance(git_commit, str) or not git_commit:
            raise RuntimeError(
                "{}/{} has no Git provenance to freeze".format(
                    dataset_id, model_id
                )
            )
        path = checkpoint_path(
            registry.model(model_id), registry.dataset(dataset_id)
        ).resolve()
        strict_load_finetuned_checkpoint(
            registry, dataset_id, model_id, device
        )
        stat = path.stat()
        frozen[model_id] = {
            "git_commit": git_commit,
            "path": path.relative_to(registry.root.resolve()).as_posix(),
            "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": serial._cached_digest(path, hash_cache),
            "manifest_record_sha256": _json_digest(record),
            "strict_state_dict_load": True,
        }
    return frozen


def _expected_commit_map(
    current_commit: str,
    model_ids: Sequence[str],
    frozen_existing: Mapping[str, Mapping[str, Any]],
) -> Dict[str, str]:
    return {
        model_id: str(
            frozen_existing.get(model_id, {}).get(
                "git_commit", current_commit
            )
        )
        for model_id in model_ids
    }


def _assert_frozen_existing_unchanged(
    serial: ModuleType,
    registry: Any,
    dataset_id: str,
    frozen_existing: Mapping[str, Mapping[str, Any]],
    *,
    deep: bool,
) -> None:
    from lgp.training_artifacts import validate_finetuned_checkpoint

    if not frozen_existing:
        return
    records = serial._unique_manifest_records(
        registry.root / "checkpoints" / dataset_id / "manifest.json"
    )
    for model_id, expected in frozen_existing.items():
        path = (registry.root / str(expected["path"])).resolve()
        try:
            path.relative_to(registry.root.resolve())
        except ValueError as exc:
            raise RuntimeError(
                "Frozen checkpoint path escapes the repository"
            ) from exc
        if not path.is_file():
            raise RuntimeError(
                "Frozen checkpoint disappeared: {}/{}".format(
                    dataset_id, model_id
                )
            )
        stat = path.stat()
        if (
            stat.st_size != int(expected["bytes"])
            or stat.st_mtime_ns != int(expected["mtime_ns"])
        ):
            raise RuntimeError(
                "Frozen checkpoint metadata changed: {}/{}".format(
                    dataset_id, model_id
                )
            )
        record = records.get(model_id)
        if (
            not isinstance(record, dict)
            or _json_digest(record) != expected["manifest_record_sha256"]
        ):
            raise RuntimeError(
                "Frozen manifest record changed: {}/{}".format(
                    dataset_id, model_id
                )
            )
        if deep:
            ready, issues = validate_finetuned_checkpoint(
                registry,
                dataset_id,
                model_id,
                deep=True,
                expected_commit=str(expected["git_commit"]),
            )
            if not ready:
                raise RuntimeError(
                    "Frozen checkpoint failed its final deep validation: "
                    "{}/{}: {}".format(
                        dataset_id, model_id, "; ".join(issues)
                    )
                )
            if serial.file_digest(path) != expected["sha256"]:
                raise RuntimeError(
                    "Frozen checkpoint SHA-256 changed: {}/{}".format(
                        dataset_id, model_id
                    )
                )


def _load_serial_queue(training_root: Path) -> ModuleType:
    src = str((training_root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)
    path = training_root / "tools" / "run_finetune_queue.py"
    spec = importlib.util.spec_from_file_location(
        "lgp_serial_finetune_queue", str(path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the audited serial queue: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _controller_digest() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _controller_git_state(serial: ModuleType) -> Dict[str, Any]:
    return serial._git_state(Path(__file__).resolve().parents[1])


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    _STOP_EVENT.set()
    with _CHILDREN_LOCK:
        children = list(_ACTIVE_CHILDREN.values())
    for child in children:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run two or more independent single-GPU workers over the canonical "
            "VOC detector panel without changing per-model batch or learning rate"
        )
    )
    parser.add_argument(
        "--training-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help=(
            "Clean repository used by training children. This may differ from "
            "the controller worktree."
        ),
    )
    parser.add_argument("--dataset", default="voc", choices=["voc"])
    parser.add_argument("--campaign", default="formal_voc_dual_seed42")
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument(
        "--frozen-existing",
        default="",
        help=(
            "Comma-separated canonical detector ids whose already-published "
            "checkpoints must be deeply validated, strict-loaded, fingerprinted "
            "and reused without rewriting their original Git provenance"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate the frozen inputs and print the worker assignment",
    )
    return parser.parse_args()


def main() -> int:
    global _STOP_SIGNAL
    args = _parse_args()
    devices = _parse_device_list(args.devices)
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be finite and positive")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal queue requires available CUDA devices")
    device_count = torch.cuda.device_count()
    for value in devices:
        index = int(value.split(":", 1)[1])
        if index >= device_count:
            raise RuntimeError(
                "{} is unavailable; torch reports {} CUDA devices".format(
                    value, device_count
                )
            )

    training_root = args.training_root.expanduser().resolve()
    configured_root = os.environ.get("LGP_PROJECT_ROOT")
    if (
        configured_root
        and Path(configured_root).expanduser().resolve() != training_root
    ):
        raise RuntimeError(
            "LGP_PROJECT_ROOT conflicts with --training-root: {}".format(
                configured_root
            )
        )
    os.environ["LGP_PROJECT_ROOT"] = str(training_root)
    serial = _load_serial_queue(training_root)
    from lgp.io import atomic_json
    from lgp.modeling import checkpoint_path
    from lgp.registry import Registry

    registry = Registry(root=training_root)
    model_ids = registry.target_ids()
    frozen_model_ids = _parse_model_list(
        args.frozen_existing, model_ids
    )
    work_model_ids = [
        model_id
        for model_id in model_ids
        if model_id not in frozen_model_ids
    ]
    if not work_model_ids:
        raise ValueError(
            "--frozen-existing cannot cover the entire detector panel"
        )
    assignments = _round_robin_assignments(work_model_ids, devices)
    controller_root = Path(__file__).resolve().parents[1]
    controller_git = _controller_git_state(serial)
    if controller_git["status"]:
        raise RuntimeError("Dual-queue controller worktree must be clean")
    training_git = serial._git_state(training_root)
    if training_git["status"]:
        raise RuntimeError("Formal training requires a clean Git worktree")

    campaign_dir = (
        training_root
        / "outputs"
        / "training"
        / "campaigns"
        / args.campaign
    ).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    state_path = campaign_dir / "queue.json"
    pid_path = campaign_dir / "queue.pid"
    lock = FileLock(str(campaign_dir / "queue.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError("Another dual fine-tune queue holds the campaign lock") from exc

    state: Dict[str, Any] = {}
    state_lock = threading.RLock()
    original_handlers: Dict[int, Any] = {}

    def write_state() -> None:
        with state_lock:
            atomic_json(state_path, state)

    def event(kind: str, **fields: Any) -> None:
        with state_lock:
            serial._event(state, kind, **fields)
            atomic_json(state_path, state)

    def assert_controller_unchanged() -> None:
        current = _controller_git_state(serial)
        if (
            current["commit"] != controller_git["commit"]
            or current["status"]
            or _controller_digest() != fingerprint["controller_sha256"]
        ):
            raise RuntimeError("Dual-queue controller changed during the campaign")

    def stop_children() -> None:
        _STOP_EVENT.set()
        with _CHILDREN_LOCK:
            children = list(_ACTIVE_CHILDREN.values())
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    try:
        identity = serial._process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("Cannot determine dual-queue process identity")
        atomic_json(pid_path, {**identity, "started_at": _utc_now()})
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _signal_handler)

        hash_cache: Dict[Path, str] = {}
        base_args = SimpleNamespace(
            device=devices[0],
            max_attempts=args.max_attempts,
            min_free_gib=args.min_free_gib,
        )
        fingerprint = serial._campaign_fingerprint(
            registry,
            [args.dataset],
            base_args,
            training_git,
            hash_cache,
        )
        frozen_existing = _capture_frozen_existing(
            serial,
            registry,
            args.dataset,
            frozen_model_ids,
            devices[0],
            hash_cache,
        )
        expected_commits = _expected_commit_map(
            fingerprint["git_commit"], model_ids, frozen_existing
        )
        fingerprint.update(
            {
                "mode": "independent_single_gpu_workers",
                "devices": devices,
                "gpu_names": [
                    torch.cuda.get_device_name(int(value.split(":", 1)[1]))
                    for value in devices
                ],
                "assignments": assignments,
                "work_model_ids": work_model_ids,
                "frozen_existing": frozen_existing,
                "controller_git_commit": controller_git["commit"],
                "controller_sha256": _controller_digest(),
            }
        )
        fingerprint.pop("device", None)
        fingerprint.pop("gpu", None)

        existing_ready: Dict[str, bool] = {}
        existing_details: Dict[str, str] = {}
        for model_id in model_ids:
            ready, reason = serial._checkpoint_ready(
                registry,
                args.dataset,
                model_id,
                expected_commits[model_id],
                hash_cache,
            )
            existing_ready[model_id] = ready
            existing_details[model_id] = reason
            destination = checkpoint_path(
                registry.model(model_id), registry.dataset(args.dataset)
            )
            if (
                destination.exists()
                and not ready
                and model_id not in frozen_existing
            ):
                raise RuntimeError(
                    "{}/{} has an existing checkpoint that is neither a "
                    "verified current-campaign artifact nor explicitly frozen: "
                    "{}".format(args.dataset, model_id, reason)
                )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "training_git_commit": fingerprint["git_commit"],
                        "controller_git_commit": controller_git["commit"],
                        "dataset": args.dataset,
                        "devices": devices,
                        "assignments": assignments,
                        "frozen_existing": frozen_existing,
                        "verified_existing": [
                            model_id
                            for model_id, ready in existing_ready.items()
                            if ready
                        ],
                        "pending": [
                            model_id
                            for model_id, ready in existing_ready.items()
                            if not ready
                        ],
                        "existing_details": existing_details,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0

        state = serial._load_json(state_path, missing_ok=True)
        if state:
            if (
                int(state.get("schema_version", -1)) != 2
                or state.get("campaign") != args.campaign
                or state.get("fingerprint") != fingerprint
            ):
                raise RuntimeError(
                    "Existing dual campaign fingerprint differs; use a new campaign name"
                )
            active = state.get("active_children", {})
            if not isinstance(active, dict):
                raise RuntimeError("Recorded active_children state is invalid")
            live = [
                record
                for record in active.values()
                if serial._identity_alive(record.get("identity"))
            ]
            if live:
                raise RuntimeError(
                    "Recorded training children are still alive; refusing duplicate spawn"
                )
            if active:
                event("stale_children_recovered", children=active)
            state["active_children"] = {}
        else:
            state = {
                "schema_version": 2,
                "campaign": args.campaign,
                "fingerprint": fingerprint,
                "started_at": _utc_now(),
                "status": "starting",
                "jobs": {},
                "events": [],
                "next_attempt_id": 1,
                "active_children": {},
                "gates": {},
            }
        for model_id in frozen_model_ids:
            job_id = "{}/{}".format(args.dataset, model_id)
            job = state["jobs"].setdefault(
                job_id,
                {
                    "status": "complete",
                    "device": None,
                    "attempts": [],
                },
            )
            if job.get("attempts") not in ([], None):
                raise RuntimeError(
                    "{} frozen job unexpectedly contains attempts".format(
                        job_id
                    )
                )
            job.update(
                {
                    "status": "complete",
                    "device": None,
                    "attempts": [],
                    "detail": "frozen_verified_existing",
                    "reused_verified_checkpoint": True,
                    "frozen_provenance": frozen_existing[model_id],
                    "updated_at": _utc_now(),
                }
            )
        state.pop("failure", None)
        state.pop("completed_at", None)
        state["status"] = "running"
        state["controller"] = identity
        serial._event(state, "dual_queue_started")
        atomic_json(state_path, state)

        def run_worker(device: str, assigned: Sequence[str]) -> None:
            worker_cache: MutableMapping[Path, str] = {}
            for model_id in assigned:
                if _STOP_EVENT.is_set():
                    raise InterruptedError("Dual queue stop requested")
                assert_controller_unchanged()
                _assert_frozen_existing_unchanged(
                    serial,
                    registry,
                    args.dataset,
                    frozen_existing,
                    deep=False,
                )
                serial._assert_campaign_unchanged(
                    registry, fingerprint, args.dataset, model_id
                )
                job_id = "{}/{}".format(args.dataset, model_id)
                ready, reason = serial._checkpoint_ready(
                    registry,
                    args.dataset,
                    model_id,
                    expected_commits[model_id],
                    worker_cache,
                )
                with state_lock:
                    job = state["jobs"].setdefault(
                        job_id,
                        {
                            "status": "pending",
                            "device": device,
                            "attempts": [],
                        },
                    )
                    if job.get("device") != device:
                        raise RuntimeError(
                            "{} device assignment changed".format(job_id)
                        )
                    if not isinstance(job.get("attempts"), list):
                        raise RuntimeError(
                            "{} attempt history is invalid".format(job_id)
                        )
                    if ready:
                        job.update(
                            {
                                "status": "complete",
                                "detail": reason,
                                "reused_verified_checkpoint": True,
                                "updated_at": _utc_now(),
                            }
                        )
                        serial._event(
                            state,
                            "verified_checkpoint_reused",
                            job=job_id,
                            device=device,
                        )
                        atomic_json(state_path, state)
                        continue

                destination = checkpoint_path(
                    registry.model(model_id), registry.dataset(args.dataset)
                )
                if destination.exists():
                    with state_lock:
                        job["status"] = "conflict"
                        job["detail"] = reason
                        job["updated_at"] = _utc_now()
                        atomic_json(state_path, state)
                    raise RuntimeError(
                        "{} has an existing unverified checkpoint; no automatic "
                        "overwrite is permitted: {}".format(job_id, reason)
                    )

                completed = False
                while not completed:
                    if _STOP_EVENT.is_set():
                        raise InterruptedError("Dual queue stop requested")
                    assert_controller_unchanged()
                    serial._assert_campaign_unchanged(
                        registry, fingerprint, args.dataset, model_id
                    )
                    with state_lock:
                        job = state["jobs"][job_id]
                        if len(job["attempts"]) >= args.max_attempts:
                            raise RuntimeError(
                                "{} exhausted {} persistent attempts".format(
                                    job_id, args.max_attempts
                                )
                            )
                    disk_checks = serial._disk_gate(
                        [
                            campaign_dir / "runs" / args.dataset / model_id,
                            destination.parent,
                        ],
                        args.min_free_gib,
                    )
                    work_dir, resume, run_attempt = serial._next_work_dir(
                        campaign_dir, args.dataset, model_id
                    )
                    with state_lock:
                        attempt_id = int(state["next_attempt_id"])
                        state["next_attempt_id"] = attempt_id + 1
                        log_path = (
                            campaign_dir
                            / "logs"
                            / args.dataset
                            / "{}_attempt_{:04d}.log".format(
                                model_id, attempt_id
                            )
                        )
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        command = [
                            sys.executable,
                            "-m",
                            "lgp",
                            "train",
                            "--dataset",
                            args.dataset,
                            "--model",
                            model_id,
                            "--work-dir",
                            str(work_dir),
                            "--device",
                            device,
                        ]
                        if resume:
                            command.extend(["--resume", resume])
                        attempt = {
                            "attempt_id": attempt_id,
                            "status": "launching",
                            "device": device,
                            "run_attempt": run_attempt,
                            "resume": resume,
                            "work_dir": work_dir.relative_to(training_root).as_posix(),
                            "log": log_path.relative_to(training_root).as_posix(),
                            "command": command,
                            "disk_preflight": disk_checks,
                            "started_at": _utc_now(),
                        }
                        job["attempts"].append(attempt)
                        job.update(
                            {
                                "status": "launching",
                                "updated_at": _utc_now(),
                                "preflight": reason,
                            }
                        )
                        serial._event(
                            state,
                            "attempt_launching",
                            job=job_id,
                            attempt_id=attempt_id,
                            device=device,
                        )
                        atomic_json(state_path, state)

                    parent_pid = os.getpid()
                    returncode: Optional[int] = None
                    child: Optional[subprocess.Popen] = None
                    child_env = os.environ.copy()
                    training_src = str((training_root / "src").resolve())
                    existing_pythonpath = child_env.get("PYTHONPATH")
                    child_env["PYTHONPATH"] = (
                        training_src
                        if not existing_pythonpath
                        else training_src + os.pathsep + existing_pythonpath
                    )
                    try:
                        with log_path.open(
                            "a", encoding="utf-8", newline="\n"
                        ) as log:
                            log.write(
                                "\n[{}] START {}\n".format(
                                    _utc_now(), " ".join(command)
                                )
                            )
                            log.flush()
                            child = subprocess.Popen(
                                command,
                                cwd=str(training_root),
                                env=child_env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                text=True,
                                preexec_fn=lambda: serial._child_setup(parent_pid),
                            )
                            with _CHILDREN_LOCK:
                                _ACTIVE_CHILDREN[device] = child
                            child_identity = serial._process_identity(child.pid)
                            with state_lock:
                                attempt["identity"] = child_identity or {
                                    "pid": child.pid
                                }
                                attempt["status"] = "running"
                                job["status"] = "running"
                                state["active_children"][device] = {
                                    "job": job_id,
                                    "attempt_id": attempt_id,
                                    "identity": attempt["identity"],
                                }
                                atomic_json(state_path, state)
                            last_heartbeat = 0.0
                            while child.poll() is None:
                                time.sleep(15)
                                now = time.monotonic()
                                if now - last_heartbeat >= 60:
                                    with state_lock:
                                        attempt["heartbeat_at"] = _utc_now()
                                        state["updated_at"] = _utc_now()
                                        atomic_json(state_path, state)
                                    last_heartbeat = now
                            returncode = int(child.returncode)
                            log.write(
                                "[{}] EXIT {}\n".format(_utc_now(), returncode)
                            )
                    except BaseException as exc:
                        with state_lock:
                            attempt["status"] = "launch_failed"
                            attempt["failure"] = "{}: {}".format(
                                type(exc).__name__, exc
                            )
                            attempt["updated_at"] = _utc_now()
                            state["active_children"].pop(device, None)
                            atomic_json(state_path, state)
                        raise
                    finally:
                        with _CHILDREN_LOCK:
                            if _ACTIVE_CHILDREN.get(device) is child:
                                _ACTIVE_CHILDREN.pop(device, None)

                    with state_lock:
                        state["active_children"].pop(device, None)
                        attempt["returncode"] = returncode
                        attempt["completed_at"] = _utc_now()
                    if _STOP_EVENT.is_set():
                        with state_lock:
                            attempt["status"] = "interrupted"
                            job["status"] = "interrupted"
                            atomic_json(state_path, state)
                        raise InterruptedError("Dual queue stop requested")

                    assert_controller_unchanged()
                    serial._assert_campaign_unchanged(
                        registry, fingerprint, args.dataset, model_id
                    )
                    worker_cache.clear()
                    ready, reason = serial._checkpoint_ready(
                        registry,
                        args.dataset,
                        model_id,
                        expected_commits[model_id],
                        worker_cache,
                    )
                    completed = returncode == 0 and ready
                    detail = serial._attempt_result_detail(
                        returncode,
                        ready,
                        reason,
                        attempt["log"],
                    )
                    with state_lock:
                        attempt["status"] = (
                            "complete" if completed else "failed"
                        )
                        attempt["detail"] = detail
                        attempt["checkpoint_gate_detail"] = reason
                        job.update(
                            {
                                "status": (
                                    "complete" if completed else "failed"
                                ),
                                "detail": detail,
                                "updated_at": _utc_now(),
                            }
                        )
                        serial._event(
                            state,
                            (
                                "attempt_complete"
                                if completed
                                else "attempt_failed"
                            ),
                            job=job_id,
                            attempt_id=attempt_id,
                            device=device,
                            returncode=returncode,
                            detail=detail,
                        )
                        atomic_json(state_path, state)
                    if not completed and destination.exists():
                        with state_lock:
                            job["status"] = "conflict"
                            atomic_json(state_path, state)
                        raise RuntimeError(
                            "{} produced an unverified checkpoint; refusing "
                            "overwrite: {}".format(job_id, detail)
                        )

        futures: List[Future[None]] = []
        first_failure: Optional[BaseException] = None
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            for device in devices:
                futures.append(
                    executor.submit(run_worker, device, assignments[device])
                )
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    if first_failure is None:
                        first_failure = exc
                        stop_children()
            if first_failure is not None:
                raise first_failure

        assert_controller_unchanged()
        _assert_frozen_existing_unchanged(
            serial,
            registry,
            args.dataset,
            frozen_existing,
            deep=True,
        )
        serial._assert_campaign_unchanged(
            registry, fingerprint, args.dataset, model_ids[-1]
        )
        hash_cache.clear()
        gate = serial._gate_dataset(
            registry,
            args.dataset,
            fingerprint["git_commit"],
            devices[0],
            hash_cache,
            expected_commits=expected_commits,
        )
        state["gates"][args.dataset] = gate
        state["status"] = "complete"
        state["completed_at"] = _utc_now()
        state["updated_at"] = state["completed_at"]
        serial._event(
            state,
            "dataset_gate_passed",
            dataset=args.dataset,
            snapshot_sha256=gate["snapshot_sha256"],
        )
        serial._event(state, "dual_queue_complete")
        atomic_json(state_path, state)
        return 0
    except BaseException as exc:
        stop_children()
        if state:
            state["status"] = (
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, InterruptedError))
                or _STOP_SIGNAL is not None
                else "failed"
            )
            state["failure"] = "{}: {}".format(type(exc).__name__, exc)
            serial._event(
                state, "dual_queue_stopped", failure=state["failure"]
            )
            atomic_json(state_path, state)
        raise
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        try:
            if pid_path.is_file():
                current = serial._load_json(pid_path)
                live = serial._process_identity(os.getpid())
                if live is not None and all(
                    current.get(key) == live.get(key)
                    for key in ("pid", "boot_id", "start_ticks")
                ):
                    pid_path.unlink()
        except (OSError, RuntimeError):
            pass
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
