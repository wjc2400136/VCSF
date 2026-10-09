from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import torch
from filelock import FileLock, Timeout
from mmengine.config import Config

from lgp.data.manager import validate_dataset
from lgp.io import atomic_json, file_digest
from lgp.modeling import checkpoint_path
from lgp.paths import project_root
from lgp.registry import Registry
from lgp.runtime_config import resume_phase_switch_boundary
from lgp.training_artifacts import (
    strict_load_finetuned_checkpoint,
    validate_finetuned_checkpoint,
)


_ACTIVE_CHILD: Optional[subprocess.Popen] = None
_STOP_SIGNAL: Optional[int] = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, *, missing_ok: bool = False) -> Dict[str, Any]:
    if not path.is_file():
        if missing_ok:
            return {}
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError("JSON state is unreadable: {}".format(path)) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("JSON state root must be a mapping: {}".format(path))
    return payload


def _git_state(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--porcelain=v1", "--untracked-files=normal"),
    }


def _process_identity(pid: int) -> Optional[Dict[str, Any]]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
        stat = Path("/proc/{}/stat".format(pid)).read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2 :].split()
        start_ticks = tail[19]
        os.kill(pid, 0)
    except (OSError, IndexError, ValueError):
        return None
    return {"pid": int(pid), "boot_id": boot_id, "start_ticks": start_ticks}


def _identity_alive(identity: Any) -> bool:
    if not isinstance(identity, dict):
        return False
    try:
        current = _process_identity(int(identity["pid"]))
    except (KeyError, TypeError, ValueError):
        return False
    return current == {
        "pid": int(identity["pid"]),
        "boot_id": identity.get("boot_id"),
        "start_ticks": str(identity.get("start_ticks")),
    }


def _signal_handler(signum: int, _frame: Any) -> None:
    global _STOP_SIGNAL
    _STOP_SIGNAL = signum
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def _child_setup(parent_pid: int) -> None:
    """Create a process group and make Linux terminate it when the queue dies."""
    import ctypes

    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        os._exit(126)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _package_versions() -> Dict[str, str]:
    distributions = (
        "torch",
        "torchvision",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmyolo",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
        "albumentations",
        "qudida",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
    )
    return {
        name: importlib.metadata.version(name)
        for name in distributions
    }


def _cached_digest(
    path: Path, cache: MutableMapping[Path, str]
) -> str:
    resolved = path.resolve()
    if resolved not in cache:
        cache[resolved] = file_digest(resolved)
    return cache[resolved]


def _unique_manifest_records(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    if int(payload.get("schema_version", -1)) != 1:
        raise RuntimeError("Unsupported checkpoint manifest schema: {}".format(path))
    records = payload.get("checkpoints")
    if not isinstance(records, list):
        raise RuntimeError("Checkpoint manifest has no record list: {}".format(path))
    indexed: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not record.get("model"):
            raise RuntimeError("Invalid checkpoint manifest record: {}".format(path))
        model_id = str(record["model"])
        if model_id in indexed:
            raise RuntimeError(
                "Duplicate {} record in {}".format(model_id, path)
            )
        indexed[model_id] = record
    return indexed


def _source_fingerprint(
    registry: Registry, hash_cache: MutableMapping[Path, str]
) -> Dict[str, str]:
    manifest_path = registry.root / "checkpoints" / "coco" / "manifest.json"
    records = _unique_manifest_records(manifest_path)
    if set(records) != set(registry.target_ids()):
        raise RuntimeError("COCO manifest must contain the canonical 16 detectors")
    coco = registry.dataset("coco")
    result: Dict[str, str] = {}
    for model_id in registry.target_ids():
        path = checkpoint_path(registry.model(model_id), coco).resolve()
        record = records[model_id]
        if (
            not path.is_file()
            or Path(str(record.get("path", ""))).name != path.name
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise RuntimeError(
                "COCO source path/size mismatch for {}".format(model_id)
            )
        digest = _cached_digest(path, hash_cache)
        if str(record.get("sha256", "")).lower() != digest.lower():
            raise RuntimeError(
                "COCO source SHA-256 mismatch for {}".format(model_id)
            )
        result[model_id] = digest
    return result


def _dataset_fingerprint(
    registry: Registry,
    datasets: List[str],
    hash_cache: MutableMapping[Path, str],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for dataset_id in datasets:
        dataset = registry.dataset(dataset_id)
        report = validate_dataset(dataset, deep=True, full=True)
        if report.get("status") != "valid":
            raise RuntimeError(
                "{} failed full deep validation: errors={!r}, warnings={!r}".format(
                    dataset_id,
                    report.get("errors", []),
                    report.get("warnings", []),
                )
            )
        annotation = (
            dataset.root / dataset.split("train").annotation
        ).resolve()
        digest = _cached_digest(annotation, hash_cache)
        result[dataset_id] = {
            "train_annotation": annotation.relative_to(
                registry.root.resolve()
            ).as_posix(),
            "train_annotation_sha256": digest,
            "splits": {
                name: int(split["annotation_images"])
                for name, split in report["splits"].items()
                if "annotation_images" in split
            },
            "train_val_shared_paths": int(
                report.get("split_overlap", {}).get("train_val_images", 0)
            ),
        }
    return result


def _campaign_fingerprint(
    registry: Registry,
    datasets: List[str],
    args: argparse.Namespace,
    git: Mapping[str, Any],
    hash_cache: MutableMapping[Path, str],
) -> Dict[str, Any]:
    root = registry.root.resolve()
    config_paths = [
        root / "configs" / "models.yaml",
        root / "configs" / "datasets" / "coco.yaml",
        root / "configs" / "datasets" / "voc.yaml",
        root / "configs" / "datasets" / "bdd100k.yaml",
    ]
    return {
        "git_commit": git["commit"],
        "datasets": datasets,
        "canonical_models": registry.target_ids(),
        "protocol": registry.training_protocol,
        "device": args.device,
        "max_attempts": args.max_attempts,
        "min_free_gib": args.min_free_gib,
        "python": platform.python_version(),
        "packages": _package_versions(),
        "cuda_runtime": torch.version.cuda,
        "gpu": (
            torch.cuda.get_device_name(torch.device(args.device).index or 0)
            if torch.cuda.is_available()
            else None
        ),
        "configs": {
            path.relative_to(root).as_posix(): _cached_digest(path, hash_cache)
            for path in config_paths
        },
        "datasets_state": _dataset_fingerprint(
            registry, datasets, hash_cache
        ),
        "coco_manifest_sha256": file_digest(
            root / "checkpoints" / "coco" / "manifest.json"
        ),
        "source_checkpoints": _source_fingerprint(registry, hash_cache),
    }


def _assert_campaign_unchanged(
    registry: Registry,
    fingerprint: Mapping[str, Any],
    dataset_id: str,
    model_id: str,
) -> Dict[str, Any]:
    git = _git_state(registry.root)
    if git["status"] or git["commit"] != fingerprint["git_commit"]:
        raise RuntimeError("Git worktree/commit changed during the campaign")
    if (
        registry.target_ids() != fingerprint["canonical_models"]
        or registry.training_protocol != fingerprint["protocol"]
        or _package_versions() != fingerprint["packages"]
    ):
        raise RuntimeError("Registry, protocol, or environment changed during campaign")
    dataset = registry.dataset(dataset_id)
    annotation = (
        dataset.root / dataset.split("train").annotation
    ).resolve()
    if (
        file_digest(annotation)
        != fingerprint["datasets_state"][dataset_id][
            "train_annotation_sha256"
        ]
    ):
        raise RuntimeError("{} train annotation changed".format(dataset_id))
    source = checkpoint_path(
        registry.model(model_id), registry.dataset("coco")
    ).resolve()
    if file_digest(source) != fingerprint["source_checkpoints"][model_id]:
        raise RuntimeError("{} COCO source checkpoint changed".format(model_id))
    if file_digest(
        registry.root / "checkpoints" / "coco" / "manifest.json"
    ) != fingerprint["coco_manifest_sha256"]:
        raise RuntimeError("COCO checkpoint manifest changed")
    return git


def _checkpoint_ready(
    registry: Registry,
    dataset_id: str,
    model_id: str,
    commit: str,
    hash_cache: Optional[MutableMapping[Path, str]] = None,
) -> Tuple[bool, str]:
    ready, issues = validate_finetuned_checkpoint(
        registry,
        dataset_id,
        model_id,
        deep=True,
        expected_commit=commit,
        hash_cache=hash_cache,
    )
    return ready, "verified" if ready else "; ".join(issues)


def _attempt_result_detail(
    returncode: int,
    checkpoint_ready: bool,
    checkpoint_detail: str,
    log_reference: str,
) -> str:
    if returncode != 0:
        return (
            "training child exited with return code {}; inspect {}; "
            "no checkpoint was accepted ({})"
        ).format(returncode, log_reference, checkpoint_detail)
    if not checkpoint_ready:
        return (
            "training child exited successfully but checkpoint validation "
            "failed: {}"
        ).format(checkpoint_detail)
    return checkpoint_detail


def _next_work_dir(
    campaign_dir: Path, dataset_id: str, model_id: str
) -> Tuple[Path, Optional[str], int]:
    model_root = campaign_dir / "runs" / dataset_id / model_id
    numbered: List[Tuple[int, Path]] = []
    if model_root.is_dir():
        for path in model_root.iterdir():
            if (
                path.is_dir()
                and path.name.startswith("attempt_")
                and path.name[8:].isdigit()
            ):
                numbered.append((int(path.name[8:]), path))
    if numbered:
        latest_number, latest = max(numbered, key=lambda item: item[0])
        if (latest / "last_checkpoint").is_file():
            pointer_value = (latest / "last_checkpoint").read_text(
                encoding="utf-8"
            ).strip()
            if not pointer_value:
                raise RuntimeError(
                    "Empty resume pointer in {}".format(latest)
                )
            pointer = Path(pointer_value)
            if not pointer.is_absolute():
                pointer = latest / pointer
            pointer = pointer.resolve()
            try:
                pointer.relative_to(latest.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    "Resume pointer escapes its work directory"
                ) from exc
            if not pointer.is_file():
                raise FileNotFoundError(pointer)
            protocol_path = latest / "protocol_config.py"
            if not protocol_path.is_file():
                raise FileNotFoundError(protocol_path)
            boundary = resume_phase_switch_boundary(
                Config.fromfile(str(protocol_path))
            )
            checkpoint_epoch: Optional[int] = None
            name = pointer.stem
            if name.startswith("epoch_") and name[6:].isdigit():
                checkpoint_epoch = int(name[6:])
            if (
                boundary is None
                or checkpoint_epoch is None
                or checkpoint_epoch < boundary
            ):
                return latest, "auto", latest_number
        next_number = latest_number + 1
    else:
        next_number = 1
    return model_root / "attempt_{:04d}".format(next_number), None, next_number


def _disk_gate(paths: List[Path], min_free_gib: float) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    seen_devices = set()
    threshold = int(min_free_gib * (1 << 30))
    for requested in paths:
        anchor = requested.resolve()
        while not anchor.exists():
            if anchor.parent == anchor:
                raise RuntimeError("No existing disk anchor for {}".format(requested))
            anchor = anchor.parent
        device = anchor.stat().st_dev
        if device in seen_devices:
            continue
        seen_devices.add(device)
        usage = shutil.disk_usage(str(anchor))
        checks.append(
            {
                "path": str(requested),
                "anchor": str(anchor),
                "free_bytes": usage.free,
                "required_free_bytes": threshold,
            }
        )
        if usage.free < threshold:
            raise RuntimeError(
                "Free disk {:.2f} GiB at {} is below {:.2f} GiB".format(
                    usage.free / float(1 << 30), anchor, min_free_gib
                )
            )
    return checks


def _event(state: Dict[str, Any], kind: str, **fields: Any) -> None:
    state.setdefault("events", []).append(
        {"at": _utc_now(), "kind": kind, **fields}
    )
    state["updated_at"] = _utc_now()


def _gate_dataset(
    registry: Registry,
    dataset_id: str,
    commit: str,
    device: str,
    hash_cache: MutableMapping[Path, str],
    expected_commits: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    model_ids = registry.target_ids()
    if expected_commits is not None and set(expected_commits) != set(model_ids):
        raise RuntimeError(
            "{} gate expected-commit map must cover the canonical detector panel".format(
                dataset_id
            )
        )
    artifacts: Dict[str, str] = {}
    artifact_git_commits: Dict[str, str] = {}
    for model_id in model_ids:
        artifact_commit = (
            commit
            if expected_commits is None
            else str(expected_commits[model_id])
        )
        ready, reason = _checkpoint_ready(
            registry, dataset_id, model_id, artifact_commit, hash_cache
        )
        if not ready:
            raise RuntimeError(
                "{}/{} failed the deep gate: {}".format(
                    dataset_id, model_id, reason
                )
            )
        strict_load_finetuned_checkpoint(
            registry, dataset_id, model_id, device
        )
        path = checkpoint_path(
            registry.model(model_id), registry.dataset(dataset_id)
        ).resolve()
        artifacts[model_id] = _cached_digest(path, hash_cache)
        artifact_git_commits[model_id] = artifact_commit
    manifest = registry.root / "checkpoints" / dataset_id / "manifest.json"
    snapshot = {
        "dataset": dataset_id,
        "git_commit": commit,
        "artifact_git_commits": artifact_git_commits,
        "validated_at": _utc_now(),
        "manifest_sha256": file_digest(manifest),
        "artifacts": artifacts,
        "strict_model_loads": len(artifacts),
    }
    canonical = json.dumps(
        snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    snapshot["snapshot_sha256"] = hashlib.sha256(canonical).hexdigest()
    return snapshot


def _assert_gate_snapshot(
    registry: Registry, snapshot: Mapping[str, Any]
) -> None:
    dataset_id = str(snapshot["dataset"])
    manifest = registry.root / "checkpoints" / dataset_id / "manifest.json"
    if file_digest(manifest) != snapshot["manifest_sha256"]:
        raise RuntimeError("{} gate manifest changed".format(dataset_id))
    for model_id, expected in snapshot["artifacts"].items():
        path = checkpoint_path(
            registry.model(model_id), registry.dataset(dataset_id)
        )
        if not path.is_file() or file_digest(path) != expected:
            raise RuntimeError(
                "{} gate artifact changed: {}".format(dataset_id, model_id)
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the audited VOC-then-BDD100K 16-detector fine-tune queue"
    )
    parser.add_argument(
        "--datasets",
        default="voc,bdd100k",
        help="Must be voc or voc,bdd100k; BDD100K is gated on verified VOC 16/16",
    )
    parser.add_argument("--campaign", default="formal_voc_bdd_seed42")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--min-free-gib", type=float, default=12.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate and hash the frozen campaign inputs without launching training",
    )
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_CHILD, _STOP_SIGNAL
    args = _parse_args()
    datasets = [
        value.strip() for value in args.datasets.split(",") if value.strip()
    ]
    if datasets not in (["voc"], ["voc", "bdd100k"]):
        raise ValueError("--datasets must preserve the exact order voc,bdd100k")
    if args.max_attempts <= 0:
        raise ValueError("--max-attempts must be positive")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        raise ValueError("--min-free-gib must be finite and positive")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal queue requires an available CUDA device")

    registry = Registry()
    root = project_root().resolve()
    campaign_dir = (
        root / "outputs" / "training" / "campaigns" / args.campaign
    ).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    queue_path = campaign_dir / "queue.json"
    pid_path = campaign_dir / "queue.pid"
    lock = FileLock(str(campaign_dir / "queue.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise RuntimeError("Another fine-tune queue holds the campaign lock") from exc

    state: Dict[str, Any] = {}
    original_handlers: Dict[int, Any] = {}
    try:
        identity = _process_identity(os.getpid())
        if identity is None:
            raise RuntimeError("Cannot determine queue process identity")
        atomic_json(pid_path, {**identity, "started_at": _utc_now()})
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _signal_handler)

        git = _git_state(root)
        if git["status"]:
            raise RuntimeError(
                "Formal training requires a clean Git worktree"
            )
        hash_cache: Dict[Path, str] = {}
        fingerprint = _campaign_fingerprint(
            registry, datasets, args, git, hash_cache
        )
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "git_commit": fingerprint["git_commit"],
                        "datasets": fingerprint["datasets_state"],
                        "source_checkpoints": len(
                            fingerprint["source_checkpoints"]
                        ),
                        "packages": fingerprint["packages"],
                        "gpu": fingerprint["gpu"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        state = _load_json(queue_path, missing_ok=True)
        if state:
            if (
                int(state.get("schema_version", -1)) != 2
                or state.get("campaign") != args.campaign
                or state.get("fingerprint") != fingerprint
            ):
                raise RuntimeError(
                    "Existing campaign fingerprint differs; use a new campaign name"
                )
            active_identity = state.get("active_child", {}).get("identity")
            if _identity_alive(active_identity):
                raise RuntimeError(
                    "Recorded training child is still alive; refusing duplicate spawn"
                )
            if state.get("active_child"):
                _event(
                    state,
                    "stale_child_recovered",
                    child=state["active_child"],
                )
                state["active_child"] = None
        else:
            state = {
                "schema_version": 2,
                "campaign": args.campaign,
                "fingerprint": fingerprint,
                "started_at": _utc_now(),
                "jobs": {},
                "gates": {},
                "events": [],
                "next_attempt_id": 1,
                "active_child": None,
            }
        state.pop("failure", None)
        state.pop("completed_at", None)
        state["status"] = "running"
        state["controller"] = identity
        _event(state, "queue_started")
        atomic_json(queue_path, state)

        for dataset_id in datasets:
            if dataset_id == "bdd100k":
                voc_gate = state.get("gates", {}).get("voc")
                if not isinstance(voc_gate, dict):
                    raise RuntimeError(
                        "BDD100K gate is closed until VOC passes 16 deep validations"
                    )
                _assert_gate_snapshot(registry, voc_gate)
            for model_id in registry.target_ids():
                if _STOP_SIGNAL is not None:
                    raise InterruptedError(
                        "Queue received signal {}".format(_STOP_SIGNAL)
                    )
                if dataset_id == "bdd100k":
                    _assert_gate_snapshot(registry, state["gates"]["voc"])
                _assert_campaign_unchanged(
                    registry, fingerprint, dataset_id, model_id
                )
                job_id = "{}/{}".format(dataset_id, model_id)
                job = state["jobs"].setdefault(
                    job_id, {"status": "pending", "attempts": []}
                )
                if not isinstance(job.get("attempts"), list):
                    raise RuntimeError("{} attempt history is invalid".format(job_id))
                ready, reason = _checkpoint_ready(
                    registry,
                    dataset_id,
                    model_id,
                    fingerprint["git_commit"],
                    hash_cache,
                )
                if ready:
                    job.update(
                        {
                            "status": "complete",
                            "detail": reason,
                            "updated_at": _utc_now(),
                        }
                    )
                    atomic_json(queue_path, state)
                    continue
                destination = checkpoint_path(
                    registry.model(model_id), registry.dataset(dataset_id)
                )
                if destination.exists():
                    job.update(
                        {
                            "status": "conflict",
                            "detail": reason,
                            "updated_at": _utc_now(),
                        }
                    )
                    atomic_json(queue_path, state)
                    raise RuntimeError(
                        "{} has an existing unverified checkpoint; no automatic "
                        "overwrite is permitted: {}".format(job_id, reason)
                    )

                completed = False
                while not completed:
                    if len(job["attempts"]) >= args.max_attempts:
                        raise RuntimeError(
                            "{} exhausted {} persistent attempts".format(
                                job_id, args.max_attempts
                            )
                        )
                    _assert_campaign_unchanged(
                        registry, fingerprint, dataset_id, model_id
                    )
                    disk_checks = _disk_gate(
                        [
                            campaign_dir / "runs" / dataset_id / model_id,
                            destination.parent,
                        ],
                        args.min_free_gib,
                    )
                    work_dir, resume, run_attempt = _next_work_dir(
                        campaign_dir, dataset_id, model_id
                    )
                    attempt_id = int(state["next_attempt_id"])
                    state["next_attempt_id"] = attempt_id + 1
                    log_path = (
                        campaign_dir
                        / "logs"
                        / dataset_id
                        / "{}_attempt_{:04d}.log".format(model_id, attempt_id)
                    )
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    command = [
                        sys.executable,
                        "-m",
                        "lgp",
                        "train",
                        "--dataset",
                        dataset_id,
                        "--model",
                        model_id,
                        "--work-dir",
                        str(work_dir),
                        "--device",
                        args.device,
                    ]
                    if resume:
                        command.extend(["--resume", resume])
                    attempt = {
                        "attempt_id": attempt_id,
                        "status": "launching",
                        "run_attempt": run_attempt,
                        "resume": resume,
                        "work_dir": work_dir.relative_to(root).as_posix(),
                        "log": log_path.relative_to(root).as_posix(),
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
                    _event(
                        state,
                        "attempt_launching",
                        job=job_id,
                        attempt_id=attempt_id,
                    )
                    atomic_json(queue_path, state)

                    parent_pid = os.getpid()
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
                            _ACTIVE_CHILD = subprocess.Popen(
                                command,
                                cwd=str(root),
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                text=True,
                                preexec_fn=lambda: _child_setup(parent_pid),
                            )
                            child_identity = _process_identity(_ACTIVE_CHILD.pid)
                            attempt["identity"] = child_identity or {
                                "pid": _ACTIVE_CHILD.pid
                            }
                            attempt["status"] = "running"
                            job["status"] = "running"
                            state["active_child"] = {
                                "job": job_id,
                                "attempt_id": attempt_id,
                                "identity": attempt["identity"],
                            }
                            atomic_json(queue_path, state)
                            last_heartbeat = 0.0
                            while _ACTIVE_CHILD.poll() is None:
                                time.sleep(15)
                                now = time.monotonic()
                                if now - last_heartbeat >= 60:
                                    attempt["heartbeat_at"] = _utc_now()
                                    state["updated_at"] = _utc_now()
                                    atomic_json(queue_path, state)
                                    last_heartbeat = now
                            returncode = int(_ACTIVE_CHILD.returncode)
                            log.write(
                                "[{}] EXIT {}\n".format(_utc_now(), returncode)
                            )
                    except BaseException as exc:
                        attempt["status"] = "launch_failed"
                        attempt["failure"] = "{}: {}".format(
                            type(exc).__name__, exc
                        )
                        attempt["updated_at"] = _utc_now()
                        state["active_child"] = None
                        atomic_json(queue_path, state)
                        raise
                    finally:
                        _ACTIVE_CHILD = None

                    state["active_child"] = None
                    attempt["returncode"] = returncode
                    attempt["completed_at"] = _utc_now()
                    if _STOP_SIGNAL is not None:
                        attempt["status"] = "interrupted"
                        job["status"] = "interrupted"
                        atomic_json(queue_path, state)
                        raise InterruptedError(
                            "Queue received signal {}".format(_STOP_SIGNAL)
                        )
                    _assert_campaign_unchanged(
                        registry, fingerprint, dataset_id, model_id
                    )
                    hash_cache.clear()
                    ready, reason = _checkpoint_ready(
                        registry,
                        dataset_id,
                        model_id,
                        fingerprint["git_commit"],
                        hash_cache,
                    )
                    completed = returncode == 0 and ready
                    detail = _attempt_result_detail(
                        returncode,
                        ready,
                        reason,
                        attempt["log"],
                    )
                    attempt["status"] = "complete" if completed else "failed"
                    attempt["detail"] = detail
                    attempt["checkpoint_gate_detail"] = reason
                    job.update(
                        {
                            "status": "complete" if completed else "failed",
                            "detail": detail,
                            "updated_at": _utc_now(),
                        }
                    )
                    _event(
                        state,
                        "attempt_complete" if completed else "attempt_failed",
                        job=job_id,
                        attempt_id=attempt_id,
                        returncode=returncode,
                        detail=detail,
                    )
                    atomic_json(queue_path, state)
                    if not completed and destination.exists():
                        job["status"] = "conflict"
                        atomic_json(queue_path, state)
                        raise RuntimeError(
                            "{} produced an unverified checkpoint; refusing overwrite: "
                            "{}".format(job_id, detail)
                        )

            hash_cache.clear()
            gate = _gate_dataset(
                registry,
                dataset_id,
                fingerprint["git_commit"],
                args.device,
                hash_cache,
            )
            state["gates"][dataset_id] = gate
            _event(
                state,
                "dataset_gate_passed",
                dataset=dataset_id,
                snapshot_sha256=gate["snapshot_sha256"],
            )
            atomic_json(queue_path, state)

        state["status"] = "complete"
        state["completed_at"] = _utc_now()
        state["updated_at"] = state["completed_at"]
        _event(state, "queue_complete")
        atomic_json(queue_path, state)
        return 0
    except BaseException as exc:
        if state:
            state["status"] = (
                "interrupted"
                if isinstance(exc, (KeyboardInterrupt, InterruptedError))
                or _STOP_SIGNAL is not None
                else "failed"
            )
            state["failure"] = "{}: {}".format(type(exc).__name__, exc)
            _event(state, "queue_stopped", failure=state["failure"])
            atomic_json(queue_path, state)
        raise
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)
        try:
            if pid_path.is_file():
                current = _load_json(pid_path)
                live = _process_identity(os.getpid())
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
