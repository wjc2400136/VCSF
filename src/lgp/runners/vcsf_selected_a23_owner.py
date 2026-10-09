"""Worker-side GPU context binding for selected-A23 formal source waves."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..io import file_digest
from .public_device_helpers import selected_physical_gpu_uuids
from .vcsf_selected_a23_plan import PROTOCOLS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Selected A23 GPU owner: " + message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def process_identity(pid: int) -> dict:
    _require(type(pid) is int and pid > 0, "invalid process ID")
    raw = (Path("/proc") / str(pid) / "stat").read_text(encoding="ascii")
    fields = raw.rsplit(") ", 1)[1].split()
    _require(len(fields) > 19, "incomplete Linux process identity")
    return {
        "pid": pid,
        "parent_pid": int(fields[1]),
        "process_group": int(fields[2]),
        "session_id": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def compute_pids(gpu_uuid: str) -> set:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
         "--format=csv,noheader,nounits"], text=True, timeout=10,
    )
    found = set()
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        _require(len(row) == 2 and row[1].strip().isdigit(),
                 "malformed NVML compute-process row")
        if row[0].strip() == gpu_uuid:
            pid = int(row[1].strip())
            _require(pid > 0, "invalid NVML process ID")
            found.add(pid)
    return found


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    raw = (json.dumps(payload, sort_keys=True, indent=2,
                      ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


class SelectedA23ContextOwner:
    """Register one worker context and seal each assigned source-group boundary."""

    def __init__(
        self, request: Mapping[str, Any], run_dir: Path, device: str,
        slot: int, task_ids: Sequence[int], torch_module: Optional[Any] = None,
    ) -> None:
        expected_keys = {
            "schema_version", "protocol", "wave_sha256", "coordinator_pid",
            "coordinator_start_ticks", "devices", "physical_gpu_uuids",
            "assignment_sha256",
        }
        _require(isinstance(request, Mapping) and set(request) == expected_keys
                 and request["schema_version"] == 1
                 and request["protocol"] in PROTOCOLS.values()
                 and isinstance(request["wave_sha256"], str)
                 and len(request["wave_sha256"]) == 64,
                 "formal owner request is incomplete")
        devices = request["devices"]
        uuids = request["physical_gpu_uuids"]
        _require(isinstance(devices, list) and isinstance(uuids, list)
                 and len(devices) == len(uuids) in (1, 2)
                 and len(set(devices)) == len(devices)
                 and len(set(uuids)) == len(uuids)
                 and type(slot) is int and 0 <= slot < len(devices)
                 and devices[slot] == device
                 and list(selected_physical_gpu_uuids([device])) == [uuids[slot]],
                 "assigned logical and physical GPU identities differ")
        _require(isinstance(task_ids, list) and task_ids
                 and all(type(value) is int and value > 0 for value in task_ids)
                 and len(task_ids) == len(set(task_ids)),
                 "assigned source-group indices are incomplete")
        root = Path(__file__).resolve().parents[3]
        run_dir = Path(run_dir)
        _require(root.name == "LGP-a23-formal-promotion-20260929-01"
                 and run_dir.is_absolute() and run_dir.is_dir()
                 and not run_dir.is_symlink()
                 and run_dir.parent.resolve() ==
                     root / "outputs" / "experiments" / request["protocol"],
                 "worker root is not an isolated formal producer")
        plan_path = run_dir / "plan.json"
        _require(plan_path.is_file() and not plan_path.is_symlink(),
                 "formal plan is absent")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        _require(plan.get("formal_execution") is True
                 and plan.get("diagnostic_only") is False
                 and plan.get("formal_preflight", {}).get("wave_sha256") ==
                     request["wave_sha256"]
                 and file_digest(run_dir / "execution_assignments.json") ==
                     request["assignment_sha256"],
                 "worker plan or assignment differs from admitted wave")
        assignment = json.loads(
            (run_dir / "execution_assignments.json").read_text(encoding="utf-8")
        )
        groups = assignment.get("attack_groups")
        _require(assignment.get("schema_version") == 1
                 and assignment.get("devices") == devices
                 and assignment.get("worker_count") == len(devices)
                 and assignment.get("clean") == []
                 and isinstance(groups, list)
                 and all(type(group.get("worker_slot")) is int
                         and 0 <= group["worker_slot"] < len(devices)
                         and group.get("device") == devices[group["worker_slot"]]
                         for group in groups)
                 and [group["canonical_index"] for group in groups
                      if group["worker_slot"] == slot] == list(task_ids),
                 "worker source groups differ from the assignment")
        coordinator_pid = request["coordinator_pid"]
        _require(type(coordinator_pid) is int and coordinator_pid > 0
                 and type(request["coordinator_start_ticks"]) is int
                 and request["coordinator_start_ticks"] > 0,
                 "coordinator identity is malformed")
        self.coordinator = process_identity(coordinator_pid)
        self.worker = process_identity(os.getpid())
        _require(self.coordinator["start_ticks"] ==
                     request["coordinator_start_ticks"]
                 and self.worker["parent_pid"] == coordinator_pid,
                 "worker parent or coordinator start ticks differ")
        self.request = dict(request)
        self.request_sha256 = _digest(self.request)
        self.run_dir = run_dir.resolve()
        self.directory = self.run_dir / "runtime_ownership"
        _require(not self.directory.is_symlink(), "linked owner directory")
        self.directory.mkdir(exist_ok=True)
        self.device = device
        self.slot = slot
        self.gpu_uuid = uuids[slot]
        self.task_ids = list(task_ids)
        self.torch = torch_module
        self.nvml_pid = None
        self._token = None
        self.cursor = 0
        self.active = None

    def register(self) -> Mapping[str, Any]:
        _require(self.nvml_pid is None, "context is already registered")
        before = compute_pids(self.gpu_uuid)
        _require(not before, "selected GPU was not idle before registration")
        torch = self.torch
        if torch is None:
            import torch as imported_torch
            torch = imported_torch
        _require(not torch.cuda.is_initialized(),
                 "worker initialized CUDA before context registration")
        torch.cuda.set_device(int(self.device.split(":", 1)[1]))
        self._token = torch.empty(1, device=self.device)
        torch.cuda.synchronize(int(self.device.split(":", 1)[1]))
        observed = compute_pids(self.gpu_uuid)
        _require(len(observed) == 1, "ambiguous or missing registered NVML context")
        self.nvml_pid = next(iter(observed))
        _require(self.nvml_pid == self.worker["pid"],
                 "NVML context PID does not identify this worker")
        self.verify()
        receipt = {
            "schema_version": 1,
            "status": "context_registered_pending_runtime_audit",
            "request_sha256": self.request_sha256,
            "protocol": self.request["protocol"],
            "wave_sha256": self.request["wave_sha256"],
            "assignment_sha256": self.request["assignment_sha256"],
            "plan_sha256": file_digest(self.run_dir / "plan.json"),
            "device": self.device, "gpu_uuid": self.gpu_uuid, "slot": self.slot,
            "coordinator": self.coordinator, "worker": self.worker,
            "nvml_pid": self.nvml_pid,
            "before_context_pids": [],
            "registered_context_pids": sorted(observed),
            "task_ids": self.task_ids,
            "registered_at_utc": _now(),
            "assumption": "NVML_enumerates_all_active_compute_contexts_on_selected_GPU",
            "runtime_gpu_ownership_accepted": False,
        }
        _write_new(self.directory / "worker-{}-registration.json".format(self.slot),
                   receipt)
        return receipt

    def verify(self) -> None:
        _require(self.nvml_pid is not None, "context has not been registered")
        _require(process_identity(self.worker["pid"]) == self.worker
                 and process_identity(self.coordinator["pid"]) == self.coordinator
                 and list(selected_physical_gpu_uuids([self.device])) ==
                     [self.gpu_uuid]
                 and compute_pids(self.gpu_uuid) == {self.nvml_pid},
                 "worker lineage, physical GPU or compute context changed")

    def start_task(self, task_id: int) -> None:
        self.verify()
        _require(self.active is None and self.cursor < len(self.task_ids)
                 and self.task_ids[self.cursor] == task_id,
                 "source group is duplicate or out of assignment order")
        self.active = task_id
        _write_new(self.directory / "worker-{}-task-{}-start.json".format(
            self.slot, task_id), {
                "task_id": task_id, "worker": self.worker,
                "nvml_pid": self.nvml_pid, "at_utc": _now(),
            })

    def finish_task(self, task_id: int) -> None:
        self.verify()
        _require(self.active == task_id, "source group lacks a matching start")
        _write_new(self.directory / "worker-{}-task-{}-finish.json".format(
            self.slot, task_id), {
                "task_id": task_id, "worker": self.worker,
                "nvml_pid": self.nvml_pid, "at_utc": _now(),
            })
        self.cursor += 1
        self.active = None

    def close(self) -> None:
        self.verify()
        _require(self.active is None and self.cursor == len(self.task_ids),
                 "worker did not complete every assigned source group")
        _write_new(self.directory / "worker-{}-terminal.json".format(self.slot), {
            "status": "worker_complete_pending_runtime_audit",
            "worker": self.worker, "nvml_pid": self.nvml_pid,
            "task_ids": self.task_ids, "completed": self.cursor,
            "at_utc": _now(), "runtime_gpu_ownership_accepted": False,
        })

    def record_failure(self, exc: BaseException) -> None:
        path = self.directory / "worker-{}-failure.json".format(self.slot)
        if not path.exists():
            _write_new(path, {
                "status": "worker_failed_preserved",
                "worker": self.worker, "active_task": self.active,
                "completed": self.cursor, "error_type": type(exc).__name__,
                "at_utc": _now(), "runtime_gpu_ownership_accepted": False,
            })


class SelectedA23RuntimeMonitor:
    """Coordinator-side NVML samples, independent of worker self-reports."""

    MAX_SAMPLE_GAP_NS = 15_000_000_000

    def __init__(
        self, request: Mapping[str, Any], run_dir: Path,
        worker_pids: Sequence[int], active_slots: Sequence[int],
    ) -> None:
        uuids = request.get("physical_gpu_uuids")
        _require(isinstance(uuids, list) and len(uuids) in (1, 2)
                 and len(set(uuids)) == len(uuids)
                 and len(worker_pids) == len(uuids)
                 and all(type(pid) is int and pid > 0 for pid in worker_pids)
                 and len(set(worker_pids)) == len(worker_pids),
                 "coordinator GPU and worker identities are incomplete")
        _require(active_slots and len(set(active_slots)) == len(active_slots)
                 and all(type(slot) is int and 0 <= slot < len(uuids)
                         for slot in active_slots),
                 "coordinator active source lanes are incomplete")
        self.uuids = list(uuids)
        coordinator_pid = request.get("coordinator_pid")
        coordinator_ticks = request.get("coordinator_start_ticks")
        _require(type(coordinator_pid) is int and coordinator_pid > 0
                 and type(coordinator_ticks) is int and coordinator_ticks > 0,
                 "coordinator identity is incomplete")
        self.coordinator = process_identity(coordinator_pid)
        _require(self.coordinator["start_ticks"] == coordinator_ticks,
                 "coordinator start ticks changed")
        self.worker_pids = list(worker_pids)
        self.active_slots = set(active_slots)
        self.request_sha256 = _digest(request)
        self.directory = Path(run_dir) / "runtime_ownership"
        _require(not self.directory.is_symlink(), "linked monitor directory")
        self.directory.mkdir(exist_ok=True)
        self.path = self.directory / "coordinator-samples.jsonl"
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self.handle = os.fdopen(fd, "w", encoding="utf-8")
        self.count = 0
        self.seen = set()
        self.worker_start_ticks = {}
        self.previous_sample_ns = None
        self.previous_gpu_query_end_ns = {}
        self.max_observed_gap_ns = 0
        self.max_gpu_observation_gap_ns = {slot: 0 for slot in range(len(uuids))}

    def sample(
        self, workers: Sequence[Mapping[str, Any]], phase: str, *,
        task_slot: Optional[int] = None, task_id: Optional[int] = None,
    ) -> None:
        _require(not self.handle.closed and len(workers) == len(self.uuids),
                 "monitor is closed or worker panel changed")
        _require(phase in {"workers_started", "interval", "task_started",
                           "stage_complete"}, "unknown owner sample phase")
        if phase == "task_started":
            _require(
                type(task_slot) is int and task_slot in self.active_slots
                and type(task_id) is int and task_id > 0
                and workers[task_slot].get("current") is not None,
                "source-start sample lacks its assigned group identity",
            )
        else:
            _require(task_slot is None and task_id is None,
                     "non-source sample carries a group identity")
        sampled_at_ns = time.monotonic_ns()
        sampled_at_utc = _now()
        elapsed_ns = (None if self.previous_sample_ns is None
                      else sampled_at_ns - self.previous_sample_ns)
        _require(elapsed_ns is None or elapsed_ns > 0,
                 "monotonic sample clock changed")
        gap_violation = elapsed_ns is not None and elapsed_ns > self.MAX_SAMPLE_GAP_NS
        if elapsed_ns is not None:
            self.max_observed_gap_ns = max(self.max_observed_gap_ns, elapsed_ns)
        readings = []
        violations = []
        missing_active = []
        for slot, gpu_uuid in enumerate(self.uuids):
            worker = workers[slot]
            _require(worker.get("slot") == slot
                     and worker.get("pid") == self.worker_pids[slot],
                     "monitored worker identity changed")
            query_started_ns = time.monotonic_ns()
            query_started_utc = _now()
            observed = compute_pids(gpu_uuid)
            query_finished_ns = time.monotonic_ns()
            query_finished_utc = _now()
            prior_gpu_query_end_ns = self.previous_gpu_query_end_ns.get(slot)
            gpu_observation_gap_ns = (
                None if prior_gpu_query_end_ns is None
                else query_finished_ns - prior_gpu_query_end_ns
            )
            if gpu_observation_gap_ns is not None:
                self.max_gpu_observation_gap_ns[slot] = max(
                    self.max_gpu_observation_gap_ns[slot], gpu_observation_gap_ns
                )
                if not 0 < gpu_observation_gap_ns <= self.MAX_SAMPLE_GAP_NS:
                    violations.append(slot)
            self.previous_gpu_query_end_ns[slot] = query_finished_ns
            expected = {self.worker_pids[slot]} if slot in self.active_slots else set()
            active = slot in self.active_slots and worker.get("status") == "running"
            identity = None
            if slot in self.active_slots:
                try:
                    identity = process_identity(self.worker_pids[slot])
                except FileNotFoundError:
                    pass
                if identity is None:
                    if phase == "task_started" and slot == task_slot:
                        violations.append(slot)
                else:
                    prior_ticks = self.worker_start_ticks.get(slot)
                    if (identity["parent_pid"] != self.coordinator["pid"]
                            or (prior_ticks is not None
                                and identity["start_ticks"] != prior_ticks)):
                        violations.append(slot)
                    self.worker_start_ticks.setdefault(slot, identity["start_ticks"])
            if observed == expected and active:
                self.seen.add(slot)
            missing = active and worker.get("current") is not None and observed != expected
            if missing:
                missing_active.append(slot)
            if (not observed.issubset(expected)
                    or (phase == "task_started" and missing)
                    or (phase == "stage_complete" and observed)):
                violations.append(slot)
            readings.append({
                "slot": slot, "gpu_uuid": gpu_uuid,
                "worker_pid": self.worker_pids[slot],
                "query_started_at_utc": query_started_utc,
                "query_finished_at_utc": query_finished_utc,
                "query_started_monotonic_ns": query_started_ns,
                "query_finished_monotonic_ns": query_finished_ns,
                "elapsed_since_prior_gpu_observation_ns": gpu_observation_gap_ns,
                "worker_status": worker.get("status"),
                "worker_has_current_group": worker.get("current") is not None,
                "worker_start_ticks": (
                    None if identity is None else identity["start_ticks"]),
                "worker_parent_pid": (
                    None if identity is None else identity["parent_pid"]),
                "observed_compute_pids": sorted(observed),
            })
        violations = sorted(set(violations))
        row = {
            "schema_version": 1, "sequence": self.count + 1,
            "request_sha256": self.request_sha256,
            "phase": phase, "at_utc": sampled_at_utc,
            "task_slot": task_slot, "task_id": task_id,
            "at_monotonic_ns": sampled_at_ns,
            "elapsed_since_prior_ns": elapsed_ns,
            "max_allowed_gap_ns": self.MAX_SAMPLE_GAP_NS,
            "gap_violation": gap_violation,
            "readings": readings, "violation_slots": violations,
            "missing_active_slots": missing_active,
            "runtime_gpu_ownership_accepted": False,
        }
        self.handle.write(json.dumps(
            row, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.count += 1
        self.previous_sample_ns = sampled_at_ns
        _require(not violations and not gap_violation,
                 "foreign owner, missing start context, terminal GPU context "
                 "or excessive GPU observation gap")

    def require_seen(self) -> None:
        _require(self.seen == self.active_slots
                 and set(self.worker_start_ticks) == self.active_slots,
                 "not every active source lane was observed with its own context")

    def finish(self, successful: bool) -> None:
        _require(not self.handle.closed, "monitor already sealed")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        complete = (successful and self.seen == self.active_slots
                    and set(self.worker_start_ticks) == self.active_slots)
        _write_new(self.directory / "coordinator-summary.json", {
            "schema_version": 1,
            "status": ("complete_pending_independent_audit" if complete
                       else "failed_or_incomplete_preserved"),
            "request_sha256": self.request_sha256,
            "coordinator": self.coordinator,
            "worker_pids": self.worker_pids,
            "worker_start_ticks": [
                self.worker_start_ticks.get(slot) for slot in range(len(self.uuids))
            ],
            "max_allowed_gap_ns": self.MAX_SAMPLE_GAP_NS,
            "max_observed_gap_ns": self.max_observed_gap_ns,
            "max_gpu_observation_gap_ns": [
                self.max_gpu_observation_gap_ns[slot]
                for slot in range(len(self.uuids))
            ],
            "active_slots": sorted(self.active_slots),
            "observed_active_slots": sorted(self.seen),
            "sample_count": self.count,
            "samples_sha256": file_digest(self.path),
            "sealed_at_utc": _now(),
            "runtime_gpu_ownership_accepted": False,
        })
