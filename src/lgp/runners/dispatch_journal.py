"""Durable process/task registration for generic Linux CUDA workers."""
from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from .training_pair_processes import process_identity
from .vcsf_gpu_context_owner import _compute_pids


def process_birth(pid):
    identity = process_identity(pid)
    return {key: identity[key] for key in (
        "pid", "parent_pid", "start_ticks", "process_group", "session_id")}


class AppendOnlyJournal:
    def __init__(self, path, header):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = 0
        self._write("x", dict(event="registration", **header))

    def _write(self, mode, payload):
        row = dict(payload, sequence=self.sequence,
                   utc=datetime.now(timezone.utc).isoformat())
        encoded = json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        with self.path.open(mode, encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1

    def append(self, event, **payload):
        self._write("a", dict(event=event, **payload))


def logical_gpu_uuid(ordinal):
    # The CUDA driver resolves the logical ordinal, including visibility masks.
    driver = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    get_device = driver.cuDeviceGet
    get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    get_device.restype = ctypes.c_int
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
    get_uuid.restype = ctypes.c_int
    value = (ctypes.c_ubyte * 16)()
    if get_device(ctypes.byref(device), ordinal) != 0 or get_uuid(value, device) != 0:
        raise RuntimeError("Cannot bind the worker logical CUDA device to its UUID")
    return "GPU-" + str(uuid.UUID(bytes=bytes(value)))


class WorkerJournal:
    def __init__(self, run_dir, stage, slot, device, tasks, coordinator_pid, torch):
        self.worker = process_birth(os.getpid())
        self.coordinator = process_birth(coordinator_pid)
        if self.worker["parent_pid"] != coordinator_pid:
            raise RuntimeError("Dispatch worker no longer belongs to its coordinator")
        self.stage, self.slot, self.device = stage, slot, device
        self.tasks = {int(task["canonical_index"]) for task in tasks}
        self.torch = torch
        self.context = None
        self.journal = AppendOnlyJournal(
            Path(run_dir) / "dispatch_journal" / stage / ("worker-{}.jsonl".format(slot)),
            dict(schema="generic_dispatch_worker_v1", stage=stage, slot=slot,
                 device=device, worker=self.worker, coordinator=self.coordinator,
                 assigned_tasks=list(tasks), gpu_context_registration=None),
        )

    def record(self, event, task_index):
        if (process_birth(os.getpid()) != self.worker
                or process_birth(self.coordinator["pid"]) != self.coordinator):
            raise RuntimeError("Dispatch process birth identity changed")
        if task_index is not None and task_index not in self.tasks:
            raise RuntimeError("Dispatch journal received an unassigned task")
        kind = event["type"]
        if kind != "fatal" and self.tasks and self.torch.cuda.is_initialized():
            ordinal = int(self.device.split(":", 1)[1])
            if self.torch.cuda.current_device() != ordinal:
                raise RuntimeError("Dispatch worker changed its logical CUDA device")
            if self.context is None:
                gpu_uuid = logical_gpu_uuid(ordinal)
                observed = _compute_pids(gpu_uuid)
                if self.worker["pid"] in observed:
                    self.context = dict(device=self.device, gpu_uuid=gpu_uuid,
                        worker_pid=self.worker["pid"], worker_start_ticks=self.worker["start_ticks"],
                        nvml_pid=self.worker["pid"], observed_context_pids=sorted(observed),
                        exclusive_context_claim=False, new_context_allocation=False)
                    self.journal.append("gpu_context_registered", registration=self.context)
                else:
                    # set_device can initialize CUDA before NVML exposes a compute context.
                    self.journal.append("gpu_context_pending", device=self.device,
                        gpu_uuid=gpu_uuid, observed_context_pids=sorted(observed),
                        gpu_context_registration=None, reason="Own NVML PID not yet observed")
        if kind == "worker_done" and self.tasks and self.context is None:
            raise RuntimeError("Completed dispatch worker has no GPU context registration")
        self.journal.append("worker_event", event_type=kind, task_index=task_index,
            current=event.get("current"), reason=event.get("reason"),
            gpu_context_registration=self.context)
        return dict(worker_start_ticks=self.worker["start_ticks"],
                    gpu_context_registration=self.context)
