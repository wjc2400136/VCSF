"""Owned Linux process groups for independent complete training branches."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import signal
import sys
import time


def process_identity(pid):
    values = (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()
    return {
        "pid": int(pid), "parent_pid": int(values[1]), "process_group": int(values[2]),
        "session_id": int(values[3]), "state": values[0], "start_ticks": int(values[19]),
    }


def rename_directory_no_replace(source, destination):
    if sys.platform != "linux":
        raise RuntimeError("Atomic directory publication requires the Linux server")
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, "renameat2", None)
    if rename is None:
        raise RuntimeError("Atomic no-replace directory publication is unavailable")
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def install_branch_scope(coordinator):
    if sys.platform != "linux":
        raise RuntimeError("Training branch containment requires the Linux server")
    own = process_identity(os.getpid())
    if own["process_group"] != own["pid"] or own["session_id"] != own["pid"]:
        raise RuntimeError("A training branch must own an independent process group")
    owner_pid = own["pid"]

    def terminate_scope(signum, frame):
        if os.getpid() == owner_pid:
            os.killpg(owner_pid, signal.SIGKILL)
        else:
            # Forked dataloader children must not terminate their whole parent group.
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate_scope)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise RuntimeError("Cannot install training parent-death guard")
    try:
        parent = process_identity(int(coordinator["pid"]))
        valid = (os.getppid() == coordinator["pid"]
                 and parent["start_ticks"] == coordinator["start_ticks"]
                 and parent["state"] != "Z")
    except (FileNotFoundError, ProcessLookupError):
        valid = False
    if not valid:
        terminate_scope(signal.SIGTERM, None)
        raise RuntimeError("Training coordinator disappeared")
    return {
        "worker_pid": own["pid"], "worker_start_ticks": own["start_ticks"],
        "process_group": own["process_group"],
        "coordinator_pid": coordinator["pid"],
        "coordinator_start_ticks": coordinator["start_ticks"],
    }


def peek_exit_code(process):
    # WNOWAIT keeps the owned leader as a PID/PGID anchor until cleanup finishes.
    result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if result is None:
        return None
    return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status


def _owned_anchor(process, identity):
    if process.pid != identity["pid"] or process.returncode is not None:
        raise RuntimeError("Training process-group anchor was reaped or changed")
    current = process_identity(process.pid)
    for field in ("pid", "parent_pid", "start_ticks", "process_group", "session_id"):
        if current[field] != identity[field]:
            raise RuntimeError("Training process ownership changed: " + field)
    if current["process_group"] != current["session_id"] or current["session_id"] != process.pid:
        raise RuntimeError("Training worker no longer owns its process group")


def _members(identity):
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            member = process_identity(int(path.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
        if member["process_group"] == member["session_id"] == identity["pid"] and member["state"] != "Z":
            if member["start_ticks"] < identity["start_ticks"]:
                raise RuntimeError("Training process-group member predates its owner")
            members.append(member)
    return members


def close_branch_process(process, identity, grace_seconds=5.0):
    if not 0 <= grace_seconds <= 30:
        raise ValueError("Training cleanup grace must be between zero and thirty seconds")
    if identity["start_ticks"] is None:
        # Popen's unreaped child still pins the PID if launch registration failed.
        current = process_identity(process.pid)
        if process.returncode is not None or any(current[key] != identity[key]
                for key in ("pid", "parent_pid", "process_group", "session_id")):
            raise RuntimeError("Unregistered training process no longer belongs to its launcher")
        identity = current
    _owned_anchor(process, identity)
    members = _members(identity)
    if members:
        os.killpg(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_seconds
        while _members(identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _members(identity):
            _owned_anchor(process, identity)
            os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + 10
    while _members(identity) and time.monotonic() < deadline:
        time.sleep(0.05)
    if _members(identity):
        raise RuntimeError("Training descendants survived owned process-group cleanup")
    process.wait(timeout=10)
    return {"status": "verified", "process_group": identity["pid"],
            "observed_live_members": len(members), "exit_code": process.returncode}
