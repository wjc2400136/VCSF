"""One or two exclusive GPU lanes for structure checks, never AP execution."""
from __future__ import annotations

import csv
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import traceback

from ..io import atomic_json


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def balanced_lanes(groups, sources, devices):
    """Alternate source-to-device assignment between complete variant/seed pairs."""
    require(len(sources) == 2 and len(set(sources)) == 2
        and 1 <= len(devices) <= 2 and len(set(devices)) == len(devices),
        "The structure schedule requires two sources and one or two distinct GPU devices")
    pairs = {}
    for group in groups:
        pair = (group["variant"], group["seed"])
        by_source = pairs.setdefault(pair, {})
        require(group["source"] in sources and group["source"] not in by_source,
            "Unknown or duplicate source in a structure pair")
        by_source[group["source"]] = group
    lanes = [[] for _ in devices]
    for pair_index, by_source in enumerate(pairs.values()):
        require(set(by_source) == set(sources), "Incomplete structure source pair")
        for source_index, source in enumerate(sources):
            lanes[(pair_index + source_index) % len(devices)].append(by_source[source])
    ids = [group["group_id"] for lane in lanes for group in lane]
    require(len(ids) == len(set(ids)) == len(groups), "Structure assignment is not one-to-one")
    return lanes


def physical_gpus(devices):
    rows = csv.reader(subprocess.check_output(["nvidia-smi",
        "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True).splitlines())
    available = {"cuda:" + row[0].strip(): row[1].strip() for row in rows if row}
    require(all(device in available for device in devices), "Requested physical GPU is missing")
    uuids = [available[device] for device in devices]
    require(len(set(uuids)) == len(devices), "GPU devices resolve to a duplicate UUID")
    return uuids


def verify_device_owner(gpu_uuid, own_pid):
    rows = csv.reader(subprocess.check_output(["nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True).splitlines())
    foreign = [int(row[1].strip()) for row in rows
        if row and row[0].strip() == gpu_uuid and int(row[1].strip()) != own_pid]
    require(not foreign, "Foreign compute process on the assigned GPU: {}".format(foreign))


def worker_environment(gpu_uuid):
    env = dict(os.environ)
    # UUID visibility is set before Python starts, so logical cuda:0 is unambiguous.
    env["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    return env


def verify_selected_devices(devices):
    for gpu_uuid in physical_gpus(devices):
        verify_device_owner(gpu_uuid, os.getpid())


def device_lock_path(gpu_uuid, lock_root=None):
    root = Path(lock_root or "/tmp/lgp-vcsf-structure-gpus")
    require(not root.is_symlink(), "GPU reservation directory cannot be a symlink")
    return root / (hashlib.sha256(gpu_uuid.encode("utf-8")).hexdigest() + ".lock")


@contextmanager
def gpu_reservations(uuids, lock_root=None):
    """Coordinate our runners across worktrees; foreign GPU users are still checked."""
    import fcntl

    require(sys.platform == "linux", "GPU reservations require the server platform")
    handles = {}
    try:
        for gpu_uuid in sorted(uuids):
            path = device_lock_path(gpu_uuid, lock_root)
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            handles[gpu_uuid] = fd
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Selected GPU is reserved by another structure run") from exc
        yield handles
    finally:
        # Never unlink or explicitly unlock: an inherited child FD keeps ownership
        # until that child exits, even if the coordinator is killed.
        for fd in handles.values():
            os.close(fd)


def verify_coordinator(expected_pid):
    require(type(expected_pid) is int and expected_pid > 1 and os.getppid() == expected_pid,
        "Structure coordinator disappeared or changed")


def install_parent_death_guard(expected_pid):
    require(sys.platform == "linux", "Parent-death containment requires the server platform")
    verify_coordinator(expected_pid)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
    libc.prctl.restype = ctypes.c_int
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0,
        "Cannot install parent-death containment: {}".format(ctypes.get_errno()))
    verify_coordinator(expected_pid)


def verify_reservation(request):
    import fcntl

    fd = request["reservation_fd"]
    require(type(fd) is int and fd >= 0, "Missing inherited GPU reservation")
    path = device_lock_path(request["gpu_uuid"])
    held, current = os.fstat(fd), os.lstat(path)
    require(stat.S_ISREG(held.st_mode) and stat.S_ISREG(current.st_mode)
        and (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)
        and [held.st_dev, held.st_ino] == request["reservation_identity"],
        "Inherited GPU reservation identity changed")
    probe = os.open(str(path), os.O_RDWR | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise RuntimeError("GPU reservation is not locked")
    finally:
        os.close(probe)


def worker_binding(request, request_sha256, worker_pid):
    return {"worker_pid": worker_pid, "coordinator_pid": request["coordinator_pid"],
        "worker_slot": request["worker_slot"], "physical_device": request["physical_device"],
        "gpu_uuid": request["gpu_uuid"], "logical_device": "cuda:0",
        "request_sha256": request_sha256, "formal_AP_eligible": False}


def require_binding(record, expected, message):
    require(all(key in record and type(record[key]) is type(value) and record[key] == value
        for key, value in expected.items()), message)


def run_worker(request_path, expected_sha256):
    from ..attacks.vcsf_research_isolation import _read_bound_json

    request_path = Path(request_path).resolve()
    request = _read_bound_json(request_path, expected_sha256)
    install_parent_death_guard(request["coordinator_pid"])
    verify_reservation(request)
    from ..registry import Registry
    from .vcsf_layered_preflight import structural_groups, verify_frozen_worktree, verify_group
    from .vcsf_research_plan import PROTOCOL, canonical_hash, compile_plan
    from ..attacks.vcsf_research_isolation import ALIAS, STUDY, validate_cost_precondition
    from .attack import run_attack

    registry = Registry()
    root = registry.root.resolve()
    request_path.relative_to(root / "outputs/diagnostics")
    require(request["kind"] == "structure_only" and request["AP_evaluations"] == 0,
        "A structure worker cannot run an efficacy job")
    verify_coordinator(request["coordinator_pid"])
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == request["gpu_uuid"],
        "Worker GPU visibility must match its assigned physical UUID")
    plan = compile_plan(registry)
    protocol = registry.protocols[PROTOCOL]
    devices = request["devices"]
    require(all(device in protocol["structure_devices"] for device in devices),
        "Unregistered structure GPU device")
    lanes = balanced_lanes(structural_groups(plan), protocol["sources"], devices)
    slot = request["worker_slot"]
    require(type(slot) is int and 0 <= slot < len(lanes), "Unknown worker slot")
    require(request["physical_device"] == devices[slot]
        and request["gpu_uuid"] == physical_gpus(devices)[slot],
        "Physical GPU assignment changed")
    require(request["group_ids"] == [g["group_id"] for g in lanes[slot]],
        "Worker assignment differs from the counterbalanced plan")
    descriptor = request["descriptor"]
    plan_path = (root / descriptor["plan"]).resolve()
    plan_path.relative_to(root)
    require(canonical_hash(_read_bound_json(plan_path, descriptor["plan_sha256"])) == canonical_hash(plan),
        "Worker plan drift")
    validate_cost_precondition(registry, descriptor["cost_acceptance"],
        descriptor["cost_acceptance_sha256"])
    verify_frozen_worktree(root, plan, request["git_head"])
    output = request_path.parent
    group_root = (root / request["group_root"]).resolve()
    require(group_root == output.parent.parent / "groups", "Worker payload root mismatch")
    binding = worker_binding(request, expected_sha256, os.getpid())
    state = {**binding, "status": "running", "pid": os.getpid(),
        "completed_groups": 0, "current": None,
        "expected_groups": len(lanes[slot]), "failed_records": 0, "formal_AP_eligible": False}
    verified = []
    atomic_json(output / "execution_state.json", state)
    try:
        for group in lanes[slot]:
            verify_coordinator(request["coordinator_pid"])
            verify_device_owner(request["gpu_uuid"], os.getpid())
            state["current"] = {key: group[key] for key in ("group_id", "source", "variant", "seed")}
            state["updated_at"] = datetime.now(timezone.utc).isoformat()
            atomic_json(output / "execution_state.json", state)
            run_dir = run_attack(registry, "coco", group["source"], ALIAS,
                output_dir=group_root / str(group["group_id"]), max_images=1,
                seed=group["seed"], device="cuda:0", strict=True,
                parameter_overrides=group["parameters"], budget_profile=protocol["budget_profile"],
                run_metadata={"protocol": PROTOCOL, "study": STUDY, "variant": group["variant"]},
                isolated_research=descriptor)
            result = verify_group(run_dir, group, request["image_id"],
                request["checkpoint_sha256"][group["source"]], Path(request["clean_path"]),
                request["clean_sha256"], descriptor)
            verify_coordinator(request["coordinator_pid"])
            verify_device_owner(request["gpu_uuid"], os.getpid())
            result.update(binding)
            verified.append(result)
            atomic_json(output / "verified_groups.json", verified)
            state.update(completed_groups=len(verified), current=None)
            atomic_json(output / "execution_state.json", state)
            print("[STRUCTURE] lane={} {}/{} group={}".format(slot, len(verified),
                len(lanes[slot]), group["group_id"]), flush=True)
            gc.collect()
        verify_frozen_worktree(root, plan, request["git_head"])
        _read_bound_json(request_path, expected_sha256)
        validate_cost_precondition(registry, descriptor["cost_acceptance"],
            descriptor["cost_acceptance_sha256"])
        verify_coordinator(request["coordinator_pid"])
        state.update(status="complete", current=None)
        atomic_json(output / "execution_state.json", state)
        return 0
    except Exception as exc:
        state.update(status="failed", failed_records=1,
            reason="{}: {}".format(type(exc).__name__, exc))
        atomic_json(output / "execution_state.json", state)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run_structure_workers(registry, plan, groups, output, descriptor, input_evidence,
    git_head, state, devices):
    from .vcsf_research_plan import PROTOCOL

    root = registry.root.resolve()
    protocol = registry.protocols[PROTOCOL]
    require(all(device in protocol["structure_devices"] for device in devices),
        "Unregistered structure GPU device")
    lanes = balanced_lanes(groups, protocol["sources"], devices)
    uuids = physical_gpus(devices)
    with gpu_reservations(uuids) as reservations:
        return _run_reserved_workers(root, groups, output, descriptor, input_evidence,
            git_head, state, devices, lanes, uuids, reservations)


def _run_reserved_workers(root, groups, output, descriptor, input_evidence,
    git_head, state, devices, lanes, uuids, reservations):
    from ..attacks.vcsf_research_isolation import _read_bound_json, _write_bound_json

    for gpu_uuid in uuids:
        verify_device_owner(gpu_uuid, os.getpid())
    entries, processes, handles = [], [], []
    started = time.perf_counter()
    try:
        for slot, lane in enumerate(lanes):
            directory = output / "workers" / str(slot)
            directory.mkdir(parents=True, exist_ok=False)
            reserved = os.fstat(reservations[uuids[slot]])
            request = {**input_evidence, "kind": "structure_only", "AP_evaluations": 0,
                "coordinator_pid": os.getpid(), "worker_slot": slot, "physical_device": devices[slot],
                "devices": list(devices),
                "gpu_uuid": uuids[slot], "group_ids": [g["group_id"] for g in lane],
                "descriptor": descriptor, "git_head": git_head,
                "group_root": (output / "groups").relative_to(root).as_posix(),
                "reservation_fd": reservations[uuids[slot]],
                "reservation_identity": [reserved.st_dev, reserved.st_ino]}
            request_path = directory / "request.json"
            request_sha256 = _write_bound_json(request_path, request)
            stdout = (directory / "stdout.log").open("w", encoding="utf-8")
            stderr = (directory / "stderr.log").open("w", encoding="utf-8")
            handles.extend([stdout, stderr])
            process = subprocess.Popen([sys.executable, "-u",
                "experiments/vcsf_layered_structure_preflight.py", "--worker-request", str(request_path),
                "--worker-request-sha256", request_sha256], cwd=root,
                env=worker_environment(uuids[slot]), stdout=stdout, stderr=stderr,
                pass_fds=(reservations[uuids[slot]],))
            processes.append(process)
            entries.append({"slot": slot, "pid": process.pid, "device": devices[slot],
                "gpu_uuid": uuids[slot], "group_ids": request["group_ids"],
                "request_sha256": request_sha256, "directory": directory.relative_to(root).as_posix()})
        assignment_sha256 = _write_bound_json(output / "execution_assignment.json", {
            "algorithm": "alternating_source_to_gpu_by_configuration_seed_pair",
            "workers": entries, "group_atomicity": True, "AP_evaluations": 0})
        while True:
            exits = [process.poll() for process in processes]
            snapshots = []
            for entry in entries:
                path = root / entry["directory"] / "execution_state.json"
                snapshot = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {
                    "status": "starting", "completed_groups": 0, "current": None}
                snapshots.append(dict(entry, state=snapshot, exit_code=exits[entry["slot"]]))
            state.update(phase="structure_worker_generation", workers=snapshots,
                completed_groups=sum(item["state"]["completed_groups"] for item in snapshots),
                current=[item["state"].get("current") for item in snapshots
                    if item["state"].get("current") is not None])
            atomic_json(output / "execution_state.json", state)
            require(all(code in (None, 0) for code in exits),
                "A structure worker failed; preserve both lanes and inspect worker logs")
            if all(code == 0 for code in exits):
                break
            time.sleep(1)
        _read_bound_json(output / "execution_assignment.json", assignment_sha256)
        results = []
        for entry in entries:
            directory = root / entry["directory"]
            request = _read_bound_json(directory / "request.json", entry["request_sha256"])
            binding = worker_binding(request, entry["request_sha256"], entry["pid"])
            require_binding(binding, {"worker_slot": entry["slot"], "physical_device": entry["device"],
                "gpu_uuid": entry["gpu_uuid"], "coordinator_pid": os.getpid()},
                "Worker request and assignment disagree")
            worker_state = json.loads((directory / "execution_state.json").read_text(encoding="utf-8"))
            rows = json.loads((directory / "verified_groups.json").read_text(encoding="utf-8"))
            require_binding(worker_state, {**binding, "expected_groups": len(entry["group_ids"]),
                "current": None}, "Worker state provenance mismatch")
            require(worker_state["status"] == "complete" and worker_state["failed_records"] == 0
                and worker_state["pid"] == entry["pid"]
                and worker_state["completed_groups"] == len(entry["group_ids"])
                and [r["group_id"] for r in rows] == entry["group_ids"], "Incomplete worker evidence")
            for row in rows:
                require_binding(row, binding, "Worker result provenance mismatch")
            results.extend(rows)
        require(len(results) == len(groups) and len({r["group_id"] for r in results}) == len(groups)
            and {r["group_id"] for r in results} == {g["group_id"] for g in groups},
            "The worker result matrix has missing or duplicate groups")
        ordered = {r["group_id"]: r for r in results}
        results = [ordered[g["group_id"]] for g in groups]
        _read_bound_json(output / "execution_assignment.json", assignment_sha256)
        atomic_json(output / "verified_groups.json", results)
        atomic_json(output / "execution_timing.json", {"wall_seconds": time.perf_counter() - started,
            "scope": "worker_structure_checks_not_isolated_cost_measurement", "workers": len(processes)})
        return results
    except BaseException:
        # Only child handles created by this coordinator may be stopped on a peer failure.
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        atomic_json(output / "worker_abort.json", {
            "reason": "coordinator_or_peer_failure", "partial_evidence_preserved": True,
            "workers": [{"pid": p.pid, "exit_code": p.returncode} for p in processes]})
        raise
    finally:
        for handle in handles:
            handle.close()
