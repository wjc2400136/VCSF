"""Owned budget scheduling with externally pinned formal admission."""
import os
from pathlib import Path
import subprocess
import sys
import time

from ..io import atomic_json, file_digest
from ..registry import Registry
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
)
from .vcsf_final_budget_contract import PROTOCOL, build_plan, generation_binding
from .vcsf_final_training_state import _read, source_identity
from .vcsf_oblivious_admission import bound_json, isolate_output


def read_permit(registry, path, digest):
    permit = bound_json(path, digest)
    limit = permit.get("max_images")
    if permit.get("protocol") != PROTOCOL:
        raise ValueError("Budget permit namespace differs")
    if permit.get("status") == "approved_budget_diagnostic":
        if type(limit) is not int or not 0 < limit <= 5000:
            raise ValueError("Only explicit bounded budget diagnostics are admitted")
    elif permit.get("status") == "approved_budget_formal":
        if limit is not None:
            raise ValueError("Formal budget permits require the full image population")
        from .vcsf_budget_admission import verify_formal_permit
        verify_formal_permit(registry, permit, recheck_evidence=False)
    else:
        raise ValueError("Budget permit status is not admitted")
    plan = build_plan(registry, permit["devices"])
    if permit["plan"] != plan or permit["source_identity"] != source_identity(registry.root):
        raise ValueError("Budget permit plan or source differs")
    generation_groups = {group["index"]: group for group in plan["generation_groups"]}
    groups = permit["group_indices"]
    if (not isinstance(groups, list) or not groups
            or any(type(index) is not int or index not in generation_groups
                   for index in groups)
            or groups != sorted(set(groups))):
        raise ValueError("Budget permit group identities are invalid")
    prepared_ref = permit["prepared"]
    terminal_path = Path(prepared_ref["file"]).resolve()
    terminal = bound_json(terminal_path, prepared_ref["sha256"])
    if terminal["status"] != "inputs_prepared_pending_execution_qualification":
        raise ValueError("Budget prepared inputs are incomplete")
    prepared = {}
    for source in registry.protocols[PROTOCOL]["sources"]:
        filename = source + ".json"
        prepared[source] = bound_json(terminal_path.parent / filename, terminal["files"][filename])
    bound_json(permit["assets"]["file"], permit["assets"]["sha256"])
    if set(permit["checkpoints"]) != set(registry.paper_order):
        raise ValueError("Budget target checkpoint panel is incomplete")
    checkpoint_ref = permit["checkpoint_bindings"]
    checkpoints = bound_json(checkpoint_ref["file"], checkpoint_ref["sha256"])
    if set(checkpoints) != set(registry.paper_order):
        raise ValueError("Budget checkpoint binding panel is incomplete")
    normalized = {target: dict(path=str(Path(row["path"]).resolve()), sha256=row["sha256"])
                  for target, row in checkpoints.items()}
    # Actual checkpoint bytes are verified immediately before each model call.
    if normalized != permit["checkpoints"]:
        raise ValueError("Budget checkpoint permit differs from bound registry")
    protected = [Path(row["inputs"]["payload"]) for row in prepared.values()]
    protected += [terminal_path.parent, Path(permit["assets"]["file"]).parent,
                  registry.root / "data", registry.root / "checkpoints", Path(path).parent]
    isolate_output(Path(permit["output"]).resolve(), protected)
    bindings = [generation_binding(registry, plan,
        prepared[generation_groups[index]["source"]], index, max_images=limit)
        for index in groups]
    return permit, bindings


def worker(request_path, request_hash):
    path = Path(request_path).resolve()
    request = bound_json(path, request_hash)
    owner = install_branch_scope(request["coordinator"])
    directory = path.parent
    started = time.time()
    try:
        registry = Registry(Path(request["source_root"]))
        permit, bindings = read_permit(registry, request["permit"]["file"], request["permit"]["sha256"])
        assigned = [item for item in bindings if item["device"] == request["device"]]
        if ([item["group_index"] for item in assigned] != request["group_indices"]
                or permit["gpu_uuids"][request["device"]] != request["gpu_uuid"]):
            raise ValueError("Budget worker ownership differs from its permit")
        from .vcsf_gpu_context_owner import RegisteredGpuOwner
        from .vcsf_final_budget_group import execute_group
        gpu = RegisteredGpuOwner(request)
        atomic_json(directory / "gpu_context_registration.json", gpu.receipt)
        results = []
        for binding in assigned:
            from .vcsf_budget_admission import require_group_space
            require_group_space(registry.root, permit)
            def progress(completed, total, image_id, status):
                if completed == 1 or completed % 25 == 0 or completed == total:
                    gpu.verify()
                    atomic_json(directory / "progress.json", dict(
                        group_index=binding["group_index"], source=binding["source"],
                        epsilon=binding["epsilon"], completed=completed, total=total,
                        image_id=image_id, status=status, completed_evaluations=len(results),
                        updated_at=time.time(), owner=owner))
            results.extend(execute_group(registry, path, request_hash, permit, binding, gpu, progress))
            atomic_json(directory / "results.json", results)
            if permit["status"] == "approved_budget_formal":
                from .vcsf_budget_lifecycle import accept_and_retire_group
                accept_and_retire_group(registry, permit, binding)
        gpu.verify()
        if source_identity(registry.root) != permit["source_identity"]:
            raise ValueError("Budget source changed during execution")
        atomic_json(directory / "terminal.json", dict(status="complete_pending_independent_acceptance",
            request_sha256=request_hash, owner=owner, results=results,
            formal_result_eligible=False, scientific_acceptance=False, seconds=time.time()-started))
    except BaseException as exc:
        atomic_json(directory / "terminal.json", dict(status="failed", request_sha256=request_hash,
            owner=owner, error=repr(exc), scientific_acceptance=False, seconds=time.time()-started))
        raise


def run(registry, permit_path, permit_sha256):
    permit_path = Path(permit_path).resolve()
    permit, bindings = read_permit(registry, permit_path, permit_sha256)
    if permit["max_images"] is None:
        from .vcsf_budget_admission import verify_formal_permit
        verify_formal_permit(registry, permit, recheck_evidence=True)
    from .vcsf_structure_workers import physical_gpus, gpu_reservations, worker_environment
    devices = permit["devices"]
    uuids = physical_gpus(devices)
    if dict(zip(devices, uuids)) != permit["gpu_uuids"]:
        raise ValueError("Budget registered physical devices changed")
    output = Path(permit["output"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "permit.json", dict(file=str(permit_path), sha256=permit_sha256))
    coordinator = process_identity(os.getpid())
    atomic_json(output / "coordinator.json", coordinator)
    handles = []
    started = time.time()
    try:
        with gpu_reservations(uuids) as reservations:
            try:
                for slot, (device, uuid) in enumerate(zip(devices, uuids)):
                    groups = [item["group_index"] for item in bindings if item["device"] == device]
                    if not groups:
                        continue
                    directory = output / "workers" / str(slot)
                    directory.mkdir(parents=True, exist_ok=False)
                    fd = reservations[uuid]
                    stat = os.fstat(fd)
                    request = dict(source_root=str(registry.root.resolve()), device=device,
                        group_indices=groups, coordinator=coordinator, gpu_uuid=uuid,
                        reservation_fd=fd, reservation_identity=[stat.st_dev, stat.st_ino],
                        permit=dict(file=str(permit_path), sha256=permit_sha256))
                    path = directory / "request.json"
                    atomic_json(path, request)
                    digest = file_digest(path)
                    env = worker_environment(uuid)
                    env.update(PYTHONPATH=str(registry.root / "src"), PYTHONDONTWRITEBYTECODE="1")
                    with (directory / "worker.log").open("xb") as log:
                        child = subprocess.Popen([sys.executable, "-m", __name__, str(path), digest],
                            cwd=str(registry.root), env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
                    identity = dict(pid=child.pid, parent_pid=os.getpid(), start_ticks=None,
                        process_group=child.pid, session_id=child.pid)
                    handles.append([child, identity, directory, digest])
                    identity = process_identity(child.pid)
                    handles[-1][1] = identity
                    atomic_json(directory / "process.json", identity)
                while True:
                    exits = [peek_exit_code(item[0]) for item in handles]
                    if any(code not in (None, 0) for code in exits):
                        raise RuntimeError("Budget worker failed; evidence retained")
                    atomic_json(output / "progress.json", dict(updated_at=time.time(),
                        workers=[dict(process=item[1], exit_code=code,
                            progress=_read(item[2] / "progress.json") if (item[2] / "progress.json").exists() else None)
                            for item, code in zip(handles, exits)]))
                    if all(code == 0 for code in exits):
                        break
                    time.sleep(5)
                results = []
                for child, identity, directory, digest in handles:
                    terminal = _read(directory / "terminal.json")
                    if (terminal.get("status") != "complete_pending_independent_acceptance"
                            or terminal.get("request_sha256") != digest):
                        raise ValueError("Budget worker terminal identity differs")
                    results.extend(terminal["results"])
                expected = [(item["group_index"], target) for item in bindings for target in registry.paper_order]
                key = lambda row: (row["group_index"], row["target"])
                if sorted(map(key, results)) != sorted(expected):
                    raise ValueError("Budget results are missing or duplicated")
                results.sort(key=lambda row: expected.index(key(row)))
                atomic_json(output / "results.json", results)
            finally:
                cleanup = []
                for child, identity, directory, digest in reversed(handles):
                    try:
                        cleanup.append(close_branch_process(child, identity))
                    except BaseException as exc:
                        cleanup.append(dict(status="failed", pid=child.pid, error=repr(exc)))
                atomic_json(output / "cleanup.json", cleanup)
                if any(row["status"] != "verified" for row in cleanup):
                    raise RuntimeError("Budget worker cleanup could not be verified")
        atomic_json(output / "terminal.json", dict(status="complete_pending_independent_acceptance",
            formal_result_eligible=False, scientific_acceptance=False, seconds=time.time()-started))
    except BaseException as exc:
        atomic_json(output / "terminal.json", dict(status="failed", error=repr(exc),
            scientific_acceptance=False, seconds=time.time()-started))
        raise
    return output


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2])
