"""Owned evaluation-only execution of the frozen Common-2 preprocessing panel."""
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
from .vcsf_final_oblivious_contract import PROTOCOL, build_plan
from .vcsf_final_oblivious_views import materialize_source, verify_view
from .vcsf_final_training_state import _read, source_identity, verify_evaluation


def execution_plan(registry, devices, max_images=None, diagnostic_jobs=None):
    plan = build_plan(registry, devices)
    if max_images is not None and (type(max_images) is not int
            or not 0 < max_images <= registry.protocols[PROTOCOL]["retained_images"]):
        raise ValueError("Diagnostic max-images outside retained population")
    if diagnostic_jobs is not None:
        if (max_images is None or type(diagnostic_jobs) is not int
                or not 0 < diagnostic_jobs <= len(plan["jobs"])):
            raise ValueError("Job truncation requires an explicit image diagnostic")
        plan["jobs"] = plan["jobs"][:diagnostic_jobs]
    if max_images is not None:
        for job in plan["jobs"]:
            job["images"] = max_images
    plan.update(dataset="coco", split="val", max_images=max_images,
        diagnostic_jobs=diagnostic_jobs, formal_scope=max_images is None)
    return plan


def verify_checkpoints(registry, path, expected_hash):
    path = Path(path).resolve()
    if file_digest(path) != expected_hash:
        raise ValueError("Checkpoint binding file changed")
    bindings = _read(path)
    if set(bindings) != set(registry.paper_order):
        raise ValueError("Checkpoint binding must cover the canonical sixteen targets")
    result = {}
    for target in registry.paper_order:
        row = bindings[target]
        checkpoint = Path(row["path"]).resolve()
        if file_digest(checkpoint) != row["sha256"]:
            raise ValueError("Target checkpoint changed: " + target)
        result[target] = dict(path=str(checkpoint), sha256=row["sha256"])
    return result


def verify_view_record(record, view):
    root = Path(view["view"]).resolve()
    if (record.get("attack") != "vcsf"
            or not record.get("adversarial_run") or not record.get("annotation")
            or Path(record["adversarial_run"]).resolve() != root
            or Path(record["annotation"]).resolve() != root / "annotations.json"):
        raise ValueError("Evaluation is not bound to the requested preprocessing view")


def worker(request_path, request_hash):
    path = Path(request_path).resolve()
    if file_digest(path) != request_hash:
        raise ValueError("Worker request changed")
    request = _read(path)
    owner = install_branch_scope(request["coordinator"])
    directory = path.parent
    started = time.time()
    try:
        root = Path(request["source_root"])
        if source_identity(root) != request["source_identity"]:
            raise ValueError("Worker source snapshot changed")
        registry = Registry(root)
        from .vcsf_gpu_context_owner import RegisteredGpuOwner
        from .evaluate import run_evaluation
        gpu = RegisteredGpuOwner(request)
        atomic_json(directory / "gpu_context_registration.json", gpu.receipt)
        plan = request["plan"]
        if plan["formal_scope"]:
            from .vcsf_oblivious_admission import bound_json
            admitted = bound_json(request["admission"]["file"], request["admission"]["sha256"])
            if (admitted["plan"] != plan or admitted["source_identity"] != request["source_identity"]
                    or admitted["payloads"] != request["payloads"]
                    or admitted["checkpoints"] != request["checkpoints"]
                    or admitted["output"] != request["output"]):
                raise ValueError("Worker admission differs from assigned run")
        sources = list(dict.fromkeys(job["source"] for job in request["jobs"]))
        results = []
        for source in sources:
            prepared = materialize_source(registry, source, request["payloads"][source],
                Path(request["output"]) / "views" / source, plan["max_images"])
            views = {row["defense"]: row for row in prepared["views"]}
            for job in (row for row in request["jobs"] if row["source"] == source):
                gpu.verify()
                target, defense = job["target"], job["defense"]
                checkpoint = request["checkpoints"][target]
                if file_digest(Path(checkpoint["path"])) != checkpoint["sha256"]:
                    raise ValueError("Target checkpoint changed before evaluation")
                view = views[defense]
                verify_view(view)
                def progress(completed, total, image_id, status):
                    if completed == 1 or completed % 25 == 0 or completed == total:
                        gpu.verify()
                        atomic_json(directory / "progress.json", dict(source=source,
                            defense=defense, target=target, completed=completed, total=total,
                            image_id=image_id, status=status, completed_jobs=len(results),
                            updated_at=time.time(), owner=owner))
                output = run_evaluation(registry, "coco", target, split="val",
                    adversarial_run=Path(view["view"]),
                    output_dir=Path(request["output"]) / "evaluations" / source / defense / target,
                    device="cuda:0", download_weights=False, keep_going=False,
                    save_visualizations=False, prediction_archive="gzip",
                    target_checkpoint=Path(checkpoint["path"]),
                    image_ids=prepared["image_ids"], progress_callback=progress)
                record = _read(output / "metrics.json")
                verify_evaluation(record, dict(plan, source=source, target=target),
                    prepared["image_ids"], checkpoint["path"], checkpoint["sha256"])
                verify_view_record(record, view)
                verify_view(view)
                results.append(dict(source=source, defense=defense, target=target,
                    metrics_sha256=file_digest(output / "metrics.json"),
                    evaluation=str(output), checkpoint_sha256=checkpoint["sha256"],
                    view_run_sha256=view["run_sha256"]))
                atomic_json(directory / "results.json", results)
        if source_identity(root) != request["source_identity"]:
            raise ValueError("Source snapshot changed during execution")
        gpu.verify()
        atomic_json(directory / "terminal.json", dict(status="complete_pending_independent_acceptance",
            request_sha256=request_hash, owner=owner, results=results,
            seconds=time.time()-started, scientific_acceptance=False))
    except BaseException as exc:
        atomic_json(directory / "terminal.json", dict(status="failed", request_sha256=request_hash,
            owner=owner, error=repr(exc), seconds=time.time()-started, scientific_acceptance=False))
        raise


def run(registry, *, payloads, checkpoint_bindings, checkpoint_bindings_sha256,
        output, devices, max_images=None, diagnostic_jobs=None, plan_only=False,
        admission=None, admission_sha256=None):
    plan = execution_plan(registry, devices, max_images, diagnostic_jobs)
    if set(payloads) != set(registry.protocols[PROTOCOL]["sources"]):
        raise ValueError("Exactly the two registered source payloads are required")
    output = Path(output).resolve()
    protected = [Path(value).resolve() for value in payloads.values()]
    protected += [registry.root / "data", registry.root / "checkpoints", Path(checkpoint_bindings).parent]
    for path in protected:
        path = path.resolve()
        if path == output or path in output.parents or output in path.parents:
            raise ValueError("Output overlaps immutable inputs")
    admission_record = None
    if not plan_only and plan["formal_scope"]:
        from .vcsf_oblivious_admission import verify_admission
        admission_record = verify_admission(registry, admission, admission_sha256, plan=plan,
            output=output, payloads=payloads, checkpoint_bindings=checkpoint_bindings,
            checkpoint_bindings_sha256=checkpoint_bindings_sha256)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    if admission_record is not None:
        atomic_json(output / "admission.json", dict(file=str(Path(admission).resolve()),
            sha256=admission_sha256, receipt=admission_record))
    if plan_only:
        atomic_json(output / "terminal.json", dict(status="planned", execution_admission=False))
        return output
    started = time.time()
    handles = []
    try:
        checkpoints = verify_checkpoints(registry, checkpoint_bindings, checkpoint_bindings_sha256)
        identity = source_identity(registry.root)
        atomic_json(output / "source_identity.json", identity)
        atomic_json(output / "checkpoint_bindings.json", dict(original=str(Path(checkpoint_bindings).resolve()),
            original_sha256=checkpoint_bindings_sha256, checkpoints=checkpoints))
        from .vcsf_structure_workers import physical_gpus, gpu_reservations, worker_environment
        uuids = physical_gpus(devices)
        coordinator = process_identity(os.getpid())
        atomic_json(output / "coordinator.json", coordinator)
        with gpu_reservations(uuids) as reservations:
            try:
                for slot, (device, uuid) in enumerate(zip(devices, uuids)):
                    jobs = [job for job in plan["jobs"] if job["device"] == device]
                    if not jobs:
                        continue
                    directory = output / "workers" / str(slot)
                    directory.mkdir(parents=True, exist_ok=False)
                    fd = reservations[uuid]
                    stat = os.fstat(fd)
                    request = dict(source_root=str(registry.root.resolve()), output=str(output),
                        source_identity=identity, plan=plan,
                        payloads={key: str(Path(value).resolve()) for key, value in payloads.items()},
                        checkpoints=checkpoints, coordinator=coordinator, gpu_uuid=uuid,
                        reservation_fd=fd, reservation_identity=[stat.st_dev, stat.st_ino], jobs=jobs)
                    if admission_record is not None:
                        request["admission"] = dict(file=str(Path(admission).resolve()), sha256=admission_sha256)
                    request_path = directory / "request.json"
                    atomic_json(request_path, request)
                    request_hash = file_digest(request_path)
                    env = worker_environment(uuid)
                    env["PYTHONPATH"] = str(registry.root / "src")
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    with (directory / "worker.log").open("xb") as log:
                        child = subprocess.Popen([sys.executable, "-m", __name__, str(request_path), request_hash],
                            cwd=str(registry.root), env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
                    child_identity = dict(pid=child.pid, parent_pid=os.getpid(), start_ticks=None,
                        process_group=child.pid, session_id=child.pid)
                    handles.append([child, child_identity, directory, request_hash])
                    child_identity = process_identity(child.pid)
                    handles[-1][1] = child_identity
                    atomic_json(directory / "process.json", child_identity)
                while True:
                    exits = [peek_exit_code(item[0]) for item in handles]
                    if any(code not in (None, 0) for code in exits):
                        raise RuntimeError("Preprocessing worker failed; original evidence retained")
                    atomic_json(output / "progress.json", dict(updated_at=time.time(),
                        workers=[dict(process=item[1], exit_code=code,
                            progress=_read(item[2] / "progress.json") if (item[2] / "progress.json").exists() else None)
                            for item, code in zip(handles, exits)]))
                    if all(code == 0 for code in exits):
                        break
                    time.sleep(5)
                results = []
                for child, child_identity, directory, request_hash in handles:
                    terminal = _read(directory / "terminal.json")
                    if (terminal.get("status") != "complete_pending_independent_acceptance"
                            or terminal.get("request_sha256") != request_hash):
                        raise ValueError("Worker terminal receipt mismatch")
                    results.extend(terminal["results"])
                key = lambda row: (row["source"], row["defense"], row["target"])
                expected = [key(job) for job in plan["jobs"]]
                if sorted(map(key, results)) != sorted(expected):
                    raise ValueError("Missing or duplicate preprocessing job")
                results.sort(key=lambda row: expected.index(key(row)))
                atomic_json(output / "results.json", results)
            finally:
                cleanup = []
                for child, child_identity, directory, request_hash in reversed(handles):
                    try:
                        cleanup.append(close_branch_process(child, child_identity))
                    except BaseException as exc:
                        cleanup.append(dict(status="failed", pid=child.pid, error=repr(exc)))
                atomic_json(output / "cleanup.json", cleanup)
                if any(row["status"] != "verified" for row in cleanup):
                    raise RuntimeError("Worker containment cleanup failed")
        atomic_json(output / "terminal.json", dict(status="complete_pending_independent_acceptance",
            formal_scope=plan["formal_scope"], scientific_acceptance=False, seconds=time.time()-started))
    except BaseException as exc:
        atomic_json(output / "terminal.json", dict(status="failed", error=repr(exc),
            scientific_acceptance=False, seconds=time.time()-started))
        raise
    return output


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2])
