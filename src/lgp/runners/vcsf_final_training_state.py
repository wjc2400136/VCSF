"""Owned evaluation-only workers for the registered frozen VCSF victim pair."""
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..registry import Registry
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
)
from .vcsf_final_training_state_contract import PROTOCOL, build_plan, ids_digest, verify_pair, verify_payload


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_identity(root):
    files = {}
    for folder, pattern in (("src", "*.py"), ("configs", "*.yaml"), ("experiments", "*.py")):
        for path in sorted((root / folder).rglob(pattern)):
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return files


def verify_evaluation(record, plan, image_ids, checkpoint, checkpoint_sha256):
    expected = dict(status="complete", dataset=plan["dataset"], split=plan["split"],
                    source=plan["source"], target=plan["target"], images=len(image_ids),
                    parameters_sha256=plan["parameters_sha256"],
                    evaluated_image_ids_sha256=ids_digest(image_ids),
                    expected_image_ids_sha256=ids_digest(image_ids),
                    evaluated_image_ids_match_expected=True,
                    checkpoint=str(Path(checkpoint).resolve()), checkpoint_sha256=checkpoint_sha256)
    if any(record.get(key) != value for key, value in expected.items()) or record.get("failures") != []:
        raise ValueError("Evaluation did not complete its exact registered image/configuration scope")
    if set(record.get("metrics", {})) != set(COCO_BBOX_METRICS):
        raise ValueError("Evaluation must retain all twelve bbox metrics")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not (value == -1 or 0 <= value <= 1) for value in record["metrics"].values()):
            raise ValueError("Evaluation metrics must be finite raw COCO values")


def require_unchanged_payload(registry, expected):
    if verify_payload(registry, expected["payload"]) != expected:
        raise ValueError("Worker payload identity changed since coordinator verification")


def require_output_isolation(output, payload, pair_root, qualification):
    for source in (Path(payload), Path(pair_root), Path(qualification).parent):
        try:
            output.relative_to(source.resolve())
        except ValueError:
            continue
        raise ValueError("Output must not be inside an immutable input root")


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
        gpu = RegisteredGpuOwner(request)
        atomic_json(directory / "gpu_context_registration.json", gpu.receipt)
        from .evaluate import run_evaluation
        from ..reporting.coco_subset import evaluate_bbox_subset
        from ..reporting.coco_summary import write_coco_bbox_summary
        plan = request["plan"]
        full_ids = request["inputs"]["image_ids"]
        image_ids = full_ids if plan["max_images"] is None else full_ids[:plan["max_images"]]
        results = []
        for job in request["jobs"]:
            state = job["victim_state"]
            checkpoint = Path(request["checkpoints"][state])
            if file_digest(checkpoint) != job["checkpoint_sha256"]:
                raise ValueError("Victim checkpoint changed before inference")
            require_unchanged_payload(registry, request["inputs"])
            annotation = _read(request["inputs"]["annotation"])
            def progress(completed, total, image_id, status):
                if completed == 1 or completed % 25 == 0 or completed == total:
                    gpu.verify()
                    atomic_json(directory / "progress.json", dict(victim_state=state,
                        completed=completed, total=total, image_id=image_id, status=status,
                        updated_at=time.time(), owner=owner))
            output = run_evaluation(registry, plan["dataset"], plan["target"], split=plan["split"],
                adversarial_run=Path(request["inputs"]["payload"]),
                output_dir=Path(request["output"]) / "evaluations" / state,
                device="cuda:0", download_weights=False, keep_going=False,
                save_visualizations=False, prediction_archive="gzip", target_checkpoint=checkpoint,
                image_ids=image_ids, progress_callback=progress)
            record = _read(output / "metrics.json")
            verify_evaluation(record, plan, image_ids, checkpoint, job["checkpoint_sha256"])
            require_unchanged_payload(registry, request["inputs"])
            archive = _read(output / "predictions_artifact.json")
            prediction_path = output / "predictions.json.gz"
            if file_digest(prediction_path) != archive["archive_sha256"]:
                raise ValueError("Prediction archive changed")
            with gzip.open(str(prediction_path), "rb") as stream:
                raw = stream.read()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != archive["uncompressed_sha256"] or digest != record["predictions_sha256"]:
                raise ValueError("Predictions do not bind to evaluator metrics")
            predictions = json.loads(raw)
            full_replay = evaluate_bbox_subset(annotation, predictions, image_ids, image_ids)
            if any(abs(full_replay["metrics"][key] - record["metrics"][key]) > 1e-12
                   for key in COCO_BBOX_METRICS):
                raise ValueError("Saved predictions disagree with full-scope metrics")
            selected = [value for value in request["inputs"]["retained_image_ids"] if value in set(image_ids)]
            subset = None
            if selected:
                subset = evaluate_bbox_subset(annotation, predictions, image_ids, selected)
                subset_dir = output / "retained_projection"
                atomic_json(subset_dir / "metrics.json", subset)
                write_coco_bbox_summary([subset["metrics"][key] for key in COCO_BBOX_METRICS],
                    subset_dir, dataset=plan["dataset"], target=plan["target"], source=plan["source"], attack="vcsf")
            results.append(dict(victim_state=state, checkpoint_sha256=job["checkpoint_sha256"],
                full_metrics=record["metrics"], retained_metrics=subset,
                metrics_sha256=file_digest(output / "metrics.json"),
                predictions_sha256=record["predictions_sha256"], archive_sha256=archive["archive_sha256"]))
            atomic_json(directory / "results.json", results)
            gpu.verify()
        if source_identity(root) != request["source_identity"]:
            raise ValueError("Source snapshot changed during execution")
        atomic_json(directory / "terminal.json", dict(status="complete_pending_independent_acceptance",
            request_sha256=request_hash, owner=owner, results=results, seconds=time.time()-started,
            scientific_acceptance=False))
    except BaseException as exc:
        atomic_json(directory / "terminal.json", dict(status="failed", request_sha256=request_hash,
            owner=owner, error=repr(exc), seconds=time.time()-started, scientific_acceptance=False))
        raise


def run(registry, *, payload, pair_root, qualification, output, devices, max_images=None, plan_only=False):
    plan = build_plan(registry, devices, max_images)
    output = Path(output).resolve()
    require_output_isolation(output, payload, pair_root, qualification)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    if plan_only:
        atomic_json(output / "terminal.json", dict(status="planned", execution_admission=False))
        return output
    started = time.time()
    try:
        inputs = verify_payload(registry, payload)
        checkpoints = verify_pair(registry, pair_root, qualification)
        identity = source_identity(registry.root)
        atomic_json(output / "input_verification.json", inputs)
        atomic_json(output / "source_identity.json", identity)
        from .vcsf_structure_workers import physical_gpus, gpu_reservations, worker_environment
        uuids = physical_gpus(devices)
        coordinator = process_identity(os.getpid())
        atomic_json(output / "coordinator.json", coordinator)
        handles = []
        with gpu_reservations(uuids) as reservations:
            try:
                for slot, (device, uuid) in enumerate(zip(devices, uuids)):
                    directory = output / "workers" / str(slot)
                    directory.mkdir(parents=True, exist_ok=False)
                    fd = reservations[uuid]
                    stat = os.fstat(fd)
                    request = dict(source_root=str(registry.root.resolve()), output=str(output),
                        source_identity=identity, plan=plan, inputs=inputs, checkpoints=checkpoints,
                        coordinator=coordinator, gpu_uuid=uuid, reservation_fd=fd,
                        reservation_identity=[stat.st_dev, stat.st_ino],
                        jobs=[job for job in plan["jobs"] if job["device"] == device])
                    request_path = directory / "request.json"
                    atomic_json(request_path, request)
                    request_hash = file_digest(request_path)
                    env = worker_environment(uuid)
                    env["PYTHONPATH"] = str(registry.root / "src")
                    env["PYTHONDONTWRITEBYTECODE"] = "1"
                    log = (directory / "worker.log").open("xb")
                    try:
                        child = subprocess.Popen([sys.executable, "-m", __name__, str(request_path), request_hash],
                            cwd=str(registry.root), env=env, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
                    finally:
                        log.close()
                    child_identity = dict(pid=child.pid, parent_pid=os.getpid(), start_ticks=None,
                                          process_group=child.pid, session_id=child.pid)
                    handles.append([child, child_identity, directory, request_hash])
                    child_identity = process_identity(child.pid)
                    handles[-1][1] = child_identity
                    atomic_json(directory / "process.json", child_identity)
                while True:
                    exits = [peek_exit_code(item[0]) for item in handles]
                    if any(code not in (None, 0) for code in exits):
                        raise RuntimeError("Training-state worker failed; original logs retained")
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
                    if terminal.get("status") != "complete_pending_independent_acceptance" or terminal.get("request_sha256") != request_hash:
                        raise ValueError("Worker terminal receipt mismatch")
                    results.extend(terminal["results"])
                expected_states = [job["victim_state"] for job in plan["jobs"]]
                if sorted(row["victim_state"] for row in results) != sorted(expected_states):
                    raise ValueError("Missing or duplicate completed victim-state job")
                results.sort(key=lambda row: expected_states.index(row["victim_state"]))
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
                    raise RuntimeError("Worker containment cleanup failed; inspect cleanup.json")
        atomic_json(output / "terminal.json", dict(status="complete_pending_independent_acceptance",
            formal_scope=plan["formal_scope"], scientific_acceptance=False, seconds=time.time()-started))
    except BaseException as exc:
        atomic_json(output / "terminal.json", dict(status="failed", error=repr(exc), scientific_acceptance=False,
                                                  seconds=time.time()-started))
        raise
    return output


if __name__ == "__main__":
    worker(sys.argv[1], sys.argv[2])
