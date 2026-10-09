"""Complete-target online visualization jobs; metadata-only planning."""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from ..io import atomic_json, file_digest
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
)


@lru_cache(maxsize=1)
def _device_helpers():
    # Reuse the original helper ASTs without importing its attack/evaluation stack.
    path = Path(__file__).with_name("formal_parallel.py")
    names = {"normalize_execution_devices", "validate_available_cuda_devices", "round_robin_indices"}
    nodes = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "_CUDA_DEVICE"):
            nodes.append(node)
    if (len(nodes) != len(names) + 1
            or {node.name for node in nodes if isinstance(node, ast.FunctionDef)} != names):
        raise RuntimeError("Original execution-device helpers are missing or duplicated")
    future = ast.parse("from __future__ import annotations").body
    tree = ast.Module(body=future + nodes, type_ignores=[])
    namespace = {"re": re}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)
    return namespace


def _selected_devices(device, devices):
    helpers = _device_helpers()
    selected = helpers["normalize_execution_devices"](device, devices)
    if len(selected) not in (1, 2):
        raise ValueError("Select one or two execution devices")
    if devices is not None and any(not helpers["_CUDA_DEVICE"].fullmatch(d) for d in selected):
        raise ValueError("--devices requires one or two distinct explicit CUDA devices")
    if devices is None and selected[0] not in ("cpu", "cuda") and not helpers["_CUDA_DEVICE"].fullmatch(selected[0]):
        raise ValueError("--device must be cpu, cuda, or an explicit CUDA ordinal")
    return selected


def _validate_devices(devices):
    import torch

    helpers = _device_helpers()
    helpers["torch"] = torch
    helpers["validate_available_cuda_devices"](devices, include_single=True)


def _absolute(path):
    # Do not probe adversarial payloads or follow runtime symlinks while planning.
    return str(Path(os.path.abspath(os.path.expanduser(str(path)))))


def _declaration(path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Invalid registry declaration: " + str(path))
    return payload


def build_plan(root, *, dataset, split, target, targets=None, device="cuda:0", devices=None,
               adversarial_run=None, max_images=20, score_threshold=0.30,
               max_detections=100, download_weights=False, stop_on_error=False, output=None):
    root = Path(_absolute(root))
    models = _declaration(root / "configs/models.yaml")
    canonical = models["paper_order"]
    if (not isinstance(canonical, list) or len(set(canonical)) != len(canonical)
            or set(canonical) != set(models["models"])):
        raise ValueError("Model registry paper_order is inconsistent")
    requested = [target] if targets is None else list(targets)
    if (not requested or len(set(requested)) != len(requested)
            or any(t not in canonical for t in requested)):
        raise ValueError("Select a unique registered target subset: " + ", ".join(canonical))
    selected = [t for t in canonical if t in requested]
    if dataset not in ("coco", "voc"):
        raise ValueError("Online visualization is maintained only for COCO/VOC")
    declaration = _declaration(root / "configs/datasets" / (dataset + ".yaml"))
    if split not in declaration["splits"]:
        raise ValueError("Dataset '{}' has no '{}' split".format(dataset, split))
    if type(max_images) is not int or max_images <= 0:
        raise ValueError("max_images must be positive")
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must lie in [0, 1]")
    if type(max_detections) is not int or max_detections <= 0:
        raise ValueError("max_detections must be positive")
    execution = _selected_devices(device, devices)
    input_kind = "adversarial" if adversarial_run is not None else "clean"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = _absolute(output if output is not None else
        root / "outputs/visualizations" / dataset / (selected[0] if len(selected) == 1 else "multi_target") / input_kind / stamp)
    jobs = [{"index": i, "dataset": dataset, "split": split, "target": t,
             "adversarial_run": _absolute(adversarial_run) if adversarial_run is not None else None,
             "max_images": max_images, "score_threshold": score_threshold,
             "max_detections": max_detections, "save_visualizations": True,
             "download_weights": bool(download_weights), "keep_going": not stop_on_error,
             "evaluation_relative_dir": "." if len(selected) == 1 else "targets/" + t}
            for i, t in enumerate(selected)]
    lanes = _device_helpers()["round_robin_indices"](len(jobs), len(execution))
    assignments = sorted([{"job_index": i, "slot": slot, "device": execution[slot]}
                          for slot, indices in enumerate(lanes) for i in indices], key=lambda row: row["job_index"])
    return {"schema_version": 1, "purpose": "online_post_NMS_prediction_visualization",
            "planning_only_runtime_inputs_not_validated": True,
            "formal": False, "science": False, "whole_project_release": False,
            "input_kind": input_kind, "output_dir": output_dir,
            "dispatch_dir": output_dir + ".dispatch", "jobs": jobs,
            "devices": execution, "assignments": assignments,
            "job_atomicity": "one complete target evaluation; no image sharding or new attacks",
            "stop_on_error": bool(stop_on_error)}


def _evaluate_job(root, job, output, device):
    from ..registry import Registry
    from .evaluate import run_evaluation

    return run_evaluation(
        Registry(root), job["dataset"], job["target"], split=job["split"],
        adversarial_run=Path(job["adversarial_run"]) if job["adversarial_run"] is not None else None,
        output_dir=Path(output), max_images=job["max_images"], device=device,
        download_weights=job["download_weights"], keep_going=job["keep_going"],
        save_visualizations=True, visualization_score_threshold=job["score_threshold"],
        visualization_max_images=job["max_images"], visualization_max_detections=job["max_detections"],
    )


def worker_main(request_path, expected_sha256):
    request_path = Path(request_path).resolve()
    if file_digest(request_path) != expected_sha256:
        raise RuntimeError("Visualization worker request hash mismatch")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    owner = install_branch_scope(request["coordinator"])
    if request_path != Path(request["directory"]) / "request.json":
        raise RuntimeError("Visualization worker request directory mismatch")
    atomic_json(request_path.parent / "owner.json", owner)
    completed = _evaluate_job(Path(request["root"]), request["job"], request["output_dir"], request["device"])
    if Path(completed).resolve() != Path(request["output_dir"]).resolve():
        raise RuntimeError("Visualization evaluation returned an unexpected directory")


def _spawn(root, request_path, request_sha256, log):
    environment = dict(os.environ, PYTHONPATH=str(root / "src"),
                       LGP_PROJECT_ROOT=str(root), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.Popen(
        [sys.executable, "-B", "-u", "-m", __name__, str(request_path), request_sha256],
        cwd=str(root), env=environment, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def _completed_record(job, output, request_sha256):
    metrics_path = output / "metrics.json"
    result = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Evaluation metrics must be a mapping")
    expected_run = str(Path(job["adversarial_run"]).resolve()) if job["adversarial_run"] is not None else None
    if (any(result.get(field) != job[field] for field in ("dataset", "split", "target"))
            or result.get("adversarial_run") != expected_run):
        raise RuntimeError("Evaluation metrics do not match the assigned target input")
    overlays = result["prediction_visualizations"]
    if not isinstance(overlays, dict):
        raise RuntimeError("Prediction overlay metrics must be a mapping")
    if (result.get("status") != "complete" or result.get("failures")
            or result.get("evaluated_image_ids_match_expected") is not True
            or overlays.get("enabled") is not True or overlays.get("failures")):
        raise RuntimeError("Target evaluation or prediction overlays are incomplete")
    if (overlays.get("score_threshold") != job["score_threshold"]
            or overlays.get("max_images") != job["max_images"]
            or overlays.get("max_detections_per_image") != job["max_detections"]):
        raise RuntimeError("Prediction overlay settings differ from the assigned job")
    if (type(overlays.get("saved")) is not int or type(overlays.get("attempted")) is not int
            or not 0 < overlays["attempted"] <= job["max_images"]
            or overlays["saved"] != overlays["attempted"]):
        raise RuntimeError("Prediction overlays were not completely saved")
    return {"status": "complete", "metrics_path": str(metrics_path),
            "metrics_sha256": file_digest(metrics_path), "request_sha256": request_sha256,
            "saved": overlays["saved"], "attempted": overlays["attempted"],
            "visualization_directory": overlays["directory"]}


def _run_jobs(root, plan):
    output = Path(plan["output_dir"])
    dispatch = Path(plan["dispatch_dir"])
    if output.exists() or dispatch.exists():
        raise FileExistsError("Refusing to reuse visualization output or dispatch evidence")
    output.mkdir(parents=True, exist_ok=False)
    dispatch.mkdir(parents=True, exist_ok=False)
    atomic_json(dispatch / "plan.json", plan)
    coordinator = process_identity(os.getpid())
    pending = {slot: [row for row in plan["assignments"] if row["slot"] == slot]
               for slot in range(len(plan["devices"]))}
    records = [{"index": job["index"], "target": job["target"], "status": "NR",
                "output_dir": str(output / job["evaluation_relative_dir"])} for job in plan["jobs"]]
    active, cleanup, cleanup_errors = {}, [], []
    error = None

    def launch(slot, assignment):
        index = assignment["job_index"]
        job = plan["jobs"][index]
        directory = dispatch / "jobs" / "{:02d}_{}".format(index, job["target"])
        directory.mkdir(parents=True, exist_ok=False)
        request = {"schema_version": 1, "root": str(root), "directory": str(directory),
                   "coordinator": coordinator, "device": assignment["device"], "slot": slot,
                   "job": job, "output_dir": records[index]["output_dir"]}
        path = directory / "request.json"
        atomic_json(path, request)
        digest = file_digest(path)
        log = (directory / "stdout.log").open("xb")
        try:
            process = _spawn(root, path, digest, log)
        except BaseException as exc:
            records[index].update(status="ERR", reason="Worker launch failed: {}".format(exc))
            log.close()
            raise
        fallback = {"pid": process.pid, "parent_pid": os.getpid(), "start_ticks": None,
                    "process_group": process.pid, "session_id": process.pid}
        active[slot] = {"process": process, "identity": fallback, "log": log,
                        "index": index, "request_path": path, "request_sha256": digest}
        records[index].update(status="running", device=assignment["device"], slot=slot,
                              pid=process.pid, request_sha256=digest)
        identity = process_identity(process.pid)
        active[slot]["identity"] = identity
        if (identity["parent_pid"] != os.getpid() or identity["process_group"] != process.pid
                or identity["session_id"] != process.pid):
            raise RuntimeError("Visualization worker did not enter its owned session")
        records[index]["start_ticks"] = identity["start_ticks"]

    try:
        while any(pending.values()) or active:
            for slot in pending:
                if slot not in active and pending[slot]:
                    launch(slot, pending[slot].pop(0))
            for slot, owned in list(active.items()):
                code = peek_exit_code(owned["process"])
                if code is None:
                    continue
                closed = close_branch_process(owned["process"], owned["identity"])
                cleanup.append({"index": owned["index"], "pid": owned["process"].pid, **closed})
                owned["log"].close()
                del active[slot]
                index = owned["index"]
                records[index]["exit_code"] = closed["exit_code"]
                try:
                    if code != 0 or closed["exit_code"] != 0 or closed["observed_live_members"]:
                        raise RuntimeError("Target worker failed or left live descendants (exit={})".format(code))
                    if file_digest(owned["request_path"]) != owned["request_sha256"]:
                        raise RuntimeError("Visualization worker request changed during execution")
                    records[index].update(_completed_record(plan["jobs"][index], Path(records[index]["output_dir"]),
                                                            owned["request_sha256"]))
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    records[index].update(status="ERR", reason="{}: {}".format(type(exc).__name__, exc))
                    if plan["stop_on_error"]:
                        raise RuntimeError(records[index]["reason"]) from exc
            if active:
                time.sleep(0.05)
    except BaseException as exc:
        error = exc
    finally:
        for slot, owned in list(active.items()):
            try:
                closed = close_branch_process(owned["process"], owned["identity"])
                cleanup.append({"index": owned["index"], "pid": owned["process"].pid, **closed})
                records[owned["index"]].update(status="ERR", exit_code=closed["exit_code"],
                                              reason="Terminated after coordinator failure or interruption")
            except BaseException as exc:
                cleanup_errors.append("{}: {}".format(type(exc).__name__, exc))
            finally:
                owned["log"].close()
        for record in records:
            if record["status"] == "running":
                record.update(status="ERR", reason="Worker cleanup not verified")
        status = "complete" if error is None and not cleanup_errors and all(r["status"] == "complete" for r in records) else "failed"
        terminal = {"schema_version": 1, "status": status, "jobs": records,
                    "cleanup": cleanup, "cleanup_errors": cleanup_errors,
                    "coordinator_error": "{}: {}".format(type(error).__name__, error) if error else None,
                    "formal": False, "science": False, "whole_project_release": False}
        atomic_json(dispatch / "terminal.json", terminal)
    if cleanup_errors:
        raise RuntimeError("Visualization process cleanup failed: " + "; ".join(cleanup_errors))
    if error is not None:
        raise error
    return terminal


def execute_plan(root, plan):
    from ..registry import Registry

    root = Path(root).resolve()
    Registry(root)  # Keep the existing full-source identity guard; no candidate bypass.
    _validate_devices(plan["devices"])
    return _run_jobs(root, plan)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Internal visualization worker requires a request and its SHA-256")
    worker_main(Path(sys.argv[1]), sys.argv[2])
