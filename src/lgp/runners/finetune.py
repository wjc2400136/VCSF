"""Immutable public queues for registered single-GPU VOC fine-tuning."""
from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

from ..io import atomic_json, file_digest
from ..modeling import checkpoint_path
from ..registry import Registry
from .formal_parallel import normalize_execution_devices, validate_available_cuda_devices
from .training_pair_processes import close_branch_process, peek_exit_code, process_identity


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Fine-tuning evidence must be a JSON object: " + str(path))
    return value


def _relative(root, path):
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def _inside(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path.resolve().relative_to(root.resolve())
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise RuntimeError("Fine-tuning paths cannot traverse symbolic links")
    return path.resolve()


def _models(registry, requested):
    canonical = registry.target_ids()
    values = canonical if requested is None else list(requested)
    if not values or len(values) != len(set(values)) or set(values) - set(canonical):
        raise ValueError("Select distinct canonical detector IDs, or all")
    return [model for model in canonical if model in values]


def build_finetune_plan(registry, models=None, max_images=None,
                        frozen_existing=None):
    selected_datasets = ["voc"]
    selected_models = _models(registry, models)
    frozen = {"voc": _models(registry, frozen_existing) if frozen_existing else []}
    if max_images is not None:
        if isinstance(max_images, bool) or not isinstance(max_images, int) or max_images <= 0:
            raise ValueError("--max-images must be a positive training-image count")
        if any(max_images > registry.dataset(d).split("train").expected_images for d in selected_datasets):
            raise ValueError("--max-images exceeds a selected dataset's registered training count")
        if any(frozen.values()):
            raise ValueError("Diagnostic training cannot reuse canonical checkpoints")
    jobs = []
    for dataset_id in selected_datasets:
        dataset = registry.dataset(dataset_id)
        for model_id in selected_models:
            model = registry.model(model_id)
            jobs.append({
                "job_id": dataset_id + "/" + model_id,
                "dataset": dataset_id, "model": model_id, "split": "train",
                "registered_train_images": dataset.split("train").expected_images,
                "max_images": max_images,
                "epochs": int(model.finetune_epochs[dataset_id]),
                "micro_batch": model.train_batch_size,
                "reference_total_batch": model.train_base_batch_size,
                "learning_rate_scale": "{}/{}".format(model.train_batch_size, model.train_base_batch_size),
                "framework": model.framework, "upstream_config": model.config,
                "warm_start": model.checkpoint,
                "gradient_clip": model.train_grad_clip,
                "head_initialization": model.finetune_head_init.get(dataset_id),
                "classes": dataset.classes,
            })
    return {
        "schema_version": 1,
        "workflow": "registered_voc_detector_finetuning",
        "datasets": selected_datasets, "models": selected_models,
        "canonical_models": registry.target_ids(),
        "training_protocol": registry.training_protocol,
        "training_safety": registry.training_safety,
        "diagnostic": max_images is not None,
        "canonical_publication_allowed": max_images is None,
        "max_images": max_images,
        "frozen_existing": frozen,
        "automatic_retry": False, "automatic_resume": False,
        "jobs": jobs,
    }


def _helpers(root):
    path = root / "tools" / "run_finetune_dual_queue.py"
    spec = importlib.util.spec_from_file_location("lgp_public_finetune_legacy_helpers", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the retained fine-tuning validation helpers")
    dual = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dual)
    return dual._load_serial_queue(root), dual


def _diagnostic_fingerprint(registry, plan, serial):
    files = {}
    for directory, suffix in (("src", ".py"), ("configs", ".yaml")):
        for path in sorted((registry.root / directory).rglob("*" + suffix)):
            files[_relative(registry.root, path)] = file_digest(path)
    for name in ("tools/run_finetune_queue.py", "tools/run_finetune_dual_queue.py"):
        files[name] = file_digest(registry.root / name)
    for dataset_id in plan["datasets"]:
        dataset = registry.dataset(dataset_id)
        path = dataset.root / dataset.split("train").annotation
        files[_relative(registry.root, path)] = file_digest(path)
    for model_id in plan["models"]:
        path = checkpoint_path(registry.model(model_id), registry.dataset("coco"))
        files[_relative(registry.root, path)] = file_digest(path)
    return {"diagnostic": True, "files": files, "packages": serial._package_versions()}


def _assert_inputs(registry, context, dataset, model):
    serial, dual = context["serial"], context["dual"]
    if context["fingerprint"].get("diagnostic"):
        for name, expected in context["fingerprint"]["files"].items():
            if file_digest(registry.root / name) != expected:
                raise RuntimeError("Diagnostic fine-tuning input changed: " + name)
        if serial._package_versions() != context["fingerprint"]["packages"]:
            raise RuntimeError("Diagnostic fine-tuning environment changed")
    else:
        serial._assert_campaign_unchanged(registry, context["fingerprint"], dataset, model)
        for frozen_dataset, records in context["frozen"].items():
            dual._assert_frozen_existing_unchanged(serial, registry, frozen_dataset, records, deep=False)


def _prepare_execution(registry, plan, devices, min_free_gib):
    serial, dual = _helpers(registry.root)
    context = {"serial": serial, "dual": dual, "frozen": {}, "expected_commits": {}, "reused": {}}
    if plan["diagnostic"]:
        context["fingerprint"] = _diagnostic_fingerprint(registry, plan, serial)
        return context
    git = serial._git_state(registry.root)
    if git["status"]:
        raise RuntimeError("Formal fine-tuning requires a clean committed worktree")
    datasets = ["voc"]
    context["fingerprint"] = serial._campaign_fingerprint(
        registry, datasets, SimpleNamespace(device=devices[0], max_attempts=1, min_free_gib=min_free_gib), git, {},
    )
    for dataset in datasets:
        frozen = dual._capture_frozen_existing(serial, registry, dataset,
            plan["frozen_existing"].get(dataset, []), devices[0], {})
        context["frozen"][dataset] = frozen
        commits = dual._expected_commit_map(git["commit"], registry.target_ids(), frozen)
        context["expected_commits"][dataset] = commits
        for model in plan["models"] if dataset in plan["datasets"] else []:
            ready, reason = serial._checkpoint_ready(registry, dataset, model, commits[model], {})
            path = checkpoint_path(registry.model(model), registry.dataset(dataset))
            if ready:
                if model not in frozen:
                    from ..training_artifacts import strict_load_finetuned_checkpoint

                    strict_load_finetuned_checkpoint(registry, dataset, model, devices[0])
                context["reused"][dataset + "/" + model] = {
                    "checkpoint": _relative(registry.root, path), "checkpoint_sha256": file_digest(path),
                    "git_commit": commits[model], "reason": reason,
                }
            elif path.exists() or path.is_symlink():
                raise RuntimeError("Unqualified existing checkpoint must not be overwritten: " + dataset + "/" + model + ": " + reason)
    return context


def _training_worker(request_value, request_sha256, binding):
    from .train import run_training

    registry = Registry()
    request_path = _inside(registry.root, request_value)
    if file_digest(request_path) != request_sha256:
        raise RuntimeError("Fine-tuning worker request changed")
    request = _read(request_path)
    job = request["job"]
    job_dir = request_path.parent
    terminal_path = job_dir / "worker_execution.json"
    terminal = {
        "schema_version": 1, **binding, "request_sha256": request_sha256,
        "job_id": job["job_id"], "dataset": job["dataset"], "model": job["model"],
        "device": request["device"], "status": "running", "started_at": _now(),
    }
    atomic_json(terminal_path, terminal)
    try:
        result = run_training(registry, job["dataset"], job["model"],
            work_dir=job_dir / "training", device=request["device"],
            max_images=job["max_images"],
        )
        run_path = job_dir / "training" / "run.json"
        terminal.update({
            "status": "complete", "completed_at": _now(),
            "checkpoint": _relative(registry.root, result), "checkpoint_sha256": file_digest(result),
            "training_run": _relative(registry.root, run_path), "training_run_sha256": file_digest(run_path),
        })
        atomic_json(terminal_path, terminal)
    except BaseException as exc:
        terminal.update(status="failed", completed_at=_now(), failure="{}: {}".format(type(exc).__name__, exc))
        atomic_json(terminal_path, terminal)
        raise


def _worker_command(request, request_sha256, coordinator):
    payload = {"request": str(request), "request_sha256": request_sha256, "coordinator": coordinator}
    return [sys.executable, "-B", "-c", (
        "import json,sys; p=json.loads(sys.argv[1]); "
        "from lgp.runners.training_pair_processes import install_branch_scope; "
        "binding=install_branch_scope(p['coordinator']); "
        "from lgp.runners.finetune import _training_worker; "
        "_training_worker(p['request'],p['request_sha256'],binding)"
    ), json.dumps(payload)]


def _verify_terminal(registry, row, job_dir, identity, coordinator, plan, context):
    if file_digest(job_dir / "request.json") != row["request_sha256"]:
        raise RuntimeError("Fine-tuning worker request changed before acceptance")
    terminal_path = job_dir / "worker_execution.json"
    terminal = _read(terminal_path)
    expected = {
        "status": "complete", "job_id": row["job_id"], "dataset": row["dataset"],
        "model": row["model"], "device": row["device"],
        "request_sha256": row["request_sha256"],
        "worker_pid": identity["pid"], "worker_start_ticks": identity["start_ticks"],
        "process_group": identity["pid"], "coordinator_pid": coordinator["pid"],
        "coordinator_start_ticks": coordinator["start_ticks"],
    }
    if any(terminal.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Fine-tuning worker terminal identity mismatch: " + row["job_id"])
    run_path = _inside(registry.root, terminal["training_run"])
    if run_path != (job_dir / "training" / "run.json").resolve() or file_digest(run_path) != terminal["training_run_sha256"]:
        raise RuntimeError("Fine-tuning training-run binding mismatch")
    run = _read(run_path)
    checkpoint = _inside(registry.root, terminal["checkpoint"])
    if (run.get("dataset") != row["dataset"] or run.get("model") != row["model"]
            or run.get("final_checkpoint") != terminal["checkpoint"]
            or run.get("final_checkpoint_sha256") != terminal["checkpoint_sha256"]
            or file_digest(checkpoint) != terminal["checkpoint_sha256"]):
        raise RuntimeError("Fine-tuning checkpoint or training record mismatch")
    protocol = run.get("protocol", {})
    for name in ("protocol_config", "train_annotation", "source_checkpoint"):
        bound = _inside(registry.root, run[name])
        if file_digest(bound) != run[name + "_sha256"]:
            raise RuntimeError("Fine-tuning bound training input changed: " + name)
    if _inside(registry.root, run["protocol_config"]) != (job_dir / "training" / "protocol_config.py").resolve():
        raise RuntimeError("Fine-tuning protocol config is outside its training job")
    job = next(job for job in plan["jobs"] if job["job_id"] == row["job_id"])
    protocol_expected = {**plan["training_protocol"], "epochs": job["epochs"],
        "micro_batch_size": job["micro_batch"], "reference_batch_size": job["reference_total_batch"]}
    if any(protocol.get(key) != value for key, value in protocol_expected.items()):
        raise RuntimeError("Fine-tuning resolved training protocol changed")
    if plan["diagnostic"]:
        checkpoint.relative_to((job_dir / "training").resolve())
        diagnostic = run.get("diagnostic", {})
        ids = diagnostic.get("image_ids", [])
        if (run.get("status") != "diagnostic_complete" or run.get("canonical_publication_allowed") is not False
                or diagnostic.get("canonical_publication_allowed") is not False
                or diagnostic.get("images") != plan["max_images"]
                or len(ids) != len(set(ids)) or len(ids) != plan["max_images"]
                or run.get("publish_method") is not None):
            raise RuntimeError("Diagnostic fine-tuning did not establish its limited non-publication contract")
        for name in ("subset_annotation", "source_annotation"):
            bound = _inside(registry.root, diagnostic[name])
            if file_digest(bound) != diagnostic[name + "_sha256"]:
                raise RuntimeError("Diagnostic annotation binding changed")
        subset = _inside(registry.root, diagnostic["subset_annotation"])
        subset.relative_to((job_dir / "training").resolve())
        if subset != _inside(registry.root, run["train_annotation"]):
            raise RuntimeError("Diagnostic training annotation is not its recorded subset")
        for receipt in (diagnostic, diagnostic.get("runtime_dataset"), diagnostic.get("runtime_dataset_after_training")):
            if (not isinstance(receipt, dict) or receipt.get("image_ids") != ids
                    or receipt.get("images") != len(ids) or receipt.get("metadata_only") is not True
                    or not isinstance(receipt.get("leaves"), list) or not receipt["leaves"]):
                raise RuntimeError("Diagnostic dataset verification receipt is incomplete")
            for leaf in receipt["leaves"]:
                leaf_ids = leaf.get("image_ids", [])
                if (leaf.get("exact_id_set") is not True or leaf.get("images") != len(ids)
                        or len(leaf_ids) != len(ids) or set(leaf_ids) != set(ids)):
                    raise RuntimeError("Diagnostic dataset leaf differs from the selected images")
        verification = run.get("verification", {})
        verified_ids = verification.get("image_ids", [])
        if (verification.get("strict_state_dict_load") is not True
                or verification.get("split") != "train"
                or verification.get("checkpoint_epoch") != job["epochs"]
                or verification.get("images") != min(plan["max_images"], int(plan["training_protocol"]["verify_train_images"]))
                or len(verified_ids) != verification["images"] or len(verified_ids) != len(set(verified_ids))
                or not set(verified_ids).issubset(set(ids))):
            raise RuntimeError("Diagnostic checkpoint verification is incomplete or outside the selected images")
    else:
        if run.get("status") != "complete":
            raise RuntimeError("Formal training did not complete publication")
        expected_path = checkpoint_path(registry.model(row["model"]), registry.dataset(row["dataset"])).resolve()
        if checkpoint != expected_path:
            raise RuntimeError("Formal checkpoint is outside its canonical model path")
        ready, reason = context["serial"]._checkpoint_ready(registry, row["dataset"], row["model"],
            context["expected_commits"][row["dataset"]][row["model"]], {})
        if not ready:
            raise RuntimeError("Formal checkpoint failed qualification: " + reason)
    return {"terminal_sha256": file_digest(terminal_path), "training_run_sha256": file_digest(run_path),
            "checkpoint": terminal["checkpoint"], "checkpoint_sha256": terminal["checkpoint_sha256"]}


def _execute_jobs(registry, run_dir, plan, assignment, context, state, write_state, min_free_gib):
    coordinator = process_identity(os.getpid())
    state["coordinator"] = coordinator
    jobs = {job["job_id"]: job for job in plan["jobs"]}
    pending = {device: [] for device in assignment["devices"]}
    for row in assignment["jobs"]:
        if row["job_id"] in context["reused"]:
            state["jobs"][row["job_id"]].update(status="complete", reused_verified_checkpoint=True,
                                               **context["reused"][row["job_id"]])
        else:
            pending[row["device"]].append(row["job_id"])
    active = {}
    env = dict(os.environ)
    env["LGP_PROJECT_ROOT"] = str(registry.root)
    env["PYTHONPATH"] = str(registry.root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    write_state()
    try:
        while any(pending.values()) or active:
            for device in assignment["devices"]:
                if device in active or not pending[device]:
                    continue
                job_id = pending[device].pop(0)
                job, row = jobs[job_id], state["jobs"][job_id]
                _assert_inputs(registry, context, job["dataset"], job["model"])
                if file_digest(run_dir / "execution_assignments.json") != state["assignment_sha256"] or file_digest(run_dir / "plan.json") != state["plan_sha256"]:
                    raise RuntimeError("Fine-tuning plan or assignment changed")
                job_dir = run_dir / "jobs" / job["dataset"] / job["model"]
                job_dir.mkdir(parents=True, exist_ok=False)
                context["serial"]._disk_gate([job_dir, checkpoint_path(registry.model(job["model"]), registry.dataset(job["dataset"])).parent], min_free_gib)
                request = job_dir / "request.json"
                atomic_json(request, {"schema_version": 1, "job": job, "device": device,
                                     "coordinator": coordinator, "plan_sha256": state["plan_sha256"]})
                row.update(status="launching", request_sha256=file_digest(request), directory=_relative(registry.root, job_dir))
                write_state()
                with (job_dir / "stdout.log").open("xb") as log:
                    process = subprocess.Popen(_worker_command(request, row["request_sha256"], coordinator),
                        cwd=str(registry.root), env=env, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                identity = {"pid": process.pid, "parent_pid": coordinator["pid"], "process_group": process.pid,
                            "session_id": process.pid, "start_ticks": None}
                active[device] = (process, identity, job_id, job_dir)
                identity.update(process_identity(process.pid))
                if identity["parent_pid"] != coordinator["pid"] or identity["process_group"] != process.pid or identity["session_id"] != process.pid:
                    raise RuntimeError("Fine-tuning worker does not own its assigned process group")
                row.update(status="running", process_identity=identity, started_at=_now())
                atomic_json(job_dir / "launch.json", {**identity, "job_id": job_id, "device": device})
                write_state()
            for device, (process, identity, job_id, job_dir) in list(active.items()):
                row = state["jobs"][job_id]
                code = peek_exit_code(process)
                terminal = job_dir / "worker_execution.json"
                if code is None:
                    if terminal.is_file() and _read(terminal).get("status") == "failed":
                        raise RuntimeError("Fine-tuning worker reported failure: " + job_id)
                    continue
                row["exit_code"] = code
                if code != 0:
                    raise RuntimeError("Fine-tuning worker exited with {}: {}".format(code, job_id))
                row.update(_verify_terminal(registry, row, job_dir, identity, coordinator, plan, context))
                row["cleanup"] = close_branch_process(process, identity)
                row.update(status="complete", completed_at=_now())
                del active[device]
                _assert_inputs(registry, context, row["dataset"], row["model"])
                write_state()
            if active:
                write_state(throttled=True)
                time.sleep(0.2)
    finally:
        failures = []
        for device, (process, identity, job_id, job_dir) in active.items():
            row = state["jobs"][job_id]
            try:
                row["cleanup"] = close_branch_process(process, identity)
            except Exception as exc:
                failures.append(job_id + ": " + str(exc))
            row.update(status="failed", completed_at=_now(), failure="Training worker did not reach an accepted terminal state")
        for row in state["jobs"].values():
            if row["status"] == "launching":
                row.update(status="failed", failure="Training worker launch did not complete")
        write_state()
        if failures:
            raise RuntimeError("Fine-tuning cleanup was not verified: " + "; ".join(failures))


def run_finetuning(registry, *, execute=True, output_dir=None, devices=None,
                   models=None, max_images=None, frozen_existing=None, min_free_gib=12.0):
    plan = build_finetune_plan(registry, models, max_images, frozen_existing)
    selected_devices = normalize_execution_devices("cuda:0", devices)
    if len(selected_devices) not in (1, 2) or any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in selected_devices):
        raise ValueError("VOC fine-tuning requires one or two distinct explicit CUDA devices")
    if not math.isfinite(min_free_gib) or min_free_gib <= 0:
        raise ValueError("Minimum free disk space must be finite and positive")
    if execute:
        if sys.platform != "linux":
            raise RuntimeError("Execute fine-tuning in the Linux server ODA environment")
        configured = os.environ.get("LGP_PROJECT_ROOT")
        if configured and Path(configured).expanduser().resolve() != registry.root.resolve():
            raise RuntimeError("Configured project root conflicts with the VOC fine-tuning registry")
        validate_available_cuda_devices(selected_devices, include_single=True)
    output = output_dir or registry.root / "outputs" / "training" / "voc_finetuning" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _inside(registry.root, output)
    try:
        run_dir.relative_to((registry.root / "checkpoints").resolve())
    except ValueError:
        pass
    else:
        raise ValueError("Fine-tuning execution records cannot be placed inside the checkpoint registry")
    run_dir.mkdir(parents=True, exist_ok=False)
    assignment = {"schema_version": 1, "algorithm": "canonical_model_round_robin_v1", "devices": selected_devices,
        "jobs": [{"job_id": job["job_id"], "dataset": job["dataset"], "model": job["model"],
                  "canonical_index": index + 1, "worker_slot": index % len(selected_devices),
                  "device": selected_devices[index % len(selected_devices)]}
                 for index, job in enumerate(plan["jobs"])]}
    atomic_json(run_dir / "execution_assignments.json", assignment)
    assignment_sha256 = file_digest(run_dir / "execution_assignments.json")
    plan["execution_scheduler"] = {"devices": selected_devices, "assignment_sha256": assignment_sha256}
    atomic_json(run_dir / "plan.json", plan)
    state = {"schema_version": 1, "status": "planned", "created_at": _now(),
        "assignment_sha256": assignment_sha256, "plan_sha256": file_digest(run_dir / "plan.json"),
        "jobs": {row["job_id"]: {**row, "status": "planned"} for row in assignment["jobs"]}, "gates": {}}
    last_write = [0.0]

    def write_state(throttled=False):
        if throttled and time.monotonic() - last_write[0] < 2:
            return
        state["updated_at"] = _now()
        atomic_json(run_dir / "execution_state.json", state)
        last_write[0] = time.monotonic()

    def finish(status):
        state["status"] = status
        state["completed_jobs"] = sum(row["status"] == "complete" for row in state["jobs"].values())
        state["failed_jobs"] = sum(row["status"] == "failed" for row in state["jobs"].values())
        write_state()
        summary = {"schema_version": 1, "status": status, "diagnostic": plan["diagnostic"],
            "total_jobs": len(plan["jobs"]), "completed_jobs": state["completed_jobs"],
            "failed_jobs": state["failed_jobs"], "failure": state.get("failure"),
            "reused_jobs": sum(row.get("reused_verified_checkpoint") is True for row in state["jobs"].values()),
            "devices": selected_devices, "assignment_sha256": assignment_sha256,
            "voc_panel_qualified": status == "complete" and bool(state["gates"].get("voc")),
            "canonical_publication_allowed": not plan["diagnostic"],
            "execution_state_sha256": file_digest(run_dir / "execution_state.json"),
        }
        atomic_json(run_dir / "summary.json", summary)

    finish("planned")
    if not execute:
        return run_dir
    handlers = {}

    def interrupted(signum, frame):
        raise InterruptedError("Fine-tuning coordinator received signal {}".format(signum))

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, interrupted)
        context = _prepare_execution(registry, plan, selected_devices, min_free_gib)
        atomic_json(run_dir / "input_fingerprint.json", context["fingerprint"])
        atomic_json(run_dir / "frozen_existing.json", context["frozen"])
        state["status"] = "running"
        _execute_jobs(registry, run_dir, plan, assignment, context, state, write_state, min_free_gib)
        if not plan["diagnostic"]:
            for job_id, reused in context["reused"].items():
                checkpoint = _inside(registry.root, reused["checkpoint"])
                if file_digest(checkpoint) != reused["checkpoint_sha256"]:
                    raise RuntimeError("Reused VOC checkpoint changed: " + job_id)
                model_id = job_id.split("/", 1)[1]
                ready, reason = context["serial"]._checkpoint_ready(registry, "voc", model_id,
                    context["expected_commits"]["voc"][model_id], {})
                if not ready:
                    raise RuntimeError("Reused VOC checkpoint lost qualification: " + reason)
            for dataset, records in context["frozen"].items():
                context["dual"]._assert_frozen_existing_unchanged(context["serial"], registry, dataset, records, deep=True)
            if plan["models"] == plan["canonical_models"]:
                state["gates"]["voc"] = context["serial"]._gate_dataset(registry, "voc",
                    context["fingerprint"]["git_commit"], selected_devices[0], {},
                    expected_commits=context["expected_commits"]["voc"],
                )
        if any(row["status"] != "complete" for row in state["jobs"].values()):
            raise RuntimeError("Fine-tuning jobs are missing terminal acceptance")
        finish("complete")
    except BaseException as exc:
        state["failure"] = "{}: {}".format(type(exc).__name__, exc)
        finish("failed")
        raise
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return run_dir
