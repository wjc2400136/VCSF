"""Current-input evaluation of the registered COCO victim checkpoint pair."""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..registry import Registry
from .formal_parallel import normalize_execution_devices, validate_available_cuda_devices
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
)
from .training_state_pair import qualify_checkpoint_pair, _resolve_inside
from .training_state_transfer import write_training_state_reports
from .vcsf_final_training_state_contract import ids_digest
from .current_reproduction_inputs import (
    _read, _hash, _bind, _unchanged, _parameter_hash, _inside, verify_main_payloads,
)


PROTOCOL_ID = "current_training_state_transfer"






def _now():
    return datetime.now(timezone.utc).isoformat()




def build_plan(registry, methods=None, devices=None, max_images=None):
    spec = registry.protocols[PROTOCOL_ID]
    main = registry.protocols[spec["method_source_protocol"]]
    defaults = list(main["methods"])
    requested = defaults if methods is None else list(methods)
    if (not requested or any(not isinstance(method, str) for method in requested)
            or len(set(requested)) != len(requested)
            or any(method not in defaults for method in requested)):
        raise ValueError("Select unique methods from the current main comparison")
    selected = [method for method in defaults if method in requested]
    execution = normalize_execution_devices("cuda:0", devices)
    if (len(execution) not in spec["device_counts"]
            or any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) for value in execution)):
        raise ValueError("Select one or two distinct explicit CUDA devices")
    full = spec["full_images"]
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= full):
        raise ValueError("max_images must be a positive integer within the full split")
    for method in selected:
        if registry.compatibility_status(method, spec["sources"][0])["status"] != "native":
            raise ValueError("The registered training-state source does not support " + method)
    jobs = []
    for state_index, state in enumerate(spec["victim_states"]):
        for method in ["clean"] + selected:
            jobs.append(dict(
                index=len(jobs), training_state=state, record_type="clean" if method == "clean" else "attacked",
                source="clean" if method == "clean" else spec["sources"][0], attack=method,
                target=spec["targets"][0], worker_slot=state_index % len(execution),
                device=execution[state_index % len(execution)],
                parameters_sha256=None if method == "clean" else _parameter_hash(registry, method),
                budget_profile=None if method == "clean" else main.get("method_budget_profiles", {}).get(method, main["budget_profile"]),
            ))
    return dict(
        schema_version=1, protocol=PROTOCOL_ID, dataset=spec["datasets"][0], split=spec["split"],
        source=spec["sources"][0], target=spec["targets"][0], methods=selected,
        states=list(spec["victim_states"]), metrics=list(spec["metrics"]), devices=execution,
        full_images=full, selected_images=full if max_images is None else max_images,
        retained_images=spec["retained_images"], retained_selection_algorithm=spec["retained_selection_algorithm"],
        max_images=max_images, diagnostic_only=max_images is not None,
        complete_method_selection=selected == defaults, generation_jobs=0,
        inference_jobs=len(jobs), jobs=jobs, scientific_acceptance=False,
        historical_result_inheritance=False, project_wide_device_release_accepted=False,
    )










def verify_inputs(registry, plan, source_main, pair_manifest=None, pair_repository=None):
    inputs = verify_main_payloads(registry, plan, source_main)
    headers = inputs["headers"]
    repository = Path(pair_repository or registry.root).resolve()
    pair_view = SimpleNamespace(root=repository, training_state_transfer=registry.training_state_transfer)
    manifest = Path(pair_manifest).resolve() if pair_manifest is not None else repository / registry.training_state_transfer["artifacts"]["pair_manifest"]
    manifest = _inside(repository, str(manifest))
    qualification = qualify_checkpoint_pair(pair_view, manifest)
    required_tier = "strict_training_controlled_pair_eligible" if plan["max_images"] is None else "diagnostic_pair_eligible"
    if qualification.get(required_tier) is not True:
        raise ValueError("Victim checkpoint pair failed the required qualification tier: " + repr(qualification["issues"]))
    recipe = registry.training_state_transfer["pair_training"]
    invariants = qualification["paired_invariants"]
    if (invariants["pair_common_recipe_sha256"] != _hash({key: value for key, value in recipe.items() if key != "adversarial_branch"})
            or invariants["final_epoch"] != recipe["outer_epochs"]):
        raise ValueError("Victim pair differs from the registered continuation recipe")
    headers.append(_bind(manifest))
    checkpoints = {}
    for state in plan["states"]:
        state_manifest = _resolve_inside(repository, qualification["state_manifests"][state])
        state_data = _read(state_manifest)
        checkpoint = _resolve_inside(repository, state_data["checkpoint"])
        headers.append(_bind(state_manifest))
        checkpoints[state] = dict(path=str(checkpoint), sha256=qualification["checkpoint_sha256"][state], binding=_bind(checkpoint, digest=False))
    inputs.update(source_checkpoint_sha256=inputs["source_checkpoint_sha256"][plan["source"]],
        payloads=inputs["payloads"][plan["source"]], checkpoints=checkpoints,
        pair_manifest=str(manifest), pair_qualification=qualification)
    return inputs


def _check_inputs(inputs, method=None, state=None):
    for binding in inputs["headers"]:
        _unchanged(binding)
    for binding in inputs.get("source_checkpoint_bindings", {}).values():
        _unchanged(binding, rehash=False)
    if method is not None:
        for binding in inputs["payloads"][method]["images"]:
            _unchanged(binding, rehash=False)
    if state is not None:
        _unchanged(inputs["checkpoints"][state]["binding"], rehash=False)


def _verify_evaluation(record, plan, job, inputs):
    ids = inputs["selected_image_ids"]
    checkpoint = inputs["checkpoints"][job["training_state"]]
    required = dict(status="complete", dataset=plan["dataset"], split=plan["split"],
                    source=job["source"], attack=job["attack"], target=plan["target"], images=len(ids),
                    parameters_sha256=job["parameters_sha256"], evaluated_image_ids_sha256=ids_digest(ids),
                    expected_image_ids_sha256=ids_digest(ids), evaluated_image_ids_match_expected=True,
                    checkpoint=checkpoint["path"], checkpoint_sha256=checkpoint["sha256"])
    if any(record.get(key) != value for key, value in required.items()) or record.get("failures") != []:
        raise ValueError("Evaluation does not match its exact assigned input and checkpoint scope")
    metrics = record.get("metrics", {})
    if (set(metrics) != set(COCO_BBOX_METRICS)
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   or not (value == -1 or 0 <= value <= 1) for value in metrics.values())):
        raise ValueError("Evaluation must retain twelve finite raw COCO metrics")


def _result_records(registry, plan, job, inputs, output):
    from ..reporting.coco_subset import evaluate_bbox_subset

    record = _read(output / "metrics.json")
    _verify_evaluation(record, plan, job, inputs)
    archive = _read(output / "predictions_artifact.json")
    prediction_path = output / "predictions.json.gz"
    if file_digest(prediction_path) != archive["archive_sha256"]:
        raise ValueError("Prediction archive hash mismatch")
    with gzip.open(str(prediction_path), "rb") as stream:
        raw = stream.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != archive["uncompressed_sha256"] or digest != record["predictions_sha256"]:
        raise ValueError("Saved prediction bytes are not bound to evaluation metrics")
    predictions = json.loads(raw)
    annotation = _read(inputs["annotation"])
    ids = inputs["selected_image_ids"]
    replay = evaluate_bbox_subset(annotation, predictions, ids, ids)
    if any(abs(replay["metrics"][name] - record["metrics"][name]) > 1e-12 for name in COCO_BBOX_METRICS):
        raise ValueError("Official saved-prediction replay differs from evaluated metrics")
    common = dict(record, record_type=job["record_type"], training_state=job["training_state"],
                  protocol=PROTOCOL_ID, diagnostic_only=plan["diagnostic_only"],
                  payload_regenerated=False, historical_result_inheritance=False,
                  target_queries_during_attack_generation=0, target_gradients_during_attack_generation=0,
                  scientific_acceptance=False)
    full = dict(common, scope="fullval", full_coco_val_claim=not plan["diagnostic_only"])
    retained = inputs["retained_image_ids"]
    subset = evaluate_bbox_subset(annotation, predictions, ids, retained) if retained else None
    projection = {key: common[key] for key in (
        "dataset", "split", "source", "attack", "target", "checkpoint", "checkpoint_sha256",
        "parameters_sha256", "record_type", "training_state", "protocol", "diagnostic_only",
        "payload_regenerated", "historical_result_inheritance", "target_queries_during_attack_generation",
        "target_gradients_during_attack_generation", "scientific_acceptance",
    )}
    projection.update(scope="retained500", status="complete" if subset else "not_run",
        images=len(retained), metrics=subset["metrics"] if subset else {},
        evaluated_image_ids_sha256=ids_digest(retained), expected_image_ids_sha256=ids_digest(retained),
        evaluated_image_ids_match_expected=True if subset else None, full_coco_val_claim=False,
        retained500_scope_complete=len(retained) == plan["retained_images"],
        projection_source_evaluation=dict(scope="fullval", images=len(ids),
            metrics_sha256=file_digest(output / "metrics.json"),
            predictions_sha256=digest, archive_sha256=archive["archive_sha256"]),
        projection_model_calls=0)
    atomic_json(output / "retained_projection.json", projection)
    if subset:
        from ..reporting.coco_summary import write_coco_bbox_summary

        write_coco_bbox_summary([subset["metrics"][name] for name in COCO_BBOX_METRICS],
            output / "retained_projection", dataset=plan["dataset"], target=plan["target"],
            source=job["source"], attack=job["attack"])
    return [full, projection]


def _worker(request_path, expected_hash):
    request_path = Path(request_path).resolve()
    if file_digest(request_path) != expected_hash:
        raise ValueError("Worker request changed")
    request = _read(request_path)
    owner = install_branch_scope(request["coordinator"])
    directory = request_path.parent
    started = time.time()
    try:
        registry = Registry(Path(request["source_root"]))
        from ..attacks.vcsf_public import verify_public_identity
        from .vcsf_gpu_context_owner import RegisteredGpuOwner
        from .evaluate import run_evaluation

        if verify_public_identity(registry.root)["source_freeze_sha256"] != request["source_freeze_sha256"]:
            raise ValueError("Worker executable source changed")
        gpu = RegisteredGpuOwner(request)
        atomic_json(directory / "gpu_context_registration.json", gpu.receipt)
        results = []
        plan, inputs = request["plan"], request["inputs"]
        for job in request["jobs"]:
            state, method = job["training_state"], job["attack"]
            _check_inputs(inputs, None if method == "clean" else method, state)
            def progress(completed, total, image_id, status):
                if completed == 1 or completed % 25 == 0 or completed == total:
                    gpu.verify()
                    atomic_json(directory / "progress.json", dict(job_index=job["index"],
                        training_state=state, attack=method, completed_images=completed,
                        total_images=total, image_id=image_id, status=status, updated_at_utc=_now(), owner=owner))
            output = run_evaluation(
                registry, plan["dataset"], plan["target"], split=plan["split"],
                adversarial_run=None if method == "clean" else Path(inputs["payloads"][method]["root"]),
                output_dir=Path(request["output"]) / "evaluations" / state / method,
                image_ids=inputs["selected_image_ids"], device="cuda:0", download_weights=False,
                keep_going=False, prediction_archive="gzip", target_checkpoint=Path(inputs["checkpoints"][state]["path"]),
                progress_callback=progress, **request["visualizations"],
            )
            _check_inputs(inputs, None if method == "clean" else method, state)
            rows = _result_records(registry, plan, job, inputs, output)
            atomic_json(directory / ("completed_{:06d}.json".format(job["index"])), rows)
            results.extend(rows)
            gpu.verify()
        if verify_public_identity(registry.root)["source_freeze_sha256"] != request["source_freeze_sha256"]:
            raise ValueError("Worker source changed during execution")
        atomic_json(directory / "terminal.json", dict(status="complete_pending_independent_acceptance",
            request_sha256=expected_hash, owner=owner, records=len(results), scientific_acceptance=False,
            completed_files=[dict(file="completed_{:06d}.json".format(job["index"]),
                sha256=file_digest(directory / "completed_{:06d}.json".format(job["index"]))) for job in request["jobs"]],
            completed_at_utc=_now(), seconds=time.time() - started))
    except BaseException as exc:
        atomic_json(directory / "terminal.json", dict(status="failed", request_sha256=expected_hash,
            owner=owner, error=repr(exc), completed_at_utc=_now(), seconds=time.time() - started))
        raise


def _key(record):
    return record["training_state"], record["attack"], record["scope"]


def _collect(run_dir, plan, current):
    lookup = {_key(row): row for row in current}
    for job in plan["jobs"]:
        path = run_dir / "workers" / str(job["worker_slot"]) / ("completed_{:06d}.json".format(job["index"]))
        if not path.exists():
            continue
        rows = _read(path)
        if (len(rows) != 2 or [row.get("scope") for row in rows] != ["fullval", "retained500"]
                or any(any(row.get(field) != job[field] for field in ("attack", "training_state", "source", "target", "record_type")) for row in rows)):
            raise ValueError("Worker published missing, duplicate or foreign result records")
        for row in rows:
            lookup[_key(row)] = row
    return [lookup[(job["training_state"], job["attack"], scope)]
            for job in plan["jobs"] for scope in ("fullval", "retained500")]


def _reports(registry, run_dir, plan, records):
    for scope in ("fullval", "retained500"):
        label = ("full-5000" if scope == "fullval" else "retained-500")
        if plan["diagnostic_only"]:
            label = ("selected-{} diagnostic".format(plan["selected_images"])
                     if scope == "fullval" else "retained-500 intersection diagnostic")
        write_training_state_reports(
            [row for row in records if row["scope"] == scope], registry,
            run_dir / "reports" / scope, plan["methods"], scope_label=label,
            full_coco_val_claim=scope == "fullval" and not plan["diagnostic_only"],
            include_failure_markers=True,
            report_scope=dict(dataset="coco", split=plan["split"],
                              max_images=plan["max_images"], diagnostic_only=plan["diagnostic_only"]),
        )


def run(registry, *, source_main=None, pair_manifest=None, pair_repository=None,
        methods=None, devices=None, max_images=None, output=None, plan_only=False,
        save_visualizations=False, visualization_score_threshold=0.3,
        visualization_max_images=20, visualization_max_detections=100):
    plan = build_plan(registry, methods, devices, max_images)
    if not plan_only:
        if sys.platform != "linux":
            raise RuntimeError("GPU training-state execution requires Linux process ownership support")
        if source_main is None:
            raise ValueError("Provide --source-main with current main reproduction payloads")
        validate_available_cuda_devices(plan["devices"], include_single=True)
    if save_visualizations and (not 0 <= visualization_score_threshold <= 1
            or visualization_max_images <= 0 or visualization_max_detections <= 0):
        raise ValueError("Invalid optional prediction-visualization settings")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output or registry.root / "outputs/experiments" / PROTOCOL_ID / stamp).resolve()
    if not plan_only:
        protected = [Path(source_main).resolve(), registry.dataset(plan["dataset"]).root.resolve()]
        pair_path = Path(pair_manifest).resolve() if pair_manifest is not None else Path(pair_repository or registry.root).resolve() / registry.training_state_transfer["artifacts"]["pair_manifest"]
        protected.append(pair_path.parent.resolve())
        if any(run_dir == path or path in run_dir.parents or run_dir in path.parents for path in protected):
            raise ValueError("Output overlaps an immutable input root")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "plan.json", plan)
    records = [dict(job, scope=scope, status="planned", metrics={})
               for job in plan["jobs"] for scope in ("fullval", "retained500")]
    atomic_json(run_dir / "records.json", records)
    handles = []
    success, failure, collection_error, reports_error = False, None, None, None
    cleanup_error = None
    if plan_only:
        _reports(registry, run_dir, plan, records)
        atomic_json(run_dir / "summary.json", dict(status="planned", inference_jobs=plan["inference_jobs"],
            generation_jobs=0, scientific_acceptance=False, project_wide_device_release_accepted=False))
        atomic_json(run_dir / "terminal.json", dict(status="planned", execution_admission=False, scientific_acceptance=False))
        return run_dir
    try:
        inputs = verify_inputs(registry, plan, source_main, pair_manifest, pair_repository)
        atomic_json(run_dir / "input_binding.json", inputs)
        from ..attacks.vcsf_public import verify_public_identity
        from .vcsf_structure_workers import physical_gpus, gpu_reservations, worker_environment

        identity = verify_public_identity(registry.root)
        coordinator = process_identity(os.getpid())
        atomic_json(run_dir / "coordinator.json", coordinator)
        uuids = physical_gpus(plan["devices"])
        with gpu_reservations(uuids) as reservations:
            try:
                for slot, uuid in enumerate(uuids):
                    directory = run_dir / "workers" / str(slot)
                    directory.mkdir(parents=True, exist_ok=False)
                    fd = reservations[uuid]
                    stat = os.fstat(fd)
                    request = dict(source_root=str(registry.root), output=str(run_dir), coordinator=coordinator,
                        source_freeze_sha256=identity["source_freeze_sha256"], gpu_uuid=uuid,
                        reservation_fd=fd, reservation_identity=[stat.st_dev, stat.st_ino],
                        plan=plan, inputs=inputs, jobs=[job for job in plan["jobs"] if job["worker_slot"] == slot],
                        visualizations=dict(save_visualizations=save_visualizations,
                            visualization_score_threshold=visualization_score_threshold,
                            visualization_max_images=visualization_max_images,
                            visualization_max_detections=visualization_max_detections))
                    request_path = directory / "request.json"
                    atomic_json(request_path, request)
                    request_hash = file_digest(request_path)
                    environment = worker_environment(uuid)
                    environment["PYTHONPATH"] = str(registry.root / "src") + os.pathsep + str(registry.root)
                    environment["PYTHONDONTWRITEBYTECODE"] = "1"
                    with (directory / "worker.log").open("xb") as log:
                        child = subprocess.Popen([sys.executable, "-m", __name__, str(request_path), request_hash],
                            cwd=str(registry.root), env=environment, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(fd,))
                    birth = dict(pid=child.pid, parent_pid=os.getpid(), start_ticks=None,
                                 process_group=child.pid, session_id=child.pid)
                    handles.append([child, birth, directory, request_hash])
                    birth = process_identity(child.pid)
                    handles[-1][1] = birth
                    atomic_json(directory / "process.json", birth)
                while True:
                    records = _collect(run_dir, plan, records)
                    atomic_json(run_dir / "records.json", records)
                    exits = [peek_exit_code(item[0]) for item in handles]
                    atomic_json(run_dir / "execution_state.json", dict(status="running", updated_at_utc=_now(),
                        workers=[dict(process=item[1], exit_code=code,
                            progress=_read(item[2] / "progress.json") if (item[2] / "progress.json").exists() else None)
                            for item, code in zip(handles, exits)]))
                    if any(code not in (None, 0) for code in exits):
                        raise RuntimeError("Training-state worker failed; preserve original logs and completed prefix")
                    if all(code == 0 for code in exits):
                        break
                    time.sleep(5)
                for slot, (child, birth, directory, request_hash) in enumerate(handles):
                    terminal = _read(directory / "terminal.json")
                    expected_owner = dict(worker_pid=child.pid, worker_start_ticks=birth["start_ticks"],
                        process_group=child.pid, coordinator_pid=coordinator["pid"], coordinator_start_ticks=coordinator["start_ticks"])
                    assigned = [job for job in plan["jobs"] if job["worker_slot"] == slot]
                    files = [dict(file="completed_{:06d}.json".format(job["index"]),
                        sha256=file_digest(directory / "completed_{:06d}.json".format(job["index"]))) for job in assigned]
                    if (terminal.get("status") != "complete_pending_independent_acceptance"
                            or terminal.get("request_sha256") != request_hash or terminal.get("owner") != expected_owner
                            or terminal.get("records") != 2 * len(assigned) or terminal.get("completed_files") != files):
                        raise ValueError("Worker terminal does not bind to its exact owner, request and completed jobs")
                records = _collect(run_dir, plan, records)
                if any(row["status"] != "complete" for row in records if row["scope"] == "fullval"):
                    raise ValueError("Incomplete full-scope evaluation panel")
                if inputs["retained_image_ids"] and any(row["status"] != "complete" for row in records if row["scope"] == "retained500"):
                    raise ValueError("Incomplete retained-scope projection panel")
            finally:
                pending_failure = sys.exc_info()[1]
                cleanup = []
                for child, birth, directory, request_hash in reversed(handles):
                    try:
                        cleanup.append(close_branch_process(child, birth))
                    except BaseException as exc:
                        cleanup.append(dict(status="failed", pid=child.pid, error=repr(exc)))
                atomic_json(run_dir / "cleanup.json", cleanup)
                if any(row["status"] != "verified" for row in cleanup):
                    cleanup_error = RuntimeError("Worker containment cleanup failed")
                    if pending_failure is None:
                        raise cleanup_error
        success = True
    except BaseException as exc:
        failure = exc
        try:
            records = _collect(run_dir, plan, records)
        except BaseException as collection_exc:
            collection_error = repr(collection_exc)
        records = [dict(row, status="failed", reason=repr(exc)) if row["status"] == "planned" else row for row in records]
    finally:
        atomic_json(run_dir / "records.json", records)
        try:
            _reports(registry, run_dir, plan, records)
        except BaseException as exc:
            reports_error = exc
            success = False
        status = "complete_pending_independent_acceptance" if success else "failed"
        atomic_json(run_dir / "summary.json", dict(status=status,
            inference_jobs=plan["inference_jobs"], generation_jobs=0,
            completed_metric_records=sum(row["status"] == "complete" for row in records),
            failed_metric_records=sum(row["status"] == "failed" for row in records),
            diagnostic_only=plan["diagnostic_only"], scientific_acceptance=False,
            project_wide_device_release_accepted=False))
        atomic_json(run_dir / "terminal.json", dict(status=status, scientific_acceptance=False,
            error=repr(failure) if failure is not None else None, collection_error=collection_error,
            cleanup_error=repr(cleanup_error) if cleanup_error is not None else None,
            reports_error=repr(reports_error) if reports_error is not None else None, completed_at_utc=_now()))
        atomic_json(run_dir / "execution_state.json", dict(status=status, updated_at_utc=_now(),
            terminal=dict(file="terminal.json", sha256=file_digest(run_dir / "terminal.json")),
            cleanup=(dict(file="cleanup.json", sha256=file_digest(run_dir / "cleanup.json"))
                     if (run_dir / "cleanup.json").exists() else None),
            workers=[dict(process=item[1], exit_code=item[0].returncode) for item in handles],
            error=repr(failure) if failure is not None else None,
            cleanup_error=repr(cleanup_error) if cleanup_error is not None else None,
            reports_error=repr(reports_error) if reports_error is not None else None,
            terminal_authoritative=True, scientific_acceptance=False,
            project_wide_device_release_accepted=False))
    if failure is not None:
        raise failure
    if reports_error is not None:
        raise reports_error
    return run_dir


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
