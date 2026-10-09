"""Current-main payload reuse under fixed oblivious victim preprocessing."""
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

from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..registry import Registry
from ..reporting.preprocessing_cost import generation_cost_metadata, record_cost_metadata
from .current_reproduction_inputs import (
    _read, _hash, _bind, _unchanged, _parameter_hash, _inside, verify_main_payloads,
)
from .formal_parallel import normalize_execution_devices, validate_available_cuda_devices
from .preprocessing_defense import _materialize_view, _subset_annotation
from .training_pair_processes import (
    close_branch_process, install_branch_scope, peek_exit_code, process_identity,
)
from .vcsf_final_training_state_contract import ids_digest


PROTOCOL_ID = "current_oblivious_preprocessing_transfer"
ADAPTIVE_PROTOCOL_ID = "current_adaptive_preprocessing_transfer"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _selection(requested, canonical, label):
    selected = list(canonical) if requested is None else list(requested)
    if (not selected or any(type(value) is not str for value in selected)
            or len(set(selected)) != len(selected) or not set(selected) <= set(canonical)):
        raise ValueError("Select unique registered " + label)
    return [value for value in canonical if value in selected]


def build_plan(registry, methods=None, sources=None, targets=None, defenses=None,
               devices=None, max_images=None, adaptive=False):
    if type(adaptive) is not bool:
        raise ValueError("Adaptive mode must be an explicit boolean")
    protocol = ADAPTIVE_PROTOCOL_ID if adaptive else PROTOCOL_ID
    spec = registry.protocols[protocol]
    main = registry.protocols[spec["method_source_protocol"]]
    methods = _selection(methods, main["methods"], "current-main methods")
    sources = _selection(sources, spec["sources"], "Common-2 sources")
    targets = _selection(targets, registry.paper_order, "targets")
    defenses = _selection(defenses, registry.preprocessing_defense_order, "preprocessing variants")
    execution = normalize_execution_devices("cuda:0", devices)
    if (len(execution) not in spec["device_counts"]
            or any(not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) for value in execution)):
        raise ValueError("Select one or two distinct explicit CUDA devices")
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= spec["retained_images"]):
        raise ValueError("max_images must be a positive integer within retained-500")
    for source in sources:
        for method in methods:
            if registry.compatibility_status(method, source)["status"] != "native":
                raise ValueError("Current Common-2 method/source pair is not native")
    parameter_hashes = {method: _parameter_hash(registry, method) for method in methods}
    groups, jobs = [], []
    origins = [("clean", "clean", "clean")]
    origins += [("attacked", source, method) for source in sources for method in methods]
    for record_type, source, method in origins:
        for defense in defenses:
            group = dict(index=len(groups), record_type=record_type, source=source,
                attack=method, defense=defense, targets=targets,
                parameters_sha256=None if method == "clean" else parameter_hashes[method],
                budget_profile=None if method == "clean" else main.get("method_budget_profiles", {}).get(method, main["budget_profile"]),
                worker_slot=len(groups) % len(execution))
            group["device"] = execution[group["worker_slot"]]
            if adaptive:
                group["adaptive_generation"] = (record_type != "clean" and not
                    (defense == "identity" and method in spec["identity_payload_reuse_methods"]))
                group["identity_payload_reused"] = (record_type != "clean" and not group["adaptive_generation"])
            groups.append(group)
            for target in targets:
                jobs.append(dict(group, index=len(jobs), group_index=group["index"], target=target))
    plan = dict(schema_version=1, protocol=protocol, dataset="coco", split=spec["split"],
        methods=methods, sources=sources, targets=targets, defenses=defenses,
        defense_parameters={key: dict(registry.preprocessing_defenses[key]) for key in defenses},
        transform_implementation=dict(registry.preprocessing_defense_policy),
        threat_model=dict({key: value for key, value in registry.preprocessing_defense_threat_model.items()
            if key != "note"}, source_protocol=spec["method_source_protocol"]),
        devices=execution, full_images=spec["full_images"], retained_images=spec["retained_images"],
        retained_selection_algorithm=spec["retained_selection_algorithm"],
        selected_images=spec["retained_images"] if max_images is None else max_images,
        max_images=max_images, diagnostic_only=max_images is not None,
        full_registered_selection=(methods == main["methods"] and sources == spec["sources"]
            and targets == registry.paper_order and defenses == registry.preprocessing_defense_order),
        generation_jobs=0, training_jobs=0, clean_evaluations=len(defenses) * len(targets),
        attack_evaluations=len(sources) * len(methods) * len(defenses) * len(targets),
        inference_jobs=len(jobs), transform_groups=len(groups), groups=groups, jobs=jobs,
        full_coco_val_claim=False, scientific_acceptance=False,
        historical_result_inheritance=False, project_wide_device_release_accepted=False)
    if adaptive:
        policy = dict(registry.adaptive_preprocessing_policy)
        plan.update(adaptive=True, adaptive_policy=policy, seed=spec["seed"],
            seed_schedule=spec["seed_schedule"],
            threat_model=dict({key: value for key, value in policy["threat_model"].items() if key != "note"},
                source_protocol=spec["method_source_protocol"]),
            identity_payload_reuse_methods=list(spec["identity_payload_reuse_methods"]),
            generation_jobs=sum(group["adaptive_generation"] for group in groups))
    return plan


def verify_inputs(registry, plan, source_main):
    index = CocoIndex(registry.dataset("coco"), plan["split"])
    ids = [row["id"] for row in index.images]
    if (len(ids) != plan["full_images"] or any(type(value) is not int for value in ids)
            or ids != sorted(set(ids))):
        raise ValueError("Canonical COCO validation population is incomplete or duplicated")
    count = plan["retained_images"]
    retained = [ids[((2 * position + 1) * len(ids)) // (2 * count)] for position in range(count)]
    selected = retained[:plan["selected_images"]]
    inputs = verify_main_payloads(registry, plan, source_main, selected)
    clean_root = (registry.dataset("coco").root / index.split.image_prefix).resolve()
    by_id = {row["id"]: row for row in index.images}
    clean = [_bind(_inside(clean_root, str(index.image_path(by_id[value])))) for value in selected]
    checkpoints = {}
    for target in plan["targets"]:
        path = (registry.root / "checkpoints/coco" / registry.model(target).checkpoint_filename).resolve()
        bound = _bind(path)
        checkpoints[target] = dict(path=str(path), sha256=bound["sha256"], binding=bound)
    inputs.update(clean=dict(root=str(clean_root), images=clean,
        annotation=_subset_annotation(index.payload, selected)), checkpoints=checkpoints,
        selected_image_ids_sha256=ids_digest(selected), full_coco_val_claim=False)
    if plan.get("adaptive"):
        from .current_adaptive_payload import seed_mapping

        for group in plan["groups"]:
            if group["record_type"] != "clean":
                seed_mapping(plan, group, inputs)
    return inputs


def _check_inputs(inputs, group=None, target=None, rehash_headers=False):
    for binding in inputs["headers"]:
        _unchanged(binding, rehash=rehash_headers)
    for binding in inputs.get("source_checkpoint_bindings", {}).values():
        _unchanged(binding, rehash=False)
    if group is not None:
        origin = (inputs["clean"] if group["record_type"] == "clean" or group.get("adaptive_generation")
                  else inputs["payloads"][group["source"]][group["attack"]])
        for binding in origin["images"]:
            _unchanged(binding, rehash=False)
    if target is not None:
        _unchanged(inputs["checkpoints"][target]["binding"], rehash=False)


def materialize_view(registry, plan, group, inputs, output, generation_progress=None):
    if group["record_type"] == "clean":
        annotation = inputs["clean"]["annotation"]
        root = Path(inputs["clean"]["root"])
        original = inputs["clean"]
    else:
        if group.get("adaptive_generation"):
            from .current_adaptive_payload import prepare_payload

            original = prepare_payload(registry, plan, group, inputs, output, generation_progress)
        else:
            original = inputs["payloads"][group["source"]][group["attack"]]
        annotation = _subset_annotation(_read(original["annotation"]), inputs["selected_image_ids"])
        root = Path(original["image_root"])
    manifest = output / "view_manifests" / ("{:06d}.json".format(group["index"]))
    view = _materialize_view(registry, output, output / "scratch" / ("{:06d}".format(group["index"])),
        manifest, annotation, root, group["source"], group["attack"], group["defense"],
        plan["defense_parameters"][group["defense"]], protocol_id=plan["protocol"],
        threat_model=plan["threat_model"], transform_implementation=plan["transform_implementation"])
    run_path = view / "run.json"
    run = _read(run_path)
    run.update(generation_cost_metadata(group, original))
    run.update(parameters_sha256=group["parameters_sha256"], budget_profile=group["budget_profile"],
        evaluated_image_ids_sha256=inputs["selected_image_ids_sha256"],
        retained500_image_ids_sha256=inputs["retained500_image_ids_sha256"],
        diagnostic_only=plan["diagnostic_only"], full_coco_val_claim=False,
        payload_regenerated=bool(group.get("adaptive_generation")),
        adaptive_source_pipeline=bool(plan.get("adaptive")),
        original_seed_mapping=original.get("seed_mapping", []))
    atomic_json(run_path, run)
    materialized = _read(manifest)
    entry = dict(view=str(view), run_sha256=file_digest(run_path),
        annotation_sha256=file_digest(view / "annotations.json"), manifest=str(manifest),
        manifest_sha256=file_digest(manifest), original_images=original["images"],
        transformed_images=[_bind(_inside(view, row["output_file"]), digest=False)
                            for row in materialized["outputs"]])
    if group.get("adaptive_generation"):
        entry["generated_payload"] = original
    return entry


def verify_view(entry, image_ids, rehash=True):
    if "generated_payload" in entry:
        for binding in entry["generated_payload"]["headers"] + entry["generated_payload"]["images"]:
            _unchanged(binding, rehash=rehash)
    view = Path(entry["view"])
    for path, digest in [(view / "run.json", entry["run_sha256"]),
            (view / "annotations.json", entry["annotation_sha256"]),
            (Path(entry["manifest"]), entry["manifest_sha256"])]:
        if file_digest(path) != digest:
            raise ValueError("Prepared view metadata changed")
    run, annotation, manifest = _read(view / "run.json"), _read(view / "annotations.json"), _read(entry["manifest"])
    if [row["id"] for row in annotation["images"]] != image_ids or run["evaluated_image_ids_sha256"] != ids_digest(image_ids):
        raise ValueError("Prepared view image population changed")
    root = Path(run["image_root"])
    root = (root if root.is_absolute() else view / root).resolve()
    if manifest["materialization"] == "reference_only":
        bindings = {row["file"]: row for row in entry["original_images"]}
        for image in annotation["images"]:
            path = _inside(root, image["file_name"])
            _unchanged(bindings[str(path)], rehash=rehash)
    else:
        rows = manifest["outputs"]
        mapping = {row["image_id"]: row for row in rows}
        bindings = {row["file"]: row for row in entry["transformed_images"]}
        if len(mapping) != len(rows) or set(mapping) != set(image_ids):
            raise ValueError("Prepared view manifest coverage changed")
        for image in annotation["images"]:
            row = mapping[image["id"]]
            path = _inside(view, row["output_file"])
            if (path != _inside(root, image["file_name"]) or path.stat().st_size != row["output_bytes"]
                    or (rehash and file_digest(path) != row["output_sha256"])):
                raise ValueError("Prepared view pixels changed")
            _unchanged(bindings[str(path)], rehash=False)


def verify_evaluation(record, plan, job, inputs, entry):
    checkpoint = inputs["checkpoints"][job["target"]]
    expected = dict(status="complete", dataset="coco", split=plan["split"], source=job["source"],
        attack=job["attack"], target=job["target"], images=len(inputs["selected_image_ids"]),
        parameters_sha256=job["parameters_sha256"], checkpoint=checkpoint["path"],
        checkpoint_sha256=checkpoint["sha256"], adversarial_run=entry["view"],
        annotation=str(Path(entry["view"]) / "annotations.json"),
        evaluated_image_ids_sha256=inputs["selected_image_ids_sha256"],
        expected_image_ids_sha256=inputs["selected_image_ids_sha256"], evaluated_image_ids_match_expected=True)
    if any(record.get(key) != value for key, value in expected.items()) or record.get("failures") != []:
        raise ValueError("Preprocessing evaluation differs from its assigned view, IDs or checkpoint")
    metrics = record.get("metrics", {})
    if (set(metrics) != set(COCO_BBOX_METRICS) or any(type(value) not in (int, float)
            or not math.isfinite(value) or not (value == -1 or 0 <= value <= 1) for value in metrics.values())):
        raise ValueError("Preprocessing evaluation must preserve twelve finite raw COCO metrics")


def result_record(plan, job, inputs, entry, output):
    from ..reporting.coco_subset import evaluate_bbox_subset

    verify_view(entry, inputs["selected_image_ids"], rehash=False)
    record = _read(output / "metrics.json")
    verify_evaluation(record, plan, job, inputs, entry)
    archive = _read(output / "predictions_artifact.json")
    prediction_path = output / "predictions.json.gz"
    if file_digest(prediction_path) != archive["archive_sha256"]:
        raise ValueError("Prediction archive hash mismatch")
    with gzip.open(str(prediction_path), "rb") as stream:
        raw = stream.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != archive["uncompressed_sha256"] or digest != record["predictions_sha256"]:
        raise ValueError("Saved prediction bytes are not bound to evaluation metrics")
    ids = inputs["selected_image_ids"]
    replay = evaluate_bbox_subset(_read(Path(entry["view"]) / "annotations.json"), json.loads(raw), ids, ids)
    if any(abs(replay["metrics"][name] - record["metrics"][name]) > 1e-12 for name in COCO_BBOX_METRICS):
        raise ValueError("Official saved-prediction replay differs from evaluated metrics")
    # Prepared-view accounting must not replace the evaluator's identity.
    view_run = _read(Path(entry["view"]) / "run.json")
    accounting = {key: value for key, value in record_cost_metadata(view_run).items()
                  if key.startswith("generation_") or key in (
                      "gradient_evaluations_per_image", "budget_profile_definition",
                      "baseline_profile_definition", "auxiliary_passes",
                      "attack_actual_gradient_evaluations_total",
                      "attack_actual_gradient_evaluations_mean", "calibrated_whole_detector_BE")}
    return dict(record, **accounting, job_index=job["index"], group_index=job["group_index"],
        record_type=job["record_type"], defense=job["defense"], protocol=plan["protocol"],
        defense_parameters=plan["defense_parameters"][job["defense"]],
        budget_profile=job["budget_profile"], diagnostic_only=plan["diagnostic_only"],
        retained500_scope_complete=len(ids) == plan["retained_images"],
        retained500_image_ids_sha256=inputs["retained500_image_ids_sha256"],
        prepared_view={key: entry[key] for key in ("view", "run_sha256", "annotation_sha256", "manifest", "manifest_sha256")},
        full_coco_val_claim=False, payload_regenerated=bool(job.get("adaptive_generation")),
        adaptive_source_pipeline=bool(plan.get("adaptive")),
        identity_payload_reused=bool(job.get("identity_payload_reused")),
        historical_result_inheritance=False, target_queries_during_attack_generation=0,
        target_gradients_during_attack_generation=0, official_replay_model_calls=0,
        scientific_acceptance=False)


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
        plan, inputs = request["plan"], request["inputs"]
        _check_inputs(inputs, rehash_headers=True)
        completed = []
        for group in request["groups"]:
            _check_inputs(inputs, group)
            jobs = [job for job in request["jobs"] if job["group_index"] == group["index"]]
            def generation_progress(count, total, image_id, status):
                if count == 1 or count % 25 == 0 or count == total:
                    gpu.verify()
                    atomic_json(directory / "progress.json", dict(job_index=jobs[0]["index"],
                        group_index=group["index"], source=group["source"], attack=group["attack"],
                        defense=group["defense"], target=None, phase="adaptive_generation", completed_images=count,
                        total_images=total, image_id=image_id, status=status, updated_at_utc=_now(), owner=owner))
            if group.get("adaptive_generation"):
                generation_progress(0, len(inputs["selected_image_ids"]), None, "starting")
            entry = materialize_view(registry, plan, group, inputs, Path(request["output"]),
                **({"generation_progress": generation_progress} if group.get("adaptive_generation") else {}))
            view_binding_path = directory / ("view_{:06d}.json".format(group["index"]))
            atomic_json(view_binding_path, entry)
            view_binding = dict(file=str(view_binding_path), sha256=file_digest(view_binding_path))
            verify_view(entry, inputs["selected_image_ids"])
            for job in jobs:
                def progress(count, total, image_id, status):
                    if count == 1 or count % 25 == 0 or count == total:
                        gpu.verify()
                        atomic_json(directory / "progress.json", dict(job_index=job["index"],
                            group_index=group["index"], source=group["source"], attack=group["attack"],
                            defense=group["defense"], target=job["target"], phase="target_evaluation", completed_images=count,
                            total_images=total, image_id=image_id, status=status, updated_at_utc=_now(), owner=owner))
                progress(0, len(inputs["selected_image_ids"]), None, "starting")
                _check_inputs(inputs, group, job["target"])
                output = run_evaluation(
                    registry, "coco", job["target"], split=plan["split"],
                    adversarial_run=Path(entry["view"]),
                    output_dir=Path(request["output"]) / "evaluations" / ("{:06d}".format(group["index"])) / job["target"],
                    image_ids=inputs["selected_image_ids"], device="cuda:0", download_weights=False,
                    keep_going=False, prediction_archive="gzip",
                    report_scope=dict(diagnostic_only=plan["diagnostic_only"],
                                      max_images=plan["max_images"], protocol=plan["protocol"]),
                    target_checkpoint=Path(inputs["checkpoints"][job["target"]]["path"]),
                    progress_callback=progress, **request["visualizations"])
                _check_inputs(inputs, group, job["target"])
                row = result_record(plan, job, inputs, entry, output)
                row["prepared_view_binding"] = view_binding
                path = directory / ("completed_{:06d}.json".format(job["index"]))
                atomic_json(path, row)
                completed.append(dict(file=path.name, sha256=file_digest(path)))
                gpu.verify()
            verify_view(entry, inputs["selected_image_ids"])
            atomic_json(directory / ("group_{:06d}.json".format(group["index"])),
                dict(status="complete_pending_independent_acceptance", group=group, prepared_view_binding=view_binding,
                    completed_jobs=[job["index"] for job in jobs], preserved_payload=True,
                    scientific_acceptance=False))
        _check_inputs(inputs, rehash_headers=True)
        if verify_public_identity(registry.root)["source_freeze_sha256"] != request["source_freeze_sha256"]:
            raise ValueError("Worker source changed during execution")
        atomic_json(directory / "terminal.json", dict(status="complete_pending_independent_acceptance",
            request_sha256=expected_hash, owner=owner, records=len(completed), completed_files=completed,
            scientific_acceptance=False, completed_at_utc=_now(), seconds=time.time() - started))
    except BaseException as exc:
        atomic_json(directory / "terminal.json", dict(status="failed", request_sha256=expected_hash,
            owner=owner, error=repr(exc), completed_at_utc=_now(), seconds=time.time() - started))
        raise


def _collect(run_dir, plan, current):
    lookup = {row["job_index"]: row for row in current}
    for job in plan["jobs"]:
        path = run_dir / "workers" / str(job["worker_slot"]) / ("completed_{:06d}.json".format(job["index"]))
        if not path.exists():
            continue
        row = _read(path)
        required = dict(job_index=job["index"], group_index=job["group_index"], status="complete",
            protocol=plan["protocol"], dataset="coco", split=plan["split"],
            **{field: job[field] for field in ("attack", "source", "target", "defense", "record_type", "parameters_sha256", "budget_profile")})
        if any(row.get(field) != value for field, value in required.items()):
            raise ValueError("Worker published a foreign or incomplete preprocessing result")
        lookup[job["index"]] = row
    return [lookup[job["index"]] for job in plan["jobs"]]


def _reports(registry, run_dir, plan, records):
    from ..reporting.latex import write_transfer_reports

    display_rows = {}
    for defense in plan["defenses"]:
        rows = [dict(row, metrics={key: value if value != -1 else None
            for key, value in row.get("metrics", {}).items()}) for row in records if row["defense"] == defense]
        display_rows[defense] = rows
        for metric in COCO_BBOX_METRICS:
            label = "selected-{} diagnostic".format(plan["selected_images"]) if plan["diagnostic_only"] else "fixed retained-500"
            write_transfer_reports(rows, registry, run_dir / "reports" / defense, "coco", metric=metric,
                source_ids=plan["sources"], target_ids=plan["targets"], method_ids=plan["methods"],
                preserve_method_order=True, require_complete_panel=True, include_failure_markers=True,
                caption="{} on COCO {} under {} {} preprocessing. Values are 100 times raw JSON metrics.".format(
                    metric, label, defense, "adaptive source-pipeline BPDA" if plan.get("adaptive") else "oblivious victim"),
                label="tab:{}_{}_{}".format(plan["protocol"] if plan.get("adaptive") else "current_preprocessing", defense, metric))
    return display_rows


def run(registry, *, source_main=None, methods=None, sources=None, targets=None,
        defenses=None, devices=None, max_images=None, output=None, plan_only=False,
        save_visualizations=False, visualization_score_threshold=0.3,
        visualization_max_images=20, visualization_max_detections=100, adaptive=False):
    plan = build_plan(registry, methods, sources, targets, defenses, devices, max_images, adaptive=adaptive)
    if not plan_only:
        if sys.platform != "linux":
            raise RuntimeError("GPU preprocessing execution requires Linux process ownership support")
        if source_main is None:
            raise ValueError("Provide --source-main with preserved current-main payloads")
        validate_available_cuda_devices(plan["devices"], include_single=True)
    if save_visualizations and (not 0 <= visualization_score_threshold <= 1
            or visualization_max_images <= 0 or visualization_max_detections <= 0):
        raise ValueError("Invalid optional prediction-visualization settings")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(output or registry.root / "outputs/experiments" / plan["protocol"] / stamp).resolve()
    if not plan_only:
        protected = [Path(source_main).resolve(), registry.dataset("coco").root.resolve(),
                     (registry.root / "checkpoints").resolve()]
        if any(run_dir == path or path in run_dir.parents or run_dir in path.parents for path in protected):
            raise ValueError("Output overlaps an immutable input root")
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "plan.json", plan)
    records = [dict(job, job_index=job["index"], dataset="coco", split=plan["split"],
        protocol=plan["protocol"], status="planned", metrics={}, diagnostic_only=plan["diagnostic_only"])
        for job in plan["jobs"]]
    atomic_json(run_dir / "records.json", records)
    handles = []
    success, failure, collection_error, reports_error = False, None, None, None
    if plan_only:
        _reports(registry, run_dir, plan, records)
        atomic_json(run_dir / "summary.json", dict(status="planned", inference_jobs=plan["inference_jobs"],
            transform_groups=plan["transform_groups"], generation_jobs=plan["generation_jobs"], scientific_acceptance=False,
            project_wide_device_release_accepted=False))
        atomic_json(run_dir / "terminal.json", dict(status="planned", execution_admission=False, scientific_acceptance=False))
        return run_dir
    try:
        inputs = verify_inputs(registry, plan, source_main)
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
                        reservation_fd=fd, reservation_identity=[stat.st_dev, stat.st_ino], plan=plan, inputs=inputs,
                        groups=[group for group in plan["groups"] if group["worker_slot"] == slot],
                        jobs=[job for job in plan["jobs"] if job["worker_slot"] == slot],
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
                        raise RuntimeError("Preprocessing worker failed; preserve original logs and completed prefix")
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
                            or terminal.get("records") != len(assigned) or terminal.get("completed_files") != files):
                        raise ValueError("Worker terminal does not bind its owner, request and exact job files")
                records = _collect(run_dir, plan, records)
                if any(row["status"] != "complete" for row in records):
                    raise ValueError("Incomplete selected preprocessing evaluation panel")
            finally:
                cleanup = []
                for child, birth, directory, request_hash in reversed(handles):
                    try:
                        cleanup.append(close_branch_process(child, birth))
                    except BaseException as exc:
                        cleanup.append(dict(status="failed", pid=child.pid, error=repr(exc)))
                atomic_json(run_dir / "cleanup.json", cleanup)
                if any(row["status"] != "verified" for row in cleanup):
                    raise RuntimeError("Worker containment cleanup failed")
        success = True
    except BaseException as exc:
        failure = exc
        try:
            records = _collect(run_dir, plan, records)
        except BaseException as collection_exc:
            collection_error = repr(collection_exc)
        attempted = set()
        for child, birth, directory, request_hash in handles:
            path = directory / "progress.json"
            if path.exists():
                progress = _read(path)
                attempted.add(progress.get("job_index"))
                if progress.get("phase") == "adaptive_generation":
                    attempted.update(job["index"] for job in plan["jobs"]
                        if job["group_index"] == progress.get("group_index"))
        records = [dict(row, status="failed" if row["job_index"] in attempted else "not_run",
            reason=repr(exc)) if row["status"] == "planned" else row for row in records]
    finally:
        atomic_json(run_dir / "records.json", records)
        try:
            _reports(registry, run_dir, plan, records)
        except BaseException as exc:
            reports_error = exc
            success = False
        status = "complete_pending_independent_acceptance" if success else "failed"
        atomic_json(run_dir / "summary.json", dict(status=status, inference_jobs=plan["inference_jobs"],
            generation_jobs=plan["generation_jobs"], transform_groups=plan["transform_groups"],
            completed_metric_records=sum(row["status"] == "complete" for row in records),
            failed_metric_records=sum(row["status"] == "failed" for row in records),
            not_run_metric_records=sum(row["status"] == "not_run" for row in records),
            diagnostic_only=plan["diagnostic_only"], scientific_acceptance=False,
            project_wide_device_release_accepted=False))
        atomic_json(run_dir / "terminal.json", dict(status=status, scientific_acceptance=False,
            error=repr(failure) if failure is not None else None, collection_error=collection_error,
            reports_error=repr(reports_error) if reports_error is not None else None, completed_at_utc=_now()))
    if failure is not None:
        raise failure
    if reports_error is not None:
        raise reports_error
    return run_dir


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
