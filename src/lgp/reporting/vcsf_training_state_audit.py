"""Independent saved-artifact audit for the final training-state refresh."""
import gzip
import hashlib
import json
import math
from pathlib import Path

from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..runners.vcsf_final_training_state import source_identity, verify_evaluation, require_output_isolation
from ..runners.vcsf_final_training_state_contract import build_plan, verify_payload, verify_pair
from .coco_subset import evaluate_bbox_subset


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compare_metrics(actual, expected):
    require(isinstance(actual, dict) and isinstance(expected, dict)
            and set(actual) == set(expected) == set(COCO_BBOX_METRICS), "Incomplete metric set")
    for key in COCO_BBOX_METRICS:
        a, b = actual[key], expected[key]
        require(type(a) in (int, float) and type(b) in (int, float)
                and math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 1e-12,
                "Saved metric differs from official replay: " + key)


def verify_annotation_path(record, inputs):
    value = record.get("annotation")
    require(isinstance(value, str) and bool(value)
            and Path(value).resolve() == Path(inputs["annotation"]).resolve(),
            "Evaluation annotation path differs from authenticated input")


def verify_archive_metadata(archive, archive_bytes, raw, predictions, record):
    expected = dict(schema_version=1, status="verified_lossless_archive", format="gzip",
        archive_file="predictions.json.gz", uncompressed_file="predictions.json",
        archive_bytes=archive_bytes, uncompressed_bytes=len(raw), prediction_records=len(predictions),
        image_ids_with_detections=len({row["image_id"] for row in predictions}),
        fields=["image_id", "category_id", "bbox", "score"])
    require(all(type(archive.get(key)) is type(value) and archive.get(key) == value
                for key, value in expected.items()), "Archive declaration differs from saved predictions")
    require(type(record.get("detections")) is int and record["detections"] == len(predictions),
            "Evaluation detection count differs from predictions")


def verify_owner(coordinator, process, owner, gpu):
    expected = dict(worker_pid=process["pid"], worker_start_ticks=process["start_ticks"],
        process_group=process["pid"], coordinator_pid=coordinator["pid"],
        coordinator_start_ticks=coordinator["start_ticks"])
    require(owner == expected, "Worker terminal owner differs from launch identity")
    require(process["parent_pid"] == coordinator["pid"]
            and process["process_group"] == process["session_id"] == process["pid"],
            "Worker did not own its recorded process group")
    require(gpu["worker_pid"] == process["pid"]
            and gpu["worker_pid_start_ticks"] == process["start_ticks"]
            and gpu["before_context_pids"] == []
            and type(gpu["nvml_pid"]) is int and gpu["nvml_pid"] > 0
            and gpu["registered_context_pids"] == [gpu["nvml_pid"]],
            "GPU context registration does not bind to worker")


def audit(registry, run_root, output, qualification):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    require_output_isolation(output, run_root, run_root, qualification)
    require(not output.exists(), "Audit output already exists")
    evidence = {}
    def read(path):
        path = Path(path).resolve()
        digest = file_digest(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        require(file_digest(path) == digest, "Evidence changed while read")
        evidence[str(path)] = digest
        return data
    try:
        plan = read(run_root / "plan.json")
        require(plan == build_plan(registry, plan["devices"], plan["max_images"]), "Plan differs from registered jobs")
        terminal = read(run_root / "terminal.json")
        require(terminal.get("status") == "complete_pending_independent_acceptance"
                and terminal.get("formal_scope") is plan["formal_scope"]
                and terminal.get("scientific_acceptance") is False, "Execution terminal is incomplete")
        inputs = read(run_root / "input_verification.json")
        require_output_isolation(output, inputs["payload"], run_root, qualification)
        require(verify_payload(registry, inputs["payload"]) == inputs, "Original payload no longer matches")
        annotation = read(inputs["annotation"])
        ids = inputs["image_ids"] if plan["max_images"] is None else inputs["image_ids"][:plan["max_images"]]
        retained = [value for value in inputs["retained_image_ids"] if value in set(ids)]
        coordinator = read(run_root / "coordinator.json")
        source = read(run_root / "source_identity.json")
        expected_workers = {str(index) for index in range(len(plan["devices"]))}
        require({path.name for path in (run_root / "workers").iterdir()} == expected_workers,
                "Missing or unexpected worker directory")
        all_results, owners, gpu_uuids = [], [], []
        for slot, device in enumerate(plan["devices"]):
            directory = run_root / "workers" / str(slot)
            request = read(directory / "request.json")
            request_hash = evidence[str((directory / "request.json").resolve())]
            jobs = [job for job in plan["jobs"] if job["device"] == device]
            require(request["plan"] == plan and request["jobs"] == jobs
                    and request["inputs"] == inputs and request["coordinator"] == coordinator
                    and request["output"] == str(run_root) and request["source_identity"] == source,
                    "Worker request differs from registered assignment")
            require(source_identity(Path(request["source_root"])) == source, "Execution source snapshot changed")
            checkpoint_paths = request["checkpoints"]
            pair_roots = {str(Path(value).resolve().parent) for value in checkpoint_paths.values()}
            require(len(pair_roots) == 1, "Checkpoint paths do not share a pair root")
            pair_root = next(iter(pair_roots))
            require_output_isolation(output, inputs["payload"], pair_root, qualification)
            require(verify_pair(registry, pair_root, qualification) == checkpoint_paths, "Checkpoint pair binding differs")
            process = read(directory / "process.json")
            gpu = read(directory / "gpu_context_registration.json")
            worker_terminal = read(directory / "terminal.json")
            require(worker_terminal["status"] == "complete_pending_independent_acceptance"
                    and worker_terminal["request_sha256"] == request_hash
                    and worker_terminal["scientific_acceptance"] is False, "Worker terminal mismatch")
            verify_owner(coordinator, process, worker_terminal["owner"], gpu)
            require(gpu["gpu_uuid"] == request["gpu_uuid"], "GPU UUID differs from reservation request")
            owners.append(process["pid"])
            gpu_uuids.append(gpu["gpu_uuid"])
            results = read(directory / "results.json")
            require(results == worker_terminal["results"]
                    and [row["victim_state"] for row in results] == [job["victim_state"] for job in jobs],
                    "Worker results are missing, duplicated or reordered")
            for job, result in zip(jobs, results):
                evaluation = run_root / "evaluations" / job["victim_state"]
                metrics_path = evaluation / "metrics.json"
                record = read(metrics_path)
                verify_evaluation(record, plan, ids, checkpoint_paths[job["victim_state"]], job["checkpoint_sha256"])
                require(record["adversarial_run"] == inputs["payload"], "Evaluation used another payload path")
                verify_annotation_path(record, inputs)
                require(result["checkpoint_sha256"] == job["checkpoint_sha256"]
                        and result["metrics_sha256"] == evidence[str(metrics_path.resolve())], "Result does not bind metrics/checkpoint")
                archive = read(evaluation / "predictions_artifact.json")
                archive_path = evaluation / "predictions.json.gz"
                archive_hash = file_digest(archive_path)
                require(archive_hash == archive["archive_sha256"] == result["archive_sha256"], "Archive hash mismatch")
                evidence[str(archive_path.resolve())] = archive_hash
                with gzip.open(str(archive_path), "rb") as stream:
                    raw = stream.read()
                raw_hash = hashlib.sha256(raw).hexdigest()
                require(raw_hash == archive["uncompressed_sha256"] == record["predictions_sha256"] == result["predictions_sha256"],
                        "Uncompressed predictions do not bind to result")
                predictions = json.loads(raw)
                verify_archive_metadata(archive, archive_path.stat().st_size, raw, predictions, record)
                full = evaluate_bbox_subset(annotation, predictions, ids, ids)
                compare_metrics(full["metrics"], record["metrics"])
                compare_metrics(full["metrics"], result["full_metrics"])
                if retained:
                    subset = evaluate_bbox_subset(annotation, predictions, ids, retained)
                    saved = read(evaluation / "retained_projection" / "metrics.json")
                    require(saved == result["retained_metrics"] and saved["image_ids"] == retained
                            and saved["images"] == len(retained) and saved["model_calls"] == 0
                            and saved["root_acceptance"] is False and saved["scientific_acceptance"] is False,
                            "Retained projection identity differs")
                    compare_metrics(subset["metrics"], saved["metrics"])
                    require(type(saved.get("detections")) is int and saved["detections"] == subset["detections"],
                            "Retained detection count differs from saved predictions")
                else:
                    require(result["retained_metrics"] is None, "Unexpected diagnostic retained result")
            all_results.extend(results)
        require(len(set(owners)) == len(owners) and len(set(gpu_uuids)) == len(gpu_uuids), "Workers alias an owner/GPU")
        order = [job["victim_state"] for job in plan["jobs"]]
        all_results.sort(key=lambda row: order.index(row["victim_state"]))
        require(read(run_root / "results.json") == all_results, "Root results differ from complete workers")
        cleanup = read(run_root / "cleanup.json")
        require(len(cleanup) == len(owners) and sorted(row["process_group"] for row in cleanup) == sorted(owners)
                and all(row["status"] == "verified" and row["exit_code"] == 0 for row in cleanup),
                "Worker cleanup/exit evidence incomplete")
        require(all(file_digest(Path(path)) == digest for path, digest in evidence.items()), "Evidence changed during audit")
        receipt = dict(status="independently_verified_saved_results", protocol=plan["protocol"],
            run=str(run_root), formal_result_eligible=plan["formal_scope"],
            scope="two_registered_victims_from_one_frozen_source", images_per_victim=len(ids),
            retained_images_per_victim=len(retained), results=all_results, evidence=evidence,
            model_calls=0, scientific_claims_accepted=False)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "receipt.json", receipt)
        return receipt
    except BaseException:
        # Failure never writes into the original run or replaces an earlier audit.
        raise
