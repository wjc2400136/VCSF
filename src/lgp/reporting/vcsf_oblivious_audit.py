"""Read-only transform and prediction replay for frozen preprocessing results."""
import gzip
import hashlib
import json
from pathlib import Path

from PIL import Image

from ..io import atomic_json, file_digest
from ..runners.preprocessing_defense import _subset_annotation, transform_image
from ..runners.vcsf_final_oblivious import execution_plan, verify_view_record
from ..runners.vcsf_final_oblivious_contract import PROTOCOL, verify_source
from ..runners.vcsf_final_oblivious_views import verify_view
from ..runners.vcsf_final_training_state import source_identity, verify_evaluation
from ..runners.vcsf_final_training_state_contract import ids_digest
from .coco_subset import evaluate_bbox_subset
from .vcsf_oblivious_checkpoint_binding import derive
from .vcsf_training_state_audit import require, compare_metrics, verify_owner, verify_archive_metadata


def require_isolated(output, path):
    output, path = Path(output).resolve(), Path(path).resolve()
    require(output != path and path not in output.parents and output not in path.parents,
            "Audit output overlaps protected input")


def bind_source_snapshot(evidence, root, expected):
    root = Path(root).resolve()
    require(source_identity(root) == expected, "Execution source snapshot changed")
    for name, digest in expected.items():
        evidence[str((root / name).resolve())] = digest


def verify_evidence_unchanged(evidence):
    require(all(file_digest(Path(path)) == digest for path, digest in evidence.items()),
            "Evidence changed during audit")


def verify_transform(registry, source, inputs, entry, image_ids, read, evidence):
    verify_view(entry)
    view = Path(entry["view"]).resolve()
    run = read(view / "run.json")
    annotation = read(view / "annotations.json")
    manifest = read(entry["manifest"])
    spec = registry.protocols[PROTOCOL]
    defense = entry["defense"]
    params = registry.preprocessing_defenses[defense]
    original_root = Path(inputs["payload"])
    original_run = read(original_root / "run.json")
    original_annotation = read(inputs["annotation"])
    mapping = [row for row in inputs["retained_seed_mapping"] if row["image_id"] in set(image_ids)]
    expected = dict(source=source, attack="vcsf", dataset="coco", split="val",
        defense_protocol=PROTOCOL, defense=defense, defense_parameters=params,
        parameters_sha256=spec["parameters_sha256"], original_payload=str(original_root),
        original_payload_binding=spec["payload_bindings"][source],
        original_annotation_sha256=spec["annotation_sha256"],
        original_full_image_ids_sha256=spec["full_image_ids_sha256"],
        evaluated_image_ids_sha256=ids_digest(image_ids), original_seed_mapping=mapping,
        threat_model=registry.preprocessing_defense_threat_model)
    require(all(run.get(k) == v for k, v in expected.items()), "Derived view provenance differs")
    require(manifest["implementation"] == registry.preprocessing_defense_policy
            and manifest["defense_parameters"] == params and manifest["defense"] == defense
            and manifest["source"] == source and manifest["images"] == len(image_ids)
            and manifest["scratch_removed"] is False, "Transform manifest differs from registry")
    identity = params["kind"] == "identity"
    names = None if identity else {i: "images/{:012d}.png".format(i) for i in image_ids}
    require(annotation == _subset_annotation(original_annotation, image_ids, names),
            "Transformed annotation differs from original retained labels")
    original_images = {row["id"]: row for row in original_annotation["images"]}
    input_root = (original_root / original_run.get("image_root", ".")).resolve()
    require(run["image_root"] == (str(input_root) if identity else "."), "View image root differs")
    require(manifest["materialization"] == ("reference_only" if identity else "png"),
            "Unexpected materialization mode")
    for row in annotation["images"]:
        original = input_root / original_images[row["id"]]["file_name"]
        actual = (view / run["image_root"] / row["file_name"]).resolve()
        for path in (original, actual):
            evidence[str(path.resolve())] = file_digest(path)
        with Image.open(original) as image:
            expected_image = transform_image(image, params)
        with Image.open(actual) as image:
            require(image.size == expected_image.size
                    and image.convert("RGB").tobytes() == expected_image.tobytes(),
                    "Saved image differs from registered transform replay")
    return annotation


def audit(registry, run_root, output, accepted_report, accepted_report_sha256,
          admission=None, admission_sha256=None):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    def isolated(path):
        require_isolated(output, path)
    isolated(run_root)
    isolated(Path(accepted_report).parent)
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
        require(plan == execution_plan(registry, plan["devices"], plan["max_images"],
                                      plan["diagnostic_jobs"]), "Plan differs from registered scope")
        terminal = read(run_root / "terminal.json")
        require(terminal.get("status") == "complete_pending_independent_acceptance"
                and terminal.get("formal_scope") is plan["formal_scope"]
                and terminal.get("scientific_acceptance") is False, "Run terminal is incomplete")
        checkpoints, accepted_evidence = derive(registry, accepted_report, accepted_report_sha256)
        evidence.update(accepted_evidence)
        for path in accepted_evidence:
            isolated(Path(path).parent)
        saved = read(run_root / "checkpoint_bindings.json")
        isolated(Path(saved["original"]).resolve().parent)
        require(saved["checkpoints"] == checkpoints
                and file_digest(Path(saved["original"])) == saved["original_sha256"]
                and read(saved["original"]) == checkpoints, "Checkpoint bindings differ from accepted main")
        admitted = None
        if plan["formal_scope"]:
            require(admission is not None and admission_sha256 is not None,
                    "Formal audit requires externally pinned prospective admission")
            isolated(Path(admission).resolve().parent)
            startup = read(run_root / "admission.json")
            require(startup["file"] == str(Path(admission).resolve())
                    and startup["sha256"] == admission_sha256, "Startup admission differs from external pin")
            from ..runners.vcsf_oblivious_admission import verify_admission
            admitted = verify_admission(registry, admission, admission_sha256, plan=plan,
                output=run_root, payloads=startup["receipt"]["payloads"],
                checkpoint_bindings=saved["original"], checkpoint_bindings_sha256=saved["original_sha256"],
                check_payloads=False, evidence=evidence, audit_output=output)
            require(startup["receipt"] == admitted, "Startup admission contents differ")
            read(admission)
        source_identity_saved = read(run_root / "source_identity.json")
        if admitted is not None:
            from ..runners.vcsf_oblivious_admission import verify_executed_source
            verify_executed_source(source_identity_saved, admitted)
        coordinator = read(run_root / "coordinator.json")
        active = [(i, device) for i, device in enumerate(plan["devices"])
                  if any(job["device"] == device for job in plan["jobs"])]
        require({p.name for p in (run_root / "workers").iterdir()} == {str(i) for i, _ in active},
                "Unexpected worker set")
        annotations, view_entries, source_ids = {}, {}, {}
        results, owners, gpu_uuids = [], [], []
        key = lambda row: (row["source"], row["defense"], row["target"])
        for slot, device in active:
            directory = run_root / "workers" / str(slot)
            request = read(directory / "request.json")
            jobs = [job for job in plan["jobs"] if job["device"] == device]
            require(request["plan"] == plan and request["jobs"] == jobs
                    and request["checkpoints"] == checkpoints and request["coordinator"] == coordinator
                    and request["output"] == str(run_root)
                    and request["source_identity"] == source_identity_saved
                    and source_identity(Path(request["source_root"])) == source_identity_saved,
                    "Worker request or source identity differs")
            require(set(request["payloads"]) == set(registry.protocols[PROTOCOL]["sources"]),
                    "Worker payload scope differs")
            if admitted is not None:
                require(request.get("admission") == dict(file=str(Path(admission).resolve()), sha256=admission_sha256)
                        and request["payloads"] == admitted["payloads"], "Worker lacks prospective admission binding")
            bind_source_snapshot(evidence, request["source_root"], source_identity_saved)
            for source in dict.fromkeys(job["source"] for job in jobs):
                isolated(request["payloads"][source])
                inputs = read(run_root / "views" / source / "input_verification.json")
                require(verify_source(registry, source, request["payloads"][source]) == inputs,
                        "Original source payload differs")
                if admitted is not None:
                    require(inputs == admitted["verified_inputs"][source], "Execution input differs from admission")
                manifest_path = Path(inputs["payload"]) / "manifest.jsonl"
                evidence[str(manifest_path.resolve())] = registry.protocols[PROTOCOL]["payload_bindings"][source]["payload_manifest_sha256"]
                ids = inputs["retained_image_ids"]
                if plan["max_images"] is not None:
                    ids = ids[:plan["max_images"]]
                source_ids[source] = ids
                views = read(run_root / "views" / source / "views.json")
                require(views["source"] == source and views["image_ids"] == ids
                        and views["formal_scope"] is plan["formal_scope"]
                        and views["formal_result_eligible"] is False
                        and [row["defense"] for row in views["views"]] == registry.preprocessing_defense_order,
                        "View population or transform set differs")
                for entry in views["views"]:
                    expected_root = run_root / "views" / source / "scratch" / entry["defense"]
                    require(Path(entry["view"]).resolve() == expected_root,
                            "View points outside its assigned source")
                    annotations[(source, entry["defense"])] = verify_transform(
                        registry, source, inputs, entry, ids, read, evidence)
                    view_entries[(source, entry["defense"])] = entry
            process = read(directory / "process.json")
            gpu = read(directory / "gpu_context_registration.json")
            worker_terminal = read(directory / "terminal.json")
            require(worker_terminal["status"] == "complete_pending_independent_acceptance"
                    and worker_terminal["request_sha256"] == evidence[str((directory / "request.json").resolve())]
                    and worker_terminal["scientific_acceptance"] is False, "Worker terminal differs")
            verify_owner(coordinator, process, worker_terminal["owner"], gpu)
            require(gpu["gpu_uuid"] == request["gpu_uuid"], "GPU assignment differs")
            owners.append(process["pid"])
            gpu_uuids.append(gpu["gpu_uuid"])
            rows = read(directory / "results.json")
            require(rows == worker_terminal["results"] and list(map(key, rows)) == list(map(key, jobs)),
                    "Worker result set differs")
            for job, result in zip(jobs, rows):
                source, defense, target = key(job)
                entry = view_entries[(source, defense)]
                ids = source_ids[source]
                evaluation = run_root / "evaluations" / source / defense / target
                record = read(evaluation / "metrics.json")
                checkpoint = checkpoints[target]
                verify_evaluation(record, dict(plan, source=source, target=target), ids,
                                  checkpoint["path"], checkpoint["sha256"])
                verify_view_record(record, entry)
                require(result["evaluation"] == str(evaluation)
                        and result["metrics_sha256"] == evidence[str((evaluation / "metrics.json").resolve())]
                        and result["checkpoint_sha256"] == checkpoint["sha256"]
                        and result["view_run_sha256"] == entry["run_sha256"], "Result binding differs")
                archive = read(evaluation / "predictions_artifact.json")
                archive_path = evaluation / "predictions.json.gz"
                archive_hash = file_digest(archive_path)
                require(archive_hash == archive["archive_sha256"], "Prediction archive hash differs")
                evidence[str(archive_path)] = archive_hash
                with gzip.open(str(archive_path), "rb") as stream:
                    raw = stream.read()
                require(hashlib.sha256(raw).hexdigest() == archive["uncompressed_sha256"]
                        == record["predictions_sha256"], "Prediction content hash differs")
                predictions = json.loads(raw)
                verify_archive_metadata(archive, archive_path.stat().st_size, raw, predictions, record)
                replay = evaluate_bbox_subset(annotations[(source, defense)], predictions, ids, ids)
                compare_metrics(replay["metrics"], record["metrics"])
            results.extend(rows)
        require(len(set(owners)) == len(owners) and len(set(gpu_uuids)) == len(gpu_uuids), "Aliased worker ownership")
        order = list(map(key, plan["jobs"]))
        results.sort(key=lambda row: order.index(key(row)))
        require(read(run_root / "results.json") == results, "Root result set differs")
        cleanup = read(run_root / "cleanup.json")
        require(len(cleanup) == len(owners) and sorted(row["process_group"] for row in cleanup) == sorted(owners)
                and all(row["status"] == "verified" and row["exit_code"] == 0 for row in cleanup),
                "Worker cleanup incomplete")
        verify_evidence_unchanged(evidence)
        receipt = dict(status="independently_verified_saved_preprocessing" if admitted is not None
                       else "saved_preprocessing_replayed_pending_formal_admission_binding",
            protocol=PROTOCOL, run=str(run_root), cells=len(results), metric_values=12*len(results),
            formal_scope=plan["formal_scope"], formal_result_eligible=admitted is not None,
            admission_sha256=admission_sha256 if admitted is not None else None,
            scientific_claims_accepted=False, model_calls=0, results=results, evidence=evidence)
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        # A late isolation failure must never create output inside newly discovered inputs.
        raise
