"""Preserved direct-payload execution for an owned radius budget group."""
import json
import math
from fractions import Fraction
from pathlib import Path

from ..io import atomic_json, file_digest
from .vcsf_final_training_state import _read, verify_evaluation
from .vcsf_final_training_state_contract import ids_digest
from .vcsf_research_plan import canonical_hash


ALIAS = "vcsf_final_budget_isolated"
MODE = "final_budget_execution"


def verify_generated_payload(payload, binding, metadata):
    """Check saved identities and bytes, not independent numerical acceptance."""
    payload = Path(payload).resolve()
    run = _read(payload / "run.json")
    expected = dict(status="complete", dataset="coco", split="val",
        source=binding["source"], attack=ALIAS, seed=binding["seed"],
        parameters_sha256=binding["parameters_sha256"],
        budget_profile=binding["budget_profile"], run_metadata=metadata,
        successful_images=len(binding["image_ids"]), failed_images=0,
        full_payload_available=True, image_root=".", seed_schedule="explicit_offsets")
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Budget generated run differs from its assigned group")
    if canonical_hash(run["parameters"]) != binding["parameters_sha256"]:
        raise ValueError("Budget generated parameters changed")
    limit = Fraction(binding["epsilon"]) * 255
    if limit <= 0:
        raise ValueError("Budget generated radius must be positive")
    for name, key in (("manifest.jsonl", "manifest_sha256"),
                      ("annotations.json", "annotation_sha256")):
        if file_digest(payload / name) != run[key]:
            raise ValueError("Budget generated metadata changed")
    with (payload / "manifest.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    annotation = _read(payload / "annotations.json")
    ids = binding["image_ids"]
    if ([row["image_id"] for row in rows] != ids
            or [row["id"] for row in annotation["images"]] != ids):
        raise ValueError("Budget generated population differs")
    seeds = {row["image_id"]: row["attack_seed"] for row in binding["seed_mapping"]}
    offsets = [[row["image_id"], row["offset"]] for row in binding["seed_mapping"]]
    if (run["image_ids_sha256"] != ids_digest(ids)
            or run["requested_ordered_image_ids_sha256"] != ids_digest(ids)
            or run["seed_offsets_sha256"] != canonical_hash(offsets)):
        raise ValueError("Budget saved seed or image selection differs")
    for row, image in zip(rows, annotation["images"]):
        path = (payload / row["output_file"]).resolve()
        radius = row.get("linf_pixel")
        if (row.get("status") != "ok" or row.get("attack_seed") != seeds[row["image_id"]]
                or payload not in path.parents
                or path != (payload / image["file_name"]).resolve()
                or type(radius) not in (int, float) or not math.isfinite(radius)
                or not 0 <= radius <= limit
                or path.stat().st_size != row["output_bytes"]
                or file_digest(path) != row["output_sha256"]):
            raise ValueError("Budget generated image, seed or budget differs")
    return dict(payload=str(payload), run_sha256=file_digest(payload / "run.json"),
        manifest_sha256=run["manifest_sha256"], annotation_sha256=run["annotation_sha256"])


def _model_call(function, *args, **kwargs):
    from .adaptive_preprocessing_defense import _cleanup_cuda
    try:
        return function(*args, **kwargs)
    finally:
        _cleanup_cuda()


def execute_group(registry, request_path, request_hash, permit, binding, gpu, progress):
    """Run only in a registered worker; run_attack authenticates ownership again."""
    from .attack import run_attack
    from .evaluate import run_evaluation
    from .vcsf_oblivious_admission import bound_json

    request = bound_json(request_path, request_hash)
    if bound_json(request["permit"]["file"], request["permit"]["sha256"]) != permit:
        raise ValueError("Budget group permit differs from the worker request")
    output = Path(permit["output"]).resolve()
    index = binding["group_index"]
    group = output / "groups" / "{:06d}".format(index)
    payload = group / "attack"
    metadata = dict(protocol=binding["protocol"], group_index=index,
        epsilon=binding["epsilon"], permit_sha256=request["permit"]["sha256"],
        assets_sha256=permit["assets"]["sha256"], diagnostic_only=binding["max_images"] is not None)
    jobs = [job for job in permit["plan"]["generated_evaluations"]
            if job["group_index"] == index]
    if (len(jobs) != 16 or [job["target"] for job in jobs] != list(registry.paper_order)
            or any(job["source"] != binding["source"]
                   or job["epsilon"] != binding["epsilon"] for job in jobs)):
        raise ValueError("Budget group must evaluate every canonical target at its bound radius")
    gpu.verify()
    _model_call(run_attack, registry, "coco", binding["source"], ALIAS, split="val",
        output_dir=payload, max_images=binding["max_images"],
        seed=binding["seed"], device="cuda:0", strict=True,
        parameter_overrides=binding["parameters"], budget_profile=binding["budget_profile"],
        image_ids=binding["image_ids"],
        seed_offsets={row["image_id"]: row["offset"] for row in binding["seed_mapping"]},
        run_metadata=metadata, progress_callback=progress,
        isolated_research=dict(mode=MODE, execution_alias=ALIAS,
            worker_request=str(Path(request_path).resolve()),
            worker_request_sha256=request_hash, group_index=index),
        isolated_research_owner=gpu)
    before = verify_generated_payload(payload, binding, metadata)

    def require_unchanged_payload():
        if verify_generated_payload(payload, binding, metadata) != before:
            raise ValueError("Budget original payload changed after generation")

    results = []
    for job in jobs:
        gpu.verify()
        checkpoint = permit["checkpoints"][job["target"]]
        if file_digest(Path(checkpoint["path"])) != checkpoint["sha256"]:
            raise ValueError("Budget target checkpoint changed")
        require_unchanged_payload()
        destination = _model_call(run_evaluation, registry, "coco", job["target"], split="val",
            adversarial_run=payload,
            output_dir=group / "evaluations" / job["target"], device="cuda:0",
            download_weights=False, keep_going=False, save_visualizations=False,
            prediction_archive="gzip", target_checkpoint=Path(checkpoint["path"]),
            image_ids=binding["image_ids"], progress_callback=progress)
        record = _read(destination / "metrics.json")
        verify_evaluation(record, dict(job, dataset="coco", split="val",
            parameters_sha256=binding["parameters_sha256"]), binding["image_ids"],
            checkpoint["path"], checkpoint["sha256"])
        require_unchanged_payload()
        results.append(dict(group_index=index, source=binding["source"],
            epsilon=binding["epsilon"], target=job["target"], evaluation=str(destination),
            metrics_sha256=file_digest(destination / "metrics.json")))
        atomic_json(group / "results.json", results)
    require_unchanged_payload()
    atomic_json(group / "terminal.json", dict(status="complete_pending_independent_acceptance",
        results=results, formal_result_eligible=False, scientific_acceptance=False))
    return results
