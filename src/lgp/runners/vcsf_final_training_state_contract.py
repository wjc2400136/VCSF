"""Frozen-input contract for evaluation-only final VCSF training-state refresh."""
import hashlib
import json
from pathlib import Path
import re

from ..io import file_digest


PROTOCOL = "vcsf_final_training_state_refresh"


def ids_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_plan(registry, devices, max_images=None):
    spec = registry.protocols[PROTOCOL]
    if (not isinstance(devices, (list, tuple)) or len(devices) not in spec["device_counts"]
            or any(not isinstance(device, str) or not re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
                   for device in devices) or len(set(devices)) != len(devices)):
        raise ValueError("Select one or two unique explicit CUDA devices")
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= spec["full_images"]):
        raise ValueError("Diagnostic max-images must be a positive integer within the full split")
    images = spec["full_images"] if max_images is None else max_images
    jobs = [{"index": index, "victim_state": state, "device": devices[index % len(devices)],
             "images": images, "checkpoint_sha256": spec["checkpoint_sha256"][state]}
            for index, state in enumerate(spec["victim_states"])]
    return {"protocol": PROTOCOL, "dataset": spec["dataset"], "split": spec["split"],
            "source": spec["source"], "target": spec["target"],
            "parameters_sha256": spec["parameters_sha256"], "devices": list(devices),
            "max_images": max_images, "formal_scope": max_images is None,
            "generation_jobs": 0, "inference_jobs": len(jobs), "jobs": jobs,
            "retained_projection": spec["retained_mode"],
            "metric_records": spec["metric_records"] if max_images is None else None,
            "execution_admission": False}


def _bound_json(path, expected):
    if file_digest(path) != expected:
        raise ValueError("Input SHA256 differs from registered identity: " + path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root, value):
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError("Payload path escapes its original root")
    return path


def verify_payload(registry, payload_root):
    """Rehash the immutable full payload, without generating or rewriting files."""
    spec = registry.protocols[PROTOCOL]
    return verify_payload_spec(spec, payload_root)


def verify_payload_spec(spec, payload_root):
    """Apply an explicitly registered source binding to the same full-payload checks."""
    root = Path(payload_root).resolve()
    run = _bound_json(root / "run.json", spec["payload_run_sha256"])
    manifest_path = root / "manifest.jsonl"
    if file_digest(manifest_path) != spec["payload_manifest_sha256"]:
        raise ValueError("Payload manifest SHA256 differs from registered identity")
    with manifest_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    annotation_path = _inside(root, run.get("annotation", "annotations.json"))
    annotation = _bound_json(annotation_path, spec["annotation_sha256"])
    if (run.get("status") != "complete" or run.get("parameters_sha256") != spec["parameters_sha256"]
            or run.get("dataset") != spec["dataset"] or run.get("split") != spec["split"]
            or run.get("source") != spec["source"] or run.get("full_payload_available") is False):
        raise ValueError("Payload is not the complete registered final-method source")
    ids = sorted(row["id"] for row in annotation["images"])
    if (len(ids) != spec["full_images"] or any(type(value) is not int for value in ids)
            or len(set(ids)) != len(ids) or ids_digest(ids) != spec["full_image_ids_sha256"]):
        raise ValueError("Full payload image identity differs")
    if len(rows) != len(ids):
        raise ValueError("Full payload manifest is incomplete")
    image_root = _inside(root, run.get("image_root", "."))
    image_records = {row["id"]: row for row in annotation["images"]}
    files = set()
    total_bytes = 0
    for position, (row, image_id) in enumerate(zip(rows, ids)):
        if (row.get("status") != "ok" or row.get("image_id") != image_id
                or row.get("position") != position or row.get("attack_seed") != spec["seed"] + position):
            raise ValueError("Payload row order, success or seed identity differs")
        path = _inside(root, row["output_file"])
        inference_path = _inside(root, str(image_root / image_records[image_id]["file_name"]))
        if inference_path != path:
            raise ValueError("Evaluator image path differs from authenticated payload image")
        if path in files or path.stat().st_size != row["output_bytes"] or file_digest(path) != row["output_sha256"]:
            raise ValueError("Payload image bytes differ or alias another row")
        files.add(path)
        total_bytes += row["output_bytes"]
    positions = [((2 * index + 1) * len(ids)) // (2 * spec["retained_images"])
                 for index in range(spec["retained_images"])]
    retained = [ids[index] for index in positions]
    if ids_digest(retained) != spec["retained_image_ids_sha256"]:
        raise ValueError("Retained image identity differs from original selection")
    return {"image_ids": ids, "retained_image_ids": retained, "payload_files": len(files),
            "payload_bytes": total_bytes, "annotation": str(annotation_path),
            "payload": str(root), "parameters_sha256": spec["parameters_sha256"],
            "model_calls": 0, "execution_admission": False}


def verify_pair(registry, pair_root, qualification_path):
    spec = registry.protocols[PROTOCOL]
    _bound_json(Path(qualification_path), spec["pair_qualification_sha256"])
    paths = {}
    for state in spec["victim_states"]:
        path = Path(pair_root) / (spec["target"] + "_" + state + ".pth")
        if file_digest(path) != spec["checkpoint_sha256"][state]:
            raise ValueError("Victim checkpoint differs from qualified pair: " + state)
        paths[state] = str(path.resolve())
    return paths
