"""Fresh current-method generation through the unchanged registered BPDA path."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from ..defenses import AdaptivePreprocessor
from ..io import file_digest
from .current_reproduction_inputs import _bind, _hash, _inside, _read, _unchanged


def seed_mapping(plan, group, inputs):
    rows = inputs["payloads"][group["source"]][group["attack"]]["seed_mapping"]
    if ([row["image_id"] for row in rows] != inputs["selected_image_ids"]
            or any(type(row["full_position"]) is not int
                or row["attack_seed"] != plan["seed"] + row["full_position"] for row in rows)):
        raise ValueError("Adaptive seeds must retain the registered full-validation positions")
    return rows


def prepare_payload(registry, plan, group, inputs, output, progress_callback=None):
    from .attack import run_attack

    rows = seed_mapping(plan, group, inputs)
    for binding in inputs["clean"]["images"]:
        _unchanged(binding, rehash=True)
    ids = inputs["selected_image_ids"]
    attack_dir = output / "adaptive_attacks" / ("{:06d}".format(group["index"]))
    defense = plan["defense_parameters"][group["defense"]]
    metadata = dict(protocol=plan["protocol"], defense=group["defense"],
        defense_parameters=defense, adaptive_policy=plan["adaptive_policy"],
        diagnostic_only=plan["diagnostic_only"],
        selected_image_ids_sha256=inputs["selected_image_ids_sha256"],
        retained500_image_ids_sha256=inputs["retained500_image_ids_sha256"],
        seed_schedule=plan["seed_schedule"], target_queries=0, target_gradients=0)
    run_attack(registry, "coco", group["source"], group["attack"], split=plan["split"],
        output_dir=attack_dir, max_images=None, seed=plan["seed"], device="cuda:0",
        download_weights=False, keep_going=False, strict=True,
        budget_profile=group["budget_profile"], image_ids=ids,
        seed_offsets={row["image_id"]: row["full_position"] for row in rows},
        input_transform=AdaptivePreprocessor(defense), run_metadata=metadata,
        progress_callback=progress_callback)
    for binding in inputs["clean"]["images"]:
        _unchanged(binding, rehash=True)
    run = _read(attack_dir / "run.json")
    required = dict(status="complete", dataset="coco", split=plan["split"], source=group["source"],
        attack=group["attack"], seed=plan["seed"], requested_images=len(ids), successful_images=len(ids),
        failed_images=0, parameters_sha256=group["parameters_sha256"], budget_profile=group["budget_profile"],
        checkpoint_sha256=inputs["source_checkpoint_sha256"][group["source"]],
        seed_schedule="explicit_offsets", run_metadata=metadata,
        requested_ordered_image_ids_sha256=_hash(ids),
        seed_offsets_sha256=_hash([[row["image_id"], row["full_position"]] for row in rows]))
    if (any(run.get(key) != value for key, value in required.items())
            or _hash(run.get("parameters")) != group["parameters_sha256"]):
        raise ValueError("Adaptive generation changed its source, budget, parameters, IDs or seeds")
    annotation_path = _inside(attack_dir, run.get("annotation", "annotations.json"))
    manifest_path = _inside(attack_dir, run.get("manifest", "manifest.jsonl"))
    if (file_digest(annotation_path) != run.get("annotation_sha256")
            or file_digest(manifest_path) != run.get("manifest_sha256")):
        raise ValueError("Adaptive generation metadata is not hash-bound")
    annotation = _read(annotation_path)
    names = {image["id"]: image["file_name"] for image in annotation["images"]}
    canonical = inputs["clean"]["annotation"]
    expected_annotation = dict(canonical, images=[dict(image, file_name=names.get(image["id"]))
        for image in canonical["images"]])
    if (annotation != expected_annotation or [image["id"] for image in annotation["images"]] != ids
            or len(names) != len(ids)):
        raise ValueError("Adaptive annotation changed canonical retained ground truth")
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    if len(manifest) != len(ids):
        raise ValueError("Adaptive generation manifest is incomplete")
    image_root = _inside(attack_dir, run.get("image_root", "."))
    clean_by_id = {image["id"]: image for image in canonical["images"]}
    images, seen = [], set()
    epsilon_pixels = float(run["parameters"]["eps"]) * 255.0
    if not math.isfinite(epsilon_pixels) or epsilon_pixels < 0:
        raise ValueError("Adaptive epsilon must be finite and nonnegative")
    for position, (row, mapping) in enumerate(zip(manifest, rows)):
        if (row.get("status") != "ok" or row.get("position") != position
                or row.get("image_id") != mapping["image_id"] or row.get("attack_seed") != mapping["attack_seed"]):
            raise ValueError("Adaptive manifest differs from the original seed population")
        path = _inside(attack_dir, row["output_file"])
        if path != _inside(image_root, names[mapping["image_id"]]) or path in seen:
            raise ValueError("Adaptive image paths escape or alias the generation output")
        binding = _bind(path)
        if binding["sha256"] != row.get("output_sha256") or binding["bytes"] != row.get("output_bytes"):
            raise ValueError("Adaptive output pixels differ from their saved hashes")
        clean_path = _inside(Path(inputs["clean"]["root"]), clean_by_id[mapping["image_id"]]["file_name"])
        with Image.open(clean_path) as clean_image, Image.open(path) as attacked_image:
            clean = np.asarray(clean_image.convert("RGB"), dtype=np.int16)
            attacked = np.asarray(attacked_image.convert("RGB"), dtype=np.int16)
        if clean.shape != attacked.shape or float(np.abs(attacked - clean).max()) > epsilon_pixels + 1e-4:
            raise ValueError("Saved adaptive PNG violates the original-clean epsilon or image shape")
        seen.add(path);images.append(binding)
    headers = [_bind(attack_dir / "run.json"), _bind(annotation_path), _bind(manifest_path)]
    if group["attack"] == "vcsf":
        headers.append(_bind(attack_dir / "native_cost.jsonl"))
    return dict(root=str(attack_dir), annotation=str(annotation_path), image_root=str(image_root),
        images=images, seed_mapping=rows, parameters_sha256=group["parameters_sha256"],
        source_checkpoint_sha256=run["checkpoint_sha256"], headers=headers,
        generation_metadata=metadata, preserved_payload=True)