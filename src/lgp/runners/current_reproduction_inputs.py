"""Authenticate current-main attack payloads without inheriting historical results."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from ..data.coco import CocoIndex
from ..io import file_digest
from .vcsf_final_training_state_contract import ids_digest


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _parameter_hash(registry, method):
    from ..attacks.factory import ATTACK_TYPES

    config = ATTACK_TYPES[method][1].from_mapping(dict(registry.attack(method).parameters))
    config.validate()
    return _hash(asdict(config))

def _inside(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if path != root and root not in path.parents:
        raise ValueError("Input artifact escapes its declared root")
    return path

def _bind(path, digest=True):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(file=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=file_digest(path) if digest else None)

def _unchanged(binding, rehash=True):
    path = Path(binding["file"])
    stat = path.stat()
    if stat.st_size != binding["bytes"] or stat.st_mtime_ns != binding["mtime_ns"]:
        raise ValueError("Bound input artifact changed")
    if rehash and binding["sha256"] is not None and file_digest(path) != binding["sha256"]:
        raise ValueError("Bound input artifact bytes changed")

def _numeric_sources(registry, methods):
    paths = set()
    if any(method != "vcsf" for method in methods):
        paths.update(p for p in (registry.root / "src/lgp/attacks").glob("*.py")
                     if not p.name.startswith("vcsf"))
        paths.update((registry.root / "src/lgp/adapters").glob("*.py"))
        paths.add(registry.root / "src/lgp/modeling.py")
    result = {p.relative_to(registry.root).as_posix(): file_digest(p) for p in paths}
    if "vcsf" in methods:
        from ..attacks.vcsf_public import verify_public_identity

        result.update(verify_public_identity(registry.root)["numeric_source_sha256"])
    return result


def verify_main_payloads(registry, plan, source_main, image_ids=None):
    source_main = Path(source_main).resolve()
    source_plan = _read(source_main / "plan.json")
    source_protocol = registry.protocols[plan["protocol"]]["method_source_protocol"]
    if source_plan.get("protocol") != source_protocol:
        raise ValueError("Payloads must come from the current main reproduction protocol")
    if type(source_plan.get("execution_seed")) is not int:
        raise ValueError("Source-main base seed is not recorded")
    provenance = _read(source_main / "provenance.json")["source_manifest"]
    files = provenance["files"]
    if (len(files) != provenance["file_count"] or _hash(files) != provenance["tree_sha256"]
            or len({row["path"] for row in files}) != len(files)):
        raise ValueError("Source-main provenance manifest is malformed")
    recorded_sources = {row["path"]: row["sha256"] for row in files}
    expected_sources = _numeric_sources(registry, plan["methods"])
    if any(recorded_sources.get(path) != digest for path, digest in expected_sources.items()):
        raise ValueError("Source-main numerical implementation differs from the current method identity")
    index = CocoIndex(registry.dataset(plan["dataset"]), plan["split"])
    full_ids = [row["id"] for row in index.images]
    if (len(full_ids) != plan["full_images"] or any(type(value) is not int for value in full_ids)
            or full_ids != sorted(set(full_ids))):
        raise ValueError("Canonical COCO validation population is incomplete or duplicated")
    selected = full_ids[:plan["selected_images"]] if image_ids is None else list(image_ids)
    if (not selected or selected != sorted(set(selected))
            or any(type(value) is not int for value in selected) or not set(selected) <= set(full_ids)):
        raise ValueError("Selected image population differs from the canonical split")
    selected_set = set(selected)
    count = plan["retained_images"]
    positions = [((2 * n + 1) * len(full_ids)) // (2 * count) for n in range(count)]
    retained = [full_ids[position] for position in positions]
    headers = [_bind(source_main / "plan.json"), _bind(source_main / "provenance.json"), _bind(index.annotation_path)]
    sources = list(plan.get("sources") or [plan["source"]])
    payloads, checkpoints, source_bindings = {}, {}, {}
    for source in sources:
        payloads[source] = {}
        source_path = (registry.root / "checkpoints/coco" / registry.model(source).checkpoint_filename).resolve()
        source_bindings[source] = _bind(source_path)
        source_checkpoint = None
        for method in plan["methods"]:
            root = (source_main / "attacks" / plan["dataset"] / source / method / "default").resolve()
            run = _read(root / "run.json")
            expected_job = next(job for job in plan["jobs"] if job["attack"] == method and job["source"] == source)
            main_jobs = [job for job in source_plan.get("jobs", [])
                         if all(job.get(k) == v for k, v in dict(dataset=plan["dataset"], split=plan["split"],
                             source=source, attack=method, study="", variant="default").items())]
            if (not main_jobs or any(job.get("budget_profile") != expected_job["budget_profile"] for job in main_jobs)):
                raise ValueError("Payload job is absent or differs from its source-main plan")
            seeds = {job.get("seed") if job.get("seed") is not None else source_plan["execution_seed"] for job in main_jobs}
            if len(seeds) != 1 or run.get("seed") != next(iter(seeds)):
                raise ValueError("Payload seed differs from its source-main plan")
            required = dict(status="complete", dataset=plan["dataset"], split=plan["split"],
                            source=source, attack=method, failed_images=0,
                            parameters_sha256=expected_job["parameters_sha256"], budget_profile=expected_job["budget_profile"])
            if any(run.get(key) != value for key, value in required.items()):
                raise ValueError("Payload identity, completion or comparison budget differs: " + method)
            if _hash(run.get("parameters")) != expected_job["parameters_sha256"]:
                raise ValueError("Payload resolved parameters are not hash-bound")
            if (type(run.get("seed")) is not int or not re.fullmatch(r"[0-9a-f]{64}", str(run.get("checkpoint_sha256", "")))):
                raise ValueError("Payload source checkpoint or seed is missing")
            if source_checkpoint is None:
                source_checkpoint = run["checkpoint_sha256"]
            if run["checkpoint_sha256"] != source_checkpoint:
                raise ValueError("Compared payloads use different source checkpoints")
            annotation_path = _inside(root, run.get("annotation", "annotations.json"))
            annotation = _read(annotation_path)
            if any(type(row["id"]) is not int for row in annotation["images"]):
                raise ValueError("Payload image IDs must be integers")
            available = [row["id"] for row in annotation["images"]]
            if (available != full_ids[:len(available)] or not selected_set.issubset(available)
                    or (plan["max_images"] is None and available != full_ids)
                    or run.get("successful_images") != len(available) or run.get("requested_images") != len(available)):
                raise ValueError("Payload does not cover its intended image population")
            if annotation != index.adversarial_annotation({row["id"]: row["file_name"] for row in annotation["images"]}):
                raise ValueError("Payload annotations differ from canonical ground truth")
            manifest_path = root / "manifest.jsonl"
            rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
            if len(rows) != len(available):
                raise ValueError("Payload manifest is incomplete")
            image_root = _inside(root, run.get("image_root", "."))
            names = {row["id"]: row["file_name"] for row in annotation["images"]}
            images, seen, seed_mapping = [], set(), []
            for position, (row, image_id) in enumerate(zip(rows, available)):
                if (row.get("status") != "ok" or row.get("position") != position
                        or row.get("image_id") != image_id or row.get("attack_seed") != run["seed"] + position):
                    raise ValueError("Payload manifest order, success or seed differs")
                if image_id not in selected_set:
                    continue
                path = _inside(root, row["output_file"])
                if path != _inside(root, str(image_root / names[image_id])) or path in seen:
                    raise ValueError("Evaluator image path differs or aliases another image")
                bound = _bind(path)
                if bound["sha256"] != row.get("output_sha256") or bound["bytes"] != row.get("output_bytes"):
                    raise ValueError("Payload image bytes differ from the generation manifest")
                seen.add(path)
                images.append(bound)
                seed_mapping.append(dict(image_id=image_id, full_position=position, attack_seed=row["attack_seed"]))
            headers.extend([_bind(root / "run.json"), _bind(annotation_path), _bind(manifest_path)])
            payloads[source][method] = dict(root=str(root), annotation=str(annotation_path), image_root=str(image_root),
                images=images, seed_mapping=seed_mapping, parameters_sha256=expected_job["parameters_sha256"],
                source_checkpoint_sha256=source_checkpoint)
        checkpoints[source] = source_checkpoint
        if source_checkpoint != source_bindings[source]["sha256"]:
            raise ValueError("Payload source checkpoint differs from the current registered COCO source")
    return dict(source_main=str(source_main), selected_image_ids=selected,
        retained_image_ids=[value for value in retained if value in selected_set],
        full_image_ids_sha256=ids_digest(full_ids), retained500_image_ids_sha256=ids_digest(retained),
        numerical_source_sha256=expected_sources, headers=headers, payloads=payloads,
        source_checkpoint_sha256=checkpoints, annotation=str(index.annotation_path),
        source_checkpoint_bindings=source_bindings,
        historical_result_inheritance=False,
        verified_selected_payload_files=sum(len(p["images"]) for methods in payloads.values() for p in methods.values()))
