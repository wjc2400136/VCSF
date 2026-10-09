"""Preserved victim-preprocessing views of authenticated frozen attack payloads."""
import json
from pathlib import Path

from ..io import atomic_json, file_digest
from .preprocessing_defense import _materialize_view, _subset_annotation
from .vcsf_final_oblivious_contract import PROTOCOL, verify_source
from .vcsf_final_training_state_contract import ids_digest


def verify_view(entry):
    """Recheck the exact selected image bytes used by a prepared evaluator view."""
    view = Path(entry["view"]).resolve()
    for path, digest in ((view / "run.json", entry["run_sha256"]),
                         (view / "annotations.json", entry["annotation_sha256"]),
                         (Path(entry["manifest"]), entry["manifest_sha256"])):
        if file_digest(path) != digest:
            raise ValueError("Prepared view metadata changed")
    run = json.loads((view / "run.json").read_text(encoding="utf-8"))
    annotation = json.loads((view / "annotations.json").read_text(encoding="utf-8"))
    manifest = json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
    image_ids = [row["id"] for row in annotation["images"]]
    if ids_digest(image_ids) != run["evaluated_image_ids_sha256"]:
        raise ValueError("View image identities changed")
    if manifest["materialization"] == "reference_only":
        root = Path(run["original_payload"]).resolve()
        binding = run["original_payload_binding"]
        if (file_digest(root / "run.json") != binding["payload_run_sha256"]
                or file_digest(root / "manifest.jsonl") != binding["payload_manifest_sha256"]):
            raise ValueError("Original payload binding changed")
        with (root / "manifest.jsonl").open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
    else:
        root, rows = view, manifest["outputs"]
    mapping = {row["image_id"]: row for row in rows}
    if len(mapping) != len(rows):
        raise ValueError("Duplicate view image identity")
    image_root = (view / run["image_root"]).resolve()
    for image in annotation["images"]:
        row = mapping[image["id"]]
        path = (root / row["output_file"]).resolve()
        inference = (image_root / image["file_name"]).resolve()
        if (root not in path.parents or path != inference
                or path.stat().st_size != row["output_bytes"]
                or file_digest(path) != row["output_sha256"]):
            raise ValueError("View image path or bytes changed")


def materialize_source(registry, source, payload, output, max_images=None):
    """Create all registered views without model calls or destructive cleanup."""
    spec = registry.protocols[PROTOCOL]
    if max_images is not None and (type(max_images) is not int
            or not 0 < max_images <= spec["retained_images"]):
        raise ValueError("Diagnostic limit must be within the retained population")
    payload, output = Path(payload).resolve(), Path(output).resolve()
    protected = [payload, registry.root / "data", registry.root / "checkpoints"]
    for path in protected:
        for parent, child in ((path.resolve(), output), (output, path.resolve())):
            if parent == child or parent in child.parents:
                raise ValueError("Output overlaps an immutable input root")
    verified = verify_source(registry, source, payload)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "input_verification.json", verified)
    try:
        image_ids = verified["retained_image_ids"]
        if max_images is not None:
            image_ids = image_ids[:max_images]
        annotation = json.loads(Path(verified["annotation"]).read_text(encoding="utf-8"))
        annotation = _subset_annotation(annotation, image_ids)
        original = json.loads((payload / "run.json").read_text(encoding="utf-8"))
        image_root = (payload / original.get("image_root", ".")).resolve()
        binding = dict(spec["payload_bindings"][source])
        views = []
        for defense in registry.preprocessing_defense_order:
            manifest = output / "manifests" / (defense + ".json")
            view = _materialize_view(registry, output, output / "scratch" / defense,
                manifest, annotation, image_root, source, "vcsf", defense,
                registry.preprocessing_defenses[defense], protocol_id=PROTOCOL,
                threat_model=registry.preprocessing_defense_threat_model,
                transform_implementation=registry.preprocessing_defense_policy)
            run_path = view / "run.json"
            run = json.loads(run_path.read_text(encoding="utf-8"))
            run.update(parameters_sha256=spec["parameters_sha256"],
                original_payload_binding=binding, original_payload=str(payload),
                original_annotation_sha256=spec["annotation_sha256"],
                original_full_image_ids_sha256=spec["full_image_ids_sha256"],
                evaluated_image_ids_sha256=ids_digest(image_ids),
                formal_scope=max_images is None, diagnostic_max_images=max_images,
                original_seed_mapping=[row for row in verified["retained_seed_mapping"]
                                       if row["image_id"] in set(image_ids)])
            atomic_json(run_path, run)
            views.append(dict(defense=defense, view=str(view),
                run_sha256=file_digest(run_path),
                annotation_sha256=file_digest(view / "annotations.json"),
                manifest=str(manifest), manifest_sha256=file_digest(manifest)))
        if verify_source(registry, source, payload) != verified:
            raise ValueError("Original payload changed during materialization")
        receipt = dict(status="views_prepared_pending_evaluation", protocol=PROTOCOL,
            source=source, parameters_sha256=spec["parameters_sha256"],
            original_payload_binding=binding, image_ids=image_ids,
            image_ids_sha256=ids_digest(image_ids), views=views,
            formal_scope=max_images is None, model_calls=0,
            formal_result_eligible=False, scientific_acceptance=False)
        atomic_json(output / "views.json", receipt)
        atomic_json(output / "terminal.json", dict(status=receipt["status"],
            views_sha256=file_digest(output / "views.json"), formal_result_eligible=False))
        return receipt
    except BaseException as exc:
        atomic_json(output / "terminal.json", dict(status="failed", error=repr(exc),
            formal_result_eligible=False))
        raise
