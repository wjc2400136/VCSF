"""Publish full-split operator inputs without execution or reuse authority."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

from ..io import file_digest
from ..registry import Registry
from . import vcsf_ablation_execution_contract as source_binding
from .vcsf_ablation_runtime_inputs import (
    NORMALIZATION, _check_file, _effective_config, _file, _head, _index,
    _pinned_json, _read_json, model_roles,
)
from .vcsf_ablation_preflight import required_packages, verify_server
from .vcsf_efficacy_contract import child, require, verify_environment
from .vcsf_operator_factorial_plan import PROTOCOL, compile_operator_factorial_plan
from .vcsf_research_plan import canonical_hash


ENTRYPOINT = "experiments/prepare_vcsf_operator_inputs.py"
INPUT_NAMESPACE = PROTOCOL + "_inputs"
RECORD_TYPE = "vcsf_operator_runtime_input_binding"
STATUS = "operator_actual_inputs_bound_not_reuse_or_execution_admission"
_ROOT = Path(__file__).resolve().parents[3]
_ENTRY_HASH = file_digest(_ROOT / ENTRYPOINT)
_FALSE_FLAGS = (
    "formal_execution_admission", "independent_acceptance", "scientific_acceptance",
    "device_mode_qualified", "historical_reuse_qualified", "physical_cost_calibration_accepted",
    "independent_confirmation",
)
_BINDING_KEYS = {
    "schema_version", "record_type", "protocol", "status", "created_at", "operator_plan_file",
    "science_sha256", "dataset", "split", "seed", "seed_schedule", "full_split_images",
    "max_images", "diagnostic_only", "ordered_image_ids", "image_ids_sha256", "seed_mapping_sha256",
    "annotation", "clean_image_manifest", "checkpoints", "model_configs", "base_commit",
    "python_version", "packages", "runtime_sha256", "runtime_snapshot_sha256", "binding_sha256",
    "new_group_ids", "reused_group_ids", "model_calls", "AP_evaluations",
} | set(_FALSE_FLAGS)


def _snapshot(registry):
    # Reuse source-byte checks, never the core protocol's acceptance routine.
    source_binding._verify_process_binding(registry)
    entry = sys.modules.get("__main__")
    require(registry.root == _ROOT
        and getattr(entry, "__file__", None) == str(_ROOT / ENTRYPOINT)
        and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == _ENTRY_HASH
        and file_digest(_ROOT / ENTRYPOINT) == _ENTRY_HASH,
        "Operator inputs require the unchanged source-bound public entry")
    return dict(sorted(dict(source_binding._IMPORTED_SOURCE, **{ENTRYPOINT: _ENTRY_HASH}).items()))


def read_operator_plan(registry, path, digest):
    """Rebuild the actual operator plan; matching counts or core plans are insufficient."""
    pointer = _file(path)
    require(pointer["sha256"] == digest, "Operator plan file hash differs")
    saved = _read_json(pointer)
    require(isinstance(saved, dict) and isinstance(saved.get("scientific_contract"), dict),
        "Missing operator scientific plan")
    science = saved["scientific_contract"]
    current = compile_operator_factorial_plan(registry,
        max_images=science.get("max_images"), devices=saved.get("requested_devices"))
    require(canonical_hash(saved) == canonical_hash(current),
        "Operator plan is not the exact current unarmed preparation")
    _snapshot(registry)
    return current, pointer


def preview_operator_inputs(registry, plan_path, plan_sha256):
    plan, pointer = read_operator_plan(registry, plan_path, plan_sha256)
    return dict(status="operator_input_preview_no_dataset_or_checkpoint_reads",
        operator_plan_file=pointer, science_sha256=plan["science_sha256"],
        full_split_images=5000, max_images=plan["scientific_contract"]["max_images"],
        requested_devices=plan["requested_devices"], device_availability_checked=False,
        model_roles=[dict(role=role, model=model) for role, model in model_roles(plan)],
        logical_group_ids=[g["group_id"] for g in plan["scientific_contract"]["groups"]],
        new_group_ids=None, reused_group_ids=None, output_created=False,
        formal_execution_admission=False, historical_reuse_qualified=False,
        model_calls=0, AP_evaluations=0)


def _output(registry, path):
    output = source_binding._plain(path)
    require(output.parent == source_binding._plain(registry.root) / "outputs/plans" / INPUT_NAMESPACE,
        "Operator input publication requires a direct leaf in its own namespace")
    return output


def _artifact_names(plan):
    return {"clean_images.json", "runtime_inputs.json"} | {
        "model_configs/{}/{}/{}".format(role, model, name)
        for role, model in model_roles(plan) for name in ("runtime_config.py", "effective_config.json")}


def _receipt(binding, file_sha256, manifest_sha256):
    return dict(status=STATUS, protocol=PROTOCOL, binding_sha256=binding["binding_sha256"],
        science_sha256=binding["science_sha256"], runtime_inputs_file_sha256=file_sha256,
        artifact_manifest_sha256=manifest_sha256, full_split_images=5000, checkpoint_count=16,
        role_config_count=17, diagnostic_only=binding["diagnostic_only"],
        formal_execution_admission=False, historical_reuse_qualified=False,
        independent_acceptance=False, scientific_acceptance=False, model_calls=0, AP_evaluations=0)


def verify_operator_inputs(registry, binding, plan):
    """Verify publication content; consumers must use the complete published verifier."""
    verify_server()
    require(sys.version_info[:3] == (3, 8, 20), "Operator input verification requires Python 3.8.20")
    require(isinstance(binding, dict) and set(binding) == _BINDING_KEYS
        and type(binding["schema_version"]) is int and binding["schema_version"] == 1
        and binding["record_type"] == RECORD_TYPE and binding["protocol"] == PROTOCOL
        and binding["status"] == STATUS, "Wrong operator input binding schema or namespace")
    pointer = binding["operator_plan_file"]
    _check_file(pointer)
    current, actual_pointer = read_operator_plan(registry, pointer["file"], pointer["sha256"])
    science = current["scientific_contract"]
    require(canonical_hash(plan) == canonical_hash(current) and pointer == actual_pointer
        and binding["science_sha256"] == current["science_sha256"]
        and binding["dataset"] == "coco" and binding["split"] == "val"
        and type(binding["seed"]) is int and binding["seed"] == 42
        and binding["seed_schedule"] == "selected_position"
        and type(binding["full_split_images"]) is int and binding["full_split_images"] == 5000
        and type(binding["max_images"]) is type(science["max_images"])
        and binding["max_images"] == science["max_images"]
        and binding["diagnostic_only"] is science["diagnostic_only"],
        "Operator input binding changed its scientific scope")
    require(all(binding[key] is False for key in _FALSE_FLAGS)
        and binding["new_group_ids"] is None and binding["reused_group_ids"] is None
        and all(type(binding[key]) is int and binding[key] == 0 for key in ("model_calls", "AP_evaluations")),
        "Input publication cannot grant reuse, execution, device or scientific acceptance")
    require(binding["binding_sha256"] == canonical_hash({k: v for k, v in binding.items() if k != "binding_sha256"}),
        "Operator binding content changed")
    require(isinstance(binding["created_at"], str), "Missing operator publication time")
    created = datetime.fromisoformat(binding["created_at"])
    require(created.tzinfo is not None and created.utcoffset().total_seconds() == 0,
        "Operator publication time requires explicit UTC")
    require(binding["base_commit"] == _head(registry.root)
        and binding["python_version"] == "3.8.20"
        and binding["packages"] == required_packages(registry.root), "Pinned input runtime identity changed")
    verify_environment(binding)
    index, ids = _index(registry)
    require(canonical_hash(binding["ordered_image_ids"]) == canonical_hash(ids)
        and binding["image_ids_sha256"] == canonical_hash(ids)
        and canonical_hash(binding["annotation"]) == canonical_hash(_file(index.annotation_path))
        and binding["seed_mapping_sha256"] == canonical_hash([[value, i, 42+i] for i, value in enumerate(ids)]),
        "Full canonical image, annotation or seed identity changed")
    manifest_pointer = binding["clean_image_manifest"]
    output = _output(registry, source_binding._plain(manifest_pointer["file"]).parent)
    require(manifest_pointer["file"] == str(output / "clean_images.json"),
        "Operator clean-image manifest has another filename")
    rows = _read_json(manifest_pointer)
    require(isinstance(rows, list) and len(rows) == 5000, "Incomplete operator clean-image manifest")
    for position, (row, image) in enumerate(zip(rows, index.images)):
        require(isinstance(row, dict) and set(row) == {"image_id", "position", "attack_seed", "input"}
            and all(type(row[key]) is int and row[key] == expected for key, expected in
                (("image_id", ids[position]), ("position", position), ("attack_seed", 42+position)))
            and canonical_hash(row["input"]) == canonical_hash(_file(index.image_path(image))),
            "Operator clean-image bytes, canonical path, order or per-image seed differ")
    from ..modeling import checkpoint_path

    targets, checkpoints = science["targets"], binding["checkpoints"]
    require(isinstance(checkpoints, dict) and set(checkpoints) == set(targets),
        "Operator inputs require the complete checkpoint panel")
    for target in targets:
        require(canonical_hash(checkpoints[target]) == canonical_hash(_file(checkpoint_path(registry.model(target),
            registry.dataset("coco")).resolve(strict=True))), "Operator checkpoint bytes or canonical path differ")
    configs = binding["model_configs"]
    require(isinstance(configs, list) and [(r.get("role"), r.get("model")) for r in configs] == model_roles(plan),
        "Operator source/target role configs are incomplete or reordered")
    for row in configs:
        directory = output / "model_configs" / row["role"] / row["model"]
        require(set(row) == {"role", "model", "checkpoint_sha256", "resolved_config", "effective_config",
                "effective_config_sha256", "normalization"}
            and row["normalization"] == NORMALIZATION
            and row["checkpoint_sha256"] == checkpoints[row["model"]]["sha256"]
            and row["resolved_config"]["file"] == str(directory / "runtime_config.py")
            and row["effective_config"]["file"] == str(directory / "effective_config.json"),
            "Operator model-role config binding differs")
        effective = _read_json(row["effective_config"])
        _check_file(row["resolved_config"])
        actual = _effective_config(registry, row["model"], directory, dump=False,
            expected_resolved=row["resolved_config"]["file"])
        require(row["effective_config_sha256"] == canonical_hash(effective) == canonical_hash(actual),
            "Operator model/preprocessing/evaluator config changed")
    snapshot = _snapshot(registry)
    require(binding["runtime_sha256"] == snapshot
        and binding["runtime_snapshot_sha256"] == canonical_hash(snapshot), "Operator input source snapshot changed")
    return binding


def verify_published_operator_inputs(registry, binding_path, binding_sha256, receipt_path, receipt_sha256):
    binding_path, receipt_path = source_binding._plain(binding_path), source_binding._plain(receipt_path)
    output = _output(registry, binding_path.parent)
    require(binding_path.name == "runtime_inputs.json" and receipt_path == output / "runtime_inputs_receipt.json",
        "Operator input publication and receipt require the same direct leaf")
    binding = _pinned_json(binding_path, binding_sha256)
    require(isinstance(binding, dict) and binding.get("clean_image_manifest", {}).get("file") == str(output / "clean_images.json"),
        "Operator input binding references another publication leaf")
    pointer = binding["operator_plan_file"]
    plan, _ = read_operator_plan(registry, pointer["file"], pointer["sha256"])
    receipt = _pinned_json(receipt_path, receipt_sha256)
    require(isinstance(receipt, dict) and canonical_hash(receipt) == canonical_hash(
        _receipt(binding, binding_sha256, receipt.get("artifact_manifest_sha256"))),
        "Operator input receipt differs from its bound publication")
    manifest_path = output / "artifact_manifest.json"
    manifest = _pinned_json(manifest_path, receipt["artifact_manifest_sha256"])
    require(isinstance(manifest, dict) and set(manifest) == {"artifacts"}
        and isinstance(manifest["artifacts"], dict) and set(manifest["artifacts"]) == _artifact_names(plan),
        "Operator input artifact manifest is incomplete or contains foreign files")
    expected = set(manifest["artifacts"]) | {"artifact_manifest.json", "runtime_inputs_receipt.json"}

    def unchanged():
        require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected
            and not any(p.is_symlink() for p in output.rglob("*")), "Operator input leaf is partial or mixed")
        for name, value in manifest["artifacts"].items():
            require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}, "Invalid operator artifact row")
            _check_file(dict(value, file=str(child(output, name))))
        _pinned_json(manifest_path, receipt["artifact_manifest_sha256"])
        _pinned_json(receipt_path, receipt_sha256)
        _pinned_json(binding_path, binding_sha256)

    unchanged()
    verify_operator_inputs(registry, binding, plan)
    unchanged()
    return binding


def bind_operator_inputs(registry, output, plan_path, plan_sha256):
    verify_server()
    plan, pointer = read_operator_plan(registry, plan_path, plan_sha256)
    output = _output(registry, output)
    require(not output.exists(), "Operator inputs require a fresh publication leaf")
    packages = required_packages(registry.root)
    verify_environment(dict(packages=packages))
    require(sys.version_info[:3] == (3, 8, 20), "Operator binding requires Python 3.8.20")
    head, snapshot = _head(registry.root), _snapshot(registry)
    index, ids = _index(registry)
    from ..modeling import checkpoint_path

    output.mkdir(parents=True, exist_ok=False)
    try:
        rows = [dict(image_id=ids[i], position=i, attack_seed=42+i, input=_file(index.image_path(image)))
            for i, image in enumerate(index.images)]
        source_binding._write_json(output / "clean_images.json", rows)
        checkpoints = {name: _file(checkpoint_path(registry.model(name), registry.dataset("coco")).resolve(strict=True))
            for name in plan["scientific_contract"]["targets"]}
        configs = []
        for role, model in model_roles(plan):
            directory = output / "model_configs" / role / model
            directory.mkdir(parents=True, exist_ok=False)
            effective = _effective_config(registry, model, directory, dump=True)
            source_binding._write_json(directory / "effective_config.json", effective)
            configs.append(dict(role=role, model=model, checkpoint_sha256=checkpoints[model]["sha256"],
                resolved_config=_file(directory / "runtime_config.py"), effective_config=_file(directory / "effective_config.json"),
                effective_config_sha256=canonical_hash(effective), normalization=NORMALIZATION))
        science = plan["scientific_contract"]
        binding = dict(schema_version=1, record_type=RECORD_TYPE, protocol=PROTOCOL, status=STATUS,
            created_at=datetime.now(timezone.utc).isoformat(), operator_plan_file=pointer, science_sha256=plan["science_sha256"],
            dataset="coco", split="val", seed=42, seed_schedule="selected_position", full_split_images=5000,
            max_images=science["max_images"], diagnostic_only=science["diagnostic_only"], ordered_image_ids=ids,
            image_ids_sha256=canonical_hash(ids), seed_mapping_sha256=canonical_hash([[value, i, 42+i] for i, value in enumerate(ids)]),
            annotation=_file(index.annotation_path), clean_image_manifest=_file(output / "clean_images.json"),
            checkpoints=checkpoints, model_configs=configs, base_commit=head, python_version="3.8.20", packages=packages,
            runtime_sha256=deepcopy(snapshot), runtime_snapshot_sha256=canonical_hash(snapshot),
            new_group_ids=None, reused_group_ids=None, model_calls=0, AP_evaluations=0,
            **{key: False for key in _FALSE_FLAGS})
        binding["binding_sha256"] = canonical_hash(binding)
        verify_operator_inputs(registry, binding, plan)
        source_binding._write_json(output / "runtime_inputs.json", binding)
        artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=file_digest(p))
            for p in sorted(output.rglob("*")) if p.is_file()}
        require(set(artifacts) == _artifact_names(plan) and not any(p.is_symlink() for p in output.rglob("*")),
            "Unexpected operator input artifacts before publication")
        manifest = source_binding._write_json(output / "artifact_manifest.json", dict(artifacts=artifacts))
        verify_operator_inputs(registry, binding, plan)
        for name, value in artifacts.items():
            _check_file(dict(value, file=str(child(output, name))))
        require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
            == set(artifacts) | {"artifact_manifest.json"}
            and not any(p.is_symlink() for p in output.rglob("*"))
            and file_digest(output / "artifact_manifest.json") == manifest["sha256"],
            "Operator publication changed before terminal receipt")
        receipt = _receipt(binding, file_digest(output / "runtime_inputs.json"), manifest["sha256"])
        source_binding._write_json(output / "runtime_inputs_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        if not (output / "runtime_inputs_receipt.json").exists():
            source_binding._write_json(output / "binding_failure.json", dict(status="failed_partial_evidence_preserved",
                exception_type=type(exc).__name__, formal_execution_admission=False, historical_reuse_qualified=False))
        raise
