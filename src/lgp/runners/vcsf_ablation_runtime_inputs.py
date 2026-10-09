"""Bind actual full-split inputs for the core executor without granting admission."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

from ..io import file_digest
from .vcsf_ablation_execution_contract import (
    CONTRACT_PROTOCOL, _plain, _verify_process_binding, _write_json,
    verify_execution_contract,
)
from .vcsf_ablation_preflight import required_packages, verify_server
from .vcsf_efficacy_contract import child, require, verify_environment
from .vcsf_research_plan import canonical_hash


RECORD_TYPE = "vcsf_core_runtime_input_binding"
NORMALIZATION = "exclude_work_dir_json_sequences_preserve_other_fields"
_BINDING_KEYS = {
    "schema_version", "record_type", "status", "created_at", "contract_sha256", "contract_source_sha256",
    "execution_contract_file", "synthetic_stage", "dataset", "split", "seed", "seed_schedule", "full_split_images",
    "max_images", "ordered_image_ids", "image_ids_sha256", "seed_mapping_sha256", "annotation",
    "clean_image_manifest", "checkpoints", "model_configs", "base_commit", "python_version", "packages",
    "runtime_sha256", "runtime_snapshot_sha256", "new_group_ids", "reused_group_ids",
    "formal_execution_admission", "independent_acceptance", "scientific_acceptance", "device_mode_qualified",
    "historical_reuse_qualified", "physical_cost_calibration_accepted", "independent_confirmation",
    "model_calls", "AP_evaluations", "binding_sha256",
}


def _normal(value):
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Model config has a non-string key")
        return {key: _normal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normal(item) for item in value]
    require(value is None or type(value) in (str, bool, int, float),
        "Model config contains a non-declarative value")
    require(type(value) is not float or math.isfinite(value), "Model config contains a non-finite value")
    return value


def model_roles(contract):
    science = contract["scientific_contract"]
    return [("source", science["sources"][0])] + [("target", name) for name in science["targets"]]


def preview_runtime_inputs(registry, contract):
    current = verify_execution_contract(registry, contract)
    require(current["scientific_contract"]["selected_stage"] == "core",
        "Runtime input binding currently covers the exact core stage")
    return dict(status="core_runtime_input_binding_preview_no_input_reads",
        contract_sha256=current["contract_sha256"], synthetic=current["synthetic"],
        full_split_images=5000, model_roles=[dict(role=r, model=m) for r, m in model_roles(current)],
        logical_groups=current["logical_group_ids"], new_group_ids=None, reused_group_ids=None,
        model_calls=0, AP_evaluations=0, output_created=False,
        formal_execution_admission=False, independent_acceptance=False)


def _file(path):
    path = _plain(path)
    require(path.is_file(), "Missing actual runtime input: " + path.name)
    before = path.stat()
    digest = file_digest(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        "Runtime input changed while hashing: " + path.name)
    require(after.st_size > 0, "Runtime input is empty: " + path.name)
    return dict(file=str(path), bytes=after.st_size, sha256=digest)


def _check_file(value):
    require(isinstance(value, dict) and set(value) == {"file", "bytes", "sha256"}
        and isinstance(value["file"], str) and bool(value["file"])
        and type(value["bytes"]) is int and value["bytes"] > 0
        and isinstance(value["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]),
        "Invalid runtime file binding")
    require(_file(value["file"]) == value, "Actual runtime input bytes changed")


def _index(registry):
    from ..data.coco import CocoIndex

    index = CocoIndex(registry.dataset("coco"), "val")
    ids = [image["id"] for image in index.images]
    require(len(ids) == len(set(ids)) == 5000 and ids == sorted(ids)
        and all(type(value) is int and value > 0 for value in ids),
        "Runtime binding requires all 5000 unique canonical ordered validation images")
    return index, ids


def _head(root):
    value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", value), "Runtime binding requires the original Git revision")
    return value


def _effective_config(registry, model, directory, *, dump, expected_resolved=None):
    from ..runtime_config import build_runtime_config

    config = build_runtime_config(registry.model(model), registry.dataset("coco"), directory,
        mode="test", test_split="val", dump_config=dump)
    if expected_resolved is not None:
        require(Path(expected_resolved).read_bytes() == config.pretty_text.encode("utf-8"),
            "Saved resolved config is not the reconstructed runtime config")
    value = config.to_dict()
    value.pop("work_dir", None)
    return _normal(value)


def _read_json(pointer):
    _check_file(pointer)
    raw = Path(pointer["file"]).read_bytes()
    require(len(raw) == pointer["bytes"] and hashlib.sha256(raw).hexdigest() == pointer["sha256"],
        "Runtime JSON bytes changed before parsing")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Runtime JSON contains duplicate object keys")
            result[key] = value
        return result

    def constant(value):
        raise RuntimeError("Runtime JSON contains a non-finite value")

    return json.loads(raw.decode("utf-8"),
        object_pairs_hook=pairs, parse_constant=constant)


def verify_runtime_inputs(registry, binding, contract):
    """Check content during publication; use verify_published_runtime_inputs for consumption."""
    verify_server()
    current = verify_execution_contract(registry, contract)
    require(isinstance(binding, dict) and set(binding) == _BINDING_KEYS
        and binding.get("record_type") == RECORD_TYPE
        and type(binding.get("schema_version")) is int and binding["schema_version"] == 1
        and binding.get("status") == "bound_actual_inputs_not_execution_admission"
        and binding.get("contract_sha256") == current["contract_sha256"]
        and binding.get("contract_source_sha256") == current["source_snapshot_sha256"]
        and binding.get("synthetic_stage") is current["synthetic"]
        and binding.get("dataset") == "coco" and binding.get("split") == "val"
        and binding.get("seed_schedule") == "selected_position"
        and type(binding.get("seed")) is int and binding["seed"] == 42
        and type(binding.get("full_split_images")) is int and binding["full_split_images"] == 5000
        and binding.get("max_images") == current["scientific_contract"]["max_images"]
        and type(binding.get("max_images")) is type(current["scientific_contract"]["max_images"]),
        "Runtime inputs belong to another contract or scientific scope")
    require(isinstance(binding["created_at"], str), "Runtime binding creation time is missing")
    created_at = datetime.fromisoformat(binding["created_at"])
    require(created_at.tzinfo is not None and created_at.utcoffset().total_seconds() == 0,
        "Runtime binding creation time must be explicit UTC")
    require(binding.get("new_group_ids") is None and binding.get("reused_group_ids") is None
        and all(binding.get(key) is False for key in ("formal_execution_admission", "independent_acceptance",
            "scientific_acceptance", "device_mode_qualified", "historical_reuse_qualified",
            "physical_cost_calibration_accepted", "independent_confirmation"))
        and all(type(binding.get(key)) is int and binding[key] == 0 for key in ("model_calls", "AP_evaluations")),
        "Input binding cannot grant execution, reuse, cost or scientific acceptance")
    require(binding.get("binding_sha256") == canonical_hash({key: value for key, value in binding.items()
        if key != "binding_sha256"}), "Runtime binding content changed")
    original = _read_json(binding["execution_contract_file"])
    require(canonical_hash(original) == canonical_hash(current), "Original execution contract file differs")
    require(binding.get("base_commit") == _head(registry.root)
        and binding.get("python_version") == "3.8.20"
        and binding.get("packages") == required_packages(registry.root), "Pinned runtime identity changed")
    verify_environment(binding)
    index, ids = _index(registry)
    require(canonical_hash(binding.get("annotation")) == canonical_hash(_file(index.annotation_path))
        and canonical_hash(binding.get("ordered_image_ids")) == canonical_hash(ids)
        and binding.get("image_ids_sha256") == canonical_hash(ids), "Canonical full-split identity changed")
    rows = _read_json(binding["clean_image_manifest"])
    output = _plain(binding["clean_image_manifest"]["file"]).parent
    require(output.parent == _plain(registry.root) / "outputs/plans" / CONTRACT_PROTOCOL
        and binding["clean_image_manifest"]["file"] == str(output / "clean_images.json"),
        "Runtime input manifest escaped its binding output namespace")
    require(isinstance(rows, list) and len(rows) == 5000, "Missing full clean-image manifest")
    for position, (row, image) in enumerate(zip(rows, index.images)):
        require(set(row) == {"image_id", "position", "attack_seed", "input"}
            and all(type(row.get(key)) is int and row[key] == value for key, value in
                (("image_id", ids[position]), ("position", position), ("attack_seed", 42 + position)))
            and canonical_hash(row["input"]) == canonical_hash(_file(index.image_path(image))),
            "Clean-image order, seed, canonical path or bytes changed")
    require(binding.get("seed_mapping_sha256") == canonical_hash([[value, i, 42 + i] for i, value in enumerate(ids)]),
        "Selected-position seed mapping changed")
    from ..modeling import checkpoint_path

    checkpoints = binding.get("checkpoints")
    targets = current["scientific_contract"]["targets"]
    require(isinstance(checkpoints, dict) and set(checkpoints) == set(targets),
        "Runtime inputs require the complete canonical checkpoint panel")
    for target in targets:
        require(canonical_hash(checkpoints[target]) == canonical_hash(_file(
            checkpoint_path(registry.model(target), registry.dataset("coco")).resolve(strict=True))),
            "A checkpoint changed or uses a noncanonical path")
    configs = binding.get("model_configs")
    roles = model_roles(current)
    require(isinstance(configs, list) and [(r.get("role"), r.get("model")) for r in configs] == roles,
        "Source/target role config slots are missing, duplicated or reordered")
    for row in configs:
        directory = output / "model_configs" / row["role"] / row["model"]
        require(set(row) == {"role", "model", "checkpoint_sha256", "resolved_config", "effective_config",
                "effective_config_sha256", "normalization"}
            and row.get("normalization") == NORMALIZATION
            and row["resolved_config"]["file"] == str(directory / "runtime_config.py")
            and row["effective_config"]["file"] == str(directory / "effective_config.json")
            and row.get("checkpoint_sha256") == checkpoints[row["model"]]["sha256"],
            "Model-role checkpoint or normalization changed")
        effective = _read_json(row["effective_config"])
        _check_file(row["resolved_config"])
        require(row.get("effective_config_sha256") == canonical_hash(effective), "Effective model config differs")
        actual = _effective_config(registry, row["model"], Path(row["resolved_config"]["file"]).parent,
            dump=False, expected_resolved=row["resolved_config"]["file"])
        require(canonical_hash(actual) == row["effective_config_sha256"],
            "Reconstructed model/preprocessing/evaluator config changed")
    require(binding.get("runtime_sha256") == current["source_sha256"]
        and binding.get("runtime_snapshot_sha256") == canonical_hash(current["source_sha256"]),
        "Executed source/configuration snapshot differs")
    _verify_process_binding(registry)
    return binding


def _artifact_names(contract):
    return {"clean_images.json", "runtime_inputs.json"} | {
        "model_configs/{}/{}/{}".format(role, model, filename)
        for role, model in model_roles(contract) for filename in ("runtime_config.py", "effective_config.json")}


def _receipt(binding, binding_file_sha256, manifest_sha256):
    return dict(status="actual_runtime_inputs_bound_not_independent_acceptance",
        binding_sha256=binding["binding_sha256"], runtime_inputs_file_sha256=binding_file_sha256,
        artifact_manifest_sha256=manifest_sha256, full_split_images=5000, checkpoint_count=16,
        role_config_count=17, synthetic_stage=binding["synthetic_stage"], model_calls=0, AP_evaluations=0,
        formal_execution_admission=False, independent_acceptance=False, scientific_acceptance=False)


def _pinned_json(path, digest):
    path = _plain(path)
    require(path.is_file(), "Missing published runtime input artifact")
    return _read_json(dict(file=str(path), bytes=path.stat().st_size, sha256=digest))


def verify_published_runtime_inputs(registry, binding_path, binding_sha256, receipt_path, receipt_sha256, contract):
    """Consume the original complete publication, not an in-memory or partial binding."""
    binding_path, receipt_path = _plain(binding_path), _plain(receipt_path)
    output = binding_path.parent
    require(binding_path.name == "runtime_inputs.json"
        and receipt_path == output / "runtime_inputs_receipt.json"
        and output.parent == _plain(registry.root) / "outputs/plans" / CONTRACT_PROTOCOL,
        "Published runtime inputs and receipt must use the same direct binding leaf")
    binding = _pinned_json(binding_path, binding_sha256)
    require(isinstance(binding, dict)
        and binding.get("clean_image_manifest", {}).get("file") == str(output / "clean_images.json"),
        "Published runtime binding points to another output leaf")
    receipt = _pinned_json(receipt_path, receipt_sha256)
    require(isinstance(receipt, dict) and canonical_hash(receipt) == canonical_hash(_receipt(binding, binding_sha256,
        receipt.get("artifact_manifest_sha256"))), "Runtime input receipt identity or scope differs")
    manifest_path = output / "artifact_manifest.json"
    manifest = _pinned_json(manifest_path, receipt["artifact_manifest_sha256"])
    require(isinstance(manifest, dict) and set(manifest) == {"artifacts"}
        and isinstance(manifest["artifacts"], dict)
        and set(manifest["artifacts"]) == _artifact_names(contract), "Runtime input manifest has an incomplete file set")
    expected = set(manifest["artifacts"]) | {"artifact_manifest.json", "runtime_inputs_receipt.json"}

    def artifacts_unchanged():
        require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected
            and not any(p.is_symlink() for p in output.rglob("*")),
            "Published runtime input leaf is partial, mixed or contains unexpected artifacts")
        for name, value in manifest["artifacts"].items():
            require(isinstance(value, dict) and set(value) == {"bytes", "sha256"}, "Invalid runtime artifact row")
            _check_file(dict(value, file=str(child(output, name))))
        _pinned_json(manifest_path, receipt["artifact_manifest_sha256"])
        _pinned_json(receipt_path, receipt_sha256)
        _pinned_json(binding_path, binding_sha256)

    artifacts_unchanged()
    verify_runtime_inputs(registry, binding, contract)
    artifacts_unchanged()
    return binding


def bind_runtime_inputs(registry, output, contract, contract_path, contract_sha256):
    """Publish actual immutable inputs. Failures remain in their original fresh leaf."""
    verify_server()
    current = verify_execution_contract(registry, contract)
    source = _file(contract_path)
    require(source["sha256"] == contract_sha256
        and canonical_hash(_read_json(source)) == canonical_hash(current), "Execution contract file pin differs")
    output = _plain(output)
    parent = _plain(registry.root) / "outputs/plans" / CONTRACT_PROTOCOL
    require(output.parent == parent and not output.exists(), "Runtime input binding requires a fresh direct output leaf")
    packages = required_packages(registry.root)
    verify_environment(dict(packages=packages))
    require(sys.version_info[:3] == (3, 8, 20), "Runtime input binding requires pinned Python 3.8.20")
    head = _head(registry.root)
    index, ids = _index(registry)
    from ..modeling import checkpoint_path

    output.mkdir(parents=True, exist_ok=False)
    try:
        rows = [dict(image_id=ids[i], position=i, attack_seed=42 + i, input=_file(index.image_path(image)))
            for i, image in enumerate(index.images)]
        _write_json(output / "clean_images.json", rows)
        targets = current["scientific_contract"]["targets"]
        checkpoints = {name: _file(checkpoint_path(registry.model(name), registry.dataset("coco")).resolve(strict=True))
            for name in targets}
        configs = []
        for role, model in model_roles(current):
            directory = output / "model_configs" / role / model
            directory.mkdir(parents=True, exist_ok=False)
            effective = _effective_config(registry, model, directory, dump=True)
            _write_json(directory / "effective_config.json", effective)
            configs.append(dict(role=role, model=model, checkpoint_sha256=checkpoints[model]["sha256"],
                resolved_config=_file(directory / "runtime_config.py"),
                effective_config=_file(directory / "effective_config.json"),
                effective_config_sha256=canonical_hash(effective), normalization=NORMALIZATION))
        binding = dict(schema_version=1, record_type=RECORD_TYPE,
            status="bound_actual_inputs_not_execution_admission", created_at=datetime.now(timezone.utc).isoformat(),
            contract_sha256=current["contract_sha256"], contract_source_sha256=current["source_snapshot_sha256"],
            execution_contract_file=source, synthetic_stage=current["synthetic"], dataset="coco", split="val",
            seed=42, seed_schedule="selected_position", full_split_images=5000,
            max_images=current["scientific_contract"]["max_images"], ordered_image_ids=ids,
            image_ids_sha256=canonical_hash(ids), seed_mapping_sha256=canonical_hash(
                [[value, i, 42 + i] for i, value in enumerate(ids)]), annotation=_file(index.annotation_path),
            clean_image_manifest=_file(output / "clean_images.json"), checkpoints=checkpoints, model_configs=configs,
            base_commit=head, python_version="3.8.20", packages=packages,
            runtime_sha256=deepcopy(current["source_sha256"]),
            runtime_snapshot_sha256=canonical_hash(current["source_sha256"]),
            new_group_ids=None, reused_group_ids=None, formal_execution_admission=False,
            independent_acceptance=False, scientific_acceptance=False, device_mode_qualified=False,
            historical_reuse_qualified=False, physical_cost_calibration_accepted=False,
            independent_confirmation=False, model_calls=0, AP_evaluations=0)
        binding["binding_sha256"] = canonical_hash(binding)
        verify_runtime_inputs(registry, binding, current)
        _write_json(output / "runtime_inputs.json", binding)
        expected_files = _artifact_names(current)
        artifacts = {p.relative_to(output).as_posix(): dict(bytes=p.stat().st_size, sha256=file_digest(p))
            for p in sorted(output.rglob("*")) if p.is_file()}
        require(set(artifacts) == expected_files and not any(p.is_symlink() for p in output.rglob("*")),
            "Unexpected artifact before runtime input manifest publication")
        manifest = _write_json(output / "artifact_manifest.json", dict(artifacts=artifacts))
        verify_runtime_inputs(registry, binding, current)
        expected = set(artifacts) | {"artifact_manifest.json"}
        require({p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} == expected
            and not any(p.is_symlink() for p in output.rglob("*")), "Runtime binding artifact set changed")
        for name, value in artifacts.items():
            path = child(output, name)
            require(path.stat().st_size == value["bytes"] and file_digest(path) == value["sha256"],
                "Runtime binding artifact bytes changed before completion")
        require(file_digest(output / "artifact_manifest.json") == manifest["sha256"], "Runtime input manifest changed")
        receipt = _receipt(binding, file_digest(output / "runtime_inputs.json"), manifest["sha256"])
        _write_json(output / "runtime_inputs_receipt.json", receipt)
        return receipt
    except BaseException as exc:
        if not (output / "runtime_inputs_receipt.json").exists():
            _write_json(output / "binding_failure.json", dict(status="failed_partial_evidence_preserved",
                exception_type=type(exc).__name__, formal_execution_admission=False))
        raise
