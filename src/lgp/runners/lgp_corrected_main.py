"""Selective corrected-LGP main-result producer; acceptance is independent."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import yaml

from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..modeling import checkpoint_path
from ..registry import Registry
from .experiment import build_plan
from .formal_parallel import (
    ASSIGNMENT_ALGORITHM,
    build_execution_assignment,
    canonical_partial_records,
    formal_expected_record_keys,
    normalize_execution_devices,
    run_formal_parallel,
    validate_available_cuda_devices,
)
from .retention import (
    PREDICTION_ARCHIVE_GZIP,
    RETENTION_MODE,
    SELECTION_ALGORITHM,
    build_retained_image_selection,
)

MAX_DIAGNOSTIC_IMAGES = 10


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _current_clean_inputs(registry: Registry, dataset_id: str) -> Dict[str, Any]:
    dataset = registry.dataset(dataset_id)
    index = CocoIndex(dataset, "val")
    expected = dataset.split("val").expected_images
    if len(index.images) != expected:
        raise RuntimeError("Formal clean reuse image population changed")
    image_root = dataset.root.resolve()
    image_rows = []
    image_ids = []
    for image in index.images:
        image_id = int(image["id"])
        image_path = index.image_path(image)
        if image_root not in image_path.parents or not image_path.is_file():
            raise RuntimeError("Formal clean reuse image path is missing or outside dataset")
        image_ids.append(image_id)
        image_rows.append([image_id, str(image["file_name"]), file_digest(image_path)])
    if len(set(image_ids)) != expected:
        raise RuntimeError("Formal clean reuse image IDs are not unique")
    checkpoints = {}
    for target_id in registry.target_ids():
        path = checkpoint_path(registry.model(target_id), dataset)
        if not path.is_file():
            raise RuntimeError("Formal clean reuse checkpoint is missing: " + target_id)
        checkpoints[target_id] = file_digest(path)
    return {
        "dataset": dataset_id,
        "split": "val",
        "expected_images": expected,
        "annotation_sha256": file_digest(index.annotation_path),
        "ordered_image_ids_sha256": _json_digest(image_ids),
        "ordered_image_files_sha256": _json_digest(image_rows),
        "target_checkpoint_sha256": checkpoints,
    }


def _accepted_clean_binding_basis(binding: Mapping[str, Any]) -> bool:
    return (
        binding.get("status") == "accepted_current_clean_binding"
        and binding.get("basis") == "new_identity_clean_reevaluation"
        and binding.get("historical_image_byte_identity_claim") is False
    )


def _verified_clean_binding(
    registry: Registry,
    protocol_id: str,
    spec: Mapping[str, Any],
    clean_reference_sha256: str,
    clean_binding: Optional[Path],
    current_inputs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    expected_hash = spec.get("clean_reuse_binding_sha256")
    if clean_binding is None or not clean_binding.is_file() or not expected_hash:
        raise RuntimeError("Independently accepted clean-input binding is required")
    binding_sha256 = file_digest(clean_binding)
    if binding_sha256 != expected_hash:
        raise RuntimeError("Accepted clean-input binding identity changed")
    binding = json.loads(clean_binding.read_text(encoding="utf-8"))
    if (
        not isinstance(binding, dict)
        or binding.get("schema_version") != 1
        or not _accepted_clean_binding_basis(binding)
        or binding.get("protocol") != protocol_id
        or binding.get("dataset") != spec["datasets"][0]
        or binding.get("split") != "val"
        or binding.get("clean_reference_acceptance_sha256") != clean_reference_sha256
    ):
        raise RuntimeError("Clean-input binding provenance is incomplete")
    current = dict(current_inputs) if current_inputs is not None else _current_clean_inputs(
        registry, spec["datasets"][0]
    )
    if binding.get("accepted_clean_inputs") != current:
        raise RuntimeError("Clean-input binding differs from current formal inputs")
    evidence_name = binding.get("source_clean_evidence_file")
    evidence_sha256 = binding.get("source_clean_evidence_sha256")
    if (
        not isinstance(evidence_name, str)
        or not evidence_name
        or Path(evidence_name).name != evidence_name
        or not isinstance(evidence_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256)
        or evidence_sha256 != spec.get("clean_current_audit_sha256")
    ):
        raise RuntimeError("Current clean audit identity is incomplete")
    evidence_path = clean_binding.resolve().parent / evidence_name
    if not evidence_path.is_file() or file_digest(evidence_path) != evidence_sha256:
        raise RuntimeError("Current clean audit file identity changed")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, dict):
        raise RuntimeError("Current clean audit is malformed")
    target_ids = registry.target_ids()
    target_evidence = evidence.get("target_evidence")
    if (
        evidence.get("schema_version") != 1
        or evidence.get("status") != "independent_saved_prediction_replay_pass"
        or evidence.get("dataset") != spec["datasets"][0]
        or evidence.get("current_inputs_sha256") != _json_digest(current)
        or evidence.get("target_count") != len(target_ids)
        or evidence.get("official_metric_replays") != len(target_ids) * len(COCO_BBOX_METRICS)
        or evidence.get("target_and_aggregate_csv_values_checked") is not True
        or evidence.get("historical_clean_reuse_accepted") is not False
        or evidence.get("corrected_lgp_execution_authorized") is not False
        or evidence.get("model_calls") != 0
        or not isinstance(target_evidence, dict)
        or set(target_evidence) != set(target_ids)
        or any(
            not isinstance(target_evidence[target], dict)
            or not isinstance(target_evidence[target].get("official_bbox_metrics"), dict)
            or set(target_evidence[target]["official_bbox_metrics"]) != set(COCO_BBOX_METRICS)
            for target in target_ids
        )
    ):
        raise RuntimeError("Current clean audit did not accept the complete target panel")
    for field in (
        "admission_sha256", "run_identity_sha256", "run_plan_sha256",
        "run_records_sha256", "run_summary_sha256",
    ):
        value = evidence.get(field)
        if (
            not isinstance(value, str)
            or not re.fullmatch(r"[0-9a-f]{64}", value)
            or binding.get(field) != value
        ):
            raise RuntimeError("Current clean audit run identity is incomplete: " + field)
    for field in (
        "historical_clean_reuse_accepted",
        "per_image_inference_trace_independently_available",
        "rendered_tex_reviewed",
    ):
        if type(evidence.get(field)) is not bool or binding.get(field) is not evidence[field]:
            raise RuntimeError("Current clean audit limitation changed: " + field)
    return {
        "binding_sha256": binding_sha256,
        "current_inputs": current,
        "source_clean_evidence_sha256": evidence_sha256,
    }


def _parameter_digest(registry: Registry) -> str:
    path = registry.root / "configs" / "attacks" / "reference.yaml"
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    parameters = parsed["attacks"]["lgp"]["parameters"]
    encoded = json.dumps(
        parameters, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _qualified_identity(registry: Registry, spec: Mapping[str, Any]) -> Dict[str, str]:
    inputs = {
        "implementation": (
            "src/lgp/attacks/lgp.py", "qualified_lgp_implementation_sha256"
        ),
        "factory": (
            "src/lgp/attacks/factory.py", "qualified_lgp_factory_sha256"
        ),
        "registry": (
            "configs/attacks/reference.yaml", "qualified_lgp_registry_sha256"
        ),
    }
    result: Dict[str, str] = {}
    for name, (relative, field) in inputs.items():
        actual = file_digest(registry.root / relative)
        if actual != str(spec.get(field, "")):
            raise RuntimeError("Qualified LGP {} identity changed".format(name))
        result[relative] = actual
    parameter_sha256 = _parameter_digest(registry)
    if parameter_sha256 != str(spec.get("qualified_lgp_parameters_sha256", "")):
        raise RuntimeError("Qualified LGP parameter identity changed")
    result["lgp_parameters"] = parameter_sha256
    return result


def build_lgp_corrected_main_plan(
    registry: Registry, protocol_id: str
) -> Dict[str, Any]:
    spec = registry.protocols.get(protocol_id)
    if not isinstance(spec, dict) or spec.get("lgp_corrected_stage") is not True:
        raise ValueError("Not a registered corrected-LGP main stage")
    if (
        spec.get("methods") != ["lgp"]
        or spec.get("targets") != "all"
        or spec.get("split") != "val"
        or spec.get("budget_profile") != "compute_matched"
        or spec.get("entrypoint") != "experiments/lgp_corrected_main.py"
        or list(spec.get("metrics", [])) != list(COCO_BBOX_METRICS)
        or spec.get("prediction_archive", {}).get("format") != PREDICTION_ARCHIVE_GZIP
    ):
        raise RuntimeError("Corrected-LGP stage contract drifted")
    dataset_ids = list(spec.get("datasets", []))
    if len(dataset_ids) != 1 or dataset_ids[0] not in {"coco", "voc"}:
        raise RuntimeError("Corrected-LGP stage must select one formal dataset")
    dataset_id = dataset_ids[0]
    expected_images = registry.dataset(dataset_id).split("val").expected_images
    if expected_images != int(spec.get("expected_images", -1)):
        raise RuntimeError("Corrected-LGP formal population drifted")
    sources = list(spec.get("sources", []))
    if sources != [source for source in registry.source_ids() if source in sources]:
        raise RuntimeError("Corrected-LGP sources are not in canonical order")
    if len(sources) != int(spec.get("expected_attack_groups", -1)):
        raise RuntimeError("Corrected-LGP source-group count drifted")
    retention = dict(spec.get("payload_retention", {}))
    mode = retention.get("mode")
    if mode == RETENTION_MODE:
        if (
            int(retention.get("retained_images", -1)) != 500
            or retention.get("selection_algorithm") != SELECTION_ALGORITHM
            or int(retention.get("prune_after_complete_targets", -1)) != 16
        ):
            raise RuntimeError("Corrected-LGP fixed-count lifecycle drifted")
    elif mode != "keep_all" or dataset_id != "coco" or len(sources) != 2:
        raise RuntimeError("Only the COCO Common-2 stage may keep full payloads")
    identity = _qualified_identity(registry, spec)
    plan = build_plan(registry, protocol_id)
    if (
        plan["datasets"] != dataset_ids
        or plan["sources"] != sources
        or plan["methods"] != ["lgp"]
        or plan["targets"] != registry.target_ids()
        or plan.get("execution_retired")
        or len(plan["jobs"]) != int(spec.get("expected_attack_evaluations", -1))
        or len(plan["jobs"]) != len(sources) * len(registry.target_ids())
        or any(job["status"] != "ready" for job in plan["jobs"])
    ):
        raise RuntimeError("Corrected-LGP main job matrix drifted")
    plan["expected_images"] = expected_images
    plan["seed"] = int(spec["seed"])
    plan["clean_evaluations"] = 0
    plan["clean_reference_acceptance_sha256"] = str(
        spec["clean_reference_acceptance_sha256"]
    )
    plan["qualified_lgp_identity"] = identity
    plan["artifact_lifecycle"] = {
        "mode": mode,
        "retained_images": 500 if mode == RETENTION_MODE else expected_images,
        "producer_mode": "keep_all",
        "independent_full_payload_audit_before_retention": mode == RETENTION_MODE,
        "deletion_authorized": False,
        "prediction_archive": PREDICTION_ARCHIVE_GZIP,
    }
    for job in plan["jobs"]:
        job["run_metadata"] = {
            "corrected_lgp_protocol": protocol_id,
            "qualified_implementation_sha256": identity["src/lgp/attacks/lgp.py"],
            "qualified_parameters_sha256": identity["lgp_parameters"],
        }
    return plan


def select_lgp_corrected_source_wave(
    registered: Mapping[str, Any], sources: Optional[Sequence[str]]
) -> Dict[str, Any]:
    stage_sources = list(registered["sources"])
    selected = stage_sources if sources is None else list(sources)
    if not selected or selected != [source for source in stage_sources if source in selected]:
        raise ValueError("Select a nonempty canonical subset of registered sources")
    plan = dict(registered)
    plan["registered_stage_plan_sha256"] = _json_digest(registered)
    plan["registered_stage_sources"] = stage_sources
    plan["sources"] = selected
    plan["jobs"] = [job for job in registered["jobs"] if job["source"] in selected]
    if len(plan["jobs"]) != len(selected) * len(registered["targets"]):
        raise RuntimeError("Source wave does not contain complete target panels")
    plan["execution_scope"] = (
        "full_stage" if selected == stage_sources else "complete_source_wave"
    )
    return plan


def _groups(plan: Mapping[str, Any]) -> List[Tuple[tuple, List[Dict[str, Any]]]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for item in plan["jobs"]:
        job = dict(item)
        key = (
            job["dataset"], job["split"], job["source"],
            job["attack"], job["study"], job["variant"],
        )
        groups.setdefault(key, []).append(job)
    return list(groups.items())


def _formal_preflight(
    registry: Registry,
    protocol_id: str,
    spec: Mapping[str, Any],
    clean_reference: Optional[Path],
    clean_binding: Optional[Path],
    retention_decision: Optional[Path],
    source_count: int,
) -> Dict[str, Any]:
    if spec.get("execution_authorized") is not True:
        raise RuntimeError("Corrected-LGP formal stage is not admitted")
    if clean_reference is None or not clean_reference.is_file():
        raise RuntimeError("Accepted clean reference file is required")
    clean_sha256 = file_digest(clean_reference)
    if clean_sha256 != spec["clean_reference_acceptance_sha256"]:
        raise RuntimeError("Accepted clean reference identity changed")
    binding = _verified_clean_binding(
        registry, protocol_id, spec, clean_sha256, clean_binding
    )
    if retention_decision is not None:
        raise RuntimeError("Producer does not perform or authorize retention cleanup")
    base = spec.get("storage_required_free_bytes")
    additional = spec.get("storage_required_free_bytes_per_additional_source")
    if (
        type(base) is not int or base < 5_000_000_000
        or type(additional) is not int or additional < 2_000_000_000
        or source_count < 1
    ):
        raise RuntimeError("Source-wave storage estimate is not registered")
    required = base + (source_count - 1) * additional
    storage_path = registry.root / "outputs" / "experiments" / protocol_id
    while not storage_path.exists():
        storage_path = storage_path.parent
    free = shutil.disk_usage(str(storage_path.resolve())).free
    if free < required:
        raise RuntimeError("Insufficient storage for corrected-LGP formal stage")
    return {
        "clean_reference_acceptance_sha256": clean_sha256,
        "clean_reuse_binding_sha256": binding["binding_sha256"],
        "clean_inputs": binding["current_inputs"],
        "producer_retention_mode": "keep_all",
        "source_count": source_count,
        "storage_required_free_bytes": required,
        "filesystem_free_bytes": free,
    }


def run_lgp_corrected_main(
    registry: Registry,
    protocol_id: str,
    *,
    execute: bool,
    max_images: Optional[int] = None,
    device: str = "cuda:0",
    devices: Optional[Sequence[str]] = None,
    output_dir: Optional[Path] = None,
    clean_reference: Optional[Path] = None,
    clean_binding: Optional[Path] = None,
    retention_decision: Optional[Path] = None,
    sources: Optional[Sequence[str]] = None,
) -> Path:
    plan = select_lgp_corrected_source_wave(
        build_lgp_corrected_main_plan(registry, protocol_id), sources
    )
    spec = registry.protocols[protocol_id]
    execution_devices = normalize_execution_devices(device, devices)
    if len(execution_devices) not in {1, 2} or any(
        re.fullmatch(r"cuda:(0|[1-9][0-9]*)", value) is None
        for value in execution_devices
    ):
        raise ValueError("Select one or two distinct explicit CUDA devices")
    if max_images is not None and not 1 <= max_images <= MAX_DIAGNOSTIC_IMAGES:
        raise ValueError("--max-images exceeds the bounded diagnostic limit")
    formal = bool(execute and max_images is None)
    if formal and output_dir is not None:
        raise ValueError("Formal execution requires a generated immutable directory")
    preflight = (
        _formal_preflight(
            registry, protocol_id, spec, clean_reference, clean_binding,
            retention_decision, len(plan["sources"]),
        )
        if formal else None
    )
    if execute:
        validate_available_cuda_devices(execution_devices, include_single=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    category = "experiments" if formal else "diagnostics" if execute else "plans"
    run_dir = (
        output_dir or registry.root / "outputs" / category / protocol_id / stamp
    ).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    group_items = _groups(plan)
    assignment = build_execution_assignment([], group_items, execution_devices)
    atomic_json(run_dir / "execution_assignments.json", assignment)
    plan["status"] = "running" if execute else "planned"
    plan["max_images"] = max_images
    plan["formal_execution"] = formal
    plan["execution_scheduler"] = {
        "mode": "single_device_serial" if len(execution_devices) == 1 else "independent_group_workers",
        "devices": list(execution_devices),
        "worker_count": len(execution_devices),
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "assignment_manifest_sha256": file_digest(run_dir / "execution_assignments.json"),
        "group_atomicity": "complete_source_method_group_per_worker",
        "global_record_writer": "coordinator_only",
    }
    if preflight is not None:
        plan["formal_preflight"] = preflight
    atomic_json(run_dir / "plan.json", plan)
    if not execute:
        atomic_json(run_dir / "summary.json", {
            "status": "planned", "protocol": protocol_id,
            "execution_scope": plan["execution_scope"],
            "registered_attack_groups": len(plan["registered_stage_sources"]),
            "attack_groups": len(group_items),
            "attack_evaluations": len(plan["jobs"]),
            "clean_evaluations": 0,
            "formal_result_eligible": False,
        })
        return run_dir

    records: List[Dict[str, Any]] = []
    group_summaries: List[Dict[str, Any]] = []
    records_path = run_dir / "records.json"
    state_path = run_dir / "execution_state.json"
    atomic_json(records_path, records)
    started_at = datetime.now(timezone.utc).isoformat()
    state: MutableMapping[str, Any] = {
        "schema_version": 1, "status": "running", "protocol": protocol_id,
        "execution_scope": plan["execution_scope"],
        "registered_attack_groups": len(plan["registered_stage_sources"]),
        "phase": "initializing", "current": None,
        "clean_evaluations_total": 0, "clean_evaluations_completed": 0,
        "attack_groups_total": len(group_items), "attack_groups_completed": 0,
        "attack_evaluations_total": len(plan["jobs"]),
        "attack_evaluations_completed": 0, "structural_skips": 0,
        "failed_records": 0, "workers": [],
        "execution_scheduler": plan["execution_scheduler"],
        "started_at": started_at, "updated_at": started_at,
    }
    atomic_json(state_path, state)
    retention = plan["artifact_lifecycle"]["producer_mode"]
    retained_selection: Mapping[str, Any] = {}
    try:
        if formal and plan["artifact_lifecycle"]["mode"] == RETENTION_MODE:
            retained_selection = build_retained_image_selection(
                registry, plan["datasets"][0], "val", 500,
                run_dir / "retained_image_set.json",
            )
            plan["retained_image_ids_sha256"] = retained_selection[
                "retained_image_ids_sha256"
            ]
            atomic_json(run_dir / "plan.json", plan)
        run_formal_parallel(
            registry=registry,
            run_dir=run_dir,
            devices=execution_devices,
            clean_jobs=[],
            group_items=group_items,
            settings={
                "max_images": max_images, "seed": plan["seed"],
                "download_weights": False, "keep_going": False,
                "save_visualizations": False,
                "visualization_score_threshold": 0.3,
                "visualization_max_images": 0,
                "visualization_max_detections": 0,
                "prediction_archive": PREDICTION_ARCHIVE_GZIP,
                "retention_mode": retention,
                "progress_interval_images": 25,
                "generic_experiment": True,
            },
            retained_selection=retained_selection,
            records=records,
            expected_keys=formal_expected_record_keys(plan, []),
            records_path=records_path,
            execution_state=state,
            state_path=state_path,
            group_summaries=group_summaries,
        )
        records = canonical_partial_records(
            records, formal_expected_record_keys(plan, [])
        )
        if (
            len(records) != len(plan["jobs"])
            or any(record.get("status") != "complete" for record in records)
            or len(group_summaries) != len(group_items)
            or any(
                item.get("status") != "complete_full_payload_retained"
                for item in group_summaries
            )
        ):
            raise RuntimeError("Corrected-LGP producer is incomplete")
        atomic_json(records_path, records)
        atomic_json(run_dir / "group_summaries.json", group_summaries)
        summary = {
            "schema_version": 1,
            "status": "producer_complete_independent_audit_pending",
            "protocol": protocol_id,
            "execution_scope": plan["execution_scope"],
            "registered_attack_groups": len(plan["registered_stage_sources"]),
            "formal_execution": formal,
            "formal_result_eligible": False,
            "clean_evaluations": 0,
            "attack_groups": len(group_items),
            "attack_evaluations": len(records),
            "payload_retention": plan["artifact_lifecycle"]["mode"],
            "producer_retention_mode": retention,
            "records_sha256": file_digest(records_path),
            "group_summaries_sha256": file_digest(run_dir / "group_summaries.json"),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(run_dir / "summary.json", summary)
        state.update(
            status="producer_complete_independent_audit_pending",
            phase="producer_complete", current=None, workers=[],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, state)
        return run_dir
    except BaseException as exc:
        state.update(
            status="failed", phase="failed", current=None,
            failure="{}: {}".format(type(exc).__name__, exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        atomic_json(state_path, state)
        atomic_json(run_dir / "summary.json", {
            "schema_version": 1, "status": "failed", "protocol": protocol_id,
            "execution_scope": plan["execution_scope"],
            "registered_attack_groups": len(plan["registered_stage_sources"]),
            "formal_result_eligible": False,
            "records_sha256": file_digest(records_path),
        })
        raise
