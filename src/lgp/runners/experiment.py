from __future__ import annotations

from ..reporting.scope import scope_from_records

import csv
import gc
import hashlib
import json
import platform
import subprocess
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..paths import project_root
from ..registry import Registry
from ..reporting import (
    write_ablation_reports,
    write_all_methods_reports,
    write_analysis_reports,
    write_artifact_manifest,
    write_experiment_plots,
    write_formal_all_methods_audits,
    write_paper_catalog,
    write_transfer_reports,
)
from .attack import run_attack
from .evaluate import run_evaluation
from .formal_parallel import (
    ASSIGNMENT_ALGORITHM,
    build_execution_assignment,
    canonical_partial_records,
    experiment_expected_record_keys,
    normalize_execution_devices,
    run_formal_parallel,
    validate_available_cuda_devices,
    validate_generated_group,
)
from .retention import (
    PREDICTION_ARCHIVE_GZIP,
    RETENTION_MODE,
    SELECTION_ALGORITHM,
    build_retained_image_selection,
    validate_and_prune_attack_group,
)


def _selection(raw: Any, fallback: Sequence[str]) -> List[str]:
    if raw == "all":
        return list(fallback)
    if isinstance(raw, list):
        return list(raw)
    return [str(raw)]


def build_plan(
    registry: Registry,
    protocol_id: str,
    dataset_override: Optional[Sequence[str]] = None,
    source_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    study_override: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if protocol_id not in registry.protocols:
        raise ValueError(
            "Unknown protocol '{}'. Choose one of: {}".format(
                protocol_id, ", ".join(registry.protocols)
            )
        )
    protocol = dict(registry.protocols[protocol_id])
    if protocol.get("source_matched_only") is True and target_override is not None:
        raise ValueError("Source-matched white-box protocols do not permit a target override")
    if protocol.get("generic_dispatch_forbidden") is True:
        raise ValueError(
            "Protocol '{}' cannot use generic experiment dispatch; use '{}'".format(
                protocol_id, protocol["preparation_entrypoint"]
            )
        )
    split = str(protocol.get("split", "val"))
    budget_profile = str(protocol["budget_profile"])
    budget = dict(registry.budget_profiles[budget_profile])
    datasets = list(
        dataset_override
        or _selection(protocol.get("datasets", "all"), registry.datasets)
    )
    allowed_datasets = protocol.get("allowed_datasets")
    if allowed_datasets is not None and any(value not in allowed_datasets for value in datasets):
        raise ValueError("Dataset is outside this protocol's maintained scope")
    requested_sources = list(
        source_override
        or _selection(protocol.get("sources", "all"), registry.source_ids())
    )
    registry.expand_ids(requested_sources, registry.source_ids())
    selected_sources = set(requested_sources)
    sources = [
        source for source in registry.source_ids() if source in selected_sources
    ]
    methods = list(
        method_override
        or _selection(protocol.get("methods", "all"), registry.attacks)
    )
    unknown_methods = sorted(set(methods) - set(registry.attacks))
    if unknown_methods:
        raise ValueError(
            "Unknown attack method(s): {}".format(", ".join(unknown_methods))
        )
    allowed_methods = protocol.get("allowed_methods")
    if allowed_methods is not None and any(value not in allowed_methods for value in methods):
        raise ValueError("Method is outside this protocol's maintained scope")
    if protocol.get("report_method_order") == "declared":
        if len(methods) != len(set(methods)):
            raise ValueError("Declared method panels do not permit duplicate methods")
        selected_methods = set(methods)
        declared_methods = _selection(protocol.get("methods", "all"), registry.attacks)
        declared_methods.extend(allowed_methods or list(registry.attacks))
        methods = [method for method in dict.fromkeys(declared_methods) if method in selected_methods]
    method_profiles = dict(protocol.get("method_budget_profiles", {}))
    raw_studies = protocol.get("ablation_studies", [])
    if study_override is not None:
        study_ids = list(study_override)
    elif raw_studies == "all":
        study_ids = [
            study_id for study_id, study in registry.ablation_studies.items()
            if not study.get("protocol") or str(study["protocol"]) == protocol_id
        ]
    else:
        study_ids = list(raw_studies)
    unknown_studies = sorted(set(study_ids) - set(registry.ablation_studies))
    if unknown_studies:
        raise ValueError(
            "Unknown ablation study/studies: {}".format(", ".join(unknown_studies))
        )
    variants_by_method: Dict[str, List[Dict[str, Any]]] = {}
    canonical_study = protocol.get("canonical_variant_study")
    if canonical_study is not None:
        if protocol_id != "current_vcsf_final_background" or study_ids:
            raise ValueError("Canonical background projection cannot be mixed with other studies")
        study = registry.ablation_studies[canonical_study]
        variants_by_method["vcsf"] = [dict(
            study=canonical_study, variant=variant,
            parameters=dict(study["base_parameters"], **study["variants"][variant]["parameters"]),
            seed=protocol["seed"], run_metadata=dict(
                public_protocol=protocol_id, background_variant=variant))
            for variant in study["row_order"]]
    for study_id in study_ids:
        study = registry.ablation_studies[study_id]
        if study.get("protocol") and str(study["protocol"]) != protocol_id:
            raise ValueError(
                "Ablation '{}' belongs to protocol '{}'".format(study_id, study["protocol"])
            )
        study_method = str(study.get("method", "svfta"))
        if study_method not in registry.attacks:
            raise ValueError(
                "Ablation '{}' references unknown method '{}'".format(
                    study_id, study_method
                )
            )
        for variant_id, variant in study["variants"].items():
            variants_by_method.setdefault(study_method, []).append(
                {
                    "study": study_id,
                    "variant": variant_id,
                    "parameters": dict(variant.get("parameters", {})),
                    "seed": variant.get("seed"),
                }
            )
    default_variant = {
        "study": "",
        "variant": "default",
        "parameters": {},
        "seed": None,
    }
    raw_targets = protocol.get("targets", "all")
    jobs: List[Dict[str, Any]] = []
    for dataset in datasets:
        registry.dataset(dataset).split(split)
        for source in sources:
            registry.model(source)
            if target_override:
                requested_targets = list(target_override)
                registry.expand_ids(requested_targets, registry.target_ids())
                selected_targets = set(requested_targets)
                targets = [
                    target
                    for target in registry.target_ids()
                    if target in selected_targets
                ]
            elif raw_targets == "source":
                targets = [source]
            elif raw_targets == "all_blackbox":
                targets = [
                    target for target in registry.target_ids() if target != source
                ]
            else:
                targets = _selection(raw_targets, registry.target_ids())
            for method in methods:
                resolved_profile = method_profiles.get(method, budget_profile)
                resolved_budget = dict(registry.budget_profiles[resolved_profile])
                method_variants = variants_by_method.get(method, [default_variant])
                compatibility = registry.compatibility_status(method, source)
                status = {
                    "native": "ready",
                    "external": "artifact_required",
                    "adaptable": "adapter_required",
                    "unsupported": "skipped",
                }[compatibility["status"]]
                if method == "vcsf" and registry.attack(method).metadata["execution_authorized"] is False:
                    status = "qualification_pending"
                for variant in method_variants:
                    for target in targets:
                        registry.model(target)
                        jobs.append(
                            {
                                "dataset": dataset,
                                "split": split,
                                "source": source,
                                "attack": method,
                                "study": variant["study"],
                                "variant": variant["variant"],
                                **({"run_metadata": variant["run_metadata"]}
                                   if "run_metadata" in variant else {}),
                                "parameters": variant["parameters"],
                                "seed": variant["seed"],
                                "target": target,
                                "budget_profile": resolved_profile,
                                "budget": resolved_budget,
                                "status": status,
                                "compatibility": compatibility["status"],
                                "reason": (
                                    "" if status == "ready" else compatibility["reason"]
                                ),
                            }
                        )
    return {
        "schema_version": 1,
        "protocol": protocol_id,
        "description": protocol.get("description", ""),
        "split": split,
        "budget_profile": budget_profile,
        "budget": budget,
        "method_budget_profiles": method_profiles,
        "datasets": datasets,
        "sources": sources,
        "methods": methods,
        "targets": [
            target
            for target in registry.target_ids()
            if any(job["target"] == target for job in jobs)
        ],
        "image_selection": dict(protocol.get("image_selection", {})),
        "jobs": jobs,
        "execution_retired": bool(protocol.get("execution_retired")) or any(
            bool(registry.ablation_studies[study_id].get("execution_retired"))
            for study_id in {job["study"] for job in jobs if job.get("study")}
        ),
    }


def _resolve_protocol_image_ids(
    registry: Registry,
    protocol: Mapping[str, Any],
    plan: Dict[str, Any],
    execute: bool,
    max_images: Optional[int],
) -> Dict[str, List[int]]:
    selection = dict(protocol.get("image_selection", {}))
    if not selection:
        return {}
    if str(selection.get("strategy")) != "ordered_slice":
        raise ValueError("Protocol image_selection must use ordered_slice")
    if plan["split"] != "dev" or not bool(protocol.get("development_only")):
        raise ValueError(
            "Protocol image_selection is restricted to development-only dev runs"
        )
    if len(plan["datasets"]) != 1:
        raise ValueError("Protocol image_selection requires exactly one dataset")
    start = int(selection.get("start", -1))
    stop = int(selection.get("stop", -1))
    expected_total = int(selection.get("expected_total", -1))
    if start < 0 or stop <= start or expected_total < stop:
        raise ValueError(
            "Protocol ordered_slice requires 0 <= start < stop <= expected_total"
        )
    plan["image_selection"] = {
        **selection,
        "resolved": False,
    }
    if not execute:
        return {}

    dataset_id = str(plan["datasets"][0])
    index = CocoIndex(registry.dataset(dataset_id), str(plan["split"]))
    if len(index.images) != expected_total:
        raise RuntimeError(
            "Protocol image selection expected {} source images, found {}".format(
                expected_total, len(index.images)
            )
        )
    selected_ids = [
        int(image["id"]) for image in index.images[start:stop]
    ]
    effective_ids = (
        selected_ids[:max_images] if max_images is not None else selected_ids
    )

    def digest(values: Sequence[int]) -> str:
        encoded = json.dumps(
            list(values), separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    plan["image_selection"] = {
        **selection,
        "resolved": True,
        "source_annotation_sha256": file_digest(index.annotation_path),
        "selected_count": len(selected_ids),
        "selected_image_ids_sha256": digest(selected_ids),
        "effective_count": len(effective_ids),
        "effective_image_ids_sha256": digest(effective_ids),
    }
    return {dataset_id: effective_ids}


def _write_plan_csv(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "split",
        "source",
        "attack",
        "study",
        "variant",
        "parameters",
        "seed",
        "target",
        "budget_profile",
        "budget",
        "status",
        "compatibility",
        "reason",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            row = {key: job.get(key, "") for key in fields}
            row["parameters"] = json.dumps(
                job.get("parameters", {}), sort_keys=True
            )
            row["budget"] = json.dumps(job.get("budget", {}), sort_keys=True)
            writer.writerow(row)


def _read_metrics(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("Metrics payload is not an object: {}".format(path))
    return payload


def _formal_record_order(
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    registry: Registry,
) -> List[Dict[str, Any]]:
    """Return the canonical 16-clean + 1,056-cell all-method record order."""
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("source", "")),
            str(record.get("attack", "")),
            str(record.get("target", "")),
        )
        if key in by_key:
            raise RuntimeError(
                "Formal records contain a duplicate source-method-target key: "
                "{}".format(key)
            )
        by_key[key] = dict(record)
    expected_keys = [
        ("clean", "clean", target_id)
        for target_id in registry.target_ids()
    ] + [
        (str(job["source"]), str(job["attack"]), str(job["target"]))
        for job in plan.get("jobs", [])
    ]
    if len(expected_keys) != 1072 or set(by_key) != set(expected_keys):
        raise RuntimeError(
            "Formal records are not the exact 16-clean + 1,056-cell matrix"
        )
    return [by_key[key] for key in expected_keys]


def _confirmatory_component_record_order(
    records: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    registry: Registry,
) -> List[Dict[str, Any]]:
    """Validate and order the exact 16-clean + 8x16 confirmation matrix."""
    by_key: Dict[tuple, Dict[str, Any]] = {}
    for record in records:
        if str(record.get("attack", "")) == "clean":
            key = ("clean", "clean", "", "", str(record.get("target", "")))
        else:
            key = (
                str(record.get("source", "")),
                str(record.get("attack", "")),
                str(record.get("study", "")),
                str(record.get("variant", "")),
                str(record.get("target", "")),
            )
        if key in by_key:
            raise RuntimeError(
                "Confirmatory records contain a duplicate key: {}".format(key)
            )
        by_key[key] = dict(record)

    expected_keys = [
        ("clean", "clean", "", "", target_id)
        for target_id in registry.target_ids()
    ] + [
        (
            str(job["source"]),
            str(job["attack"]),
            str(job["study"]),
            str(job["variant"]),
            str(job["target"]),
        )
        for job in plan.get("jobs", [])
    ]
    if len(expected_keys) != 16 + 8 * 16 or set(by_key) != set(expected_keys):
        raise RuntimeError(
            "SVFTA confirmation is not the exact 16-clean + 8x16 matrix"
        )

    ordered = [by_key[key] for key in expected_keys]
    for record in ordered:
        if (
            str(record.get("status")) != "complete"
            or int(record.get("images", -1)) != 5000
            or record.get("evaluated_image_ids_match_expected") is not True
            or bool(record.get("failures"))
        ):
            raise RuntimeError(
                "SVFTA confirmation contains an incomplete 5,000-image record: "
                "{}/{}/{}".format(
                    record.get("variant", "clean"),
                    record.get("source"),
                    record.get("target"),
                )
            )
        metrics = record.get("metrics") or {}
        if any(metric not in metrics for metric in COCO_BBOX_METRICS):
            raise RuntimeError(
                "SVFTA confirmation record is missing a standard COCO metric"
            )
        artifact = record.get("predictions_artifact") or {}
        if artifact.get("status") != "verified_lossless_archive":
            raise RuntimeError(
                "SVFTA confirmation requires a verified lossless prediction archive"
            )
        if record.get("attack") != "clean" and int(
            record.get("gradient_evaluations_per_image", -1)
        ) != 20:
            raise RuntimeError(
                "SVFTA confirmation requires exactly 20 gradient evaluations"
            )
    return ordered


def _source_manifest(root: Optional[Path] = None) -> Dict[str, Any]:
    """Hash executable source/config state without recording machine paths."""
    root = (root or project_root()).resolve()
    selected: List[Path] = []
    for directory, pattern in (
        ("src", "*.py"),
        ("configs", "*.yaml"),
        ("experiments", "*.py"),
    ):
        selected.extend((root / directory).rglob(pattern))
    for name in ("environment.yml", "requirements/locked-cu118.txt"):
        path = root / name
        if path.is_file():
            selected.append(path)
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": file_digest(path),
        }
        for path in sorted(
            set(selected), key=lambda item: item.relative_to(root).as_posix()
        )
    ]
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "algorithm": "sha256",
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def _formal_command(
    execution_devices: Sequence[str],
    entrypoint: str = "experiments/coco_all_methods.py",
) -> str:
    if (
        not entrypoint.startswith("experiments/")
        or not entrypoint.endswith(".py")
        or any(character.isspace() for character in entrypoint)
    ):
        raise ValueError("Formal protocol entrypoint is invalid")
    base = "conda run -n oda python {}".format(entrypoint)
    if len(execution_devices) > 1:
        base += " --devices {}".format(",".join(execution_devices))
    elif list(execution_devices) != ["cuda:0"]:
        base += " --device {}".format(execution_devices[0])
    return base + " --no-visualize-predictions"


def _run_provenance(
    protocol_id: str,
    protocol: Mapping[str, Any],
    execution_scheduler: Mapping[str, Any],
) -> Dict[str, Any]:
    root = project_root()
    commit = None
    worktree_clean = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout
        worktree_clean = not bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    versions: Dict[str, Optional[str]] = {}
    for package in (
        "torch",
        "torchvision",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmyolo",
        "numpy",
        "pycocotools",
        "Pillow",
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "schema_version": 1,
        "protocol": protocol_id,
        "git_commit": commit,
        "git_worktree_clean_at_start": worktree_clean,
        "source_manifest": _source_manifest(),
        "python": platform.python_version(),
        "packages": versions,
        "canonical_command": (
            _formal_command(
                execution_scheduler["devices"],
                str(protocol.get("entrypoint", "")),
            )
            if bool(
                protocol.get("formal_all_methods")
                or protocol.get("formal_component_confirmatory")
            )
            else None
        ),
        "execution_scheduler": dict(execution_scheduler),
        "privacy": "No hostname, hardware identifier, credential, or absolute project path is recorded.",
    }


def run_experiment(
    registry: Registry,
    protocol_id: str,
    execute: bool = False,
    output_dir: Optional[Path] = None,
    dataset_override: Optional[Sequence[str]] = None,
    source_override: Optional[Sequence[str]] = None,
    target_override: Optional[Sequence[str]] = None,
    method_override: Optional[Sequence[str]] = None,
    study_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda:0",
    devices: Optional[Sequence[str]] = None,
    download_weights: bool = False,
    keep_going: bool = True,
    save_visualizations: bool = False,
    visualization_score_threshold: float = 0.3,
    visualization_max_images: int = 20,
    visualization_max_detections: int = 100,
    payload_retention: Optional[str] = None,
    retained_image_count: Optional[int] = None,
    prediction_archive: Optional[str] = None,
    generic_report_projections: bool = True,
) -> Path:
    execution_devices = normalize_execution_devices(device, devices)
    protocol = dict(registry.protocols.get(protocol_id, {}))
    is_formal_all_methods = bool(protocol.get("formal_all_methods"))
    is_formal_component_confirmatory = bool(
        protocol.get("formal_component_confirmatory")
    )
    is_formal_component_execution = bool(
        execute and is_formal_component_confirmatory and max_images is None
    )
    if len(execution_devices) not in {1, 2}:
        raise ValueError(
            "Experiment protocols support one serial device or two "
            "independent group-worker devices"
        )
    execution_scheduler = {
        "mode": (
            "single_device_serial"
            if len(execution_devices) == 1
            else "independent_group_workers"
        ),
        "devices": list(execution_devices),
        "worker_count": len(execution_devices),
        "assignment_algorithm": ASSIGNMENT_ALGORITHM,
        "clean_barrier_before_attack_groups": True,
        "group_atomicity": "complete_source_method_group_per_worker",
        "global_record_writer": "coordinator_only",
    }
    plan = build_plan(
        registry,
        protocol_id,
        dataset_override=dataset_override,
        source_override=source_override,
        target_override=target_override,
        method_override=method_override,
        study_override=study_override,
    )
    if execute and "vcsf" in plan["methods"]:
        from ..attacks.vcsf_public import guard_public_execution

        guard_public_execution(registry)
    if execute and plan.get("execution_retired"):
        raise ValueError(
            "Retired protocol or ablation study cannot execute; "
            "use --plan-only to inspect its preserved historical declaration."
        )
    retention_config = dict(protocol.get("payload_retention", {}))
    prediction_config = dict(protocol.get("prediction_archive", {}))
    resolved_retention = str(
        payload_retention or retention_config.get("mode", "keep_all")
    )
    resolved_retained_count = int(
        retained_image_count
        if retained_image_count is not None
        else retention_config.get("retained_images", 0)
    )
    if prediction_archive == "none":
        resolved_prediction_archive = None
    else:
        resolved_prediction_archive = prediction_archive
    if prediction_archive is None:
        configured_archive = str(prediction_config.get("format", "none"))
        resolved_prediction_archive = (
            None if configured_archive == "none" else configured_archive
        )
    if resolved_retention not in {"keep_all", RETENTION_MODE}:
        raise ValueError(
            "Unknown payload-retention mode: {}".format(resolved_retention)
        )
    if resolved_prediction_archive not in {None, PREDICTION_ARCHIVE_GZIP}:
        raise ValueError(
            "Unknown prediction archive format: {}".format(
                resolved_prediction_archive
            )
        )
    plan["prediction_visualizations"] = {
        "enabled": bool(save_visualizations),
        "score_threshold": float(visualization_score_threshold),
        "max_images_per_evaluation": int(visualization_max_images),
        "max_detections_per_image": int(visualization_max_detections),
    }
    plan["execution_seed"] = int(seed)
    plan["execution_scheduler"] = dict(execution_scheduler)
    plan["artifact_lifecycle"] = {
        "payload_retention": resolved_retention,
        "retained_images_per_attack_set": resolved_retained_count,
        "prune_only_after_complete_target_panel": True,
        "prediction_archive": resolved_prediction_archive or "none",
        "raw_predictions_removed_after_verified_archive": bool(
            resolved_prediction_archive
        ),
    }
    formal_dataset: Optional[str] = None
    if (
        execute
        and resolved_retention == RETENTION_MODE
        and not is_formal_all_methods
    ):
        raise ValueError(
            "The fixed-count formal lifecycle is registered only for "
            "formal all-method protocols"
        )
    if execute and is_formal_all_methods:
        registered_datasets = list(
            _selection(protocol.get("datasets", []), registry.datasets)
        )
        if len(registered_datasets) != 1:
            raise RuntimeError(
                "Formal all-method protocol must register exactly one dataset"
            )
        formal_dataset = registered_datasets[0]
        if output_dir is not None:
            raise ValueError(
                "Formal all-method execution requires its generated immutable "
                "UTC timestamp directory"
            )
        if resolved_retention != RETENTION_MODE:
            raise ValueError(
                "Formal all-method execution requires its registered "
                "post-16/16 retention lifecycle"
            )
        if (
            plan["datasets"] != registered_datasets
            or plan["split"] != str(protocol.get("split", "val"))
        ):
            raise ValueError(
                "Formal all-method retention forbids dataset or split overrides"
            )
        if max_images is not None:
            raise ValueError(
                "Formal all-method retention forbids --max-images; use a "
                "separate diagnostic protocol for a smoke run"
            )
        if int(seed) != int(protocol.get("seed", -1)):
            raise ValueError(
                "Formal all-method execution requires the registered seed"
            )
        if download_weights:
            raise ValueError(
                "Formal all-method execution forbids checkpoint downloads"
            )
        if plan["sources"] != registry.source_ids():
            raise ValueError(
                "Formal all-method retention requires all six canonical sources"
            )
        if plan["targets"] != registry.target_ids():
            raise ValueError(
                "Formal all-method retention requires all sixteen canonical targets"
            )
        if plan["methods"] != list(protocol.get("methods", [])):
            raise ValueError(
                "Formal all-method retention forbids a method override"
            )
        if (
            len(plan["methods"]) != 11
            or list(protocol.get("metrics", [])) != list(COCO_BBOX_METRICS)
            or retention_config.get("selection_algorithm")
            != SELECTION_ALGORITHM
            or int(retention_config.get("prune_after_complete_targets", -1))
            != 16
            or prediction_config.get("remove_uncompressed_after_verification")
            is not True
            or not str(protocol.get("entrypoint", "")).startswith(
                "experiments/"
            )
        ):
            raise RuntimeError(
                "Formal all-method protocol registry invariants are incomplete"
            )
        status_counts = {
            status: sum(job["status"] == status for job in plan["jobs"])
            for status in ("ready", "skipped", "artifact_required", "adapter_required")
        }
        ready_groups = {
            (job["source"], job["attack"])
            for job in plan["jobs"]
            if job["status"] == "ready"
        }
        if len(plan["jobs"]) != 1056 or status_counts != {
            "ready": 848,
            "skipped": 208,
            "artifact_required": 0,
            "adapter_required": 0,
        } or len(ready_groups) != 53:
            raise RuntimeError(
                "Formal all-method matrix must resolve to 1,056 cells, "
                "848 evaluations, 208 structural skips and 53 payloads"
            )
        if resolved_retained_count != 500:
            raise ValueError(
                "Formal all-method retention requires exactly 500 retained images"
            )
        if resolved_prediction_archive != PREDICTION_ARCHIVE_GZIP:
            raise ValueError(
                "Formal all-method retention requires lossless gzip predictions"
            )
        if keep_going:
            raise ValueError(
                "Formal all-method retention requires fail-fast execution"
            )
        if save_visualizations:
            raise ValueError(
                "Formal all-method execution disables prediction overlays"
            )

    if is_formal_component_execution:
        expected_variants = list(
            registry.ablation_studies["components"]["variants"]
        )
        if output_dir is not None:
            raise ValueError(
                "Formal component confirmation requires its generated immutable "
                "UTC timestamp directory"
            )
        if (
            plan["datasets"] != ["coco"]
            or plan["split"] != "val"
            or plan["budget_profile"] != "compute_matched"
            or plan["sources"] != ["faster_rcnn_r50"]
            or plan["targets"] != registry.target_ids()
            or plan["methods"] != ["svfta"]
            or len(plan["jobs"]) != 8 * 16
            or {str(job["study"]) for job in plan["jobs"]} != {"components"}
            or list(dict.fromkeys(str(job["variant"]) for job in plan["jobs"]))
            != expected_variants
        ):
            raise RuntimeError(
                "Formal SVFTA component confirmation must be one source, eight "
                "frozen variants and all sixteen canonical targets"
            )
        if (
            list(protocol.get("metrics", [])) != list(COCO_BBOX_METRICS)
            or str(protocol.get("entrypoint", ""))
            != "experiments/svfta_component_confirmatory.py"
        ):
            raise RuntimeError(
                "Formal SVFTA component protocol registry invariants are incomplete"
            )
        if int(seed) != int(protocol.get("seed", -1)):
            raise ValueError("Formal component confirmation requires seed 42")
        if download_weights:
            raise ValueError(
                "Formal component confirmation forbids checkpoint downloads; "
                "download and verify them before execution"
            )
        if keep_going:
            raise ValueError("Formal component confirmation requires fail-fast execution")
        if save_visualizations:
            raise ValueError("Formal component confirmation disables prediction overlays")
        if resolved_prediction_archive != PREDICTION_ARCHIVE_GZIP:
            raise ValueError(
                "Formal component confirmation requires lossless gzip predictions"
            )

    if execute:
        validate_available_cuda_devices(execution_devices, include_single=True)
    protocol_image_ids = _resolve_protocol_image_ids(
        registry,
        protocol,
        plan,
        execute,
        max_images,
    )

    provenance = _run_provenance(
        protocol_id, protocol, execution_scheduler
    )
    if execute and is_formal_all_methods:
        if provenance.get("git_worktree_clean_at_start") is not True:
            raise RuntimeError(
                "Formal all-method execution requires a clean Git worktree"
            )
        commit = str(provenance.get("git_commit") or "").lower()
        if len(commit) != 40 or any(
            character not in "0123456789abcdef" for character in commit
        ):
            raise RuntimeError(
                "Formal all-method execution requires an exact Git commit"
            )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_dir
        or project_root() / "outputs" / "experiments" / protocol_id / stamp
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite non-empty experiment: {}".format(run_dir)
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "plan.json", plan)
    _write_plan_csv(run_dir / "plan.csv", plan["jobs"])
    atomic_json(run_dir / "provenance.json", provenance)

    records: List[Dict[str, Any]] = [
        {
            "dataset": job["dataset"],
            "split": job["split"],
            "source": job["source"],
            "attack": job["attack"],
            "study": job["study"],
            "variant": job["variant"],
            "parameters": job["parameters"],
            "target": job["target"],
            "budget_profile": job["budget_profile"],
            "budget": job["budget"],
            "status": (
                job["status"]
                if execute
                else ("planned" if job["status"] == "ready" else job["status"])
            ),
            "reason": job["reason"],
            "metrics": {},
        }
        for job in plan["jobs"]
        if not execute or job["status"] != "ready"
    ]
    records_path = run_dir / "records.json"
    atomic_json(records_path, records)

    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for job in plan["jobs"]:
        if job["status"] == "ready":
            groups.setdefault(
                (
                    job["dataset"],
                    job["split"],
                    job["source"],
                    job["attack"],
                    job["study"],
                    job["variant"],
                ),
                [],
            ).append(job)
    clean_jobs = sorted(
        {
            (job["dataset"], job["split"], job["target"])
            for job in plan["jobs"]
        },
        key=lambda item: (
            item[0],
            item[1],
            registry.model_position(item[2]),
        ),
    )
    expected_record_keys = experiment_expected_record_keys(plan, clean_jobs)
    records[:] = canonical_partial_records(records, expected_record_keys)
    assignment = build_execution_assignment(
        clean_jobs, list(groups.items()), execution_devices
    )
    assignment_path = run_dir / "execution_assignments.json"
    atomic_json(assignment_path, assignment)
    plan["execution_scheduler"]["assignment_manifest"] = assignment_path.name
    plan["execution_scheduler"]["assignment_manifest_sha256"] = file_digest(assignment_path)
    atomic_json(run_dir / "plan.json", plan)
    started_at = datetime.now(timezone.utc).isoformat()
    execution_state: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running" if execute else "planned",
        "protocol": protocol_id,
        "phase": "initializing" if execute else "plan_only",
        "current": None,
        "clean_evaluations_total": len(clean_jobs),
        "clean_evaluations_completed": 0,
        "attack_groups_total": len(groups),
        "attack_groups_completed": 0,
        "attack_evaluations_total": sum(
            len(group_jobs) for group_jobs in groups.values()
        ),
        "attack_evaluations_completed": 0,
        "structural_skips": sum(
            job["status"] == "skipped" for job in plan["jobs"]
        ),
        "failed_records": 0,
        "execution_scheduler": dict(plan["execution_scheduler"]),
        "workers": [],
        "started_at": started_at,
        "updated_at": started_at,
    }
    state_path = run_dir / "execution_state.json"
    atomic_json(state_path, execution_state)
    retained_selection: Optional[Dict[str, Any]] = None
    group_summaries: List[Dict[str, Any]] = []

    def update_state(**changes: Any) -> None:
        execution_state.update(changes)
        execution_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, execution_state)

    try:
        if execute and resolved_retention == RETENTION_MODE:
            if formal_dataset is None:
                raise RuntimeError(
                    "Formal retention dataset was not resolved"
                )
            retained_selection = build_retained_image_selection(
                registry,
                formal_dataset,
                plan["split"],
                resolved_retained_count,
                run_dir / "retained_image_set.json",
            )
            plan["artifact_lifecycle"]["retained_image_ids_sha256"] = (
                retained_selection["retained_image_ids_sha256"]
            )
            atomic_json(run_dir / "plan.json", plan)

        if execute and len(execution_devices) == 1:
            for dataset, split, target in clean_jobs:
                update_state(
                    phase="clean_evaluation",
                    current={"dataset": dataset, "target": target},
                )
                clean_dir = run_dir / "evaluations" / dataset / "clean" / target
                try:
                    completed = run_evaluation(
                        registry,
                        dataset,
                        target,
                        split=split,
                        output_dir=clean_dir,
                        max_images=max_images,
                        device=execution_devices[0],
                        download_weights=download_weights,
                        keep_going=keep_going,
                        save_visualizations=save_visualizations,
                        visualization_score_threshold=(
                            visualization_score_threshold
                        ),
                        visualization_max_images=visualization_max_images,
                        visualization_max_detections=(
                            visualization_max_detections
                        ),
                        prediction_archive=resolved_prediction_archive,
                        image_ids=protocol_image_ids.get(dataset),
                    )
                    records.append(_read_metrics(completed / "metrics.json"))
                except Exception as exc:
                    records.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "source": "clean",
                            "attack": "clean",
                            "target": target,
                            "status": "failed",
                            "reason": "clean evaluation failed: {}: {}".format(
                                type(exc).__name__, exc
                            ),
                            "metrics": {},
                        }
                    )
                    atomic_json(records_path, records)
                    if not keep_going:
                        raise
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                atomic_json(records_path, records)
                update_state(
                    clean_evaluations_completed=(
                        execution_state["clean_evaluations_completed"] + 1
                    ),
                    failed_records=sum(
                        str(record.get("status")) == "failed"
                        for record in records
                    ),
                )

            for group_index, (group_key, group_jobs) in enumerate(
                groups.items(), start=1
            ):
                dataset, split, source, attack, study, variant = group_key
                suffix = Path(study) / variant if study else Path(variant)
                attack_dir = (
                    run_dir
                    / "attacks"
                    / dataset
                    / source
                    / attack
                    / suffix
                )
                group_status_path = (
                    run_dir
                    / "group_status"
                    / dataset
                    / source
                    / attack
                    / suffix.with_suffix(".json")
                )
                group_status: Dict[str, Any] = {
                    "schema_version": 1,
                    "status": "generating",
                    "index": group_index,
                    "total": len(groups),
                    "dataset": dataset,
                    "split": split,
                    "source": source,
                    "attack": attack,
                    "study": study,
                    "variant": variant,
                    "targets_total": len(group_jobs),
                    "targets_completed": 0,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                }
                atomic_json(group_status_path, group_status)
                update_state(
                    phase="attack_generation",
                    current={
                        "group_index": group_index,
                        "dataset": dataset,
                        "source": source,
                        "attack": attack,
                        "variant": variant,
                    },
                )
                job_seed = group_jobs[0].get("seed")
                try:
                    generated = run_attack(
                        registry,
                        dataset,
                        source,
                        attack,
                        split=split,
                        output_dir=attack_dir,
                        max_images=max_images,
                        seed=int(seed if job_seed is None else job_seed),
                        device=execution_devices[0],
                        download_weights=download_weights,
                        keep_going=keep_going,
                        strict=True,
                        parameter_overrides=group_jobs[0].get("parameters", {}),
                        budget_profile=group_jobs[0]["budget_profile"],
                        image_ids=protocol_image_ids.get(dataset),
                        **({"run_metadata": group_jobs[0]["run_metadata"]}
                           if "run_metadata" in group_jobs[0] else {}),
                    )
                    validate_generated_group(
                        generated, dataset, split, source, attack,
                        protocol_image_ids.get(dataset), max_images,
                    )
                except Exception as exc:
                    group_status.update(
                        status="failed_generation",
                        reason="{}: {}".format(type(exc).__name__, exc),
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    atomic_json(group_status_path, group_status)
                    for job in group_jobs:
                        records.append(
                            {
                                **job,
                                "status": "failed",
                                "reason": (
                                    "attack generation failed: {}: {}".format(
                                        type(exc).__name__, exc
                                    )
                                ),
                                "metrics": {},
                            }
                        )
                    atomic_json(records_path, records)
                    if not keep_going:
                        raise
                    continue
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                evaluation_dirs: Dict[str, Path] = {}
                group_failed = False
                for target_index, job in enumerate(group_jobs, start=1):
                    evaluation_dir = (
                        run_dir
                        / "evaluations"
                        / dataset
                        / source
                        / attack
                        / suffix
                        / job["target"]
                    )
                    evaluation_dirs[job["target"]] = evaluation_dir
                    group_status.update(
                        status="evaluating_targets",
                        current_target=job["target"],
                        current_target_index=target_index,
                    )
                    atomic_json(group_status_path, group_status)
                    update_state(
                        phase="attack_target_evaluation",
                        current={
                            "group_index": group_index,
                            "dataset": dataset,
                            "source": source,
                            "attack": attack,
                            "variant": variant,
                            "target": job["target"],
                            "target_index": target_index,
                        },
                    )
                    try:
                        completed = run_evaluation(
                            registry,
                            dataset,
                            job["target"],
                            split=job["split"],
                            adversarial_run=generated,
                            output_dir=evaluation_dir,
                            max_images=max_images,
                            device=execution_devices[0],
                            download_weights=download_weights,
                            keep_going=keep_going,
                            save_visualizations=save_visualizations,
                            visualization_score_threshold=(
                                visualization_score_threshold
                            ),
                            visualization_max_images=visualization_max_images,
                            visualization_max_detections=(
                                visualization_max_detections
                            ),
                            prediction_archive=resolved_prediction_archive,
                            image_ids=protocol_image_ids.get(dataset),
                        )
                        result = _read_metrics(completed / "metrics.json")
                        result["study"] = study
                        result["variant"] = variant
                        result["parameters"] = group_jobs[0].get(
                            "parameters", {}
                        )
                        result["budget_profile"] = job["budget_profile"]
                        result["budget"] = job["budget"]
                        records.append(result)
                        group_failed = group_failed or result.get("status") != "complete"
                    except Exception as exc:
                        group_failed = True
                        records.append(
                            {
                                **job,
                                "status": "failed",
                                "reason": "evaluation failed: {}: {}".format(
                                    type(exc).__name__, exc
                                ),
                                "metrics": {},
                            }
                        )
                        atomic_json(records_path, records)
                        if not keep_going:
                            raise
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    atomic_json(records_path, records)
                    group_status["targets_completed"] = target_index
                    atomic_json(group_status_path, group_status)
                    update_state(
                        attack_evaluations_completed=(
                            execution_state["attack_evaluations_completed"] + 1
                        ),
                        failed_records=sum(
                            str(record.get("status")) == "failed"
                            for record in records
                        ),
                    )

                if group_failed:
                    group_status.update(
                        status="failed_target_panel",
                        completed_at=datetime.now(timezone.utc).isoformat(),
                    )
                    atomic_json(group_status_path, group_status)
                    continue

                if resolved_retention == RETENTION_MODE:
                    if retained_selection is None:
                        raise RuntimeError(
                            "Formal retention selection was not frozen"
                        )
                    group_status["status"] = "validating_before_prune"
                    atomic_json(group_status_path, group_status)
                    update_state(phase="group_validation_and_prune")
                    retention_summary = validate_and_prune_attack_group(
                        registry,
                        dataset,
                        split,
                        source,
                        attack,
                        generated,
                        evaluation_dirs,
                        [job["target"] for job in group_jobs],
                        retained_selection,
                    )
                    retention_path = generated / "retention_summary.json"
                    group_status["retention_summary"] = str(
                        retention_path.relative_to(run_dir)
                    )
                    group_status["retention_summary_sha256"] = file_digest(
                        retention_path
                    )
                    group_status["retained_images"] = int(
                        retention_summary["retained_images"]
                    )
                    group_status["pruned_images"] = int(
                        retention_summary["images_to_prune"]
                    )
                    group_status["status"] = (
                        "complete_validated_and_pruned"
                    )
                else:
                    group_status["status"] = "complete_full_payload_retained"
                group_status["completed_at"] = datetime.now(
                    timezone.utc
                ).isoformat()
                atomic_json(group_status_path, group_status)
                group_summaries.append(dict(group_status))
                update_state(
                    attack_groups_completed=(
                        execution_state["attack_groups_completed"] + 1
                    )
                )

        if execute and len(execution_devices) > 1:
            if resolved_retention == RETENTION_MODE and retained_selection is None:
                raise RuntimeError(
                    "Parallel retention requires the frozen retained-image selection"
                )
            run_formal_parallel(
                registry=registry,
                run_dir=run_dir,
                devices=execution_devices,
                clean_jobs=clean_jobs,
                group_items=list(groups.items()),
                settings={
                    "max_images": max_images,
                    "seed": int(seed),
                    "download_weights": bool(download_weights),
                    "save_visualizations": bool(save_visualizations),
                    "visualization_score_threshold": float(
                        visualization_score_threshold
                    ),
                    "visualization_max_images": int(visualization_max_images),
                    "visualization_max_detections": int(
                        visualization_max_detections
                    ),
                    "prediction_archive": resolved_prediction_archive,
                    "retention_mode": resolved_retention,
                    "keep_going": bool(keep_going),
                    "image_ids_by_dataset": protocol_image_ids,
                    "generic_experiment": not is_formal_all_methods,
                },
                retained_selection=retained_selection or {},
                records=records,
                expected_keys=expected_record_keys,
                records_path=records_path,
                execution_state=execution_state,
                state_path=state_path,
                group_summaries=group_summaries,
            )
            update_state(workers=[], current=None)

        if execute and is_formal_all_methods:
            records = _formal_record_order(records, plan, registry)
        elif is_formal_component_execution:
            records = _confirmatory_component_record_order(records, plan, registry)
        else:
            records = canonical_partial_records(records, expected_record_keys)
        atomic_json(records_path, records)
        update_state(phase="report_generation", current=None)
        is_ablation = any(job.get("study") for job in plan["jobs"])
        # Callers with a source-aware dedicated reporter can omit generic projections.
        for dataset in sorted({job["dataset"] for job in plan["jobs"]}
                              if generic_report_projections else set()):
            dataset_records = [
                record for record in records
                if record.get("dataset") == dataset
                and record.get("split") == plan["split"]
            ]
            metrics = list(
                protocol.get(
                    "metrics",
                    ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75"],
                )
            )
            report_scope = scope_from_records(
                dataset_records, dataset=dataset, split=plan["split"],
                max_images=max_images, diagnostic_only=max_images is not None)
            if is_ablation:
                for metric in metrics:
                    write_ablation_reports(
                        dataset_records,
                        registry,
                        run_dir / "reports" / dataset,
                        dataset=dataset,
                        metric=metric,
                        report_scope=report_scope,
                    )
            else:
                for metric in metrics:
                    diagonal_options = {}
                    if protocol.get("source_matched_only") is True:
                        diagonal_options = {
                            "include_failure_markers": True,
                            "caption": (
                                "Source-matched white-box diagonal on {} for {}; "
                                "off-diagonal cells are not run. No black-box panel or "
                                "BB Mean is evaluated. Values are 100 times the raw "
                                "[0,1] JSON metrics."
                            ).format(dataset, metric),
                        }
                    write_transfer_reports(
                        dataset_records,
                        registry,
                        run_dir / "reports" / dataset,
                        dataset=dataset,
                        metric=metric,
                        source_ids=plan["sources"],
                        target_ids=plan["targets"],
                        method_ids=plan["methods"],
                        preserve_method_order=protocol.get("report_method_order") == "declared",
                        report_scope=report_scope,
                        **diagonal_options,
                    )
            write_analysis_reports(
                dataset_records,
                registry,
                run_dir / "reports" / dataset,
                dataset=dataset,
                include_qualitative_payload_pairs=save_visualizations,
                split=plan["split"],
                report_scope=report_scope,
            )
            write_experiment_plots(
                dataset_records,
                registry,
                run_dir / "reports" / dataset / "figures",
                dataset=dataset,
                method_ids=plan["methods"],
                report_scope=report_scope,
            )
        write_paper_catalog(
            registry, run_dir / "reports" / "paper_catalog"
        )

        ready = sum(job["status"] == "ready" for job in plan["jobs"])
        completed_ready = sum(
            record.get("attack") != "clean"
            and record.get("status") == "complete"
            and bool(record.get("metrics"))
            for record in records
        )
        completed_clean = sum(
            record.get("attack") == "clean"
            and record.get("status") == "complete"
            and bool(record.get("metrics"))
            for record in records
        )
        failed_records = sum(
            str(record.get("status"))
            in {"failed", "complete_with_failures"}
            for record in records
        )
        groups_complete = (
            len(group_summaries) == len(groups) if execute else False
        )
        completion_ok = (
            execute
            and completed_ready == ready
            and completed_clean == len(clean_jobs)
            and failed_records == 0
            and groups_complete
        )
        summary = {
            "status": (
                "complete"
                if completion_ok
                else ("planned" if not execute else "incomplete")
            ),
            "protocol": protocol_id,
            "formal_eligible": bool(
                completion_ok
                and (
                    (
                        is_formal_all_methods
                        and max_images is None
                        and resolved_retention == RETENTION_MODE
                        and resolved_retained_count == 500
                    )
                    or is_formal_component_execution
                )
                and resolved_prediction_archive == PREDICTION_ARCHIVE_GZIP
                and int(seed) == int(protocol.get("seed", -1))
                and not download_weights
                and not keep_going
                and not save_visualizations
                and _source_manifest()["tree_sha256"]
                == provenance["source_manifest"]["tree_sha256"]
            ),
            "max_images": max_images,
            "seed": int(seed),
            "jobs": len(plan["jobs"]),
            "ready": ready,
            "completed_ready": completed_ready,
            "clean_evaluations": len(clean_jobs),
            "completed_clean": completed_clean,
            "artifact_required": sum(
                job["status"] == "artifact_required"
                for job in plan["jobs"]
            ),
            "adapter_required": sum(
                job["status"] == "adapter_required"
                for job in plan["jobs"]
            ),
            "skipped": sum(
                job["status"] == "skipped" for job in plan["jobs"]
            ),
            "failed_records": failed_records,
            "image_failures": sum(
                len(record.get("failures") or []) for record in records
            ),
            "records": len(records),
            "attack_groups": len(groups),
            "attack_groups_completed": len(group_summaries),
            "payload_retention": resolved_retention,
            "retained_images_per_attack_set": resolved_retained_count,
            "prediction_archive": resolved_prediction_archive or "none",
            "execution_scheduler": dict(plan["execution_scheduler"]),
            "retained_image_ids_sha256": (
                retained_selection.get("retained_image_ids_sha256")
                if retained_selection
                else None
            ),
            "image_selection": dict(plan.get("image_selection", {})),
            "records_sha256": file_digest(records_path),
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if "vcsf" in plan["methods"]:
            summary["qualification_pending"] = sum(
                job["status"] == "qualification_pending" for job in plan["jobs"]
            )
        atomic_json(run_dir / "summary.json", summary)
        if is_formal_component_execution and not summary["formal_eligible"]:
            raise RuntimeError(
                "Formal SVFTA component confirmation failed its completion or "
                "source-immutability audit"
            )
        if (
            execute
            and not completion_ok
            and resolved_retention == RETENTION_MODE
        ):
            raise RuntimeError(
                "Formal all-method completion audit failed"
            )
        if protocol_id == "coco_all_methods":
            update_state(phase="all_method_report_validation", current=None)
            write_all_methods_reports(
                records,
                registry,
                run_dir / "reports" / "coco",
                summary=summary,
                plan=plan,
                provenance=provenance,
                run_dir=run_dir,
            )
        elif completion_ok and is_formal_all_methods:
            if formal_dataset is None:
                raise RuntimeError(
                    "Formal report dataset was not resolved"
                )
            update_state(
                phase="all_method_report_validation", current=None
            )
            write_formal_all_methods_audits(
                records,
                registry,
                run_dir / "reports" / formal_dataset,
                summary=summary,
                plan=plan,
                provenance=provenance,
                run_dir=run_dir,
                protocol_id=protocol_id,
            )
        update_state(
            status=summary["status"],
            failed_records=failed_records,
            phase=(
                "complete" if completion_ok else execution_state["phase"]
            ),
            current=None,
        )
        if completion_ok and is_formal_all_methods:
            write_artifact_manifest(run_dir)
        return run_dir
    except (Exception, KeyboardInterrupt) as exc:
        record_order_error = None
        try:
            records[:] = canonical_partial_records(records, expected_record_keys)
        except RuntimeError as order_exc:
            record_order_error = str(order_exc)
        atomic_json(records_path, records)
        failure_summary = {
            "status": "failed",
            "protocol": protocol_id,
            "formal_eligible": False,
            "max_images": max_images,
            "seed": int(seed),
            "jobs": len(plan["jobs"]),
            "ready": sum(
                job["status"] == "ready" for job in plan["jobs"]
            ),
            "skipped": sum(
                job["status"] == "skipped" for job in plan["jobs"]
            ),
            "records": len(records),
            "failed_records": sum(
                str(record.get("status"))
                in {"failed", "complete_with_failures"}
                for record in records
            ),
            "reason": "{}: {}".format(type(exc).__name__, exc),
            "record_order_error": record_order_error,
            "payload_retention": resolved_retention,
            "prediction_archive": resolved_prediction_archive or "none",
            "execution_scheduler": dict(plan["execution_scheduler"]),
            "started_at": started_at,
            "failed_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(run_dir / "summary.json", failure_summary)
        update_state(
            status="failed",
            phase="failed",
            failed_records=failure_summary["failed_records"],
            reason=failure_summary["reason"],
        )
        raise
