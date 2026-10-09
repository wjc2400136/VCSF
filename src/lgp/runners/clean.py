from __future__ import annotations

import csv
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import torch

from ..io import atomic_json, file_digest
from ..paths import project_root
from ..registry import Registry
from ..reporting.clean_map import (
    COCO_BBOX_METRICS,
    clean_command,
    runtime_provenance,
    write_clean_reports,
)
from ..reporting.formatting import format_metric
from .evaluate import run_evaluation
from .formal_parallel import (
    ASSIGNMENT_ALGORITHM,
    build_execution_assignment,
    canonical_partial_records,
    experiment_expected_record_keys,
    normalize_execution_devices,
    run_formal_parallel,
    validate_available_cuda_devices,
)


CLEAN_METRICS = COCO_BBOX_METRICS


def _default_output(dataset: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root() / "outputs" / "experiments" / "clean_map" / dataset / stamp


def _write_plan_csv(
    path: Path, targets: Sequence[str], registry: Registry
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_order",
                "paper_group",
                "target",
                "source",
                "held_out",
                "status",
            ],
        )
        writer.writeheader()
        for target in targets:
            model = registry.model(target)
            writer.writerow(
                {
                    "paper_order": registry.model_position(target),
                    "paper_group": registry.model_group(target).id,
                    "target": target,
                    "source": model.source,
                    "held_out": model.held_out,
                    "status": "ready",
                }
            )


def _write_clean_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    output_dir: Path,
    *,
    context: Optional[Mapping[str, Any]] = None,
    include_manifest: bool = True,
) -> Dict[str, Path]:
    return write_clean_reports(
        records,
        registry,
        dataset,
        output_dir,
        context=context,
        include_manifest=include_manifest,
    )


def _run_serial_targets(
    registry: Registry, dataset_id: str, split: str, target_ids: Sequence[str],
    run_dir: Path, device: str, settings: Mapping[str, Any],
    records: List[Dict[str, Any]], publish_progress: Callable[[], None],
) -> None:
    for index, target in enumerate(target_ids):
        evaluation_dir = run_dir / "evaluations" / target
        print("Clean detector {}/{}: {} ({})".format(
            index + 1, len(target_ids), registry.model(target).display_name, target,
        ), flush=True)
        try:
            completed = run_evaluation(
                registry, dataset_id, target, split=split, output_dir=evaluation_dir,
                max_images=settings["max_images"], device=device,
                download_weights=settings["download_weights"],
                keep_going=settings["keep_going"],
                save_visualizations=settings["save_visualizations"],
                visualization_score_threshold=settings["visualization_score_threshold"],
                visualization_max_images=settings["visualization_max_images"],
                visualization_max_detections=settings["visualization_max_detections"],
            )
            record = json.loads((completed / "metrics.json").read_text(encoding="utf-8"))
            canonical_partial_records([record], experiment_expected_record_keys(
                {"jobs": []}, [(dataset_id, split, target)],
            ))
            record["predictions_sha256"] = file_digest(completed / "predictions.json")
            records[index] = record
            metrics = record["metrics"]
            print("Completed {}: AP={}, AP50={}, AP75={}".format(
                target, format_metric(metrics["bbox_mAP"]),
                format_metric(metrics["bbox_mAP_50"]), format_metric(metrics["bbox_mAP_75"]),
            ), flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            records[index] = {
                "dataset": dataset_id, "split": split, "source": "clean", "attack": "clean",
                "target": target, "status": "failed",
                "reason": "{}: {}".format(type(exc).__name__, exc), "metrics": {},
            }
            print("Failed {}: {}: {}".format(target, type(exc).__name__, exc), flush=True)
            if isinstance(exc, KeyboardInterrupt) or not settings["keep_going"]:
                raise
        finally:
            publish_progress()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_clean_benchmark(
    registry: Registry,
    dataset_id: str,
    targets: Optional[Sequence[str]] = None,
    split: str = "val",
    execute: bool = False,
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    device: str = "cuda:0",
    download_weights: bool = False,
    keep_going: bool = True,
    save_visualizations: bool = False,
    visualization_score_threshold: float = 0.3,
    visualization_max_images: int = 20,
    visualization_max_detections: int = 100,
    devices: Optional[Sequence[str]] = None,
) -> Path:
    dataset = registry.dataset(dataset_id)
    dataset.split(split)
    requested_targets = list(targets or registry.target_ids())
    registry.expand_ids(requested_targets, registry.target_ids())
    selected_targets = set(requested_targets)
    target_ids = [
        target for target in registry.target_ids() if target in selected_targets
    ]
    execution_devices = normalize_execution_devices(device, devices)
    if len(execution_devices) not in {1, 2}:
        raise ValueError("Clean evaluation supports one or two execution devices")
    if execute:
        validate_available_cuda_devices(execution_devices, include_single=True)
    device = execution_devices[0]
    clean_jobs = [(dataset_id, split, target) for target in target_ids]
    expected_keys = experiment_expected_record_keys({"jobs": []}, clean_jobs)
    scheduler = {
        "mode": "single_device_serial" if len(execution_devices) == 1 else "independent_target_workers",
        "devices": execution_devices, "worker_count": len(execution_devices),
        "assignment_algorithm": ASSIGNMENT_ALGORITHM, "global_record_writer": "coordinator_only",
        "job_atomicity": "complete_target_evaluation_per_worker",
    }
    run_dir = (output_dir or _default_output(dataset_id)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Refusing to overwrite non-empty clean evaluation: {}".format(run_dir)
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    provenance = runtime_provenance(registry.root)
    report_context: Dict[str, Any] = {
        "dataset": dataset_id,
        "split": split,
        "targets": target_ids,
        "execute": bool(execute),
        "max_images": max_images,
        "device": device,
        "execution_scheduler": scheduler,
        "prediction_visualizations": bool(save_visualizations),
        "command": clean_command(
            dataset_id,
            split=split,
            targets=target_ids,
            canonical_targets=registry.target_ids(),
            execute=execute,
            max_images=max_images,
            device=device,
            save_visualizations=save_visualizations,
            devices=execution_devices,
        ),
        **provenance,
    }
    assignment_path = run_dir / "execution_assignments.json"
    atomic_json(assignment_path, build_execution_assignment(clean_jobs, [], execution_devices))
    scheduler.update(assignment_manifest=assignment_path.name,
                     assignment_manifest_sha256=file_digest(assignment_path))
    atomic_json(
        run_dir / "plan.json",
        {
            "schema_version": 2,
            "purpose": "clean detector baseline",
            "dataset": dataset_id,
            "split": split,
            "targets": target_ids,
            "max_images": max_images,
            "execute": execute,
            "execution_scheduler": scheduler,
            "evaluation_provenance": provenance,
            "reproduction_command": report_context["command"],
            "prediction_visualizations": {
                "enabled": bool(save_visualizations),
                "score_threshold": float(visualization_score_threshold),
                "max_images": int(visualization_max_images),
                "max_detections_per_image": int(
                    visualization_max_detections
                ),
            },
        },
    )
    _write_plan_csv(run_dir / "plan.csv", target_ids, registry)
    records = [
        {
            "dataset": dataset_id,
            "split": split,
            "source": "clean",
            "attack": "clean",
            "target": target,
            "status": "planned",
            "metrics": {},
        }
        for target in target_ids
    ]
    records_path = run_dir / "records.json"
    state_path = run_dir / "execution_state.json"
    execution_state: Dict[str, Any] = {
        "schema_version": 1, "status": "running" if execute else "planned",
        "phase": "initializing", "current": None, "workers": [],
        "coordinator_pid": os.getpid(), "execution_scheduler": scheduler,
        "clean_evaluations_total": len(clean_jobs), "clean_evaluations_completed": 0,
        "attack_groups_total": 0, "attack_groups_completed": 0,
        "attack_evaluations_total": 0, "attack_evaluations_completed": 0,
        "failed_records": 0, "started_at": datetime.now(timezone.utc).isoformat(),
    }
    settings = {
        "max_images": max_images, "download_weights": download_weights,
        "keep_going": keep_going, "save_visualizations": save_visualizations,
        "visualization_score_threshold": visualization_score_threshold,
        "visualization_max_images": visualization_max_images,
        "visualization_max_detections": visualization_max_detections,
        "prediction_archive": None, "retention_mode": "keep_all",
        "clean_output_layout": "flat_targets",
    }

    def publish_progress() -> None:
        records[:] = canonical_partial_records(records, expected_keys)
        atomic_json(records_path, records)
        execution_state["clean_evaluations_completed"] = sum(
            record.get("status") in {"complete", "complete_with_failures"}
            for record in records
        )
        execution_state["failed_records"] = sum(
            record.get("status") in {"failed", "complete_with_failures"} for record in records
        )
        execution_state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, execution_state)
        _write_clean_reports(records, registry, dataset_id, run_dir / "reports",
                             context=report_context, include_manifest=False)

    def finalize(reason: Optional[str] = None) -> None:
        records[:] = canonical_partial_records(records, expected_keys)
        failed_targets = sum(record.get("status") == "failed" for record in records)
        image_failures = sum(len(record.get("failures") or []) for record in records)
        completed = sum(record.get("status") in {"complete", "complete_with_failures"}
                        for record in records)
        all_complete = (completed == len(target_ids) and not failed_targets
                        and not image_failures
                        and all(record.get("status") == "complete" for record in records))
        status = ("failed" if reason else "planned" if not execute else
                  "complete" if all_complete else "complete_with_failures")
        summary = {
            "status": status, "dataset": dataset_id, "split": split,
            "targets": len(target_ids), "completed": completed, "failed": failed_targets,
            "image_failures": image_failures, "max_images": max_images,
            "execution_scheduler": scheduler,
        }
        if reason:
            summary["reason"] = reason
            report_context["execution_failed"] = True
        execution_state.update(
            status=status, phase="terminal", current=None,
            clean_evaluations_completed=completed,
            failed_records=sum(record.get("status") in {"failed", "complete_with_failures"}
                               for record in records),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        if reason:
            execution_state["reason"] = reason
        atomic_json(records_path, records)
        atomic_json(run_dir / "summary.json", summary)
        atomic_json(state_path, execution_state)

    parallel_records: List[Dict[str, Any]] = []
    try:
        publish_progress()
        if execute and len(execution_devices) == 1:
            execution_state["phase"] = "clean_evaluation"
            _run_serial_targets(registry, dataset_id, split, target_ids, run_dir,
                                device, settings, records, publish_progress)
        elif execute:
            try:
                run_formal_parallel(
                    registry=registry, run_dir=run_dir, devices=execution_devices,
                    clean_jobs=clean_jobs, group_items=[], settings=settings,
                    retained_selection={}, records=parallel_records, expected_keys=expected_keys,
                    records_path=records_path, execution_state=execution_state,
                    state_path=state_path, group_summaries=[],
                )
            finally:
                # Keep unstarted targets as planned records, including on worker failure.
                actual = canonical_partial_records(parallel_records, expected_keys)
                by_target = {record["target"]: record for record in actual}
                for worker in execution_state.get("workers", []):
                    current = worker.get("current") or {}
                    target = current.get("target")
                    if target and target not in by_target:
                        interrupted = {
                            "dataset": dataset_id, "split": split, "source": "clean",
                            "attack": "clean", "target": target, "status": "failed",
                            "metrics": {}, "reason": (
                                "Clean evaluation started but returned no terminal record; "
                                "worker status: {}".format(worker["status"])
                            ),
                        }
                        canonical_partial_records([interrupted], expected_keys)
                        by_target[target] = interrupted
                records[:] = [by_target.get(record["target"], record) for record in records]
        finalize()
        _write_clean_reports(records, registry, dataset_id, run_dir / "reports",
                             context=report_context, include_manifest=True)
    except (Exception, KeyboardInterrupt) as exc:
        finalize("{}: {}".format(type(exc).__name__, exc))
        try:
            _write_clean_reports(records, registry, dataset_id, run_dir / "reports",
                                 context=report_context, include_manifest=True)
        except Exception as reporting_exc:
            # Preserve the execution failure even if its diagnostic report also fails.
            finalize("{}: {}; report error: {}: {}".format(
                type(exc).__name__, exc, type(reporting_exc).__name__, reporting_exc,
            ))
        raise
    return run_dir
