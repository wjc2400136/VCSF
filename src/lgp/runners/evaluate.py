from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from mmdet.apis import inference_detector
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from ..data.coco import CocoIndex, select_coco_images
from ..io import atomic_json, file_digest
from ..modeling import load_detector
from ..paths import project_root
from ..registry import Registry
from ..reporting.coco_summary import write_coco_bbox_summary
from ..reporting.scope import scope_from_records
from ..reporting.formatting import format_metric
from ..visualization import render_detection_overlay
from .retention import archive_predictions


def _interactive_progress_available() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, OSError):
        return False


def _runtime_config_digest(model_dir: Path, expected: Optional[str] = None) -> str:
    path = model_dir / "runtime_config.py"
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("Resolved detector runtime config is missing or linked")
    digest = file_digest(path)
    if expected is not None and digest != expected:
        raise RuntimeError("Resolved detector runtime config changed during evaluation")
    return digest


def _default_output(dataset: str, source: str, attack: str, target: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        project_root()
        / "outputs"
        / "evaluations"
        / dataset
        / source
        / attack
        / target
        / stamp
    )


def _input_from_run(run_dir: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    metadata_path = run_dir / "run.json"
    if not metadata_path.is_file():
        raise FileNotFoundError("Adversarial run has no run.json: {}".format(run_dir))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") not in {"complete", "complete_with_failures"}:
        raise RuntimeError("Adversarial run is not evaluable: {}".format(metadata.get("status")))
    if metadata.get("full_payload_available") is False:
        raise RuntimeError(
            "Adversarial payload was pruned after its complete target panel was "
            "evaluated; use the preserved predictions and metrics instead"
        )
    annotation_value = metadata.get("annotation", "annotations.json")
    annotation = Path(str(annotation_value))
    if not annotation.is_absolute():
        annotation = run_dir / annotation
    image_root_value = metadata.get("image_root", ".")
    image_root = Path(str(image_root_value))
    if not image_root.is_absolute():
        image_root = run_dir / image_root
    if not annotation.is_file() or not image_root.is_dir():
        raise FileNotFoundError(
            "Adversarial artifact paths are missing: annotation={}, image_root={}".format(
                annotation, image_root
            )
        )
    return annotation.resolve(), image_root.resolve(), metadata


def _category_ap(
    evaluator: COCOeval, category_names: Sequence[str]
) -> Dict[str, Optional[float]]:
    precision = evaluator.eval.get("precision")
    if precision is None:
        return {}
    result = {}
    for index, name in enumerate(category_names):
        values = precision[:, :, index, 0, -1]
        valid = values[values > -1]
        # A limited diagnostic subset may contain no ground truth for a
        # category. JSON ``null`` makes that undefined value explicit without
        # publishing NaN; formal full-split gates still require all categories
        # to be finite.
        result[name] = float(np.mean(valid)) if valid.size else None
    return result


def run_evaluation(
    registry: Registry,
    dataset_id: str,
    target_id: str,
    split: str = "val",
    adversarial_run: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    device: str = "cuda:0",
    download_weights: bool = False,
    keep_going: bool = False,
    save_visualizations: bool = False,
    visualization_score_threshold: float = 0.3,
    visualization_max_images: int = 20,
    visualization_max_detections: int = 100,
    prediction_archive: Optional[str] = None,
    target_checkpoint: Optional[Path] = None,
    image_ids: Optional[Sequence[int]] = None,
    progress_callback: Any = None,
    public_execution_context: Any = None,
    report_scope: Optional[Mapping[str, Any]] = None,
) -> Path:
    if public_execution_context is not None:
        from .vcsf_a10_main_context import MainExecutionContext
        if type(public_execution_context) is not MainExecutionContext:
            raise RuntimeError("Native evaluation requires the exact A10 main worker capability")
        public_execution_context.guard_evaluation(registry, dataset_id, split, target_id, adversarial_run, output_dir, image_ids, device,
            max_images, download_weights, keep_going, save_visualizations, prediction_archive, target_checkpoint)
    if save_visualizations:
        if not 0.0 <= visualization_score_threshold <= 1.0:
            raise ValueError("visualization_score_threshold must lie in [0, 1]")
        if visualization_max_images <= 0:
            raise ValueError("visualization_max_images must be positive")
        if visualization_max_detections <= 0:
            raise ValueError("visualization_max_detections must be positive")
    dataset = registry.dataset(dataset_id)
    target = registry.model(target_id)
    source_id = "clean"
    attack_id = "clean"
    run_metadata: Dict[str, Any] = {}
    if adversarial_run is not None:
        annotation_path, image_root, run_metadata = _input_from_run(adversarial_run.resolve())
        if run_metadata.get("dataset") != dataset_id:
            raise ValueError("Adversarial run dataset does not match --dataset")
        artifact_split = run_metadata.get("split")
        if artifact_split is not None and artifact_split != split:
            raise ValueError(
                "Adversarial run split '{}' does not match --split '{}'".format(
                    artifact_split, split
                )
            )
        source_id = str(run_metadata["source"])
        attack_id = str(run_metadata["attack"])
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = sorted(payload.get("images", []), key=lambda item: int(item["id"]))
        image_path = lambda item: (image_root / str(item["file_name"])).resolve()
    else:
        index = CocoIndex(dataset, split)
        annotation_path = index.annotation_path
        payload = index.payload
        images = index.images
        image_path = index.image_path
    images = select_coco_images(
        images,
        max_images=max_images,
        image_ids=image_ids,
    )
    expected_image_ids = [int(image["id"]) for image in images]
    run_dir = (
        output_dir
        or _default_output(dataset_id, source_id, attack_id, target_id)
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty evaluation: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    model, loaded_cfg, checkpoint = load_detector(
        target,
        dataset,
        run_dir / "model",
        device=device,
        download=download_weights,
        split=split,
        checkpoint_override=target_checkpoint,
    )
    if public_execution_context is not None:
        public_execution_context.loaded(model, loaded_cfg, checkpoint, target_id, run_dir / "model")
    runtime_config_sha256 = _runtime_config_digest(run_dir / "model")
    category_id_by_name = {
        str(category["name"]): int(category["id"])
        for category in payload.get("categories", [])
    }
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    evaluated_ids: List[int] = []
    inference_seconds = 0.0
    visualization_attempts = 0
    visualization_records: List[Dict[str, Any]] = []
    visualization_failures: List[Dict[str, Any]] = []
    visualization_dir = run_dir / "visualizations" / "predictions"
    progress = tqdm(
        images,
        desc="Clean/evaluate {}".format(target_id),
        unit="image",
        dynamic_ncols=True,
        disable=not _interactive_progress_available(),
    )
    for position, image in enumerate(progress):
        current_path = image_path(image)
        image_id = int(image["id"])
        try:
            if public_execution_context is not None:
                public_execution_context.before_evaluation_image(image_id, current_path)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            sample = inference_detector(model, str(current_path))
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            instances = sample.pred_instances.to("cpu")
            boxes = instances.bboxes.numpy()
            scores = instances.scores.numpy()
            labels = instances.labels.numpy()
            if (
                save_visualizations
                and visualization_attempts < visualization_max_images
            ):
                visualization_attempts += 1
                file_stem = Path(str(image.get("file_name", image_id))).stem
                safe_stem = "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in file_stem
                )
                overlay_path = visualization_dir / (
                    "{:012d}_{}.png".format(image_id, safe_stem or "image")
                )
                try:
                    visualization_record = render_detection_overlay(
                        current_path,
                        overlay_path,
                        boxes,
                        scores,
                        labels,
                        dataset.classes,
                        score_threshold=visualization_score_threshold,
                        max_detections=visualization_max_detections,
                        title="{} | {} | score >= {}".format(
                            target.display_name,
                            (
                                "clean"
                                if attack_id == "clean"
                                else "{} / {}".format(source_id, attack_id)
                            ),
                            format_metric(visualization_score_threshold),
                        ),
                    )
                    visualization_record.update(
                        {
                            "image_id": image_id,
                            "file_name": str(image.get("file_name", "")),
                            "dataset": dataset_id,
                            "split": split,
                            "source": source_id,
                            "attack": attack_id,
                            "target": target_id,
                        }
                    )
                    visualization_records.append(visualization_record)
                except Exception as exc:
                    visualization_failures.append(
                        {
                            "image_id": image_id,
                            "path": str(current_path),
                            "reason": "{}: {}".format(type(exc).__name__, exc),
                        }
                    )
            for box, score, label in zip(boxes, scores, labels):
                label_index = int(label)
                if label_index < 0 or label_index >= len(dataset.classes):
                    continue
                category_name = dataset.classes[label_index]
                category_id = category_id_by_name.get(category_name)
                if category_id is None:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                results.append(
                    {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        "score": float(score),
                    }
                )
            evaluated_ids.append(image_id)
            if progress_callback is not None:
                progress_callback(position + 1, len(images), image_id, "ok")
        except Exception as exc:
            failures.append(
                {
                    "position": position,
                    "image_id": image_id,
                    "path": str(current_path),
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                }
            )
            if not keep_going:
                raise
            if progress_callback is not None:
                progress_callback(position + 1, len(images), image_id, "failed")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions_path = run_dir / "predictions.json"
    atomic_json(predictions_path, results)
    predictions_sha256 = file_digest(predictions_path)
    ground_truth = COCO(str(annotation_path))
    if results:
        detections = ground_truth.loadRes(str(predictions_path))
    else:
        # pycocotools.COCO.loadRes indexes the first result and raises on an
        # empty list.  Construct the mathematically correct empty result set
        # so a complete detector failure evaluates to zero AP, not an error.
        detections = COCO()
        detections.dataset = deepcopy(ground_truth.dataset)
        detections.dataset["annotations"] = []
        detections.createIndex()
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = evaluated_ids
    evaluator.evaluate()
    evaluator.accumulate()
    # COCOeval hard-codes three decimal places in summarize(). Capture that
    # presentation-only output and print the same headline metrics uniformly
    # at the project-wide four-decimal precision instead.
    with redirect_stdout(StringIO()):
        evaluator.summarize()
    stats = evaluator.stats
    metrics = {
        "bbox_mAP": float(stats[0]),
        "bbox_mAP_50": float(stats[1]),
        "bbox_mAP_75": float(stats[2]),
        "bbox_mAP_small": float(stats[3]),
        "bbox_mAP_medium": float(stats[4]),
        "bbox_mAP_large": float(stats[5]),
        "bbox_AR_1": float(stats[6]),
        "bbox_AR_10": float(stats[7]),
        "bbox_AR_100": float(stats[8]),
        "bbox_AR_small": float(stats[9]),
        "bbox_AR_medium": float(stats[10]),
        "bbox_AR_large": float(stats[11]),
    }
    scope_context = report_scope or {}
    report_cap = max_images if max_images is not None else scope_context.get("max_images")
    if report_cap is None:
        report_cap = run_metadata.get("max_images")
    report_diagnostic = scope_context.get("diagnostic_only")
    if report_cap is not None or failures or run_metadata.get("diagnostic_only") is True:
        report_diagnostic = True
    elif report_diagnostic is not True and report_diagnostic is not False:
        # An image-ID subset alone declares no scientific role.
        report_diagnostic = (False if len(expected_image_ids) == dataset.split(split).expected_images
                             else None)
    coco_summary_artifacts = write_coco_bbox_summary(
        stats,
        run_dir,
        dataset=dataset_id,
        target=target_id,
        source=source_id,
        attack=attack_id,
        report_scope=scope_from_records(
            dataset=dataset_id, split=split, sample_n=len(evaluated_ids),
            max_images=report_cap, diagnostic_only=report_diagnostic,
        ),
    )
    print(coco_summary_artifacts["text"], flush=True)
    visualization_manifest = run_dir / "visualizations" / "manifest.json"
    prediction_visualizations = {
        "enabled": bool(save_visualizations),
        "score_threshold": float(visualization_score_threshold),
        "max_images": int(visualization_max_images),
        "max_detections_per_image": int(visualization_max_detections),
        "attempted": visualization_attempts,
        "saved": len(visualization_records),
        "directory": (
            str(visualization_dir.resolve()) if save_visualizations else None
        ),
        "manifest": (
            str(visualization_manifest.resolve())
            if save_visualizations
            else None
        ),
        "failures": visualization_failures,
    }
    if save_visualizations:
        atomic_json(
            visualization_manifest,
            {
                "schema_version": 1,
                "purpose": "post-NMS prediction overlays",
                "dataset": dataset_id,
                "split": split,
                "source": source_id,
                "attack": attack_id,
                "target": target_id,
                "score_threshold": float(visualization_score_threshold),
                "max_detections_per_image": int(
                    visualization_max_detections
                ),
                "images": visualization_records,
                "failures": visualization_failures,
            },
        )
        print(
            "Prediction overlays: {} saved under {}".format(
                len(visualization_records),
                visualization_dir,
            ),
            flush=True,
        )
    prediction_artifact = (
        archive_predictions(predictions_path, prediction_archive)
        if prediction_archive is not None
        else None
    )
    evaluated_image_ids_sha256 = hashlib.sha256(
        json.dumps(
            evaluated_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    expected_image_ids_sha256 = hashlib.sha256(
        json.dumps(
            expected_image_ids,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = {
        "status": "complete" if not failures else "complete_with_failures",
        "dataset": dataset_id,
        "split": split,
        "source": source_id,
        "attack": attack_id,
        "target": target_id,
        "adversarial_run": str(adversarial_run.resolve()) if adversarial_run else None,
        "images": len(images),
        "detections": len(results),
        "failures": failures,
        "evaluated_image_ids_sha256": evaluated_image_ids_sha256,
        "expected_image_ids_sha256": expected_image_ids_sha256,
        "evaluated_image_ids_match_expected": evaluated_ids == expected_image_ids,
        "metrics": metrics,
        "per_category_ap": _category_ap(evaluator, dataset.classes),
        "inference_seconds_total": inference_seconds,
        "inference_seconds_mean": inference_seconds / max(len(evaluated_ids), 1),
        "attack_quality_mean": run_metadata.get("quality_mean"),
        "attack_quality_definition": run_metadata.get("quality_definition"),
        "attack_pixel_identical_images": run_metadata.get(
            "pixel_identical_images"
        ),
        "attack_actual_gradient_evaluations_total": run_metadata.get(
            "actual_gradient_evaluations_total"
        ),
        "attack_actual_gradient_evaluations_mean": run_metadata.get(
            "actual_gradient_evaluations_mean"
        ),
        "attack_runtime_seconds_mean": run_metadata.get(
            "runtime_seconds_mean", run_metadata.get("attack_seconds_per_image")
        ),
        "attack_peak_cuda_memory_mb": run_metadata.get(
            "peak_cuda_memory_mb", run_metadata.get("peak_memory_mb")
        ),
        "budget_profile": run_metadata.get("budget_profile"),
        "budget_profile_definition": run_metadata.get("budget_profile_definition"),
        "baseline_profile_definition": run_metadata.get("baseline_profile_definition"),
        "gradient_evaluations_per_image": run_metadata.get(
            "gradient_evaluations_per_image"
        ),
        "auxiliary_passes": run_metadata.get("auxiliary_passes"),
        "parameters_sha256": run_metadata.get("parameters_sha256"),
        "fidelity_status": run_metadata.get("fidelity_status"),
        "fidelity_basis": run_metadata.get("fidelity_basis"),
        "semantic_contract": run_metadata.get("semantic_contract"),
        "paper": run_metadata.get("paper"),
        "audited_repository": run_metadata.get("audited_repository"),
        "audit_commit": run_metadata.get("audit_commit"),
        "runtime_compatibility": run_metadata.get("runtime_compatibility"),
        "implementation_adjudication": run_metadata.get(
            "implementation_adjudication"
        ),
        "implementation_adjudication_details": run_metadata.get(
            "implementation_adjudication_details"
        ),
        "code_commit": run_metadata.get("code_commit"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "runtime_config_sha256": _runtime_config_digest(
            run_dir / "model", runtime_config_sha256),
        "predictions_sha256": predictions_sha256,
        "predictions_artifact": prediction_artifact,
        "annotation": str(annotation_path),
        "coco_summary": {
            "decimal_places": coco_summary_artifacts["decimal_places"],
            "rows": coco_summary_artifacts["rows"],
            "text": str(coco_summary_artifacts["text_path"]),
            "csv": str(coco_summary_artifacts["csv_path"]),
            "tex": str(coco_summary_artifacts["tex_path"]),
        },
        "prediction_visualizations": prediction_visualizations,
    }
    atomic_json(run_dir / "metrics.json", summary)
    return run_dir
