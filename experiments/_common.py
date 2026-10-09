from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lgp.registry import Registry
from lgp.runners.experiment import run_experiment

from visualization_settings import (
    PREDICTION_VISUALIZATION_MAX_DETECTIONS,
    PREDICTION_VISUALIZATION_MAX_IMAGES,
    PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
    SAVE_PREDICTION_VISUALIZATIONS,
)


def _csv(value: Optional[str]) -> Optional[list[str]]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def studies_for_method(registry: Registry, method: str) -> list[str]:
    return [
        study_id
        for study_id, study in registry.ablation_studies.items()
        if str(study.get("method", "svfta")) == method
    ]


def run_protocol(
    protocol: str,
    description: str,
    fixed_method: Optional[str] = None,
    fixed_studies: Optional[Sequence[str]] = None,
    default_datasets: Optional[Sequence[str]] = None,
    default_sources: Optional[Sequence[str]] = None,
    default_stop_on_error: bool = False,
    default_payload_retention: Optional[str] = None,
    default_retained_image_count: Optional[int] = None,
    default_prediction_archive: Optional[str] = None,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Execute the run (default, retained for backward compatibility).",
    )
    mode.add_argument(
        "--plan-only",
        dest="execute",
        action="store_false",
        help="Only write the plan and placeholder reports; do not run GPU work.",
    )
    parser.set_defaults(execute=True)
    parser.add_argument("--datasets", help="Comma-separated datasets, for example coco,voc")
    parser.add_argument("--sources", help="Comma-separated source detector IDs")
    parser.add_argument("--targets", help="Comma-separated target detector IDs")
    if fixed_method is None:
        parser.add_argument("--methods", help="Comma-separated attack IDs")
    parser.add_argument(
        "--studies",
        help="Comma-separated study IDs; default: every study covered by this script.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Debug image limit; omit for a formal run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    execution_device = parser.add_mutually_exclusive_group()
    execution_device.add_argument(
        "--device",
        help=(
            "Single serial PyTorch device (default: cuda:0). This remains the "
            "beginner-friendly and one-GPU execution mode."
        ),
    )
    execution_device.add_argument(
        "--devices",
        help=(
            "Comma-separated independent group-worker devices, for example "
            "cuda:0,cuda:1. One or two GPUs run the same declared jobs; "
            "each group stays on one worker."
        ),
    )
    parser.add_argument("--download-weights", action="store_true")
    failure_mode = parser.add_mutually_exclusive_group()
    failure_mode.add_argument(
        "--stop-on-error",
        dest="stop_on_error",
        action="store_true",
        help="Stop on the first error.",
    )
    failure_mode.add_argument(
        "--keep-going",
        dest="stop_on_error",
        action="store_false",
        help="Record failures and continue (not permitted by the formal all-method protocol).",
    )
    parser.set_defaults(stop_on_error=default_stop_on_error)
    parser.add_argument(
        "--payload-retention",
        choices=["keep_all", "fixed_count_after_group_validation"],
        default=default_payload_retention,
        help="Generated-image lifecycle; protocol default is used when omitted.",
    )
    parser.add_argument(
        "--retained-image-count",
        type=int,
        default=default_retained_image_count,
        help="Evidence images retained only after a complete target-panel audit.",
    )
    parser.add_argument(
        "--prediction-archive",
        choices=["none", "gzip"],
        default=default_prediction_archive,
        help="Lossless prediction-record archive; protocol default is used when omitted.",
    )
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        "--visualize-predictions",
        dest="save_visualizations",
        action="store_true",
        help="Save post-NMS boxes, class labels and confidence scores.",
    )
    visualization.add_argument(
        "--no-visualize-predictions",
        dest="save_visualizations",
        action="store_false",
        help="Disable prediction overlays.",
    )
    parser.set_defaults(
        save_visualizations=SAVE_PREDICTION_VISUALIZATIONS
    )
    parser.add_argument(
        "--visualization-score-threshold",
        type=float,
        default=PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--visualization-max-images",
        type=int,
        default=PREDICTION_VISUALIZATION_MAX_IMAGES,
    )
    parser.add_argument(
        "--visualization-max-detections",
        type=int,
        default=PREDICTION_VISUALIZATION_MAX_DETECTIONS,
    )
    parser.add_argument("--output", type=Path, help="Custom output directory")
    args = parser.parse_args()

    registry = Registry(PROJECT_ROOT)
    studies = _csv(args.studies)
    if studies is None and fixed_studies is not None:
        studies = list(fixed_studies)
    if (
        studies is None
        and fixed_method is not None
        and protocol == "method_ablation"
    ):
        studies = studies_for_method(registry, fixed_method)

    result = run_experiment(
        registry,
        protocol,
        execute=args.execute,
        output_dir=args.output.resolve() if args.output else None,
        dataset_override=_csv(args.datasets) or (
            list(default_datasets) if default_datasets is not None else None
        ),
        source_override=_csv(args.sources) or (
            list(default_sources) if default_sources is not None else None
        ),
        target_override=_csv(args.targets),
        method_override=(
            [fixed_method]
            if fixed_method is not None
            else _csv(getattr(args, "methods", None))
        ),
        study_override=studies,
        max_images=args.max_images,
        seed=args.seed,
        device=args.device or "cuda:0",
        devices=_csv(args.devices),
        download_weights=args.download_weights,
        keep_going=not args.stop_on_error,
        save_visualizations=args.save_visualizations,
        visualization_score_threshold=(
            args.visualization_score_threshold
        ),
        visualization_max_images=args.visualization_max_images,
        visualization_max_detections=(
            args.visualization_max_detections
        ),
        payload_retention=args.payload_retention,
        retained_image_count=args.retained_image_count,
        prediction_archive=args.prediction_archive,
    )
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", result)
    print(
        "Status: {status}; jobs={jobs}; ready={ready}; "
        "artifact_required={artifact_required}; skipped={skipped}".format(**summary)
    )
    print("TeX and figures:", result / "reports")
    if not args.execute:
        print("Plan only. Run this file without --plan-only to execute.")
    return 0
