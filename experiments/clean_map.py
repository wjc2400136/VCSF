from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lgp.registry import Registry
from lgp.runners.clean import run_clean_benchmark

from visualization_settings import (
    PREDICTION_VISUALIZATION_MAX_DETECTIONS,
    PREDICTION_VISUALIZATION_MAX_IMAGES,
    PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
    SAVE_PREDICTION_VISUALIZATIONS,
)


# Direct-run defaults. Edit only this block if you want another clean panel.
DEFAULT_DATASET = "coco"
DEFAULT_SPLIT = "val"
DEFAULT_TARGETS = "all"
DEFAULT_MAX_IMAGES = None
DEFAULT_DEVICE = "cuda:0"


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate clean mAP for the registered target detectors only."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["coco", "voc", "bdd100k"],
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--targets",
        default=DEFAULT_TARGETS,
        help="all or comma-separated detector IDs",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Run inference (default, retained for backward compatibility).",
    )
    mode.add_argument(
        "--plan-only",
        dest="execute",
        action="store_false",
        help="Only write the plan and placeholder reports; do not run inference.",
    )
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--max-images",
        type=int,
        default=DEFAULT_MAX_IMAGES,
        help="Debug-only image limit; omit for formal mAP.",
    )
    execution_device = parser.add_mutually_exclusive_group()
    execution_device.add_argument("--device", default=DEFAULT_DEVICE)
    execution_device.add_argument(
        "--devices",
        help="One or two comma-separated GPUs; complete target evaluations remain independent.",
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
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
        help=(
            "Do not save per-image box overlays; aggregate metrics and the "
            "clean summary figure are still generated."
        ),
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registry = Registry(PROJECT_ROOT)
    targets = (
        registry.target_ids()
        if args.targets == "all"
        else registry.expand_ids(_csv(args.targets), registry.target_ids())
    )
    output = run_clean_benchmark(
        registry,
        args.dataset,
        targets=targets,
        split=args.split,
        execute=args.execute,
        output_dir=args.output.resolve() if args.output else None,
        max_images=args.max_images,
        device=args.device,
        devices=_csv(args.devices) if args.devices is not None else None,
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
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print(
        "Status: {status}; targets={targets}; completed={completed}; failed={failed}".format(
            **summary
        )
    )
    print("CSV/TeX/figures:", output / "reports")
    if not args.execute:
        print("Plan only. Run this file without --plan-only to execute.")
