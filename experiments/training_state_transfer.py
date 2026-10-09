"""Evaluate current COCO source payloads against the controlled victim pair."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from lgp.registry import Registry
from lgp.runners.current_training_state import run
from visualization_settings import (
    SAVE_PREDICTION_VISUALIZATIONS, PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
    PREDICTION_VISUALIZATION_MAX_IMAGES, PREDICTION_VISUALIZATION_MAX_DETECTIONS,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Inspect without reading payloads, checkpoints or using GPUs.")
    parser.add_argument("--source-main", type=Path, help="Current main-transfer run with preserved Swin-T source PNGs.")
    parser.add_argument("--pair-manifest", type=Path, help="Qualified checkpoint-pair manifest; defaults to its registered location.")
    parser.add_argument("--pair-repository", type=Path, help="Repository containing the pair's relative artifact paths; default: this repository.")
    parser.add_argument("--methods", help="Optional current-main method subset, comma separated.")
    parser.add_argument("--max-images", type=int, help="Diagnostic limit; omit for the full COCO 5,000-image scope.")
    parser.add_argument("--devices", default="cuda:0", help="One or two distinct explicit CUDA devices, comma separated.")
    parser.add_argument("--output", type=Path, help="New immutable output directory.")
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument("--visualize-predictions", dest="save_visualizations", action="store_true")
    visualization.add_argument("--no-visualize-predictions", dest="save_visualizations", action="store_false")
    parser.set_defaults(save_visualizations=SAVE_PREDICTION_VISUALIZATIONS)
    parser.add_argument("--visualization-score-threshold", type=float, default=PREDICTION_VISUALIZATION_SCORE_THRESHOLD)
    parser.add_argument("--visualization-max-images", type=int, default=PREDICTION_VISUALIZATION_MAX_IMAGES)
    parser.add_argument("--visualization-max-detections", type=int, default=PREDICTION_VISUALIZATION_MAX_DETECTIONS)
    args = parser.parse_args(argv)
    if not args.plan_only and args.source_main is None:
        parser.error("--source-main is required for execution")
    output = run(
        Registry(ROOT), source_main=args.source_main, pair_manifest=args.pair_manifest,
        pair_repository=args.pair_repository,
        methods=None if args.methods is None else [value.strip() for value in args.methods.split(",")],
        devices=[value.strip() for value in args.devices.split(",")], max_images=args.max_images,
        output=args.output, plan_only=args.plan_only, save_visualizations=args.save_visualizations,
        visualization_score_threshold=args.visualization_score_threshold,
        visualization_max_images=args.visualization_max_images,
        visualization_max_detections=args.visualization_max_detections,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print("Status: {}; inference jobs={}; generation jobs={}".format(summary["status"], summary["inference_jobs"], summary["generation_jobs"]))
    print("Records, predictions and reports:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
