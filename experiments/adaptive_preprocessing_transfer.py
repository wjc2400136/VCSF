"""Generate current COCO retained-500 attacks through registered source BPDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from lgp.registry import Registry
from lgp.runners.current_preprocessing import run
from visualization_settings import (
    SAVE_PREDICTION_VISUALIZATIONS, PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
    PREDICTION_VISUALIZATION_MAX_IMAGES, PREDICTION_VISUALIZATION_MAX_DETECTIONS,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Inspect without reading payloads, checkpoints or using GPUs.")
    parser.add_argument("--source-main", type=Path, help="Current main-transfer run binding Common-2 IDs, seeds, checkpoints and identity pixels.")
    parser.add_argument("--methods", help="Optional current-main method subset, comma separated.")
    parser.add_argument("--sources", help="Optional Common-2 source subset, comma separated.")
    parser.add_argument("--targets", help="Optional registered target subset, comma separated.")
    parser.add_argument("--defenses", help="Optional registered preprocessing subset, comma separated.")
    parser.add_argument("--max-images", type=int, help="Diagnostic retained-500 prefix limit; omit for all fixed 500 images.")
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
    choices = {key: None if getattr(args, key) is None else [value.strip() for value in getattr(args, key).split(",")]
               for key in ("methods", "sources", "targets", "defenses")}
    output = run(Registry(ROOT), source_main=args.source_main, **choices,
        devices=[value.strip() for value in args.devices.split(",")], max_images=args.max_images,
        output=args.output, plan_only=args.plan_only, adaptive=True, save_visualizations=args.save_visualizations,
        visualization_score_threshold=args.visualization_score_threshold,
        visualization_max_images=args.visualization_max_images,
        visualization_max_detections=args.visualization_max_detections)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print("Status: {}; inference jobs={}; generation jobs={}".format(summary["status"], summary["inference_jobs"], summary["generation_jobs"]))
    print("Preserved transformed images, predictions and reports:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())