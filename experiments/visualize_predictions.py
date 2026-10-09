from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lgp.runners.prediction_visualization import build_plan, execute_plan

from visualization_settings import (
    PREDICTION_VISUALIZATION_MAX_DETECTIONS,
    PREDICTION_VISUALIZATION_MAX_IMAGES,
    PREDICTION_VISUALIZATION_SCORE_THRESHOLD,
)


# Direct-run defaults. For adversarial visualization, paste the completed
# attack-run directory into DEFAULT_ADVERSARIAL_RUN. None means clean images.
DEFAULT_DATASET = "coco"
DEFAULT_SPLIT = "val"
DEFAULT_TARGET = "faster_rcnn_r50"
DEFAULT_ADVERSARIAL_RUN = None
DEFAULT_MAX_IMAGES = PREDICTION_VISUALIZATION_MAX_IMAGES
DEFAULT_SCORE_THRESHOLD = PREDICTION_VISUALIZATION_SCORE_THRESHOLD
DEFAULT_MAX_DETECTIONS = PREDICTION_VISUALIZATION_MAX_DETECTIONS
DEFAULT_DEVICE = "cuda:0"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Generate a small panel of detector predictions with boxes, class "
            "names and confidence scores."
        )
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["coco", "voc"],
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--target", default=DEFAULT_TARGET)
    target_group.add_argument(
        "--targets", nargs="+", help="Complete target jobs in canonical registry order."
    )
    parser.add_argument(
        "--adversarial-run",
        type=Path,
        default=(
            Path(DEFAULT_ADVERSARIAL_RUN)
            if DEFAULT_ADVERSARIAL_RUN is not None
            else None
        ),
    )
    parser.add_argument("--max-images", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=DEFAULT_MAX_DETECTIONS,
    )
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--device", default=DEFAULT_DEVICE)
    device_group.add_argument(
        "--devices", nargs="+", help="One or two distinct explicit GPUs, e.g. cuda:0 cuda:1."
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print jobs and assignments without runtime inputs, models or CUDA."
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = build_plan(
            PROJECT_ROOT, dataset=args.dataset, split=args.split,
            target=args.target, targets=args.targets,
            device=args.device, devices=args.devices,
            adversarial_run=args.adversarial_run, max_images=args.max_images,
            score_threshold=args.score_threshold, max_detections=args.max_detections,
            download_weights=args.download_weights, stop_on_error=args.stop_on_error,
            output=args.output,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.plan_only:
        print(json.dumps(plan, indent=2, allow_nan=False))
        return 0
    try:
        terminal = execute_plan(PROJECT_ROOT, plan)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, "Visualization failed: {}\n".format(exc))
    for record in terminal["jobs"]:
        print("{}: {}".format(record["target"], record["status"]))
        if record["status"] == "complete":
            print("Output directory:", record["output_dir"])
            print("Saved prediction overlays: {}/{}".format(record["saved"], record["attempted"]))
            print("Open this folder:", record["visualization_directory"])
        else:
            print(record.get("reason", "Not run after interruption"))
    print("Dispatch evidence:", plan["dispatch_dir"])
    print(
        "This is a qualitative panel; limited-image AP is not a formal benchmark result."
    )
    return 0 if terminal["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
