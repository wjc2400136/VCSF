"""Train the strict COCO victim checkpoint pair used by state transfer.

The default runs two independent single-GPU branches at once; one selected
device runs the same branches serially. Both
start from the exact same released Faster R-CNN checkpoint and consume the
same COCO train2017 order and augmentation trace. The only predeclared branch
difference is benign versus L-infinity adversarial continuation. Formal
training never attaches val2017 and publishes only the fixed final epoch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lgp.registry import Registry
from lgp.runners.training_state_pair import run_pair_training


def _csv(value: Optional[str]) -> Optional[list]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the strictly controlled standard/adversarial Faster R-CNN "
            "checkpoint pair for the retained-500 transfer benchmark."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", dest="execute", action="store_true")
    mode.add_argument(
        "--plan-only",
        dest="execute",
        action="store_false",
        help="Write the exact training plan without reading data or using a GPU.",
    )
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="One or two distinct CUDA devices; one device runs both branches serially.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Diagnostic train-image limit; omit for the formal 118,287-image pair.",
    )
    parser.add_argument("--output", type=Path, help="Custom immutable output directory")
    args = parser.parse_args()

    output = run_pair_training(
        Registry(PROJECT_ROOT),
        execute=args.execute,
        output_dir=args.output.resolve() if args.output else None,
        devices=_csv(args.devices),
        max_images=args.max_images,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print(
        "Status: {status}; selected_train_images={images}; strict_pair_eligible={eligible}".format(
            status=summary["status"],
            images=summary["selected_train_images"],
            eligible=summary["strict_training_controlled_pair_eligible"],
        )
    )
    if not args.execute:
        print("Plan only. Omit --plan-only to train the checkpoint pair.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
