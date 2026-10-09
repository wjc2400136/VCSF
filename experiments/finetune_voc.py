"""Fine-tune the registered VOC detector panel on one or two independent GPUs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lgp.registry import Registry
from lgp.runners.finetune import run_finetuning


def _models(value):
    return None if value == "all" else [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run immutable VOC detector fine-tuning with the registered training recipes.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", dest="execute", action="store_true")
    mode.add_argument("--plan-only", dest="execute", action="store_false", help="Write the complete plan without data, checkpoints or GPU access.")
    parser.set_defaults(execute=True)
    parser.add_argument("--devices", default="cuda:0", help="One or two distinct CUDA devices, such as cuda:0,cuda:1.")
    parser.add_argument("--models", default="all", help="Canonical detector IDs, comma-separated, or all.")
    parser.add_argument("--max-images", type=int, help="Diagnostic number of distinct usable training images; never publishes canonical weights.")
    parser.add_argument("--frozen-existing", default="", help="Already qualified VOC model IDs to preserve under their original commits, or all.")
    parser.add_argument("--min-free-gib", type=float, default=12.0, help="Minimum free disk space before launching each model.")
    parser.add_argument("--output", type=Path, help="New immutable output directory; existing directories are rejected.")
    args = parser.parse_args()
    registry = Registry(PROJECT_ROOT)
    frozen = registry.target_ids() if args.frozen_existing == "all" else _models(args.frozen_existing)
    output = run_finetuning(registry, execute=args.execute, output_dir=args.output,
        devices=[item.strip() for item in args.devices.split(",") if item.strip()],
        models=_models(args.models), max_images=args.max_images, frozen_existing=frozen,
        min_free_gib=args.min_free_gib,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print("Status: {}; jobs={}/{}; VOC panel qualified={}".format(summary["status"], summary["completed_jobs"], summary["total_jobs"], summary["voc_panel_qualified"]))
    if not args.execute:
        print("Plan only. Omit --plan-only to execute the selected VOC jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
