"""Run the paired COCO retained-500 adaptive preprocessing benchmark.

Every Common-2 source/method/defense payload is regenerated from the same
frozen clean image IDs through the known preprocessing-to-source-detector
pipeline.  Exact deployed preprocessing is used in the forward pass and the
predeclared BPDA rule supplies its derivative.  The target detector is never
queried during generation.  Omit ``--max-images`` for the formal retained-500
scope; use ``--devices cuda:0,cuda:1`` for the canonical two-worker run.
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
from lgp.runners.adaptive_preprocessing_defense import (
    run_adaptive_preprocessing_defense,
)


def _csv(value: Optional[str]) -> Optional[list]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "COCO retained-500 Common-2 eleven-method, sixteen-target, "
            "source-pipeline-adaptive preprocessing benchmark."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute",
        dest="execute",
        action="store_true",
        help="Execute the benchmark (default).",
    )
    mode.add_argument(
        "--plan-only",
        dest="execute",
        action="store_false",
        help="Write the exact plan and placeholder TeX without GPU work.",
    )
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--paired-acceptance",
        type=Path,
        help=(
            "Accepted paired oblivious benchmark JSON. The default is frozen "
            "in configs/experiments/protocols.yaml."
        ),
    )
    parser.add_argument(
        "--sources",
        help="Diagnostic Common-2 source subset; omit for both sources.",
    )
    parser.add_argument(
        "--methods",
        help="Diagnostic comma-separated method subset; omit for all eleven.",
    )
    parser.add_argument(
        "--targets",
        help="Diagnostic comma-separated target subset; omit for all sixteen.",
    )
    parser.add_argument(
        "--defenses",
        help="Diagnostic comma-separated defense subset; omit for all eleven.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help=(
            "Diagnostic image limit. Omit for retained-500 eligibility; "
            "limited runs are never benchmark-eligible."
        ),
    )
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--device",
        help="One serial PyTorch device (default: cuda:0).",
    )
    execution.add_argument(
        "--devices",
        help=(
            "Comma-separated independent workers, for example cuda:0,cuda:1. "
            "Each worker owns complete source-method groups."
        ),
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--output", type=Path, help="Custom immutable output directory")
    args = parser.parse_args()

    output = run_adaptive_preprocessing_defense(
        Registry(PROJECT_ROOT),
        execute=args.execute,
        output_dir=args.output.resolve() if args.output else None,
        paired_acceptance=(
            args.paired_acceptance.resolve() if args.paired_acceptance else None
        ),
        source_override=_csv(args.sources),
        method_override=_csv(args.methods),
        target_override=_csv(args.targets),
        defense_override=_csv(args.defenses),
        max_images=args.max_images,
        device=args.device or "cuda:0",
        devices=_csv(args.devices),
        download_weights=args.download_weights,
    )
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    print("Output directory:", output)
    print(
        "Status: {status}; jobs={jobs}; records={records}; "
        "adaptive_retained500_benchmark_eligible={eligible}".format(
            status=summary["status"],
            jobs=summary.get("jobs", summary.get("records", 0)),
            records=summary.get("records", 0),
            eligible=summary["adaptive_retained500_benchmark_eligible"],
        )
    )
    print("TeX and CSV:", output / "reports" / "coco")
    if not args.execute:
        print("Plan only. Run this file without --plan-only to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
