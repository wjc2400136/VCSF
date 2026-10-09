"""Run the COCO retained-500 Common-2 preprocessing-defense benchmark.

This direct entry point evaluates all eleven maintained attacks from the two
white-box sources that support every method, under eleven deterministic
victim-side input transforms and all sixteen target detectors. The default
executes on one GPU; pass ``--devices cuda:0,cuda:1`` for the formal two-worker
run. Omit ``--max-images`` for the retained-500 benchmark.

The threat model is deliberately non-adaptive: accepted adversarial payloads
are reused read-only and the attacker does not know or optimize through the
defense. The resulting metrics are eligible only for the retained-500 defense
benchmark and never for a full-val2017 COCO claim.
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
from lgp.runners.preprocessing_defense import run_preprocessing_defense


def _csv(value: Optional[str]) -> Optional[list]:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "COCO retained-500 Common-2 eleven-method, sixteen-target, "
            "oblivious input-preprocessing defense benchmark."
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
        "--source-acceptance",
        type=Path,
        help=(
            "Accepted COCO evidence JSON. Default: "
            "outputs/reports/coco/formal_all_methods_acceptance.json."
        ),
    )
    parser.add_argument(
        "--sources",
        help="Diagnostic subset of Common-2 source IDs; omit for both sources.",
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
            "Diagnostic image limit. Omit for the formal retained-500 scope; "
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
            "Workers own complete clean-variant or source-method groups."
        ),
    )
    parser.add_argument("--download-weights", action="store_true")
    parser.add_argument("--output", type=Path, help="Custom immutable output directory")
    args = parser.parse_args()

    registry = Registry(PROJECT_ROOT)
    output = run_preprocessing_defense(
        registry,
        execute=args.execute,
        output_dir=args.output.resolve() if args.output else None,
        source_acceptance=(
            args.source_acceptance.resolve() if args.source_acceptance else None
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
        "Status: {status}; jobs={jobs}; records={records}; retained500_benchmark_eligible={eligible}".format(
            status=summary["status"],
            jobs=summary.get("jobs", summary["records"]),
            records=summary["records"],
            eligible=summary["retained500_benchmark_eligible"],
        )
    )
    print("TeX, CSV and figures:", output / "reports" / "coco")
    if not args.execute:
        print("Plan only. Run this file without --plan-only to execute.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
