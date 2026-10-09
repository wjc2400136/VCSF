"""Render fixed COCO current-main saved predictions offline on CPU, never author-certified figures."""
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path,
                        help="One complete, unlimited main_transfer run; never a merged/historical directory.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Without inputs: display contract only. With --main-run: full loader validation, no output.")
    parser.add_argument("--max-images", type=int, choices=(1, 2, 3),
                        help="Display diagnostic only; always validate the full 5000-image scope.")
    parser.add_argument("--output", type=Path,
                        help="New directory outside --main-run; PNG, manifest and bound request only, no PDF.")
    args = parser.parse_args(argv)
    if not args.plan_only and args.main_run is None:
        parser.error("Execution requires --main-run; author-certified figures are never substituted.")
    if not args.plan_only and args.output is None:
        parser.error("Execution requires --output; it must be a new directory outside --main-run.")
    from lgp.reporting.current_qualitative import plan_current_display, run_current_qualitative

    try:
        if args.main_run is None:
            result = plan_current_display(PROJECT_ROOT, args.max_images)
        else:
            from lgp.registry import Registry

            result = run_current_qualitative(Registry(PROJECT_ROOT), args.main_run,
                                            plan_only=args.plan_only, output=args.output,
                                            max_images=args.max_images)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
