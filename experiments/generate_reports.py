from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lgp.cli import main


DEFAULT_INPUT = None
DEFAULT_DATASET = "coco"
DEFAULT_METRIC = "all"


def _latest_records(dataset: str) -> Path:
    candidates = sorted(
        (PROJECT_ROOT / "outputs").rglob("records.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(records, list):
            continue
        if any(
            record.get("dataset") == dataset
            and any(
                value is not None
                for value in (record.get("metrics") or {}).values()
            )
            for record in records
        ):
            return path
    raise FileNotFoundError(
        "No executed {} records.json was found under outputs.".format(dataset)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild CSV, TeX, PNG and PDF reports from records.json."
    )
    parser.add_argument("input", nargs="?", type=Path)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=["coco", "voc", "bdd100k"],
    )
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        input_path = (
            args.input.resolve()
            if args.input is not None
            else (
                Path(DEFAULT_INPUT).resolve()
                if DEFAULT_INPUT is not None
                else _latest_records(args.dataset)
            )
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        args.output.resolve()
        if args.output is not None
        else input_path.parent / "reports_regenerated" / args.dataset / stamp
    )
    raise SystemExit(
        main(
            [
                "report",
                "--input",
                str(input_path),
                "--dataset",
                args.dataset,
                "--metric",
                args.metric,
                "--output",
                str(output_path),
            ]
        )
    )
