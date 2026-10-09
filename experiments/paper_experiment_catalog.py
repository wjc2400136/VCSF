from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lgp.registry import Registry
from lgp.reporting import write_paper_catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the paper-experiment coverage catalog as CSV, Markdown and TeX."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "paper_experiment_catalog",
    )
    args = parser.parse_args()
    paths = write_paper_catalog(Registry(PROJECT_ROOT), args.output.resolve())
    for name, path in paths.items():
        print("{}: {}".format(name, path))
