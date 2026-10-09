"""Run the exact selected-A10 final-background controls on fresh COCO inputs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tools", ROOT / "src"):
    sys.path.insert(0, str(directory))
from lgp.runners.current_background import main

if __name__ == "__main__":
    raise SystemExit(main())