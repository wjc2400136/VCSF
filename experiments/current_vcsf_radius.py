"""Run the current-input selected-A10 five-radius sensitivity programme."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "tools", ROOT / "src"):
    sys.path.insert(0, str(directory))
from lgp.runners.current_radius import main

if __name__ == "__main__":
    raise SystemExit(main())
