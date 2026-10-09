"""Direct server-only paired runtime measurement; never computes AP."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from lgp.runners.cost_calibration import main
    raise SystemExit(main())
