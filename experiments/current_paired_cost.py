"""Direct current-input paired cost reproduction; computes no AP."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    from lgp.runners.current_cost import main
    raise SystemExit(main())
