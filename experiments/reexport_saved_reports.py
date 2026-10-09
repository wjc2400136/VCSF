"""Re-export labels from saved JSON using CPU only; no detector or evaluation."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgp.reporting.reexport import main

if __name__ == "__main__":
    raise SystemExit(main())