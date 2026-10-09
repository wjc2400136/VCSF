"""Verify the versioned lightweight source package without models or payload reads."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-images", type=int, choices=(1,), default=1,
                        help="Reserved metadata diagnostic cap; no images are read")
    args = parser.parse_args(argv)
    from lgp.public_package import verify_package, canonical, FREEZE, MANIFEST
    files = verify_package(ROOT)
    manifest_data = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    print(json.dumps(dict(schema=manifest_data["schema"], files=len(files),
        source_manifest_sha256=canonical(files), manifest=MANIFEST, freeze=FREEZE,
        candidate_only=manifest_data["candidate_only"], rights_status=manifest_data["rights_status"],
        release=manifest_data["release"], project_license=manifest_data.get("licensing", {}).get("project_license"),
        model_calls=0, CUDA_calls=0, inference_calls=0, attack_calls=0, AP_calls=0,
        plan_only=args.plan_only), sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
