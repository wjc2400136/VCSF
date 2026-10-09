"""CPU-only public A10 source identity qualification; never loads a detector."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lgp.attacks.factory import build_attack
from lgp.attacks.vcsf_public import verify_public_identity
from lgp.registry import Registry


def pinned_cpu_runtime():
    import torch

    expected = {"torch": "2.0.0+cu118", "torchvision": "0.15.1+cu118", "mmcv": "2.0.1",
                "mmengine": "0.7.4", "mmdet": "3.0.0", "mmyolo": "0.6.0"}
    versions = {name: importlib.metadata.version(name) for name in expected}
    if (Path(sys.prefix).name != "oda" or sys.version_info[:3] != (3, 8, 20)
            or versions != expected or torch.version.cuda != "11.8"
            or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or torch.cuda.is_initialized()):
        raise RuntimeError("Require pinned ODA with CUDA hidden and uninitialized")
    return {"python": "3.8.20", "packages": versions, "torch_cuda": "11.8",
            "cuda_hidden": True, "cuda_initialized": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--max-images", type=int, choices=(1,), default=1,
                        help="Reserved toy cap; no dataset images or models are accessed")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.plan_only and args.output is not None:
        parser.error("--plan-only does not write output")
    runtime = pinned_cpu_runtime()
    registry = Registry(ROOT)
    identity = verify_public_identity(ROOT, registry.attack("vcsf"))
    attack = build_attack("vcsf", object(), registry.attack("vcsf").parameters)
    if asdict(attack.config) != identity["parameters"]:
        raise RuntimeError("Factory A10 configuration differs from the registered builder")
    report = dict(identity, schema_version=1,
        status="A10_public_source_plan_only" if args.plan_only else "A10_public_source_CPU_identity_checked",
        runtime=runtime, sources=registry.source_ids(),
        real_model_qualification="NR", result_reuse_bridge="not_accepted",
        source201_qualification="not_qualified", device_mode_release="not_validated",
        model_loads=0, gpu_calls=0, AP_evaluations=0, dataset_images_accessed=0,
        toy_path_equivalence="separate_focused_tests_required", max_images=args.max_images)
    if not args.plan_only:
        parent = ROOT / "outputs/diagnostics/vcsf_a10_public_qualification"
        output = (args.output or parent / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")).absolute()
        if output.resolve() != output or output.parent != parent or output.exists():
            raise RuntimeError("Use a fresh direct public A10 CPU qualification output")
        output.mkdir(parents=True, exist_ok=False)
        with (output / "receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        report["output"] = output.relative_to(ROOT).as_posix()
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
