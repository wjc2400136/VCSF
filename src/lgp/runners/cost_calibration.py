"""Immutable, paired physical-cost measurements of accepted attack settings."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone

from ..io import atomic_json, file_digest
from ..paths import project_root


PROTOCOL = "vcsf_common2_cost_calibration"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def now():
    return datetime.now(timezone.utc).isoformat()


def ordered_methods(methods, position):
    count = len(methods)
    require(count > 0 and count % 2 == 0, "Balanced carryover schedule requires an even panel")
    # Williams rows balance both ordinal position and within-image predecessors.
    base = [0]
    for i in range(1, count):
        base.append((i + 1) // 2 if i % 2 else count - i // 2)
    return [methods[(value + position) % count] for value in base]


def source_snapshot(root):
    paths = [path for folder in ("src", "configs")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.suffix in (".py", ".yaml")]
    paths.append(root / "experiments" / "vcsf_common2_cost_calibration.py")
    return {str(path.relative_to(root)): file_digest(path) for path in sorted(paths)}


def environment(server_only=True):
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda", "Scientific execution is server-only")
    if server_only:
        require(Path(sys.prefix).name == "oda", "The pinned ODA environment is required")
        require(Path(sys.executable).as_posix() == str(Path(sys.prefix) / "bin" / "python"),
            "The pinned server ODA interpreter is required")
    pinned = {"torch": "2.0.0", "torchvision": "0.15.1", "mmcv": "2.0.1",
        "mmengine": "0.7.4", "mmdet": "3.0.0", "mmyolo": "0.6.0"}
    versions = {name: importlib.metadata.version(name) for name in pinned}
    require(sys.version_info[:3] == (3, 8, 20), "Python version drift")
    for name, expected in pinned.items():
        require(versions[name].split("+")[0] == expected, "Version drift: " + name)
    return {"python": sys.version, "executable": sys.executable,
        "versions": versions, "gpu": subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader"], text=True).strip().splitlines()}


def gpu_exclusive():
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    require(all(not p.strip() or p.strip() == str(os.getpid()) for p in active),
        "Concurrent GPU computation detected; contaminated timing is not accepted")


def seed_call(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accepted_scope(row, image_hash):
    failures = row.get("failures")
    zero_failures = (type(failures) is int and failures == 0) or failures == []
    return (row.get("dataset") == "coco" and row.get("split") == "val" and
        row.get("images") == 5000 and zero_failures and
        row.get("evaluated_image_ids_match_expected") is True and
        row.get("expected_image_ids_sha256") == image_hash and
        row.get("evaluated_image_ids_sha256") == image_hash)


def build_plan(args, root, output):
    from ..registry import Registry
    from ..data.coco import CocoIndex
    from .retention import build_retained_image_selection
    registry = Registry(root)
    protocol = registry.protocols[PROTOCOL]
    require("vcsf" not in registry.attacks, "Global VCSF registration is forbidden")
    from ..attacks.factory import ATTACK_TYPES
    require("vcsf" not in ATTACK_TYPES, "Global VCSF factory entry is forbidden")
    require(file_digest(root / "src/lgp/attacks/vcsf_final_candidate.py") ==
        protocol["frozen_vcsf_sha256"], "Frozen VCSF implementation drift")
    canonical = args.current_main.resolve()
    records_path = canonical / "records.json"
    acceptance_path = canonical / "canonical_acceptance.json"
    require(file_digest(records_path) == args.records_sha256, "Records hash mismatch")
    require(file_digest(acceptance_path) == args.acceptance_sha256,
        "Acceptance hash mismatch")
    acceptance = read_json(acceptance_path)
    require(acceptance["status"] == "accepted_canonical_projection",
        "Canonical source was not accepted")
    require(acceptance["records_sha256"] == args.records_sha256,
        "Acceptance does not bind the supplied records")
    reuse_reference = acceptance["legacy_reference"]
    reuse_path = Path(reuse_reference["reuse_preflight"])
    require(file_digest(reuse_path) == reuse_reference["reuse_preflight_sha256"],
        "Legacy equivalence acceptance changed")
    accepted_preflight = read_json(reuse_path)
    reuse = accepted_preflight["baseline_reuse"]
    require(reuse["status"] == "reusable", "Baseline reuse was not accepted")
    require(file_digest(root / "src/lgp/adapters/openmmlab.py") ==
        reuse["adapter_identity"]["current_sha256"], "Accepted adapter equivalence changed")
    selection = build_retained_image_selection(registry, "coco", "val",
        int(protocol["retained_images"]), output / "selection.json")
    for key in ("all_image_ids_sha256", "retained_image_ids_sha256"):
        require(selection[key] == protocol[key], "Frozen selection mismatch: " + key)
    index = CocoIndex(registry.dataset("coco"), "val")
    require(file_digest(index.annotation_path) == accepted_preflight["data_and_checkpoints"]
        ["datasets"]["coco"]["annotation_sha256"], "Accepted COCO annotation changed")
    selected = selection["retained"]
    if args.max_images is not None:
        require(1 <= args.max_images <= len(selected), "Invalid diagnostic size")
        selected = selected[:args.max_images]
    image_map = {int(image["id"]): image for image in index.images}
    image_hashes = {str(item["image_id"]): file_digest(
        index.image_path(image_map[item["image_id"]])) for item in selected}
    records = read_json(records_path)
    groups = []
    for source in protocol["sources"]:
        for method in protocol["methods"]:
            rows = [r for r in records if r.get("source") == source and
                r.get("attack") == method and r.get("status") == "complete"]
            require(len(rows) == len(registry.target_ids()), "Incomplete accepted group")
            require({r["target"] for r in rows} == set(registry.target_ids()),
                "Accepted target panel mismatch")
            require(all(accepted_scope(r, protocol["all_image_ids_sha256"])
                for r in rows), "Accepted group scope mismatch")
            hashes = {r["parameters_sha256"] for r in rows}
            roots = {r["adversarial_run"] for r in rows}
            require(len(hashes) == len(roots) == 1, "Ambiguous accepted configuration")
            run_path = Path(next(iter(roots))) / "run.json"
            run = read_json(run_path)
            require(run["status"] == "complete", "Unfinished source payload")
            require(run["source"] == source and run["attack"] == method,
                "Payload identity mismatch")
            require(digest(run["parameters"]) == next(iter(hashes)), "Parameter drift")
            require(float(run["parameters"]["eps"]) == 4.0 / 255.0, "Epsilon drift")
            checkpoint = Path(run["checkpoint"])
            require(file_digest(checkpoint) == run["checkpoint_sha256"],
                "Source checkpoint mismatch")
            if method != "vcsf":
                require(registry.compatibility_status(method, source)["status"] ==
                    "native", "Unsupported cost group")
            implementation = ("src/lgp/attacks/vcsf_final_candidate.py" if method == "vcsf"
                else "src/" + ATTACK_TYPES[method][0].__module__.replace(".", "/") + ".py")
            origin = subprocess.check_output(["git", "show",
                run["code_commit"] + ":" + implementation], cwd=root)
            require(hashlib.sha256(origin).hexdigest() == file_digest(root / implementation),
                "Accepted attack implementation changed: " + method)
            groups.append({"source": source, "method": method,
                "parameters": run["parameters"], "parameters_sha256": next(iter(hashes)),
                "source_run": str(run_path), "source_run_sha256": file_digest(run_path),
                "checkpoint": str(checkpoint), "checkpoint_sha256": run["checkpoint_sha256"],
                "accepted_commit": run["code_commit"], "implementation": implementation,
                "implementation_sha256": file_digest(root / implementation)})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(not subprocess.check_output(["git", "diff", "HEAD", "--name-only"],
        cwd=root, text=True).strip(), "Tracked source tree is not frozen")
    return registry, index, {"schema_version": 1, "protocol": PROTOCOL,
        "purpose": "physical_cost_only_no_AP", "created_at": now(),
        "git_head": head, "environment": environment(), "definition": protocol,
        "source_snapshot": source_snapshot(root), "selected": selected,
        "image_sha256": image_hashes, "annotation_sha256": file_digest(index.annotation_path),
        "groups": groups, "expected_measured_calls": len(groups) * len(selected),
        "diagnostic_only": args.max_images is not None,
        "canonical_records_sha256": args.records_sha256,
        "canonical_acceptance_sha256": args.acceptance_sha256,
        "adapter_equivalence_preflight": str(reuse_path),
        "adapter_equivalence_preflight_sha256": file_digest(reuse_path),
        "device": args.device, "evaluation_calls": 0, "payload_files": 0}


def hooks_signature(model):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks), len(m._backward_hooks))
        for m in model.modules()]


def instantiate(method, adapter, parameters):
    from ..attacks.factory import build_attack
    from ..attacks.vcsf_final_candidate import VCSFFinalCandidate, VCSFFinalCandidateConfig
    attack = (VCSFFinalCandidate(adapter, VCSFFinalCandidateConfig.from_mapping(parameters))
        if method == "vcsf" else build_attack(method, adapter, parameters))
    require(digest(asdict(attack.config)) == digest(parameters), "Resolved parameter mismatch")
    return attack


def strict_checkpoint(model, path, source):
    import torch
    from ..modeling import finite_state_dict_audit
    raw = torch.load(str(path), map_location="cpu")
    state = raw.get("state_dict", raw)
    finite = finite_state_dict_audit(state)
    excluded = [key for key in state if key.startswith("roi_head.mask_head.") and
        source == "mask_rcnn_swin_t"]
    validate_mask_exclusions(excluded, source)
    filtered = {key: value for key, value in state.items() if key not in excluded}
    model.load_state_dict(filtered, strict=True)
    return {"strict": True, "excluded_mask_head_keys": excluded, "finite": finite}


def validate_mask_exclusions(excluded, source):
    expected = {"roi_head.mask_head.convs.{}.conv.{}".format(i, suffix)
        for i in range(4) for suffix in ("weight", "bias")}
    expected.update("roi_head.mask_head.{}.{}".format(layer, suffix)
        for layer in ("upsample", "conv_logits") for suffix in ("weight", "bias"))
    require(set(excluded) == (expected if source == "mask_rcnn_swin_t" else set()),
        "Unused checkpoint keys differ from the released bbox-only exception")


def summarize(rows, plan):
    expected = {(g["source"], g["method"], item["image_id"])
        for g in plan["groups"] for item in plan["selected"]}
    keys = [(r["source"], r["method"], r["image_id"]) for r in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected, "Cost matrix incomplete")
    require(all(r["status"] == "ok" and math.isfinite(r["wall_seconds"]) and
        r["wall_seconds"] > 0 and 0 <= r["linf_pixel"] <= 4.0001 for r in rows),
        "Invalid cost observation")
    result = []
    reference = plan["definition"]["reference_method"]
    lookup = {(r["source"], r["method"], r["image_id"]): r for r in rows}

    def complete_counts(values, field):
        counts = [row.get(field) for row in values]
        if all(count is None for count in counts):
            return None, None
        require(all(type(count) is int and count >= 0 for count in counts),
            "Incomplete or invalid recorded cost count: " + field)
        return statistics.mean(counts), sum(counts)

    for group in plan["groups"]:
        values = [r for r in rows if r["source"] == group["source"] and
            r["method"] == group["method"]]
        times = [r["wall_seconds"] for r in values]
        paired = [r["wall_seconds"] / lookup[(r["source"], reference,
            r["image_id"])]["wall_seconds"] for r in values]
        gradient_mean, gradient_total = complete_counts(
            values, "actual_logical_gradients_from_method_diagnostics")
        source_forward_mean, source_forward_total = complete_counts(
            values, "actual_source_row_forwards_from_method_diagnostics")
        auxiliary_forward_mean, auxiliary_forward_total = complete_counts(
            values, "actual_auxiliary_forward_from_method_diagnostics")
        result.append({"source": group["source"], "method": group["method"],
            "images": len(values), "mean_seconds": statistics.mean(times),
            "median_seconds": statistics.median(times),
            "total_gpu_occupied_hours": sum(times) / 3600.0,
            "mean_paired_ratio_to_vcsf": statistics.mean(paired),
            "median_paired_ratio_to_vcsf": statistics.median(paired),
            "maximum_peak_allocated_mib": max(r["peak_allocated_mib"] for r in values),
            "mean_recorded_logical_gradients": gradient_mean,
            "total_recorded_logical_gradients": gradient_total,
            "logical_gradient_count_sources": sorted({r.get("logical_gradient_count_source", "unavailable") for r in values}),
            "mean_method_reported_source_row_forwards": source_forward_mean,
            "total_method_reported_source_row_forwards": source_forward_total,
            "mean_method_reported_auxiliary_forwards": auxiliary_forward_mean,
            "total_method_reported_auxiliary_forwards": auxiliary_forward_total,
            "declared_logical_gradients": sorted(set(r["declared_logical_gradients"] for r in values)),
            "declared_auxiliary_forward": sorted(set(r["declared_auxiliary_forward"] for r in values), key=lambda x: (x is None, x or 0)),
            "declared_auxiliary_backward": sorted(set(r["declared_auxiliary_backward"] for r in values), key=lambda x: (x is None, x or 0))})
    return {"status": "complete_pending_independent_cost_audit" if not plan["diagnostic_only"] else
        "passed_cost_diagnostic_not_publishable", "formal_AP_eligible": False,
        "cost_report_eligible": False, "failed_records": 0,
        "cost_counter_summary_version": 1,
        "measured_calls": len(rows), "groups": result,
        "interpretation": "Observed cost ratios, not a proof of equal physical computation."}


def execute(registry, index, plan, output, attack_builder=None):
    import torch
    from ..adapters import OpenMMLabAdapter
    from ..modeling import load_detector
    from .attack import _load_bgr
    torch.cuda.set_device(plan["device"])
    gpu_exclusive()
    atomic_json(output / "runtime_backend.json", {"torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cpu_threads": torch.get_num_threads(),
        "rng_policy": "reset_python_numpy_torch_before_each_call_outside_timer",
        "input_policy": "raw_BGR_CPU_decode_then_GPU_preload_outside_timer"})
    definition = plan["definition"]
    methods = definition["methods"]
    image_map = {int(image["id"]): image for image in index.images}
    rows = []
    started = time.perf_counter()
    state = {"status": "running", "phase": "loading", "completed_calls": 0,
        "expected_calls": plan["expected_measured_calls"], "pid": os.getpid(),
        "interpreter": sys.executable, "failed_records": 0, "started_at": now()}

    def update(**changes):
        state.update(changes)
        state["updated_at"] = now()
        atomic_json(output / "execution_state.json", state)

    try:
        for source in definition["sources"]:
            group_map = {g["method"]: g for g in plan["groups"] if g["source"] == source}
            checkpoint = group_map[methods[0]]["checkpoint"]
            require(len({g["checkpoint_sha256"] for g in group_map.values()}) == 1,
                "Different source checkpoints across methods")
            update(phase="loading", current={"source": source})
            model, _, _ = load_detector(registry.model(source), registry.dataset("coco"),
                output / "models" / source, device=plan["device"], download=False,
                checkpoint_override=Path(checkpoint))
            atomic_json(output / (source + "_load_audit.json"), strict_checkpoint(model, checkpoint, source))
            builder = instantiate if attack_builder is None else attack_builder
            attacks = {m: builder(m, OpenMMLabAdapter(model), group_map[m]["parameters"])
                for m in methods}
            signature = hooks_signature(model)
            warmup = plan["selected"][:min(definition["warmup_images"], len(plan["selected"]))]
            for phase, selected in (("warmup", warmup), ("measuring", plan["selected"])):
                for position, item in enumerate(selected):
                    gpu_exclusive()
                    image_id = item["image_id"]
                    image_path = index.image_path(image_map[image_id])
                    require(file_digest(image_path) == plan["image_sha256"][str(image_id)],
                        "Clean image changed")
                    clean = _load_bgr(image_path).to(plan["device"])
                    box_list, label_list = index.instances(image_id)
                    boxes = torch.tensor(box_list, dtype=torch.float32, device=plan["device"]).reshape(-1, 4)
                    labels = torch.tensor(label_list, dtype=torch.long, device=plan["device"])
                    for ordinal, method in enumerate(ordered_methods(methods, position)):
                        update(phase=phase, current={"source": source, "method": method,
                            "image_id": image_id, "image_position": position, "method_position": ordinal})
                        attack = attacks[method]
                        require(hooks_signature(model) == signature, "Leaked model hook before call")
                        seed_call(definition["seed"] + item["position"])
                        model.zero_grad(set_to_none=True)
                        gpu_exclusive()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        base_allocated = torch.cuda.memory_allocated()
                        t0 = time.perf_counter()
                        result = attack(clean, boxes, labels, seed=definition["seed"] + item["position"])
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter() - t0
                        peak = torch.cuda.max_memory_allocated()
                        reserved = torch.cuda.max_memory_reserved()
                        gpu_exclusive()
                        require(hooks_signature(model) == signature, "Leaked model hook after call")
                        require(bool(torch.isfinite(result.adversarial_bgr).all()) and
                            bool(torch.isfinite(result.perturbation).all()), "Nonfinite attack output")
                        actual_linf = float((result.adversarial_bgr - clean).abs().max().item())
                        require(math.isfinite(actual_linf) and actual_linf <= 4.0001,
                            "L-infinity budget violation")
                        identity = [d for d in result.diagnostics
                            if d.get("empty_target_behavior") == "identity_output"]
                        source_row_forwards = (
                            attack.source_row_forwards_for_output(result)
                            if callable(getattr(attack, "source_row_forwards_for_output", None)) else None)
                        auxiliary_forwards = (
                            attack.auxiliary_forward_passes_for_output(result)
                            if callable(getattr(attack, "auxiliary_forward_passes_for_output", None)) else None)
                        risk_surfaces = [d["risk_surface"] for d in result.diagnostics
                            if d.get("phase") == "risk_initialization" and "risk_surface" in d]
                        row = {"status": "ok", "source": source, "method": method,
                            "image_id": image_id, "seed": definition["seed"] + item["position"],
                            "position": position, "ordinal": ordinal, "wall_seconds": elapsed,
                            "input_shape": list(clean.shape), "linf_pixel": actual_linf,
                            "gpu_exclusivity_checks": "before_and_after_call_no_foreign_pid",
                            "base_allocated_mib": base_allocated / 2 ** 20,
                            "peak_allocated_mib": peak / 2 ** 20,
                            "peak_reserved_mib_shared_allocator": reserved / 2 ** 20,
                            "declared_logical_gradients": int(attack.gradient_evaluations_per_image),
                            "declared_auxiliary_forward": getattr(attack, "auxiliary_forward_passes_per_image", None),
                            "declared_auxiliary_backward": getattr(attack, "auxiliary_backward_passes_per_image", None),
                            "actual_source_row_forwards_from_method_diagnostics": source_row_forwards,
                            "actual_auxiliary_forward_from_method_diagnostics": auxiliary_forwards,
                            "logical_gradient_count_source": "identity_output" if identity else (
                                "method_diagnostics" if callable(getattr(attack, 'gradient_evaluations_for_output', None))
                                else "declared_cap"),
                            "counter_scope": "method_reported_when_available_else_declared_cap_not_physical_kernel_accounting",
                            "identity_output": bool(identity),
                            "actual_logical_gradients_from_method_diagnostics": int(identity[0].get("actual_gradient_evaluations", 0))
                                if identity else int(
                                    attack.gradient_evaluations_for_output(result)
                                    if callable(getattr(attack, 'gradient_evaluations_for_output', None))
                                    else attack.gradient_evaluations_per_image),
                            "vcsf_risk_surfaces": risk_surfaces if method == "vcsf" else None,
                            "vcsf_reference_pairs_from_phase_diagnostics": sum(d.get("phase") == "feature"
                                for d in result.diagnostics) if method == "vcsf" else None,
                            "parameters_sha256": group_map[method]["parameters_sha256"], "timestamp": now()}
                        require(0 <= row["actual_logical_gradients_from_method_diagnostics"] <=
                                row["declared_logical_gradients"], "Invalid actual gradient accounting")
                        with (output / ("measurements.jsonl" if phase == "measuring" else "warmup.jsonl")).open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, allow_nan=False) + "\n")
                            handle.flush()
                        if phase == "measuring":
                            rows.append(row)
                            update(completed_calls=len(rows))
                        del result
                    if position % 10 == 0:
                        print("[{}] {} {}/{} calls={}/{}".format(phase, source,
                            position + 1, len(selected), len(rows), plan["expected_measured_calls"]), flush=True)
                    del clean, boxes, labels
            del attacks, attack, model
            gc.collect()
            torch.cuda.empty_cache()
        require(source_snapshot(project_root()) == plan["source_snapshot"], "Source tree changed during run")
        reread = [json.loads(line) for line in (output / "measurements.jsonl").read_text().splitlines()]
        require(reread == rows, "Observation round-trip mismatch")
        summary = summarize(reread, plan)
        summary["total_elapsed_seconds_including_load_warmup_validation"] = time.perf_counter() - started
        atomic_json(output / "summary.json", summary)
        with (output / "cost_by_source_method.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["groups"][0]))
            writer.writeheader()
            writer.writerows(summary["groups"])
        state.update(status="complete", phase="complete", current=None, reason=None,
            updated_at=now(), summary_sha256=file_digest(output / "summary.json"))
        atomic_json(output / "final_state.json", state)
        manifest = {str(p.relative_to(output)): file_digest(p) for p in output.rglob("*")
            if p.is_file() and p.name != "execution_state.json"}
        atomic_json(output / "artifact_manifest.json", manifest)
        (output / "artifact_manifest.sha256").write_text(
            file_digest(output / "artifact_manifest.json") + "  artifact_manifest.json\n")
        atomic_json(output / "execution_state.json", state)
    except BaseException as exc:
        error = traceback.format_exc()
        (output / "traceback.txt").write_text(error, encoding="utf-8")
        update(status="failed", phase="failed", failed_records=1,
            reason="{}: {}".format(type(exc).__name__, exc))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-main", type=Path, required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--acceptance-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    environment()
    root = project_root()
    output = args.output or root / "outputs/diagnostics" / PROTOCOL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    try:
        registry, index, plan = build_plan(args, root, output)
    except BaseException as exc:
        atomic_json(output / "execution_state.json", {"status": "preflight_failed",
            "model_calls": 0, "reason": "{}: {}".format(type(exc).__name__, exc)})
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    atomic_json(output / "plan.json", plan)
    print("[PLAN] groups={} measured_calls={} diagnostic={} output={}".format(
        len(plan["groups"]), plan["expected_measured_calls"], plan["diagnostic_only"], output), flush=True)
    if args.plan_only:
        atomic_json(output / "execution_state.json", {"status": "planned", "model_calls": 0})
        return 0
    import fcntl
    lock_path = root / "outputs" / (PROTOCOL + ".lock")
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            atomic_json(output / "execution_state.json", {"status": "launch_blocked",
                "model_calls": 0, "reason": "Another calibration owns the exclusive lock"})
            raise
        execute(registry, index, plan, output)
    return 0
