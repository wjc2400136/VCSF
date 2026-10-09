"""Server-only one-image execution check of every registered layered variant."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

from ..attacks.vcsf_research_isolation import (
    ALIAS, CANDIDATE_ID, STUDY, _read_bound_json, _write_bound_json, research_execution_metadata,
    validate_cost_precondition,
)
from ..data.coco import CocoIndex
from ..io import atomic_json, file_digest
from ..modeling import checkpoint_path
from ..registry import Registry
from .cost_calibration import environment
from .vcsf_research_plan import PROTOCOL, canonical_hash, compile_plan, require


TEST_FILES = (
    "tests/test_vcsf_final_candidate.py",
    "tests/test_vcsf_research_isolated.py",
    "tests/test_vcsf_research_isolation.py",
    "tests/test_vcsf_layered_preflight.py",
    "tests/test_vcsf_structure_workers.py",
    "tests/test_vcsf_layered_contrasts.py",
    "tests/test_vcsf_image_bootstrap.py",
    "tests/test_vcsf_paired_statistics.py",
    "tests/test_vcsf_prediction_evidence.py",
    "tests/test_vcsf_coco_replay.py",
)


def structural_groups(plan):
    """Seed-42 covers every unique setting; five-seed AP is a later stage."""
    groups = [g for g in plan["groups"] if g["seed"] == 42]
    require(len(groups) == 2 * plan["totals_before_qualified_reuse"]["unique_variants"],
        "Every variant must have exactly two structure-preflight source groups")
    require(len({(g["variant"], g["source"]) for g in groups}) == len(groups),
        "Duplicate structural group")
    return groups


def verify_frozen_worktree(root, plan, expected_head=None):
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, text=True)
    require(not status.strip(), "Freeze a clean server worktree, including untracked files")
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root, text=True).split("\0"))
    require(set(plan["file_sha256"]) <= tracked,
        "Every source/config/test in the prepared snapshot must be Git-tracked")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(expected_head is None or head == expected_head,
        "Frozen Git HEAD changed during structure preflight")
    return head


def verify_group(run_dir, group, image_id, checkpoint_sha256, clean_path, clean_sha256, descriptor):
    """Verify serialized output as well as the declared logical schedule."""
    import numpy as np
    from PIL import Image, ImageOps

    run_data = (run_dir / "run.json").read_bytes()
    manifest_data = (run_dir / "manifest.jsonl").read_bytes()
    run = json.loads(run_data.decode("utf-8"))
    rows = [json.loads(line) for line in manifest_data.decode("utf-8").splitlines() if line.strip()]
    require(run.get("status") == "complete" and run.get("successful_images") == 1
        and run.get("failed_images") == 0 and len(rows) == 1, "Incomplete structure payload")
    row = rows[0]
    require(row.get("status") == "ok" and row["image_id"] == image_id
        and row["attack_seed"] == group["seed"], "Structure image/seed mismatch")
    require(run["source"] == group["source"] and run["attack"] == ALIAS
        and run["parameters_sha256"] == group["parameters_sha256"]
        and canonical_hash(run["parameters"]) == group["parameters_sha256"],
        "Structure identity drift")
    require(run["checkpoint_sha256"] == checkpoint_sha256, "Structure source weights changed")
    require(row["actual_gradient_evaluations"] == group["logical_gradients"]
        and run["actual_gradient_evaluations_total"] == group["logical_gradients"],
        "Incorrect logical gradient accounting")
    require(row["auxiliary_forward_passes"] == 0 and row["auxiliary_backward_passes"] == 0,
        "Unexpected explicit auxiliary calls")
    require(run["auxiliary_passes"]["forward_total"] == 0
        and run["auxiliary_passes"]["backward_total"] == 0,
        "Summary auxiliary counts disagree with the structure record")
    diagnostics = row["diagnostics"]
    phases = ["risk_initialization"] * group["risk_updates"] + ["feature"] * group["feature_updates"]
    require(len(phases) == group["logical_gradients"]
        and all(type(d.get("step")) is int for d in diagnostics)
        and [(d.get("step"), d.get("phase")) for d in diagnostics] == list(enumerate(phases)),
        "Diagnostic phase schedule mismatch")
    require(all(d["risk_direction_carryover_applied"] is False
        and d["uncertainty_gate_active"] is False for d in diagnostics),
        "Removed state was reintroduced")
    require(run.get("formal_AP_eligible") is False
        and canonical_hash(run.get("research_execution")) == canonical_hash(
            research_execution_metadata(descriptor, group)),
        "Structure output was mislabeled")
    require(hashlib.sha256(manifest_data).hexdigest() == run["manifest_sha256"],
        "Structure manifest changed")
    png = (run_dir / row["output_file"]).resolve()
    png.relative_to(run_dir.resolve())
    png_data, clean_data = png.read_bytes(), clean_path.read_bytes()
    require(hashlib.sha256(png_data).hexdigest() == row["output_sha256"], "Serialized image hash mismatch")
    require(Path(row["source_file"]).resolve() == clean_path.resolve()
        and hashlib.sha256(clean_data).hexdigest() == clean_sha256, "Clean input identity changed")
    with Image.open(BytesIO(clean_data)) as clean_image:
        clean = np.asarray(ImageOps.exif_transpose(clean_image).convert("RGB"), dtype=np.int16)
    with Image.open(BytesIO(png_data)) as adversarial_image:
        require(adversarial_image.format == "PNG", "Structure output is not lossless PNG")
        adversarial = np.asarray(adversarial_image.convert("RGB"), dtype=np.int16)
    require(clean.shape == adversarial.shape, "Serialized image shape changed")
    linf = int(np.abs(adversarial - clean).max())
    reported_linf = float(row["linf_pixel"])
    require(linf <= 4 and math.isfinite(reported_linf) and 0 <= reported_linf <= 4.0001,
        "Structure perturbation exceeds the declared radius")
    # Integer clean pixels and nearest-integer serialization differ by at most half a pixel.
    require(abs(reported_linf - linf) <= 0.5001,
        "Reported and serialized perturbation norms disagree beyond quantization tolerance")
    return {"group_id": group["group_id"], "source": group["source"],
        "variant": group["variant"], "seed": group["seed"], "image_id": image_id,
        "logical_gradients": group["logical_gradients"], "decoded_linf_pixel": linf,
        "manifest_sha256": run["manifest_sha256"], "run_sha256": hashlib.sha256(run_data).hexdigest(),
        "research_execution": run["research_execution"],
        "parameters_sha256": run["parameters_sha256"], "checkpoint_sha256": checkpoint_sha256,
        "png_sha256": row["output_sha256"], "clean_sha256": clean_sha256,
        "feature_surface_shapes": [d.get("feature_surface_shapes") for d in diagnostics
            if d["phase"] == "feature"]}


def verify_terminal_authority(registry, plan, descriptor, git_head, output, authority_sha256):
    require(set(authority_sha256) == {"plan.json", "environment.json", "provenance.json",
        "input_identity.json", "descriptor.json", "targeted_tests.log"},
        "Incomplete startup authority binding")
    for name, digest in authority_sha256.items():
        require(file_digest(output / name) == digest, "Startup evidence changed: " + name)
    require(canonical_hash(_read_bound_json(output / "descriptor.json",
        authority_sha256["descriptor.json"])) == canonical_hash(descriptor),
        "Structure descriptor differs from the executed authority")
    require(canonical_hash(_read_bound_json(registry.root / descriptor["plan"],
        descriptor["plan_sha256"])) == canonical_hash(plan), "Copied structure plan changed")
    validate_cost_precondition(registry, descriptor["cost_acceptance"],
        descriptor["cost_acceptance_sha256"])
    require(canonical_hash(compile_plan(registry)) == canonical_hash(plan),
        "Source/config snapshot changed during structure preflight")
    verify_frozen_worktree(registry.root, plan, expected_head=git_head)


def seal_structure_receipt(output, receipt):
    required = {"plan.json", "descriptor.json", "provenance.json", "input_identity.json",
        "environment.json", "execution_state.json", "execution_assignment.json",
        "execution_timing.json", "verified_groups.json", "targeted_tests.log"}
    excluded = {"structure_receipt.json", "artifact_manifest.json", "artifact_manifest.sha256"}
    paths = {p.relative_to(output).as_posix(): p for p in output.rglob("*") if p.is_file()}
    require(not any(p.is_symlink() for p in output.rglob("*")), "Structure artifacts cannot be symlinks")
    require(required <= set(paths), "Missing structural authority files")
    require(not (excluded & set(paths)), "Refusing to reseal an existing structure receipt")
    manifest = {"schema_version": 1, "scope": "structure_only_not_AP",
        "files": {name: file_digest(path) for name, path in sorted(paths.items())}}
    atomic_json(output / "artifact_manifest.json", manifest)
    digest = file_digest(output / "artifact_manifest.json")
    (output / "artifact_manifest.sha256").write_text(digest + "  artifact_manifest.json\n", encoding="ascii")
    # Validate before publishing a receipt; no authority file is rewritten afterward.
    for name, expected in manifest["files"].items():
        require(file_digest(output / name) == expected, "Structure evidence changed while sealing: " + name)
    atomic_json(output / "structure_receipt.json", dict(receipt, artifact_manifest_sha256=digest))
    verify_structure_receipt(output, file_digest(output / "structure_receipt.json"))


def verify_structure_receipt(output, expected_sha256):
    receipt = _read_bound_json(output / "structure_receipt.json", expected_sha256)
    manifest = _read_bound_json(output / "artifact_manifest.json", receipt["artifact_manifest_sha256"])
    require(manifest.get("schema_version") == 1 and manifest.get("scope") == "structure_only_not_AP",
        "Structure manifest scope mismatch")
    require((output / "artifact_manifest.sha256").read_text(encoding="ascii") ==
        receipt["artifact_manifest_sha256"] + "  artifact_manifest.json\n", "Structure sidecar mismatch")
    excluded = {"structure_receipt.json", "artifact_manifest.json", "artifact_manifest.sha256"}
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} - excluded
    require(actual == set(manifest["files"]), "Structure artifact membership changed")
    for name, expected in manifest["files"].items():
        path = (output / name).resolve()
        path.relative_to(output.resolve())
        require(file_digest(path) == expected, "Structure artifact hash mismatch: " + name)
    descriptor = _read_bound_json(output / "descriptor.json", manifest["files"]["descriptor.json"])
    state = _read_bound_json(output / "execution_state.json", manifest["files"]["execution_state.json"])
    provenance = _read_bound_json(output / "provenance.json", manifest["files"]["provenance.json"])
    groups = _read_bound_json(output / "verified_groups.json", manifest["files"]["verified_groups.json"])
    assignment = _read_bound_json(output / "execution_assignment.json",
        manifest["files"]["execution_assignment.json"])
    require(receipt["plan_sha256"] == manifest["files"]["plan.json"] == descriptor["plan_sha256"]
        and receipt["cost_acceptance_sha256"] == descriptor["cost_acceptance_sha256"]
            == provenance["cost_acceptance_sha256"],
        "Structure receipt authority mismatch")
    for field, name in {"assignment_sha256": "execution_assignment.json",
        "timing_sha256": "execution_timing.json", "verified_groups_sha256": "verified_groups.json",
        "test_log_sha256": "targeted_tests.log"}.items():
        require(receipt[field] == manifest["files"][name], "Structure receipt hash binding mismatch")
    count = receipt["verified_groups"]
    require(type(count) is int and count > 0
        and len(groups) == count == state["expected_groups"] == state["completed_groups"]
            == receipt["model_attack_calls"]
        and len(assignment["workers"]) == receipt["worker_count"] == len(receipt["devices"])
        and receipt["AP_evaluations"] == 0
        and receipt["status"] == "structure_only_complete_pending_independent_review",
        "Structure receipt count or scope mismatch")
    require(state["status"] == "complete" and state["failed_records"] == 0
        and state["formal_AP_eligible"] is False and state["efficacy_runner_armed"] is False
        and receipt["formal_AP_eligible"] is False and receipt["efficacy_runner_armed"] is False,
        "Structure terminal state mismatch")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="No tests or model calls.")
    parser.add_argument("--cost-acceptance", type=Path, help="Independent accepted cost receipt.")
    parser.add_argument("--cost-acceptance-sha256", help="Expected immutable receipt SHA-256.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", help="Comma-separated physical GPUs; default uses the registered pair.")
    parser.add_argument("--worker-request", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-request-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda" and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        "All layered tests and model execution must use the pinned server ODA interpreter")
    if args.worker_request is not None:
        from .vcsf_structure_workers import run_worker
        require(args.worker_request_sha256 is not None, "Worker request hash is required")
        return run_worker(args.worker_request, args.worker_request_sha256)
    require(args.worker_request_sha256 is None, "Worker hash requires a worker request")
    registry = Registry()
    plan = compile_plan(registry)
    groups = structural_groups(plan)
    from .vcsf_structure_workers import balanced_lanes, run_structure_workers, verify_selected_devices
    protocol = registry.protocols[PROTOCOL]
    devices = list(protocol["structure_devices"]) if args.devices is None else args.devices.split(",")
    require(all(device in protocol["structure_devices"] for device in devices),
        "Choose one or both registered structure devices")
    lanes = balanced_lanes(groups, protocol["sources"], devices)
    if args.plan_only:
        print(json.dumps({"status": "plan_only", "source_groups": len(groups),
            "images_per_group": 1, "AP_evaluations": 0, "efficacy_runner_armed": False,
            "devices": devices, "groups_per_worker": [len(lane) for lane in lanes]}, indent=2))
        return 0
    require(args.cost_acceptance is not None and args.cost_acceptance_sha256 is not None,
        "GPU structure preflight needs the completed independent cost receipt and its hash")
    receipt = validate_cost_precondition(registry, args.cost_acceptance,
        args.cost_acceptance_sha256)
    root = registry.root.resolve()
    runtime = environment()
    verify_selected_devices(devices)
    git_head = verify_frozen_worktree(root, plan)
    output = (args.output or root / "outputs/diagnostics/vcsf_layered_structure_preflight" /
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")).resolve()
    output.relative_to(root / "outputs/diagnostics")
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "running", "phase": "tests", "completed_groups": 0,
        "expected_groups": len(groups), "failed_records": 0, "current": None,
        "formal_AP_eligible": False, "efficacy_runner_armed": False}
    atomic_json(output / "execution_state.json", state)
    authority_sha256 = {"plan.json": _write_bound_json(output / "plan.json", plan),
        "environment.json": _write_bound_json(output / "environment.json", runtime)}
    authority_sha256["provenance.json"] = _write_bound_json(output / "provenance.json", {
        "git_head": git_head,
        "cost_acceptance": str(args.cost_acceptance.resolve()),
        "cost_acceptance_sha256": args.cost_acceptance_sha256,
        "test_sha256": {name: file_digest(root / name) for name in TEST_FILES}})
    verified = []
    try:
        with (output / "targeted_tests.log").open("w", encoding="utf-8") as handle:
            tests = subprocess.run([sys.executable, "-m", "pytest", "-q", *TEST_FILES],
                cwd=root, stdout=handle, stderr=subprocess.STDOUT)
        require(tests.returncode == 0, "Targeted server tests failed; inspect targeted_tests.log")
        authority_sha256["targeted_tests.log"] = file_digest(output / "targeted_tests.log")
        index = CocoIndex(registry.dataset("coco"), "val")
        ids = [int(image["id"]) for image in index.images]
        require(len(ids) == 5000 and canonical_hash(ids) == registry.protocols[PROTOCOL][
            "all_image_ids_sha256"], "Full-val image IDs do not match the frozen protocol")
        cost_plan = _read_bound_json(Path(receipt["run"]) / "plan.json", receipt["plan_sha256"])
        require(file_digest(index.annotation_path) == cost_plan["annotation_sha256"],
            "COCO annotation changed since accepted cost execution")
        clean_path = index.image_path(index.images[0])
        clean_sha256 = file_digest(clean_path)
        authority_sha256["input_identity.json"] = _write_bound_json(output / "input_identity.json", {"image_id": ids[0],
            "image_sha256": clean_sha256, "annotation_sha256": file_digest(index.annotation_path),
            "all_image_ids_sha256": canonical_hash(ids)})
        expected_checkpoints = {}
        for source in registry.protocols[PROTOCOL]["sources"]:
            hashes = {g["checkpoint_sha256"] for g in cost_plan["groups"] if g["source"] == source}
            require(len(hashes) == 1, "Accepted cost source weights are inconsistent")
            path = checkpoint_path(registry.model(source), registry.dataset("coco"))
            require(path.is_file() and file_digest(path) in hashes, "Research source weight drift")
            expected_checkpoints[source] = next(iter(hashes))
        descriptor = {"candidate_id": CANDIDATE_ID, "execution_alias": ALIAS,
            "mode": "structure_preflight", "plan": (output / "plan.json").relative_to(root).as_posix(),
            "plan_sha256": authority_sha256["plan.json"],
            "cost_acceptance": str(args.cost_acceptance.resolve()),
            "cost_acceptance_sha256": args.cost_acceptance_sha256}
        authority_sha256["descriptor.json"] = _write_bound_json(output / "descriptor.json", descriptor)
        verified = run_structure_workers(registry, plan, groups, output, descriptor,
            {"image_id": ids[0], "clean_path": str(clean_path), "clean_sha256": clean_sha256,
                "checkpoint_sha256": expected_checkpoints}, git_head, state, devices)
        for group, result in zip(groups, verified):
            checked = verify_group(output / "groups" / str(group["group_id"]), group,
                ids[0], expected_checkpoints[group["source"]], clean_path, clean_sha256, descriptor)
            require(all(result.get(key) == value for key, value in checked.items()),
                "Worker result differs from coordinator payload verification")
        verify_selected_devices(devices)
        verify_terminal_authority(registry, plan, descriptor, git_head, output, authority_sha256)
        state.update(status="complete", phase="structure_checks_complete", current=None)
        atomic_json(output / "execution_state.json", state)
        seal_structure_receipt(output, {
            "status": "structure_only_complete_pending_independent_review",
            "formal_AP_eligible": False, "efficacy_runner_armed": False,
            "errors": [], "verified_groups": len(verified), "model_attack_calls": len(verified),
            "devices": devices, "worker_count": len(devices),
            "AP_evaluations": 0, "plan_sha256": descriptor["plan_sha256"],
            "cost_acceptance_sha256": descriptor["cost_acceptance_sha256"],
            "assignment_sha256": file_digest(output / "execution_assignment.json"),
            "timing_sha256": file_digest(output / "execution_timing.json"),
            "verified_groups_sha256": file_digest(output / "verified_groups.json"),
            "test_log_sha256": file_digest(output / "targeted_tests.log"),
            "remaining_gates": ["independent_structure_review", "six_source_fixed_state_reference",
                "all_target_checkpoint_and_evaluator_provenance", "reuse_adjudication",
                "analysis_policy_lock", "payload_lifecycle_and_disk", "stage_runner_acceptance"],
        })
        print("[STRUCTURE ONLY] No AP, no full-val efficacy run, no registry switch.")
        return 0
    except Exception as exc:
        state.update(status="failed", reason="{}: {}".format(type(exc).__name__, exc), failed_records=1)
        atomic_json(output / "execution_state.json", state)
        (output / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
