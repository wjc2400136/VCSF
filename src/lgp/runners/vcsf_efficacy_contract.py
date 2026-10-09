"""Hash-bound execution contract for the already registered layered study."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from ..io import file_digest
from ..reporting.vcsf_analysis_inputs import compile_input_slots
from .vcsf_research_plan import PROTOCOL, canonical_hash, compile_plan
from .vcsf_structure_workers import balanced_lanes


EXECUTION_PROTOCOL = "vcsf_layered_efficacy_execution"
ENTRYPOINT = "experiments/vcsf_layered_efficacy.py"
ADDITIONAL_FILES = (
    ENTRYPOINT,
    "tools/prepare_vcsf_efficacy_execution.py",
    "tools/audit_vcsf_efficacy_execution.py",
    "tests/test_vcsf_efficacy_contract.py",
    "tests/test_vcsf_efficacy_runner.py",
)
ALLOWED_BASE_CHANGES = {
    "configs/experiments/protocols.yaml", "src/lgp/registry.py", "src/lgp/runners/attack.py",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_bound(path, expected_sha256):
    path = Path(path).resolve(strict=True)
    data = path.read_bytes()
    require(type(expected_sha256) is str and hashlib.sha256(data).hexdigest() == expected_sha256,
        "Bound evidence bytes changed: " + path.name)
    return json.loads(data.decode("utf-8"))


def child(root, relative):
    root = Path(root).resolve()
    raw = Path(relative)
    require(not raw.is_absolute() and ".." not in raw.parts, "Expected a contained relative path")
    path = root / raw
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
        "Output/source symlinks are not accepted")
    path.resolve().relative_to(root)
    return path


def runtime_snapshot(root):
    root = Path(root).resolve()
    names = {path.relative_to(root).as_posix()
        for directory, suffix in (("src/lgp", ".py"), ("configs", ".yaml"))
        for path in (root / directory).rglob("*") if path.is_file() and path.suffix == suffix}
    names.update(ADDITIONAL_FILES)
    return {name: file_digest(child(root, name)) for name in sorted(names)}


def verify_base_delta(original, current):
    changed = {name for name, digest in original.items()
        if name in current and current[name] != digest}
    missing = set(original) - set(current)
    # The prepared snapshot also contains historical tests and a structure entry.
    missing_runtime = {name for name in missing if name.startswith(("src/lgp/", "configs/"))}
    require(not missing_runtime and changed <= ALLOWED_BASE_CHANGES,
        "A scientific implementation or undeclared base file changed")
    return {"changed_base_files": sorted(changed),
        "added_files": sorted(set(current) - set(original))}


def verify_original_worktree(root, prepared, expected_head):
    root = Path(root).resolve(strict=True)
    actual_root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=root, text=True).strip()).resolve()
    require(root == actual_root, "The original evidence root is not its own worktree")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True).strip()
    require(head == expected_head and not dirty, "The original evidence worktree changed")
    require(all(file_digest(root / name) == digest for name, digest in prepared["file_sha256"].items()),
        "A prepared source binding changed in the original worktree")


def compile_execution_plan(registry, index_path, index_sha256, original_root):
    index = read_bound(index_path, index_sha256)
    require(index.get("status") == "partial_bound_analysis_inputs" and index.get("errors") == []
        and index.get("efficacy_runner_armed") is False
        and index.get("scientific_acceptance") is False, "Unexpected original input-index scope")
    prepared = read_bound(index["plan_file"], index["plan_sha256"])
    semantic = read_bound(index["semantic_file"], index["semantic_sha256"])
    slots = compile_input_slots(prepared, semantic, index["plan_sha256"])
    require(index.get("groups") == slots and index.get("source_sha256") == prepared["file_sha256"],
        "Input-index group/source binding differs")
    require(index.get("sources") == prepared["sources"] and index.get("targets") == prepared["targets"]
        and index.get("registered_cells") == sum(len(g["targets"]) for g in slots)
        and index.get("pending_new_groups") == semantic["new_groups"]
        and index.get("qualified_original_groups") == semantic["qualified_reuse_groups"],
        "Input-index matrix counts differ")
    expected_keys = [(g["group_id"], target) for g in slots for target in g["targets"]]
    cells = index.get("cells", [])
    require([(c["group_id"], c["target"]) for c in cells] == expected_keys,
        "Input-index cells are missing, duplicated or out of order")
    slot_by_id = {g["group_id"]: g for g in slots}
    for cell in cells:
        pending = slot_by_id[cell["group_id"]]["disposition"] == "new_registered_measurement"
        require(cell.get("bootstrap_status") == "NR", "Original input index was relabeled")
        if pending:
            require(cell.get("status") == "NR" and all(cell.get(field) is None
                for field in ("input_file", "input_sha256", "normalized_cell", "normalized_cell_sha256")),
                "A pending group must not inherit an observation")
        else:
            require(cell.get("status") == "metadata_bound_pending_archive_replay"
                and cell.get("normalized_cell") is not None, "A reused observation is missing")
            read_bound(Path(index_path).parent / cell["input_file"], cell["input_sha256"])
    current = compile_plan(registry)
    require(canonical_hash({k: v for k, v in current.items() if k != "file_sha256"})
        == canonical_hash({k: v for k, v in prepared.items() if k != "file_sha256"}),
        "The prepared scientific plan, analysis or selection policy changed")
    original_root = Path(original_root).resolve(strict=True)
    root = registry.root.resolve()
    require(root != original_root and root not in original_root.parents
        and original_root not in root.parents, "Use separate, non-nested execution/evidence worktrees")
    verify_original_worktree(original_root, prepared, semantic["current_git"])
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    require(head == semantic["current_git"], "The executor must retain the attested base commit")
    snapshot = runtime_snapshot(root)
    delta = verify_base_delta(prepared["file_sha256"], snapshot)
    definition = registry.protocols[EXECUTION_PROTOCOL]
    protocol = registry.protocols[PROTOCOL]
    require(definition["parent_protocol"] == PROTOCOL and protocol["execution_authorized"] is True,
        "Parent research execution is not authorized")
    pending_ids = [g["group_id"] for g in slots if g["disposition"] == "new_registered_measurement"]
    pending = [g for g in prepared["groups"] if g["group_id"] in pending_ids]
    require(len(pending) == semantic["new_groups"] and pending, "Empty or inconsistent pending matrix")
    balanced_lanes(pending, prepared["sources"], ["cuda:0", "cuda:1"])
    references = [Path(p) for p in semantic["input_sha256"] if Path(p).name == "reuse_acceptance.json"]
    require(len(references) == 1, "Reference provenance is not uniquely bound")
    reference = read_bound(references[0], semantic["input_sha256"][str(references[0])])
    require(reference.get("status") == "verified_reference_reuse_and_shared_provenance"
        and reference.get("errors") == [], "Original reference receipt is not accepted")
    return {
        "schema_version": 1, "status": "prepared_execution_pending_admission",
        "protocol": EXECUTION_PROTOCOL, "parent_protocol": PROTOCOL,
        "definition": definition, "definition_sha256": canonical_hash(definition),
        "base_commit": head, "runtime_sha256": snapshot,
        "runtime_snapshot_sha256": canonical_hash(snapshot), "source_delta": delta,
        "original_root": str(original_root), "input_index_file": str(Path(index_path).resolve()),
        "input_index_sha256": index_sha256, "prepared_plan_file": index["plan_file"],
        "prepared_plan_sha256": index["plan_sha256"], "semantic_file": index["semantic_file"],
        "semantic_sha256": index["semantic_sha256"], "prepared_plan": prepared,
        "groups": prepared["groups"], "input_slots": slots,
        "new_group_ids": pending_ids, "reused_group_ids": [
            g["group_id"] for g in slots if g["disposition"] != "new_registered_measurement"],
        "sources": prepared["sources"], "targets": prepared["targets"],
        "new_groups": len(pending), "new_images": sum(g["images"] for g in pending),
        "new_evaluations": sum(len(g["targets"]) for g in pending),
        "packages": index["packages"], "checkpoint_sha256": reference["target_checkpoint_sha256"],
        "annotation_sha256": reference["annotation_sha256"],
        "image_ids_sha256": reference["image_ids_sha256"],
        "analysis": prepared["analysis"], "analysis_sha256": canonical_hash(prepared["analysis"]),
        "selection_namespace": prepared["selection_namespace"], "independent_confirmation": False,
        "formal_metrics_eligible": False, "runner_armed": False,
        "payload_authorization": {
            "id": definition["payload_retention"]["authorization"],
            "scope": "new_execution_root_complete_groups_only",
            "explicit_owner_confirmation": True, "historical_roots_allowed": False,
            "failed_or_partial_groups_allowed": False,
            "retained_images": definition["payload_retention"]["retained_images"],
            "preserve_predictions_and_manifests": True,
        },
    }


def verify_execution_plan(registry, plan, *, check_original=True):
    require(plan.get("status") == "prepared_execution_pending_admission"
        and plan.get("protocol") == EXECUTION_PROTOCOL and plan.get("parent_protocol") == PROTOCOL
        and plan.get("runner_armed") is False and plan.get("independent_confirmation") is False,
        "Unexpected efficacy plan identity")
    require(plan.get("definition") == registry.protocols[EXECUTION_PROTOCOL]
        and plan.get("definition_sha256") == canonical_hash(plan["definition"]),
        "Execution/lifecycle declaration drifted")
    snapshot = runtime_snapshot(registry.root)
    require(snapshot == plan.get("runtime_sha256")
        and canonical_hash(snapshot) == plan.get("runtime_snapshot_sha256"),
        "The tested execution runtime changed")
    require(verify_base_delta(plan["prepared_plan"]["file_sha256"], snapshot) == plan["source_delta"],
        "The declared source delta changed")
    current = compile_plan(registry)
    require(canonical_hash({k: v for k, v in current.items() if k != "file_sha256"})
        == canonical_hash({k: v for k, v in plan["prepared_plan"].items() if k != "file_sha256"}),
        "Scientific plan or analysis policy changed")
    require(plan["groups"] == plan["prepared_plan"]["groups"]
        and plan["analysis"] == plan["prepared_plan"]["analysis"]
        and plan["analysis_sha256"] == canonical_hash(plan["analysis"]), "Embedded plan differs")
    index = read_bound(plan["input_index_file"], plan["input_index_sha256"])
    semantic = read_bound(plan["semantic_file"], plan["semantic_sha256"])
    prepared = read_bound(plan["prepared_plan_file"], plan["prepared_plan_sha256"])
    slots = compile_input_slots(prepared, semantic, plan["prepared_plan_sha256"])
    require(prepared == plan["prepared_plan"] and slots == plan["input_slots"] == index["groups"],
        "Prepared/reuse/index provenance changed")
    pending_ids = [g["group_id"] for g in slots if g["disposition"] == "new_registered_measurement"]
    reused_ids = [g["group_id"] for g in slots if g["disposition"] != "new_registered_measurement"]
    require(plan["new_group_ids"] == pending_ids and plan["reused_group_ids"] == reused_ids
        and plan["new_groups"] == len(pending_ids)
        and plan["new_images"] == sum(g["images"] for g in plan["groups"] if g["group_id"] in pending_ids)
        and plan["new_evaluations"] == sum(len(g["targets"]) for g in plan["groups"] if g["group_id"] in pending_ids),
        "The pending execution matrix changed")
    require(plan["sources"] == prepared["sources"] and plan["targets"] == prepared["targets"],
        "Canonical detector order changed")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=registry.root, text=True).strip()
    require(head == plan["base_commit"] == semantic["current_git"], "Execution base commit changed")
    if check_original:
        verify_original_worktree(plan["original_root"], prepared, plan["base_commit"])
    reconstructed = compile_execution_plan(registry, plan["input_index_file"],
        plan["input_index_sha256"], plan["original_root"])
    require(canonical_hash(reconstructed) == canonical_hash(plan),
        "The execution contract differs from its complete input-derived reconstruction")
    return plan


def execution_groups(plan, max_images):
    groups = [g for g in plan["groups"] if g["group_id"] in plan["new_group_ids"]]
    require(groups, "No new groups are scheduled")
    if max_images is None:
        return groups
    require(type(max_images) is int and 0 < max_images < groups[0]["images"],
        "A smoke limit must be positive and smaller than the full formal split")
    pair = (groups[0]["variant"], groups[0]["seed"])
    smoke = [g for g in groups if (g["variant"], g["seed"]) == pair]
    require([g["source"] for g in smoke] == plan["sources"], "Smoke must preserve a complete source pair")
    return smoke


def execution_metadata(plan, plan_sha256, group, max_images):
    return {
        "protocol": PROTOCOL, "execution_protocol": EXECUTION_PROTOCOL,
        "study": "vcsf_finalization_layered", "variant": group["variant"],
        "seed": group["seed"], "group_id": group["group_id"],
        "execution_plan_sha256": plan_sha256,
        "prepared_plan_sha256": plan["prepared_plan_sha256"],
        "runtime_snapshot_sha256": plan["runtime_snapshot_sha256"],
        "selection_namespace": plan["selection_namespace"], "independent_confirmation": False,
        "smoke_max_images": max_images, "formal_metrics_eligible": False,
    }


def verify_admission(registry, plan, plan_sha256, admission_path, admission_sha256, max_images):
    admission = read_bound(admission_path, admission_sha256)
    mode = "formal" if max_images is None else "smoke"
    require(admission.get("status") == "independently_verified_layered_execution"
        and admission.get("errors") == [] and admission.get("execution_plan_sha256") == plan_sha256
        and admission.get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"]
        and admission.get("analysis_sha256") == plan["analysis_sha256"]
        and admission.get("independent_confirmation") is False
        and mode in admission.get("allowed_modes", []), "No matching independent execution admission")
    require(admission.get("auditor_sha256") == file_digest(
        registry.root / "tools/audit_vcsf_efficacy_execution.py"), "Execution admission auditor changed")
    if mode == "formal":
        require(admission.get("all_launch_gates_passed") is True
            and admission.get("payload_authorization") == plan["payload_authorization"],
            "Formal launch or exact new-root retention authorization is missing")
    else:
        require(type(admission.get("smoke_max_images")) is int
            and max_images <= admission["smoke_max_images"],
            "Smoke execution exceeds its independent admission")
    for bound in admission.get("bound_evidence", []):
        read_bound(bound["file"], bound["sha256"])
    require(admission.get("bound_evidence"), "Admission has no supporting receipts")
    return admission


def verify_environment(plan):
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda"
        and sys.version_info[:3] == (3, 8, 20), "Use the pinned server ODA interpreter")
    actual = {name: importlib.metadata.version(name) for name in plan["packages"]}
    require(actual == plan["packages"], "The accepted evaluation environment changed")
    return actual
