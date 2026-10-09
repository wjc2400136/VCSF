"""Full adaptive permission checks, separate from frozen numerical method code."""
from pathlib import Path
from copy import deepcopy
import shutil

from ..io import file_digest
from .vcsf_oblivious_admission import bound_json, isolate_output
from .vcsf_final_training_state import source_identity


READINESS_SHA256 = "6f6437d101f1d3d5ae0f4cd8e88ef6a92ea33d45c9d63df8d6489783ab1fd371"
BRIDGE_FILES = {
    "src/lgp/runners/vcsf_adaptive_admission.py",
    "src/lgp/runners/vcsf_final_adaptive.py",
    "src/lgp/runners/vcsf_final_adaptive_group.py",
    "src/lgp/attacks/vcsf_final_adaptive_isolation.py",
    "experiments/vcsf_final_adaptive.py",
}


def scientific_plan(plan):
    value = deepcopy(plan)
    value.pop("devices", None)
    for name in ["generation_groups", "generated_evaluations", "identity_reuse", "canonical_jobs"]:
        for row in value.get(name, []):
            if isinstance(row, dict):
                row.pop("device", None)
    return value


def output_free_bytes(output):
    path = Path(output).resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def verify_formal_permit(registry, permit, *, recheck_evidence=False):
    """Requires a prospective externally pinned permit from the caller."""
    if (permit.get("status") != "approved_adaptive_formal"
            or permit.get("max_images", "missing") is not None
            or permit.get("protocol") != "vcsf_final_adaptive_refresh"):
        raise ValueError("Formal adaptive permission is missing or scope-limited")
    from .vcsf_final_adaptive_contract import build_plan
    plan = build_plan(registry, permit["devices"])
    if (permit.get("status") != "approved_adaptive_formal"
            or permit.get("max_images") is not None
            or permit.get("group_indices") != list(range(20))
            or any(type(index) is not int for index in permit["group_indices"])
            or permit.get("plan") != plan
            or len(plan["generated_evaluations"]) != 320
            or len(plan["identity_reuse"]) != 32
            or any(group["images"] != 500 for group in plan["generation_groups"])):
        raise ValueError("Formal adaptive permission does not cover the exact full matrix")
    ref = permit["readiness"]
    if ref["sha256"] != READINESS_SHA256:
        raise ValueError("Unqualified adaptive full-scope readiness receipt")
    ready = bound_json(ref["file"], ref["sha256"])
    if (ready["status"] != "adaptive_full_scope_prerequisites_bound"
            or ready["plan"] != build_plan(registry, ready["plan"]["devices"])
            or scientific_plan(ready["plan"]) != scientific_plan(plan)):
        raise ValueError("Adaptive prerequisite scope differs")
    for name in ["prepared", "assets", "checkpoints", "checkpoint_bindings", "identity_reuse"]:
        if permit[name] != ready[name]:
            raise ValueError("Formal adaptive prerequisite binding changed: " + name)
    diagnostic_ref = ready["diagnostic_receipt"]
    diagnostic = bound_json(diagnostic_ref["file"], diagnostic_ref["sha256"])
    diagnostic_permits = [path for path, digest in diagnostic["evidence"].items()
                          if digest == diagnostic["permit_sha256"]]
    if len(diagnostic_permits) != 1:
        raise ValueError("Diagnostic permit evidence is ambiguous")
    original = bound_json(diagnostic_permits[0], diagnostic["permit_sha256"])
    current = source_identity(registry.root)
    if current != permit["source_identity"]:
        raise ValueError("Formal adaptive source differs from prospective permission")
    before = original["source_identity"]
    changes = {name: dict(before=before.get(name), after=current.get(name))
        for name in set(before) | set(current) if before.get(name) != current.get(name)}
    if changes != permit["reviewed_source_changes"] or not set(changes).issubset(BRIDGE_FILES):
        raise ValueError("Formal bridge modifies unqualified numerical or shared source")
    audit_ref = permit["acceptance_tool"]
    if file_digest(Path(audit_ref["file"])) != audit_ref["sha256"]:
        raise ValueError("Prospective independent acceptance tool changed")
    protected = [Path(ref["file"]).parent, Path(diagnostic_ref["file"]).parent,
                 Path(ready["identity_receipt"]["file"]).parent,
                 Path(diagnostic_permits[0]).parent, Path(audit_ref["file"]).parent,
                 Path(original["output"])]
    isolate_output(permit["output"], protected)
    storage = ready["storage"]
    if (not storage["estimate_fits"] or permit["storage"] != storage
            or storage["requires_runtime_free_space_guard"] is not True):
        raise ValueError("Formal adaptive storage plan is missing or changed")
    if recheck_evidence:
        evidence = dict(ready["evidence"])
        evidence.update(diagnostic["evidence"])
        for name, digest in evidence.items():
            if file_digest(Path(name)) != digest:
                raise ValueError("Qualified adaptive prerequisite evidence changed")
        if output_free_bytes(permit["output"]) < storage["estimated_required_bytes"]:
            raise ValueError("Current free space no longer covers the formal storage estimate")
    return ready


def require_group_space(root, permit):
    """Stop before the next group if two concurrent groups threaten the reserve."""
    if permit["max_images"] is not None:
        return
    storage = permit["storage"]
    concurrent = len(permit["devices"])
    group_allowance = (storage["image_allowance_bytes"] + storage["prediction_allowance_bytes"]) // 20
    required = storage["reserve_bytes"] + concurrent * group_allowance
    if output_free_bytes(permit["output"]) < required:
        raise RuntimeError("Insufficient free space for the next owned groups; retain all existing evidence")
