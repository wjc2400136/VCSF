"""Fail-closed bridge for the unregistered layered-study structure preflight.

This module does not authorize efficacy execution. A prepared plan is a
configuration contract, not an accepted or armed experiment.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from ..io import atomic_json, file_digest
from ..registry import AttackSpec, Registry
from .factory import ATTACK_TYPES
from .vcsf_research_isolated import VCSFResearchCandidate, VCSFResearchConfig


ALIAS = "vcsf_research_isolated"
CANDIDATE_ID = "vcsf_layered_research"
STUDY = "vcsf_finalization_layered"


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _read_bound_json(path, expected_digest):
    path = Path(path).resolve()
    _require(path.is_file(), "Missing evidence file: {}".format(path))
    data = path.read_bytes()
    _require(hashlib.sha256(data).hexdigest() == expected_digest,
        "Evidence hash mismatch: {}".format(path))
    return json.loads(data.decode("utf-8"))


def _write_bound_json(path, value):
    from ..runners.vcsf_research_plan import canonical_hash

    atomic_json(path, value)
    data = Path(path).read_bytes()
    _require(canonical_hash(json.loads(data.decode("utf-8"))) == canonical_hash(value),
        "Written evidence differs from its authorized value: {}".format(path))
    return hashlib.sha256(data).hexdigest()


def research_execution_metadata(descriptor, group):
    from ..runners.vcsf_research_plan import PROTOCOL

    return {
        "candidate_id": CANDIDATE_ID, "protocol": PROTOCOL, "study": STUDY,
        "variant": group["variant"], "configuration_seed_group_id": group["group_id"],
        "prepared_plan_sha256": descriptor["plan_sha256"],
        "cost_acceptance_sha256": descriptor["cost_acceptance_sha256"],
        "fidelity_status": "isolated_structure_diagnostic_not_efficacy",
        "fidelity_basis": "registered_plan_and_independent_cost_bound_preflight",
        "semantic_contract": "retrospective_layered_structure_only",
        "global_registry_mutated": False, "global_factory_mutated": False,
        "formal_AP_eligible": False, "independent_confirmation": False,
    }


def _require_separate_worktrees(research_root, cost_run):
    research_root, cost_run = Path(research_root).resolve(), Path(cost_run).resolve()
    roots = [Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=path, text=True).strip()).resolve()
        for path in (research_root, cost_run)]
    _require(roots[0] == research_root, "Research root must be its own Git worktree")
    cost_worktree = roots[1]
    _require(cost_worktree in cost_run.parents,
        "Accepted cost run must be inside its recorded worktree")
    _require(research_root != cost_worktree
        and research_root not in cost_worktree.parents
        and cost_worktree not in research_root.parents,
        "Use separate non-nested worktrees; the accepted cost worktree stays immutable")


def validate_cost_precondition(registry, receipt_path, receipt_sha256):
    """Recheck the accepted cost root's bindings, without rerunning measurements."""
    receipt = _read_bound_json(receipt_path, receipt_sha256)
    _require(receipt.get("status") == "independently_verified_cost"
        and receipt.get("errors") == []
        and receipt.get("cost_report_eligible") is True
        and receipt.get("formal_AP_eligible") is False,
        "An independent cost acceptance is required before GPU preflight")
    _require(receipt.get("verified_measured_calls") == 10000
        and len(receipt.get("recomputed_groups", [])) == 20,
        "Cost acceptance does not cover the complete Common-2 matrix")
    root = Path(receipt["run"]).resolve()
    _require_separate_worktrees(registry.root, root)
    bound_files = {
        "plan.json": "plan_sha256", "summary.json": "summary_sha256",
        "measurements.jsonl": "measurements_sha256",
        "execution_state.json": "execution_state_sha256",
        "final_state.json": "final_state_sha256",
        "artifact_manifest.json": "artifact_manifest_sha256",
    }
    for name, field in bound_files.items():
        if name in ("plan.json", "execution_state.json"):
            continue
        path = root / name
        _require(path.is_file() and file_digest(path) == receipt.get(field),
            "Accepted cost evidence changed: " + name)
    state = _read_bound_json(root / "execution_state.json", receipt.get("execution_state_sha256"))
    _require(state.get("status") == "complete", "Cost measurement is not complete")
    plan = _read_bound_json(root / "plan.json", receipt.get("plan_sha256"))
    _require(plan.get("protocol") == "vcsf_common2_cost_calibration"
        and plan.get("definition") == registry.protocols["vcsf_common2_cost_calibration"],
        "Cost receipt belongs to a different registered protocol")
    implementation = "src/lgp/attacks/vcsf_final_candidate.py"
    frozen_sha256 = file_digest(registry.root / implementation)
    _require(plan["definition"].get("frozen_vcsf_sha256") == frozen_sha256
        and registry.protocols["vcsf_fullval_retrospective_ablation"].get(
            "frozen_vcsf_sha256") == frozen_sha256
        and plan.get("source_snapshot", {}).get(implementation) == frozen_sha256,
        "Accepted cost measurement is not bound to the frozen research anchor")
    sources = plan["definition"]["sources"]
    vcsf_groups = [group for group in plan.get("groups", []) if group.get("method") == "vcsf"]
    _require(sources == registry.protocols["vcsf_fullval_retrospective_ablation"]["sources"]
        and len(vcsf_groups) == len(sources)
        and {group.get("source") for group in vcsf_groups} == set(sources)
        and all(group.get("implementation") == implementation
            and group.get("implementation_sha256") == frozen_sha256 for group in vcsf_groups),
        "Accepted VCSF cost groups do not match the frozen anchor and research sources")
    _require(receipt.get("auditor_sha256") == file_digest(
        registry.root / "tools/audit_cost_calibration.py"), "Cost auditor identity changed")
    return receipt


def resolve_research_preflight(
    registry: Registry,
    descriptor: Mapping[str, Any],
    *,
    dataset_id: str,
    split: str,
    source_id: str,
    attack_id: str,
    seed: int,
    max_images: Any,
    parameter_overrides: Mapping[str, Any],
    run_metadata: Mapping[str, Any],
    budget_profile: str,
    image_ids: Any = None,
    seed_offsets: Any = None,
    input_transform: Any = None,
) -> AttackSpec:
    # Import lazily: the planner knows the configuration class, not this bridge.
    from ..runners.vcsf_research_plan import PROTOCOL, canonical_hash, compile_plan

    _require(descriptor.get("candidate_id") == CANDIDATE_ID
        and descriptor.get("execution_alias") == ALIAS and attack_id == ALIAS,
        "Unsupported research execution descriptor")
    _require(ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        "Research must not enter the global method registry or factory")
    _require(descriptor.get("mode") == "structure_preflight",
        "Layered efficacy execution is not armed; only structure_preflight is supported")
    _require(type(max_images) is int and max_images == 1,
        "Research structure preflight requires exactly one image, not an AP run")
    _require(dataset_id == "coco" and split == "val"
        and image_ids is None and seed_offsets is None and input_transform is None,
        "Structure preflight must use the canonical first COCO-val image and seed offset")
    _require(run_metadata.get("protocol") == PROTOCOL
        and run_metadata.get("study") == STUDY,
        "Research job is not bound to the registered layered protocol")
    protocol = registry.protocols[PROTOCOL]
    _require(budget_profile == protocol["budget_profile"], "Research cost regime changed")
    raw_path = Path(str(descriptor.get("plan", "")))
    _require(not raw_path.is_absolute(), "The prepared plan must use a project-relative path")
    path = (registry.root / raw_path).resolve()
    path.relative_to(registry.root.resolve())
    plan = _read_bound_json(path, descriptor.get("plan_sha256"))
    _require(canonical_hash(plan) == canonical_hash(compile_plan(registry)),
        "Prepared research plan or its implementation snapshot drifted")
    variant = run_metadata.get("variant")
    matches = [g for g in plan["groups"] if g["variant"] == variant
        and g["seed"] == seed and g["source"] == source_id]
    _require(type(seed) is int and len(matches) == 1,
        "Unscheduled research variant/source/seed")
    group = matches[0]
    # Defaults or partial overrides could silently select another variant.
    _require(canonical_hash(dict(parameter_overrides)) == group["parameters_sha256"],
        "Research parameters must exactly match the complete resolved group")
    config = VCSFResearchConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    _require(canonical_hash(asdict(config)) == group["parameters_sha256"],
        "Research configuration normalization changed the frozen parameters")
    validate_cost_precondition(registry, descriptor["cost_acceptance"],
        descriptor["cost_acceptance_sha256"])
    return AttackSpec(id=ALIAS, display_name="VCSF research: " + variant,
        executor="native", parameters=dict(group["parameters"]),
        metadata=research_execution_metadata(descriptor, group))


def build_research_preflight(adapter, parameters):
    return VCSFResearchCandidate(adapter,
        VCSFResearchConfig.from_mapping(dict(parameters)))
