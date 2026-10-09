from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import yaml

from ..registry import AttackSpec, Registry
from .factory import ATTACK_TYPES
from .svfta_cross_stage_claim_confirmation import (
    SVFTACrossStageClaimConfirmation,
    SVFTACrossStageClaimConfirmationConfig,
)
from .vcsf_a23_isolated import A23_PARAMETERS_SHA256, VCSFA23Candidate
from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
from .vcsf_final_candidate import (
    VCSFFinalCandidate,
    VCSFFinalCandidateConfig,
)


VCSF_CANDIDATE_ID = "vcsf_final_candidate"
VCSF_EXECUTION_MODES = ("final_candidate", "final_exact_ablation", "selected_a23")
A23_IMPLEMENTATION = "src/lgp/attacks/vcsf_a23_isolated.py"
A23_CONFIG = "configs/attacks/vcsf_selected_a23.yaml"
A23_EXECUTING_FILES = (
    A23_IMPLEMENTATION,
    "src/lgp/attacks/vcsf_common_anchor_isolated.py",
    "src/lgp/attacks/vcsf_scale_isolated.py",
    "src/lgp/attacks/vcsf_research_isolated.py",
    "src/lgp/attacks/vcsf_final_candidate.py",
    "src/lgp/attacks/common.py",
    "src/lgp/attacks/base.py",
    "src/lgp/attacks/isolated.py",
    "src/lgp/runners/attack.py",
    "src/lgp/runners/vcsf_research_plan.py",
    A23_CONFIG,
)
VCSF_FINAL_ABLATION_VARIANTS = (
    "neck_identity_reset",
    "neck_scale_reset",
    "cross_stage_identity_reset",
    "cross_stage_scale_feature_only",
    "cross_stage_scale_reset",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_file(root: Path, raw: str) -> Path:
    path = (root / str(raw)).resolve()
    path.relative_to(root.resolve())
    if not path.is_file():
        raise RuntimeError("Missing isolated-candidate file: {}".format(path))
    return path


def validate_isolated_vcsf_descriptor(
    registry: Registry,
    descriptor: Mapping[str, Any],
) -> Tuple[AttackSpec, Tuple[str, ...]]:
    """Validate the process-local VCSF descriptor without changing registries."""
    candidate_id = str(descriptor.get("candidate_id", ""))
    alias = str(descriptor.get("execution_alias", ""))
    if alias == "vcsf":
        raise RuntimeError("Historical A01/A23 overlays cannot use public vcsf; only A10 is prepared")
    if candidate_id != VCSF_CANDIDATE_ID or alias != "vcsf":
        raise RuntimeError("Unsupported isolated candidate descriptor")
    execution_mode = str(
        descriptor.get("execution_mode", "final_candidate")
    )
    if execution_mode not in VCSF_EXECUTION_MODES:
        raise RuntimeError("Unsupported isolated VCSF execution mode")
    if alias in registry.attacks or alias in ATTACK_TYPES:
        raise RuntimeError(
            "The isolated VCSF alias must not exist in the global registry or factory"
        )

    root = registry.root.resolve()
    implementation = _project_file(root, str(descriptor["implementation"]))
    config_path = _project_file(root, str(descriptor["config"]))
    freeze_path = _project_file(root, str(descriptor["freeze_manifest"]))
    expected_hashes = dict(descriptor.get("file_sha256", {}))
    if execution_mode == "selected_a23":
        if (implementation.relative_to(root).as_posix() != A23_IMPLEMENTATION
                or config_path.relative_to(root).as_posix() != A23_CONFIG
                or set(expected_hashes) != set(A23_EXECUTING_FILES)):
            raise RuntimeError("Selected A23 descriptor has an incomplete execution identity")
        checked_paths = [_project_file(root, relative) for relative in A23_EXECUTING_FILES]
    else:
        checked_paths = (implementation, config_path)
    for path in checked_paths:
        relative = path.relative_to(root).as_posix()
        if _sha256(path) != str(expected_hashes.get(relative, "")):
            raise RuntimeError(
                "Isolated VCSF file hash drift: {}".format(relative)
            )

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_status = (
        "selected_a23_pre_promotion_formal_closed"
        if execution_mode == "selected_a23"
        else "frozen_development_candidate_implementation_verified_formal_pending"
    )
    if freeze.get("status") != expected_status:
        raise RuntimeError("VCSF candidate freeze is not formal-pending")
    if (execution_mode == "selected_a23"
            and (freeze.get("parameters_sha256") != A23_PARAMETERS_SHA256
                 or freeze.get("formal_execution_authorized") is not False
                 or freeze.get("file_sha256") != expected_hashes)):
        raise RuntimeError("Selected A23 freeze identity or gate drifted")
    for relative, expected in expected_hashes.items():
        if freeze.get("file_sha256", {}).get(relative) != expected:
            raise RuntimeError(
                "VCSF descriptor disagrees with the candidate freeze: {}".format(
                    relative
                )
            )

    ablation_evidence: Dict[str, Any] = {}
    if execution_mode == "final_exact_ablation":
        ablation_implementation = _project_file(
            root, str(descriptor.get("ablation_implementation", ""))
        )
        ablation_registry = _project_file(
            root, str(descriptor.get("ablation_registry", ""))
        )
        ablation_hashes = dict(descriptor.get("ablation_file_sha256", {}))
        for path in (ablation_implementation, ablation_registry):
            relative = path.relative_to(root).as_posix()
            if _sha256(path) != str(ablation_hashes.get(relative, "")):
                raise RuntimeError(
                    "Isolated VCSF ablation file hash drift: {}".format(
                        relative
                    )
                )
        implementation_relative = ablation_implementation.relative_to(
            root
        ).as_posix()
        if freeze.get("file_sha256", {}).get(
            implementation_relative
        ) != _sha256(ablation_implementation):
            raise RuntimeError(
                "VCSF freeze does not bind the ablation oracle"
            )
        if tuple(descriptor.get("allowed_variants", ())) != (
            VCSF_FINAL_ABLATION_VARIANTS
        ):
            raise RuntimeError("VCSF final-ablation variant order drifted")
        ablation_evidence = {
            "ablation_implementation": implementation_relative,
            "ablation_registry": ablation_registry.relative_to(root).as_posix(),
            "allowed_variants": list(VCSF_FINAL_ABLATION_VARIANTS),
        }

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("id") != candidate_id:
        raise RuntimeError("Invalid isolated VCSF configuration")
    parameters = dict(payload.get("parameters", {}))
    if execution_mode == "selected_a23":
        config = VCSFCommonAnchorConfig.from_mapping(parameters)
        config.validate()
        resolved = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"),
                              allow_nan=False).encode("utf-8")
        if hashlib.sha256(resolved).hexdigest() != A23_PARAMETERS_SHA256:
            raise RuntimeError("Selected A23 complete parameter map drifted")
    else:
        VCSFFinalCandidateConfig.from_mapping(parameters).validate()

    sources = tuple(str(value) for value in descriptor.get("native_sources", []))
    if sources != tuple(registry.source_ids()):
        raise RuntimeError("Isolated VCSF source panel drifted")
    metadata: Dict[str, Any] = {
        key: value for key, value in payload.items() if key != "parameters"
    }
    metadata.update(
        {
            "fidelity_status": (
                "selected_a23_pre_promotion_formal_closed"
                if execution_mode == "selected_a23"
                else "frozen_formal_candidate"
            ),
            "fidelity_basis": (
                "hash_bound_selected_candidate_not_formal_accepted"
                if execution_mode == "selected_a23"
                else "verified_process_local_candidate_overlay"
            ),
            "semantic_contract": "view_consistent_cross_stage_feature_survival",
            "implementation": implementation.relative_to(root).as_posix(),
            "candidate_id": candidate_id,
            "global_registry_mutated": False,
            "global_factory_mutated": False,
            "execution_mode": execution_mode,
            **ablation_evidence,
        }
    )
    if execution_mode == "final_exact_ablation":
        metadata["implementation"] = ablation_evidence[
            "ablation_implementation"
        ]
        metadata["base_implementation"] = implementation.relative_to(
            root
        ).as_posix()
    return (
        AttackSpec(
            id=alias,
            display_name="VCSF",
            executor="native",
            parameters=parameters,
            metadata=metadata,
        ),
        sources,
    )


def resolve_isolated_vcsf(
    registry: Registry,
    attack_id: str,
    source_id: str,
    descriptor: Mapping[str, Any],
) -> AttackSpec:
    spec, sources = validate_isolated_vcsf_descriptor(registry, descriptor)
    if attack_id != spec.id:
        raise RuntimeError("Isolated candidate alias does not match the attack job")
    if source_id not in sources:
        raise RuntimeError(
            "VCSF is not authorized on source '{}'".format(source_id)
        )
    return spec


def build_isolated_vcsf(
    adapter: Any,
    parameters: Mapping[str, Any],
    descriptor: Optional[Mapping[str, Any]] = None,
) -> Any:
    raise RuntimeError(
        "Historical A01/A23 overlay builder is retired for public vcsf; "
        "preserved research classes are not public alternatives"
    )


def install_process_local_report_alias(
    registry: Registry,
    descriptor: Mapping[str, Any],
) -> None:
    """Expose VCSF only on one Registry instance used by report writers."""
    spec, sources = validate_isolated_vcsf_descriptor(registry, descriptor)
    registry.attacks[spec.id] = spec
    methods = registry.compatibility.setdefault("methods", {})
    methods[spec.id] = {
        "native": list(sources),
        "external": [],
        "adaptable": [],
        "unsupported": [],
        "reason": "Process-local formal-candidate reporting alias.",
    }

