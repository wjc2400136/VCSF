"""Selected A10 numerical identity and new-input public reproduction dispatch."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

import yaml


PARAMETERS_SHA256 = "2551944131096c1d74011f2eb8d789b858e6dce79748f5b5aada54ea18254328"
PROTOCOL_SHA256 = "b9c8ddf613a70dd6623cdb71c5070a83352a39c0bfe755ddaee24600cffc0084"
RECOMMENDATION_SHA256 = "159e0883d58e9c802b33269897dd4e4e92eec7a913183cc6ddfafbd32025b955"
CONFIG = "configs/attacks/vcsf.yaml"
PROTOCOL = "docs/research/vcsf-a10-selection-protocol.json"
RECOMMENDATION = "docs/research/vcsf-a10-configuration-recommendation-provenance.md"
FREEZE = "docs/research/vcsf-a10-public-source-freeze.json"
QUALIFICATION_ENTRY = "experiments/qualify_vcsf_a10_public.py"
STATUS = "selected_A10_public_reproduction_candidate_not_project_wide_release"


def _hash(value):
    from ..runners.vcsf_research_plan import canonical_hash

    return canonical_hash(value)


def _plain_file(root, relative):
    path = root / relative
    if (Path(relative).is_absolute() or path.resolve() != path
            or root not in path.parents or not path.is_file()):
        raise RuntimeError("Public identity requires a plain project file: " + relative)
    return path


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest(root):
    from ..public_package import verify_package

    return verify_package(root)


def _registered_config(parameters, expected):
    from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig

    if not isinstance(parameters, dict) or set(parameters) != set(expected):
        raise ValueError("Public A10 requires the complete registered parameter map")
    config = VCSFCommonAnchorConfig.from_mapping(parameters)
    config.validate()
    if _hash(asdict(config)) != PARAMETERS_SHA256 or _hash(expected) != PARAMETERS_SHA256:
        raise ValueError("Public A10 complete parameter identity drifted")
    return config


def verify_public_identity(root, attack_spec=None):
    root = Path(root).resolve()
    protocol_path = _plain_file(root, PROTOCOL)
    if _digest(protocol_path) != PROTOCOL_SHA256:
        raise RuntimeError("Registered selection protocol bytes changed")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    decision = protocol["decision"]
    execution = protocol["execution_identity"]
    if (decision["unique_working_candidate"] != "A10"
            or decision["parameters_sha256"] != PARAMETERS_SHA256
            or _hash(decision["parameters"]) != PARAMETERS_SHA256):
        raise RuntimeError("Registered A10 selection parameter identity changed")
    if _digest(_plain_file(root, RECOMMENDATION)) != RECOMMENDATION_SHA256:
        raise RuntimeError("Configuration recommendation provenance changed")
    numeric = execution["numeric_source_sha256"]
    if _hash(numeric) != execution["numeric_source_map_sha256"]:
        raise RuntimeError("Registered numerical source map changed")
    for relative, expected in numeric.items():
        if _digest(_plain_file(root, relative)) != expected:
            raise RuntimeError("Registered numerical source changed: " + relative)

    declaration = yaml.safe_load(_plain_file(root, CONFIG).read_text(encoding="utf-8"))
    required = {
        "schema_version": 1, "id": "vcsf", "display_name": "VCSF", "executor": "native",
        "internal_configuration": "A10", "status": "public_reproduction_candidate_execution_enabled",
        "implementation": "src/lgp/attacks/vcsf_common_anchor_isolated.py",
        "public_builder": "lgp.attacks.vcsf_public.build_public_vcsf",
        "parameters_sha256": PARAMETERS_SHA256, "selection_protocol": PROTOCOL,
        "source_freeze": FREEZE, "fidelity_status": "registered_A10_numeric_identity_preserved",
        "fidelity_basis": "actual_common_anchor_builder_not_alias_derived_identity",
        "execution_authorized": True, "scientific_selection_closed": True,
        "science": False, "formal": False, "release": False,
    }
    if (not isinstance(declaration, dict) or set(declaration) != set(required) | {"parameters"}
            or any(type(declaration[k]) is not type(v) or declaration[k] != v
                   for k, v in required.items())):
        raise RuntimeError("Public A10 declaration or closed acceptance gates drifted")
    config = _registered_config(declaration["parameters"], decision["parameters"])
    if attack_spec is not None:
        metadata = {key: value for key, value in declaration.items() if key != "parameters"}
        if (attack_spec.id != "vcsf" or attack_spec.display_name != "VCSF"
                or attack_spec.executor != "native" or dict(attack_spec.metadata) != metadata
                or dict(attack_spec.parameters) != declaration["parameters"]):
            raise RuntimeError("Registry public VCSF differs from its sole declaration")

    # The six original documents remain immutable; the public closure has a new identity.
    from ..public_package import FREEZE as package_freeze, read_json
    manifest = source_manifest(root)
    package_identity = read_json(root, package_freeze)
    return {
        "parameters": asdict(config), "parameters_sha256": PARAMETERS_SHA256,
        "selection_protocol_sha256": PROTOCOL_SHA256,
        "recommendation_sha256": RECOMMENDATION_SHA256,
        "numeric_source_sha256": dict(numeric),
        "numeric_source_map_sha256": execution["numeric_source_map_sha256"],
        "historical_actual_builder": execution["historical_actual_builder"],
        "actual_class_mro": list(execution["actual_class_mro"]),
        "source_manifest_sha256": _hash(manifest), "source_freeze_sha256": _digest(root / package_freeze),
        "original_source_freeze_sha256": _digest(root / FREEZE),
        "source_identity_schema": package_identity["schema"],
        "old_full_freeze_equivalence_claimed": False,
        "execution_authorized": True, "scientific_selection_closed": True,
        "whole_execution_equivalence_claimed": False,
        "science": False, "formal": False, "release": False,
    }


def build_public_vcsf(adapter, parameters):
    root = Path(__file__).resolve().parents[3]
    identity = verify_public_identity(root)
    _registered_config(dict(parameters), identity["parameters"])
    from .vcsf_common_anchor_execution_isolation import build_common_anchor_execution

    attack = build_common_anchor_execution(adapter, dict(parameters))
    ancestry = [cls.__module__ + "." + cls.__name__ for cls in type(attack).__mro__]
    if ancestry != identity["actual_class_mro"] or _hash(asdict(attack.config)) != PARAMETERS_SHA256:
        raise RuntimeError("Public builder changed the actual registered common-anchor path")
    return attack


def guard_public_execution(registry, parameter_overrides=None, execution_context=None, invocation=None):
    identity = verify_public_identity(registry.root, registry.attack("vcsf"))
    parameters = dict(registry.attack("vcsf").parameters)
    parameters.update(dict(parameter_overrides or {}))
    if (execution_context is None and type(invocation) is dict
            and (invocation.get("run_metadata") or {}).get("public_protocol")
            == "current_vcsf_final_background"):
        from .current_vcsf_background import validate_invocation

        validate_invocation(registry, invocation, parameter_overrides)
        return identity
    if (execution_context is None and type(invocation) is dict
            and invocation.get("budget_profile") == "vcsf_final_radius_observed_cost"):
        from .current_vcsf_radius import validate_invocation

        validate_invocation(registry, invocation, parameter_overrides)
        return identity
    _registered_config(parameters, identity["parameters"])
    if execution_context is not None:
        from ..runners.vcsf_a10_main_context import MainExecutionContext

        if (type(execution_context) is not MainExecutionContext
                or execution_context.registry is not registry
                or execution_context.root != Path(__file__).resolve().parents[3]
                or type(invocation) is not dict):
            raise RuntimeError("Public native execution requires the exact admitted A10 main worker capability")
        execution_context.guard_attack(invocation)
        return identity
    if invocation is not None:
        validate_reproduction_invocation(registry, invocation)
    return identity


def validate_reproduction_invocation(registry, invocation):
    """Bind current user inputs without inheriting an author's historical run."""
    if type(invocation) is not dict:
        raise ValueError("Public reproduction requires explicit invocation arguments")
    if (type(invocation.get("dataset_id")) is not str
            or type(invocation.get("split")) is not str
            or invocation.get("dataset_id") not in {"coco", "voc"}
            or invocation.get("split") not in {"dev", "val"}
            or invocation.get("source_id") not in registry.source_ids()
            or invocation.get("attack_id") != "vcsf"):
        raise ValueError("Public VCSF reproduction is restricted to current COCO/VOC source inputs")
    maximum = invocation.get("max_images")
    if maximum is not None and (type(maximum) is not int or maximum <= 0):
        raise ValueError("max_images must be a positive integer or None")
    if type(invocation.get("seed")) is not int:
        raise ValueError("Public reproduction requires an integer seed")
    device = invocation.get("device")
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:[0-9]+", device):
        raise ValueError("Public reproduction requires a CPU or explicit CUDA device")
    profile = invocation.get("budget_profile")
    if profile is not None and profile != "vcsf_final_background_observed_cost":
        raise ValueError("Public VCSF requires its registered logical-update observed-cost profile")
