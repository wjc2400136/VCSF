"""Compile the registered layered study without arming or running experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from ..attacks.factory import ATTACK_TYPES
from ..attacks.vcsf_research_isolated import VCSFResearchConfig
from ..io import atomic_json, file_digest
from ..registry import Registry
from .vcsf_layered_contrasts import compile_contrasts


PROTOCOL = "vcsf_fullval_retrospective_ablation"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compile_plan(registry):
    protocol = registry.protocols[PROTOCOL]
    study = registry.ablation_studies[protocol["ablation_study"]]
    require(protocol["split"] == "val" and protocol["images"] == 5000,
        "Every efficacy group must use full COCO val2017")
    require(protocol["independent_confirmation"] is False and
        protocol["selection_namespace"] == "retrospective_fullval_reuse", "Undisclosed reused-set selection")
    require(study["method"] not in registry.attacks and study["method"] not in ATTACK_TYPES,
        "Research must remain process-local")
    reference = registry.root / "src/lgp/attacks/vcsf_final_candidate.py"
    require(file_digest(reference) == protocol["frozen_vcsf_sha256"], "Frozen VCSF changed")
    sources = protocol["sources"]
    targets = registry.target_ids()
    require(sources == registry.source_ids()[:2], "Common-2 source scope/order changed")
    require(len(targets) == 16 and all(s in targets for s in sources), "Incomplete target panel")
    resolved = {}
    fingerprints = set()
    for name, variant in study["variants"].items():
        config = VCSFResearchConfig.from_mapping(dict(study["defaults"], **variant["parameters"]))
        config.validate()
        raw = asdict(config)
        fingerprint = canonical_hash(raw)
        require(fingerprint not in fingerprints, "Duplicate configuration under different names: " + name)
        fingerprints.add(fingerprint)
        resolved[name] = {"parameters": raw, "parameters_sha256": fingerprint,
            "is_frozen_reference": config.is_reference(),
            "logical_gradients": config.iterations,
            "risk_updates": int(config.initialization == "detector"),
            "feature_updates": config.iterations - int(config.initialization == "detector"),
            "random_pixel_initialization": config.initialization == "random_sign"}
    require(resolved["full"]["is_frozen_reference"], "The anchor must be the unmodified reference")
    ordered_pairs, stages, seen = [], [], set()
    for block_name in study["execution_order"]:
        block = study["blocks"][block_name]
        require(len(set(block["variants"])) == len(block["variants"]) and
            len(set(block["seeds"])) == len(block["seeds"]), "Duplicate block contrast")
        shared, introduced = [], []
        for variant in block["variants"]:
            require(variant in resolved, "Unknown block variant")
            for seed in block["seeds"]:
                require(type(seed) is int, "Seed must be an integer")
                pair = (variant, seed)
                if pair in seen:
                    shared.append(pair)
                else:
                    seen.add(pair)
                    ordered_pairs.append(pair)
                    introduced.append(pair)
        stages.append({"block": block_name, "purpose": block["purpose"],
            "introduced_pairs": introduced, "shared_pairs": shared,
            "requires": ["independent_cost_acceptance"] if not stages else [stages[-1]["block"]]})
    require({variant for variant, _ in seen} == set(resolved), "Unscheduled variant")
    groups = []
    for variant, seed in ordered_pairs:
        for source in sources:
            groups.append({"group_id": len(groups) + 1, "variant": variant, "seed": seed,
                "source": source, "images": protocol["images"], "targets": targets,
                "blackbox_targets": [t for t in targets if t != source],
                "source_matched_target": source, **resolved[variant],
                "reuse_status": "unassessed_do_not_subtract", "execution_status": "not_armed"})
    totals = {"unique_variants": len(resolved), "configuration_seed_pairs": len(ordered_pairs),
        "source_groups": len(groups), "generated_image_instances": len(groups) * protocol["images"],
        "attack_evaluations": len(groups) * len(targets)}
    require(totals == study["planning_totals_before_reuse"], "Declared planning counts are inconsistent")
    analysis = compile_contrasts(study, resolved, ordered_pairs)
    files = sorted({path.relative_to(registry.root).as_posix()
        for directory, suffix in (("src/lgp", ".py"), ("configs", ".yaml"))
        for path in (registry.root / directory).rglob("*")
        if path.is_file() and path.suffix == suffix})
    files.extend(["experiments/vcsf_layered_structure_preflight.py",
        "tools/audit_cost_calibration.py", "tests/test_vcsf_final_candidate.py",
        "tests/test_vcsf_research_isolated.py", "tests/test_vcsf_research_isolation.py",
        "tests/test_vcsf_layered_preflight.py", "tests/test_vcsf_structure_workers.py",
        "tests/test_vcsf_layered_contrasts.py", "tests/test_vcsf_image_bootstrap.py",
        "tests/test_vcsf_paired_statistics.py", "tests/test_vcsf_prediction_evidence.py",
        "tests/test_vcsf_coco_replay.py"])
    return {"status": "prepared_not_executable", "protocol": PROTOCOL,
        "scientific_evidence": False, "runner_armed": False, "model_calls": 0,
        "configuration_registry_sha256": canonical_hash(study),
        "protocol_registry_sha256": canonical_hash(protocol),
        "file_sha256": {name: file_digest(registry.root / name) for name in files},
        "totals_before_qualified_reuse": totals, "stages": stages, "groups": groups,
        "sources": sources, "targets": targets,
        "analysis": analysis,
        "blackbox_cells_per_configuration_seed": len(sources) * (len(targets) - 1),
        "selection_namespace": protocol["selection_namespace"], "independent_confirmation": False,
        "remaining_execution_gates": ["accepted_cost_root", "server_tests", "all_variant_structure",
            "six_source_reference_compatibility", "padding_extremes", "source_layer_signature",
            "data_checkpoint_evaluator_hashes", "reuse_adjudication", "prediction_archive_lifecycle",
            "analysis_and_selection_policy_lock", "disk_and_duplicate_run_check", "armed_stage_runner"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="This planning entry never runs models.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or Path(sys.prefix).name != "oda" or sys.executable != str(Path(sys.prefix) / "bin" / "python"):
        raise RuntimeError("The research workflow uses the pinned server ODA interpreter")
    registry = Registry()
    output = args.output or registry.root / "outputs/plans/vcsf_finalization_layered" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    result = compile_plan(registry)
    atomic_json(output / "plan.json", result)
    print(json.dumps(result["totals_before_qualified_reuse"], indent=2))
    print("[PREPARED] Not armed. No model calls, AP evaluation, or reuse acceptance.")
    return 0
