"""Prepare a new single-source scale study without running or resuming models."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import sys

from ..attacks.factory import ATTACK_TYPES
from ..attacks.vcsf_scale_isolated import VCSFScaleConfig
from ..io import atomic_json, file_digest
from ..registry import Registry
from .vcsf_research_plan import canonical_hash, require


PROTOCOL = "vcsf_single_source_scale_research"


def compile_scale_plan(registry, max_images=None):
    if max_images is not None:
        require(type(max_images) is int and 0 < max_images < 5000,
            "A diagnostic image limit must be an integer between 1 and 4999")
    protocol = registry.protocols[PROTOCOL]
    study = registry.ablation_studies[protocol["ablation_study"]]
    base = registry.ablation_studies[study["base_study"]]
    baseline = dict(base["defaults"], **base["variants"][study["base_variant"]]["parameters"])
    require(study["base_variant"] == "levels_two" and baseline["levels_per_stage"] == 2,
        "The working baseline must preserve the registered four-level deletion")
    require(study["method"] not in registry.attacks and study["method"] not in ATTACK_TYPES,
        "Scale research must not modify the global attack registry")
    rows = study["row_order"]
    widths = [Fraction(str(value)) for value in study["half_widths"]]
    require(rows == list(study["variants"]) and len(rows) == len(set(rows)) == 4,
        "Scale rows must be unique and in registered order")
    require(widths == sorted(set(widths)) and widths[0] == 0 and len(widths) == 5,
        "Scale widths must be five ordered distinct half-widths including identity")
    targets, sources = registry.target_ids(), protocol["sources"]
    require(len(targets) == 16 and sources == registry.source_ids()[:1],
        "Single-source research must retain the full canonical target panel")
    anchor_row = study["anchor"]["row"]
    anchor_width = Fraction(str(study["anchor"]["half_width"]))
    require(anchor_row == rows[0] and anchor_width in widths[1:], "Invalid working anchor")
    variants, display = {}, []
    for row in rows:
        for width_index, width in enumerate(widths):
            canonical_row = rows[0] if width == 0 else row
            name = "{}_h{}".format(canonical_row, width_index)
            parameters = dict(baseline, **study["variants"][canonical_row]["parameters"])
            parameters.update(scale_min=float(1 - width), scale_max=float(1 + width))
            config = VCSFScaleConfig.from_mapping(parameters)
            config.validate()
            resolved = asdict(config)
            if name not in variants:
                variants[name] = dict(variant=name, row=canonical_row, half_width=str(width),
                    parameters=resolved, parameters_sha256=canonical_hash(resolved),
                    potential_historical_reference=canonical_row == anchor_row and width == anchor_width)
            display.append(dict(row=row, half_width=str(width), variant=name,
                identity_alias_requires_validation=width == 0 and row != canonical_row))
    require(len({value["parameters_sha256"] for value in variants.values()}) == len(variants),
        "Duplicate non-identity configuration")
    ordered, stages = [], []
    for name in study["stage_order"]:
        stage = study["stages"][name]
        introduced = []
        for width in stage["half_widths"]:
            index = widths.index(Fraction(str(width)))
            for row in stage["rows"]:
                variant = "{}_h{}".format(rows[0] if index == 0 else row, index)
                require(variant in variants and variant not in ordered, "Duplicate or unknown scheduled configuration")
                ordered.append(variant)
                introduced.append(variant)
        stages.append(dict(stage=name, variants=introduced))
    require(set(ordered) == set(variants), "Unscheduled scale configuration")
    images = protocol["images"] if max_images is None else max_images
    groups = [dict(group_id=index + 1, source=sources[0], seed=protocol["seed"],
        images=images, targets=targets, blackbox_targets=[t for t in targets if t != sources[0]],
        logical_gradients=20, detector_initialization_updates=1, feature_updates=19,
        reference_forwards_not_free=True, reuse_status="unassessed_do_not_subtract",
        execution_status="not_armed", **variants[name]) for index, name in enumerate(ordered)]
    totals = dict(display_cells=len(display), unique_configurations=len(groups),
        source_groups=len(groups), generated_image_instances=len(groups) * protocol["images"],
        target_evaluations=len(groups) * len(targets))
    require(totals == study["planning_totals_before_reuse"], "Scale study totals differ from the registry")
    anchor_index = widths.index(anchor_width)
    identity = "{}_h0".format(anchor_row)
    contrasts = []
    def add(name, kind, terms):
        coefficients = {}
        for variant, coefficient in terms:
            coefficients[variant] = coefficients.get(variant, 0) + coefficient
        contrasts.append(dict(name=name, kind=kind,
            terms=[dict(variant=variant, coefficient=str(coefficient))
                for variant, coefficient in coefficients.items() if coefficient]))
    for row in rows:
        for index in range(1, len(widths)):
            variant = "{}_h{}".format(row, index)
            add(variant + "_minus_identity", "scale_effect", [(variant, 1), (identity, -1)])
            if row == anchor_row:
                continue
            same_width = "{}_h{}".format(anchor_row, index)
            add(variant + "_minus_A", "operator_effect", [(variant, 1), (same_width, -1)])
            if index != anchor_index:
                add(variant + "_interaction", "difference_in_differences", [(variant, 1),
                    (same_width, -1), ("{}_h{}".format(row, anchor_index), -1),
                    ("{}_h{}".format(anchor_row, anchor_index), 1)])
    source_files = ["configs/experiments/ablations.yaml", "configs/experiments/protocols.yaml",
        "src/lgp/registry.py", "src/lgp/attacks/vcsf_scale_isolated.py",
        "src/lgp/attacks/vcsf_research_isolated.py", "src/lgp/attacks/vcsf_final_candidate.py",
        "src/lgp/attacks/common.py", "src/lgp/runners/vcsf_scale_plan.py",
        "src/lgp/runners/vcsf_research_plan.py", "experiments/prepare_vcsf_single_source_scale.py",
        "src/lgp/attacks/vcsf_scale_isolation.py", "src/lgp/attacks/vcsf_research_isolation.py",
        "src/lgp/runners/attack.py", "src/lgp/runners/vcsf_scale_preflight.py",
        "experiments/vcsf_single_source_scale_preflight.py", "src/lgp/attacks/base.py",
        "src/lgp/adapters/openmmlab.py", "src/lgp/modeling.py", "src/lgp/data/coco.py",
        "src/lgp/io.py", "src/lgp/paths.py", "configs/models.yaml", "configs/datasets/coco.yaml",
        "configs/experiments/budgets.yaml", "src/lgp/runners/vcsf_structure_workers.py"]
    return dict(schema_version=1, protocol=PROTOCOL, status="prepared_not_executable",
        runner_armed=False, model_calls=0, independent_confirmation=False,
        automatic_promotion=False, scientific_acceptance=False, max_images=max_images,
        diagnostic_only=max_images is not None, original_run_restart=False,
        sources=sources, targets=targets, working_baseline=study["base_variant"],
        anchor="{}_h{}".format(anchor_row, anchor_index),
        base_study_sha256=canonical_hash(base), protocol_sha256=canonical_hash(protocol),
        study_sha256=canonical_hash(study), source_sha256={p: file_digest(registry.root / p) for p in source_files},
        totals_before_qualified_reuse=totals, diagnostic_image_instances=len(groups) * images,
        stages=stages, groups=groups, display_cells=display, contrasts=contrasts,
        analysis_policy=study["analysis_policy"],
        remaining_gates=["ODA_operator_forward_backward_rng_tests", "identity_alias_equivalence",
            "real_source_structure_and_exact_anchor_bridge", "historical_reuse_adjudication",
            "data_checkpoint_evaluator_binding", "paired_analysis_contract_validation",
            "independent_execution_and_lifecycle_validation", "fresh_run_admission"])


def compile_stage_plan(registry, stage_name, devices, max_images=None, analysis_mode='legacy_bootstrap'):
    from .vcsf_scale_execution_contract import balanced_lanes

    prepared = compile_scale_plan(registry, max_images)
    execution = registry.protocols["vcsf_single_source_scale_execution"]
    require(analysis_mode in ('legacy_bootstrap', 'point_estimate'), 'Unknown scale analysis mode')
    point = analysis_mode == 'point_estimate'
    definition = registry.protocols['vcsf_scale_point_stages'] if point else execution["staged_analysis"]
    order = [stage["stage"] for stage in prepared["stages"]]
    require(order == definition["stage_order"] and stage_name in order,
        "Select a registered scale stage")
    names = [contrast["name"] for contrast in prepared["contrasts"]]
    require(len(set(names)) == len(names), "Duplicate registered scale contrast")
    available, assigned, families = set(), set(), []
    for stage, expected_size in zip(prepared["stages"], definition["contrast_counts" if point else "family_sizes"]):
        available.update(stage["variants"])
        family = [deepcopy(c) for c in prepared["contrasts"] if c["name"] not in assigned
            and {term["variant"] for term in c["terms"]} <= available]
        require(len(family) == expected_size, "Scale stage contrast family differs from its registration")
        assigned.update(c["name"] for c in family)
        families.append(family)
    require(assigned == set(names), "Unassigned scale contrast")
    index = order.index(stage_name)
    introduced = set(prepared["stages"][index]["variants"])
    previous = {variant for stage in prepared["stages"][:index] for variant in stage["variants"]}
    groups = [deepcopy(g) for g in prepared["groups"] if g["variant"] in introduced]
    prerequisites = [deepcopy(g) for g in prepared["groups"] if g["variant"] in previous]
    lanes = balanced_lanes(groups, prepared["sources"], devices)
    sampling_keys = ("metric", "scalar", "unit", "resampling", "replicates", "rng", "seed",
        "image_occurrence_order", "alpha", "multiplicity", "seed_estimand", "finite_sample_guarantee")
    if point:
        analysis = {k: deepcopy(definition[k]) for k in ('metric', 'scalar', 'seed_estimand')}
        analysis.update(analysis_mode='point_estimate', family_size=len(families[index]),
            contrasts=families[index], bootstrap_required=False, significance_claimed=False,
            confidence_intervals_produced=False, independent_confirmation=False,
            selection_adjusted_inference=False, prior_intervals_reused=False)
    else:
        analysis = {k: deepcopy(execution["first_stage_analysis"][k]) for k in sampling_keys}
        analysis.update(family_size=len(families[index]), contrasts=families[index],
            multiplicity_scope=definition["multiplicity_scope"], across_stage_error_control_claimed=False,
            prior_intervals_reused=False, selection_adjusted_inference=False, independent_confirmation=False)
    slots = [dict(group_id=g["group_id"], variant=g["variant"], target=target, seed=g["seed"],
        parameters_sha256=g["parameters_sha256"], images=g["images"],
        disposition="previous_stage_evidence_required" if g["variant"] in previous else "stage_work_unassessed",
        bootstrap_required=not point and target != g["source"])
        for g in prerequisites + groups for target in prepared["targets"]]
    return dict(schema_version=2 if point else 1, status="prepared_stage_not_executable", stage=stage_name,
        protocol=prepared["protocol"], prepared_plan_sha256=canonical_hash(prepared), prepared_plan=prepared,
        stage_definition=deepcopy(definition), stage_definition_sha256=canonical_hash(definition),
        groups=groups, previous_stage_groups=prerequisites, previous_stages=order[:index],
        sources=prepared["sources"], targets=prepared["targets"],
        introduced_groups_before_qualified_reuse=len(groups), qualified_reuse_groups=0,
        scheduled_images_before_qualified_reuse=sum(g["images"] for g in groups),
        target_evaluations_before_qualified_reuse=len(groups) * len(prepared["targets"]),
        lane_group_ids=[[g["group_id"] for g in lane] for lane in lanes], devices=list(devices),
        analysis=analysis, analysis_sha256=canonical_hash(analysis), analysis_slots=slots,
        all_target_cells=len(slots), blackbox_bootstrap_cells=sum(s["bootstrap_required"] for s in slots),
        no_early_efficacy_pruning=True, historical_reuse_granted=False,
        remaining_gates=["previous_stage_full_panel_and_paired_analysis_independent_acceptance",
            "previous_stage_cost_and_complexity_decision", "exact_previous_cell_reuse_binding",
            "stage_analysis_runtime_and_source_bound_tests", "actual_single_and_dual_device_smokes",
            "fresh_execution_plan_and_independent_admission"],
        runner_armed=False, model_calls=0, scientific_acceptance=False, automatic_promotion=False,
        independent_confirmation=False, diagnostic_only=max_images is not None, max_images=max_images)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Print the plan without writing an output directory.")
    parser.add_argument("--max-images", type=int, help="Diagnostic planning limit; never formal efficacy.")
    parser.add_argument("--stage", help="Inspect one registered stage, including prerequisites and exact contrasts.")
    parser.add_argument("--analysis-mode", choices=('legacy_bootstrap', 'point_estimate'), default='point_estimate',
        help="Point estimates are the current exploration path; legacy mode only inspects original contracts.")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"],
        help="Stage-planning device assignment; no device is initialized or claimed available.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or Path(sys.prefix).name != "oda" or sys.executable != str(Path(sys.prefix) / "bin" / "python"):
        raise RuntimeError("The research workflow requires the pinned server ODA interpreter")
    result = (compile_stage_plan(Registry(), args.stage, args.devices, args.max_images, args.analysis_mode)
        if args.stage else compile_scale_plan(Registry(), args.max_images))
    if not args.plan_only:
        output = args.output or Path.cwd() / "outputs/plans" / PROTOCOL / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output.mkdir(parents=True, exist_ok=False)
        atomic_json(output / "plan.json", result)
    summary = (dict(stage=result["stage"], introduced_groups=result["introduced_groups_before_qualified_reuse"],
        prerequisite_groups=len(result["previous_stage_groups"]), lane_group_ids=result["lane_group_ids"],
        all_target_cells=result["all_target_cells"], blackbox_bootstrap_cells=result["blackbox_bootstrap_cells"],
        contrasts=result["analysis"]["family_size"]) if args.stage else
        dict(totals=result["totals_before_qualified_reuse"], contrasts=len(result["contrasts"])))
    print(json.dumps(dict(summary, status=result["status"], model_calls=0, runner_armed=False), indent=2))
    return 0
