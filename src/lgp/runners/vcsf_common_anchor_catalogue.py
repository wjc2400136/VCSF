"""Internal, outcome-independent common-anchor preparation; never arm jobs.

Explicit parameters are a proposed binding, not evidence of operator acceptance.
No historical endpoint resolver, result reader, or execution adapter is used.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, fields

from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
from ..attacks.vcsf_operator_factorial_isolated import VCSFOperatorFactorialConfig
from .vcsf_layered_contrasts import compile_contrasts
from .vcsf_research_plan import canonical_hash, require


_PROTOCOL = "vcsf_single_source_ablation_research"


def _parameters(raw, config_type):
    require(isinstance(raw, Mapping), "Explicit complete anchor parameters are required")
    require(set(raw) == {field.name for field in fields(config_type)},
        "Parameters must be complete with no unknown fields")
    config = config_type.from_mapping(deepcopy(dict(raw)))
    config.validate()
    return asdict(config)


def _resolved(parameters, budget):
    parameters = _parameters(parameters, VCSFCommonAnchorConfig)
    iterations = parameters["iterations"]
    initialization = parameters["initialization"]
    require(iterations in budget["allowed_iterations_by_initialization"][initialization],
        "Configuration is outside its registered initializer budget")
    risk = int(initialization == "detector")
    return dict(parameters=parameters, parameters_sha256=canonical_hash(parameters),
        logical_gradients=iterations, risk_updates=risk, feature_updates=iterations - risk,
        feature_terms=parameters["levels_per_stage"] *
            (2 if parameters["surface"] == "cross_stage" else 1),
        random_pixel_initialization=initialization == "random_sign",
        reference_forwards_not_free=True)


def compile_common_anchor_catalogue(registry, anchor_parameters):
    """Bind B once and prepare exact-identity groups and background-local contrasts.

    The caller must supply every config field. Acceptance, input/implementation
    binding, historical reuse and execution admission remain external gates.
    """
    baseline = _parameters(anchor_parameters, VCSFOperatorFactorialConfig)
    baseline = _parameters(baseline, VCSFCommonAnchorConfig)
    require((baseline["scale_min"], baseline["scale_max"]) == (2.0 / 3.0, 4.0 / 3.0),
        "Common anchor requires nonzero half-width 1/3")
    registry._validate()
    protocol = deepcopy(registry.protocols[_PROTOCOL])
    study = deepcopy(registry.ablation_studies[protocol["ablation_study"]])
    budget = deepcopy(registry.budget_profiles[protocol["budget_profile"]])
    require(len(study["variants"]) == 27 and study["analysis_policy"]["anchor"] == "full"
        and study["variants"]["full"]["parameters"] == {}, "Registered catalogue/anchor changed")
    order = study["execution_order"]
    require(len(order) == len(set(order)) and set(order) == set(study["blocks"]),
        "Every block must have exactly one stage mapping")
    require(protocol["seed"] == study["analysis_policy"]["primary_seed"] == 42
        and protocol["sources"] == ["faster_rcnn_r50"] and protocol["images"] == 5000,
        "Common exploration source/image/seed scope changed")
    require(all(block["seeds"] == [42] for block in study["blocks"].values()),
        "Common exploration block seeds changed")
    geometry_names = study["blocks"]["correspondence"]["variants"]
    require(geometry_names == ["full", "independent_offsets", "independent_geometry"]
        and study["blocks"]["correspondence"]["allowed_fields"] == ["correspondence"],
        "Correspondence must retain exactly three declared controls")
    centered = baseline["placement"] == "center"
    backgrounds = {"B": baseline}
    if centered:
        backgrounds["Q"] = dict(baseline, placement="random")
    resolved = {role: {} for role in backgrounds}
    for name, spec in study["variants"].items():
        role = "Q" if centered and name in geometry_names[1:] else "B"
        resolved[role][name] = _resolved(dict(backgrounds[role], **spec["parameters"]), budget)
    if centered:
        resolved["Q"]["full"] = _resolved(backgrounds["Q"], budget)
    require(resolved["B"]["full"]["parameters"] == baseline, "B must remain fixed")

    block_roles = {}
    for stage, block in study["blocks"].items():
        role = "Q" if centered and stage == "correspondence" else "B"
        block_roles[stage] = role
        allowed = block.get("allowed_fields", [])
        require(allowed and len(allowed) == len(set(allowed)) and set(allowed) <= set(baseline),
            "Missing or invalid allowed parameter fields for block: " + stage)
        for name in block["variants"]:
            require(name in resolved[role], "Missing block configuration: " + name)
            parameters = resolved[role][name]["parameters"]
            require(all(value == backgrounds[role][key] for key, value in parameters.items()
                if key not in allowed), "Non-factor parameter confound in block variant: " + name)
    matched = study["blocks"]["matched_terms"]
    controls = [resolved["B"][name] for name in matched["variants"]]
    require(all(row["feature_terms"] == matched["feature_terms"] for row in controls),
        "Matched-stage controls do not have equal feature-term count")
    require(all(all(row["parameters"][key] == controls[0]["parameters"][key]
        for key in baseline if key not in ("surface", "levels_per_stage")) for row in controls),
        "Matched-stage controls differ outside stage and layer count")
    schedules = study["blocks"]["initialization_specificity"]
    require(set(schedules["schedules"]) == set(schedules["variants"]),
        "Initializer schedule coverage changed")
    for name, schedule in schedules["schedules"].items():
        require(set(schedule) == {"risk_updates", "feature_updates"}
            and all(resolved["B"][name][key] == value for key, value in schedule.items()),
            "Initializer risk/feature budget mismatch: " + name)

    targets = list(registry.target_ids())
    groups, by_hash, bindings, stages = [], {}, {role: {} for role in backgrounds}, []
    for stage in order:
        role, block = block_roles[stage], study["blocks"][stage]
        introduced, shared, ids = [], [], []
        for name in block["variants"]:
            row = resolved[role][name]
            digest = row["parameters_sha256"]
            if digest not in by_hash:
                group = dict(group_id=len(groups) + 1, variant=role + ":" + name,
                    seed=42, source=protocol["sources"][0], images=protocol["images"],
                    targets=list(targets), blackbox_targets=[t for t in targets if t != protocol["sources"][0]],
                    source_matched_target=protocol["sources"][0], bindings=[],
                    reuse_status="unassessed_do_not_subtract", execution_status="not_armed", **deepcopy(row))
                groups.append(group)
                by_hash[digest] = group
                introduced.append(group["group_id"])
            else:
                group = by_hash[digest]
                require(group["parameters"] == row["parameters"], "Parameter hash collision")
                shared.append(group["group_id"])
            binding = dict(background=role, variant=name)
            if binding not in group["bindings"]:
                group["bindings"].append(binding)
            bindings[role][name] = group["group_id"]
            ids.append(group["group_id"])
        stages.append(dict(stage=stage, background=role, purpose=block["purpose"],
            variants=list(block["variants"]), group_ids=ids,
            introduced_group_ids=introduced, shared_group_ids=shared,
            baseline_sha256=canonical_hash(baseline),
            background_sha256=canonical_hash(backgrounds[role]), execution_status="not_armed"))
    require(all(set(bindings[role]) == set(rows) for role, rows in resolved.items()),
        "Some configurations are not scheduled")
    require(len(groups) == (28 if centered else 27), "Common-anchor unique configuration count changed")

    # Separate compiler calls prevent its every-nonanchor policy creating Q-minus-B.
    analyses = {}
    for role, rows in resolved.items():
        fragment = deepcopy(study)
        fragment["variants"] = {name: deepcopy(study["variants"][name]) for name in rows}
        fragment["blocks"] = {name: deepcopy(block) for name, block in study["blocks"].items()
            if block_roles[name] == role}
        fragment["execution_order"] = [name for name in order if block_roles[name] == role]
        fragment.pop("planning_totals_before_reuse", None)
        policy = fragment["analysis_policy"]
        for key in ("factorials", "additional_pairs"):
            policy[key] = [spec for spec in policy[key] if spec["block"] in fragment["blocks"]]
        analysis = compile_contrasts(fragment, rows, [(name, 42) for name in rows])
        for contrast in analysis["contrasts"]:
            contrast["id"] = role + ":" + contrast["id"]
            contrast["background"] = role
            for term in contrast["terms"]:
                term["group_id"] = bindings[role][term["variant"]]
                term["parameters_sha256"] = rows[term["variant"]]["parameters_sha256"]
        analysis.update(background=role, background_sha256=canonical_hash(backgrounds[role]),
            scope="random_placement_diagnostic_not_centered_method_attribution" if role == "Q"
                else "conditional_common_anchor_exploration")
        analyses[role] = analysis
    return dict(schema_version=1, record_type="vcsf_common_anchor_catalogue",
        status="prepared_unarmed", runner_armed=False, scientific_acceptance=False,
        model_calls=0, AP_evaluations=0, automatic_promotion=False, independent_confirmation=False,
        anchor_acceptance_status="unassessed", reuse_status="unassessed_do_not_subtract",
        baseline=dict(role="B", parameters=deepcopy(baseline), parameters_sha256=canonical_hash(baseline)),
        backgrounds={role: dict(role=role, parameters=deepcopy(parameters),
            parameters_sha256=canonical_hash(parameters), anchor_group_id=bindings[role]["full"])
            for role, parameters in backgrounds.items()},
        groups=groups, stages=stages, variant_group_ids=bindings, analyses_by_background=analyses,
        study_sha256=canonical_hash(study), protocol_sha256=canonical_hash(protocol),
        budget_sha256=canonical_hash(budget),
        totals_before_qualified_reuse=dict(unique_variants=len(groups), source_groups=len(groups),
            generated_image_instances=len(groups) * protocol["images"],
            target_evaluations=len(groups) * len(targets)),
        remaining_gates=["complete_eight_operator_acceptance_and_explicit_anchor_decision",
            "input_and_implementation_binding", "historical_reuse_qualification",
            "server_ODA_control_and_budget_validation", "block_execution_and_independent_acceptance",
            "explicit_execution_admission"])
