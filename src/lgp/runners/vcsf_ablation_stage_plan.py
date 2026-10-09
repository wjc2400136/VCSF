"""Stage-local, unarmed ablation preparation; never execute an attack or infer reuse.

Endpoint records contain normalized ``parameters`` and a separately hashed
``original`` identity. ``resolve_scale_endpoint`` makes synthetic originals by
default; a real evidence adapter supplies reconstructed originals instead.
The bound adapter owns evidence acceptance, not this compiler.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import csv
import inspect
from pathlib import Path
import re

from ..attacks.factory import ATTACK_TYPES
from ..attacks.vcsf_ablation_isolated import VCSFAblationCandidate, VCSFAblationConfig
from ..attacks.vcsf_research_isolated import VCSFResearchConfig
from ..attacks.vcsf_scale_isolated import VCSFScaleConfig
from ..io import atomic_json, file_digest
from .vcsf_layered_contrasts import compile_contrasts
from .vcsf_research_plan import canonical_hash, require


PROTOCOL = "vcsf_single_source_ablation_research"
SCALE_PROTOCOL = "vcsf_single_source_scale_research"
RECORD_TYPE = "vcsf_ablation_stage_preparation"
SAME_AS_WORKING_BASELINE = "same_as_working_baseline"
_SCALE_FIELDS = {"scale_min", "scale_max"}
_CORE_FIELDS = _SCALE_FIELDS | {"surface", "initialization"}
_SOURCE_FILES = (
    "src/lgp/runners/vcsf_ablation_stage_plan.py",
    "src/lgp/runners/vcsf_stage_evidence.py",
    "experiments/prepare_vcsf_single_source_ablation_stage.py",
    "src/lgp/runners/vcsf_layered_contrasts.py",
    "src/lgp/runners/vcsf_research_plan.py",
    "src/lgp/attacks/vcsf_ablation_isolated.py",
    "src/lgp/attacks/vcsf_scale_isolated.py",
    "src/lgp/attacks/vcsf_research_isolated.py",
    "src/lgp/attacks/vcsf_final_candidate.py",
    "src/lgp/attacks/common.py", "src/lgp/attacks/base.py",
    "src/lgp/registry.py", "src/lgp/io.py",
    "configs/models.yaml", "configs/compatibility.yaml",
    "configs/experiments/ablations.yaml", "configs/experiments/protocols.yaml",
    "configs/experiments/budgets.yaml",
)


def _hashes(value, label):
    require(isinstance(value, Mapping) and bool(value), label + " must contain artifact hashes")
    require(all(isinstance(k, str) and k and isinstance(v, str)
        and re.fullmatch(r"[0-9a-f]{64}", v) for k, v in value.items()),
        label + " contains an invalid SHA-256")
    return deepcopy(dict(value))


def _normal_parameters(raw, config_type=VCSFAblationConfig, complete=False):
    require(isinstance(raw, Mapping), "Endpoint parameters must be a mapping")
    config = config_type.from_mapping(deepcopy(dict(raw)))
    config.validate()
    result = asdict(config)
    require(not complete or set(raw) == set(result), "Endpoint parameters must be complete")
    return result


def _original(raw, parameters, synthetic, parameter_projection=None):
    require(isinstance(raw, Mapping), "Missing original endpoint provenance")
    result = deepcopy(dict(raw))
    for key in ("protocol", "study", "variant", "run_id"):
        require(isinstance(result.get(key), str) and result[key], "Missing original " + key)
    require(type(result.get("group_id")) is int and result["group_id"] > 0,
        "Original group_id must be an explicit positive integer")
    implementation = result.get("implementation")
    require(isinstance(implementation, Mapping) and all(
        isinstance(implementation.get(k), str) and implementation[k]
        for k in ("implementation", "class_name", "config_class_name")),
        "Original executable and config-class identity must be explicit")
    result["source_sha256"] = _hashes(result.get("source_sha256"), "Original source inventory")
    require(implementation["implementation"] in result["source_sha256"],
        "Original source inventory omits its executable")
    original_type = VCSFResearchConfig if implementation["config_class_name"] == "VCSFResearchConfig" else VCSFAblationConfig
    original_parameters = _normal_parameters(result.get("parameters"), config_type=original_type, complete=True)
    require(result.get("parameters_sha256") == canonical_hash(original_parameters),
        "Spoofed original parameters hash")
    if original_parameters != parameters:
        require(synthetic is False and isinstance(parameter_projection, Mapping)
            and parameter_projection.get("status") == "verified_original_anchor_parameter_projection_for_preparation"
            and parameter_projection.get("original_identity_sha256") == canonical_hash(result)
            and parameter_projection.get("original_parameters_sha256") == result["parameters_sha256"]
            and canonical_hash(parameter_projection.get("projected_parameters")) == canonical_hash(parameters)
            and parameter_projection.get("projected_parameters_sha256") == canonical_hash(parameters),
            "Original/current endpoint parameters differ without a reconstructed original anchor projection")
        defaults = parameter_projection.get("explicit_historical_geometry_defaults")
        require(defaults == dict(image_interpolation="bilinear", image_padding="reflect", placement="random")
            and not (set(defaults) & set(original_parameters))
            and dict(original_parameters, **defaults) == parameters
            and parameter_projection.get("formal_reuse_qualified") is False,
            "Historical projection changed more than the original explicit geometry defaults")
    else:
        require(parameter_projection is None, "An unchanged original must not acquire a parameter projection")
    require(result.get("synthetic") is synthetic, "Original synthetic provenance does not match preparation mode")
    result["parameters"] = original_parameters
    return result


def resolve_scale_endpoint(registry, variant, provenance=None, parameter_projection=None):
    """Resolve one canonical registered scale cell, never either full scale planner.

    ``provenance`` is an original record with protocol/study/variant/run_id,
    group_id, parameters/hash, implementation (including config_class_name),
    source_sha256 and synthetic. It is descriptive here, not independently
    accepted evidence. Bound mode must still call the real evidence verifier.
    Identity aliases such as B_h0 are deliberately not canonical endpoints.
    """
    registry._validate()
    protocol = registry.protocols[SCALE_PROTOCOL]
    study = registry.ablation_studies[protocol["ablation_study"]]
    base = registry.ablation_studies[study["base_study"]]
    require(isinstance(variant, str), "Scale variant must be an explicit registered key")
    rows = study["row_order"]
    widths = [Fraction(str(v)) for v in study["half_widths"]]
    require(rows == list(study["variants"]) and len(rows) == len(set(rows)), "Invalid scale row order")
    require(widths and widths == sorted(set(widths)) and widths[0] == 0, "Invalid rational scale widths")
    matches = [(row, i, width) for row in rows for i, width in enumerate(widths)
        if variant == "{}_h{}".format(row, i) and (width > 0 or row == rows[0])]
    require(len(matches) == 1, "Unknown scale endpoint or unbound identity alias: " + variant)
    row, index, width = matches[0]
    parameters = dict(base["defaults"], **base["variants"][study["base_variant"]]["parameters"])
    require(study["base_variant"] == "levels_two" and parameters["levels_per_stage"] == 2,
        "Scale baseline must preserve the registered four-level deletion")
    parameters.update(study["variants"][row]["parameters"])
    parameters.update(scale_min=float(1 - width), scale_max=float(1 + width))
    parameters = _normal_parameters(parameters, VCSFScaleConfig)
    scheduled = []
    for stage in study["stage_order"]:
        for value in study["stages"][stage]["half_widths"]:
            width_index = widths.index(Fraction(str(value)))
            for scheduled_row in study["stages"][stage]["rows"]:
                scheduled.append("{}_h{}".format(rows[0] if width_index == 0 else scheduled_row, width_index))
    require(scheduled.count(variant) == 1, "Scale endpoint has missing or duplicate schedule membership")
    if provenance is None:
        implementation = deepcopy(protocol["isolated_candidate"])
        files = (implementation["implementation"], "src/lgp/attacks/vcsf_research_isolated.py",
            "src/lgp/attacks/vcsf_final_candidate.py", "src/lgp/attacks/common.py")
        provenance = dict(protocol=SCALE_PROTOCOL, study=protocol["ablation_study"],
            variant=variant, run_id="synthetic_fixture_no_original_run", group_id=scheduled.index(variant) + 1,
            implementation=implementation, source_sha256={p: file_digest(registry.root / p) for p in files},
            parameters=deepcopy(parameters), parameters_sha256=canonical_hash(parameters), synthetic=True)
    original = _original(provenance, parameters, provenance.get("synthetic") is True, parameter_projection)
    endpoint = dict(variant=variant, row=row, half_width=str(width), parameters=parameters,
        parameters_sha256=canonical_hash(parameters), original=original, original_sha256=canonical_hash(original))
    if parameter_projection is not None:
        endpoint["original_parameter_projection"] = deepcopy(parameter_projection)
    return endpoint


def _endpoint(registry, value, synthetic):
    if isinstance(value, str):
        require(synthetic, "Bound endpoints must be reconstructed records, not fixture selectors")
        return resolve_scale_endpoint(registry, value)
    require(isinstance(value, Mapping), "Missing explicit working/comparator endpoint")
    result = deepcopy(dict(value))
    require(isinstance(result.get("variant"), str) and result["variant"], "Endpoint variant is required")
    parameters = _normal_parameters(result.get("parameters"), complete=True)
    require(result.get("parameters_sha256") == canonical_hash(parameters), "Spoofed endpoint parameters hash")
    parameter_projection = result.get("original_parameter_projection")
    original = _original(result.get("original"), parameters, synthetic, parameter_projection)
    require(result.get("original_sha256") == canonical_hash(original), "Spoofed original identity hash")
    result.update(parameters=parameters, original=original)
    scale_study = registry.ablation_studies[registry.protocols[SCALE_PROTOCOL]["ablation_study"]]
    is_scale_key = any(result["variant"].startswith(row + "_h") for row in scale_study["row_order"])
    if is_scale_key:
        registered = resolve_scale_endpoint(registry, result["variant"], provenance=original,
            parameter_projection=parameter_projection)
        require(registered["parameters"] == parameters, "Endpoint differs from its registered scale cell")
        for key in ("row", "half_width"):
            require(key not in result or result[key] == registered[key], "Spoofed scale endpoint " + key)
            result[key] = registered[key]
    return result


def _same_reference(value):
    return value == SAME_AS_WORKING_BASELINE or (isinstance(value, Mapping)
        and value.get("reference") == SAME_AS_WORKING_BASELINE)


def _bindings(registry, stage, working_baseline, nonzero_comparator, mode, evidence):
    proof = None
    if mode == "bound_preparation":
        # Only this main-owned adapter may reconstruct real receipt/source/decision evidence.
        from .vcsf_stage_evidence import verify_stage_evidence
        proof = verify_stage_evidence(registry, evidence, working_baseline, nonzero_comparator)
        require(isinstance(proof, Mapping)
            and proof.get("status") == "verified_complete_scale_evidence_for_unarmed_preparation",
            "Evidence verifier must return its reconstructed complete-scale proof")
        proof = deepcopy(dict(proof))
        require(isinstance(proof.get("proof"), Mapping) and isinstance(proof.get("references"), (Mapping, list)),
            "Evidence verifier omitted reconstructed proof/references")
        _hashes({k: proof.get(k) for k in ("proof_sha256", "selection_decision_sha256")},
            "Reconstructed decision/proof hashes")
        working = _endpoint(registry, working_baseline, False)
        comparator = deepcopy(working) if _same_reference(nonzero_comparator) else _endpoint(
            registry, nonzero_comparator, False)
    else:
        require(evidence is None, "Fixture mode does not consume or simulate accepted evidence")
        working = _endpoint(registry, working_baseline, True)
        comparator = deepcopy(working) if _same_reference(nonzero_comparator) else _endpoint(
            registry, nonzero_comparator, True)
    if _same_reference(nonzero_comparator) and isinstance(nonzero_comparator, Mapping):
        require(set(nonzero_comparator) <= {"reference", "retention_reason"},
            "same_as_working_baseline must not override original identity")
    require(nonzero_comparator is not None, "An explicit nonzero comparator is required")
    require(comparator["parameters"]["scale_min"] < 1 < comparator["parameters"]["scale_max"],
        "Comparator must have a registered positive scale width")
    if stage == "core":
        registered = resolve_scale_endpoint(registry, comparator["variant"], comparator["original"],
            comparator.get("original_parameter_projection"))
        require(registered["parameters"] == comparator["parameters"],
            "Core nonzero comparator must be a registered scale endpoint")
    identity = working["parameters"]["scale_min"] == working["parameters"]["scale_max"] == 1
    projected = deepcopy(working["parameters"])
    alias = None
    if not identity:
        require(working["parameters"] == comparator["parameters"]
            and working["parameters_sha256"] == comparator["parameters_sha256"]
            and working["original"] == comparator["original"],
            "Nonzero W/N must have the same complete original configuration identity")
    else:
        require(not _same_reference(nonzero_comparator), "Identity W requires a separately bound nonzero N")
        reason = comparator.get("retention_reason")
        if isinstance(nonzero_comparator, Mapping):
            reason = nonzero_comparator.get("retention_reason", reason)
        require(isinstance(reason, str) and bool(reason.strip()), "Identity comparator requires a retention reason")
        comparator["retention_reason"] = reason
        desired = dict(comparator["parameters"], scale_min=1.0, scale_max=1.0)
        if projected != desired:
            alias = None if proof is None else proof.get("identity_context_projection")
            require(isinstance(alias, Mapping), "Non-scale W/N mismatch requires a proven identity-context bridge")
            require(alias.get("original_sha256") == working["original_sha256"]
                and alias.get("original_parameters_sha256") == working["parameters_sha256"]
                and alias.get("projected_parameters") == desired
                and alias.get("projected_parameters_sha256") == canonical_hash(desired),
                "Identity bridge does not bind the original and projected endpoints")
            changes = {k: dict(original=projected[k], projected=desired[k]) for k in projected if projected[k] != desired[k]}
            require(alias.get("transformation") == changes, "Identity bridge transformation is not exact")
            artifacts = _hashes(alias.get("artifact_sha256"), "Identity bridge evidence")
            require({"geometry", "rng", "forward", "backward", "acceptance"} <= set(artifacts),
                "Identity bridge lacks qualified geometry/RNG/forward/backward evidence")
            projected = _normal_parameters(desired, complete=True)
    return working, comparator, projected, identity, deepcopy(alias), proof


def _projection(study, stage, anchor):
    require(stage in study["blocks"], "Unknown selected ablation stage")
    block = study["blocks"][stage]
    policy = study["analysis_policy"]
    factorials = [deepcopy(f) for f in policy["factorials"] if f["block"] == stage]
    pairs = [deepcopy(p) for p in policy["additional_pairs"] if p["block"] == stage]
    names = list(block["variants"])
    reasons = {name: ["selected_block"] for name in names}
    for name, reason in [(anchor, "explicit_working_anchor")] + [
            (f[k], "factorial_" + k) for f in factorials for k in ("low", "high")] + [
            (p[k], "additional_pair_" + k) for p in pairs for k in ("candidate", "reference")]:
        require(name in study["variants"], "Missing selected-stage dependency: " + name)
        if name not in names:
            names.append(name)
        reasons.setdefault(name, []).append(reason)
    projected = deepcopy(study)
    projected["blocks"] = {stage: deepcopy(block)}
    projected["execution_order"] = [stage]
    projected.pop("planning_totals_before_reuse", None)
    projected["variants"] = {name: deepcopy(study["variants"][name]) for name in names}
    projected["analysis_policy"].update(anchor=anchor, factorials=factorials, additional_pairs=pairs)
    if stage == "core":
        require(len(names) == len(block["variants"]) == 8 and len(factorials) == 1 and not pairs,
            "Core requires exactly eight cells and no ninth anchor or extra comparisons")
        require(policy["anchor"] == "full" and factorials[0]["high"] == "full"
            and factorials[0]["axes"] == {"scale": ["scale_min", "scale_max"],
                "stage": ["surface"], "initializer_schedule": ["initialization"]}
            and set(block["allowed_fields"]) == _CORE_FIELDS,
            "Core factorial axes/anchor declaration changed")
    return projected, names, reasons


def compile_stage_plan(registry, stage, working_baseline, nonzero_comparator=None,
        mode="fixture", evidence=None, max_images=None, devices=("cuda:0",)):
    """Compile only selected controls/dependencies; both modes remain unarmed.

    The main-owned verifier reconstructs and checks the supplied complete
    endpoint originals. Its proof/references/decision hashes are recorded intact;
    this compiler does not infer endpoints from an acceptance boolean or count.
    The current real adapter qualifies complete scale evidence only, not a future
    prior-stage acceptance chain. Fixture originals may describe prior stages.
    """
    registry._validate()
    require(mode in ("fixture", "bound_preparation"), "Unknown preparation mode")
    require(max_images is None or (type(max_images) is int and 0 < max_images < 5000),
        "Diagnostic image limit must be an integer from 1 through 4999")
    require(isinstance(devices, (tuple, list)) and 1 <= len(devices) <= 2
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices), "Planning requires one or two distinct explicit CUDA ordinals")
    protocol = registry.protocols[PROTOCOL]
    budget = registry.budget_profiles[protocol["budget_profile"]]
    study = registry.ablation_studies[protocol["ablation_study"]]
    require(stage in study["blocks"], "Unknown selected ablation stage")
    require(study["method"] not in registry.attacks and study["method"] not in ATTACK_TYPES,
        "Stage preparation must not alter the global method registry")
    implementation = protocol["isolated_candidate"]
    require(implementation["implementation"] == VCSFAblationCandidate.implementation_path
        and implementation["class_name"] == VCSFAblationCandidate.__name__
        and implementation["config_class_name"] == VCSFAblationConfig.__name__
        and Path(inspect.getfile(VCSFAblationCandidate)).resolve()
            == (registry.root / implementation["implementation"]).resolve(),
        "Current stage executable/config class does not match the declared implementation")
    working, comparator, working_parameters, identity, alias, proof = _bindings(
        registry, stage, working_baseline, nonzero_comparator, mode, evidence)
    baseline = comparator["parameters"] if stage == "core" else working_parameters
    working_variant = "cross_identity_detector" if stage == "core" and identity else "full"
    projection, names, reasons = _projection(study, stage, working_variant)
    block = projection["blocks"][stage]
    allowed = block.get("allowed_fields", [])
    require(allowed and len(allowed) == len(set(allowed)) and set(allowed) <= set(baseline),
        "Invalid selected-stage allowed fields")
    resolved, seen = {}, set()
    for name in names:
        raw = dict(baseline, **projection["variants"][name]["parameters"])
        # Scale endpoints use rational-width floats; YAML identity integers must
        # not create a different hash for the very same selected endpoint.
        if stage == "core":
            for key in _SCALE_FIELDS:
                raw[key] = float(Fraction(str(raw[key])))
        parameters = _normal_parameters(raw)
        require(all(parameters[k] == baseline[k] for k in parameters if k not in allowed),
            "Non-factor parameter confound: " + name)
        require(parameters["iterations"] in budget["allowed_iterations_by_initialization"][parameters["initialization"]],
            "Configuration is outside the registered initializer budget")
        digest = canonical_hash(parameters)
        require(digest not in seen, "Duplicate selected-stage configuration: " + name)
        seen.add(digest)
        risk = int(parameters["initialization"] == "detector")
        resolved[name] = dict(parameters=parameters, parameters_sha256=digest,
            logical_gradients=parameters["iterations"], risk_updates=risk,
            feature_updates=parameters["iterations"] - risk,
            feature_terms=parameters["levels_per_stage"] * (2 if parameters["surface"] == "cross_stage" else 1),
            random_pixel_initialization=parameters["initialization"] == "random_sign",
            reference_forwards_not_free=True)
    matches = [n for n in names if resolved[n]["parameters"] == working_parameters]
    require(matches == [working_variant], "W is not the expected unique selected endpoint")
    require(resolved[working_variant]["parameters_sha256"] == canonical_hash(working_parameters),
        "Selected W cell does not preserve the complete normalized endpoint hash")
    if stage == "core":
        require(resolved["full"]["parameters"] == comparator["parameters"], "Core high cell differs from N")
    if stage == "matched_terms":
        controls = [resolved[n] for n in block["variants"]]
        require(all(c["feature_terms"] == block["feature_terms"] for c in controls),
            "Matched-stage controls do not have equal registered term counts")
        require(all(all(c["parameters"][k] == controls[0]["parameters"][k]
            for k in baseline if k not in ("surface", "levels_per_stage")) for c in controls),
            "Matched-stage controls differ outside stage and layer count")
    if stage == "initialization_specificity":
        require(set(block["schedules"]) == set(block["variants"]), "Initializer schedule coverage changed")
        for name, schedule in block["schedules"].items():
            require(set(schedule) == {"risk_updates", "feature_updates"}
                and all(resolved[name][k] == v for k, v in schedule.items()),
                "Initializer risk/feature schedule mismatch: " + name)
    targets, sources = registry.target_ids(), protocol["sources"]
    require(len(targets) == 16 and len(set(targets)) == 16 and sources == registry.source_ids()[:1],
        "Stage scope must preserve one canonical source and all sixteen targets")
    images = protocol["images"] if max_images is None else max_images
    groups = [dict(group_id=i + 1, variant=name, source=sources[0], seed=seed,
        images=images, targets=list(targets), blackbox_targets=[t for t in targets if t != sources[0]],
        source_matched_target=sources[0], selected_control=name in block["variants"],
        dependency_reasons=reasons[name], reuse_status="unassessed_do_not_subtract",
        execution_status="not_armed", **resolved[name])
        for i, (name, seed) in enumerate((n, s) for n in names for s in block["seeds"])]
    analysis = compile_contrasts(projection, resolved, [(g["variant"], g["seed"]) for g in groups])
    analysis.update(mode="point_estimate_only", uncertainty=dict(status="not_requested_not_computed",
        bootstrap_required=False, significance_claim=False), all_twelve_target_metrics_required=True)
    if stage == "core":
        require(len(groups) == 8 and analysis["family_size"] == 32, "Core must have eight cells and 32 contrasts")
    source_hashes = {p: file_digest(registry.root / p) for p in sorted(_SOURCE_FILES)}
    authority_hashes = {}
    for key in ("owner_decision", "design_contract"):
        raw = Path(protocol[key])
        path = (registry.root / raw).resolve()
        require(not raw.is_absolute() and registry.root.resolve() in path.parents
            and path.suffix == ".md" and path.is_file(), "Invalid stage authority document")
        authority_hashes[raw.as_posix()] = file_digest(path)
    current = dict(implementation=deepcopy(implementation), source_sha256=source_hashes)
    shared = {k: v for k, v in baseline.items() if k not in allowed}
    plan = dict(schema_version=1, record_type=RECORD_TYPE,
        status="prepared_fixture_not_executable" if mode == "fixture" else "prepared_bound_stage_not_executable",
        mode=mode, synthetic=mode == "fixture", protocol=PROTOCOL, study=protocol["ablation_study"],
        selected_stage=stage, runner_armed=False, model_calls=0, AP_evaluations=0,
        scientific_acceptance=False, independent_confirmation=False, automatic_promotion=False,
        old_run_restart=False, source_sha256=source_hashes, current_stage_identity=current,
        current_stage_identity_sha256=canonical_hash(current), original_source_inventories=dict(
            working_baseline=working["original"]["source_sha256"],
            nonzero_comparator=comparator["original"]["source_sha256"]),
        working_baseline=working, nonzero_comparator=comparator, working_is_identity=identity,
        working_stage_parameters=working_parameters, working_stage_parameters_sha256=canonical_hash(working_parameters),
        selected_working_variant=working_variant, factorial_high_variant="full" if stage == "core" else None,
        comparator_role="retained_nonzero_rejection_deletion_comparator" if identity else "same_selected_identity",
        identity_context_projection=alias, evidence_proof=proof,
        evidence_proof_sha256=None if proof is None else canonical_hash(proof),
        shared_parameters=shared, shared_parameters_sha256=canonical_hash(shared),
        stage_dependency_keys=list(names), dependency_only_keys=[n for n in names if n not in block["variants"]],
        dependency_reasons=reasons, sources=list(sources), targets=list(targets),
        dataset=protocol["dataset"], split=protocol["split"], seeds=list(block["seeds"]),
        max_images=max_images, diagnostic_only=max_images is not None,
        groups=groups, analysis=analysis, study_sha256=canonical_hash(study),
        selected_fragment=projection, selected_fragment_sha256=canonical_hash(projection),
        protocol_sha256=canonical_hash(protocol), budget_profile=protocol["budget_profile"],
        budget_sha256=canonical_hash(budget), authority_sha256=authority_hashes,
        totals_before_qualified_reuse=dict(unique_variants=len(names), source_groups=len(groups),
            generated_image_instances=len(groups) * images, target_evaluations=len(groups) * len(targets)),
        requested_group_ids=[g["group_id"] for g in groups], requested_image_instances=len(groups) * images,
        remaining_gates=["real_complete_scale_evidence_and_selection" if mode == "fixture" else "fresh_stage_admission",
            "new_control_server_and_cost_contract_validation", "historical_reuse_independent_adjudication",
            "executor_lifecycle_and_exclusive_device_validation", "bound_prediction_index_and_analysis_family_lock"])
    # Devices only partition complete jobs; neither availability nor execution is implied.
    plan["science_sha256"] = canonical_hash(plan)
    plan.update(requested_devices=list(devices), device_availability_checked=False,
        lane_plan=[dict(device=d, group_ids=[g["group_id"] for g in groups[i::len(devices)]])
            for i, d in enumerate(devices)])
    return plan


def write_stage_plan(output, plan):
    """Write a fresh immutable preparation root, never merge or overwrite it."""
    require(plan.get("record_type") == RECORD_TYPE and plan.get("runner_armed") is False,
        "Only an unarmed stage preparation may be written")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    columns = ("group_id", "variant", "source", "seed", "images", "selected_control",
        "feature_terms", "risk_updates", "feature_updates", "parameters_sha256", "reuse_status", "execution_status")
    with (output / "groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for group in plan["groups"]:
            writer.writerow([group[k] for k in columns])
    with (output / "contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("contrast_id", "kind", "variant", "seed", "coefficient"))
        for contrast in plan["analysis"]["contrasts"]:
            for term in contrast["terms"]:
                writer.writerow((contrast["id"], contrast["kind"], term["variant"], term["seed"], term["coefficient"]))
    manifest = dict(record_type=RECORD_TYPE, files={name: file_digest(output / name)
        for name in ("plan.json", "groups.csv", "contrasts.csv")}, runner_armed=False,
        synthetic=plan["synthetic"], model_calls=0, AP_evaluations=0)
    atomic_json(output / "artifact_manifest.json", manifest)
    return manifest
