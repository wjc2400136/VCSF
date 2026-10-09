"""Prospective ablation cost contracts and unaccepted saved-trace recounts.

Counts describe source-visible operations, not physical detector equivalents.
This module cannot accept a run, calibrate hardware cost, or launch a model.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import statistics

from ..attacks.vcsf_ablation_isolated import VCSFAblationConfig
from ..io import atomic_json, file_digest
from ..runners.vcsf_ablation_stage_plan import compile_stage_plan
from ..runners.vcsf_research_plan import canonical_hash, require


RECORD_TYPE = "vcsf_ablation_cost_contract"
# These hashes identify interpreted code, not a second parameter registry.
INTERPRETED_SOURCE = {
    "src/lgp/attacks/vcsf_ablation_isolated.py": "3200ee8d6f3257f076597c65525fbb3b6d1ec506f02236378f290248f1b1d1de",
    "src/lgp/attacks/vcsf_scale_isolated.py": "65827ce3ed4eb1d16700a828648ed9d6595f5ff6d92103408343ab56df5047ef",
    "src/lgp/attacks/vcsf_research_isolated.py": "b78fbbddc1638f4f74ddb44771c131a6f5301272bc5d75110c700828f40099b0",
    "src/lgp/attacks/vcsf_final_candidate.py": "99423c6892db7afc172dff0eb375970a603b44ed9885b0c04f71faa845da2c4a",
    "src/lgp/attacks/common.py": "850fbdfcb8908064c0380e5a30f66cec1c62814a287c26f7f388a980811709d3",
    "src/lgp/attacks/base.py": "9288b61e8168b1738d84d3e4c6860c713a340b5cfaf45b0a984694dc4fd3259e",
    "src/lgp/adapters/openmmlab.py": "553ab418de02fa4c9e2c5920e066b4be248225af4b68b660e17e2866998c4348",
    "src/lgp/runners/attack.py": "6e4e73162394c23fff1c3d1cd35f5c3453a93d6d31a3c781bd30e71e4aa5e96e",
}
COUNT_FIELDS = (
    "logical_gradient_updates", "detector_score_backward_updates", "feature_partial_backward_updates",
    "paired_feature_extractor_calls", "detached_clean_reference_image_slots", "differentiable_feature_image_slots",
    "image_geometry_calls", "view_mask_geometry_calls", "feature_objective_terms",
    "feature_mask_conversion_calls", "feature_mask_resize_calls", "post_nms_forward_attempts",
    "pre_nms_fallback_forward_calls", "explicit_auxiliary_forward_calls", "explicit_auxiliary_backward_calls",
    "random_sign_initialization_tensors", "clean_energy_terms", "feature_aggregation_calls",
)
UNMEASURED = ("runtime_seconds", "peak_cuda_memory_mb", "physical_full_detector_backward_equivalents",
    "measured_kernel_flops", "per_image_peak_memory")
QUALIFIERS = dict(
    count_basis="source_visible_calls_and_objective_terms_not_hardware_work",
    reference_basis="one_detached_clean_input_slot_in_each_paired_batch_not_a_free_or_separate_no_grad_forward",
    extractor_basis="complete_adapter_extract_features_called_before_objective_surface_selection",
    backward_basis="one_loss_backward_per_update_not_one_backward_per_feature_term",
    detached_reference_limit="detach_does_not_establish_zero_physical_backward_work_for_the_clean_batch_slot",
    geometry_basis="view_image_and_view_mask_calls_separate_from_per_term_model_mask_conversion_and_resize",
    time_basis="saved_per_image_generation_timer_not_randomized_isolated_calibration",
    memory_basis="original_run_peak_including_resident_model_not_per_image_or_attack_only",
    shape_pairing_basis="selected_objective_surface_signatures_aligned_by_feature_update_index_only_when_counts_match",
)


def _parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate JSON key")
            value[key] = item
        return value
    def reject(value):
        raise ValueError("Non-finite JSON value: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def read_stage_plan(path, sha256):
    path = Path(path)
    require(isinstance(sha256, str) and re.fullmatch(r"[0-9a-f]{64}", sha256), "Invalid external stage-plan hash")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink stage plan is not accepted")
    require(file_digest(path) == sha256, "Stage-plan bytes differ from their external hash")
    value = _parse(path.read_bytes())
    require(isinstance(value, dict), "Stage plan must be a JSON object")
    require(file_digest(path) == sha256, "Stage plan changed while being read")
    return value


def _parameters(raw):
    require(isinstance(raw, dict), "Complete ablation parameters are required")
    config = VCSFAblationConfig.from_mapping(raw)
    config.validate()
    normalized = asdict(config)
    require(set(raw) == set(normalized) and canonical_hash(raw) == canonical_hash(normalized),
        "Cost parameters are incomplete or not normalized")
    return normalized


def _counts(parameters, pre_nms=None):
    risk = int(parameters["initialization"] == "detector")
    features = parameters["iterations"] - risk
    terms = parameters["levels_per_stage"] * (2 if parameters["surface"] == "cross_stage" else 1)
    if risk == 0:
        require(pre_nms in (None, 0), "No-initializer trace cannot have a detector fallback")
        pre_nms = 0
    require(pre_nms is None or type(pre_nms) is int and 0 <= pre_nms <= risk, "Invalid fallback count")
    return dict(logical_gradient_updates=parameters["iterations"], detector_score_backward_updates=risk,
        feature_partial_backward_updates=features, paired_feature_extractor_calls=features,
        detached_clean_reference_image_slots=features, differentiable_feature_image_slots=features,
        image_geometry_calls=2 * features,
        view_mask_geometry_calls=features * (1 if parameters["correspondence"] == "shared" else 2),
        feature_objective_terms=features * terms, feature_mask_conversion_calls=features * terms,
        feature_mask_resize_calls=features * terms, post_nms_forward_attempts=risk,
        pre_nms_fallback_forward_calls=pre_nms, explicit_auxiliary_forward_calls=0,
        explicit_auxiliary_backward_calls=0,
        random_sign_initialization_tensors=int(parameters["initialization"] == "random_sign"),
        clean_energy_terms=features * terms if parameters["spatial_weight"] == "energy" else 0,
        feature_aggregation_calls=features)


def _fixture_endpoint(endpoint):
    value = deepcopy(endpoint)
    require(value.get("original_sha256") == canonical_hash(value["original"]), "Changed original endpoint identity")
    value["original"]["synthetic"] = True
    value["original_sha256"] = canonical_hash(value["original"])
    return value


def _validate_stage(registry, plan):
    require(plan.get("record_type") == "vcsf_ablation_stage_preparation"
        and type(plan.get("schema_version")) is int and plan["schema_version"] == 1,
        "Not a stage-only preparation record")
    mode = plan.get("mode")
    require(mode in ("fixture", "bound_preparation")
        and plan.get("status") == ("prepared_fixture_not_executable" if mode == "fixture"
            else "prepared_bound_stage_not_executable") and plan.get("synthetic") is (mode == "fixture"),
        "Preparation mode/status mismatch")
    for key in ("runner_armed", "scientific_acceptance", "independent_confirmation", "automatic_promotion", "old_run_restart"):
        require(plan.get(key) is False, "Preparation must not claim " + key)
    for key in ("model_calls", "AP_evaluations"):
        require(type(plan.get(key)) is int and plan[key] == 0, "Preparation has an executed-work claim")
    content = {k: v for k, v in plan.items() if k not in
        ("science_sha256", "requested_devices", "device_availability_checked", "lane_plan")}
    require(canonical_hash(content) == plan.get("science_sha256"), "Stage scientific content hash differs")
    require(plan.get("identity_context_projection") is None,
        "Projected identity needs a separately qualified cost bridge")
    for key in ("working_baseline", "nonzero_comparator"):
        require(plan[key]["original"].get("synthetic") is (mode == "fixture"),
            "Original endpoint mode disagrees with its preparation")
    proof = plan.get("evidence_proof")
    if mode == "fixture":
        require(proof is None and plan.get("evidence_proof_sha256") is None, "Fixture carries a real-evidence claim")
    else:
        require(isinstance(proof, dict) and proof.get("status") == "verified_complete_scale_evidence_for_unarmed_preparation"
            and plan.get("evidence_proof_sha256") == canonical_hash(proof), "Bound preparation proof identity differs")
    # Reconstruct configuration/dependency/contrast content only. This does not
    # re-accept real point evidence or upgrade a fixture into bound preparation.
    reconstructed = compile_stage_plan(registry, plan["selected_stage"],
        _fixture_endpoint(plan["working_baseline"]), _fixture_endpoint(plan["nonzero_comparator"]),
        max_images=plan["max_images"], devices=plan["requested_devices"])
    for key in ("groups", "analysis", "selected_fragment", "shared_parameters", "selected_working_variant",
            "factorial_high_variant", "working_stage_parameters", "stage_dependency_keys", "dependency_only_keys",
            "requested_group_ids", "totals_before_qualified_reuse", "sources", "targets", "dataset", "split", "seeds",
            "source_sha256", "current_stage_identity", "protocol", "study", "protocol_sha256", "study_sha256",
            "budget_sha256", "authority_sha256", "lane_plan", "diagnostic_only"):
        require(canonical_hash(plan.get(key)) == canonical_hash(reconstructed[key]),
            "Stage cost structure differs from registration: " + key)
    for name, digest in INTERPRETED_SOURCE.items():
        path = registry.root / name
        require(not any(p.is_symlink() for p in (path, *path.parents)) and file_digest(path) == digest,
            "Cost interpretation does not cover the current source: " + name)


def compile_cost_contract(registry, stage_plan, devices=None):
    _validate_stage(registry, stage_plan)
    devices = stage_plan["requested_devices"] if devices is None else devices
    require(isinstance(devices, (list, tuple)) and 1 <= len(devices) <= 2
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices), "Select one or two distinct explicit CUDA ordinals")
    groups = []
    for group in stage_plan["groups"]:
        parameters = _parameters(group["parameters"])
        require(canonical_hash(parameters) == group["parameters_sha256"], "Group parameter hash differs")
        counts = _counts(parameters)
        groups.append(dict(group_id=group["group_id"], variant=group["variant"], seed=group["seed"],
            source=group["source"], images=group["images"], parameters_sha256=group["parameters_sha256"],
            parameters=parameters, declared_per_image=counts,
            declared_group_totals={k: None if v is None else v * group["images"] for k, v in counts.items()},
            pre_nms_fallback_per_image_bounds=dict(minimum=0, maximum=counts["detector_score_backward_updates"]),
            unmeasured={key: None for key in UNMEASURED}, observed_trace_rows=0,
            saved_cost_disclosure_accepted=False, reuse_status="unassessed_do_not_subtract"))
    sources = dict(INTERPRETED_SOURCE)
    for name in ("src/lgp/reporting/vcsf_ablation_cost.py", "experiments/prepare_vcsf_ablation_cost_contract.py"):
        sources[name] = file_digest(registry.root / name)
    contract = dict(schema_version=1, record_type=RECORD_TYPE,
        status="prepared_ablation_cost_contract_not_execution_or_disclosure",
        selected_stage=stage_plan["selected_stage"], stage_plan_science_sha256=stage_plan["science_sha256"],
        stage_preparation_mode=stage_plan["mode"], stage_evidence_reverified=False,
        groups=groups, sources=list(stage_plan["sources"]), targets=list(stage_plan["targets"]),
        max_images=stage_plan["max_images"], diagnostic_only=stage_plan["diagnostic_only"],
        analysis_contrast_count=stage_plan["analysis"]["family_size"],
        contrasts=deepcopy(stage_plan["analysis"]["contrasts"]), source_sha256=sources,
        qualifiers=deepcopy(QUALIFIERS), runner_armed=False, model_calls=0, AP_evaluations=0,
        cost_disclosure_accepted=False, physical_cost_calibration_accepted=False,
        scientific_acceptance=False, independent_confirmation=False, bootstrap_required=False,
        timer_includes=["seed_resolution", "attack_and_its_input_device_transfer", "diagnostics",
            "output_and_linf_finiteness_checks", "diagnostic_json_serialization", "final_cuda_synchronize"],
        timer_excludes=["image_decode_and_annotation_lookup", "initial_cuda_synchronize", "budget_threshold_check",
            "png_write_and_hash", "quality_metrics", "run_record_serialization"],
        remaining_gates=["complete_real_stage_binding_and_fresh_execution_admission",
            "real_source_control_and_count_validation", "accepted_original_generation_manifests",
            "paired_time_memory_source_and_worker_binding", "independent_saved_cost_recount_and_export_audit"])
    contract["science_sha256"] = canonical_hash(contract)
    contract.update(requested_devices=list(devices), device_availability_checked=False,
        lane_plan=[dict(device=d, group_ids=[g["group_id"] for g in groups[i::len(devices)]])
            for i, d in enumerate(devices)])
    return contract


def _number(value, label, positive=False, nonnegative=False):
    require(type(value) in (int, float) and math.isfinite(value)
        and (not positive or value > 0) and (not nonnegative or value >= 0), "Invalid " + label)
    return float(value)


def recount_image_trace(row, parameters, seed=42, position=0):
    """Recount one complete trace; caller still owns source/input acceptance."""
    parameters = _parameters(parameters)
    require(type(position) is int and position >= 0 and type(seed) is int, "Invalid trace position/seed")
    require(row.get("status") == "ok" and type(row.get("position")) is int and row["position"] == position
        and type(row.get("image_id")) is int and row["image_id"] > 0
        and type(row.get("attack_seed")) is int and row["attack_seed"] == seed + position, "Trace image/seed/status differs")
    steps = row.get("diagnostics")
    require(isinstance(steps, list) and len(steps) == parameters["iterations"]
        and type(row.get("actual_gradient_evaluations")) is int
        and row["actual_gradient_evaluations"] == len(steps), "Incomplete logical gradient trace")
    for key in ("auxiliary_forward_passes", "auxiliary_backward_passes"):
        require(type(row.get(key)) is int and row[key] == 0, "Explicit auxiliary declaration changed")
    selected = {"neck", "backbone"} if parameters["surface"] == "cross_stage" else {parameters["surface"]}
    terms = len(selected) * parameters["levels_per_stage"]
    signatures, risk_surface = [], None
    for index, step in enumerate(steps):
        risk = parameters["initialization"] == "detector" and index == 0
        require(type(step.get("step")) is int and step["step"] == index
            and step.get("phase") == ("risk_initialization" if risk else "feature"), "Trace schedule differs")
        for key in ("initialization", "surface", "spatial_weight", "aggregation", "support", "correspondence", "gradient_centering"):
            require(type(step.get(key)) is type(parameters[key]) and step[key] == parameters[key], "Trace setting differs: " + key)
        _number(step.get("loss"), "loss")
        _number(step.get("image_gradient_abs_mean"), "image gradient", nonnegative=True)
        linf = _number(step.get("linf_pixel"), "L_inf", nonnegative=True)
        require(linf <= parameters["eps"] * 255.0 + 1e-5, "Trace exceeds its perturbation bound")
        require(step.get("risk_direction_carryover_applied") is False and step.get("uncertainty_gate_active") is False,
            "Trace carries an unregistered optimizer or uncertainty path")
        if risk:
            risk_surface = step.get("risk_surface")
            require(risk_surface in ("post_nms", "pre_nms") and step.get("score_risk_defined") is True,
                "Unavailable detector initializer path")
            require(not step.get("feature_surface_shapes"), "Risk update contains an unexpected feature path")
            continue
        require(step.get("risk_surface") in (None, "not_checked"), "Feature update contains a detector risk path")
        require(type(step.get("feature_surface_level_count")) is int and step["feature_surface_level_count"] == terms,
            "Objective term count differs")
        surface_label = "neck_backbone" if parameters["surface"] == "cross_stage" \
            and parameters["levels_per_stage"] == 3 and parameters["aggregation"] == "log_mean_exp" else parameters["surface"]
        require(step.get("feature_surface") == surface_label, "Feature surface diagnostic alias differs")
        shapes = step.get("feature_surface_shapes")
        require(isinstance(shapes, dict) and set(shapes) == selected, "Selected objective stages differ")
        for shapes_at_stage in shapes.values():
            require(isinstance(shapes_at_stage, (list, tuple)) and len(shapes_at_stage) == parameters["levels_per_stage"],
                "Incomplete selected feature levels")
            for shape in shapes_at_stage:
                require(isinstance(shape, (list, tuple)) and len(shape) == 4
                    and all(type(v) is int and v > 0 for v in shape) and shape[0] == 2,
                    "Feature trace does not preserve paired batch two")
        for key in ("image_interpolation", "placement", "image_padding"):
            require(step.get(key) == parameters[key], "View operator trace differs: " + key)
        require(step.get("mask_interpolation") == "nearest" and step.get("mask_padding") == "zero"
            and step.get("offset_random_draws_preserved") is True, "View-mask/RNG trace differs")
        signatures.append(canonical_hash(shapes))
    counts = _counts(parameters, int(risk_surface == "pre_nms"))
    require(len(signatures) == counts["feature_partial_backward_updates"], "Missing feature-update signatures")
    return dict(status="recounted_trace_not_independent_acceptance", image_id=row["image_id"],
        runtime_seconds=_number(row.get("runtime_seconds"), "saved elapsed time", positive=True),
        counts=counts, risk_surface=risk_surface, feature_signatures=signatures)


def recount_group_traces(rows, group):
    observations = [recount_image_trace(row, group["parameters"], group["seed"], position)
        for position, row in enumerate(rows)]
    ids = [r["image_id"] for r in observations]
    require(type(group["images"]) is int and group["images"] > 0 and len(ids) == group["images"]
        and ids == sorted(set(ids)), "Incomplete or noncanonical group image set")
    require(canonical_hash(group["parameters"]) == group["parameters_sha256"], "Group parameter binding differs")
    return dict(status="recounted_group_traces_not_independent_acceptance", ids=ids,
        times=[r["runtime_seconds"] for r in observations], shapes=[r["feature_signatures"] for r in observations],
        per_image_counts=[r["counts"] for r in observations],
        counts={key: sum(r["counts"][key] for r in observations) for key in COUNT_FIELDS},
        image_ids_sha256=canonical_hash(ids), physical_cost_calibration_accepted=False)


def compare_paired_costs(contrasts, observations):
    """Pair by image ID; no truncation when initializer schedules differ."""
    results = []
    names = set()
    for contrast in contrasts:
        name, terms = contrast["id"], contrast["terms"]
        require(isinstance(name, str) and name not in names, "Duplicate cost contrast identity")
        names.add(name)
        keys = [(t["variant"], t["seed"]) for t in terms]
        require(len(keys) >= 2 and len(keys) == len(set(keys))
            and all(isinstance(v, str) and v and type(s) is int for v, s in keys)
            and all(isinstance(t["coefficient"], str) for t in terms), "Invalid rational cost contrast")
        weights = [Fraction(t["coefficient"]) for t in terms]
        require(all(weights) and sum(weights) == 0, "Cost contrast is not a signed difference")
        rows = [observations[key] for key in keys]
        ids = rows[0]["ids"]
        require(ids and all(type(i) is int and i > 0 for i in ids) and ids == sorted(set(ids)) and all(r["ids"] == ids
            and len(r["times"]) == len(r["shapes"]) == len(r["per_image_counts"]) == len(ids) for r in rows),
            "Cost observations are not completely paired")
        differences = []
        for index in range(len(ids)):
            elapsed = [_number(r["times"][index], "saved elapsed time", positive=True) for r in rows]
            differences.append(float(sum((w * Fraction.from_float(t) for w, t in zip(weights, elapsed)), Fraction(0))))
            for row in rows:
                counts = row["per_image_counts"][index]
                require(set(counts) == set(COUNT_FIELDS) and all(type(v) is int and v >= 0 for v in counts.values()),
                    "Missing or invalid observed source counts")
                require(len(row["shapes"][index]) == counts["feature_partial_backward_updates"],
                    "Feature signature count differs from its actual schedule")
        pair = len(terms) == 2 and weights == [Fraction(1), Fraction(-1)]
        compatible = pair and all(len(a) == len(b) for a, b in zip(rows[0]["shapes"], rows[1]["shapes"]))
        shape_differences = sum(a != b for left, right in zip(rows[0]["shapes"], rows[1]["shapes"])
            for a, b in zip(left, right)) if compatible else None
        mean_counts = {key: str(sum((w * sum(r["per_image_counts"][i][key] for i in range(len(ids)))
            for w, r in zip(weights, rows)), Fraction(0)) / len(ids)) for key in COUNT_FIELDS}
        results.append(dict(id=name, terms=deepcopy(terms), images=len(ids),
            paired_mean_seconds=math.fsum(differences) / len(ids), paired_median_seconds=statistics.median(differences),
            ratio_of_total_times=math.fsum(rows[0]["times"]) / math.fsum(rows[1]["times"]) if pair else None,
            paired_mean_source_count_differences=mean_counts,
            objective_surface_steps_with_different_signatures=shape_differences,
            shape_comparison_status="paired_by_feature_update_index" if compatible else
                "different_feature_step_counts" if pair else "not_a_directed_pair",
            physical_cost_calibration_accepted=False, independent_acceptance=False))
    return results


def write_cost_contract(output, contract):
    require(contract.get("record_type") == RECORD_TYPE and contract.get("runner_armed") is False
        and contract.get("cost_disclosure_accepted") is False, "Only an unarmed prospective cost contract may be written")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "cost_contract.json", contract)
    with (output / "groups.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("group_id", "variant", "seed", "planned_images", *COUNT_FIELDS,
            "count_scope", "runtime_seconds", "peak_cuda_memory_mb", "physical_full_detector_backward_equivalents"))
        for group in contract["groups"]:
            writer.writerow((group["group_id"], group["variant"], group["seed"], group["images"],
                *("N/A" if group["declared_per_image"][k] is None else group["declared_per_image"][k] for k in COUNT_FIELDS),
                "prospective_per_image_source_counts_not_measurements", "N/A", "N/A", "N/A"))
    lines = [r"\begin{tabular}{lrrrr}", r"Variant & Updates & Risk & Feature & Objective terms \\"]
    for group in contract["groups"]:
        c = group["declared_per_image"]
        lines.append("{} & {} & {} & {} & {} ".format(group["variant"].replace("_", r"\_"),
            c["logical_gradient_updates"], c["detector_score_backward_updates"],
            c["feature_partial_backward_updates"], c["feature_objective_terms"]) + r"\\")
    lines.extend([r"\end{tabular}", r"\par\noindent\footnotesize "
        "Prospective per-image source counts, not observed cost or execution admission. "
        "Each feature update uses one complete paired feature extraction even for a single selected stage. "
        "Time, memory, physical detector equivalents and kernel FLOPs are unmeasured (N/A).", ""])
    (output / "cost_contract.tex").write_text("\n".join(lines), encoding="utf-8")
    manifest = dict(record_type=RECORD_TYPE, files={name: file_digest(output / name) for name in
        ("cost_contract.json", "groups.csv", "cost_contract.tex")}, model_calls=0, AP_evaluations=0,
        cost_disclosure_accepted=False, runner_armed=False)
    atomic_json(output / "artifact_manifest.json", manifest)
    return manifest
