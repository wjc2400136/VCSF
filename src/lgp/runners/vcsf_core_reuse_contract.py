"""Keep logical core cells separate from independently assessed result origins."""
from __future__ import annotations

from copy import deepcopy
import re

from .vcsf_research_plan import canonical_hash, require


PROTOCOL = "vcsf_single_source_ablation_reuse"
STATES = ("qualified_reuse", "requires_new", "unresolved")
_OPERATOR_FIELDS = {"image_interpolation", "placement", "image_padding"}


def parameter_relation(current, original):
    """A possible identity-operator alias is unresolved, never an implicit match."""
    require(isinstance(current, dict) and isinstance(original, dict), "Missing complete parameter mappings")
    if canonical_hash(current) == canonical_hash(original):
        return "exact_parameters"
    common = (set(current) | set(original)) - _OPERATOR_FIELDS
    if all(key in current and key in original for key in common) and canonical_hash(
            {key: current[key] for key in common}) == canonical_hash({key: original[key] for key in common}):
        identity = all(current.get(key) == original.get(key) == 1 for key in ("scale_min", "scale_max"))
        if identity or set(original) < set(current) and set(current) - set(original) <= _OPERATOR_FIELDS:
            return "operator_or_legacy_projection_unresolved"
    return "different_configuration"


def _validate_design(contract):
    science = contract["scientific_contract"]
    groups, targets = science["groups"], science["targets"]
    require(science["selected_stage"] == "core"
        and [g["group_id"] for g in groups] == list(range(1, 9))
        and contract["logical_group_ids"] == list(range(1, 9))
        and len(targets) == len(set(targets)) == 16
        and len(contract["result_slots"]) == 128
        and contract["analysis_contract"]["count"] == 32,
        "Reuse must preserve the complete registered core design")
    require(contract["new_group_ids"] is None and contract["qualified_reused_group_ids"] is None
        and contract["historical_reuse_qualified"] is False,
        "Original execution preparation was mutated into a reuse decision")
    require(all(g["reuse_status"] == "unassessed_do_not_subtract" for g in groups)
        and all(s["status"] == "NR" and all(v is None for v in s["metrics"].values())
            for s in contract["result_slots"]), "Original preparation contains filled or reassessed cells")
    require(len({g["parameters_sha256"] for g in groups}) == 8
        and all(g["parameters_sha256"] == canonical_hash(g["parameters"]) for g in groups),
        "Core parameter identities are incomplete or duplicated")
    return groups, targets


def compile_partition(contract, original_rows, selections, assessments, *, prerequisite_proof):
    """Compile verified adapter outputs; this function does not accept external evidence itself.

    Public consumers must reconstruct the original stage/runtime and every
    assessment before calling this arithmetic layer. A partition is not admission.
    """
    before = canonical_hash(contract)
    groups, targets = _validate_design(contract)
    require(isinstance(selections, list) and [s.get("group_id") for s in selections] == list(range(1, 9))
        and all(type(s.get("group_id")) is int
            and set(s) == {"group_id", "original_generation_sha256"} for s in selections),
        "Explicit selections must cover all eight cells once, in order")
    require(isinstance(original_rows, list) and len(original_rows) == 17
        and len({r["original"]["generation_sha256"] for r in original_rows}) == 17
        and all(isinstance(r["original"]["generation_sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", r["original"]["generation_sha256"])
            and r["group"]["parameters_sha256"] == canonical_hash(r["group"]["parameters"])
            for r in original_rows),
        "Reuse assessment requires all seventeen distinct accepted scale origins")
    by_hash = {r["original"]["generation_sha256"]: r for r in original_rows}
    require(isinstance(assessments, dict) and set(assessments) <= set(range(1, 9))
        and all(type(key) is int for key in assessments),
        "Assessment references unknown core cells")
    require(isinstance(prerequisite_proof, dict)
        and type(prerequisite_proof.get("complete_scale_and_selection_reconstructed")) is bool
        and isinstance(prerequisite_proof.get("references"), dict), "Missing prerequisite proof")
    prerequisites = prerequisite_proof["complete_scale_and_selection_reconstructed"]
    rows, selected_origins = [], set()
    for group, selection in zip(groups, selections):
        gid = group["group_id"]
        inventory = [dict(original_generation_sha256=r["original"]["generation_sha256"],
            parameters_sha256=r["group"]["parameters_sha256"],
            relation=parameter_relation(group["parameters"], r["group"]["parameters"])) for r in original_rows]
        potential = {r["original_generation_sha256"] for r in inventory
            if r["relation"] != "different_configuration"}
        chosen = selection["original_generation_sha256"]
        require(chosen is None or isinstance(chosen, str) and chosen in by_hash,
            "Selected original generation is not in complete accepted scale evidence")
        require(chosen is None or chosen in potential,
            "Selected original has a different scientific configuration")
        require(chosen is None or chosen not in selected_origins,
            "The same original generation cannot fill two distinct logical core cells")
        if chosen is not None:
            selected_origins.add(chosen)
        status, reason = "unresolved", "complete_scale_acceptance_or_selection_not_ready"
        proof, origins = None, None
        if prerequisites and not potential:
            require(chosen is None and gid not in assessments, "No-match cell has unexpected reuse evidence")
            status, reason = "requires_new", "complete_original_inventory_has_no_matching_configuration"
        elif prerequisites and chosen is None:
            reason = "potential_original_exists_but_no_explicit_selection"
        elif prerequisites:
            assessment = assessments.get(gid)
            if assessment is None:
                reason = "selected_original_evidence_not_reconstructed"
            else:
                require(assessment["group_id"] == gid and assessment["original_generation_sha256"] == chosen,
                    "Reconstructed assessment identifies a different cell or generation")
                require(assessment["status"] in ("qualified_reuse", "unresolved")
                    and isinstance(assessment["reason"], str) and assessment["reason"].strip(),
                    "An applicable original must be qualified or explicitly unresolved")
                proof = deepcopy(assessment)
                status, reason = assessment["status"], assessment["reason"]
                if status == "qualified_reuse":
                    require(assessment.get("parameters_sha256") == group["parameters_sha256"]
                        and assessment.get("input_identity_verified") is True
                        and assessment.get("numerical_source_mapping_verified") is True
                        and assessment.get("original_result_acceptance_verified") is True
                        and assessment.get("original_saved_cost_attribution_verified") is True
                        and assessment.get("physical_cost_inheritance") is False
                        and assessment.get("trajectory_identity_claim") is False,
                        "Qualified reuse omitted independently reconstructed evidence boundaries")
                    origins = deepcopy(assessment["result_origins"])
                    require(isinstance(origins, list) and [s["target"] for s in origins] == targets
                        and all(s["group_id"] == gid and s["original_generation_sha256"] == chosen
                            and s["images"] == group["images"] and s["status"] == "original_accepted_reference"
                            for s in origins), "Reuse must bind all sixteen original result origins")
        rows.append(dict(group_id=gid, variant=group["variant"], parameters_sha256=group["parameters_sha256"],
            status=status, reason=reason, selected_original_generation_sha256=chosen,
            complete_original_comparison=inventory, assessment=proof, result_origins=origins))
    partition = {state: [r["group_id"] for r in rows if r["status"] == state] for state in STATES}
    require(sorted(sum(partition.values(), [])) == list(range(1, 9)), "Incomplete or overlapping core partition")
    resolved = not partition["unresolved"]
    publish = resolved and prerequisites and contract["synthetic"] is False \
        and contract["scientific_contract"]["max_images"] is None
    require(canonical_hash(contract) == before, "Reuse compilation mutated the original preparation")
    result = dict(schema_version=1, record_type="vcsf_core_original_result_reuse_partition", protocol=PROTOCOL,
        status="resolved_core_reuse_partition_pending_independent_acceptance" if publish else
            "core_reuse_partition_unresolved_or_synthetic_not_admission",
        synthetic=contract["synthetic"], original_execution_contract_sha256=contract["contract_sha256"],
        original_execution_science_sha256=contract["science_sha256"], logical_group_ids=list(range(1, 9)),
        logical_target_slots=128, registered_contrasts=32, groups=rows, assessed_partition=partition,
        partition_published=publish, new_group_ids=partition["requires_new"] if publish else None,
        qualified_reused_group_ids=partition["qualified_reuse"] if publish else None,
        unresolved_group_ids=partition["unresolved"],
        new_generation_image_instances=sum(g["images"] for g in groups
            if g["group_id"] in partition["requires_new"]) if publish else None,
        new_target_evaluations=len(partition["requires_new"]) * len(targets) if publish else None,
        prerequisite_proof=deepcopy(prerequisite_proof), original_result_slots_unchanged=True,
        original_run_provenance_preserved=True, source_or_determinism_settings_changed=False,
        current_physical_cost_inheritance=False, trajectory_identity_claim=False,
        independent_reuse_acceptance=False, formal_execution_admission=False, runner_armed=False,
        scientific_acceptance=False, independent_confirmation=False, automatic_promotion=False,
        new_model_calls=0, new_AP_evaluations=0)
    result["partition_sha256"] = canonical_hash(result)
    return result
