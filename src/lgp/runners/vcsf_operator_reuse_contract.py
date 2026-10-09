"""Bounded operator reuse arithmetic over caller-reconstructed evidence only."""
from __future__ import annotations

from copy import deepcopy
import re

from ..metrics import COCO_BBOX_METRICS
from .vcsf_core_reuse_contract import STATES, parameter_relation
from .vcsf_operator_factorial_plan import PROTOCOL, compile_operator_factorial_plan
from .vcsf_research_plan import canonical_hash, require
from .vcsf_scale_plan import compile_scale_plan


def _original_inventory(registry, original_rows, targets):
    declared = compile_scale_plan(registry)["groups"]
    require(isinstance(original_rows, list) and len(original_rows) == len(declared),
        "All registered original scale generations are required")
    generations, variants = set(), set()
    for row, expected in zip(original_rows, declared):
        group, original = row["group"], row["original"]
        require(type(group["group_id"]) is int and type(original["group_id"]) is int,
            "Original group identities must be integers")
        digest = original["generation_sha256"]
        require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest)
            and digest not in generations, "Original generations are malformed or duplicated")
        generations.add(digest)
        # The scale anchor was actually produced by the older layered study.
        legacy = (expected["potential_historical_reference"]
            and group["variant"] == "levels_two" and group["group_id"] == 29)
        require(legacy or (group["variant"] == expected["variant"]
            and type(group["group_id"]) is int and group["group_id"] == expected["group_id"]),
            "Original inventory is incomplete, reordered or relabelled")
        require(group["variant"] not in variants, "Duplicate original study row")
        variants.add(group["variant"])
        relation = parameter_relation(expected["parameters"], group["parameters"])
        require(relation != "different_configuration" if legacy else relation == "exact_parameters",
            "Original parameters differ from the registered scale study")
        require(group["parameters_sha256"] == canonical_hash(group["parameters"])
            and original["parameters_sha256"] == group["parameters_sha256"]
            and canonical_hash(original["parameters"]) == group["parameters_sha256"]
            and original["group_id"] == group["group_id"]
            and original["variant"] == group["variant"]
            and all(canonical_hash(group[key]) == canonical_hash(expected[key])
                for key in ("source", "seed", "images", "targets", "blackbox_targets")),
            "Original group, parameters and scientific scope disagree")
        origins = row["result_origins"]
        require(isinstance(origins, list) and [o["target"] for o in origins] == targets,
            "Original result panel must retain every target in canonical order")
        for origin in origins:
            require(type(origin["group_id"]) is int and origin["group_id"] == group["group_id"]
                and origin["original_generation_sha256"] == digest
                and origin["original_parameters_sha256"] == group["parameters_sha256"]
                and origin["original_identity_sha256"] == canonical_hash(original)
                and type(origin["images"]) is int and origin["images"] == group["images"]
                and origin["status"] == "original_accepted_reference"
                and origin.get("physical_cost_inheritance") is False
                and origin.get("trajectory_identity_claim") is False
                and "original_group_id" not in origin and "original_origin_sha256" not in origin,
                "Original result origin is relabelled or overclaims inheritance")
            metrics = origin["metrics"]
            require(isinstance(metrics, dict) and set(metrics) == set(COCO_BBOX_METRICS),
                "Original origins require all twelve raw COCO metrics")
    return {row["original"]["generation_sha256"]: row for row in original_rows}


def _assessment(group, original_row, assessment):
    gid = group["group_id"]
    digest = original_row["original"]["generation_sha256"]
    require(isinstance(assessment, dict) and type(assessment.get("group_id")) is int
        and assessment["group_id"] == gid
        and assessment.get("original_generation_sha256") == digest
        and assessment.get("parameters_sha256") == group["parameters_sha256"]
        and assessment.get("status") in ("qualified_reuse", "unresolved")
        and isinstance(assessment.get("reason"), str) and assessment["reason"].strip(),
        "Assessment must bind the selected cell, generation and parameters")
    if assessment["status"] != "qualified_reuse":
        return None
    require(all(assessment.get(key) is True for key in (
        "input_identity_verified", "numerical_source_mapping_verified",
        "original_result_acceptance_verified", "original_saved_cost_attribution_verified"))
        and assessment.get("physical_cost_inheritance") is False
        and assessment.get("trajectory_identity_claim") is False,
        "Qualified assessment omitted caller-reconstructed evidence boundaries")
    remapped = []
    for origin in original_row["result_origins"]:
        item = deepcopy(origin)
        item.update(group_id=gid, original_group_id=origin["group_id"],
            original_origin_sha256=canonical_hash(origin))
        remapped.append(item)
    require(canonical_hash(assessment.get("result_origins")) == canonical_hash(remapped),
        "Qualified result origins must be the exact original remap, including metrics and provenance")
    return remapped


def compile_operator_partition(registry, prepared_plan, original_rows, selections, assessments):
    """Compile a four-old/four-new partition, NEVER acceptance or admission.

    The caller MUST already reconstruct the complete original evidence and each
    supplied assessment. This checks arithmetic and internal bindings, not the
    truth of upstream verified flags, file contents, inputs or numerical claims.
    ``original_rows`` is the collector's complete ``origins['groups']`` list.
    Result origins change only group_id and add original_group_id plus the hash
    of the untouched original origin. Pixel-audit and clean-byte scope is neither
    upgraded nor reinterpreted here. Missing historical proof stays unresolved.
    """
    before = canonical_hash([prepared_plan, original_rows, selections, assessments])
    rebuilt = compile_operator_factorial_plan(registry, devices=prepared_plan["requested_devices"])
    require(canonical_hash(prepared_plan) == canonical_hash(rebuilt),
        "Prepared plan differs from the entire reconstructed canonical operator plan")
    science = rebuilt["scientific_contract"]
    groups, targets = science["groups"], science["targets"]
    require(science["max_images"] is None and science["diagnostic_only"] is False
        and [g["variant"] for g in groups] == list("ABCDEFGH")
        and all(type(g["images"]) is int and g["images"] == 5000 for g in groups),
        "Operator reuse requires the full non-diagnostic eight-cell scientific scope")
    ids = [g["group_id"] for g in groups]
    old_ids = [g["group_id"] for g in groups if g["historical_variant"] is not None]
    new_ids = [g["group_id"] for g in groups if g["requested_new"]]
    require(old_ids == ids[:4] and new_ids == ids[4:], "Registered four-old/four-new scope changed")
    require(isinstance(selections, list) and len(selections) == len(groups)
        and all(isinstance(s, dict) and set(s) == {"group_id", "original_generation_sha256"}
            and type(s["group_id"]) is int for s in selections)
        and [s["group_id"] for s in selections] == ids,
        "Explicit selections must cover all eight cells once, in order")
    require(isinstance(assessments, dict) and all(type(gid) is int for gid in assessments)
        and set(assessments) <= set(old_ids), "Assessments cannot expand the historical reuse scope")
    by_hash = _original_inventory(registry, original_rows, targets)
    selected, mapped, remapped = set(), {}, {}
    for group, selection in zip(groups, selections):
        gid, chosen = group["group_id"], selection["original_generation_sha256"]
        require(chosen is None or isinstance(chosen, str) and chosen in by_hash,
            "Selected generation is absent from the complete original inventory")
        require(chosen is None or chosen not in selected, "Duplicate selected original generation")
        if chosen is None:
            require(gid not in assessments, "Assessment has no explicitly selected original")
            continue
        selected.add(chosen)
        require(gid in old_ids, "Only registered historical cells may select an original")
        original = by_hash[chosen]
        require(parameter_relation(group["parameters"], original["group"]["parameters"])
            != "different_configuration", "Selected original has different parameters")
        expected_variant = group["historical_variant"]
        require(original["group"]["variant"] == expected_variant
            or (gid == old_ids[0] and original["group"]["variant"] == "levels_two"
                and original["group"]["group_id"] == 29),
            "Selection does not identify this cell's registered historical source")
        if gid in assessments:
            origins = _assessment(group, original, assessments[gid])
            if origins is not None:
                remapped[gid] = origins
                mapped[chosen] = group

    rows = []
    for group, selection in zip(groups, selections):
        gid, chosen = group["group_id"], selection["original_generation_sha256"]
        inventory = []
        for original in original_rows:
            digest = original["original"]["generation_sha256"]
            relation = parameter_relation(group["parameters"], original["group"]["parameters"])
            # A legacy projection is broad. Only a qualified caller mapping can
            # exclude it from other cells; its raw comparison remains inspectable.
            mapped_group = mapped.get(digest)
            effective = (parameter_relation(group["parameters"], mapped_group["parameters"])
                if mapped_group is not None else relation)
            inventory.append(dict(original_generation_sha256=digest,
                original_group_id=original["group"]["group_id"],
                parameters_sha256=original["group"]["parameters_sha256"], relation=relation,
                effective_relation=effective,
                caller_mapping_group_id=None if mapped_group is None else mapped_group["group_id"]))
        potential = any(item["effective_relation"] != "different_configuration" for item in inventory)
        status, reason = "unresolved", "historical_selection_or_assessment_missing"
        if gid in new_ids:
            status = "unresolved" if potential else "requires_new"
            reason = ("potential_original_must_be_resolved_before_new_work" if potential else
                "registered_new_cell_has_no_matching_original_after_caller_mappings")
        elif gid in assessments:
            status, reason = assessments[gid]["status"], assessments[gid]["reason"]
        rows.append(dict(group_id=gid, variant=group["variant"],
            parameters_sha256=group["parameters_sha256"], status=status, reason=reason,
            selected_original_generation_sha256=chosen, complete_original_comparison=inventory,
            assessment=deepcopy(assessments.get(gid)), result_origins=deepcopy(remapped.get(gid))))
    partition = {state: [r["group_id"] for r in rows if r["status"] == state] for state in STATES}
    publish = partition == dict(qualified_reuse=old_ids, requires_new=new_ids, unresolved=[])
    require(canonical_hash([prepared_plan, original_rows, selections, assessments]) == before,
        "Operator reuse compilation mutated caller inputs")
    result = dict(schema_version=1, record_type="vcsf_operator_original_result_reuse_partition",
        protocol=PROTOCOL, arithmetic_only=True, upstream_evidence_validated=False,
        status="resolved_operator_reuse_partition_pending_independent_acceptance" if publish else
            "operator_reuse_partition_unresolved_not_admission",
        prepared_plan_sha256=canonical_hash(rebuilt), science_sha256=rebuilt["science_sha256"],
        original_rows_sha256=canonical_hash(original_rows), original_rows=deepcopy(original_rows),
        selections=deepcopy(selections), logical_group_ids=ids,
        logical_target_slots=len(groups) * len(targets),
        registered_contrasts=science["analysis"]["family_size"],
        analysis_contract=deepcopy(science["analysis"]), groups=rows, assessed_partition=partition,
        partition_published=publish, new_group_ids=new_ids if publish else None,
        qualified_reused_group_ids=old_ids if publish else None,
        unresolved_group_ids=partition["unresolved"],
        new_generation_image_instances=sum(g["images"] for g in groups
            if g["group_id"] in new_ids) if publish else None,
        new_target_evaluations=len(new_ids) * len(targets) if publish else None,
        original_result_slots_unchanged=True, original_run_provenance_preserved=True,
        current_physical_cost_inheritance=False, physical_cost_inheritance=False,
        trajectory_identity_claim=False, independent_reuse_acceptance=False,
        formal_execution_admission=False, runner_armed=False, scientific_acceptance=False,
        independent_confirmation=False, automatic_promotion=False,
        new_model_calls=0, new_AP_evaluations=0)
    result["partition_sha256"] = canonical_hash(result)
    return result
