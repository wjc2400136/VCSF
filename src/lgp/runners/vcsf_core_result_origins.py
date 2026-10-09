"""Collect original accepted scale cells without relabelling or granting reuse."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .vcsf_research_plan import canonical_hash, require
from .vcsf_stage_evidence import _bound_json, verify_stage_evidence


class _BoundMetadata:
    def __init__(self, inventory):
        self.allowed = {}
        self.checked = {}
        self.documents = {}
        self.extend(inventory)

    def extend(self, inventory):
        require(isinstance(inventory, dict), "Missing accepted input inventory")
        for name, digest in inventory.items():
            require(name not in self.allowed or self.allowed[name] == digest,
                "Conflicting accepted origin input hash")
            self.allowed[name] = digest

    def reference(self, path, digest=None):
        name = str(path)
        require(name in self.allowed and (digest is None or self.allowed[name] == digest),
            "Original input is absent from the reconstructed acceptance inventory: " + name)
        return dict(file=name, sha256=self.allowed[name])

    def json(self, path, digest=None):
        ref = self.reference(path, digest)
        if ref["file"] not in self.documents:
            self.documents[ref["file"]] = _bound_json(ref)
            self.checked[ref["file"]] = ref["sha256"]
        return self.documents[ref["file"]]

    def matching(self, digest, filename):
        matches = [self.reference(name, digest) for name, value in sorted(self.allowed.items())
            if value == digest and Path(name).name == filename]
        require(matches, "No accepted original metadata reference matches the cell")
        return matches

    def unchanged(self):
        for name, digest in self.checked.items():
            require(canonical_hash(_bound_json(dict(file=name, sha256=digest)))
                == canonical_hash(self.documents[name]), "Origin metadata changed during collection")


def _pixel_scope(evidence):
    if evidence["status"] == "independently_verified_scale_group_artifacts_pending_metric_replay":
        require(evidence["max_images"] is None and evidence["images"] == 5000
            and evidence["png_pixels_verified"] == evidence["clean_image_bytes_bound"] == 5000
            and evidence["clean_byte_binding_scope"] == "actual_bytes_consumed_by_independent_audit",
            "Scale origin lost its complete independent content-audit scope")
        return dict(independently_decoded_pngs=5000, clean_content_hashes=5000,
            clean_content_hash_scope=evidence["clean_byte_binding_scope"],
            generation_instant_clean_content_hash_claim=False)
    require(evidence["status"] == "independently_verified_new_group_artifacts_pending_metric_replay"
        and evidence["retained_png_pixels_verified"] == 500
        and evidence["pruned_pngs_verified_absent"] == 4500
        and evidence["pruned_png_pixels_independently_replayed"] is False
        and evidence["producer_pre_prune_decoded_pngs"] == 5000,
        "Historical origin lost its accepted retained-payload audit scope")
    return dict(independently_decoded_pngs=500, producer_pre_prune_decoded_pngs=5000,
        pruned_png_pixels_independently_replayed=False, clean_content_hashes=None,
        clean_content_hash_scope="accepted_original_protocol_and_retained_payload_audit",
        generation_instant_clean_content_hash_claim=False)


def _original_group(group):
    result = dict(group)
    if "source_matched_target" in result:
        require(result.pop("source_matched_target") == group["source"],
            "Analysis group changed its source-matched target")
    return result


def _extend_previous_point_inputs(read, point_plan):
    plan = point_plan
    references = []
    for stage, status in (("remaining_widths", "independently_verified_operator_point_analysis"),
            ("operators_at_anchor_width", "independently_verified_scale_point_analysis")):
        execution = read.json(plan["execution_plan_file"], plan["execution_plan_sha256"])
        require(execution["stage"] == stage, "Original point predecessor stage changed")
        ref = read.reference(execution["previous_analysis_audit_file"], execution["previous_analysis_audit_sha256"])
        accepted = read.json(ref["file"], ref["sha256"])
        require(accepted["status"] == status and accepted["diagnostic"] is False,
            "A diagnostic or different predecessor cannot supply original inputs")
        read.extend(accepted["checked_input_sha256"])
        plan = read.json(accepted["point_plan_file"], accepted["point_plan_sha256"])
        references.append(ref)
    return references


def _collect_target(read, row, cell, ref, point_reference):
    group, original = row["group"], row["original"]
    gid, target = group["group_id"], cell["target"]
    require(ref["group_id"] == gid and ref["target"] == target
        and ref["normalized_cell_sha256"] == canonical_hash(cell)
        and cell["status"] == "complete" and cell["failures"] == []
        and all(cell[key] == group[key] for key in
            ("group_id", "variant", "parameters_sha256", "source", "seed", "images"))
        and cell["images"] == 5000 and cell["image_ids_sha256"] == row["image_ids_sha256"],
        "Accepted target was relabelled or has a partial panel")
    outer = read.json(ref["cell_file"], ref["cell_sha256"])
    replay = read.json(ref["replay_file"], ref["replay_sha256"])
    # The first accepted replay schema owns its input inventory in the outer
    # cell, rather than flattening it into every successor's plan.
    read.extend(outer.get("checked_input_sha256", {}))
    # Earlier accepted cells have several outer schemas. Their full independent
    # verifier has already reconstructed them; preserve the explicit references.
    require(canonical_hash(outer["normalized_cell"]) == canonical_hash(cell)
        and outer.get("normalized_cell_sha256", canonical_hash(cell)) == canonical_hash(cell)
        and replay["normalized_cell_sha256"] == canonical_hash(cell),
        "Original replay no longer binds this normalized cell")
    require(replay["input_sha256"] == cell["input_evidence_sha256"],
        "Original replay binds another target input")
    input_refs = read.matching(cell["input_evidence_sha256"], "input.json")
    document = read.json(input_refs[0]["file"], input_refs[0]["sha256"])
    require(all(read.json(r["file"], r["sha256"]) == document for r in input_refs),
        "Equivalent original input copies disagree")
    require(canonical_hash(_original_group(document["group"])) == canonical_hash(_original_group(group))
        and document["image_ids_sha256"] == cell["image_ids_sha256"]
        and document.get("max_images") is None,
        "Original target input changed group or image scope")
    evidence = read.json(document["group_evidence_file"], document["group_evidence_sha256"])
    read.extend(evidence["checked_input_sha256"])
    pixel_scope = _pixel_scope(evidence)
    receipt_path = Path(document["group_root"])/"group_acceptance.json"
    receipt = read.json(receipt_path, evidence["source_group_receipt_sha256"])
    records_path = Path(document["group_root"])/"records.json"
    records = read.json(records_path, receipt["records_sha256"])
    records_matches = [(i, r) for i, r in enumerate(records)
        if r.get("group_id") == gid and r.get("target") == target]
    require(len(records) == 16 and len(records_matches) == 1,
        "Original records file has an incomplete or duplicate target panel")
    position, record = records_matches[0]
    require(record == document["original_record"]
        and canonical_hash(record) == cell["original_record_sha256"],
        "Original record row or its selector changed")
    metadata = read.json(document["metadata_file"], document["metadata_sha256"])
    require(metadata == document["metadata"] and metadata["metrics"] == cell["metrics"]
        and metadata["target"] == target and metadata["images"] == 5000
        and metadata["status"] == "complete" and metadata["failures"] == [],
        "Saved raw target metrics disagree with the accepted cell")
    artifact = read.json(document["prediction_artifact_file"], document["prediction_artifact_sha256"])
    require(artifact == document["archive"] == metadata["predictions_artifact"]
        and artifact["status"] == "verified_lossless_archive" and artifact["format"] == "gzip"
        and artifact["archive_file"] == "predictions.json.gz"
        and artifact["uncompressed_sha256"] == metadata["predictions_sha256"],
        "Original prediction archive identity changed")
    archive_ref = read.reference(Path(document["prediction_artifact_file"]).parent/
        artifact["archive_file"], artifact["archive_sha256"])
    generation = read.json(row["generation_file"], row["generation_sha256"])
    require(Path(row["generation_file"]) == Path(document["attack_root"])/"run.json"
        and generation["image_ids_sha256"] == cell["image_ids_sha256"]
        and generation["run_metadata"]["protocol"] == original["protocol"]
        and document["execution_plan_sha256"] == original["execution_plan_sha256"],
        "Original target and generation provenance diverge")
    return dict(group_id=gid, target=target, images=5000,
        original_generation_sha256=original["generation_sha256"],
        status="original_accepted_reference", original_identity_sha256=canonical_hash(original),
        original_parameters_sha256=group["parameters_sha256"], metrics=deepcopy(cell["metrics"]),
        original_normalized_cell_sha256=canonical_hash(cell),
        original_record=dict(file=str(records_path), sha256=receipt["records_sha256"],
            position=position, group_id=gid, target=target, row_sha256=canonical_hash(record)),
        generation=read.reference(row["generation_file"], row["generation_sha256"]),
        manifest=read.reference(Path(row["generation_file"]).parent/"manifest.jsonl",
            generation["manifest_sha256"]),
        execution_plan=read.reference(document["execution_plan_file"], document["execution_plan_sha256"]),
        group_receipt=read.reference(receipt_path, evidence["source_group_receipt_sha256"]),
        group_evidence=read.reference(document["group_evidence_file"], document["group_evidence_sha256"]),
        input_references=input_refs, point_cell=read.reference(ref["cell_file"], ref["cell_sha256"]),
        official_replay=read.reference(ref["replay_file"], ref["replay_sha256"]),
        point_acceptance=deepcopy(point_reference), metrics_file=read.reference(
            document["metadata_file"], document["metadata_sha256"]),
        prediction_artifact=read.reference(document["prediction_artifact_file"],
            document["prediction_artifact_sha256"]), prediction_archive=archive_ref,
        prediction_uncompressed_sha256=artifact["uncompressed_sha256"],
        canonical_annotation=read.reference(document["canonical_annotation_file"],
            document["canonical_annotation_sha256"]),
        exported_annotation=read.reference(document["annotation_file"], document["annotation_sha256"]),
        image_ids_sha256=cell["image_ids_sha256"], checkpoint_sha256=record["checkpoint_sha256"],
        original_code_commit=record["code_commit"], pixel_audit_scope=pixel_scope,
        current_input_equivalence_assessed=False, numerical_source_mapping_assessed=False,
        physical_cost_inheritance=False, trajectory_identity_claim=False)


def _collect_reconstructed_origins(stage):
    """Internal only: caller must reconstruct stage evidence, not accept this dict."""
    proof, point_ref = stage["proof"], stage["references"]["point_acceptance"]
    read = _BoundMetadata(proof["checked_input_sha256"])
    read.extend({point_ref["file"]: point_ref["sha256"]})
    accepted = read.json(point_ref["file"], point_ref["sha256"])
    require(accepted["status"] == "independently_verified_remaining_width_point_analysis"
        and accepted["diagnostic"] is False and accepted["max_images"] is None
        and accepted["groups"] == 17 and accepted["cells"] == 272
        and accepted["images_per_group"] == 5000 and accepted["all_twelve_metrics_verified"] is True,
        "Original complete scale acceptance is required")
    read.extend(accepted["checked_input_sha256"])
    plan = read.json(accepted["point_plan_file"], accepted["point_plan_sha256"])
    predecessor_refs = _extend_previous_point_inputs(read, plan)
    report = read.json(Path(accepted["analysis_root"])/"reports/analysis.json", accepted["report_sha256"])
    cells = report["normalized_cells"]
    require(len(proof["groups"]) == 17 and len(proof["targets"]) == 16 and len(cells) == 272
        and [r["group"]["group_id"] for r in proof["groups"]] == [29] + list(range(2, 18))
        and cells == plan["previous_cells"] + [c for g in plan["groups"] for c in g["cells"]],
        "Original full scale result inventory is incomplete")
    refs = deepcopy(plan["contract"]["previous_point_replay_refs"])
    require(len(refs) == 80, "Original accepted eighty-cell reference panel is incomplete")
    for cell in cells[80:]:
        path = Path(accepted["analysis_root"])/"cells"/"g{:06d}".format(cell["group_id"])/cell["target"]/"cell.json"
        outer = read.json(path)
        refs.append(dict(group_id=cell["group_id"], target=cell["target"], cell_file=str(path),
            cell_sha256=read.reference(path)["sha256"], replay_file=outer["replay_file"],
            replay_sha256=outer["replay_sha256"], normalized_cell_sha256=outer["normalized_cell_sha256"]))
    rows = []
    for index, row in enumerate(proof["groups"]):
        panel = cells[index*16:(index+1)*16]
        require([c["target"] for c in panel] == proof["targets"],
            "Original target order changed or a target is duplicated")
        origins = [_collect_target(read, row, cell, ref, point_ref)
            for cell, ref in zip(panel, refs[index*16:(index+1)*16])]
        rows.append(dict(original=deepcopy(row["original"]), group=deepcopy(row["group"]),
            result_origins=origins))
    read.unchanged()
    result = dict(schema_version=1, status="collected_reconstructed_original_scale_result_origins",
        stage_proof_sha256=stage["proof_sha256"], point_acceptance=deepcopy(point_ref),
        predecessor_point_acceptances=predecessor_refs,
        groups=rows, targets=deepcopy(proof["targets"]), original_groups=17, original_target_cells=272,
        checked_metadata_sha256=read.checked, formal_reuse_qualified=False,
        current_input_equivalence_assessed=False, numerical_source_mapping_assessed=False,
        physical_cost_inheritance=False, trajectory_identity_claim=False,
        formal_execution_admission=False, scientific_acceptance=False, new_model_calls=0,
        new_AP_evaluations=0, bootstrap_replicates_computed=0)
    result["origins_sha256"] = canonical_hash(result)
    return result


def collect_stage_result_origins(registry, evidence, working_baseline, nonzero_comparator):
    """Reconstruct original acceptance before collecting its complete target origins."""
    stage = verify_stage_evidence(registry, evidence, working_baseline, nonzero_comparator)
    return _collect_reconstructed_origins(stage)
