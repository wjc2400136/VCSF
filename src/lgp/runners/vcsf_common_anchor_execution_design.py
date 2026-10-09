"""Complete-job common-anchor execution design, separate from runtime admission."""
from copy import deepcopy

from .vcsf_common_anchor_catalogue import compile_common_anchor_catalogue
from .vcsf_operator_execution_contract import balanced_lanes
from .vcsf_research_plan import canonical_hash, require


def compile_execution_design(registry, binding, partition, devices):
    """Preserve every logical result slot and dispatch each qualified new group once.

    Callers must bind and independently verify the anchor and reuse publication.
    This function checks their supplied identities and scheduling algebra only;
    it cannot issue input, implementation or GPU execution admission.
    """
    require(isinstance(binding, dict)
        and binding.get("status") == "common_exploration_anchor_bound_not_final_or_armed"
        and binding.get("execution_armed") is False and binding.get("final_method_frozen") is False
        and binding.get("subsequent_blocks_may_rebase") is False,
        "Require a fixed non-final common-anchor binding")
    catalogue = binding["common_catalogue"]
    expected = compile_common_anchor_catalogue(registry, binding["selected_operator_group"]["parameters"])
    require(canonical_hash(catalogue) == canonical_hash(expected), "Common catalogue changed after B binding")
    groups = catalogue["groups"]
    ids = [g["group_id"] for g in groups]
    require(isinstance(partition, dict) and set(partition) == {
        "catalogue_content_sha256", "status", "groups", "independent_partition_acceptance"}
        and partition["catalogue_content_sha256"] == canonical_hash(catalogue)
        and partition["status"] == "qualified_common_anchor_reuse_partition"
        and partition["independent_partition_acceptance"] is True,
        "Require an explicitly qualified partition for this exact catalogue")
    require(isinstance(partition["groups"], list)
        and [r.get("group_id") for r in partition["groups"]] == ids,
        "Reuse partition must cover the full ordered catalogue")
    new, reused, reuse_origins = [], [], set()
    for group, row in zip(groups, partition["groups"]):
        require(set(row) == {"group_id", "parameters_sha256", "state", "proof_sha256", "original_identity"}
            and type(row["group_id"]) is int and row["parameters_sha256"] == group["parameters_sha256"]
            and isinstance(row["proof_sha256"], str) and len(row["proof_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["proof_sha256"]),
            "Partition group identity or evidence hash differs")
        require(row["state"] in ("qualified_reuse", "requires_new"),
            "Unresolved reuse cannot silently become a new job")
        if row["state"] == "qualified_reuse":
            identity = row["original_identity"]
            require(isinstance(identity, str) and identity.strip() and identity not in reuse_origins,
                "Reused generations must have unique original identities")
            reuse_origins.add(identity)
            reused.append(group["group_id"])
        else:
            require(row["original_identity"] is None, "New group must not inherit an original generation")
            new.append(group["group_id"])
    selected = [deepcopy(g) for g in groups if g["group_id"] in new]
    require(selected, "No new jobs remain; consume existing results instead of launching")
    lanes = balanced_lanes(selected, ["faster_rcnn_r50"], devices)
    blocks = []
    for stage in catalogue["stages"]:
        required = stage["group_ids"]
        require(len(required) == len(set(required)) and set(required) <= set(ids), "Invalid block result dependencies")
        blocks.append(dict(stage=stage["stage"], background=stage["background"],
            required_group_ids=deepcopy(required),
            new_group_ids=[gid for gid in required if gid in new],
            reused_group_ids=[gid for gid in required if gid in reused],
            baseline_sha256=stage["baseline_sha256"], background_sha256=stage["background_sha256"],
            recommendation_may_rebase=False))
    require({gid for block in blocks for gid in block["required_group_ids"]} == set(ids),
        "Every catalogue group must belong to a registered block")
    return dict(schema_version=1, record_type="vcsf_common_anchor_execution_design",
        status="complete_job_design_pending_runtime_and_execution_admission",
        anchor_binding_content_sha256=canonical_hash(binding),
        catalogue_content_sha256=canonical_hash(catalogue), partition_content_sha256=canonical_hash(partition),
        source="faster_rcnn_r50", seed=42, images_per_group=5000,
        targets=list(registry.target_ids()), groups=deepcopy(groups),
        new_group_ids=new, reused_group_ids=reused,
        requested_devices=list(devices), lanes=[[g["group_id"] for g in lane] for lane in lanes],
        execution_stages=[dict(name="common_anchor_exploration", group_ids=new)],
        analysis_blocks=blocks, analyses_by_background=deepcopy(catalogue["analyses_by_background"]),
        result_slots=[dict(group_id=g["group_id"], target=t, status="NR", metrics=None)
            for g in groups for t in g["targets"]],
        logical_configurations=len(groups), new_generated_images=5000 * len(new),
        new_target_evaluations=len(registry.target_ids()) * len(new),
        device_count_changes_method=False, independent_complete_jobs=True,
        execution_armed=False, input_qualification=False, numerical_source_qualification=False,
        final_method_frozen=False, automatic_promotion=False)
