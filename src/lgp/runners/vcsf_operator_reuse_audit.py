"""Independent operator reuse reconstruction and bounded publication consumption."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .vcsf_core_reuse_inputs import verify_original_inputs
from . import vcsf_operator_reuse as reuse
from .vcsf_operator_reuse_contract import compile_operator_partition
from .vcsf_research_plan import canonical_hash, require
from .vcsf_stage_evidence import _bound_json


AUDIT_TYPE = "vcsf_operator_independent_original_result_reuse_audit"
RESOLVED = dict(qualified_reuse=[1, 2, 3, 4], requires_new=[5, 6, 7, 8], unresolved=[])
REPORT_KEYS = set(("schema_version record_type protocol status request_reference request_sha256 "
    "operator_plan_reference prepared_plan_sha256 science_sha256 runtime_binding_reference "
    "runtime_binding_sha256 input_evidence source_sha256 source_snapshot_sha256 original_origins_sha256 "
    "stage_proof_sha256 partition checked_file_sha256 checked_files_sha256 max_images synthetic reuse_sha256").split()) \
    | set(reuse.FALSE_FLAGS) | set(reuse.ZERO_FIELDS)
AUDIT_KEYS = set(("schema_version record_type protocol status request_reference report_reference "
    "request_sha256 reuse_sha256 partition_sha256 original_origins_sha256 stage_proof_sha256 "
    "operator_plan_reference prepared_plan_sha256 science_sha256 runtime_binding_reference "
    "runtime_receipt_reference runtime_binding_sha256 input_evidence source_sha256 source_snapshot_sha256 "
    "checked_file_sha256 checked_files_sha256 independently_reconstructed_partition "
    "record_reconstruction_accepted all_groups_resolved audit_sha256").split()) \
    | set(reuse.FALSE_FLAGS) | set(reuse.ZERO_FIELDS)


def _independent_assessment(registry, context, group, generation):
    """Call the actual input and operator numerical verifiers, not the producer."""
    from .vcsf_operator_numerical_mapping import map_operator_numerical_sources

    require(generation in context["original_rows"], "Audit selection has no original")
    row = context["original_rows"][generation]
    original = row["original"]
    checks, failures = {}, {}
    try:
        checks["input"] = verify_original_inputs(context["binding"], row, context["read"])
        require(isinstance(checks["input"], dict) and checks["input"], "Missing reconstructed input")
    except (ValueError, RuntimeError) as exc:
        checks["input"], failures["input"] = None, str(exc)
    try:
        checks["numerical"] = map_operator_numerical_sources(registry.root,
            context["binding"]["runtime_sha256"], reuse.original_root(original), original,
            group["parameters"], context["proof_rows"][generation].get("original_parameter_projection"))
        require(isinstance(checks["numerical"], dict) and checks["numerical"], "Missing reconstructed numerical")
    except (ValueError, RuntimeError) as exc:
        checks["numerical"], failures["numerical"] = None, str(exc)
    checks["saved_cost"] = deepcopy(context["costs"][generation])
    require(isinstance(checks["saved_cost"], dict) and checks["saved_cost"], "Missing independently bound saved cost")
    qualified = not failures
    origins = None
    if qualified:
        origins = []
        for original_origin in row["result_origins"]:
            origin = deepcopy(original_origin)
            origin["original_group_id"] = original_origin["group_id"]
            origin["original_origin_sha256"] = canonical_hash(original_origin)
            origin["group_id"] = group["group_id"]
            origins.append(origin)
    return dict(group_id=group["group_id"], original_generation_sha256=generation,
        parameters_sha256=group["parameters_sha256"],
        status="qualified_reuse" if qualified else "unresolved",
        reason="exact_bound_original_operator_conditions" if qualified else "selected_original_mapping_unresolved",
        input_identity_verified=checks["input"] is not None,
        numerical_source_mapping_verified=checks["numerical"] is not None,
        original_result_acceptance_verified=True, original_saved_cost_attribution_verified=True,
        checks=checks, mapping_failures=failures, result_origins=origins,
        physical_cost_inheritance=False, trajectory_identity_claim=False)


def _partition(registry, plan, originals, selections, assessments, actual):
    # The compiler is arithmetic only. Every supplied assessment is independently
    # reconstructed above on the initial audit, or covered by its pinned proof later.
    rebuilt = compile_operator_partition(registry, plan, originals, selections, assessments)
    require(canonical_hash(actual) == canonical_hash(rebuilt), "Operator partition differs from exact reconstruction")
    states = {key: [] for key in RESOLVED}
    for row in actual["groups"]:
        require(row["status"] in states, "Unknown reuse partition state")
        states[row["status"]].append(row["group_id"])
    resolved = states == RESOLVED
    require(actual["assessed_partition"] == states and actual["partition_published"] is resolved,
        "Partition status disagrees with independently enumerated cells")
    if resolved:
        require(actual["new_group_ids"] == [5, 6, 7, 8]
            and actual["qualified_reused_group_ids"] == [1, 2, 3, 4]
            and actual["new_generation_image_instances"] == 20000
            and actual["new_target_evaluations"] == 64 and actual["logical_target_slots"] == 128,
            "Resolved operator reuse is not exactly four-old/four-new")
    return states, resolved


def _report_identity(report, request_ref, request, plan, binding, input_evidence, source):
    require(set(report) == REPORT_KEYS, "Unexpected operator reuse report schema")
    reuse.check_hash(report, "reuse_sha256")
    expected = dict(schema_version=1, record_type=reuse.REPORT_TYPE, protocol=reuse.PROTOCOL,
        status="bound_operator_reuse_pending_independent_acceptance", request_reference=request_ref,
        request_sha256=canonical_hash(request), operator_plan_reference=request["operator_plan"],
        prepared_plan_sha256=canonical_hash(plan), science_sha256=plan["science_sha256"],
        runtime_binding_reference=request["runtime_inputs"], runtime_binding_sha256=binding["binding_sha256"],
        input_evidence=input_evidence, source_sha256=source, source_snapshot_sha256=canonical_hash(source),
        max_images=None, synthetic=False,
        **dict.fromkeys(reuse.FALSE_FLAGS, False), **dict.fromkeys(reuse.ZERO_FIELDS, 0))
    require(canonical_hash({key: report[key] for key in expected}) == canonical_hash(expected),
        "Report changed its bound inputs, source identity or authority")
    require(report["checked_files_sha256"] == canonical_hash(report["checked_file_sha256"]),
        "Report evidence inventory hash changed")


def audit_saved_reuse(registry, request_ref, report_ref):
    """Initial expensive audit: independently reconstruct input/result/cost/numerics."""
    report, publication_files = reuse.read_publication(registry, report_ref, "assessment")
    context = reuse.reconstruct_context(registry, request_ref)
    _report_identity(report, request_ref, context["request"], context["plan"],
        context["binding"], context["input_evidence"], context["source"])
    origins_path = Path(report_ref["file"]).parent / "origins.json"
    saved_origins = _bound_json(dict(file=str(origins_path), sha256=publication_files[str(origins_path)]))
    require(canonical_hash(saved_origins) == canonical_hash(context["origins"])
        and report["original_origins_sha256"] == context["origins"]["origins_sha256"]
        and report["stage_proof_sha256"] == context["stage"]["proof_sha256"],
        "Producer origins or stage proof do not reconstruct")
    assessments = {}
    for group, selection in zip(context["plan"]["scientific_contract"]["groups"], context["request"]["selections"]):
        digest = selection["original_generation_sha256"]
        if digest is not None:
            assessments[group["group_id"]] = _independent_assessment(registry, context, group, digest)
    states, resolved = _partition(registry, context["plan"], context["origins"]["groups"],
        context["request"]["selections"], assessments, report["partition"])
    files = reuse.evidence_files(registry, context, assessments)
    require(files == report["checked_file_sha256"], "Producer evidence inventory did not reconstruct independently")
    reuse._merge_files(files, publication_files)
    reuse.rehash_files(files)
    audit = dict(schema_version=1, record_type=AUDIT_TYPE, protocol=reuse.PROTOCOL,
        status="independently_verified_operator_reuse_partition" if resolved else "independently_verified_unresolved_operator_reuse_record",
        request_reference=deepcopy(request_ref), report_reference=deepcopy(report_ref),
        request_sha256=canonical_hash(context["request"]), reuse_sha256=report["reuse_sha256"],
        partition_sha256=report["partition"]["partition_sha256"],
        original_origins_sha256=context["origins"]["origins_sha256"], stage_proof_sha256=context["stage"]["proof_sha256"],
        operator_plan_reference=deepcopy(context["request"]["operator_plan"]),
        prepared_plan_sha256=canonical_hash(context["plan"]), science_sha256=context["plan"]["science_sha256"],
        runtime_binding_reference=deepcopy(context["request"]["runtime_inputs"]),
        runtime_receipt_reference=deepcopy(context["request"]["runtime_receipt"]),
        runtime_binding_sha256=context["binding"]["binding_sha256"], input_evidence=context["input_evidence"],
        source_sha256=context["source"], source_snapshot_sha256=canonical_hash(context["source"]),
        checked_file_sha256=files, checked_files_sha256=canonical_hash(files),
        independently_reconstructed_partition=states, record_reconstruction_accepted=True,
        all_groups_resolved=resolved,
        **dict.fromkeys(reuse.FALSE_FLAGS, False), **dict.fromkeys(reuse.ZERO_FIELDS, 0))
    audit["independent_reuse_acceptance"] = resolved
    audit["audit_sha256"] = canonical_hash(audit)
    return audit


def write_audit(registry, output, audit):
    require(set(audit) == AUDIT_KEYS, "Unexpected operator audit schema")
    reuse.check_hash(audit, "audit_sha256")
    return reuse.publish(registry, output, {"audit.json": audit}, kind="audit")


def verify_reuse_publication(registry, request_ref, report_ref, audit_ref):
    """Consume pinned independent proof; never repeat the historical reconstruction.

    The audit reference is the caller's explicit trust anchor, not a discovered
    latest receipt. Its complete report identity, sources and evidence bytes are
    checked afresh, as is the current operator input publication.
    """
    source = reuse.source_snapshot(registry)
    request = reuse.read_request(request_ref)
    report, report_files = reuse.read_publication(registry, report_ref, "assessment")
    audit, audit_files = reuse.read_publication(registry, audit_ref, "audit")
    require(set(audit) == AUDIT_KEYS, "Unexpected independently audited proof schema")
    plan, binding, input_evidence = reuse.current_inputs(registry, request)
    _report_identity(report, request_ref, request, plan, binding, input_evidence, source)
    assessments = {row["group_id"]: row["assessment"] for row in report["partition"]["groups"]
        if row["assessment"] is not None}
    states, resolved = _partition(registry, plan, report["partition"]["original_rows"],
        request["selections"], assessments, report["partition"])
    require(resolved, "Unresolved operator reuse cannot be consumed for execution preparation")
    expected_files = deepcopy(report["checked_file_sha256"])
    reuse._merge_files(expected_files, report_files)
    expected = dict(schema_version=1, record_type=AUDIT_TYPE, protocol=reuse.PROTOCOL,
        status="independently_verified_operator_reuse_partition",
        request_reference=request_ref, report_reference=report_ref, request_sha256=canonical_hash(request),
        reuse_sha256=report["reuse_sha256"], partition_sha256=report["partition"]["partition_sha256"],
        original_origins_sha256=report["original_origins_sha256"], stage_proof_sha256=report["stage_proof_sha256"],
        operator_plan_reference=request["operator_plan"], prepared_plan_sha256=canonical_hash(plan),
        science_sha256=plan["science_sha256"], runtime_binding_reference=request["runtime_inputs"],
        runtime_receipt_reference=request["runtime_receipt"], runtime_binding_sha256=binding["binding_sha256"],
        input_evidence=input_evidence, source_sha256=source, source_snapshot_sha256=canonical_hash(source),
        checked_file_sha256=expected_files, checked_files_sha256=canonical_hash(expected_files),
        independently_reconstructed_partition=states, record_reconstruction_accepted=True, all_groups_resolved=True,
        **dict.fromkeys(reuse.FALSE_FLAGS, False), **dict.fromkeys(reuse.ZERO_FIELDS, 0))
    expected["independent_reuse_acceptance"] = True
    expected["audit_sha256"] = canonical_hash(expected)
    require(canonical_hash(audit) == canonical_hash(expected),
        "Independent audit is not the exact proof for this request/report/current source")
    reuse.rehash_files(expected_files)
    reuse.rehash_files(audit_files)
    require(reuse.source_snapshot(registry) == source and reuse.read_request(request_ref) == request,
        "Reuse source/request changed during consumer verification")
    return dict(request=request, report=report, binding=binding, input_evidence=input_evidence)
