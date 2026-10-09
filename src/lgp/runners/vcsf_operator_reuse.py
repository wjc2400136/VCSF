"""Prepare source-bound operator reuse evidence without granting acceptance."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import sys

from ..io import file_digest
from . import vcsf_ablation_execution_contract as source_binding
from .vcsf_core_result_origins import _BoundMetadata, _collect_reconstructed_origins
from .vcsf_core_reuse_inputs import bind_original_saved_cost, verify_original_inputs
from .vcsf_operator_input_consumer import verify_operator_inputs_for_consumer
from .vcsf_operator_factorial_plan import PROTOCOL, compile_operator_factorial_plan
from .vcsf_operator_reuse_contract import compile_operator_partition
from .vcsf_research_plan import canonical_hash, require
from .vcsf_stage_evidence import _bound_json, verify_stage_evidence


ENTRYPOINT = "experiments/vcsf_operator_reuse.py"
NAMESPACE = PROTOCOL + "_reuse"
REQUEST_KIND = "vcsf_operator_original_result_reuse_request"
REPORT_TYPE = "vcsf_operator_bound_original_result_reuse_assessment"
REQUEST_KEYS = {"schema_version", "kind", "operator_plan", "runtime_inputs",
    "runtime_receipt", "stage_evidence", "original_origins", "selections"}
FALSE_FLAGS = ("independent_reuse_acceptance", "formal_execution_admission",
    "scientific_acceptance", "physical_cost_inheritance", "trajectory_identity_claim",
    "original_roots_modified", "independent_confirmation", "automatic_promotion")
ZERO_FIELDS = ("new_model_calls", "new_AP_evaluations", "bootstrap_replicates_computed")
_ENTRY_HASH = file_digest(Path(__file__).resolve().parents[3] / ENTRYPOINT)


def source_snapshot(registry, *, public=False):
    source_binding._verify_process_binding(registry)
    require(file_digest(registry.root / ENTRYPOINT) == _ENTRY_HASH,
        "Operator reuse entry changed after import")
    if public:
        entry = sys.modules.get("__main__")
        require(getattr(entry, "__file__", None) == str(registry.root / ENTRYPOINT)
            and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == _ENTRY_HASH,
            "Reuse publication requires its source-bound public entry")
    return dict(sorted(dict(source_binding._IMPORTED_SOURCE,
        **{ENTRYPOINT: _ENTRY_HASH,
            "tests/test_vcsf_operator_reuse.py": file_digest(registry.root / "tests/test_vcsf_operator_reuse.py")}).items()))


def read_request(request_ref):
    request = _bound_json(request_ref)
    require(isinstance(request, dict) and set(request) == REQUEST_KEYS
        and type(request["schema_version"]) is int and request["schema_version"] == 1
        and request["kind"] == REQUEST_KIND, "Invalid operator reuse request schema")
    for key in REQUEST_KEYS - {"schema_version", "kind", "selections"}:
        ref = request[key]
        require(isinstance(ref, dict) and set(ref) == {"file", "sha256"}
            and isinstance(ref["file"], str) and Path(ref["file"]).is_absolute()
            and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
            "Operator request needs exact file/hash references")
    rows = request["selections"]
    require(isinstance(rows, list) and len(rows) == 8
        and all(isinstance(row, dict) and set(row) == {"group_id", "original_generation_sha256"}
            and type(row["group_id"]) is int for row in rows)
        and [row["group_id"] for row in rows] == list(range(1, 9)),
        "Selections must cover eight ordered integer group IDs")
    selected = []
    for row in rows:
        digest = row["original_generation_sha256"]
        require(digest is None or isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Malformed selected generation digest")
        require(row["group_id"] <= 4 or digest is None, "E-H cannot select historical results")
        if digest is not None:
            selected.append(digest)
    require(len(selected) == len(set(selected)), "A historical generation was selected twice")
    return request


def current_inputs(registry, request):
    refs = [request[k] for k in ("operator_plan", "runtime_inputs", "runtime_receipt")]
    plan = _bound_json(refs[0])
    rebuilt = compile_operator_factorial_plan(registry, devices=plan["requested_devices"])
    require(canonical_hash(plan) == canonical_hash(rebuilt)
        and plan["scientific_contract"]["max_images"] is None,
        "Reuse requires the exact full-split operator plan")
    binding, evidence = verify_operator_inputs_for_consumer(registry,
        *[value for ref in refs for value in (ref["file"], ref["sha256"])])
    require(binding["max_images"] is None and binding["diagnostic_only"] is False,
        "Diagnostic inputs cannot qualify formal historical reuse")
    return plan, binding, evidence


def original_root(original):
    parts = Path(original["generation_file"]).parts
    require(parts.count("outputs") == 1, "Original workspace is ambiguous")
    return Path(*parts[:parts.index("outputs")])


def reconstruct_context(registry, request_ref):
    """Reconstruct real historical acceptance once per assessment or initial audit."""
    source = source_snapshot(registry)
    request = read_request(request_ref)
    plan, binding, input_evidence = current_inputs(registry, request)
    evidence = _bound_json(request["stage_evidence"])
    decision = _bound_json(evidence["selection_decision"])
    # These endpoints reconstruct the old seventeen rows, never select an A-H winner.
    stage = verify_stage_evidence(registry, evidence,
        decision["working_baseline"], decision["nonzero_comparator"])
    saved_origins = _bound_json(request["original_origins"])
    origins = _collect_reconstructed_origins(stage)
    require(canonical_hash(saved_origins) == canonical_hash(origins),
        "Provided original origins differ from real acceptance reconstruction")
    costs = bind_original_saved_cost(stage)
    read = _BoundMetadata(stage["proof"]["checked_input_sha256"])
    read.extend(origins["checked_metadata_sha256"])
    originals = {r["original"]["generation_sha256"]: r for r in origins["groups"]}
    proofs = {r["original"]["generation_sha256"]: r for r in stage["proof"]["groups"]}
    require(len(originals) == 17 and set(originals) == set(proofs) == set(costs),
        "Historical result, input and cost inventories disagree")
    return dict(request=request, request_reference=deepcopy(request_ref), plan=plan,
        binding=binding, input_evidence=input_evidence, stage=stage, origins=origins,
        costs=costs, read=read, original_rows=originals, proof_rows=proofs, source=source)


def assess_selected(registry, context, group, generation):
    from .vcsf_operator_numerical_mapping import map_operator_numerical_sources

    require(generation in context["original_rows"], "Selected historical generation is absent")
    row = context["original_rows"][generation]
    original = row["original"]
    checks, failures = {}, {}
    for name, function in (
        ("input", lambda: verify_original_inputs(context["binding"], row, context["read"])),
        ("numerical", lambda: map_operator_numerical_sources(registry.root,
            context["binding"]["runtime_sha256"], original_root(original), original,
            group["parameters"], context["proof_rows"][generation].get("original_parameter_projection"))),
        ("saved_cost", lambda: deepcopy(context["costs"][generation])),
    ):
        try:
            checks[name] = function()
            require(isinstance(checks[name], dict) and checks[name], "Missing reconstructed " + name)
        except (RuntimeError, ValueError) as exc:
            checks[name], failures[name] = None, str(exc)
    qualified = not failures
    remapped = [dict(deepcopy(origin), group_id=group["group_id"],
        original_group_id=origin["group_id"], original_origin_sha256=canonical_hash(origin))
        for origin in row["result_origins"]] if qualified else None
    return dict(group_id=group["group_id"], original_generation_sha256=generation,
        parameters_sha256=group["parameters_sha256"],
        status="qualified_reuse" if qualified else "unresolved",
        reason="exact_bound_original_operator_conditions" if qualified else "selected_original_mapping_unresolved",
        input_identity_verified=checks["input"] is not None,
        numerical_source_mapping_verified=checks["numerical"] is not None,
        original_result_acceptance_verified=True,
        original_saved_cost_attribution_verified=checks["saved_cost"] is not None,
        checks=checks, mapping_failures=failures, result_origins=remapped,
        physical_cost_inheritance=False, trajectory_identity_claim=False)


def _merge_files(files, mapping):
    require(isinstance(mapping, dict), "Missing evidence file inventory")
    for name, digest in mapping.items():
        path = Path(name)
        require(path.is_absolute() and path == path.resolve()
            and not any(p.is_symlink() for p in (path, *path.parents))
            and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest),
            "Evidence inventory contains a noncanonical file or hash")
        require(name not in files or files[name] == digest, "Conflicting bound evidence hashes")
        files[name] = digest


def _references(value, files, root=None):
    if isinstance(value, dict):
        if isinstance(value.get("file"), str) and isinstance(value.get("sha256"), str):
            path = Path(value["file"])
            if path.is_absolute():
                _merge_files(files, {str(path): value["sha256"]})
            elif root is not None:
                _merge_files(files, {str(root / path): value["sha256"]})
        for key, item in value.items():
            if key in ("checked_input_sha256", "checked_metadata_sha256", "checked_auxiliary_sha256"):
                _merge_files(files, item)
            else:
                _references(item, files, root)
    elif isinstance(value, list):
        for item in value:
            _references(item, files, root)


def evidence_files(registry, context, assessments):
    files = {}
    _merge_files(files, context["read"].allowed)
    for value in (context["request_reference"], context["request"], context["binding"], context["stage"],
            context["origins"], context["costs"], context["read"].documents):
        _references(value, files)
    clean_ref = context["binding"]["clean_image_manifest"]
    clean_rows = _bound_json({key: clean_ref[key] for key in ("file", "sha256")})
    _references(clean_rows, files)
    for module in context["stage"]["proof"]["modules"].values():
        _references(module, files, registry.root)
    for assessment in assessments.values():
        _references(assessment, files)
        mapping = assessment["checks"]["numerical"]
        if mapping is not None:
            original = context["original_rows"][assessment["original_generation_sha256"]]["original"]
            root = original_root(original)
            # Bind the historical numerical source files actually compared, not unrelated old code.
            for name, digest in mapping["unchanged_sources"].items():
                _merge_files(files, {str(root / name): digest})
            from .vcsf_core_numerical_mapping import DRIVER
            _merge_files(files, {str(root / DRIVER): original["source_sha256"][DRIVER]})
            _references(mapping["original_builder"], files, root)
            implementation = original["implementation"]["implementation"]
            _merge_files(files, {str(root / implementation): original["source_sha256"][implementation]})
    for name, digest in context["source"].items():
        _merge_files(files, {str(registry.root / name): digest})
    receipt = _bound_json(context["request"]["runtime_receipt"])
    root = Path(context["request"]["runtime_inputs"]["file"]).parent
    manifest_ref = dict(file=str(root / "artifact_manifest.json"), sha256=receipt["artifact_manifest_sha256"])
    manifest = _bound_json(manifest_ref)
    _references(manifest_ref, files)
    for name, row in manifest["artifacts"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Runtime artifact escaped publication")
        _merge_files(files, {str(root / name): row["sha256"]})
    rehash_files(files)
    context["read"].unchanged()
    require(source_snapshot(registry) == context["source"]
        and read_request(context["request_reference"]) == context["request"],
        "Reuse sources or request changed during reconstruction")
    return dict(sorted(files.items()))


def rehash_files(files):
    checked = {}
    _merge_files(checked, files)
    require(checked, "Empty reuse evidence snapshot")
    for name, digest in checked.items():
        require(file_digest(Path(name)) == digest, "Bound reuse evidence changed: " + name)


def assess_reuse(registry, request_ref):
    context = reconstruct_context(registry, request_ref)
    assessments = {}
    for group, selection in zip(context["plan"]["scientific_contract"]["groups"], context["request"]["selections"]):
        generation = selection["original_generation_sha256"]
        if generation is not None:
            assessments[group["group_id"]] = assess_selected(registry, context, group, generation)
    partition = compile_operator_partition(registry, context["plan"], context["origins"]["groups"],
        context["request"]["selections"], assessments)
    files = evidence_files(registry, context, assessments)
    report = dict(schema_version=1, record_type=REPORT_TYPE, protocol=PROTOCOL,
        status="bound_operator_reuse_pending_independent_acceptance",
        request_reference=deepcopy(request_ref), request_sha256=canonical_hash(context["request"]),
        operator_plan_reference=deepcopy(context["request"]["operator_plan"]),
        prepared_plan_sha256=canonical_hash(context["plan"]), science_sha256=context["plan"]["science_sha256"],
        runtime_binding_reference=deepcopy(context["request"]["runtime_inputs"]),
        runtime_binding_sha256=context["binding"]["binding_sha256"],
        input_evidence=context["input_evidence"], source_sha256=context["source"],
        source_snapshot_sha256=canonical_hash(context["source"]),
        original_origins_sha256=context["origins"]["origins_sha256"],
        stage_proof_sha256=context["stage"]["proof_sha256"], partition=partition,
        checked_file_sha256=files, checked_files_sha256=canonical_hash(files),
        max_images=None, synthetic=False,
        **dict.fromkeys(FALSE_FLAGS, False), **dict.fromkeys(ZERO_FIELDS, 0))
    report["reuse_sha256"] = canonical_hash(report)
    return report, context["origins"]


def check_hash(record, key):
    require(record.get(key) == canonical_hash({k: v for k, v in record.items() if k != key}),
        "Invalid canonical record hash: " + key)


def publish(registry, output, records, *, kind):
    """Fresh immutable leaves; exceptions deliberately retain partial evidence."""
    source_snapshot(registry, public=True)
    output = source_binding._plain(output)
    require(kind in ("assessment", "audit") and output.parent == registry.root / "outputs/plans" / (NAMESPACE + "_" + kind)
        and not output.exists(), "Reuse publication requires a fresh scoped direct leaf")
    expected = {"reuse.json", "origins.json"} if kind == "assessment" else {"audit.json"}
    require(set(records) == expected, "Unexpected reuse publication records")
    output.mkdir(parents=True, exist_ok=False)
    for name, value in records.items():
        source_binding._write_json(output / name, value)
    manifest = {name: file_digest(output / name) for name in sorted(records)}
    source_binding._write_json(output / "manifest.json", manifest)
    source_snapshot(registry, public=True)
    primary = records["reuse.json" if kind == "assessment" else "audit.json"]
    rehash_files(primary["checked_file_sha256"])
    require(source_snapshot(registry, public=True) == primary["source_sha256"],
        "Reuse source changed before receipt publication")
    require(all(file_digest(output / name) == digest for name, digest in manifest.items()),
        "Reuse records changed during publication")
    receipt = dict(schema_version=1, kind="vcsf_operator_reuse_" + kind + "_publication",
        artifact_manifest_sha256=file_digest(output / "manifest.json"),
        record_sha256=primary["reuse_sha256" if kind == "assessment" else "audit_sha256"],
        independent_reuse_acceptance=primary["independent_reuse_acceptance"],
        formal_execution_admission=False, scientific_acceptance=False)
    source_binding._write_json(output / "receipt.json", receipt)
    require({p.name for p in output.iterdir()} == expected | {"manifest.json", "receipt.json"},
        "Unexpected reuse publication inventory")
    return dict(output=str(output), receipt=receipt,
        reference=dict(file=str(output / ("reuse.json" if kind == "assessment" else "audit.json")),
            sha256=manifest["reuse.json" if kind == "assessment" else "audit.json"]))


def write_reuse(registry, output, report, origins):
    require(report["independent_reuse_acceptance"] is False, "Producer cannot accept reuse")
    return publish(registry, output, {"reuse.json": report, "origins.json": origins}, kind="assessment")


def read_publication(registry, reference, kind):
    path = source_binding._plain(reference["file"])
    names = {"reuse.json", "origins.json"} if kind == "assessment" else {"audit.json"}
    require(kind in ("assessment", "audit") and path.name == ("reuse.json" if kind == "assessment" else "audit.json")
        and path.parent.parent == registry.root / "outputs/plans" / (NAMESPACE + "_" + kind)
        and {p.name for p in path.parent.iterdir()} == names | {"manifest.json", "receipt.json"}
        and all(p.is_file() and not p.is_symlink() for p in path.parent.iterdir()),
        "Wrong reuse publication namespace or inventory")
    receipt_ref = dict(file=str(path.parent / "receipt.json"), sha256=file_digest(path.parent / "receipt.json"))
    receipt = _bound_json(receipt_ref)
    manifest_ref = dict(file=str(path.parent / "manifest.json"), sha256=receipt["artifact_manifest_sha256"])
    manifest = _bound_json(manifest_ref)
    require(set(manifest) == names and manifest[path.name] == reference["sha256"], "Publication manifest identity mismatch")
    files = {str(path.parent / name): digest for name, digest in manifest.items()}
    _references([receipt_ref, manifest_ref], files)
    rehash_files(files)
    record = _bound_json(reference)
    check_hash(record, "reuse_sha256" if kind == "assessment" else "audit_sha256")
    expected = dict(schema_version=1, kind="vcsf_operator_reuse_" + kind + "_publication",
        artifact_manifest_sha256=manifest_ref["sha256"],
        record_sha256=record["reuse_sha256" if kind == "assessment" else "audit_sha256"],
        independent_reuse_acceptance=record["independent_reuse_acceptance"],
        formal_execution_admission=False, scientific_acceptance=False)
    require(canonical_hash(receipt) == canonical_hash(expected), "Publication receipt changed its identity or scope")
    return record, files


def verify_reuse_publication(registry, request_ref, report_ref, audit_ref):
    from .vcsf_operator_reuse_audit import verify_reuse_publication as verify

    return verify(registry, request_ref, report_ref, audit_ref)
