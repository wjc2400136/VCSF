"""Operator execution layout; input/reuse admission is a separate prerequisite.

These deterministic profile helpers neither qualify historical results nor
admit execution. The public runner must verify the complete bound execution
plan and its independent admission before using them.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

from ..io import file_digest
from ..registry import Registry
from .vcsf_efficacy_contract import child, verify_environment
from .vcsf_operator_factorial_plan import PROTOCOL, compile_operator_factorial_plan
from .vcsf_research_plan import canonical_hash, require


EXECUTION_PROTOCOL = "vcsf_single_source_operator_factorial_execution"
ENTRYPOINT = "experiments/vcsf_operator_factorial.py"
ALIAS = "vcsf_operator_factorial_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "single_source_operator_factorial_execution"
ALLOW_EMPTY_LANES = False
READINESS_STATUS = "independently_verified_operator_execution_readiness"
FOUR_DEVICE_LAUNCH_STATUS = "independently_verified_operator_four_device_smoke_launch"
FORMAL_READINESS_STATUS = "independently_verified_operator_real_runtime_readiness"
READINESS_CHECKS = (
    "compile_and_registry", "operator_forward_backward_rng", "exact_execution_resolver",
    "one_device_complete_jobs", "two_device_complete_jobs", "worker_failure_containment",
    "full_stage_keep_all_lifecycle",
)
FOUR_DEVICE_LAUNCH_CHECKS = (
    "four_device_plan_and_admission", "operator_forward_backward_rng", "exact_execution_resolver",
    "four_device_complete_jobs", "four_device_failure_containment", "four_device_keep_all_lifecycle",
)


def balanced_lanes(groups, sources, devices):
    """Schedule complete operator groups without broadening legacy profiles."""
    require(isinstance(devices, (list, tuple)) and len(devices) in (1, 2, 4)
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices),
        "Select one, two or four distinct explicit physical CUDA devices")
    require(sources == ["faster_rcnn_r50"] and groups
        and all(g["source"] == sources[0] for g in groups)
        and all(type(g["group_id"]) is int and g["group_id"] > 0 for g in groups)
        and len({g["group_id"] for g in groups}) == len(groups),
        "Invalid or duplicate source-only operator group")
    require(len(groups) >= len(devices), "Operator execution does not admit empty lanes")
    return [groups[i::len(devices)] for i in range(len(devices))]


def verify_device_admission(receipt, devices):
    """An allowed protocol mode is not proof that its device count was tested."""
    require(isinstance(devices, (list, tuple)) and len(devices) in (1, 2, 4)
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices)
        and len(set(devices)) == len(devices), "Invalid requested operator devices")
    require(isinstance(receipt, dict) and isinstance(receipt.get("device_counts"), list)
        and all(type(n) is int for n in receipt["device_counts"])
        and len(devices) in receipt["device_counts"],
        "Readiness evidence does not admit the requested operator device count")


def read_bound(path, expected_sha256):
    """Read exact plain JSON bytes, rejecting duplicate keys and nonfinite values."""
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Execution evidence must use plain paths without traversal")
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
        "Malformed execution evidence SHA-256")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Bound execution evidence changed")

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate execution JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise RuntimeError("Nonfinite execution JSON value: " + value)

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)
    require(isinstance(value, dict), "Execution evidence must be a JSON object")
    return value


def read_reference(registry, reference):
    require(isinstance(reference, dict) and set(reference) == {"file", "sha256"}
        and isinstance(reference["file"], str), "References require exactly file and sha256")
    path = Path(reference["file"])
    path = path if path.is_absolute() else child(registry.root, reference["file"])
    return read_bound(path, reference["sha256"])


def runtime_snapshot(root):
    root = Path(root).resolve()
    names = {p.relative_to(root).as_posix() for directory, suffixes in (
        ("src", (".py",)), ("configs", (".yaml", ".yml")),
        ("tests", (".py",)), ("tools", (".py",)))
        for p in (root / directory).rglob("*") if p.is_file() and p.suffix in suffixes}
    names.update((ENTRYPOINT, "environment.yml", "requirements/locked-cu118.txt"))
    return {name: file_digest(child(root, name)) for name in sorted(names)}


_ROOT = Path(__file__).resolve().parents[3]


def verify_process_source(registry, snapshot):
    from . import vcsf_ablation_execution_contract as source_binding

    source_binding._verify_process_binding(registry)
    require(registry.root == _ROOT and runtime_snapshot(registry.root) == snapshot,
        "Execution process source changed")
    entry = sys.modules.get("__main__")
    require(getattr(entry, "__file__", None) == str(_ROOT / ENTRYPOINT)
        and getattr(entry, "_ENTRY_SOURCE_SHA256", None) == snapshot[ENTRYPOINT],
        "Operator execution requires the unchanged source-only public entry")


def compile_runtime_plan(registry, request_ref, report_ref, audit_ref, *, max_images=None,
        capacity_contract):
    """Bind current input/reuse publications, never issue execution admission."""
    from .vcsf_operator_reuse_audit import verify_reuse_publication
    from .vcsf_efficacy_runner import keep_all_policy

    require(max_images is None or type(max_images) is int and 0 < max_images < 5000,
        "Invalid operator diagnostic image limit")
    require(isinstance(capacity_contract, dict) and set(capacity_contract) == {
        "policy", "group_budget_bytes", "failure_reserve_bytes", "required_free_bytes"},
        "Capacity requires explicit complete byte budgets and failure reserve")
    refs = [request_ref, report_ref, audit_ref]
    saved = [read_reference(registry, ref) for ref in refs]
    require(len({str(Path(ref["file"]).absolute()) for ref in refs}) == 3,
        "Reuse request, report and independent audit must be separate publications")
    context = verify_reuse_publication(registry, request_ref, report_ref, audit_ref)
    require(isinstance(context, dict) and set(context) == {"request", "report", "binding", "input_evidence"},
        "Unexpected independent reuse consumer interface")
    request, report, binding = context["request"], context["report"], context["binding"]
    require(canonical_hash([request, report]) == canonical_hash(saved[:2]),
        "Reuse verifier returned another publication")
    prepared = read_reference(registry, request["operator_plan"])
    inputs = read_reference(registry, request["runtime_inputs"])
    read_reference(registry, request["runtime_receipt"])
    require(canonical_hash(inputs) == canonical_hash(binding), "Reuse context has another input binding")
    current = compile_operator_factorial_plan(registry, devices=prepared["requested_devices"])
    require(canonical_hash(prepared) == canonical_hash(current), "Original operator plan changed")
    science, partition = prepared["scientific_contract"], report["partition"]
    groups = science["groups"]
    new = [g["group_id"] for g in groups if g["requested_new"]]
    reused = [g["group_id"] for g in groups if not g["requested_new"]]
    require(len(groups) == 8 and len(new) == len(reused) == 4
        and science["max_images"] is None and science["diagnostic_only"] is False
        and all(type(g["images"]) is int and g["images"] == 5000 for g in groups),
        "Execution requires the original full5000 four-old/four-new operator design")
    require(partition.get("partition_published") is True
        and canonical_hash(partition.get("assessed_partition")) == canonical_hash(dict(
            qualified_reuse=reused, requires_new=new, unresolved=[]))
        and canonical_hash(partition.get("new_group_ids")) == canonical_hash(new)
        and canonical_hash(partition.get("qualified_reused_group_ids")) == canonical_hash(reused)
        and partition.get("unresolved_group_ids") == []
        and partition.get("prepared_plan_sha256") == canonical_hash(prepared)
        and partition.get("science_sha256") == prepared["science_sha256"]
        and partition.get("partition_sha256") == canonical_hash({
            k: v for k, v in partition.items() if k != "partition_sha256"}),
        "Only independently qualified exact4 reuse plus exact4 new cells can execute")
    definition = deepcopy(registry.protocols[PROTOCOL])
    require(definition.get("execution_authorized") is True
        and definition.get("device_counts") == [1, 2, 4]
        and definition.get("execution_lifecycle") == "full_stage_keep_all_v1"
        and definition.get("execution_entrypoint") == ENTRYPOINT
        and definition.get("prediction_archive") in ("gzip", "plain")
        and definition.get("isolated_candidate", {}).get("execution_alias") == ALIAS,
        "Missing registered operator execution interface and keep-all declaration")
    require(binding["science_sha256"] == prepared["science_sha256"]
        and binding["max_images"] is None and binding["diagnostic_only"] is False
        and binding["full_split_images"] == 5000,
        "Operator execution requires full-split published inputs")
    snapshot = runtime_snapshot(registry.root)
    verify_process_source(registry, snapshot)
    selected = [g for g in groups if g["requested_new"]]
    plan = dict(schema_version=1, record_type=EXECUTION_PROTOCOL, protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, status="prepared_operator_execution_pending_independent_readiness",
        definition=definition, definition_sha256=canonical_hash(definition),
        prepared_operator_plan=deepcopy(prepared), prepared_plan=deepcopy(science),
        prepared_plan_sha256=canonical_hash(science), science_sha256=prepared["science_sha256"],
        groups=deepcopy(groups), sources=list(science["sources"]), targets=list(science["targets"]),
        stage="operator_factorial", max_images=max_images, diagnostic_only=max_images is not None,
        new_group_ids=new, reused_group_ids=reused, new_groups=len(new),
        new_images=sum(g["images"] if max_images is None else max_images for g in selected),
        new_evaluations=sum(len(g["targets"]) for g in selected),
        reuse_request=deepcopy(request_ref), reuse_report=deepcopy(report_ref), reuse_audit=deepcopy(audit_ref),
        operator_plan=deepcopy(request["operator_plan"]), runtime_inputs=deepcopy(request["runtime_inputs"]),
        runtime_receipt=deepcopy(request["runtime_receipt"]),
        input_binding_sha256=binding["binding_sha256"], input_evidence=deepcopy(context["input_evidence"]),
        reuse_assessment_sha256=partition["partition_sha256"],
        annotation_sha256=binding["annotation"]["sha256"], image_ids_sha256=binding["image_ids_sha256"],
        seed_mapping_sha256=binding["seed_mapping_sha256"],
        clean_image_manifest=deepcopy(binding["clean_image_manifest"]),
        checkpoint_sha256={t: binding["checkpoints"][t]["sha256"] for t in science["targets"]},
        model_config_bindings=deepcopy(binding["model_configs"]),
        packages=deepcopy(binding["packages"]), base_commit=binding["base_commit"],
        runtime_sha256=snapshot, runtime_snapshot_sha256=canonical_hash(snapshot),
        analysis=deepcopy(science["analysis"]), analysis_sha256=canonical_hash(science["analysis"]),
        selection_namespace=definition["selection_namespace"],
        execution_lifecycle=definition["execution_lifecycle"], capacity_contract=deepcopy(capacity_contract),
        payload_authorization=dict(explicit_owner_confirmation=False, deletion_authorized=False,
            historical_roots_allowed=False, failed_or_partial_groups_allowed=False),
        runner_armed=False, formal_execution_admission=False, scientific_acceptance=False,
        independent_confirmation=False, formal_metrics_eligible=False, automatic_promotion=False,
        old_run_restart=False)
    policy = keep_all_policy(plan, selected, max_images)
    plan["keep_all_policy_sha256"] = canonical_hash(policy)
    execution_groups(plan, max_images)
    require(runtime_snapshot(registry.root) == snapshot, "Execution source changed during compilation")
    for reference, value in zip(refs, saved):
        require(canonical_hash(read_reference(registry, reference)) == canonical_hash(value),
            "Reuse publication changed during execution preparation")
    return plan


def verify_execution_plan(registry, plan):
    require(isinstance(plan, dict) and plan.get("record_type") == EXECUTION_PROTOCOL,
        "Not an operator runtime execution plan")
    expected = compile_runtime_plan(registry, plan["reuse_request"], plan["reuse_report"],
        plan["reuse_audit"], max_images=plan["max_images"], capacity_contract=plan["capacity_contract"])
    require(canonical_hash(plan) == canonical_hash(expected),
        "Complete operator execution plan or its bound evidence changed")
    return plan


def publish_execution_plan(registry, output, plan):
    verify_execution_plan(registry, plan)
    output = Path(output).absolute()
    parent = registry.root / "outputs/plans" / EXECUTION_PROTOCOL
    require(output.parent == parent, "Plan publication requires a fresh direct protocol leaf")
    child(registry.root, output.relative_to(registry.root))
    output.mkdir(parents=True, exist_ok=False)
    path = output / "execution_plan.json"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(plan, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    reference = dict(file=str(path), sha256=file_digest(path))
    require(canonical_hash(read_reference(registry, reference)) == canonical_hash(plan),
        "Execution plan publication readback differs")
    return reference


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    """Engineering evidence admits max1 only; full5000 needs real runtime readback."""
    require(max_images is None or type(max_images) is int and max_images == 1,
        "Operator admission supports only max1 diagnostic or qualified full5000 execution")
    if max_images == 1:
        return verify_engineering_admission(registry, plan, plan_hash, path, digest, max_images)
    verify_execution_plan(registry, plan)
    verify_environment(plan)
    execution_groups(plan, None)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from tools.audit_vcsf_operator_runtime_readiness import verify_runtime_admission

    return verify_runtime_admission(registry, plan, plan_hash, path, digest)


def verify_engineering_admission(registry, plan, plan_hash, path, digest, max_images):
    """Consume independent source-bound readiness, not input/reuse/result acceptance.

    The independent ODA validation owner publishes this receipt only after the
    named checks execute. This runtime module deliberately cannot mint it.
    Evidence records bind each check to this plan, source, environment and scope.
    """
    require(type(max_images) is int and max_images == 1
        and type(plan.get("max_images")) is int and plan["max_images"] == 1,
        "Engineering readiness admits only max_images=1 diagnostics, never full5000")
    verify_execution_plan(registry, plan)
    verify_environment(plan)
    execution_groups(plan, max_images)
    path = Path(path).absolute()
    require(not (path.parent / "failure.json").exists()
        and not (path.parent / "failure.json").is_symlink(),
        "Failed readiness publication cannot admit execution")
    receipt = read_bound(path, digest)
    launch_only = receipt.get("status") == FOUR_DEVICE_LAUNCH_STATUS
    required_checks = FOUR_DEVICE_LAUNCH_CHECKS if launch_only else READINESS_CHECKS
    admitted_counts = [4] if launch_only else [1, 2]
    expected_keys = {"schema_version", "status", "execution_plan", "runtime_snapshot_sha256",
        "input_binding_sha256", "reuse_audit", "max_images", "device_counts", "checks",
        "auditor_source", "independent_of_plan_producer", "errors", "execution_admitted",
        "result_acceptance", "scientific_acceptance", "automatic_promotion"}
    require(set(receipt) == expected_keys and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == (FOUR_DEVICE_LAUNCH_STATUS if launch_only else READINESS_STATUS)
        and receipt["execution_plan"]["sha256"] == plan_hash
        and canonical_hash(read_reference(registry, receipt["execution_plan"])) == canonical_hash(plan)
        and receipt["runtime_snapshot_sha256"] == plan["runtime_snapshot_sha256"]
        and receipt["input_binding_sha256"] == plan["input_binding_sha256"]
        and receipt["reuse_audit"] == plan["reuse_audit"]
        and canonical_hash(receipt["max_images"]) == canonical_hash(max_images)
        and canonical_hash(receipt["device_counts"]) == canonical_hash(admitted_counts) and receipt["errors"] == []
        and receipt["independent_of_plan_producer"] is True and receipt["execution_admitted"] is True
        and all(receipt[k] is False for k in ("result_acceptance", "scientific_acceptance", "automatic_promotion")),
        "Missing independent source-bound operator execution-readiness admission")
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from tools import audit_vcsf_operator_readiness as independent
    selectors = independent.FOUR_DEVICE_LAUNCH_CHECKS if launch_only else independent.CHECKS

    auditor = receipt["auditor_source"]
    require(isinstance(auditor, dict) and set(auditor) == {"file", "sha256"},
        "Readiness auditor requires an immutable source reference")
    auditor_path = Path(auditor["file"])
    auditor_path = auditor_path if auditor_path.is_absolute() else child(registry.root, auditor["file"])
    relative = auditor_path.relative_to(registry.root).as_posix()
    require(relative == independent.AUDITOR
        and Path(independent.__file__).resolve() == registry.root / independent.AUDITOR
        and file_digest(child(registry.root, relative)) == auditor["sha256"]
        and plan["runtime_sha256"].get(relative) == auditor["sha256"],
        "Readiness audit must bind the independently implemented registered auditor source")
    checks = receipt["checks"]
    require(isinstance(checks, dict) and set(checks) == set(required_checks),
        "Incomplete independent execution-readiness checks")
    require(len({canonical_hash(ref) for ref in checks.values()}) == len(required_checks),
        "Each readiness gate requires its own executed evidence")
    for name in required_checks:
        evidence = read_reference(registry, checks[name])
        require(evidence.get("check") == name and evidence.get("status") == "passed"
            and evidence.get("executed") is True and type(evidence.get("returncode")) is int
            and evidence["returncode"] == 0 and evidence.get("errors") == []
            and evidence.get("execution_plan_sha256") == plan_hash
            and evidence.get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"]
            and evidence.get("auditor_source") == auditor
            and evidence.get("base_commit") == plan["base_commit"]
            and evidence.get("packages") == plan["packages"]
            and evidence.get("python_version") == "3.8.20"
            and evidence.get("environment") == "oda"
            and canonical_hash(evidence.get("max_images")) == canonical_hash(max_images)
            and evidence.get("scientific_acceptance") is False
            and evidence.get("scope") == "engineering_test_adapters_not_real_detector_jobs"
            and type(evidence.get("detector_calls")) is int and evidence["detector_calls"] == 0
            and type(evidence.get("formal_AP_evaluations")) is int and evidence["formal_AP_evaluations"] == 0
            and evidence.get("test_selectors") == selectors[name],
            "Readiness check lacks actual matching ODA evidence: " + name)
        require(Path(checks[name]["file"]).absolute() == path.parent / (name + ".json"),
            "Readiness evidence escaped its independent audit publication")
        for field, suffix in (("junit", ".xml"), ("log", ".log"),
                ("collected_nodeids", "_collected.json"), ("executed_nodeids", "_executed.json")):
            ref = evidence.get(field)
            require(isinstance(ref, dict) and set(ref) == {"file", "sha256"}
                and isinstance(ref["file"], str)
                and Path(ref["file"]).absolute() == path.parent / (name + suffix)
                and isinstance(ref["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", ref["sha256"]),
                "Readiness test output reference is missing or foreign")
            artifact = Path(ref["file"]).absolute()
            require(not any(p.is_symlink() for p in (artifact, *artifact.parents))
                and file_digest(artifact) == ref["sha256"], "Readiness test output bytes changed")
        require(canonical_hash(independent._junit(evidence["junit"]["file"]))
            == canonical_hash(evidence.get("test_cases")), "Readiness JUnit case enumeration changed")
        require(independent.verify_collection(evidence["collected_nodeids"], evidence["executed_nodeids"],
            evidence["test_cases"], auditor["sha256"]) == evidence.get("nodeids"),
            "Readiness collection differs from the complete actually executed nodeids")
    require(not (path.parent / "failure.json").exists()
        and not (path.parent / "failure.json").is_symlink()
        and canonical_hash(read_bound(path, digest)) == canonical_hash(receipt),
        "Readiness publication failed or changed during admission")
    return receipt


def execution_groups(plan, max_images):
    """Select only the registered missing cells, never failed reuse candidates."""
    require(isinstance(plan, dict) and plan.get("record_type") == EXECUTION_PROTOCOL,
        "Not an operator execution layout")
    require(max_images is None or type(max_images) is int and 0 < max_images < 5000,
        "Invalid operator diagnostic image limit")
    require(canonical_hash(plan.get("max_images")) == canonical_hash(max_images),
        "Operator execution request changed its registered image limit")
    prepared = plan["prepared_operator_plan"]
    current = compile_operator_factorial_plan(Registry(), devices=prepared["requested_devices"])
    require(canonical_hash(prepared) == canonical_hash(current),
        "Operator layout is not the complete current registered scientific plan")
    science = prepared["scientific_contract"]
    require(prepared["science_sha256"] == canonical_hash(science)
        and plan["science_sha256"] == prepared["science_sha256"],
        "Operator layout changed its scientific identity")
    require(science["max_images"] is None and science["diagnostic_only"] is False,
        "Execution layout must retain the original full-split scientific design")
    groups = science["groups"]
    require(canonical_hash(plan["groups"]) == canonical_hash(groups),
        "Operator execution groups differ from the complete original design")
    require(all(type(g["group_id"]) is int and g["group_id"] > 0
        and type(g["requested_new"]) is bool for g in groups)
        and len({g["group_id"] for g in groups}) == len(groups),
        "Invalid or duplicate operator group identity")
    new = [g["group_id"] for g in groups if g["requested_new"]]
    reused = [g["group_id"] for g in groups if not g["requested_new"]]
    require(canonical_hash(plan["new_group_ids"]) == canonical_hash(new)
        and canonical_hash(plan["reused_group_ids"]) == canonical_hash(reused)
        and new and reused,
        "Unresolved historical cells cannot become new operator jobs")
    require(canonical_hash(plan["sources"]) == canonical_hash(science["sources"])
        and canonical_hash(plan["targets"]) == canonical_hash(science["targets"]),
        "Operator source or target panel changed")
    return deepcopy([g for g in groups if g["requested_new"]])


def scheduled_stages(plan, selected):
    require(canonical_hash(selected) == canonical_hash(execution_groups(plan, plan["max_images"])),
        "Operator stage must cover the entire registered new partition in order")
    return [dict(name="operator_factorial", group_ids=list(plan["new_group_ids"]))]


def execution_metadata(plan, plan_sha256, group, max_images):
    selected = execution_groups(plan, max_images)
    require(any(canonical_hash(group) == canonical_hash(candidate) for candidate in selected),
        "Foreign, altered or historical-reuse operator group")
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL,
        stage="operator_factorial", variant=group["variant"], group_id=group["group_id"],
        seed=group["seed"], execution_plan_sha256=plan_sha256,
        science_sha256=plan["science_sha256"],
        prepared_plan_sha256=canonical_hash(plan["prepared_operator_plan"]),
        runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        input_binding_sha256=plan["input_binding_sha256"],
        reuse_assessment_sha256=plan["reuse_assessment_sha256"],
        factor_bits=deepcopy(group["factor_bits"]), parameters_sha256=group["parameters_sha256"],
        max_images=max_images, diagnostic_only=max_images is not None,
        formal_metrics_eligible=False, independent_confirmation=False,
        automatic_promotion=False)
