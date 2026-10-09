"""Execute bounded ODA engineering checks and publish source-bound admission.

This is not detector execution, result acceptance, or scientific acceptance.
The public operator entry supplies its verified plan and source-only loader.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from lgp.io import file_digest
from lgp.runners import vcsf_operator_execution_contract as contract
from lgp.runners.vcsf_research_plan import canonical_hash, require


AUDITOR = "tools/audit_vcsf_operator_readiness.py"
ISOLATION = "tests/test_vcsf_operator_execution_isolation.py"
CHECKS = {
    "compile_and_registry": ["tests/test_vcsf_operator_factorial_plan.py",
        "tests/test_vcsf_operator_execution_contract.py",
        "tests/test_vcsf_operator_numerical_mapping.py",
        "tests/test_vcsf_operator_readiness.py",
        "tests/test_vcsf_core_numerical_mapping.py"],
    "operator_forward_backward_rng": ["tests/test_vcsf_operator_factorial_isolated.py"],
    "exact_execution_resolver": [ISOLATION, "-k",
        "not operator_complete_jobs and not operator_worker_failure and not operator_keep_all"],
    "one_device_complete_jobs": [ISOLATION + "::test_operator_complete_jobs_one_device"],
    "two_device_complete_jobs": [ISOLATION + "::test_operator_complete_jobs_two_devices"],
    "worker_failure_containment": [ISOLATION + "::test_operator_worker_failure_preserves_payload"],
    "full_stage_keep_all_lifecycle": [ISOLATION + "::test_operator_keep_all_lifecycle"],
}
FOUR_DEVICE_LAUNCH_CHECKS = {
    "four_device_plan_and_admission": ["tests/test_vcsf_operator_factorial_plan.py",
        "tests/test_vcsf_operator_execution_contract.py::test_operator_device_lanes_preserve_every_complete_group",
        "tests/test_vcsf_operator_execution_contract.py::test_operator_device_lanes_reject_unregistered_or_ambiguous_devices",
        "tests/test_vcsf_operator_execution_contract.py::test_operator_device_lanes_reject_empty_worker",
        "tests/test_vcsf_operator_execution_contract.py::test_operator_device_count_requires_matching_verified_evidence",
        "tests/test_vcsf_operator_execution_contract.py::test_four_device_launch_admission_is_not_full_release"],
    "operator_forward_backward_rng": CHECKS["operator_forward_backward_rng"],
    "exact_execution_resolver": CHECKS["exact_execution_resolver"],
    "four_device_complete_jobs": [ISOLATION + "::test_operator_complete_jobs_four_devices"],
    "four_device_failure_containment": [ISOLATION + "::test_operator_worker_failure_four_devices"],
    "four_device_keep_all_lifecycle": [ISOLATION + "::test_operator_keep_all_four_devices"],
}


def _write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return dict(file=str(path), sha256=file_digest(path))


def _junit(path):
    root = ET.parse(str(path)).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    cases = [case for suite in suites for case in suite.findall("testcase")]
    require(suites and cases and sum(int(s.get("tests", "0")) for s in suites) == len(cases),
        "Readiness JUnit does not enumerate all executed cases")
    require(all(int(s.get(key, "0")) == 0 for s in suites for key in ("errors", "failures", "skipped"))
        and all(not list(c) for c in cases), "Readiness requires no failed, skipped or errored cases")
    identities = [(c.get("classname", ""), c.get("name", "")) for c in cases]
    require(all(name for _, name in identities) and len(identities) == len(set(identities)),
        "Readiness case identities are empty or duplicated")
    return [dict(classname=module, name=name) for module, name in identities]


def pytest_collection_finish(session):
    """Capture selected nodeids in both collection-only and executed children."""
    destination = os.environ.get("LGP_OPERATOR_COLLECTION_RECEIPT")
    if destination:
        nodes = [item.nodeid for item in session.items]
        require(nodes and len(nodes) == len(set(nodes)), "Empty or duplicate readiness collection")
        _write(destination, dict(nodeids=nodes, collection_only=bool(session.config.option.collectonly),
            auditor_sha256=file_digest(Path(__file__))))


def verify_collection(expected_path, executed_path, cases, auditor_sha256):
    expected = contract.read_bound(expected_path["file"], expected_path["sha256"])
    executed = contract.read_bound(executed_path["file"], executed_path["sha256"])
    require(set(expected) == set(executed) == {"nodeids", "collection_only", "auditor_sha256"}
        and expected["collection_only"] is True and executed["collection_only"] is False
        and expected["auditor_sha256"] == executed["auditor_sha256"] == auditor_sha256
        and expected["nodeids"] == executed["nodeids"], "Readiness selected nodeid inventory changed")
    nodes = expected["nodeids"]
    require(isinstance(nodes, list) and nodes and all(isinstance(n, str) and "::" in n for n in nodes)
        and len(nodes) == len(set(nodes))
        and sorted(n.rsplit("::", 1)[-1] for n in nodes) == sorted(c["name"] for c in cases),
        "Readiness executed JUnit differs from complete collected nodeids")
    return nodes


def audit_readiness(registry, plan_path, plan_sha256, output, *, launch_only=False):
    """Run real checks once; failures preserve their fresh root without admission."""
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda" and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        "Operator readiness runs only in authorized ODA")
    plan_path = Path(plan_path).absolute()
    plan = contract.read_bound(plan_path, plan_sha256)
    contract.verify_execution_plan(registry, plan)
    contract.verify_environment(plan)
    require(type(plan["max_images"]) is int and plan["max_images"] == 1,
        "Engineering-only readiness admits a one-image diagnostic, never formal execution")
    snapshot = plan["runtime_sha256"]
    contract.verify_process_source(registry, snapshot)
    require(set(CHECKS) == set(contract.READINESS_CHECKS), "Readiness gate catalogue drifted")
    require(type(launch_only) is bool, "Launch-only selection must be explicit")
    require(set(FOUR_DEVICE_LAUNCH_CHECKS) == set(contract.FOUR_DEVICE_LAUNCH_CHECKS),
        "Four-device launch catalogue drifted")
    selected_checks = FOUR_DEVICE_LAUNCH_CHECKS if launch_only else CHECKS
    require(Path(__file__).resolve() == registry.root / AUDITOR
        and file_digest(registry.root / AUDITOR) == snapshot[AUDITOR], "Readiness auditor source changed")
    auditor = dict(file=str(registry.root / AUDITOR), sha256=snapshot[AUDITOR])
    output = Path(output).absolute()
    require(output.parent == registry.root / "outputs/diagnostics" / contract.EXECUTION_PROTOCOL,
        "Readiness requires a fresh direct diagnostic leaf")
    contract.child(registry.root, output.relative_to(registry.root))
    output.mkdir(parents=True, exist_ok=False)
    plan_ref = dict(file=str(plan_path), sha256=plan_sha256)
    evidence = {}
    started = datetime.now(timezone.utc).isoformat()
    _write(output / "request.json", dict(execution_plan=plan_ref, auditor_source=auditor,
        checks=selected_checks, started_at=started,
        scope="four_device_smoke_launch_not_full_release" if launch_only else
            "engineering_tests_with_test_adapters_not_real_detector_jobs",
        detector_calls=0, formal_AP_evaluations=0, scientific_acceptance=False))
    try:
        for relative in snapshot:
            if relative.endswith(".py"):
                compile((registry.root / relative).read_bytes(), str(registry.root / relative),
                    "exec", dont_inherit=True)
        for name, selectors in selected_checks.items():
            contract.verify_process_source(registry, snapshot)
            xml = output / (name + ".xml")
            log = output / (name + ".log")
            command = [sys.executable, "-B", str(registry.root / contract.ENTRYPOINT),
                "--test-worker", "-c", "/dev/null", "--rootdir", str(registry.root),
                "--noconftest", "-q", "-p", "no:cacheprovider",
                "-p", "tools.audit_vcsf_operator_readiness", *selectors]
            environment = dict(os.environ)
            for key in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_CURRENT_TEST"):
                environment.pop(key, None)
            environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTHONPATH=os.pathsep.join((str(registry.root / "src"), str(registry.root))))
            collected = output / (name + "_collected.json")
            executed = output / (name + "_executed.json")
            environment["LGP_OPERATOR_COLLECTION_RECEIPT"] = str(collected)
            with (output / (name + "_collection.log")).open("xb") as handle:
                collection = subprocess.run(command + ["--collect-only"], cwd=str(registry.root),
                    env=environment, stdout=handle, stderr=subprocess.STDOUT, timeout=300, check=False)
            require(collection.returncode == 0, "Readiness collection failed: " + name)
            expected_ref = dict(file=str(collected), sha256=file_digest(collected))
            environment["LGP_OPERATOR_COLLECTION_RECEIPT"] = str(executed)
            command += ["--junitxml=" + str(xml)]
            with log.open("xb") as handle:
                process = subprocess.Popen(command, cwd=str(registry.root), env=environment,
                    stdout=handle, stderr=subprocess.STDOUT)
                _write(output / (name + "_launch.json"), dict(pid=process.pid,
                    started_at=datetime.now(timezone.utc).isoformat(), command=command,
                    execution_plan=plan_ref, auditor_source=auditor))
                try:
                    returncode = process.wait(timeout=3600)
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
            require(returncode == 0, "Readiness check failed: " + name)
            cases = _junit(xml)
            executed_ref = dict(file=str(executed), sha256=file_digest(executed))
            nodes = verify_collection(expected_ref, executed_ref, cases, auditor["sha256"])
            contract.verify_process_source(registry, snapshot)
            require(file_digest(plan_path) == plan_sha256, "Readiness plan changed during checks")
            evidence[name] = _write(output / (name + ".json"), dict(check=name,
                status="passed", executed=True, returncode=returncode, errors=[],
                execution_plan_sha256=plan_sha256, runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
                auditor_source=auditor, base_commit=plan["base_commit"], packages=plan["packages"],
                python_version="3.8.20", environment="oda", max_images=plan["max_images"],
                scientific_acceptance=False, scope="engineering_test_adapters_not_real_detector_jobs",
                detector_calls=0, formal_AP_evaluations=0, test_cases=cases,
                collected_nodeids=expected_ref, executed_nodeids=executed_ref, nodeids=nodes,
                test_selectors=selectors, junit=dict(file=str(xml), sha256=file_digest(xml)),
                log=dict(file=str(log), sha256=file_digest(log))))
        receipt = dict(schema_version=1,
            status=contract.FOUR_DEVICE_LAUNCH_STATUS if launch_only else contract.READINESS_STATUS,
            execution_plan=plan_ref,
            runtime_snapshot_sha256=plan["runtime_snapshot_sha256"], input_binding_sha256=plan["input_binding_sha256"],
            reuse_audit=plan["reuse_audit"], max_images=plan["max_images"],
            device_counts=[4] if launch_only else [1, 2],
            checks=evidence, auditor_source=auditor, independent_of_plan_producer=True,
            errors=[], execution_admitted=True, result_acceptance=False,
            scientific_acceptance=False, automatic_promotion=False)
        reference = _write(output / "admission.json", receipt)
        contract.verify_admission(registry, plan, plan_sha256, reference["file"], reference["sha256"], plan["max_images"])
        _write(output / "terminal.json", dict(status="verified_four_device_smoke_launch_not_full_release"
            if launch_only else "verified_engineering_readiness_not_result_acceptance",
            admission=reference, started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
            checks=list(evidence), errors=[], scientific_acceptance=False))
        return reference
    except BaseException as exc:
        _write(output / "failure.json", dict(status="failed_preserved_no_valid_admission",
            error_type=type(exc).__name__, error=str(exc), completed_checks=list(evidence),
            execution_plan=plan_ref, scientific_acceptance=False))
        raise
