"""Reverify operator input publications through their source-bound public entry."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import subprocess
import sys

from . import vcsf_ablation_execution_contract as source_binding
from . import vcsf_operator_runtime_inputs as runtime
from .vcsf_ablation_runtime_inputs import _file, _pinned_json
from .vcsf_research_plan import canonical_hash, require


def _source_snapshot(registry):
    source_binding._verify_process_binding(registry)
    require(registry.root == runtime._ROOT, "Operator consumer requires its own source workspace")
    entry = _file(registry.root / runtime.ENTRYPOINT)
    require(entry["sha256"] == runtime._ENTRY_HASH, "Operator input entry changed after import")
    return dict(sorted(dict(source_binding._IMPORTED_SOURCE,
        **{runtime.ENTRYPOINT: entry["sha256"]}).items()))


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Operator verifier returned duplicate JSON keys")
        result[key] = value
    return result


def verify_operator_inputs_for_consumer(registry, plan_path, plan_sha256,
        binding_path, binding_sha256, receipt_path, receipt_sha256, *, timeout_seconds=3600):
    """Return checked inputs and invocation evidence, never reuse/admission authority.

    This consumer must itself run with source-only imports. The child performs
    complete publication verification in its required __main__ context. A failed
    or timed-out invocation propagates without retry or publication writes.
    """
    runtime.verify_server()
    require(sys.version_info[:3] == (3, 8, 20), "Operator consumer requires Python 3.8.20")
    require(type(timeout_seconds) is int and timeout_seconds > 0,
        "Operator verifier timeout must be a positive integer")
    paths = [source_binding._plain(path) for path in (plan_path, binding_path, receipt_path)]
    pins = [plan_sha256, binding_sha256, receipt_sha256]
    source = _source_snapshot(registry)
    initial = [_pinned_json(path, pin) for path, pin in zip(paths, pins)]
    plan, binding, receipt = initial
    require(isinstance(plan, dict) and isinstance(binding, dict) and isinstance(receipt, dict),
        "Operator consumer requires three bound JSON objects")
    require(binding.get("operator_plan_file") == _file(paths[0])
        and binding.get("runtime_sha256") == source
        and binding.get("runtime_snapshot_sha256") == canonical_hash(source),
        "Operator publication belongs to another plan or source snapshot")
    limit = binding.get("max_images")
    require(limit is None or type(limit) is int and 0 < limit < 5000,
        "Operator publication has an invalid diagnostic limit")
    command = [sys.executable, str(registry.root / runtime.ENTRYPOINT),
        "--operator-plan", str(paths[0]), "--operator-plan-sha256", plan_sha256,
        "--runtime-inputs", str(paths[1]), "--runtime-inputs-sha256", binding_sha256,
        "--receipt", str(paths[2]), "--receipt-sha256", receipt_sha256]
    if limit is not None:
        command.extend(["--max-images", str(limit)])
    completed = subprocess.run(command, cwd=str(registry.root), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command,
            output=completed.stdout, stderr=completed.stderr)
    answer = json.loads(completed.stdout.decode("utf-8"), object_pairs_hook=_unique_object)
    expected = dict(status="verified_operator_input_publication_not_independent_acceptance",
        binding_sha256=binding["binding_sha256"], science_sha256=binding["science_sha256"],
        formal_execution_admission=False, historical_reuse_qualified=False,
        model_calls=0, AP_evaluations=0)
    require(canonical_hash(answer) == canonical_hash(expected),
        "Operator verifier returned a different identity or acceptance claim")
    require(_source_snapshot(registry) == source, "Operator consumer source changed during verification")
    for path, pin, value in zip(paths, pins, initial):
        require(canonical_hash(_pinned_json(path, pin)) == canonical_hash(value),
            "Operator publication changed during verification")
    evidence = dict(status="operator_inputs_reverified_via_bound_entry_not_reuse_acceptance",
        entrypoint=runtime.ENTRYPOINT, entrypoint_sha256=source[runtime.ENTRYPOINT],
        runtime_snapshot_sha256=canonical_hash(source),
        plan_file_sha256=plan_sha256, binding_file_sha256=binding_sha256,
        receipt_file_sha256=receipt_sha256, verifier_returncode=completed.returncode,
        verifier_stdout_sha256=hashlib.sha256(completed.stdout).hexdigest(),
        verifier_stderr_sha256=hashlib.sha256(completed.stderr).hexdigest(),
        timeout_seconds=timeout_seconds, verifier_invocations=1, max_images=limit,
        formal_execution_admission=False, historical_reuse_qualified=False,
        independent_acceptance=False, model_calls=0, AP_evaluations=0)
    return deepcopy(binding), evidence
