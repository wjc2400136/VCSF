"""Prospective, externally pinned permission for one frozen preprocessing run."""
import json
from pathlib import Path

from ..io import file_digest
from .vcsf_final_training_state import source_identity
from .vcsf_final_oblivious_contract import verify_source


BRIDGE_FILES = {
    "src/lgp/runners/vcsf_oblivious_admission.py",
    "src/lgp/runners/vcsf_final_oblivious.py",
    "src/lgp/reporting/vcsf_oblivious_audit.py",
    "experiments/vcsf_final_oblivious.py",
}

ACCEPTED_REPORT_SHA256 = "c2958513e58bb512c7eb9826427905e53cb7ea7910217fe48910f5c975e1be7f"


def isolate_output(output, protected):
    output = Path(output).resolve()
    for name in protected:
        path = Path(name).resolve()
        if output == path or path in output.parents or output in path.parents:
            raise ValueError("Admission output overlaps protected evidence")


def verify_executed_source(saved, admitted):
    if saved != admitted["source_identity"]:
        raise ValueError("Executed source differs from prospective admission")


def bound_json(path, digest):
    path = Path(path).resolve()
    if not digest or file_digest(path) != digest:
        raise ValueError("Admission evidence hash mismatch")
    data = json.loads(path.read_text(encoding="utf-8"))
    if file_digest(path) != digest:
        raise ValueError("Admission evidence changed while read")
    return data


def verify_bridge(current, original, changes):
    actual = {name: dict(before=original.get(name), after=current.get(name))
        for name in set(current) | set(original) if current.get(name) != original.get(name)}
    if actual != changes or not set(actual).issubset(BRIDGE_FILES):
        raise ValueError("Unreviewed change from qualified diagnostic source")


def verify_admission(registry, path, digest, *, plan, output, payloads,
                     checkpoint_bindings, checkpoint_bindings_sha256, check_payloads=True,
                     evidence=None, audit_output=None):
    if path is None or digest is None:
        raise ValueError("Formal preprocessing execution has not been admitted")
    receipt = bound_json(path, digest)
    dependencies = {str(Path(path).resolve()): digest}
    expected_payloads = {key: str(Path(value).resolve()) for key, value in payloads.items()}
    expected = dict(status="approved_prospective_execution", protocol=plan["protocol"],
        plan=plan, output=str(Path(output).resolve()), payloads=expected_payloads,
        checkpoint_bindings=str(Path(checkpoint_bindings).resolve()),
        checkpoint_bindings_sha256=checkpoint_bindings_sha256)
    if (not plan["formal_scope"] or plan["max_images"] is not None
            or plan["diagnostic_jobs"] is not None or len(plan["jobs"]) != 352
            or any(receipt.get(key) != value for key, value in expected.items())):
        raise ValueError("Admission does not authorize this exact full run")
    current = source_identity(registry.root)
    if receipt.get("source_identity") != current:
        raise ValueError("Admitted execution source changed")
    diagnostic = receipt["diagnostic"]
    accepted = bound_json(diagnostic["receipt"], diagnostic["sha256"])
    dependencies[str(Path(diagnostic["receipt"]).resolve())] = diagnostic["sha256"]
    if (accepted.get("protocol") != plan["protocol"] or accepted.get("cells") != 17
            or accepted.get("metric_values") != 204 or accepted.get("formal_scope") is not False
            or accepted.get("status") != "saved_preprocessing_replayed_pending_formal_admission_binding"):
        raise ValueError("Qualified diagnostic receipt differs")
    for name, expected_hash in accepted["evidence"].items():
        if file_digest(Path(name)) != expected_hash:
            raise ValueError("Qualified diagnostic evidence changed")
        dependencies[str(Path(name).resolve())] = expected_hash
    original_path = Path(accepted["run"]) / "source_identity.json"
    original = bound_json(original_path, accepted["evidence"][str(original_path)])
    verify_bridge(current, original, receipt["reviewed_source_changes"])
    bindings = bound_json(checkpoint_bindings, checkpoint_bindings_sha256)
    if bindings != receipt["checkpoints"]:
        raise ValueError("Admitted checkpoint binding changed")
    dependencies[str(Path(checkpoint_bindings).resolve())] = checkpoint_bindings_sha256
    report = receipt["accepted_report"]
    if report["sha256"] != ACCEPTED_REPORT_SHA256:
        raise ValueError("Admission accepted-main trust anchor differs")
    from ..reporting.vcsf_oblivious_checkpoint_binding import derive
    qualified, accepted_evidence = derive(registry, report["file"], report["sha256"])
    if qualified != bindings:
        raise ValueError("Checkpoint binding differs from accepted main results")
    dependencies.update(accepted_evidence)
    authority = receipt["authority"]
    if file_digest(Path(authority["file"])) != authority["sha256"]:
        raise ValueError("Execution authorization record changed")
    dependencies[str(Path(authority["file"]).resolve())] = authority["sha256"]
    protected = [Path(name).parent for name in dependencies] + [Path(accepted["run"])]
    isolate_output(output, protected)
    if audit_output is not None:
        isolate_output(audit_output, protected)
    if check_payloads:
        for source, payload in expected_payloads.items():
            if verify_source(registry, source, payload) != receipt["verified_inputs"][source]:
                raise ValueError("Admitted original input changed")
    if evidence is not None:
        evidence.update(dependencies)
    return receipt
