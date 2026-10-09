"""Read-only reconstruction of accepted scale evidence for stage preparation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from .vcsf_research_plan import canonical_hash, require


KIND = "vcsf_complete_scale_stage_evidence"
PROOF_MARKER = "LGP_STAGE_SCALE_PROOF="


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate evidence JSON key")
            result[key] = value
        return result

    def reject(value):
        raise ValueError("Non-finite evidence JSON: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _bound_json(reference):
    require(isinstance(reference, dict) and set(reference) == {"file", "sha256"},
        "Evidence references require exactly file and sha256")
    require(isinstance(reference["file"], str) and isinstance(reference["sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]), "Invalid evidence reference")
    path = Path(reference["file"])
    require(path.is_absolute() and path == path.resolve()
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "Evidence requires an absolute canonical non-symlink path")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == reference["sha256"], "Evidence bytes changed")
    return _parse(raw)


_ORIGIN_HELPER = r'''
def original_identity(read, item, origin, origin_plan):
    from pathlib import Path
    import hashlib
    import yaml
    generation, group = item['generation'], item['group']
    protocol_name = generation['run_metadata']['protocol']
    relative = 'configs/experiments/protocols.yaml'
    expected = origin_plan['runtime_sha256'][relative]
    candidates = {Path(p) for p, h in read.checked.items()
        if h == expected and Path(p).as_posix().endswith('/'+relative)}
    # Old cost inputs need not contain every historical declarative source.
    # Add only sources from explicit original generation/plan workspaces.
    for path in [item['path']] + [Path(p) for p in origin['execution_plan_files']]:
        parts = path.parts
        audit.require(parts.count('outputs') == 1, 'Ambiguous original workspace')
        workspace = Path(*parts[:parts.index('outputs')])
        candidate = workspace/relative
        if candidate.is_file() and hashlib.sha256(candidate.read_bytes()).hexdigest() == expected:
            candidates.add(candidate)
    audit.require(candidates, 'No explicit original protocol registry matches its execution-plan hash')
    declarations = [yaml.safe_load(read.bind(p, expected).read_text())['protocols'][protocol_name]
        for p in sorted(candidates)]
    audit.require(all(d == declarations[0] for d in declarations), 'Original protocol copies disagree')
    implementation = declarations[0]['isolated_candidate']
    audit.require(all(isinstance(implementation.get(k), str) and implementation[k]
        for k in ('implementation', 'class_name', 'config_class_name'))
        and implementation['implementation'] in origin_plan['runtime_sha256'],
        'Original executable/config declaration is absent from original runtime identity')
    return dict(protocol=protocol_name, study=generation['run_metadata']['study'],
        variant=group['variant'], run_id=item['path'].parents[3].name, group_id=group['group_id'],
        implementation=implementation, source_sha256=origin_plan['runtime_sha256'],
        parameters=group['parameters'], parameters_sha256=group['parameters_sha256'], synthetic=False,
        generation_file=str(item['path']), generation_sha256=item['sha256'],
        execution_plan_sha256=origin['execution_plan_sha256'], runtime_snapshot_sha256=origin['runtime_snapshot_sha256'])

def original_anchor_projection(read, original):
    from pathlib import Path
    canonical = audit.contract.old.canonical
    defaults = dict(image_interpolation='bilinear', image_padding='reflect', placement='random')
    parameters = original['parameters']
    if set(defaults) <= set(parameters):
        return None
    audit.require(original['implementation']['class_name'] == 'VCSFResearchCandidate'
        and original['implementation']['config_class_name'] == 'VCSFResearchConfig'
        and original['variant'] == 'levels_two' and original['group_id'] == 29
        and not set(defaults).intersection(parameters), 'Unknown original parameter projection')
    candidates = []
    for name, digest in list(read.checked.items()):
        if Path(name).name != 'execution_plan.json':
            continue
        plan = read.json(name, digest)
        mapping = plan.get('historical_mapping')
        if not isinstance(mapping, dict) or mapping.get('original_parameters_sha256') != original['parameters_sha256']:
            continue
        audit.require(mapping.get('status') == 'qualified_original_group_mapped_to_exact_effective_anchor'
            and mapping.get('original_group_id') == original['group_id']
            and mapping.get('original_variant') == original['variant']
            and mapping.get('destination_group_id') == 1 and mapping.get('destination_variant') == 'A_h2'
            and mapping.get('explicit_historical_geometry_defaults') == defaults
            and mapping.get('seed_schedule') == 'selected_position'
            and mapping.get('scope') == 'same_effective_single_group_conditions_not_same_whole_study_protocol'
            and mapping.get('physical_cost_inheritance') is False
            and mapping.get('bitwise_trajectory_equivalence_claimed') is False,
            'Original accepted anchor mapping has different scope')
        projected = dict(parameters, **defaults)
        audit.require(mapping['destination_parameters_sha256'] == canonical(projected)
            and mapping['original_parameters_sha256'] == canonical(parameters)
            and plan.get('historical_plan_sha256') == original['execution_plan_sha256'],
            'Anchor mapping does not bind this exact original source plan and parameters')
        candidates.append(dict(file=name, sha256=digest, mapping=mapping))
    audit.require(candidates and len({canonical(c['mapping']) for c in candidates}) == 1,
        'Missing or ambiguous already accepted original anchor mapping')
    return dict(status='verified_original_anchor_parameter_projection_for_preparation',
        original_identity_sha256=canonical(original), original_parameters_sha256=original['parameters_sha256'],
        projected_parameters=projected, projected_parameters_sha256=canonical(projected),
        explicit_historical_geometry_defaults=defaults,
        original_mapping=candidates[0]['mapping'], mapping_plan_references=[
            dict(file=c['file'], sha256=c['sha256']) for c in candidates],
        formal_reuse_qualified=False, scientific_acceptance=False, original_identity_modified=False)
'''


# Import the unchanged historical auditors from their bound source, never pyc.
# The reconstruction below mirrors their read-only main path and compares the
# entire receipt; it does not call their output-writing or model-running paths.
_PROBE = _ORIGIN_HELPER + r'''
import hashlib, importlib.abc, importlib.machinery, importlib.util, json, sys
from pathlib import Path
request = json.loads(sys.stdin.read())
trusted, reference = Path(request['trusted_root']), request['cost_acceptance']
receipt_path = Path(reference['file'])
raw = receipt_path.read_bytes()
assert hashlib.sha256(raw).hexdigest() == reference['sha256'], 'Cost receipt changed'
saved = json.loads(raw)
root = receipt_path.parents[4]
assert root != trusted, 'Reconstruction must use the original isolated cost auditor'
assert receipt_path.parent.parent == root/'outputs/diagnostics/vcsf_remaining_cost_audit', 'Wrong audit namespace'
sources, loaded = saved['tests']['source_sha256'], {}
assert 'tools/audit_vcsf_remaining_cost.py' in sources, 'Missing cost auditor source'
for name, expected in sources.items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts, 'Invalid source path'
    for base in (trusted, root):
        path = base/relative
        assert not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink source'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'Untrusted or changed auditor source: '+name
sys.path[:0] = [str(root/'tools')]
class SourceOnly(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = Path(self.path)
        relative, data = path.relative_to(root).as_posix(), path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        assert sources.get(relative) == digest, 'Unbound imported source'
        loaded[fullname] = dict(file=relative, sha256=digest)
        return compile(data, str(path), 'exec', dont_inherit=True)
class FrozenSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        source = root/'tools'/(fullname+'.py')
        if '.' in fullname or not source.is_file():
            return None
        return importlib.util.spec_from_file_location(fullname, str(source),
            loader=SourceOnly(fullname, str(source)))
sys.meta_path.insert(0, FrozenSources())
import audit_vcsf_remaining_cost as audit
contract, Reader = audit.contract, audit.Reader
contract.operator.server_only()
test_read, read = Reader(), Reader()
saved = read.json(receipt_path, reference['sha256'])
# Receipt identity is checked separately; original checked_input_sha256 excludes itself.
read.checked.pop(str(receipt_path))
point_ref = request['point_acceptance']
tests = contract.verify_tests(test_read, root, Path(saved['tests']['receipt_file']), saved['tests']['receipt_sha256'])
bound = contract.bind_inputs(read, Path(point_ref['file']), point_ref['sha256'], False)
input_bindings = dict(read.checked)
path = Path(saved['cost_report_file'])
report = read.json(path, saved['cost_report_sha256'])
reporting_root = contract.old.plain(report['reporting_workspace'])
audit.require(reporting_root != root and root not in reporting_root.parents and reporting_root not in root.parents,
    'Independent audit workspace overlap')
audit.require(report['point_acceptance_file'] == point_ref['file']
    and report['point_acceptance_sha256'] == point_ref['sha256']
    and report['original_point_proof'] == bound['proof'], 'Point proof differs')
report_test_read = Reader()
report_tests = contract.verify_tests(report_test_read, reporting_root,
    Path(report['tests']['receipt_file']), report['tests']['receipt_sha256'])
audit.require(report['tests'] == report_tests and report['source_sha256']
    == report_tests['source_sha256'] == tests['source_sha256'], 'Different audit/report source')
for name, digest in report_test_read.checked.items():
    audit.require(name not in input_bindings or input_bindings[name] == digest, 'Conflicting report input')
    input_bindings[name] = digest
audit.require(report['checked_input_sha256'] == input_bindings, 'Changed report input inventory')
audit.require(path.name == 'cost.json' and path.parent.parent == reporting_root/'outputs/diagnostics'/contract.PROTOCOL,
    'Cost report namespace differs')
for name, digest in input_bindings.items():
    read.bind(name, digest)
independent = audit.audit_report(report, bound, False)
audit.verify_exports(read, path, report, saved['manifest_sha256'])
expected = dict(status='independently_verified_remaining_saved_cost_disclosure', errors=[],
    stage=contract.STAGE, diagnostic=False, groups=len(report['groups']), images_per_group=5000,
    contrasts=len(independent), independent_contrasts=independent,
    source_trace_steps_replayed=sum(g['counts']['logical_gradient_updates'] for g in report['groups']),
    execution_plan_sha256=report['execution_plan_sha256'], point_acceptance_sha256=point_ref['sha256'],
    original_point_proof=bound['proof'], previous_cost_binding=bound['previous_cost_binding'],
    cost_report_file=str(path), cost_report_sha256=saved['cost_report_sha256'], manifest_sha256=saved['manifest_sha256'],
    saved_cost_disclosure_accepted=True, physical_cost_calibration_accepted=False,
    scientific_acceptance=False, independent_confirmation=False, new_model_calls=0,
    bootstrap_replicates_computed=0, tests=tests, checked_test_sha256=test_read.checked,
    checked_input_sha256=read.checked)
actual = dict(saved)
actual.pop('completed_at')
audit.require(actual == expected, 'Original full cost acceptance does not reconstruct')
rows = []
for item, origin in zip(bound['rows'], bound['timer_source_binding']):
    generation, group = item['generation'], item['group']
    origin_plan = read.json(origin['execution_plan_files'][0], origin['execution_plan_sha256'])
    original = original_identity(read, item, origin, origin_plan)
    parameter_projection = original_anchor_projection(read, original)
    rows.append(dict(group=group, generation_file=str(item['path']), generation_sha256=item['sha256'],
        original=original, generation_protocol=original['protocol'],
        original_parameter_projection=parameter_projection,
        original_execution_plan_sha256=origin['execution_plan_sha256'],
        runtime_snapshot_sha256=origin['runtime_snapshot_sha256'],
        runtime_sha256=origin_plan['runtime_sha256'], image_ids_sha256=generation['image_ids_sha256']))
for reader in (read, test_read, report_test_read):
    reader.unchanged()
assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == reference['sha256'], 'Receipt changed during reconstruction'
print('LGP_STAGE_SCALE_PROOF='+json.dumps(dict(status='reconstructed_complete_scale_point_and_saved_cost',
    cost_acceptance_sha256=reference['sha256'], point_acceptance_sha256=point_ref['sha256'],
    groups=rows, targets=bound['execution']['targets'], contrasts=bound['contrasts'],
    execution_plan_sha256=bound['plan']['execution_plan_sha256'],
    cost_report_sha256=saved['cost_report_sha256'], checked_input_sha256=read.checked,
    modules=loaded, new_model_calls=0, AP_evaluations=0, bootstrap_replicates_computed=0,
    physical_cost_calibration_accepted=False, scientific_acceptance=False, independent_confirmation=False)))
'''


def _reconstruct(registry, evidence):
    request = dict(trusted_root=str(registry.root.resolve()),
        cost_acceptance=evidence["cost_acceptance"], point_acceptance=evidence["point_acceptance"])
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    process = subprocess.run([sys.executable, "-I", "-B", "-c", _PROBE],
        input=json.dumps(request), cwd=registry.root, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(process.returncode == 0,
        "Original full-scale evidence reconstruction failed: " + process.stderr[-4000:])
    markers = [line[len(PROOF_MARKER):] for line in process.stdout.splitlines()
        if line.startswith(PROOF_MARKER)]
    require(len(markers) == 1, "Missing unique full-scale reconstruction proof")
    return _parse(markers[0])


def verify_stage_evidence(registry, evidence, working_baseline, nonzero_comparator):
    """Verify real inputs; no synthetic fixture can self-declare acceptance."""
    require(sys.platform == "linux" and Path(sys.prefix).name == "oda" and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        "Real stage evidence reconstruction requires the pinned server ODA interpreter")
    require(isinstance(evidence, dict) and set(evidence) == {
        "schema_version", "kind", "cost_acceptance", "point_acceptance", "selection_decision"}
        and type(evidence["schema_version"]) is int and evidence["schema_version"] == 1
        and evidence["kind"] == KIND, "Not a complete scale-stage evidence request")
    originals = {key: _bound_json(evidence[key]) for key in
        ("cost_acceptance", "point_acceptance", "selection_decision")}
    require(originals["cost_acceptance"].get("status") ==
        "independently_verified_remaining_saved_cost_disclosure"
        and originals["point_acceptance"].get("status") ==
        "independently_verified_remaining_width_point_analysis", "Diagnostics cannot bind a real baseline")
    proof = _reconstruct(registry, evidence)
    require(proof["status"] == "reconstructed_complete_scale_point_and_saved_cost"
        and proof["cost_acceptance_sha256"] == evidence["cost_acceptance"]["sha256"]
        and proof["point_acceptance_sha256"] == evidence["point_acceptance"]["sha256"]
        and proof["targets"] == registry.target_ids(), "Reconstructed evidence scope changed")
    decision = originals["selection_decision"]
    _verify_endpoints(proof, working_baseline, nonzero_comparator)
    _verify_decision(decision, proof, working_baseline, nonzero_comparator)
    for key, value in originals.items():
        require(_bound_json(evidence[key]) == value, "Stage evidence changed while being verified")
    return dict(status="verified_complete_scale_evidence_for_unarmed_preparation",
        evidence_kind=KIND, proof_sha256=canonical_hash(proof), proof=proof,
        selection_decision_sha256=evidence["selection_decision"]["sha256"],
        references=evidence, new_model_calls=0, AP_evaluations=0,
        scientific_acceptance=False, independent_confirmation=False,
        candidate_frozen=False, executor_admission=False)


def _verify_endpoints(proof, working_baseline, nonzero_comparator):
    same = nonzero_comparator == "same_as_working_baseline" or (
        isinstance(nonzero_comparator, dict)
        and nonzero_comparator.get("reference") == "same_as_working_baseline")
    endpoints = [working_baseline] if same else [working_baseline, nonzero_comparator]
    for endpoint in endpoints:
        require(isinstance(endpoint, dict) and isinstance(endpoint.get("original"), dict),
            "Real endpoint requires its complete original accepted identity")
        matching = [row for row in proof["groups"]
            if canonical_hash(row["original"]) == canonical_hash(endpoint["original"])]
        require(len(matching) == 1, "Endpoint original does not match a reconstructed accepted scale group")
        row = matching[0]
        display = "A_h2" if row["group"]["variant"] == "levels_two" else row["group"]["variant"]
        projection = row.get("original_parameter_projection")
        parameters = row["group"]["parameters"] if projection is None else projection["projected_parameters"]
        parameter_hash = row["group"]["parameters_sha256"] if projection is None else projection["projected_parameters_sha256"]
        require(endpoint.get("variant") == display
            and canonical_hash(endpoint.get("parameters")) == canonical_hash(parameters)
            and endpoint.get("parameters_sha256") == parameter_hash
            and canonical_hash(endpoint.get("original_parameter_projection")) == canonical_hash(projection)
            and endpoint.get("original_sha256") == canonical_hash(row["original"]),
            "Endpoint parameters, registered alias or original hash were relabelled")


def _verify_decision(decision, proof, working_baseline, nonzero_comparator):
    require(isinstance(decision, dict) and type(decision.get("schema_version")) is int
        and decision["schema_version"] == 1
        and decision.get("kind") == "vcsf_scale_working_baseline_decision"
        and decision.get("status") == "working_selection_not_final_freeze",
        "Missing explicitly recorded scale working-baseline decision")
    for key in ("cost_acceptance_sha256", "point_acceptance_sha256", "execution_plan_sha256"):
        require(decision.get(key) == proof[key], "Selection decision evidence mismatch: " + key)
    require(canonical_hash(decision.get("working_baseline")) == canonical_hash(working_baseline)
        and canonical_hash(decision.get("nonzero_comparator")) == canonical_hash(nonzero_comparator),
        "Stage endpoints differ from the hashed working-selection decision")
    rows = proof["groups"]
    expected = [row["group"]["variant"] for row in rows]
    require(len(expected) == len(set(expected)) == 17, "Selection needs all seventeen original groups")
    reviews = decision.get("group_assessments")
    require(isinstance(reviews, list) and [r.get("variant") for r in reviews] == expected,
        "Selection must preserve every original group outcome in canonical order")
    for review, row in zip(reviews, rows):
        require(review.get("parameters_sha256") == row["group"]["parameters_sha256"]
            and review.get("disposition") in ("keep", "change", "reject")
            and all(isinstance(review.get(k), str) and review[k].strip() for k in
                ("paired_ap_assessment", "saved_cost_assessment", "implementation_complexity_assessment", "reason")),
            "Incomplete original outcome assessment")
    comparisons = decision.get("comparisons_considered")
    require(comparisons == [c["name"] for c in proof["contrasts"]],
        "Selection decision must identify the complete registered paired contrast family")
    for key in ("independent_confirmation", "significance_claimed", "final_freeze", "executor_admission"):
        require(decision.get(key) is False, "Working selection overclaims: " + key)
    require(isinstance(decision.get("rationale"), str) and decision["rationale"].strip(),
        "Missing full-panel working selection rationale")
