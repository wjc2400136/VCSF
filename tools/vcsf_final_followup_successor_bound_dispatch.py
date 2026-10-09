"""Bind successor groups; a missing admission consumer is a hard failure."""
from pathlib import Path

from tools.audit_vcsf_cpu_analysis import check, plain, is_sha
from tools.run_prepare_vcsf_final_followup_inputs import _catalogue
from tools.vcsf_final_followup_bound_dispatch import prepare_bound_group_calls
from lgp.runners.vcsf_final_followup_worker import execution_projection
from lgp.runners.vcsf_research_plan import canonical_hash


def load_bound_successor_execution_group(read, registry, runtime_reference,
        admission_reference, group_id, group_directory, device, *, max_images=None):
    from lgp.runners.vcsf_final_followup_successor_admission import read_successor_execution_admission

    check(type(runtime_reference) is dict and set(runtime_reference) == {'file', 'sha256'}
        and type(runtime_reference['file']) is str and is_sha(runtime_reference['sha256']),
        'Require a pinned successor runtime')
    path = plain(Path(runtime_reference['file']))
    check(path.name == 'runtime.json' and len(path.parents) >= 5
        and path.parents[1].name == 'vcsf_final_followup_execution'
        and path.parents[2].name == 'plans' and path.parents[3].name == 'outputs'
        and not (path.parent / 'failure.json').exists(), 'Invalid successor runtime path')
    runtime = read.json(path, runtime_reference['sha256'])
    request = runtime['preparation_request']
    catalogue = _catalogue(read, registry, request['preparation'], request['input_report'])
    admitted = read_successor_execution_admission(read, registry, catalogue,
        runtime['scope'], runtime_reference, admission_reference)
    scope = admitted['scope']
    check(type(group_id) is int and group_id in scope['group_ids']
        and canonical_hash(max_images) == canonical_hash(scope['max_images']),
        'Group or limit is outside admitted successor scope')
    group = next(g for g in scope['groups'] if g['group_id'] == group_id)
    refs = admitted['input_references'][group['dataset']]
    prepared = prepare_bound_group_calls(read, registry, refs['publication'], refs['audit'],
        group_id, group_directory, device, max_images=max_images,
        expected_catalogue_sha256=scope['catalogue_content_sha256'])
    bindings = [row for row in admitted['group_input_bindings'] if row['group_id'] == group_id]
    check(len(bindings) == 1, 'Missing or repeated successor input binding')
    projected = prepared['projected_inputs']
    expected = dict(group_id=group_id, input_binding_sha256=projected['binding_sha256'],
        execution_input_binding_sha256=execution_projection(projected, max_images)['binding_sha256'])
    check(canonical_hash(bindings[0]) == canonical_hash(expected)
        and prepared['input_binding_sha256'] == projected['binding_sha256'],
        'Successor input projection differs from admitted runtime')
    # Rehash the already reconstructed evidence instead of rebuilding its graph.
    getattr(read, 'full_unchanged', read.unchanged)()
    return dict(admission=admitted, prepared=prepared, group=group, catalogue=catalogue)
