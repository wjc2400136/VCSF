"""Connect authenticated input publications to call planning, never execution."""
from pathlib import Path
from copy import deepcopy

from audit_vcsf_cpu_analysis import check, child, plain, is_sha
from run_prepare_vcsf_final_followup_inputs import _catalogue, _publication
from lgp.runners.vcsf_final_followup_dispatch import prepare_group_calls
from lgp.runners.vcsf_final_followup_inputs import project_group_inputs
from lgp.runners.vcsf_research_plan import canonical_hash
from lgp.runners.vcsf_final_followup_execution_contract import read_execution_admission


def load_bound_execution_group(read, registry, runtime_reference, admission_reference,
        group_id, group_directory, device, *, max_images=None):
    """Join authentic catalogue/input reconstruction with the admission consumer.

    This loads no models and does not verify live process/device ownership.
    """
    check(isinstance(runtime_reference, dict) and set(runtime_reference) == {'file', 'sha256'}
        and is_sha(runtime_reference['sha256']), 'Require a pinned runtime reference')
    runtime_path = plain(Path(runtime_reference['file']))
    check(runtime_path.name == 'runtime.json' and len(runtime_path.parents) >= 5 and
        runtime_path.parents[1].name == 'vcsf_final_followup_execution' and
        runtime_path.parents[2].name == 'plans' and runtime_path.parents[3].name == 'outputs'
        and not (runtime_path.parent / 'failure.json').exists(), 'Runtime publication is unsafe or failed')
    runtime = read.json(runtime_path, runtime_reference['sha256'])
    inputs = runtime['input_references']
    check(isinstance(inputs, dict) and inputs, 'Runtime has no bound dataset inputs')
    first = inputs[sorted(inputs)[0]]
    _, publication = _publication(read, first['publication'], 'vcsf_final_followup_inputs')
    request = publication['request']
    catalogue = _catalogue(read, registry, request['preparation'], request['input_report'])
    admitted = read_execution_admission(read, registry, catalogue, runtime['scope'],
        runtime_reference, admission_reference)
    scope = admitted['scope']
    check(type(group_id) is int and group_id in scope['group_ids'] and
        canonical_hash(max_images) == canonical_hash(scope['max_images']),
        'Group or image limit is outside the admitted execution scope')
    group = next(g for g in scope['groups'] if g['group_id'] == group_id)
    references = admitted['input_references'][group['dataset']]
    prepared = prepare_bound_group_calls(read, registry,
        references['publication'], references['audit'], group_id, group_directory, device,
        max_images=max_images, expected_catalogue_sha256=scope['catalogue_content_sha256'])
    admitted = read_execution_admission(read, registry, catalogue, runtime['scope'],
        runtime_reference, admission_reference)
    return dict(admission=admitted, prepared=prepared, group=group, catalogue=catalogue)


def prepare_bound_group_calls(read, registry, publication_reference, audit_reference,
        group_id, group_directory, device, *, max_images=None, expected_catalogue_sha256=None):
    """External callers provide trusted hashes; matching receipts do not admit GPU work."""
    root, completion = _publication(read, publication_reference, 'vcsf_final_followup_inputs')
    check(completion['status'] == 'final_followup_dataset_inputs_pending_independent_readback',
        'Input publication is not complete')
    check({p.name for p in root.iterdir()} ==
        {'assets', 'shared_inputs.json', 'group_views.json', 'completion.json'},
        'Input publication contains partial or unexpected artifacts')
    flags = ('reuse_accepted', 'formal_execution_admission',
        'checkpoint_qualification_accepted', 'implementation_bridge_accepted')
    check(all(completion[k] is False for k in flags), 'Unexpected input publication admission')
    audit_path = plain(Path(audit_reference['file']))
    check(not (audit_path.parent / 'failure.json').exists(),
        'Failed input audit cannot be consumed')
    check(audit_path.name == 'receipt.json' and len(audit_path.parents) >= 5 and
        audit_path.parents[1].name == 'vcsf_final_followup_inputs' and
        audit_path.parents[2].name == 'audits' and audit_path.parents[3].name == 'outputs',
        'Input audit namespace differs')
    audit = read.json(audit_path, audit_reference['sha256'])
    check(audit['status'] == 'dataset_asset_reconstruction_matches_pending_outer_acceptance' and
        audit['publication_reference'] == publication_reference and
        audit['actual_asset_reconstruction_verified'] is True and
        all(audit[k] is False for k in flags) and
        audit['model_calls'] == audit['AP_replays'] == 0,
        'Input audit scope or reference differs')
    check(isinstance(audit['source_sha256'], dict) and audit['source_sha256'],
        'Missing audit source binding')
    for name, digest in audit['source_sha256'].items():
        read.bytes(child(audit_path.parents[4], name), digest)
    request = completion['request']
    plan = _catalogue(read, registry, request['preparation'], request['input_report'])
    check(expected_catalogue_sha256 is None or canonical_hash(plan) == expected_catalogue_sha256,
        'Dataset publication belongs to another execution catalogue')
    artifacts = completion['artifacts_sha256']
    check(set(artifacts) == {'shared_inputs.json', 'group_views.json'}, 'Unexpected input artifacts')
    shared = read.json(root / 'shared_inputs.json', artifacts['shared_inputs.json'])
    views = read.json(root / 'group_views.json', artifacts['group_views.json'])
    check(shared['dataset'] == request['dataset'] == audit['dataset'] and
        audit['shared_input_content_sha256'] == canonical_hash(shared) and
        audit['group_views_content_sha256'] == canonical_hash(views),
        'Audited input content differs')
    manifest = shared['clean_image_manifest']
    check(plain(Path(shared['dataset_output_directory'])) == root / 'assets' and
        plain(Path(manifest['file'])) == root / 'assets' / 'clean_images.json',
        'Input assets escaped publication')
    rows = read.json(Path(manifest['file']), manifest['sha256'])
    groups = [g for g in plan['groups'] if g['dataset'] == shared['dataset']]
    expected = [project_group_inputs(registry, g, shared, rows) for g in groups]
    check(canonical_hash(expected) == canonical_hash(views) and
        type(completion['groups']) is int and completion['groups'] == len(groups) == audit['groups'],
        'Saved projected groups differ')
    selected = [v for g, v in zip(groups, views) if g['group_id'] == group_id]
    check(len(selected) == 1, 'Group is not in this dataset publication')
    result = prepare_group_calls(registry, plan, group_id, selected[0],
        group_directory, device, max_images=max_images)
    read.unchanged()
    check(not (audit_path.parent / 'failure.json').exists(),
        'Input audit failed during call preparation')
    result.update(input_publication_reference=dict(publication_reference),
        input_audit_reference=dict(audit_reference),
        input_publication_authenticated=True, historical_input_readback_bound=True,
        current_asset_reconstruction_performed=False, checkpoint_qualification_accepted=False,
        implementation_bridge_accepted=False, projected_inputs=deepcopy(selected[0]))
    return result
