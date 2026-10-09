"""Exact follow-up execution scope for later admission and worker binding.

This module does not issue admission, qualify reuse, or execute models.
"""
from copy import deepcopy
from pathlib import Path
import re

from .vcsf_final_followup_plan import PROTOCOL, FREEZE_SHA256
from .vcsf_research_plan import canonical_hash, require


MODE = 'final_followup_execution'
EXECUTION_PROTOCOL = 'vcsf_final_followup_execution'
ENTRYPOINT = 'experiments/vcsf_final_followup_experiments.py'


def compile_execution_scope(registry, catalogue, *, group_ids, max_images=None):
    """Consume an externally authenticated catalogue and explicit group selection.

    Formal scope is the entire prospective-new partition. A diagnostic selects
    one canonical image per explicitly listed group and cannot qualify reuse or
    provide formal metrics. Lanes are filtered from the registered assignment.
    """
    definition = registry.protocols[PROTOCOL]
    require(catalogue['protocol_id'] == PROTOCOL and
        catalogue['protocol_content_sha256'] == canonical_hash(definition) and
        catalogue['freeze_sha256'] == FREEZE_SHA256 and
        catalogue['groups_sha256'] == canonical_hash(catalogue['groups']),
        'Execution scope requires the unchanged frozen catalogue')
    require(max_images is None or (type(max_images) is int and max_images == 1),
        'Follow-up diagnostics require exactly one canonical image per group')
    require(isinstance(group_ids, list) and group_ids and
        all(type(i) is int and i > 0 for i in group_ids) and
        len(set(group_ids)) == len(group_ids), 'Require explicit unique group identities')
    groups = catalogue['groups']
    ids = [g['group_id'] for g in groups]
    require(all(type(i) is int and i > 0 for i in ids) and len(ids) == len(set(ids)),
        'Invalid catalogue group identities')
    new_ids = [g['group_id'] for g in groups if g['disposition'] == 'prospective_new']
    require(catalogue['prospective_new_group_ids'] == new_ids and
        group_ids == [i for i in new_ids if i in group_ids],
        'Execution selection is missing, reordered, reused or unregistered')
    require(max_images is not None or group_ids == new_ids,
        'Formal follow-up cannot inherit a diagnostic subset')
    devices = catalogue['devices']
    require(isinstance(devices, list) and len(devices) in definition['device_counts'] and
        all(isinstance(d, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) for d in devices)
        and len(set(devices)) == len(devices), 'Invalid registered device selection')
    lanes = [new_ids[i::len(devices)] for i in range(len(devices))]
    require(canonical_hash(lanes) == canonical_hash(catalogue['prospective_lanes']),
        'Catalogue worker assignment changed')
    selected = [deepcopy(g) for g in groups if g['group_id'] in group_ids]
    for group in selected:
        dataset = group['dataset']
        require(dataset in definition['datasets'] and
            group['split'] == definition['datasets'][dataset]['split'] and
            type(group['images']) is int and group['images'] == definition['datasets'][dataset]['images']
            and group['source'] in registry.source_ids() and group['targets'] == registry.target_ids(),
            'Execution group differs from canonical dataset or detector scope')
        seeds = definition['stability_seeds'] if (
            dataset == definition['stability_dataset'] and
            group['source'] == definition['stability_source']) else [definition['main_seed']]
        require(type(group['seed']) is int and group['seed'] in seeds and
            group['seed_schedule'] == definition['seed_schedule'] and
            group['parameters_sha256'] == canonical_hash(catalogue['parameters']) and
            group['implementation_inventory_sha256'] == canonical_hash(catalogue['producer_runtime_sha256']),
            'Execution group changed method, inputs or seed semantics')
    scope = dict(schema_version=1, mode=MODE, protocol=EXECUTION_PROTOCOL,
        catalogue_content_sha256=canonical_hash(catalogue), freeze_sha256=FREEZE_SHA256,
        group_ids=list(group_ids), groups=selected, devices=list(devices),
        lanes=[[i for i in lane if i in group_ids] for lane in lanes],
        parameters_sha256=canonical_hash(catalogue['parameters']),
        producer_inventory_sha256=canonical_hash(catalogue['producer_runtime_sha256']),
        max_images=max_images, diagnostic_only=max_images is not None,
        images=sum(max_images or g['images'] for g in selected),
        target_cells=sum(len(g['targets']) for g in selected),
        budget_profile=definition['budget_profile'],
        payload_policy='keep_all_until_independent_acceptance',
        canonical_image_selection=True, selected_position_seeds=True,
        independent_confirmation=False, formal_execution_admission=False,
        reuse_accepted=False, model_calls=0)
    scope['scope_sha256'] = canonical_hash(scope)
    return scope


def verify_execution_scope(registry, catalogue, scope):
    """Reconstruct every field; a self-hash alone does not qualify a scope."""
    require(isinstance(scope, dict), 'Execution scope must be an object')
    expected = compile_execution_scope(registry, catalogue,
        group_ids=scope['group_ids'], max_images=scope['max_images'])
    require(canonical_hash(expected) == canonical_hash(scope),
        'Execution scope or its derived counts changed')
    return expected


def read_execution_admission(read, registry, catalogue, scope, runtime_reference,
        admission_reference):
    """Consume a separately issued, source-bound readiness receipt.

    The caller authenticates the catalogue, dataset publications and input
    readbacks separately. Receipt matching never establishes current process
    ownership, current GPU availability or scientific result acceptance.
    """
    from tools.audit_vcsf_cpu_analysis import child, is_sha, plain
    from .vcsf_final_followup_method import METHOD_FILES
    from .vcsf_common_anchor_execution_contract import runtime_snapshot

    expected_scope = verify_execution_scope(registry, catalogue, scope)

    def reference_path(reference, category, namespace, filename):
        require(isinstance(reference, dict) and set(reference) == {'file', 'sha256'}
            and isinstance(reference['file'], str) and is_sha(reference['sha256']),
            'Require a pinned execution evidence reference')
        path = plain(Path(reference['file']))
        require(path.name == filename and len(path.parents) >= 5 and
            path.parents[1].name == namespace and path.parents[2].name == category
            and path.parents[3].name == 'outputs', 'Execution evidence namespace differs')
        require(not (path.parent / 'failure.json').exists(), 'Failed execution evidence cannot be consumed')
        return path

    runtime_path = reference_path(runtime_reference, 'plans', EXECUTION_PROTOCOL, 'runtime.json')
    admission_path = reference_path(admission_reference, 'audits',
        'vcsf_final_followup_admission', 'receipt.json')
    runtime = read.json(runtime_path, runtime_reference['sha256'])
    require(set(runtime) == {'schema_version', 'status', 'scope', 'runtime_sha256', 'input_references'}
        and type(runtime['schema_version']) is int and runtime['schema_version'] == 1
        and runtime['status'] == 'prepared_final_followup_runtime_not_admitted'
        and canonical_hash(runtime['scope']) == canonical_hash(expected_scope),
        'Runtime preparation differs from the reconstructed execution scope')
    sources = runtime['runtime_sha256']
    required_sources = {'src/lgp/runners/attack.py',
        'src/lgp/runners/vcsf_final_followup_dispatch.py',
        'src/lgp/runners/vcsf_final_followup_execution_contract.py',
        'src/lgp/attacks/vcsf_common_anchor_isolated.py',
        'src/lgp/attacks/vcsf_research_isolated.py'}
    require(isinstance(sources, dict) and required_sources.union(METHOD_FILES) <= set(sources),
        'Runtime preparation omits required execution sources')
    require(all(sources[name] == catalogue['producer_runtime_sha256'].get(name)
        for name in METHOD_FILES), 'Runtime changed the frozen numerical source')
    require(sources == runtime_snapshot(registry.root, entrypoint=ENTRYPOINT),
        'Runtime source inventory is incomplete or changed')
    for name, digest in sources.items():
        read.bytes(child(registry.root, name), digest)
    inputs = runtime['input_references']
    datasets = {g['dataset'] for g in expected_scope['groups']}
    require(isinstance(inputs, dict) and set(inputs) == datasets,
        'Runtime inputs do not cover the selected datasets exactly')
    for item in inputs.values():
        require(isinstance(item, dict) and set(item) == {'publication', 'audit'},
            'Runtime needs both input publication and independent readback')
        for key, filename, category in (('publication', 'completion.json', 'plans'),
                ('audit', 'receipt.json', 'audits')):
            path = reference_path(item[key], category, 'vcsf_final_followup_inputs', filename)
            read.bytes(path, item[key]['sha256'])
    receipt = read.json(admission_path, admission_reference['sha256'])
    diagnostic = expected_scope['diagnostic_only']
    require(receipt['status'] == ('independently_admitted_final_followup_diagnostic' if diagnostic
            else 'independently_admitted_final_followup_formal') and receipt['errors'] == []
        and receipt['runtime_reference'] == runtime_reference
        and receipt['scope_sha256'] == expected_scope['scope_sha256']
        and receipt['runtime_snapshot_sha256'] == canonical_hash(sources)
        and canonical_hash(receipt['input_references']) == canonical_hash(inputs)
        and receipt['devices'] == expected_scope['devices']
        and receipt['diagnostic_execution_admission'] is diagnostic
        and receipt['formal_execution_admission'] is (not diagnostic),
        'Admission receipt does not bind this runtime, scope, inputs and device mode')
    checks = receipt['verified_checks']
    required_checks = {'input_identity', 'frozen_method_identity', 'strict_checkpoint_loading',
        'loaded_process_binding', 'worker_isolation', 'resource_capacity', 'full_target_dispatch'}
    require(isinstance(checks, dict) and set(checks) == required_checks and
        all(value is True for value in checks.values()), 'Runtime readiness is incomplete')
    auditor = receipt['auditor_source_sha256']
    require(isinstance(auditor, dict) and auditor, 'Admission lacks auditor source provenance')
    for name, digest in auditor.items():
        read.bytes(child(admission_path.parents[4], name), digest)
    diagnostic_path = None
    if diagnostic:
        require(receipt['diagnostic_acceptance'] is None,
            'Diagnostic admission cannot claim completed diagnostic acceptance')
    else:
        reference = receipt['diagnostic_acceptance']
        diagnostic_path = reference_path(reference, 'audits', 'vcsf_final_followup_diagnostic', 'receipt.json')
        proof = read.json(diagnostic_path, reference['sha256'])
        require(proof['status'] == 'independently_verified_final_followup_diagnostic'
            and proof['errors'] == [] and proof['formal_scope_sha256'] == expected_scope['scope_sha256']
            and proof['formal_runtime_reference'] == runtime_reference
            and proof['runtime_snapshot_sha256'] == canonical_hash(sources)
            and type(proof['device_count']) is int and proof['device_count'] == len(expected_scope['devices'])
            and proof['model_execution_verified'] is True and proof['full_target_panel_verified'] is True,
            'Formal admission lacks matching executed diagnostic acceptance')
        proof_sources = proof['auditor_source_sha256']
        require(isinstance(proof_sources, dict) and proof_sources,
            'Diagnostic acceptance lacks auditor source provenance')
        for name, digest in proof_sources.items():
            read.bytes(child(diagnostic_path.parents[4], name), digest)
    read.unchanged()
    require(sources == runtime_snapshot(registry.root, entrypoint=ENTRYPOINT),
        'Runtime source inventory changed during admission readback')
    for path in (runtime_path, admission_path):
        require(not (path.parent / 'failure.json').exists(),
            'Execution evidence failed during admission readback')
    for item in inputs.values():
        for reference in item.values():
            require(not (Path(reference['file']).parent / 'failure.json').exists(),
                'Input evidence failed during admission readback')
    if diagnostic_path is not None:
        require(not (diagnostic_path.parent / 'failure.json').exists(),
            'Diagnostic acceptance failed during admission readback')
    return dict(status='execution_admission_receipt_bound_pending_worker_ownership',
        scope=expected_scope, input_references=deepcopy(inputs),
        runtime_sha256=deepcopy(sources), admission_reference=deepcopy(admission_reference),
        worker_ownership_verified=False, current_device_availability_verified=False,
        scientific_acceptance=False, model_calls=0)
