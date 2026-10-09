"""Prepare a pinned successor runtime; never admit, launch or stop execution.

The caller supplies a real source-bound preparation entry. This module does
not invent a successor CLI or claim that preparation proves executor readiness.
Scope/partition file hashes are checked, but their stop and qualification
references remain opaque obligations for a separate admission consumer.
"""
from copy import deepcopy
from pathlib import Path
import sys

from tools.audit_vcsf_cpu_analysis import Evidence, check, child, is_sha, plain
from tools.run_prepare_vcsf_final_followup_inputs import _catalogue
from tools.vcsf_final_followup_bound_dispatch import prepare_bound_group_calls

from ..io import atomic_json, file_digest
from .vcsf_common_anchor_execution_contract import runtime_snapshot
from .vcsf_final_followup_method import METHOD_FILES
from .vcsf_final_followup_successor_contract import EXECUTION_PROTOCOL, verify_successor_scope
from .vcsf_final_followup_worker import execution_projection
from .vcsf_research_plan import canonical_hash


STATUS = 'prepared_final_followup_successor_runtime_not_admitted'
RECONSTRUCTED_STATUS = 'reconstructed_final_followup_successor_runtime_not_admitted'


def _snapshot(registry, entry_reference):
    from tools.run_vcsf_core_points import BOOTSTRAP

    entrypoint = Path(entry_reference['file']).relative_to(registry.root).as_posix()
    sources = runtime_snapshot(registry.root, entrypoint=entrypoint)
    # The shared snapshot includes only its chosen experiment entry, not the
    # separate bootstrap whose executed bytes source_binding also verifies.
    sources[BOOTSTRAP] = file_digest(child(registry.root, BOOTSTRAP))
    return sources


def _reference_path(reference):
    check(type(reference) is dict and set(reference) == {'file', 'sha256'}
        and type(reference['file']) is str and bool(reference['file'].strip())
        and reference['file'] == reference['file'].strip()
        and not any(ord(c) < 32 for c in reference['file']) and is_sha(reference['sha256']),
        'Require exact file/sha256 reference')
    path = plain(Path(reference['file']))
    check(not (path.parent / 'failure.json').exists(), 'Failed runtime input evidence')
    return path


def verify_loaded_source(registry, sources, preparation_entry_reference):
    """Verify the actual preparation caller, not an as-yet unbound executor."""
    from tools.run_vcsf_core_points import source_binding, ROOT
    from .vcsf_ablation_execution_contract import _verify_process_binding

    path = _reference_path(preparation_entry_reference)
    check(path.parent == registry.root / 'experiments' and path.suffix == '.py'
        and registry.root == ROOT, 'Preparation entry must belong to this source workspace')
    entrypoint = path.relative_to(registry.root).as_posix()
    entry = sys.modules['__main__']
    check(getattr(entry, '__file__', None) == str(path)
        and getattr(entry, '_ENTRY_SOURCE_SHA256', None) == preparation_entry_reference['sha256']
        and sources.get(entrypoint) == preparation_entry_reference['sha256'],
        'Use the pinned source-bound preparation entry')
    _verify_process_binding(registry)
    loaded = source_binding(entrypoint, preparation_entry_reference['sha256'],
        getattr(entry, '_BOOTSTRAP_SOURCE_SHA256', None))
    check(all(sources.get(name) == digest for name, digest in loaded.items()),
        'Loaded preparation source differs from runtime snapshot')


def _unchanged(read, registry, request, sources):
    """Check evidence and snapshot stability, independent of the caller process."""
    for key in ('preparation', 'scope_reference', 'partition_reference', 'preparation_entry_reference'):
        _reference_path(request[key])
    for item in request['input_references'].values():
        for reference in item.values():
            _reference_path(reference)
    read.unchanged()
    check(sources == _snapshot(registry, request['preparation_entry_reference']),
        'Runtime source changed during preparation or publication')


def _request_sources(read, registry, request, max_images):
    check(type(request) is dict and set(request) == {'preparation', 'input_report',
        'input_references', 'scope_reference', 'partition_reference',
        'preparation_entry_reference', 'max_images'}, 'Unexpected successor runtime request fields')
    check(canonical_hash(request['max_images']) == canonical_hash(max_images),
        'Requested image limit differs from pinned runtime request')
    for key in ('preparation', 'scope_reference', 'partition_reference', 'preparation_entry_reference'):
        _reference_path(request[key])
    entry_path = Path(request['preparation_entry_reference']['file'])
    check(entry_path.parent == registry.root / 'experiments' and entry_path.suffix == '.py',
        'Require a direct preparation entry in this workspace')
    sources = _snapshot(registry, request['preparation_entry_reference'])
    entrypoint = entry_path.relative_to(registry.root).as_posix()
    check(sources.get(entrypoint) == request['preparation_entry_reference']['sha256'],
        'Pinned preparation entry differs from snapshot')
    read.bytes(entry_path, request['preparation_entry_reference']['sha256'])
    return sources


def reconstruct_runtime(read, registry, request, *, devices=None, max_images=None):
    """Rebuild pinned evidence without asserting a preparation-process identity.

    Admission consumers may call this under their own entry. The caller pins
    the request; all scope, projection, source and unchanged checks still apply.
    No stop, qualification, process ownership or execution admission is granted.
    """
    sources = _request_sources(read, registry, request, max_images)
    catalogue = _catalogue(read, registry, request['preparation'], request['input_report'])
    partition = read.json(_reference_path(request['partition_reference']), request['partition_reference']['sha256'])
    saved_scope = read.json(_reference_path(request['scope_reference']), request['scope_reference']['sha256'])
    scope = verify_successor_scope(registry, catalogue, partition, saved_scope,
        partition_reference=request['partition_reference'])
    check(canonical_hash(scope['max_images']) == canonical_hash(max_images),
        'Pinned scope image limit differs')
    check(devices is None or canonical_hash(devices) == canonical_hash(scope['devices']),
        'Requested devices differ from reconstructed successor assignment')
    check({2, 3} <= set(partition['completed_group_ids']),
        'Current groups 2/3 must be complete-group origins, never successor recomputation')
    check(all(name in sources and sources[name] == catalogue['producer_runtime_sha256'].get(name)
        for name in METHOD_FILES), 'Frozen numerical implementation changed')
    inputs = request['input_references']
    check(type(inputs) is dict and set(inputs) == {g['dataset'] for g in scope['groups']},
        'Inputs must cover exactly the remaining datasets')
    for item in inputs.values():
        check(type(item) is dict and set(item) == {'publication', 'audit'},
            'Require input publication and independent readback')
        for reference in item.values():
            _reference_path(reference)
    bindings = []
    category = 'diagnostics' if max_images is not None else 'experiments'
    for group in scope['groups']:
        refs = inputs[group['dataset']]
        directory = registry.root / 'outputs' / category / EXECUTION_PROTOCOL / 'preparation' / 'groups' / '{:06d}'.format(group['group_id'])
        prepared = prepare_bound_group_calls(read, registry, refs['publication'], refs['audit'],
            group['group_id'], str(directory), 'cuda:0', max_images=max_images,
            expected_catalogue_sha256=scope['catalogue_content_sha256'])
        projected = prepared['projected_inputs']
        executed = execution_projection(projected, max_images)
        check(prepared['group_id'] == group['group_id']
            and prepared['input_binding_sha256'] == projected['binding_sha256']
            and [row['target_id'] for row in prepared['evaluations']] == group['targets'],
            'Prepared group identity or full target panel differs')
        bindings.append(dict(group_id=group['group_id'],
            input_binding_sha256=projected['binding_sha256'],
            execution_input_binding_sha256=executed['binding_sha256']))
    for name, digest in sources.items():
        read.bytes(child(registry.root, name), digest)
    _unchanged(read, registry, request, sources)
    return dict(schema_version=1, status=RECONSTRUCTED_STATUS, scope=scope,
        scope_reference=deepcopy(request['scope_reference']),
        partition_reference=deepcopy(request['partition_reference']),
        preparation_entry_reference=deepcopy(request['preparation_entry_reference']),
        preparation_request=deepcopy(request), runtime_sha256=deepcopy(sources),
        input_references=deepcopy(inputs), group_input_bindings=bindings,
        preparation_source_binding_verified=False, scope_and_partition_bytes_verified=True,
        predecessor_stop_verified=False, qualification_references_authenticated=False,
        current_groups_completion_authenticated=False, device_availability_verified=False,
        execution_armed=False, formal_execution_admission=False, scientific_acceptance=False,
        model_calls=0, AP_replays=0)


def compile_runtime(read, registry, request, *, devices=None, max_images=None):
    """Wrap reconstruction with actual preparation-process checks on both sides."""
    sources = _request_sources(read, registry, request, max_images)
    verify_loaded_source(registry, sources, request['preparation_entry_reference'])
    runtime = reconstruct_runtime(read, registry, request, devices=devices, max_images=max_images)
    check(sources == runtime['runtime_sha256'], 'Runtime source changed across reconstruction')
    verify_loaded_source(registry, sources, request['preparation_entry_reference'])
    runtime.update(status=STATUS, preparation_source_binding_verified=True)
    return runtime


def prepare_runtime(registry, request_reference, *, devices=None, max_images=None, output=None):
    """Inspect without writes, or publish a fresh runtime.json, always unarmed."""
    read = Evidence()
    request = read.json(_reference_path(request_reference), request_reference['sha256'])
    runtime = compile_runtime(read, registry, request, devices=devices, max_images=max_images)
    runtime['request_reference'] = deepcopy(request_reference)
    _reference_path(request_reference)
    if output is None:
        return dict(status='successor_runtime_inspected_not_published', scope=runtime['scope'],
            execution_armed=False, formal_execution_admission=False)
    output = plain(Path(output).absolute())
    check(output.parent == registry.root / 'outputs/plans' / EXECUTION_PROTOCOL and not output.exists(),
        'Use a fresh direct successor runtime leaf; never overwrite or resume')
    output.mkdir(parents=True, exist_ok=False)
    try:
        _unchanged(read, registry, request, runtime['runtime_sha256'])
        verify_loaded_source(registry, runtime['runtime_sha256'], request['preparation_entry_reference'])
        atomic_json(output / 'runtime.json', runtime)
        reference = dict(file=str(output / 'runtime.json'), sha256=file_digest(output / 'runtime.json'))
        read.json(output / 'runtime.json', reference['sha256'])
        _unchanged(read, registry, request, runtime['runtime_sha256'])
        verify_loaded_source(registry, runtime['runtime_sha256'], request['preparation_entry_reference'])
        _reference_path(request_reference)
        return dict(status=STATUS, runtime_reference=reference, scope_sha256=runtime['scope']['scope_sha256'],
            group_ids=runtime['scope']['group_ids'], execution_armed=False, formal_execution_admission=False)
    except BaseException as exc:
        atomic_json(output / 'failure.json', dict(status='failed', error_type=type(exc).__name__,
            execution_armed=False, formal_execution_admission=False))
        raise
