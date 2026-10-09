"""Read one complete predecessor group, pending outer migration qualification.

collect_group_inputs takes pinned original runtime/admission, worker_group and
worker request references (exact file/sha256). It does not require coordinator
or worker completion. Original scientific checks are reused without alteration.

planned_stop_failure_references is an explicit caller declaration of preserved
planned-stop metadata, limited to this execution root and owning worker's
failure.json. References are authenticated as bytes, not as proof of the cause
of failure or of shutdown. The outer qualifier MUST authenticate that cause.
No failure anywhere in the selected group is permitted, including attack and
target failures. Undeclared ancestor failures are rejected. Other groups are
not collected or accepted. No files, models, processes or AP replay are created.
"""
from copy import deepcopy
from pathlib import Path

from tools.audit_vcsf_cpu_analysis import Evidence, check, plain, same, sha
from tools import vcsf_final_followup_result_inputs as original
from tools.vcsf_final_followup_terminal_audit import _group
from lgp.registry import Registry
from lgp.runners.vcsf_research_plan import canonical_hash


STATUS = 'whole_group_inputs_validated_pending_outer_qualification'


class GroupInputsError(RuntimeError):
    """Preserve failure_report externally; it is never partial qualification."""

    def __init__(self, report):
        self.failure_report = report
        super().__init__(report['error'])


def _reference_path(reference):
    check(type(reference) is dict and set(reference) == {'file', 'sha256'}
        and type(reference['file']) is str and original.is_sha(reference['sha256']),
        'Require exact file/sha256 reference')
    return plain(reference['file'])


def _control_json(read, path, reference=None):
    path = plain(path)
    reference = dict(file=str(path), sha256=sha(path)) if reference is None else reference
    check(_reference_path(reference) == path, 'Control reference escaped expected path')
    value = original._json(read.bytes(path, reference['sha256'], original.METADATA_BYTES))
    return deepcopy(reference), value


def _ancestor_failures(read, root, worker_directory, references):
    check(type(references) is list, 'Require explicit planned-stop failure reference list')
    allowed = {root / 'failure.json', worker_directory / 'failure.json'}
    supplied, records = set(), []
    for reference in references:
        path = _reference_path(reference)
        check(path in allowed and path not in supplied,
            'Duplicate or non-ancestor planned-stop failure reference')
        supplied.add(path)
        ref, value = _control_json(read, path, reference)
        check(type(value) is dict and value.get('status') == 'failed'
            and value.get('scientific_acceptance', False) is False,
            'Unexpected ancestor failure metadata')
        records.append(dict(reference=ref, metadata=value))
    observed = {p for p in allowed if p.exists() or p.is_symlink()}
    check(observed == supplied, 'Undeclared or missing ancestor failure metadata')
    return sorted(records, key=lambda row: row['reference']['file'])


def _positive(value):
    return type(value) is int and value > 0


def _lane_binding(read, root, registry, scope, runtime_reference, admission_reference,
        request_reference, group_id):
    relative = root.relative_to(plain(registry.root)).as_posix()
    binding_ref, binding = _control_json(read, root / 'plan_binding.json')
    same(binding, dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
        scope_sha256=scope['scope_sha256'], execution_root=relative, devices=scope['devices'],
        lanes=scope['lanes'], payload_policy=scope['payload_policy']), 'Original plan binding differs')
    assignment_ref, assignment = _control_json(read, root / 'execution_assignment.json')
    check(type(assignment) is dict and set(assignment) == {'workers', 'lanes'},
        'Unexpected assignment schema')
    same(assignment['lanes'], scope['lanes'], 'Original assignment lanes differ')
    entries = assignment['workers']
    slots = [slot for slot, lane in enumerate(scope['lanes']) if lane]
    check(type(entries) is list and len(entries) == len(slots), 'Incomplete worker assignment')
    pids = set()
    selected = None
    for slot, entry in zip(slots, entries):
        check(type(entry) is dict and set(entry) == {'slot', 'pid', 'group_ids', 'request_reference'}
            and type(entry['slot']) is int and entry['slot'] == slot and _positive(entry['pid'])
            and entry['pid'] not in pids, 'Invalid or duplicate assigned worker identity')
        pids.add(entry['pid'])
        same(entry['group_ids'], scope['lanes'][slot], 'Assigned group lane differs')
        check(_reference_path(entry['request_reference']) == root / 'workers' / str(slot) / 'request.json',
            'Assigned worker request escaped its slot')
        if group_id in entry['group_ids']:
            check(selected is None, 'Group assigned more than once')
            selected = entry
    check(selected is not None, 'Group has no original worker lane')
    same(selected['request_reference'], request_reference, 'Caller request differs from assigned request')
    _, request = _control_json(read, _reference_path(request_reference), request_reference)
    slot = selected['slot']
    expected = dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
        execution_root=relative, scope_sha256=scope['scope_sha256'], worker_slot=slot,
        physical_device=scope['devices'][slot], group_ids=scope['lanes'][slot],
        run_binding_sha256=binding_ref['sha256'])
    ownership = {'coordinator_pid', 'coordinator_start_ticks', 'reservation_fd',
        'reservation_identity', 'gpu_uuid'}
    check(type(request) is dict and set(request) == set(expected) | ownership,
        'Unexpected original request schema')
    for key, value in expected.items():
        same(request[key], value, 'Original request ' + key + ' differs')
    check(_positive(request['coordinator_pid']) and _positive(request['coordinator_start_ticks'])
        and type(request['reservation_fd']) is int and request['reservation_fd'] >= 0
        and type(request['reservation_identity']) is list and len(request['reservation_identity']) == 2
        and all(type(i) is int and i >= 0 for i in request['reservation_identity'])
        and type(request['gpu_uuid']) is str and request['gpu_uuid'].startswith('GPU-')
        and len(request['gpu_uuid']) > 4, 'Invalid saved worker ownership identity')
    return dict(plan_binding_reference=binding_ref, assignment_reference=assignment_ref,
        worker_request_reference=deepcopy(request_reference), worker_request=request,
        assigned_worker=deepcopy(selected))


def _completed_prefix(read, request_reference, lane, group_id, worker):
    directory = _reference_path(request_reference).parent
    reference, completed = _control_json(read, directory / 'completed_groups.json')
    check(type(completed) is list and completed and all(type(row) is dict
        and type(row.get('group_id')) is int for row in completed), 'Invalid completed group prefix')
    ids = [row['group_id'] for row in completed]
    same(ids, lane[:len(ids)], 'Completed groups are not a unique original-lane prefix')
    check(group_id in ids, 'Selected group is absent from completed prefix')
    same(completed[ids.index(group_id)], worker, 'Worker group differs from completed prefix')
    return reference


def _inventory(read, payloads, directory, projected, cells, before):
    expected_pngs = {'attack/images/%012d.png' % i for i in projected['ordered_image_ids']}
    check({p for p in before if p.lower().endswith('.png')} == expected_pngs,
        'Whole-group PNG inventory differs')
    for suffix, key in (('metrics.json', 'metrics_reference'),
            ('predictions_artifact.json', 'sidecar_reference'), ('.gz', 'archive_reference')):
        expected = {_reference_path(c[key]).relative_to(directory).as_posix() for c in cells}
        check({p for p in before if p.endswith(suffix)} == expected, 'Whole-group target inventory differs')
    same(sorted(p.name for p in (directory / 'evaluations').iterdir()),
        sorted(projected['targets']), 'Unexpected target directory')
    inventory = {}
    for name in before:
        path = directory / name
        absolute = str(path)
        digest = read.checked.get(absolute)
        if absolute in payloads.files:
            payload_digest = payloads.files[absolute]['sha256']
            check(digest is None or digest == payload_digest, 'Conflicting group file digest')
            digest = payload_digest
        if digest is None:
            digest = sha(path)
            read.bytes(path, digest)
        inventory[name] = digest
    return inventory


def collect_group_inputs(runtime_reference, admission_reference, worker_group_reference,
        worker_request_reference, *, registry=None, planned_stop_failure_references=None):
    """Validate one full group (2..16); never qualify reuse or certify shutdown.

    The original runtime/admission must still pass the unchanged consumer on
    its matching source workspace. Ancestor-stop declarations do not exempt
    runtime, admission, input publications or selected-group failure checks.
    """
    read, controls, payloads = original._Metadata(), Evidence(), original._Payloads()
    context = dict(stage='authentication', group_id=None, target=None, image_id=None)
    cells = []
    references = dict(runtime_reference=deepcopy(runtime_reference),
        admission_reference=deepcopy(admission_reference), worker_group_reference=deepcopy(worker_group_reference),
        worker_request_reference=deepcopy(worker_request_reference),
        planned_stop_failure_references=deepcopy([] if planned_stop_failure_references is None
            else planned_stop_failure_references))
    try:
        runtime_path = _reference_path(runtime_reference)
        check(len(runtime_path.parents) >= 5, 'Incomplete runtime namespace')
        producer = runtime_path.parents[4]
        registry = Registry(producer) if registry is None else registry
        check(plain(registry.root) == producer, 'Registry differs from original producer')
        path = _reference_path(worker_group_reference)
        directory, root = path.parent, path.parent.parent.parent
        check(path.name == 'worker_group.json' and directory.parent.name == 'groups'
            and root.parent == producer / 'outputs/experiments/vcsf_final_followup_execution',
            'Group escaped original formal execution namespace')
        original._healthy(directory.parent)
        read.directories.add(directory.parent)
        worker = original._reference(read, worker_group_reference, directory / 'worker_group.json')
        group_id = worker.get('group_id')
        check(type(group_id) is int and 2 <= group_id <= 16 and directory.name == '%06d' % group_id,
            'Invalid original whole-group identity')
        context['group_id'] = group_id
        request_path = _reference_path(worker_request_reference)
        check(request_path.name == 'request.json' and request_path.parent.parent == root / 'workers'
            and request_path.parent.name in ('0', '1'), 'Worker request escaped original root')
        original._healthy(request_path.parent.parent)
        read.directories.add(request_path.parent.parent)
        failures = _ancestor_failures(controls, root, request_path.parent,
            [] if planned_stop_failure_references is None else planned_stop_failure_references)
        runtime = original._reference(read, runtime_reference)
        bound = original.load_bound_execution_group(read, registry, runtime_reference, admission_reference,
            group_id, str(directory), 'cuda:0', max_images=None)
        scope, group, prepared = bound['admission']['scope'], bound['group'], bound['prepared']
        original._formal_scope(scope, list(registry.target_ids()))
        same(scope, runtime['scope'], 'Authenticated original scope differs')
        same(group, next(g for g in scope['groups'] if g['group_id'] == group_id), 'Bound group differs')
        binding = _lane_binding(controls, root, registry, scope, runtime_reference,
            admission_reference, worker_request_reference, group_id)
        prefix = _completed_prefix(controls, worker_request_reference,
            binding['worker_request']['group_ids'], group_id, worker)
        _group(read, root, group, worker, runtime_reference, admission_reference,
            worker_request_reference, scope)
        projected = prepared['projected_inputs']
        same(worker['input_binding_sha256'], projected['binding_sha256'], 'Full input projection differs')
        same(worker['execution_input_binding_sha256'], projected['binding_sha256'], 'Partial execution binding')
        check(type(projected['images']) is int and projected['images'] == group['images']
            and len(projected['ordered_image_ids']) == group['images'], 'Incomplete whole-group manifest')
        image_ids = projected['ordered_image_ids']
        check(type(image_ids) is list and all(type(i) is int for i in image_ids)
            and image_ids == sorted(set(image_ids)), 'Invalid full canonical image IDs')
        same(projected['targets'], group['targets'], 'Projected full target panel differs')
        before = original._tree(directory)
        context['stage'] = 'projection_metadata'
        original._projection_metadata(read, projected)
        parameters = prepared['attack']['parameter_overrides']
        context['stage'] = 'generation_payloads'
        generated = original._generation(read, payloads, directory, worker, projected, parameters, context)
        for target in worker['targets']:
            context.update(stage='target_payload', target=target['target'])
            cells.append(original._cell(read, payloads, directory, group, projected, generated, target, parameters))
        same([(c['group_id'], c['target']) for c in cells],
            [(group_id, t) for t in group['targets']], 'Whole-group cell coverage differs')
        check(len(cells) == 16 and generated['png_count'] == group['images'], 'Incomplete group payloads')
        context.update(stage='final_readback', target=None, image_id=None)
        inventory = _inventory(read, payloads, directory, projected, cells, before)
        auditor_root, sources = original._sources(read)
        source = plain(Path(__file__).absolute())
        sources[source.relative_to(plain(auditor_root)).as_posix()] = sha(source)
        read.bytes(source, sources[source.relative_to(plain(auditor_root)).as_posix()])
        refreshed = original.load_bound_execution_group(read, registry, runtime_reference, admission_reference,
            group_id, str(directory), 'cuda:0', max_images=None)
        check(refreshed == bound, 'Original admission or prepared inputs changed')
        payloads.unchanged(rehash=True)
        read.unchanged()
        controls.unchanged()
        same(original._tree(directory), before, 'Selected group inventory changed')
        same(_ancestor_failures(controls, root, request_path.parent,
            [] if planned_stop_failure_references is None else planned_stop_failure_references),
            failures, 'Ancestor failure metadata changed')
        controls.unchanged()
        return original._seal(dict(schema_version=1, status=STATUS, **references,
            execution_root=str(root), group_id=group_id, group=deepcopy(group),
            catalogue_content_sha256=canonical_hash(bound['catalogue']), scope_sha256=scope['scope_sha256'],
            runtime_snapshot_sha256=canonical_hash(runtime['runtime_sha256']),
            projected_inputs=deepcopy(projected), original_lane_binding=binding,
            completed_prefix_reference=prefix, generation=generated, cells=cells,
            group_count=1, images=group['images'], png_count=generated['png_count'], target_cells=16,
            max_images=None, diagnostic_only=False, raw_inventory=inventory,
            payload_files=deepcopy(payloads.files), metadata_sha256=dict(read.checked),
            control_metadata_sha256=dict(controls.checked), auditor_source_root=auditor_root,
            auditor_source_sha256=sources, preserved_ancestor_failures=failures,
            planned_stop_cause_authenticated=False, predecessor_stop_verified=False,
            complete_group_qualification_accepted=False, full_coordinator_completion_required=False,
            original_input_admission_authenticated=True, group_payload_inputs_verified=True,
            payload_readback='sha256_and_stat_identity', producer_files_modified=False,
            **original._flags()))
    except Exception as error:
        report = original._seal(dict(schema_version=1, status='whole_group_inputs_failed',
            **references, context=context, error_type=type(error).__name__, error=str(error),
            checked_cells=[dict(group_id=c['group_id'], target=c['target']) for c in cells],
            metadata_sha256=dict(read.checked), control_metadata_sha256=dict(controls.checked),
            payload_files=deepcopy(payloads.files), partial_evidence_is_acceptance=False,
            complete_group_qualification_accepted=False, producer_files_modified=False, **original._flags()))
        raise GroupInputsError(report) from error
