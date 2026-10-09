"""Read terminal follow-up evidence without issuing execution/result acceptance."""
from copy import deepcopy

from tools.audit_vcsf_cpu_analysis import check, child, is_sha, plain, same, sha, value_sha


TERMINAL = 'finished_pending_independent_acceptance'
GROUP_TERMINAL = 'generated_and_evaluated_pending_independent_acceptance'


def _integer(value, expected, label):
    check(type(value) is int and value == expected, label + ' differs')


def _positive(value, label):
    check(type(value) is int and value > 0, 'Invalid ' + label)


def _reference(read, reference, expected_path=None):
    check(type(reference) is dict and set(reference) == {'file', 'sha256'}
        and type(reference['file']) is str and is_sha(reference['sha256']),
        'Invalid pinned reference')
    path = plain(reference['file'])
    check(expected_path is None or path == expected_path, 'Reference escaped expected leaf')
    return read.json(path, reference['sha256'])


def _pending(record):
    check(record['independent_result_acceptance'] is False
        and record['scientific_acceptance'] is False, 'Unsupported terminal acceptance claim')


def _inventory(root):
    paths = tuple(sorted(root.rglob('*')))
    for path in paths:
        plain(path)
        check(path.name != 'failure.json', 'Execution contains failure evidence')
    return tuple(str(p.relative_to(root)) for p in paths)


def _group(read, root, expected, record, runtime_reference, admission_reference,
        request_reference, scope):
    group_id = expected['group_id']
    directory = root / 'groups' / ('%06d' % group_id)
    path = directory / 'worker_group.json'
    same(read.json(path, sha(path)), record, 'Individual group differs from completed array')
    check(record['status'] == GROUP_TERMINAL, 'Group is not terminal')
    _integer(record['group_id'], group_id, 'Group identity')
    _pending(record)
    for key, value in (('runtime_reference', runtime_reference),
            ('admission_reference', admission_reference), ('scope_sha256', scope['scope_sha256']),
            ('diagnostic_only', scope['diagnostic_only']), ('payload_policy', scope['payload_policy'])):
        same(record[key], value, 'Group ' + key)
    images = scope['max_images'] if scope['max_images'] is not None else expected['images']
    _integer(record['images'], images, 'Group image count')
    same([t['target'] for t in record['targets']], expected['targets'], 'Group target order')
    full_binding = record['input_binding_sha256']
    execution_binding = record['execution_input_binding_sha256']
    check(is_sha(full_binding) and is_sha(execution_binding), 'Invalid group input binding')
    if scope['max_images'] is None:
        same(full_binding, execution_binding, 'Full execution input binding')
    run = _reference(read, record['generation'], directory / 'attack' / 'run.json')
    check(run['status'] == 'complete' and run['formal_AP_eligible'] is False,
        'Generation is incomplete or claims formal eligibility')
    for key in ('dataset', 'split', 'source', 'seed', 'seed_schedule', 'parameters_sha256'):
        same(run[key], expected[key], 'Run ' + key)
    check(type(run['seed']) is int, 'Invalid run seed')
    same(run['parameters_sha256'], scope['parameters_sha256'], 'Scope parameters')
    same(value_sha(run['parameters']), scope['parameters_sha256'], 'Run parameter content')
    for key, value in (('requested_images', images), ('successful_images', images), ('failed_images', 0)):
        _integer(run[key], value, 'Run ' + key)
    metadata = dict(protocol=scope['protocol'], scope_sha256=scope['scope_sha256'],
        group_id=group_id, dataset=expected['dataset'], source=expected['source'],
        seed=expected['seed'], seed_schedule=expected['seed_schedule'],
        parameters_sha256=expected['parameters_sha256'], input_binding_sha256=full_binding,
        runtime_reference=runtime_reference, admission_reference=admission_reference,
        worker_request_reference=request_reference, diagnostic_only=scope['diagnostic_only'])
    same(run['run_metadata'], metadata, 'Invocation metadata differs')
    research = dict(metadata,
        fidelity_status='frozen_final_followup_pending_result_acceptance',
        semantic_contract='final_configuration_after_fullval_selection_not_independent_confirmation',
        global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False)
    same(run['research_execution'], research, 'Research execution metadata differs')
    summary = record['generation_metadata']
    same(summary['projected_binding_sha256'], execution_binding, 'Execution projection binding')
    same(summary['run_content_sha256'], value_sha(run), 'Generation metadata run binding')
    return dict(group_id=group_id, generation_reference=deepcopy(record['generation']),
        input_binding_sha256=full_binding, execution_input_binding_sha256=execution_binding,
        completed_evaluations=len(expected['targets']))


def audit_terminal_chain(read, execution_root, runtime_reference, admission_reference, scope,
        *, expected_device_count=2):
    """Check saved two-worker terminal linkage using an existing Evidence reader.

    The caller authenticates catalogue, inputs, projections, runtime/admission
    provenance and metric payloads separately. Scope must be its reconstructed
    scope, not a scope inferred from this execution. Unpinned terminal files are
    snapshotted and rehashed, not treated as independently signed receipts.
    Returns a pending-acceptance report; raises on missing/conflicting evidence.
    Does not inspect /proc, NVML, reservations, or create a GPU owner.
    """
    check(type(expected_device_count) is int and expected_device_count in (2, 4),
        'Unsupported explicitly expected device count')
    root = plain(execution_root)
    before = _inventory(root)
    runtime = _reference(read, runtime_reference)
    admission = _reference(read, admission_reference)
    same(runtime['scope'], scope, 'Runtime scope differs')
    same(scope['scope_sha256'], value_sha({k: v for k, v in scope.items()
        if k != 'scope_sha256'}), 'Scope content digest differs')
    same(admission['runtime_reference'], runtime_reference, 'Admission runtime differs')
    same(admission['scope_sha256'], scope['scope_sha256'], 'Admission scope differs')
    groups, ids, lanes = scope['groups'], scope['group_ids'], scope['lanes']
    check(type(ids) is list and ids and all(type(i) is int and i > 0 for i in ids)
        and len(set(ids)) == len(ids), 'Invalid scope group IDs')
    same([g['group_id'] for g in groups], ids, 'Scope group order differs')
    check(len(scope['devices']) == len(lanes) == expected_device_count
        and all(type(lane) is list and lane for lane in lanes)
        and len(set(scope['devices'])) == expected_device_count,
        'Require the explicitly expected nonempty distinct device lanes')
    flattened = [i for lane in lanes for i in lane]
    same(sorted(flattened), sorted(ids), 'Scope lanes omit or duplicate groups')
    for lane in lanes:
        same(lane, [i for i in ids if i in lane], 'Scope lane order differs')
    diagnostic = scope['max_images'] is not None
    check(scope['max_images'] is None or (type(scope['max_images']) is int and scope['max_images'] == 1),
        'Unsupported image limit')
    check(scope['diagnostic_only'] is diagnostic, 'Scope diagnostic classification differs')
    for group in groups:
        _positive(group['images'], 'scope image count')
        check(type(group['seed']) is int and type(group['targets']) is list and group['targets']
            and len(set(group['targets'])) == len(group['targets']), 'Invalid scope group')
        same(group['parameters_sha256'], scope['parameters_sha256'], 'Scope group parameters')
    _integer(scope['target_cells'], sum(len(g['targets']) for g in groups), 'Scope target cells')
    _integer(scope['images'], sum(scope['max_images'] or g['images'] for g in groups), 'Scope images')

    def local(path):
        return read.json(path, sha(path))

    binding_path = root / 'plan_binding.json'
    binding = local(binding_path)
    runtime_path = plain(runtime_reference['file'])
    check(len(runtime_path.parents) >= 5, 'Runtime path lacks workspace namespace')
    relative = binding['execution_root']
    check(child(runtime_path.parents[4], relative) == root, 'Execution root binding differs')
    category = 'diagnostics' if diagnostic else 'experiments'
    check(root.parent == runtime_path.parents[4] / 'outputs' / category / scope['protocol'],
        'Execution root scope namespace differs')
    same(binding, dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
        scope_sha256=scope['scope_sha256'], execution_root=relative, devices=scope['devices'],
        lanes=lanes, payload_policy=scope['payload_policy']), 'Plan binding differs')
    assignment = local(root / 'execution_assignment.json')
    same(assignment['lanes'], lanes, 'Assignment lanes differ')
    entries = assignment['workers']
    same([e['slot'] for e in entries], list(range(expected_device_count)),
        'Require exactly the expected ordered workers')
    completion = local(root / 'completion.json')
    same(local(root / 'execution_state.json'), completion, 'Root state/completion differ')
    expected_root = dict(status=TERMINAL, execution_root=relative, scope_sha256=scope['scope_sha256'],
        group_ids=ids, worker_count=expected_device_count,
        requested_device_count=expected_device_count, diagnostic_only=diagnostic,
        independent_result_acceptance=False, scientific_acceptance=False,
        payload_policy=scope['payload_policy'])
    same(completion, expected_root, 'Root is not successful terminal scope')
    by_id = {g['group_id']: g for g in groups}
    results, coordinators, uuids, pids, reservations = [], [], [], [], []
    for slot, entry in enumerate(entries):
        lane = lanes[slot]
        directory = root / 'workers' / str(slot)
        same(entry['group_ids'], lane, 'Assignment group IDs differ')
        _positive(entry['pid'], 'assigned worker PID')
        reference = entry['request_reference']
        request = _reference(read, reference, directory / 'request.json')
        expected_request = dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
            execution_root=relative, scope_sha256=scope['scope_sha256'], worker_slot=slot,
            physical_device=scope['devices'][slot], group_ids=lane,
            run_binding_sha256=read.checked[str(binding_path)])
        owner_keys = {'coordinator_pid', 'coordinator_start_ticks', 'reservation_fd',
            'reservation_identity', 'gpu_uuid'}
        check(set(request) == set(expected_request) | owner_keys, 'Request schema differs')
        for key, value in expected_request.items():
            same(request[key], value, 'Request ' + key)
        for key in ('coordinator_pid', 'coordinator_start_ticks'):
            _positive(request[key], key)
        check(type(request['reservation_fd']) is int and request['reservation_fd'] >= 0,
            'Invalid reservation descriptor')
        identity = request['reservation_identity']
        check(type(identity) is list and len(identity) == 2
            and all(type(v) is int and v >= 0 for v in identity), 'Invalid reservation identity')
        check(type(request['gpu_uuid']) is str and request['gpu_uuid'].startswith('GPU-')
            and len(request['gpu_uuid']) > 4, 'Invalid GPU UUID')
        coordinators.append((request['coordinator_pid'], request['coordinator_start_ticks']))
        reservations.append((request['reservation_fd'], tuple(identity)))
        uuids.append(request['gpu_uuid'])
        pids.append(entry['pid'])
        done = local(directory / 'completion.json')
        state = local(directory / 'execution_state.json')
        same(state, {k: v for k, v in done.items() if k not in
            ('groups', 'independent_result_acceptance', 'scientific_acceptance')},
            'Worker state/completion differ')
        check(done['status'] == TERMINAL and done['current'] is None, 'Worker is not terminal')
        _pending(done)
        same(done['request_reference'], reference, 'Worker completion request differs')
        same(done['group_ids'], lane, 'Worker completion lane differs')
        _integer(done['worker_slot'], slot, 'Worker slot')
        _integer(done['completed_groups'], len(lane), 'Worker completed groups')
        evaluations = sum(len(by_id[i]['targets']) for i in lane)
        _integer(done['completed_evaluations'], evaluations, 'Worker completed evaluations')
        _integer(done['failed_records'], 0, 'Worker failed records')
        same(done['groups'], local(directory / 'completed_groups.json'), 'Completed group arrays differ')
        same([g['group_id'] for g in done['groups']], lane, 'Completed group order differs')
        gpu = done['gpu_context_registration']
        check(gpu['schema'] == 'vcsf_gpu_context_registration_v1', 'Unknown GPU registration schema')
        same(gpu['gpu_uuid'], request['gpu_uuid'], 'GPU registration UUID differs')
        _integer(gpu['worker_pid'], entry['pid'], 'GPU registration worker PID')
        _positive(gpu['worker_pid_start_ticks'], 'GPU registration start ticks')
        _positive(gpu['nvml_pid'], 'registered NVML PID')
        same(gpu['before_context_pids'], [], 'GPU was not recorded idle')
        same(gpu['registered_context_pids'], [gpu['nvml_pid']], 'Ambiguous GPU registration')
        same(gpu['assumption'], 'NVML_enumerates_all_active_compute_context_processes_on_selected_device',
            'GPU registration assumption differs')
        checked_groups = [_group(read, root, by_id[g['group_id']], g, runtime_reference,
            admission_reference, reference, scope) for g in done['groups']]
        results.append(dict(worker_slot=slot, group_ids=deepcopy(lane), completed_groups=len(lane),
            completed_evaluations=evaluations, request_reference=deepcopy(reference),
            gpu_context_registration=deepcopy(gpu), groups=checked_groups))
    check(len(set(coordinators)) == 1
        and len(set(uuids)) == len(set(pids)) == expected_device_count,
        'Worker/coordinator/GPU identities conflict')
    check(len({r[0] for r in reservations})
        == len({r[1] for r in reservations}) == expected_device_count,
        'Workers share a reservation identity')
    same(sorted(p.relative_to(root).as_posix() for p in root.rglob('worker_group.json')),
        sorted('groups/%06d/worker_group.json' % i for i in ids), 'Unexpected individual group records')
    same(sorted(p.name for p in (root / 'workers').iterdir()),
        [str(i) for i in range(expected_device_count)], 'Unexpected worker leaves')
    same(sorted(p.name for p in (root / 'groups').iterdir()),
        sorted('%06d' % i for i in ids), 'Unexpected group leaves')
    read.unchanged()
    same(_inventory(root), before, 'Execution directory changed during audit')
    return dict(status='terminal_chain_verified_pending_acceptance', execution_root=str(root),
        runtime_reference=deepcopy(runtime_reference), admission_reference=deepcopy(admission_reference),
        scope_sha256=scope['scope_sha256'], group_ids=deepcopy(ids), workers=results,
        completed_groups=len(ids), completed_evaluations=scope['target_cells'], failed_records=0,
        evidence_sha256=dict(read.checked), terminal_chain_verified=True,
        independent_result_acceptance=False, scientific_acceptance=False,
        formal_execution_admission=False, formal_metrics_verified=False,
        model_execution_verified=False, current_gpu_ownership_verified=False,
        full_input_binding_authenticated=False, model_calls=0, AP_replays=0,
        evidence_limits=[
            'Caller authenticates catalogue, inputs, projections, admission and runtime provenance.',
            'Saved terminal records are not independent execution or scientific acceptance.',
            'Worker start ticks exist only in registration; assignment binds PID, request binds UUID.',
            'Coordinator PID/start ticks agree across requests, without an independent launch receipt.',
            'No live PID, NVML, reservation or physical-device mapping was inspected.',
            'No metrics, predictions, images or formal-admission bridge were verified.'])
