"""Consume independent successor admission, never issue it or observe live GPUs.

Runtime and group qualification use their existing producer schemas. The
following missing publication schemas are REQUIRED contracts, not proof that
their producers have run. All references are exact file/sha256, all source maps
are source-root-relative path/SHA256, and all receipts live in fresh direct
outputs/audits/<namespace>/<leaf>/receipt.json publications without failure.json.

ADMISSION_FIELDS describes vcsf_final_followup_successor_admission. Formal
admission requires DIAGNOSTIC_FIELDS in vcsf_final_followup_successor_diagnostic;
max-one diagnostic admission has diagnostic_acceptance=None (no circular proof).
Its independent review binds the terminal reference and a byte evidence seal.
The diagnostic proof pins its admission_reference; that max-one admission is
fully consumed without recursively requiring completed diagnostic evidence.
device_count describes four allocated devices, not four concurrent jobs: only
nonempty lanes require worker/context evidence.

STOP_FIELDS describes vcsf_final_followup_stop. capture_reference is the exact
unaltered capture_shutdown return object. CAPTURE_PROVENANCE_FIELDS describes
a separately source-bound capture invocation, not a Boolean attestation: it
binds original function/entry bytes, arguments, output, boot and observation
interval. monitor_launch_reference and every worker request/state bind the
expected stopped PID/start-tick set and original reserved devices. A raw capture
object alone is insufficient. No capture, signal, reservation, model or AP call
occurs here; launch-time ownership is still the executor's responsibility.

C payloads are not successor computation inputs. This admission consumer reads
qualification receipts, source bindings, reports, replay JSON and original group
identity records, NOT completed PNGs, prediction archives or group file trees.
The qualifier checked those payloads at qualification time. Final cross-root
result acceptance must recheck them; execution admission makes no statement
about their current integrity. This is a scope boundary, not a trust cache.
"""
from copy import deepcopy
from pathlib import Path
import re
import sys

from ..registry import Registry
from .vcsf_research_plan import canonical_hash, require
from .vcsf_common_anchor_execution_contract import runtime_snapshot
from .vcsf_final_followup_successor_contract import verify_successor_scope, compile_successor_scope
from . import vcsf_final_followup_successor_runtime as runtime_module
from tools.audit_vcsf_cpu_analysis import child, is_sha, plain
from tools import vcsf_final_followup_group_qualification as qualification
from tools import vcsf_final_followup_result_inputs as input_module
from tools.vcsf_final_followup_bound_dispatch import load_bound_execution_group
from tools import vcsf_final_followup_group_inputs as group_inputs
from tools.vcsf_followup_stop_evidence import validate_stop_observations


ENTRYPOINT = 'experiments/vcsf_final_followup_successor.py'
NAMESPACE = 'vcsf_final_followup_successor_admission'
STOP_NAMESPACE = 'vcsf_final_followup_stop'
DIAGNOSTIC_NAMESPACE = 'vcsf_final_followup_successor_diagnostic'
CHECKS = {'input_identity', 'frozen_method_identity', 'strict_checkpoint_loading',
    'loaded_process_binding', 'worker_isolation', 'resource_capacity', 'full_target_dispatch'}
ADMISSION_FIELDS = {'schema_version', 'status', 'errors', 'runtime_reference', 'scope_sha256',
    'partition_reference', 'executor_entry_reference', 'executor_runtime_sha256',
    'input_references', 'group_input_bindings', 'devices', 'predecessor_stop_reference',
    'completed_group_qualification_references', 'verified_checks', 'diagnostic_acceptance',
    'diagnostic_execution_admission', 'formal_execution_admission', 'auditor_source_sha256'}
STOP_FIELDS = {'schema_version', 'status', 'errors', 'capture_reference',
    'capture_provenance_reference', 'monitor_launch_reference', 'monitor_contract_reference',
    'worker_references', 'runtime_reference', 'admission_reference', 'execution_root',
    'auditor_source_sha256'}
CAPTURE_PROVENANCE_FIELDS = {'schema_version', 'status', 'function', 'entry_reference',
    'source_sha256', 'expected_processes', 'devices', 'capture_reference', 'boot_id',
    'started_monotonic_ns', 'finished_monotonic_ns'}
DIAGNOSTIC_FIELDS = {'schema_version', 'status', 'errors', 'formal_runtime_reference',
    'formal_scope_sha256', 'partition_reference', 'executor_entry_reference',
    'executor_runtime_sha256', 'runtime_reference', 'admission_reference', 'terminal_reference',
    'independent_review_reference', 'evidence_seal_reference', 'device_count', 'devices',
    'group_ids', 'target_cells', 'auditor_source_sha256'}
CAPTURE_SOURCES = {'tools/vcsf_followup_stop_capture.py', 'tools/vcsf_followup_stop_evidence.py',
    'tools/inspect_vcsf_followup_migration.py', 'src/lgp/runners/vcsf_structure_workers.py',
    'src/lgp/runners/vcsf_research_plan.py'}


def _exact(value, fields, label):
    require(type(value) is dict and set(value) == fields, label + ' schema differs')


def _version(value):
    require(type(value.get('schema_version')) is int and value['schema_version'] == 1,
        'Unsupported schema version')


def _same(left, right, message):
    require(canonical_hash(left) == canonical_hash(right), message)


def _integer(value, expected=None):
    require(type(value) is int and (value > 0 if expected is None else value == expected),
        'Invalid integer identity/count')
    return value


class _Evidence:
    """Track healthy publications separately from explicitly historical records."""

    def __init__(self, reader):
        self.reader, self.healthy = reader, set()

    def path(self, ref):
        _exact(ref, {'file', 'sha256'}, 'Reference')
        require(type(ref['file']) is str and ref['file'] == ref['file'].strip()
            and bool(ref['file']) and not any(ord(c) < 32 for c in ref['file'])
            and is_sha(ref['sha256']), 'Invalid file/SHA256 reference')
        return plain(ref['file'])

    def json(self, ref, *, expected=None, historical=False):
        path = self.path(ref)
        require(expected is None or path == expected, 'Evidence escaped bound path')
        if not historical:
            input_module._healthy(path.parent)
            self.healthy.add(path.parent)
        read_bytes = (getattr(self.reader, 'historical_bytes', self.reader.bytes)
            if historical else self.reader.bytes)
        return input_module._json(read_bytes(path, ref['sha256'], input_module.METADATA_BYTES))

    def receipt(self, ref, namespace):
        path = self.path(ref)
        require(len(path.parents) >= 5 and path.name == 'receipt.json'
            and path.parents[1].name == namespace and path.parents[2].name == 'audits'
            and path.parents[3].name == 'outputs', 'Receipt namespace differs')
        return self.json(ref), path

    def sources(self, root, sources, required=()):
        require(type(sources) is dict and sources and set(required) <= set(sources),
            'Incomplete source provenance')
        for name, digest in sources.items():
            require(type(name) is str and bool(name) and Path(name).suffix in ('.py', '.yaml', '.yml', '.txt')
                and is_sha(digest), 'Invalid source map')
            self.reader.bytes(child(root, name), digest)

    def unchanged(self):
        for path in self.healthy:
            input_module._healthy(path)
        self.reader.unchanged()
        for path in self.healthy:
            input_module._healthy(path)


def _executor_snapshot(registry):
    from tools.run_vcsf_core_points import BOOTSTRAP
    from ..io import file_digest
    result = runtime_snapshot(registry.root, entrypoint=ENTRYPOINT)
    result[BOOTSTRAP] = file_digest(child(registry.root, BOOTSTRAP))
    return result


def _verify_executor(registry, sources):
    from .vcsf_ablation_execution_contract import _verify_process_binding
    from tools.run_vcsf_core_points import source_binding, ROOT, BOOTSTRAP
    entry = sys.modules.get('__main__')
    require(registry.root == ROOT and getattr(entry, '__file__', None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, '_ENTRY_SOURCE_SHA256', None) == sources[ENTRYPOINT],
        'Use the exact source-bound successor executor entry')
    _verify_process_binding(registry)
    loaded = source_binding(ENTRYPOINT, sources[ENTRYPOINT], getattr(entry, '_BOOTSTRAP_SOURCE_SHA256', None))
    require(loaded.get(BOOTSTRAP) == sources[BOOTSTRAP]
        and all(sources.get(k) == v for k, v in loaded.items()), 'Loaded executor differs from snapshot')


def _runtime(evidence, registry, catalogue, scope, reference):
    path = evidence.path(reference)
    require(path.name == 'runtime.json' and path.parent.parent ==
        registry.root / 'outputs/plans/vcsf_final_followup_execution', 'Runtime namespace differs')
    saved = evidence.json(reference)
    request = evidence.json(saved['request_reference'])
    _same(request, saved['preparation_request'], 'Runtime request bytes differ')
    rebuilt = runtime_module.reconstruct_runtime(evidence.reader, registry, request,
        devices=scope['devices'], max_images=scope['max_images'])
    expected = dict(deepcopy(rebuilt), status=runtime_module.STATUS,
        preparation_source_binding_verified=True, request_reference=deepcopy(saved['request_reference']))
    _same(saved, expected, 'Published runtime differs from evidence reconstruction')
    verified = verify_successor_scope(registry, catalogue, rebuilt['scope']['partition'], scope,
        partition_reference=rebuilt['partition_reference'])
    _same(verified, rebuilt['scope'], 'Caller catalogue/scope differs from bound runtime')
    require(verified['group_ids'] and len(verified['devices']) == len(verified['lanes']) == 4,
        'Admission requires a nonempty four-device successor')
    require({2, 3} <= set(verified['partition']['completed_group_ids']), 'Groups 2/3 are not completed origins')
    return saved, verified


def _qualification(evidence, ref, group, catalogue):
    """Read the qualification chain, not current C PNG/archive bytes or trees.

    Stored inventories remain part of authenticated report semantics and replay
    bindings. They are not a request to re-read the completed group's payloads.
    """
    from tools import vcsf_final_followup_completed_group as completed
    if evidence.path(ref).parent.parent.name == completed.NAMESPACE:
        return completed.read_completed_group(evidence.reader, ref, group, catalogue)
    receipt, path = evidence.receipt(ref, qualification.NAMESPACE)
    _version(receipt)
    require(receipt.get('report_content_sha256') == canonical_hash(
        {k: v for k, v in receipt.items() if k != 'report_content_sha256'}), 'Qualification consistency seal differs')
    require(receipt['status'] == qualification.STATUS and receipt['errors'] == []
        and receipt['complete_group_qualification_accepted'] is True
        and receipt['official_full_panel_replayed'] is True
        and receipt['original_runtime_admission_reauthenticated'] is True, 'Group qualification incomplete')
    for name, count in (('group_id', group['group_id']), ('group_count', 1), ('images', group['images']),
            ('target_cells', 16), ('metric_values', 192), ('AP_replays', 16), ('model_calls', 0)):
        _integer(receipt[name], count)
    _same(receipt['group'], group, 'Qualified original group identity differs')
    _same(receipt['targets'], group['targets'], 'Qualification target panel differs')
    require(receipt['max_images'] is None and receipt['diagnostic_only'] is False
        and all(receipt[k] is False for k in ('successor_admitted', 'formal_execution_admission',
            'scientific_acceptance', 'reuse_accepted', 'independent_confirmation', 'formal_metrics_eligible')),
        'Qualification exceeds its original full-group scope')
    require(receipt['group_content_sha256'] == canonical_hash(group)
        and receipt['catalogue_content_sha256'] == canonical_hash(catalogue), 'Qualified catalogue identity differs')
    evidence.sources(path.parents[4], receipt['auditor_source_sha256'],
        {'tools/vcsf_final_followup_group_qualification.py', 'tools/vcsf_final_followup_result_replay.py'})
    reports = [evidence.json(receipt[key], expected=(path.parent / name if name else None))
        for key, name in (('collector_report_reference', None),
            ('recollected_inputs_reference', 'recollected_inputs.json'), ('final_inputs_reference', 'final_inputs.json'))]
    semantics = [qualification._semantic(r) for r in reports]
    require(all(canonical_hash(r) == receipt['collector_semantic_sha256'] for r in semantics),
        'Qualified input reports have different semantics')
    final = reports[-1]
    _same(receipt['preserved_ancestor_failures'], final['preserved_ancestor_failures'], 'Ancestor evidence differs')
    _same(final['group'], group, 'Qualified payload belongs to another group')
    for key in ('runtime_reference', 'admission_reference', 'worker_group_reference',
            'worker_request_reference', 'scope_sha256', 'runtime_snapshot_sha256'):
        _same(final[key], receipt[key], 'Qualification input binding differs: ' + key)
    evidence.sources(plain(final['auditor_source_root']), final['auditor_source_sha256'])
    for mapping in (final['metadata_sha256'], final['control_metadata_sha256']):
        require(type(mapping) is dict and mapping, 'Missing qualified input readback inventory')
        require(all(type(name) is str and bool(name) and is_sha(digest)
            for name, digest in mapping.items()), 'Invalid qualified metadata digest inventory')
    inventory = qualification._inventory(final)
    root = plain(final['execution_root'])
    directory = root / 'groups' / ('%06d' % group['group_id'])
    worker = evidence.json(final['worker_group_reference'], expected=directory / 'worker_group.json')
    _integer(worker['group_id'], group['group_id'])
    _integer(worker['images'], group['images'])
    require(worker['status'] == 'generated_and_evaluated_pending_independent_acceptance'
        and worker['diagnostic_only'] is False, 'Qualified original worker group is not complete')
    _same([t['target'] for t in worker['targets']], group['targets'], 'Original worker target panel differs')
    for key in ('runtime_reference', 'admission_reference', 'scope_sha256'):
        _same(worker[key], final[key], 'Original worker binding differs: ' + key)
    _same(worker['input_binding_sha256'], final['projected_inputs']['binding_sha256'], 'Original full projection differs')
    _same(worker['execution_input_binding_sha256'], worker['input_binding_sha256'], 'Original group was partial')
    rows = receipt['replay_references']
    require(type(rows) is list and len(rows) == 16, 'Incomplete qualified replay panel')
    _same([r['target'] for r in rows], group['targets'], 'Qualified target order differs')
    annotation = {k: final['projected_inputs']['annotation'][k] for k in ('file', 'sha256')}
    normalized = []
    for row in rows:
        _exact(row, {'target', 'reference'}, 'Qualified replay row')
        result = evidence.json(row['reference'], expected=path.parent / 'cells' / row['target'] / 'replay.json')
        qualification._validate_replay(result, final, row['target'], inventory, annotation)
        normalized.append(result['normalized_cell'])
    _same(evidence.json(receipt['normalized_cells_reference'], expected=path.parent / 'normalized_cells.json'),
        normalized, 'Saved normalized group differs from verified replays')
    return final


def _ticks(value):
    require(type(value) is str and re.fullmatch(r'[1-9][0-9]*', value), 'Invalid monitor start ticks')
    return int(value)


def _stop(evidence, ref, originals, catalogue):
    stop, path = evidence.receipt(ref, STOP_NAMESPACE)
    _exact(stop, STOP_FIELDS, 'Stop receipt')
    _version(stop)
    require(stop['status'] == 'independently_bound_predecessor_shutdown_capture' and stop['errors'] == [],
        'Missing independent stop identity binding')
    evidence.sources(path.parents[4], stop['auditor_source_sha256'])
    root = plain(stop['execution_root'])
    old_runtime = evidence.json(stop['runtime_reference'])
    old_scope = old_runtime['scope']
    producer = evidence.path(stop['runtime_reference']).parents[4]
    require(root.parent == producer / 'outputs/experiments/vcsf_final_followup_execution',
        'Stopped root namespace differs')
    require(old_scope['devices'] == catalogue['devices'] and len(old_scope['devices']) == 2,
        'Stop receipt does not describe original two-card predecessor')
    # Reuse the original admission consumer without substituting successor scope.
    first = originals[0]
    bound = load_bound_execution_group(evidence.reader, Registry(producer), stop['runtime_reference'],
        stop['admission_reference'], first['group_id'],
        str(root / 'groups' / ('%06d' % first['group_id'])), 'cuda:0', max_images=None)
    _same(bound['catalogue'], catalogue, 'Stopped predecessor used another original catalogue')
    for original in originals:
        require(plain(original['execution_root']) == root, 'Qualified groups span different predecessors')
        for key in ('runtime_reference', 'admission_reference'):
            _same(original[key], stop[key], 'Qualified group is not from stopped predecessor')
    contract = evidence.json(stop['monitor_contract_reference'], historical=True)
    launch_path = evidence.path(stop['monitor_launch_reference'])
    require(launch_path.name == 'launch.json' and launch_path.parent.parent ==
        producer / 'outputs/experiments/vcsf_final_followup_monitor', 'Original monitor namespace differs')
    launch = evidence.json(stop['monitor_launch_reference'], historical=True)
    for key in ('runtime_reference', 'admission_reference', 'execution_output'):
        expected = stop[key] if key != 'execution_output' else str(root)
        _same(contract[key], expected, 'Monitor contract belongs to another predecessor')
        _same(launch[key], expected, 'Monitor launch belongs to another predecessor')
    _same(launch['contract_reference'], stop['monitor_contract_reference'], 'Monitor contract reference differs')
    require(contract['root'] == str(producer) and contract['monitor'] == str(launch_path.parent)
        and contract['devices'] == old_scope['devices'] and contract['scope_sha256'] == old_scope['scope_sha256']
        and launch['phase'] == 'coordinator' and launch['command_identity_verified'] is True
        and launch['observed_process_role'] == 'coordinator', 'Original monitor launch identity incomplete')
    source_ref = launch['launcher_source_reference']
    require(source_ref['sha256'] == launch['monitor_source_sha256'] == contract['monitor_source_sha256'],
        'Original monitor source identity differs')
    evidence.reader.bytes(evidence.path(source_ref), source_ref['sha256'])
    evidence.reader.bytes(launch_path.parent / 'supervisor.py', source_ref['sha256'])
    command = [contract['python_executable'], '-u', str(producer / 'experiments/vcsf_final_followup_experiments.py'),
        '--runtime', stop['runtime_reference']['file'], '--runtime-sha256', stop['runtime_reference']['sha256'],
        '--admission', stop['admission_reference']['file'], '--admission-sha256', stop['admission_reference']['sha256'],
        '--devices'] + old_scope['devices'] + ['--output', str(root)]
    _same(launch['command'], command, 'Monitor launched another predecessor command')
    monitor = dict(pid=_integer(launch['monitor_pid']), start_ticks=_ticks(launch['monitor_start_ticks']))
    coordinator = dict(pid=_integer(launch['coordinator_pid']), start_ticks=_ticks(launch['coordinator_start_ticks']))
    for key, identity in (('monitor_identity', monitor), ('coordinator_identity', coordinator)):
        observed = launch[key]
        require(type(observed) is dict and observed['pid'] == identity['pid']
            and type(observed['pid']) is int and _ticks(observed['start_ticks']) == identity['start_ticks'],
            'Monitor process identity record differs')
    _same(launch['coordinator_identity']['command'], command, 'Observed coordinator command differs')
    require(launch['coordinator_identity']['parent_pid'] == monitor['pid'], 'Coordinator parent differs')
    expected_processes = [monitor, coordinator]
    workers = stop['worker_references']
    require(type(workers) is list and len(workers) == 2, 'Stop proof must bind both predecessor workers')
    request_refs, uuids = [], []
    for slot, item in enumerate(workers):
        _exact(item, {'request_reference', 'state_reference'}, 'Stopped worker evidence')
        request = evidence.json(item['request_reference'], expected=root / 'workers' / str(slot) / 'request.json', historical=True)
        state = evidence.json(item['state_reference'], expected=root / 'workers' / str(slot) / 'execution_state.json', historical=True)
        lane_binding = group_inputs._lane_binding(evidence.reader, root, Registry(producer), old_scope,
            stop['runtime_reference'], stop['admission_reference'], item['request_reference'], old_scope['lanes'][slot][0])
        _same(request['group_ids'], old_scope['lanes'][slot], 'Stopped worker lane differs')
        for key, value in (('runtime_reference', stop['runtime_reference']), ('admission_reference', stop['admission_reference']),
                ('scope_sha256', old_scope['scope_sha256']), ('worker_slot', slot),
                ('physical_device', old_scope['devices'][slot]), ('coordinator_pid', coordinator['pid']),
                ('coordinator_start_ticks', coordinator['start_ticks'])):
            _same(request[key], value, 'Stopped worker request differs: ' + key)
        _same(state['request_reference'], item['request_reference'], 'Worker state request differs')
        gpu = state['gpu_context_registration']
        require(gpu['schema'] == 'vcsf_gpu_context_registration_v1' and gpu['gpu_uuid'] == request['gpu_uuid'],
            'Stopped worker GPU registration differs')
        _integer(gpu['worker_pid'], lane_binding['assigned_worker']['pid'])
        uuids.append(gpu['gpu_uuid'])
        expected_processes.append(dict(pid=_integer(gpu['worker_pid']), start_ticks=_integer(gpu['worker_pid_start_ticks'])))
        request_refs.append(item['request_reference'])
    require(len(set(uuids)) == 2, 'Predecessor workers share a GPU')
    for original in originals:
        require(original['worker_request_reference'] in request_refs, 'Qualified group request is outside stopped workers')
    _capture(evidence, stop, expected_processes, old_scope['devices'])
    return stop


def _capture(evidence, stop, expected_processes, devices):
    """Bind the actual capture output to a sourced invocation and derived IDs."""
    capture = evidence.json(stop['capture_reference'])
    _exact(capture, {'status', 'expected_processes', 'devices', 'observations', 'consistency',
        'reservation_probe_succeeded', 'reservations_held_on_return', 'predecessor_run_authenticated',
        'successor_admitted', 'signals_sent'}, 'Actual capture_shutdown result')
    require(capture['status'] == 'shutdown_captured_pending_run_identity_binding'
        and capture['reservation_probe_succeeded'] is True and capture['reservations_held_on_return'] is False
        and capture['predecessor_run_authenticated'] is False and capture['successor_admitted'] is False,
        'Capture does not match the real producer contract')
    _integer(capture['signals_sent'], 0)
    _same(capture['expected_processes'], expected_processes, 'Captured process set is not the original monitor/coordinator/workers')
    _same(capture['devices'], devices, 'Captured devices differ from predecessor')
    _same(capture['consistency'], validate_stop_observations(expected_processes, capture['observations']),
        'Captured observations failed reconstruction')
    provenance = evidence.json(stop['capture_provenance_reference'])
    _exact(provenance, CAPTURE_PROVENANCE_FIELDS, 'Capture execution provenance')
    _version(provenance)
    require(provenance['status'] == 'source_bound_capture_shutdown_executed'
        and provenance['function'] == 'tools.vcsf_followup_stop_capture.capture_shutdown',
        'Missing actual capture invocation provenance')
    for key in ('expected_processes', 'devices'):
        _same(provenance[key], capture[key], 'Capture invocation arguments differ')
    _same(provenance['capture_reference'], stop['capture_reference'], 'Capture provenance output differs')
    entry = evidence.path(provenance['entry_reference'])
    require(entry.parent.name in ('tools', 'experiments') and entry.suffix == '.py', 'Invalid capture entry')
    source_root = entry.parent.parent
    evidence.sources(source_root, provenance['source_sha256'], CAPTURE_SOURCES | {entry.relative_to(source_root).as_posix()})
    require(provenance['source_sha256'][entry.relative_to(source_root).as_posix()] == provenance['entry_reference']['sha256'],
        'Capture entry digest differs from executed source map')
    observations = capture['observations']
    require(provenance['boot_id'] == observations[0]['boot_id']
        and _integer(provenance['started_monotonic_ns']) <= observations[0]['monotonic_ns']
        < observations[1]['monotonic_ns'] <= _integer(provenance['finished_monotonic_ns']),
        'Capture provenance interval/boot differs')
    return capture


def _diagnostic(evidence, ref, registry, catalogue, scope, runtime_ref, executor_ref, executor_sources):
    proof, path = evidence.receipt(ref, DIAGNOSTIC_NAMESPACE)
    _exact(proof, DIAGNOSTIC_FIELDS, 'Independent four-card diagnostic proof')
    _version(proof)
    require(proof['status'] == 'independently_verified_final_followup_successor_diagnostic'
        and proof['errors'] == [], 'Independent four-card diagnostic did not pass')
    for key, value in (('formal_runtime_reference', runtime_ref), ('formal_scope_sha256', scope['scope_sha256']),
            ('partition_reference', scope['partition_reference']), ('executor_entry_reference', executor_ref),
            ('executor_runtime_sha256', executor_sources), ('devices', scope['devices']),
            ('group_ids', scope['group_ids']), ('target_cells', scope['target_cells'])):
        _same(proof[key], value, 'Diagnostic does not bind formal successor: ' + key)
    _integer(proof['device_count'], 4)
    evidence.sources(path.parents[4], proof['auditor_source_sha256'])
    diagnostic_scope = compile_successor_scope(registry, catalogue, scope['partition'],
        partition_reference=scope['partition_reference'], max_images=1)
    diagnostic_runtime, _ = _runtime(evidence, registry, catalogue, diagnostic_scope, proof['runtime_reference'])
    # Consume the complete diagnostic authority chain. Diagnostic mode requires
    # diagnostic_acceptance=None, so this cannot recurse into another proof.
    evidence.json(proof['admission_reference'])
    diagnostic_admission = read_stored_successor_execution_admission(evidence.reader, registry,
        catalogue, diagnostic_scope, proof['runtime_reference'], proof['admission_reference'])
    _same(diagnostic_admission['scope'], diagnostic_scope, 'Diagnostic admission scope differs')
    _same(diagnostic_admission['runtime_sha256'], executor_sources, 'Diagnostic admission executor differs')
    worker_count = sum(bool(lane) for lane in diagnostic_scope['lanes'])
    require(len(diagnostic_scope['devices']) == len(set(diagnostic_scope['devices'])) == 4
        and len(diagnostic_scope['lanes']) == 4 and 0 < worker_count <= 4,
        'Diagnostic requires four allocated devices and a nonempty remainder')
    terminal = evidence.json(proof['terminal_reference'])
    terminal_path = evidence.path(proof['terminal_reference'])
    require(terminal_path.name == 'completion.json' and terminal_path.parent.parent ==
        registry.root / 'outputs/diagnostics/vcsf_final_followup_execution', 'Diagnostic terminal namespace differs')
    expected_terminal = dict(status='finished_pending_independent_acceptance',
        execution_root=terminal_path.parent.relative_to(registry.root).as_posix(),
        scope_sha256=diagnostic_scope['scope_sha256'], group_ids=scope['group_ids'],
        worker_count=worker_count, requested_device_count=4, diagnostic_only=True,
        independent_result_acceptance=False, scientific_acceptance=False,
        payload_policy=scope['payload_policy'])
    _same(terminal, expected_terminal, 'Diagnostic did not complete the four-card scope')
    seal = evidence.json(proof['evidence_seal_reference'])
    _exact(seal, {'status', 'evidence_sha256'}, 'Diagnostic byte seal')
    require(seal['status'] == 'diagnostic_evidence_snapshot_sealed'
        and type(seal['evidence_sha256']) is dict and seal['evidence_sha256'], 'Missing diagnostic raw evidence seal')
    for name, digest in seal['evidence_sha256'].items():
        evidence.json({'file': name, 'sha256': digest}) if name.endswith('.json') else evidence.reader.bytes(plain(name), digest)
    require(seal['evidence_sha256'].get(str(terminal_path)) == proof['terminal_reference']['sha256'],
        'Diagnostic terminal omitted from independent evidence seal')
    review = evidence.json(proof['independent_review_reference'])
    _exact(review, {'schema_version', 'status', 'errors', 'terminal_reference', 'evidence_seal_reference',
        'executor_entry_reference', 'executor_runtime_sha256', 'verified_checks', 'auditor_source_sha256'},
        'Independent diagnostic review')
    _version(review)
    require(review['status'] == 'independent_successor_four_card_diagnostic_review_pass' and review['errors'] == [],
        'Missing independent diagnostic review')
    for key in ('terminal_reference', 'evidence_seal_reference', 'executor_entry_reference', 'executor_runtime_sha256'):
        _same(review[key], proof[key], 'Diagnostic review binding differs')
    _same(review['verified_checks'], {k: True for k in CHECKS}, 'Diagnostic review checks incomplete')
    review_path = evidence.path(proof['independent_review_reference'])
    require(len(review_path.parents) >= 5 and review_path.parents[2].name == 'audits'
        and review_path.parents[3].name == 'outputs', 'Independent review namespace differs')
    evidence.sources(review_path.parents[4], review['auditor_source_sha256'])
    # Saved worker receipts must demonstrate four distinct contexts, not count=4 alone.
    uuids, pids = [], []
    coordinator_ids = set()
    for slot, lane in enumerate(scope['lanes']):
        if not lane:
            continue
        done_path = terminal_path.parent / 'workers' / str(slot) / 'completion.json'
        require(str(done_path) in seal['evidence_sha256'], 'Diagnostic worker omitted from seal')
        done = evidence.json(dict(file=str(done_path), sha256=seal['evidence_sha256'][str(done_path)]))
        require(done['status'] == 'finished_pending_independent_acceptance' and done['current'] is None,
            'Diagnostic worker not terminal')
        _same(done['group_ids'], lane, 'Diagnostic worker lane differs')
        for key, count in (('worker_slot', slot), ('completed_groups', len(lane)),
                ('completed_evaluations', len(lane) * 16), ('failed_records', 0)):
            _integer(done[key], count)
        request = evidence.json(done['request_reference'], expected=done_path.parent / 'request.json')
        require(seal['evidence_sha256'].get(str(done_path.parent / 'request.json')) ==
            done['request_reference']['sha256'], 'Diagnostic worker request omitted from seal')
        _same(request['admission_reference'], proof['admission_reference'], 'Diagnostic worker admission differs')
        _same(request['runtime_reference'], proof['runtime_reference'], 'Diagnostic worker runtime differs')
        _same(request['scope_sha256'], diagnostic_runtime['scope']['scope_sha256'], 'Diagnostic worker scope differs')
        _same(request['group_ids'], lane, 'Diagnostic request lane differs')
        _same(request['physical_device'], scope['devices'][slot], 'Diagnostic worker device differs')
        _integer(request['worker_slot'], slot)
        coordinator_ids.add((_integer(request['coordinator_pid']), _integer(request['coordinator_start_ticks'])))
        gpu = done['gpu_context_registration']
        require(gpu['schema'] == 'vcsf_gpu_context_registration_v1' and gpu['gpu_uuid'] == request['gpu_uuid']
            and gpu['before_context_pids'] == [] and gpu['registered_context_pids'] == [gpu['nvml_pid']],
            'Diagnostic GPU ownership evidence incomplete')
        uuids.append(gpu['gpu_uuid'])
        pids.append(_integer(gpu['worker_pid']))
        _same([g['group_id'] for g in done['groups']], lane, 'Diagnostic completed group panel differs')
        for row in done['groups']:
            gid = row['group_id']
            group = next(g for g in diagnostic_scope['groups'] if g['group_id'] == gid)
            directory = terminal_path.parent / 'groups' / ('%06d' % gid)
            group_path = directory / 'worker_group.json'
            require(str(group_path) in seal['evidence_sha256'], 'Diagnostic group omitted from seal')
            _same(evidence.json(dict(file=str(group_path), sha256=seal['evidence_sha256'][str(group_path)])),
                row, 'Diagnostic group record differs')
            require(row['status'] == 'generated_and_evaluated_pending_independent_acceptance'
                and row['diagnostic_only'] is True and row['scientific_acceptance'] is False
                and row['independent_result_acceptance'] is False, 'Diagnostic group is not complete')
            _integer(row['images'], 1)
            _same(row['runtime_reference'], proof['runtime_reference'], 'Diagnostic group runtime differs')
            _same(row['admission_reference'], proof['admission_reference'], 'Diagnostic group admission differs')
            _same(row['scope_sha256'], diagnostic_scope['scope_sha256'], 'Diagnostic group scope differs')
            _same([t['target'] for t in row['targets']], group['targets'], 'Diagnostic omitted a target')
            for target in row['targets']:
                target_path = directory / 'evaluations' / target['target'] / 'metrics.json'
                require(seal['evidence_sha256'].get(str(target_path)) == target['record']['sha256'],
                    'Diagnostic target omitted from seal')
                metrics = evidence.json(target['record'], expected=target_path)
                require(metrics['status'] == 'complete' and metrics['failures'] == [], 'Diagnostic target failed')
                _integer(metrics['images'], 1)
                for key in ('dataset', 'split', 'source', 'parameters_sha256'):
                    _same(metrics[key], group[key], 'Diagnostic target identity differs')
                _same(metrics['target'], target['target'], 'Diagnostic target identity differs')
                archive_path = directory / 'evaluations' / target['target'] / 'predictions.json.gz'
                require(evidence.path(target['prediction_archive']) == archive_path
                    and seal['evidence_sha256'].get(str(archive_path)) == target['prediction_archive']['sha256'],
                    'Diagnostic archive omitted from seal')
    require(len(set(uuids)) == len(set(pids)) == worker_count and len(coordinator_ids) == 1,
        'Diagnostic reused a GPU/worker or mixed coordinators')


def read_stored_successor_execution_admission(read, registry, catalogue, scope, runtime_reference, admission_reference):
    """Authenticate stored proof chains independently of the auditor's __main__.

    Runtime/source bytes, C qualifications, stop and diagnostic proof remain
    mandatory. This does not verify loaded executor identity, live ownership or
    current C payload integrity. Diagnostic-mode proof terminates without recursion.
    """
    evidence = _Evidence(read)
    runtime, expected = _runtime(evidence, registry, catalogue, scope, runtime_reference)
    receipt, receipt_path = evidence.receipt(admission_reference, NAMESPACE)
    _exact(receipt, ADMISSION_FIELDS, 'Successor admission')
    _version(receipt)
    diagnostic = expected['diagnostic_only']
    require(receipt['status'] == ('independently_admitted_final_followup_successor_diagnostic' if diagnostic
        else 'independently_admitted_final_followup_successor_formal') and receipt['errors'] == []
        and receipt['diagnostic_execution_admission'] is diagnostic
        and receipt['formal_execution_admission'] is (not diagnostic), 'Wrong successor admission mode')
    for key, value in (('runtime_reference', runtime_reference), ('scope_sha256', expected['scope_sha256']),
            ('partition_reference', expected['partition_reference']), ('input_references', runtime['input_references']),
            ('group_input_bindings', runtime['group_input_bindings']), ('devices', expected['devices']),
            ('predecessor_stop_reference', expected['partition']['predecessor_stop_reference'])):
        _same(receipt[key], value, 'Successor admission binding differs: ' + key)
    _same(receipt['verified_checks'], {k: True for k in CHECKS}, 'Incomplete successor readiness')
    executor = _executor_snapshot(registry)
    _same(receipt['executor_runtime_sha256'], executor, 'Executor snapshot differs')
    executor_ref = dict(file=str(registry.root / ENTRYPOINT), sha256=executor[ENTRYPOINT])
    _same(receipt['executor_entry_reference'], executor_ref, 'Wrong successor executor entry')
    evidence.sources(registry.root, executor)
    evidence.sources(receipt_path.parents[4], receipt['auditor_source_sha256'])
    rows = expected['partition']['completed_groups']
    refs = [dict(group_id=r['group']['group_id'], reference=r['qualification_reference']) for r in rows]
    _same(receipt['completed_group_qualification_references'], refs, 'Admission C differs from partition C')
    originals = [_qualification(evidence, r['qualification_reference'], r['group'], catalogue) for r in rows]
    _stop(evidence, receipt['predecessor_stop_reference'], originals, catalogue)
    if diagnostic:
        require(receipt['diagnostic_acceptance'] is None, 'Diagnostic admission cannot claim its own completed proof')
    else:
        _diagnostic(evidence, receipt['diagnostic_acceptance'], registry, catalogue, expected,
            runtime_reference, executor_ref, executor)
    evidence.unchanged()
    _same(_executor_snapshot(registry), executor, 'Executor changed during admission consumption')
    return dict(status='successor_admission_receipt_bound_pending_worker_ownership', scope=expected,
        input_references=deepcopy(runtime['input_references']), group_input_bindings=deepcopy(runtime['group_input_bindings']),
        runtime_sha256=deepcopy(executor), preparation_runtime_sha256=deepcopy(runtime['runtime_sha256']),
        admission_reference=deepcopy(admission_reference),
        completed_group_references_authenticated=True,
        qualification_references_authenticated=not any(r.get('metric_replay_deferred', False) for r in originals),
        metric_replay_deferred=any(r.get('metric_replay_deferred', False) for r in originals),
        completed_groups_metric_qualified=not any(r.get('metric_replay_deferred', False) for r in originals),
        loaded_executor_identity_verified=False,
        completed_payload_current_integrity_verified=False,
        predecessor_stop_evidence_authenticated=True, worker_ownership_verified=False,
        current_device_availability_verified=False, scientific_acceptance=False, model_calls=0, AP_replays=0)


def read_successor_execution_admission(read, registry, catalogue, scope, runtime_reference, admission_reference):
    """Consume stored authority with loaded executor checks on both sides."""
    executor = _executor_snapshot(registry)
    _verify_executor(registry, executor)
    result = read_stored_successor_execution_admission(read, registry, catalogue, scope,
        runtime_reference, admission_reference)
    _same(result['runtime_sha256'], executor, 'Executor changed across stored admission validation')
    _verify_executor(registry, executor)
    result['loaded_executor_identity_verified'] = True
    return result
