"""Publish operational whole-group exclusion, NOT metric/result qualification.

The pinned owner policy defers replay until the entire follow-up queue finishes.
Only original admission, task/config/seed identities and complete stored JSON
metadata are reconstructed. PNG bytes are not read; prediction archives are
only checked for existence and declared size. Their declared hashes are bound,
NOT verified. No tree scan, model call, AP replay or stop observation occurs.
The successor consumer separately authenticates predecessor stop evidence.

Run this source-bound tool in a separate ODA issuer clone with --request FILE
--request-sha256 SHA --output FRESH_LEAF. Request fields are REQUEST_FIELDS;
every reference is exactly file/sha256, and planned_stop_failure_references is
the explicit list of preserved root/selected-worker failure records (or []).
No full coordinator completion is required. Publication failure is preserved.
"""
import hashlib
from pathlib import Path
import sys

if __name__ == '__main__' and '_ENTRY_SOURCE_SHA256' not in globals():
    _path = Path(__file__).resolve()
    _raw = _path.read_bytes()
    _ENTRY_SOURCE_SHA256 = hashlib.sha256(_raw).hexdigest()
    __file__ = str(_path)
    exec(compile(_raw, str(_path), 'exec', dont_inherit=True), globals())
    raise RuntimeError('Source-bound publisher returned without terminal status')

ROOT = Path(__file__).resolve().parents[1]
ENTRY = 'tools/vcsf_final_followup_completed_group.py'
if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    _bootstrap = ROOT / 'experiments/vcsf_single_source_ablation_executor_smoke.py'
    _raw = _bootstrap.read_bytes()
    _BOOTSTRAP_SOURCE_SHA256 = hashlib.sha256(_raw).hexdigest()
    exec(compile(_raw, str(_bootstrap), 'exec', dont_inherit=True),
        {'__file__': str(_bootstrap), '__name__': '_completed_group_bootstrap'})
    sys.path.insert(0, str(ROOT / 'tools'))

import argparse
from copy import deepcopy
import json

from lgp.registry import Registry
from lgp.runners import vcsf_final_followup_successor_admission as consumer
from tools import vcsf_final_followup_group_inputs as inputs
from tools.audit_vcsf_cpu_analysis import Evidence, check, plain, same, sha, is_sha
from tools.audit_vcsf_operator_readiness import _write
from tools.vcsf_final_followup_record_binding import bind_target_to_generation
from tools.vcsf_final_followup_terminal_audit import _group

NAMESPACE = 'vcsf_final_followup_completed_group'
STATUS = 'complete_execution_metadata_authenticated_for_scheduling_exclusion'
POLICY = 'docs/research/vcsf-followup-deferred-replay-20260917.json'
POLICY_SHA256 = 'a1c48475375b77c2d60a050b1dbaad426c4f956f4449e96426c681776b4c918c'
REQUEST_FIELDS = {'runtime_reference', 'admission_reference', 'worker_group_reference',
    'worker_request_reference', 'policy_reference', 'planned_stop_failure_references'}


def _flags():
    return dict(AP_replays=0, model_calls=0, metric_replay_deferred=True,
        operational_task_exclusion_only=True, complete_group_qualification_accepted=False,
        independent_result_acceptance=False, scientific_acceptance=False,
        qualified_metrics_verified=False, formal_metrics_eligible=False,
        completed_payload_current_integrity_verified=False, prediction_archive_verified=False,
        generated_pixels_verified=False, reuse_accepted=False, formal_execution_admission=False,
        predecessor_stop_verified=False)


def _policy(evidence, reference, issuer):
    same(reference, dict(file=str(issuer / POLICY), sha256=POLICY_SHA256),
        'Require the exact owner deferred-replay policy path and hash')
    return evidence.json(reference)


def _archive(read, leaf, record, reference):
    """Bind declared digests and size without opening the archive."""
    path = inputs._reference_path(reference)
    artifact = record.get('predictions_artifact')
    check(type(artifact) is dict, 'Missing prediction artifact declaration')
    check(path == leaf / 'predictions.json.gz' and path.is_file(), 'Missing or foreign archive')
    for key, value in (('format', 'gzip'), ('archive_file', 'predictions.json.gz'),
            ('uncompressed_file', 'predictions.json')):
        check(artifact.get(key) == value, 'Archive declaration differs: ' + key)
    for key in ('archive_bytes', 'uncompressed_bytes', 'prediction_records'):
        check(type(artifact.get(key)) is int and artifact[key] >= 0, 'Invalid archive count')
    check(is_sha(artifact.get('archive_sha256')) and is_sha(artifact.get('uncompressed_sha256')),
        'Invalid declared archive digest')
    same(reference['sha256'], artifact['archive_sha256'], 'Worker archive declaration differs')
    if 'predictions_sha256' in record:
        same(record['predictions_sha256'], artifact['uncompressed_sha256'], 'Prediction digest differs')
    check(path.stat().st_size == artifact['archive_bytes'], 'Archive size differs')
    sidecar = leaf / 'predictions_artifact.json'
    sidecar_reference = None
    if sidecar.exists():
        sidecar_reference = dict(file=str(sidecar), sha256=sha(sidecar))
        same(inputs.original._reference(read, sidecar_reference), artifact, 'Archive sidecar differs')
    return dict(reference=deepcopy(reference), declared_artifact=deepcopy(artifact),
        sidecar_reference=sidecar_reference, existence_and_size_verified=True,
        archive_digest_verified=False, archive_content_verified=False)


def _collect(read, request):
    consumer._exact(request, REQUEST_FIELDS, 'Completed group request')
    runtime_ref, admission_ref = request['runtime_reference'], request['admission_reference']
    producer = inputs._reference_path(runtime_ref).parents[4]
    registry = Registry(producer)
    path = inputs._reference_path(request['worker_group_reference'])
    directory, root = path.parent, path.parent.parent.parent
    check(path.name == 'worker_group.json' and directory.parent.name == 'groups'
        and root.parent == producer / 'outputs/experiments/vcsf_final_followup_execution',
        'Group escaped original execution namespace')
    worker = inputs.original._reference(read, request['worker_group_reference'])
    gid = worker.get('group_id')
    check(type(gid) is int and 2 <= gid <= 16 and directory.name == '%06d' % gid,
        'Invalid complete original group identity')
    bound = inputs.original.load_bound_execution_group(read, registry, runtime_ref,
        admission_ref, gid, str(directory), 'cuda:0', max_images=None)
    scope, group, prepared = bound['admission']['scope'], bound['group'], bound['prepared']
    inputs.original._formal_scope(scope, list(registry.target_ids()))
    same(scope, inputs.original._reference(read, runtime_ref)['scope'], 'Original runtime scope differs')
    same(group, next(g for g in scope['groups'] if g['group_id'] == gid), 'Original group differs')
    request_ref = request['worker_request_reference']
    binding = inputs._lane_binding(read, root, registry, scope, runtime_ref, admission_ref, request_ref, gid)
    prefix = inputs._completed_prefix(read, request_ref, binding['worker_request']['group_ids'], gid, worker)
    failures = inputs._ancestor_failures(read, root, inputs._reference_path(request_ref).parent,
        request['planned_stop_failure_references'])
    healthy = [directory.parent, directory, directory / 'attack', directory / 'evaluations', root / 'workers']
    healthy.extend(directory / 'evaluations' / target for target in group['targets'])
    for leaf in healthy:
        inputs.original._healthy(leaf)
    _group(read, root, group, worker, runtime_ref, admission_ref, request_ref, scope)
    projected = prepared['projected_inputs']
    same(worker['input_binding_sha256'], projected['binding_sha256'], 'Original input projection differs')
    check(type(projected['images']) is int and projected['images'] == group['images']
        and len(projected['seeded_clean_rows']) == group['images'], 'Incomplete projected inputs')
    same(projected['targets'], group['targets'], 'Projected target panel differs')
    inputs.original._projection_metadata(read, projected)
    run = inputs.original._reference(read, worker['generation'], directory / 'attack/run.json')
    check(run.get('manifest') == 'manifest.jsonl' and run.get('annotation') == 'annotations.json'
        and run.get('image_root') == '.', 'Generation metadata paths differ')
    manifest_ref = dict(file=str(directory / 'attack/manifest.jsonl'), sha256=run['manifest_sha256'])
    raw = read.bytes(plain(manifest_ref['file']), manifest_ref['sha256'], inputs.original.METADATA_BYTES)
    rows = [inputs.original._json(line) for line in raw.splitlines()]
    same(len(rows), group['images'], 'Incomplete manifest')
    for row, clean in zip(rows, projected['seeded_clean_rows']):
        check(row.get('output_file') == 'images/%012d.png' % clean['image_id']
            and is_sha(row.get('output_sha256')) and type(row.get('output_bytes')) is int
            and row['output_bytes'] > 0, 'Invalid declared PNG identity')
        if 'source_file' in row:
            same(row['source_file'], clean['input']['file'], 'Manifest source identity differs')
    annotation_ref = dict(file=str(directory / 'attack/annotations.json'), sha256=run['annotation_sha256'])
    generated = inputs.original._annotation(read, annotation_ref, projected['ordered_image_ids'])
    canonical = inputs.original._annotation(read, projected['annotation'], projected['ordered_image_ids'])
    same(generated['categories'], canonical['categories'], 'Annotation categories differ')
    same(generated['annotations'], canonical['annotations'], 'Annotation ground truth differs')
    same(sorted(generated['images'], key=lambda r: r['id']),
        [dict(r, file_name='images/%012d.png' % r['id'])
            for r in sorted(canonical['images'], key=lambda r: r['id'])], 'Generated image identities differ')
    cells = []
    parameters = prepared['attack']['parameter_overrides']
    for target in worker['targets']:
        name = target['target']
        leaf = directory / 'evaluations' / name
        record = inputs.original._reference(read, target['record'], leaf / 'metrics.json')
        linked = bind_target_to_generation(read, worker['generation'], target['record'],
            projected, name, inputs.original.ALIAS, parameters, parameters['iterations'])
        same(linked['generation_metadata'], worker['generation_metadata'], 'Generation summary differs')
        same(linked, target['metadata_check'], 'Target metadata summary differs')
        check(set(record['metrics']) == set(inputs.original.COCO_BBOX_METRICS), 'Incomplete metric record')
        same(record.get('annotation'), str(directory / 'attack/annotations.json'), 'Target annotation differs')
        if 'annotation_sha256' in record:
            same(record['annotation_sha256'], annotation_ref['sha256'], 'Target annotation hash differs')
        cells.append(dict(target=name, record_reference=deepcopy(target['record']),
            archive=_archive(read, leaf, record, target['prediction_archive'])))
    same([c['target'] for c in cells], group['targets'], 'Incomplete or duplicate targets')
    check(len(cells) == 16, 'Require all sixteen targets')
    read.unchanged()
    for leaf in healthy:
        inputs.original._healthy(leaf)
    for cell in cells:
        archive = cell['archive']
        check(plain(archive['reference']['file']).stat().st_size == archive['declared_artifact']['archive_bytes'],
            'Archive size changed during metadata collection')
    return dict(schema_version=1, status=STATUS, group_id=gid, group=deepcopy(group),
        catalogue_content_sha256=consumer.canonical_hash(bound['catalogue']), execution_root=str(root),
        runtime_reference=deepcopy(runtime_ref), admission_reference=deepcopy(admission_ref),
        worker_group_reference=deepcopy(request['worker_group_reference']), worker_request_reference=deepcopy(request_ref),
        scope_sha256=scope['scope_sha256'], input_binding_sha256=projected['binding_sha256'],
        original_lane_binding=binding, completed_prefix_reference=prefix,
        generation_reference=deepcopy(worker['generation']), manifest_reference=manifest_ref,
        annotation_reference=annotation_ref, cells=cells, preserved_ancestor_failures=failures,
        images=group['images'], target_cells=16, max_images=None, diagnostic_only=False,
        full_coordinator_completion_required=False, original_input_admission_authenticated=True, **_flags())


def read_completed_group(read, reference, group, catalogue):
    """Authenticate operational evidence, returning original bindings for _stop.

    No claim is made about current PNG/archive integrity or stored metric truth.
    qualification_reference in the partition is a compatibility name only.
    """
    evidence = consumer._Evidence(read)
    receipt, path = evidence.receipt(reference, NAMESPACE)
    consumer._exact(receipt, {'schema_version', 'status', 'request_reference', 'summary',
        'auditor_source_sha256'} | set(_flags()), 'Operational completed group receipt')
    consumer._version(receipt)
    check(receipt['status'] == STATUS, 'Not an operational completed-group receipt')
    same({k: receipt[k] for k in _flags()}, _flags(), 'Operational receipt exceeds deferred policy')
    issuer = path.parents[4]
    evidence.sources(issuer, receipt['auditor_source_sha256'], {ENTRY,
        'tools/vcsf_final_followup_group_inputs.py', 'tools/vcsf_final_followup_record_binding.py',
        'tools/vcsf_final_followup_terminal_audit.py', 'src/lgp/runners/vcsf_final_followup_records.py'})
    request = evidence.json(receipt['request_reference'], expected=path.parent / 'request.json')
    _policy(evidence, request['policy_reference'], issuer)
    summary = _collect(read, request)
    same(summary, receipt['summary'], 'Operational receipt differs from actual metadata reconstruction')
    same(summary['group'], group, 'Operational receipt belongs to another group')
    same(summary['catalogue_content_sha256'], consumer.canonical_hash(catalogue), 'Original catalogue differs')
    evidence.unchanged()
    return summary


def _sources(evidence):
    from tools.run_vcsf_core_points import source_binding, BOOTSTRAP
    from lgp.runners.vcsf_ablation_execution_contract import _verify_process_binding
    main = sys.modules['__main__']
    check(getattr(main, '__file__', None) == str(ROOT / ENTRY), 'Use the source-bound publisher entry')
    _verify_process_binding(Registry(ROOT))
    loaded = source_binding(ENTRY, getattr(main, '_ENTRY_SOURCE_SHA256', None),
        getattr(main, '_BOOTSTRAP_SOURCE_SHA256', None))
    sources = consumer.runtime_snapshot(ROOT, entrypoint=ENTRY)
    sources[BOOTSTRAP] = sha(ROOT / BOOTSTRAP)
    check(all(sources.get(k) == v for k, v in loaded.items()), 'Loaded publisher differs from snapshot')
    evidence.sources(ROOT, sources)
    return sources


def publish(request_reference, *, output):
    check(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        'Requires authorized server ODA')
    output = plain(Path(output).absolute())
    check(output.parent == ROOT / 'outputs/audits' / NAMESPACE and not output.exists(), 'Require a fresh issuer leaf')
    read = Evidence()
    evidence = consumer._Evidence(read)
    sources = _sources(evidence)
    request = evidence.json(request_reference)
    consumer._exact(request, REQUEST_FIELDS, 'Completed group request')
    _policy(evidence, request['policy_reference'], ROOT)
    producer = inputs._reference_path(request['runtime_reference']).parents[4]
    check(ROOT != producer and ROOT not in producer.parents and producer not in ROOT.parents,
        'Use a separate nonnested issuer clone')
    output.mkdir(parents=True, exist_ok=False)
    try:
        request_ref = _write(output / 'request.json', request)
        summary = _collect(read, request)
        same(_sources(evidence), sources, 'Publisher source changed')
        evidence.unchanged()
        receipt = dict(schema_version=1, status=STATUS, request_reference=request_ref,
            summary=summary, auditor_source_sha256=sources, **_flags())
        reference = _write(output / 'receipt.json', receipt)
        # Read back using the actual original catalogue, never a self-hash as admission.
        bound = inputs.original.load_bound_execution_group(read, Registry(producer),
            request['runtime_reference'], request['admission_reference'], summary['group_id'],
            str(plain(summary['execution_root']) / 'groups' / ('%06d' % summary['group_id'])),
            'cuda:0', max_images=None)
        read_completed_group(read, reference, summary['group'], bound['catalogue'])
        same(_sources(evidence), sources, 'Publisher changed after readback')
        evidence.unchanged()
        return dict(status=STATUS, receipt_reference=reference, **_flags())
    except BaseException as error:
        _write(output / 'failure.json', dict(status='completed_group_publication_failed',
            error_type=type(error).__name__, error=str(error), **_flags()))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(publish(dict(file=str(args.request.absolute()), sha256=args.request_sha256),
            output=args.output), sort_keys=True))
        return 0
    except Exception as error:
        print(json.dumps(dict(status='completed_group_publication_failed', error=str(error))), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
