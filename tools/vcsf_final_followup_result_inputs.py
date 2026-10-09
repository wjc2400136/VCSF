"""Read-only full follow-up inputs, before official replay or result acceptance.

Public API: collect_formal_result_inputs(runtime_reference, admission_reference,
completion_reference, *, registry=None). References contain exactly file/sha256.
The completion reference selects an already terminal immutable execution root.

Schema 1 returns groups (complete projections and generation references), cells
(group-major/canonical-target-major, 240 records), payload_files (absolute file
to SHA/bytes), raw_inventory (run-relative path to SHA), metadata_sha256,
terminal_chain and report_content_sha256. Cell
input_binding contains everything needed by a separate official replay consumer.
No predictions are retained, no scratch is created and no file is written.
FormalResultInputsError.failure_report is a sealed failure snapshot for the
outer publisher to preserve in its own fresh output tree; never publish it as
successful inputs. Partial groups/cells cannot qualify a subset or a union.

The existing streaming parser limits each compressed/decompressed cell to 8 GiB
and each JSON record to 1 MiB. Metadata is capped at 512 MiB. PNG limits remain
those of audit_generated_png. These are explicit resource limits, not detector
limits; no predictions are truncated, filtered or reordered.
"""
from copy import deepcopy
import gzip
import json
import math
import os
from pathlib import Path

from tools.audit_vcsf_cpu_analysis import Evidence, check, is_sha, plain, same, sha, signature
from tools.vcsf_final_followup_bound_dispatch import (
    load_bound_execution_group, prepare_bound_group_calls,
)
from tools.vcsf_final_followup_record_binding import audit_generated_png
from tools.vcsf_final_followup_terminal_audit import audit_terminal_chain
from lgp.metrics import COCO_BBOX_METRICS
from lgp.registry import Registry
from lgp.reporting.qualitative_inputs import (
    _ArchiveReader, _prediction_records, MAX_ARCHIVE_BYTES, MAX_DECOMPRESSED_BYTES,
    MAX_RECORD_BYTES, STREAM_CHUNK_BYTES,
)
from lgp.runners.vcsf_final_followup_dispatch import ALIAS
from lgp.runners.vcsf_final_followup_records import (
    inspect_generation_metadata, inspect_target_record,
)
from lgp.runners.vcsf_research_plan import canonical_hash


METADATA_BYTES = 512 << 20
STATUS = 'formal_followup_result_inputs_verified_pending_replay'


def _flags():
    return dict(model_calls=0, AP_replays=0, formal_execution_admission=False,
        independent_result_acceptance=False, scientific_acceptance=False,
        formal_metrics_verified=False, formal_metrics_eligible=False,
        reuse_accepted=False, union_accepted=False, independent_confirmation=False,
        current_gpu_ownership_verified=False)


def _seal(value):
    value['report_content_sha256'] = canonical_hash(value)
    return value


class FormalResultInputsError(RuntimeError):
    """The caller may persist failure_report; no partial acceptance is issued."""

    def __init__(self, failure_report):
        self.failure_report = failure_report
        super().__init__(failure_report['error'])


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            check(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)

    def finite(value):
        result = float(value)
        check(math.isfinite(result), 'Non-finite JSON number')
        return result

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid, parse_float=finite)


def _healthy(directory):
    marker = plain(directory) / 'failure.json'
    check(not marker.exists() and not marker.is_symlink(), 'Failed evidence publication')


class _Metadata(Evidence):
    def __init__(self):
        super().__init__()
        self.directories = set()

    def bytes(self, path, expected, cap=None):
        path = plain(path)
        if path.suffix in ('.json', '.jsonl'):
            self.directories.add(path.parent)
            _healthy(path.parent)
        return super().bytes(path, expected, cap)

    def json(self, path, expected):
        return _json(self.bytes(path, expected, METADATA_BYTES))

    def historical_bytes(self, path, expected, cap=None):
        """Read protocol-designated history without accepting its failed parent."""
        return super().bytes(plain(path), expected, cap)

    def unchanged(self):
        for directory in self.directories:
            _healthy(directory)
        super().unchanged()
        for directory in self.directories:
            _healthy(directory)


def _reference(read, reference, expected_path=None):
    check(type(reference) is dict and set(reference) == {'file', 'sha256'}
        and type(reference['file']) is str and is_sha(reference['sha256']), 'Invalid reference')
    path = plain(reference['file'])
    check(expected_path is None or path == expected_path, 'Reference escaped expected leaf')
    return read.json(path, reference['sha256'])


def _snapshot(read, path):
    path = plain(path)
    reference = dict(file=str(path), sha256=sha(path))
    return reference, _reference(read, reference)


class _Payloads:
    """Keep digests/stat identities, never payload buffers or prediction lists."""

    def __init__(self):
        self.files = {}
        self.signatures = {}

    def add(self, path, digest, size, before):
        path = plain(path)
        name = str(path)
        check(is_sha(digest) and type(size) is int and size >= 0
            and signature(path.stat()) == before and before[2] == size,
            'Payload changed during consumption')
        reference = dict(sha256=digest, bytes=size)
        check(name not in self.files or (self.files[name] == reference
            and self.signatures[name] == before), 'Conflicting payload identity')
        self.files[name], self.signatures[name] = reference, before

    def absorb(self, read):
        for name, digest in read.checked.items():
            before = read.signatures[name]
            self.add(name, digest, before[2], before)

    def unchanged(self, names=None, rehash=False):
        # Helpers rehash their own bounded payloads; the final sweep detects
        # ordinary replacement/in-place writes without an O(images^2) audit.
        for name in (self.signatures if names is None else names):
            before = self.signatures[name]
            check(signature(plain(name).stat()) == before, 'Payload changed after verification')
            if rehash:
                check(sha(name) == self.files[name]['sha256']
                    and signature(plain(name).stat()) == before, 'Payload changed during group readback')


def _tree(root):
    members = []
    for path in root.rglob('*'):
        plain(path)
        check(path.name != 'failure.json', 'Execution contains failure evidence')
        if path.is_file():
            members.append(path.relative_to(root).as_posix())
        else:
            check(path.is_dir(), 'Execution contains a special file')
    return tuple(sorted(members))


def _formal_scope(scope, targets):
    check(scope['diagnostic_only'] is False and scope['max_images'] is None
        and scope['group_ids'] == list(range(2, 17))
        and scope['images'] == 74712 and scope['target_cells'] == 240
        and scope['payload_policy'] == 'keep_all_until_independent_acceptance',
        'Require the complete formal fifteen-group scope')
    check(len(targets) == len(set(targets)) == 16, 'Require canonical sixteen targets')
    same([g['group_id'] for g in scope['groups']], scope['group_ids'], 'Group order differs')
    for group in scope['groups']:
        check(group['dataset'] in ('coco', 'voc') and group['split'] == 'val'
            and type(group['images']) is int
            and group['images'] == (5000 if group['dataset'] == 'coco' else 4952),
            'Require full registered dataset image count')
        same(group['targets'], targets, 'Canonical target order differs')


def _authenticate(read, runtime_reference, admission_reference, completion_reference, registry):
    runtime = _reference(read, runtime_reference)
    runtime_path = plain(runtime_reference['file'])
    check(len(runtime_path.parents) >= 5, 'Runtime namespace is incomplete')
    producer = runtime_path.parents[4]
    registry = registry if registry is not None else Registry(producer)
    check(plain(registry.root) == producer, 'Registry must belong to the bound producer')
    scope = runtime['scope']
    _formal_scope(scope, list(registry.target_ids()))
    root = plain(completion_reference['file']).parent
    _reference(read, completion_reference, root / 'completion.json')
    first = scope['group_ids'][0]
    bound = load_bound_execution_group(read, registry, runtime_reference, admission_reference,
        first, root / 'groups' / ('%06d' % first), 'cuda:0', max_images=None)
    same(bound['admission']['scope'], scope, 'Authenticated scope differs')
    terminal = audit_terminal_chain(read, root, runtime_reference, admission_reference, scope)
    return registry, runtime, root, bound, terminal


def _annotation(read, reference, ids):
    value = _reference(read, {k: reference[k] for k in ('file', 'sha256')})
    if 'bytes' in reference:
        check(type(reference['bytes']) is int
            and plain(reference['file']).stat().st_size == reference['bytes'], 'Annotation size differs')
    check(type(value) is dict and all(type(value.get(k)) is list
        for k in ('images', 'categories', 'annotations')), 'Incomplete annotation')
    for key in ('images', 'categories', 'annotations'):
        numbers = [r['id'] for r in value[key]]
        check(all(type(n) is int for n in numbers) and len(numbers) == len(set(numbers)),
            'Duplicate or invalid annotation IDs')
    same(sorted(r['id'] for r in value['images']), ids, 'Canonical full image IDs differ')
    categories = {r['id'] for r in value['categories']}
    check(categories and all(i > 0 for i in categories), 'Invalid annotation categories')
    image_set = set(ids)
    check(all(r['image_id'] in image_set and r['category_id'] in categories
        for r in value['annotations']), 'Annotation contains foreign image/category')
    return value


def _projection_metadata(read, projected):
    roles = projected['model_configs']
    same([(row['role'], row['model']) for row in roles],
        [('source', projected['source'])] + [('target', t) for t in projected['targets']],
        'Projection role configuration order differs')
    for row in roles:
        for key in ('resolved_config', 'effective_config'):
            reference = row[key]
            check(type(reference) is dict and set(reference) == {'file', 'sha256', 'bytes'}
                and type(reference['bytes']) is int and 0 < reference['bytes'] <= METADATA_BYTES,
                'Invalid role configuration reference')
            raw = read.bytes(plain(reference['file']), reference['sha256'], reference['bytes'])
            check(len(raw) == reference['bytes'], 'Role configuration byte count differs')


def _generation(read, payloads, directory, worker, projected, parameters, context=None):
    reference = worker['generation']
    run = _reference(read, reference, directory / 'attack/run.json')
    attack = directory / 'attack'
    check(run.get('manifest') == 'manifest.jsonl' and run.get('annotation') == 'annotations.json'
        and run.get('image_root') == '.', 'Generation paths differ')
    manifest_ref = dict(file=str(attack / 'manifest.jsonl'), sha256=run['manifest_sha256'])
    raw = read.bytes(attack / 'manifest.jsonl', run['manifest_sha256'], METADATA_BYTES)
    rows = [_json(line) for line in raw.splitlines()]
    summary = inspect_generation_metadata(run, raw, projected, ALIAS, parameters, parameters['iterations'])
    same(summary, worker['generation_metadata'], 'Worker generation summary differs')
    canonical = _annotation(read, projected['annotation'], projected['ordered_image_ids'])
    annotation_ref = dict(file=str(attack / 'annotations.json'), sha256=run['annotation_sha256'])
    generated = _annotation(read, annotation_ref, projected['ordered_image_ids'])
    same(generated['categories'], canonical['categories'], 'Generated categories differ')
    same(generated['annotations'], canonical['annotations'], 'Generated ground truth differs')
    originals = {r['id']: r for r in canonical['images']}
    expected_images = [dict(originals[i], file_name='images/%012d.png' % i)
        for i in projected['ordered_image_ids']]
    same(sorted(generated['images'], key=lambda r: r['id']), expected_images,
        'Generated image metadata differs')
    clean_rows = projected['seeded_clean_rows']
    check(len(rows) == len(clean_rows) == projected['images'], 'Full manifest coverage differs')
    max_delta = 0
    for row, clean in zip(rows, clean_rows):
        image_id = clean['image_id']
        if context is not None:
            context['image_id'] = image_id
        image = originals[image_id]
        pixel_read = Evidence()
        pixel = audit_generated_png(pixel_read, attack, row, clean['input'],
            'images/%012d.png' % image_id, image_id, [image['height'], image['width'], 3])
        payloads.absorb(pixel_read)
        max_delta = max(max_delta, pixel['observed_linf_uint8'])
    expected_pngs = {'images/%012d.png' % i for i in projected['ordered_image_ids']}
    check({p.relative_to(attack).as_posix() for p in attack.rglob('*.png')} == expected_pngs,
        'Unexpected or missing group PNGs')
    if context is not None:
        context['image_id'] = None
    return dict(generation_reference=deepcopy(reference), manifest_reference=manifest_ref,
        annotation_reference=annotation_ref, canonical_annotation_reference=deepcopy(projected['annotation']),
        generation_metadata=summary, png_count=len(rows), maximum_linf_uint8=max_delta,
        category_ids=sorted(r['id'] for r in canonical['categories']))


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _prediction_payload(read, payloads, leaf, record, worker_archive, image_ids, category_ids):
    sidecar_ref, artifact = _snapshot(read, leaf / 'predictions_artifact.json')
    same(artifact, record['predictions_artifact'], 'Sidecar differs from metrics')
    check(type(artifact.get('schema_version')) is int and artifact['schema_version'] == 1
        and artifact.get('status') == 'verified_lossless_archive' and artifact.get('format') == 'gzip'
        and artifact.get('archive_file') == 'predictions.json.gz'
        and artifact.get('uncompressed_file') == 'predictions.json'
        and artifact.get('fields') == ['image_id', 'category_id', 'bbox', 'score'], 'Archive schema differs')
    for key in ('archive_bytes', 'uncompressed_bytes', 'prediction_records', 'image_ids_with_detections'):
        check(type(artifact.get(key)) is int and artifact[key] >= 0, 'Invalid archive count: ' + key)
    check(0 < artifact['archive_bytes'] <= MAX_ARCHIVE_BYTES
        and 2 <= artifact['uncompressed_bytes'] <= MAX_DECOMPRESSED_BYTES, 'Archive resource cap exceeded')
    check(is_sha(artifact.get('archive_sha256')) and is_sha(artifact.get('uncompressed_sha256'))
        and record.get('predictions_sha256') == artifact['uncompressed_sha256']
        and type(record.get('detections')) is int and record['detections'] == artifact['prediction_records'],
        'Archive provenance/count differs')
    raw_path = leaf / 'predictions.json'
    check(not raw_path.exists() and not raw_path.is_symlink(), 'Unexpected raw prediction side file')
    path = plain(leaf / 'predictions.json.gz')
    archive_ref = dict(file=str(path), sha256=artifact['archive_sha256'])
    same(worker_archive, archive_ref, 'Worker archive reference differs')
    before = signature(path.stat())
    check(before[2] == artifact['archive_bytes'], 'Archive byte count differs')
    images, categories, detected = set(image_ids), set(category_ids), set()
    count = 0
    with path.open('rb') as handle:
        check(signature(os.fstat(handle.fileno())) == before, 'Archive changed before reading')
        compressed = _ArchiveReader(handle, artifact['archive_bytes'], artifact['archive_sha256'])
        with gzip.GzipFile(fileobj=compressed, mode='rb') as stream:
            for item in _prediction_records(stream, artifact['uncompressed_bytes'], artifact['uncompressed_sha256']):
                check(type(item) is dict and set(item) == {'image_id', 'category_id', 'bbox', 'score'},
                    'Unexpected prediction fields')
                check(type(item['image_id']) is int and item['image_id'] in images
                    and type(item['category_id']) is int and item['category_id'] in categories,
                    'Prediction image/category outside full scope')
                box, score = item['bbox'], item['score']
                check(type(box) is list and len(box) == 4 and all(_number(v) for v in box)
                    and box[2] >= 0 and box[3] >= 0 and _number(score) and 0 <= score <= 1,
                    'Invalid prediction geometry or confidence')
                check(all(_number(v) for v in (box[0] + box[2], box[1] + box[3], box[2] * box[3])),
                    'Prediction derived geometry overflows')
                count += 1
                check(count <= artifact['prediction_records'], 'Prediction record count overflow')
                detected.add(item['image_id'])
        compressed.verify()
        check(signature(os.fstat(handle.fileno())) == before, 'Archive changed while reading')
    check(count == artifact['prediction_records'] and len(detected) == artifact['image_ids_with_detections'],
        'Prediction record/detected-image counts differ')
    payloads.add(path, artifact['archive_sha256'], artifact['archive_bytes'], before)
    return dict(sidecar_reference=sidecar_ref, archive_reference=archive_ref,
        archive_bytes=artifact['archive_bytes'], uncompressed_bytes=artifact['uncompressed_bytes'],
        uncompressed_sha256=artifact['uncompressed_sha256'], prediction_records=count,
        image_ids_with_detections=len(detected), images_without_detections=len(images) - len(detected),
        full_prediction_array_validated=True, gzip_crc_verified=True)


def _cell(read, payloads, directory, group, projected, generated, target, parameters):
    name = target['target']
    leaf = directory / 'evaluations' / name
    record = _reference(read, target['record'], leaf / 'metrics.json')
    target_summary = inspect_target_record(record, projected, name, ALIAS)
    check(set(record['metrics']) == set(COCO_BBOX_METRICS), 'Require exactly twelve raw metrics')
    check(type(record.get('adversarial_run')) is str
        and plain(record['adversarial_run']) == directory / 'attack'
        and type(record.get('annotation')) is str
        and plain(record['annotation']) == plain(generated['annotation_reference']['file']),
        'Target generation or annotation path differs')
    if 'annotation_sha256' in record:
        same(record['annotation_sha256'], generated['annotation_reference']['sha256'], 'Target annotation hash differs')
    check(type(record.get('gradient_evaluations_per_image')) is int
        and record['gradient_evaluations_per_image'] == parameters['iterations']
        and type(record.get('attack_actual_gradient_evaluations_total')) is int
        and record['attack_actual_gradient_evaluations_total'] == projected['images'] * parameters['iterations'],
        'Target inherited another generation schedule')
    expected_check = dict(status='record_files_linked_pending_payload_and_execution_acceptance',
        generation_reference=generated['generation_reference'], target_reference=target['record'],
        generation_metadata=generated['generation_metadata'], target_metadata=target_summary,
        recorded_generation_path_matches=True, referenced_bytes_verified=True,
        input_authentication_verified=False, generated_pixels_verified=False,
        prediction_archive_verified=False, runtime_implementation_verified=False,
        independent_result_acceptance=False, reuse_accepted=False, formal_execution_admission=False)
    same(target['metadata_check'], expected_check, 'Worker target metadata summary differs')
    payload = _prediction_payload(read, payloads, leaf, record, target['prediction_archive'],
        projected['ordered_image_ids'], generated['category_ids'])
    binding = dict(group_id=group['group_id'], dataset=group['dataset'], split=group['split'],
        source=group['source'], seed=group['seed'], target=name, attack=ALIAS,
        parameters_sha256=group['parameters_sha256'], checkpoint_sha256=record['checkpoint_sha256'],
        code_commit=record['code_commit'], images=projected['images'],
        image_ids_sha256=projected['image_ids_sha256'], projected_binding_sha256=projected['binding_sha256'],
        generation_reference=generated['generation_reference'], manifest_reference=generated['manifest_reference'],
        annotation_reference=generated['annotation_reference'],
        canonical_annotation_reference=generated['canonical_annotation_reference'],
        metrics_reference=deepcopy(target['record']), sidecar_reference=payload['sidecar_reference'],
        archive_reference=payload['archive_reference'])
    return dict(binding, metrics=deepcopy(record['metrics']), prediction_payload=payload,
        input_binding=deepcopy(binding), input_binding_sha256=canonical_hash(binding),
        status='formal_cell_inputs_verified_pending_official_replay',
        max_images=None, diagnostic_only=False, **_flags())


def _sources(read):
    root = Path(__file__).absolute().parents[1]
    names = ('tools/vcsf_final_followup_result_inputs.py', 'tools/audit_vcsf_cpu_analysis.py',
        'tools/vcsf_final_followup_bound_dispatch.py', 'tools/run_prepare_vcsf_final_followup_inputs.py',
        'tools/vcsf_final_followup_terminal_audit.py', 'tools/vcsf_final_followup_record_binding.py',
        'src/lgp/reporting/qualitative_inputs.py', 'src/lgp/runners/vcsf_final_followup_records.py',
        'src/lgp/registry.py', 'src/lgp/metrics.py', 'src/lgp/reporting/coco_summary.py',
        'src/lgp/runners/vcsf_final_followup_inputs.py',
        'src/lgp/runners/vcsf_final_followup_execution_contract.py',
        'src/lgp/runners/vcsf_final_followup_dispatch.py',
        'src/lgp/runners/vcsf_final_followup_method.py',
        'src/lgp/runners/vcsf_final_followup_plan.py',
        'src/lgp/runners/vcsf_common_anchor_execution_contract.py',
        'src/lgp/runners/vcsf_research_plan.py')
    result = {}
    for name in names:
        path = plain(root / name)
        result[name] = sha(path)
        read.bytes(path, result[name])
    return str(root), result


def collect_formal_result_inputs(runtime_reference, admission_reference, completion_reference, *, registry=None):
    return _collect_result_inputs(runtime_reference, admission_reference,
        completion_reference, registry=registry, successor=False)


def collect_successor_result_inputs(runtime_reference, admission_reference,
        completion_reference, *, registry=None):
    """Collect the admitted four-device groups4..16, never the cross-root union."""
    return _collect_result_inputs(runtime_reference, admission_reference,
        completion_reference, registry=registry, successor=True)


def _authenticate_successor(read, runtime_reference, admission_reference,
        completion_reference, registry):
    from tools.run_prepare_vcsf_final_followup_inputs import _catalogue
    from lgp.runners.vcsf_final_followup_successor_admission import read_stored_successor_execution_admission

    runtime = _reference(read, runtime_reference)
    path = plain(runtime_reference['file'])
    check(len(path.parents) >= 5, 'Runtime namespace is incomplete')
    producer = path.parents[4]
    registry = registry if registry is not None else Registry(producer)
    check(plain(registry.root) == producer, 'Registry must belong to bound successor')
    scope = runtime['scope']
    check(scope['diagnostic_only'] is False and scope['max_images'] is None
        and scope['group_ids'] == list(range(4, 17)) and scope['images'] == 64712
        and scope['target_cells'] == 208
        and scope['payload_policy'] == 'keep_all_until_independent_acceptance',
        'Require the complete registered thirteen-group successor scope')
    targets = list(registry.target_ids())
    check(len(targets) == len(set(targets)) == 16, 'Require canonical sixteen targets')
    same([g['group_id'] for g in scope['groups']], scope['group_ids'], 'Group order differs')
    for group in scope['groups']:
        check(group['dataset'] in ('coco', 'voc') and group['split'] == 'val'
            and type(group['images']) is int
            and group['images'] == (5000 if group['dataset'] == 'coco' else 4952),
            'Require full registered dataset image count')
        same(group['targets'], targets, 'Canonical target order differs')
    root = plain(completion_reference['file']).parent
    _reference(read, completion_reference, root / 'completion.json')
    request = runtime['preparation_request']
    catalogue = _catalogue(read, registry, request['preparation'], request['input_report'])
    # Result auditing authenticates stored producer evidence, not a live executor.
    admitted = read_stored_successor_execution_admission(read, registry, catalogue,
        scope, runtime_reference, admission_reference)
    same(admitted['scope'], scope, 'Authenticated successor scope differs')
    group = scope['groups'][0]
    refs = admitted['input_references'][group['dataset']]
    prepared = prepare_bound_group_calls(read, registry, refs['publication'], refs['audit'],
        group['group_id'], root / 'groups/000004', 'cuda:0', max_images=None,
        expected_catalogue_sha256=scope['catalogue_content_sha256'])
    projected = prepared['projected_inputs']
    bindings = [row for row in admitted['group_input_bindings']
        if row['group_id'] == group['group_id']]
    same(bindings, [dict(group_id=group['group_id'],
        input_binding_sha256=projected['binding_sha256'],
        execution_input_binding_sha256=projected['binding_sha256'])],
        'Successor full input projection differs from admission')
    same(prepared['input_binding_sha256'], projected['binding_sha256'],
        'Prepared successor binding differs')
    bound = dict(admission=admitted, prepared=prepared, group=group, catalogue=catalogue)
    terminal = audit_terminal_chain(read, root, runtime_reference, admission_reference,
        scope, expected_device_count=4)
    return registry, runtime, root, bound, terminal


def _collect_result_inputs(runtime_reference, admission_reference, completion_reference,
        *, registry=None, successor=False):
    """Consume all fifteen completed groups; raise with a sealed failure snapshot.

    This never writes a report, modifies a run, invokes models/official replay,
    accepts seed42 reuse, or issues result acceptance. The outer publisher owns
    success/failure publication and independently pins this report's file hash.
    """
    read, payloads = _Metadata(), _Payloads()
    groups, cells = [], []
    context = dict(stage='authentication', group_id=None, target=None, image_id=None)
    references = dict(runtime_reference=deepcopy(runtime_reference),
        admission_reference=deepcopy(admission_reference), completion_reference=deepcopy(completion_reference))
    try:
        auditor_root, sources = _sources(read)
        if successor:
            for name in ('tools/vcsf_final_followup_successor_bound_dispatch.py',
                    'src/lgp/runners/vcsf_final_followup_successor_admission.py'):
                path = plain(Path(auditor_root) / name)
                sources[name] = sha(path)
                read.bytes(path, sources[name])
        authenticate = _authenticate_successor if successor else _authenticate
        registry, runtime, root, bound, terminal = authenticate(read, runtime_reference,
            admission_reference, completion_reference, registry)
        scope, catalogue = runtime['scope'], bound['catalogue']
        inventory = _tree(root)
        for group in scope['groups']:
            group_id = group['group_id']
            context.update(stage='group_inputs', group_id=group_id, target=None)
            directory = root / 'groups' / ('%06d' % group_id)
            if group_id == scope['group_ids'][0]:
                prepared = bound['prepared']
            else:
                refs = runtime['input_references'][group['dataset']]
                prepared = prepare_bound_group_calls(read, registry, refs['publication'], refs['audit'],
                    group_id, directory, 'cuda:0', max_images=None,
                    expected_catalogue_sha256=scope['catalogue_content_sha256'])
            projected = prepared['projected_inputs']
            _projection_metadata(read, projected)
            worker_ref, worker = _snapshot(read, directory / 'worker_group.json')
            check(worker['input_binding_sha256'] == worker['execution_input_binding_sha256']
                == projected['binding_sha256'], 'Worker full projection differs')
            same([t['target'] for t in worker['targets']], group['targets'], 'Target panel differs')
            parameters = prepared['attack']['parameter_overrides']
            context['stage'] = 'generation_payloads'
            generated = _generation(read, payloads, directory, worker, projected, parameters, context)
            for target in worker['targets']:
                context.update(stage='target_payload', target=target['target'])
                cells.append(_cell(read, payloads, directory, group, projected, generated, target, parameters))
            context.update(stage='group_payload_readback', target=None)
            group_payloads = {name for name in payloads.files if directory in Path(name).parents}
            group_payloads.update(str(plain(row['input']['file'])) for row in projected['seeded_clean_rows'])
            payloads.unchanged(group_payloads, rehash=True)
            groups.append(dict(group_id=group_id, group=deepcopy(group),
                projected_inputs=deepcopy(projected), worker_group_reference=worker_ref, **generated))
        context.update(stage='final_stability', group_id=None, target=None)
        expected_pairs = [(g['group_id'], t) for g in scope['groups'] for t in g['targets']]
        same([(c['group_id'], c['target']) for c in cells], expected_pairs, 'Full cell order/coverage differs')
        expected_cells, expected_images = (208, 64712) if successor else (240, 74712)
        check(len(cells) == expected_cells and sum(g['png_count'] for g in groups) == expected_images,
            'Formal full payload totals differ')
        expected_pngs = {'groups/%06d/attack/images/%012d.png' % (g['group_id'], image_id)
            for g in groups for image_id in g['projected_inputs']['ordered_image_ids']}
        check({n for n in inventory if n.lower().endswith('.png')} == expected_pngs, 'Global PNG inventory differs')
        for suffix, reference_key in (('metrics.json', 'metrics_reference'),
                ('predictions_artifact.json', 'sidecar_reference'), ('.gz', 'archive_reference')):
            expected = {Path(c[reference_key]['file']).relative_to(root).as_posix() for c in cells}
            check({n for n in inventory if n.endswith(suffix)} == expected, 'Target file inventory differs')
        read.unchanged()
        payloads.unchanged()
        same(_tree(root), inventory, 'Execution inventory changed during consumption')
        raw_inventory = {Path(name).relative_to(root).as_posix(): digest
            for name, digest in read.checked.items() if root in Path(name).parents}
        for name, payload in payloads.files.items():
            if root in Path(name).parents:
                relative = Path(name).relative_to(root).as_posix()
                check(relative not in raw_inventory or raw_inventory[relative] == payload['sha256'],
                    'Conflicting raw inventory entry')
                raw_inventory[relative] = payload['sha256']
        return _seal(dict(schema_version=1, status=STATUS, execution_root=str(root), **references,
            scope_sha256=scope['scope_sha256'], runtime_snapshot_sha256=canonical_hash(runtime['runtime_sha256']),
            scope=deepcopy(scope), input_references=deepcopy(runtime['input_references']),
            catalogue_content_sha256=canonical_hash(catalogue), catalogue=deepcopy(catalogue),
            group_ids=list(scope['group_ids']), targets=list(registry.target_ids()),
            groups=groups, cells=cells, group_count=13 if successor else 15, png_count=expected_images,
            target_cells=expected_cells, metric_values=12 * expected_cells, max_images=None, diagnostic_only=False,
            terminal_chain=terminal, input_authentication_verified=True, full_payloads_verified=True,
            prediction_records_retained=False, payload_stability_check='device_inode_size_mtime_ctime',
            payload_files=deepcopy(payloads.files), raw_inventory=raw_inventory,
            metadata_sha256=dict(read.checked),
            checked_publication_directories=sorted(str(p) for p in read.directories),
            auditor_source_root=auditor_root, auditor_source_sha256=sources,
            resource_limits=dict(metadata_bytes=METADATA_BYTES, archive_bytes=MAX_ARCHIVE_BYTES,
                decompressed_bytes=MAX_DECOMPRESSED_BYTES, prediction_record_bytes=MAX_RECORD_BYTES,
                prediction_chunk_bytes=STREAM_CHUNK_BYTES), **_flags()))
    except Exception as error:
        failure = _seal(dict(schema_version=1, status='formal_followup_result_inputs_failed',
            **references, context=dict(context), error_type=type(error).__name__, error=str(error),
            completed_group_ids=[g['group_id'] for g in groups],
            completed_cells=[dict(group_id=c['group_id'], target=c['target'],
                input_binding_sha256=c['input_binding_sha256']) for c in cells],
            metadata_sha256=dict(read.checked), payload_files=deepcopy(payloads.files),
            partial_evidence_is_acceptance=False, producer_files_modified=False, **_flags()))
        raise FormalResultInputsError(failure) from error
