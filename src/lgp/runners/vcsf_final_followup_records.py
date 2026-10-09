"""Metadata checks for follow-up records, never independent result acceptance."""
import hashlib
import json
import math

from ..metrics import COCO_BBOX_METRICS
from ..reporting.coco_summary import coco_bbox_summary_rows
from .vcsf_research_plan import canonical_hash, require


def inspect_generation_metadata(run, manifest_bytes, projected_inputs, attack_id,
        parameters, logical_gradients):
    """Bind recorded seeds to manifest bytes; do not claim actual file execution."""
    binding = projected_inputs
    require(binding['binding_sha256'] == canonical_hash(
        {k: v for k, v in binding.items() if k != 'binding_sha256'}),
        'Projected input content changed')
    require(type(logical_gradients) is int and logical_gradients > 0 and
        isinstance(attack_id, str) and bool(attack_id), 'Invalid method expectations')
    require(canonical_hash(parameters) == binding['parameters_sha256'] and
        canonical_hash(run.get('parameters')) == binding['parameters_sha256'],
        'Generated parameter content differs')
    for key in ('dataset', 'split', 'source', 'parameters_sha256', 'seed_schedule'):
        require(run.get(key) == binding[key], 'Generation scope differs: ' + key)
    require(type(run.get('seed')) is int and run['seed'] == binding['seed'] and
        run.get('attack') == attack_id and run.get('code_commit') == binding['base_commit'] and
        run.get('checkpoint_sha256') == binding['checkpoints'][binding['source']]['sha256'],
        'Generation seed, attack, revision or source checkpoint differs')
    require(run.get('status') == 'complete', 'Generation is incomplete')
    for key, expected in (('requested_images', binding['images']),
            ('successful_images', binding['images']), ('failed_images', 0),
            ('gradient_evaluations_per_image', logical_gradients),
            ('actual_gradient_evaluations_total', binding['images'] * logical_gradients)):
        require(type(run.get(key)) is int and run[key] == expected,
            'Generation count differs: ' + key)
    ids = binding['ordered_image_ids']
    require(len(ids) == binding['images'] and len(set(ids)) == len(ids) and
        all(type(v) is int and v > 0 for v in ids) and ids == sorted(ids) and
        canonical_hash(ids) == binding['image_ids_sha256'], 'Bound image order differs')
    for key in ('image_ids_sha256', 'requested_ordered_image_ids_sha256'):
        require(run.get(key) == binding['image_ids_sha256'], 'Generation image order differs')
    offsets = canonical_hash([[v, i] for i, v in enumerate(ids)])
    require(run.get('seed_schedule') == 'selected_position' and
        run.get('seed_offsets_sha256') == binding['seed_offsets_sha256'] == offsets,
        'Generation seed offsets differ')
    require(isinstance(manifest_bytes, bytes) and
        hashlib.sha256(manifest_bytes).hexdigest() == run.get('manifest_sha256'),
        'Generation manifest bytes differ')
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    require(len(rows) == len(ids), 'Generation manifest count differs')
    for position, (row, image_id) in enumerate(zip(rows, ids)):
        require(row.get('status') == 'ok', 'Generation manifest contains failed image')
        for key, expected in (('image_id', image_id), ('position', position),
                ('attack_seed', binding['seed'] + position),
                ('actual_gradient_evaluations', logical_gradients)):
            require(type(row.get(key)) is int and row[key] == expected,
                'Generation manifest identity differs: ' + key)
    mapping = canonical_hash([[v, i, binding['seed'] + i] for i, v in enumerate(ids)])
    require(mapping == binding['seed_mapping_sha256'], 'Bound seed mapping differs')
    return dict(status='generation_metadata_consistent_not_independently_accepted',
        projected_binding_sha256=binding['binding_sha256'], run_content_sha256=canonical_hash(run),
        manifest_sha256=run['manifest_sha256'], seed_mapping_sha256=mapping,
        metadata_consistent=True, input_authentication_verified=False,
        generated_pixels_verified=False, runtime_implementation_verified=False,
        independent_result_acceptance=False, reuse_accepted=False,
        formal_execution_admission=False)


def inspect_target_record(record, projected_inputs, target, attack_id):
    """Caller authenticates the projection and binds generation/prediction files.

    A self-consistent projection is not an authenticated or accepted input.
    This helper deliberately does not admit execution or qualify result reuse.
    """
    binding = projected_inputs
    require(binding['binding_sha256'] == canonical_hash(
        {k: v for k, v in binding.items() if k != 'binding_sha256'}),
        'Projected input content changed')
    require(target in binding['targets'] and isinstance(attack_id, str) and bool(attack_id),
        'Target or attack identity missing')
    require(record.get('status') == 'complete' and record.get('failures') == [],
        'Target record is incomplete')
    for key in ('dataset', 'split', 'source', 'parameters_sha256'):
        require(record.get(key) == binding[key], 'Target scope differs: ' + key)
    require(type(record.get('images')) is int and record['images'] == binding['images'] and
        record.get('target') == target and record.get('attack') == attack_id and
        record.get('checkpoint_sha256') == binding['checkpoints'][target]['sha256'] and
        record.get('code_commit') == binding['base_commit'],
        'Target image count, model, attack, checkpoint or revision differs')
    require(record.get('evaluated_image_ids_sha256') == binding['image_ids_sha256'] and
        record.get('expected_image_ids_sha256') == binding['image_ids_sha256'] and
        record.get('evaluated_image_ids_match_expected') is True,
        'Target image set differs')
    metrics = record.get('metrics', {})
    require(isinstance(metrics, dict), 'Missing metric mapping')
    values = [metrics.get(key) for key in COCO_BBOX_METRICS]
    require(all(type(value) in (int, float) and math.isfinite(value) and
        (value == -1 or 0 <= value <= 1) for value in values) and values[0] >= 0,
        'Missing or invalid unrounded twelve-metric panel')
    summary = record.get('coco_summary', {})
    require(isinstance(summary, dict), 'Missing COCO summary')
    # Canonical JSON also distinguishes booleans from numeric row fields.
    require(canonical_hash(summary.get('rows')) == canonical_hash(coco_bbox_summary_rows(values)),
        'COCO summary semantics, order or values differ from raw metrics')
    return dict(status='target_metadata_consistent_not_independently_accepted',
        dataset=binding['dataset'], target=target,
        projected_binding_sha256=binding['binding_sha256'], record_content_sha256=canonical_hash(record),
        metadata_consistent=True, input_authentication_verified=False,
        generation_binding_verified=False, prediction_archive_verified=False,
        independent_result_acceptance=False, reuse_accepted=False,
        formal_execution_admission=False)
