"""Read pinned generation/target bytes without accepting a result or executing models."""
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath

from tools.audit_vcsf_cpu_analysis import check, plain, no_duplicates, no_constant, same


PAYLOAD_BYTE_LIMIT = 128 << 20
PIXEL_LIMIT = 16 << 20


def _payload_path(group_directory, relative_name):
    check(type(relative_name) is str and bool(relative_name) and
        '\\' not in relative_name and ':' not in relative_name and
        not PureWindowsPath(relative_name).drive and
        all(part not in ('', '.', '..') for part in relative_name.split('/')),
        'Unsafe payload relative path')
    root = plain(group_directory)
    path = plain(root / relative_name)
    check(path != root and root in path.parents and path.is_file(),
        'Payload is not a file under the authenticated group directory')
    return path


def _count(value, label, cap=None):
    check(type(value) is int and value >= 0 and (cap is None or value <= cap),
        'Invalid or excessive ' + label)
    return value


def _limit(value):
    check(type(value) is int and 0 < value <= PAYLOAD_BYTE_LIMIT,
        'Payload byte limit must be between 1 and 128 MiB')
    return value


def _pending_flags():
    return dict(input_authentication_verified=False, runtime_implementation_verified=False,
        independent_result_acceptance=False, reuse_accepted=False,
        formal_execution_admission=False, formal_metrics_verified=False)


def audit_generated_png(read, group_directory, manifest_row, clean_reference,
        expected_relative_path, expected_image_id, expected_shape, budget_uint8=4,
        max_bytes=PAYLOAD_BYTE_LIMIT):
    """Audit one pinned PNG, not a generation run or model execution.

    Caller authenticates group, manifest row, clean reference and expectations.
    Shape is (height, width, 3); clean reference has file/sha256/bytes.
    Clean decoding follows the attack's EXIF transpose and RGB conversion.
    """
    import numpy as np
    from PIL import Image, ImageOps

    cap = _limit(max_bytes)
    check(type(budget_uint8) is int and 0 <= budget_uint8 <= 4, 'Invalid uint8 budget')
    check(type(expected_shape) in (tuple, list) and len(expected_shape) == 3 and
        all(type(v) is int and v > 0 for v in expected_shape) and
        expected_shape[2] == 3 and expected_shape[0] * expected_shape[1] <= PIXEL_LIMIT,
        'Invalid or excessive pinned RGB shape')
    check(type(expected_image_id) is int and expected_image_id > 0 and
        type(manifest_row.get('image_id')) is int and
        manifest_row['image_id'] == expected_image_id, 'PNG image ID differs')
    check(manifest_row.get('output_file') == expected_relative_path,
        'PNG path differs from expected group payload')
    path = _payload_path(group_directory, expected_relative_path)
    clean_path = plain(clean_reference['file'])
    if 'source_file' in manifest_row:
        check(plain(manifest_row['source_file']) == clean_path, 'PNG clean source differs')
    raw = read.bytes(path, manifest_row['output_sha256'], cap)
    clean_raw = read.bytes(clean_path, clean_reference['sha256'], cap)
    check(len(raw) == _count(manifest_row.get('output_bytes'), 'PNG bytes', cap),
        'PNG byte count differs')
    check(len(clean_raw) == _count(clean_reference.get('bytes'), 'clean bytes', cap),
        'Clean byte count differs')
    size = (expected_shape[1], expected_shape[0])
    with Image.open(io.BytesIO(raw)) as generated, Image.open(io.BytesIO(clean_raw)) as clean:
        check(generated.format == 'PNG' and generated.mode == 'RGB' and
            getattr(generated, 'n_frames', 1) == 1, 'Generated payload is not single-frame RGB PNG')
        check(getattr(clean, 'n_frames', 1) == 1, 'Clean image has multiple frames')
        clean_rgb = ImageOps.exif_transpose(clean).convert('RGB')
        check(generated.size == clean_rgb.size == size, 'PNG or clean dimensions differ')
        # Cast before subtraction: uint8 arithmetic would wrap at 0 and 255.
        delta = np.asarray(generated, dtype=np.int16) - np.asarray(clean_rgb, dtype=np.int16)
    observed = int(np.abs(delta).max())
    check(observed <= budget_uint8, 'PNG exceeds uint8 L-infinity budget')
    read.unchanged()
    return dict(status='png_payload_verified_not_independently_accepted',
        file=str(path), image_id=expected_image_id, sha256=hashlib.sha256(raw).hexdigest(),
        bytes=len(raw), shape=list(expected_shape), format='PNG', mode='RGB',
        observed_linf_uint8=observed, observed_delta_min=int(delta.min()),
        observed_delta_max=int(delta.max()), pixel_identical=observed == 0,
        budget_uint8=budget_uint8, generated_pixels_verified=True,
        referenced_bytes_verified=True, **_pending_flags())


def audit_prediction_archive(read, group_directory, metrics_reference, image_ids,
        category_ids, expected_relative_path='predictions.json.gz',
        sidecar_reference=None, max_bytes=PAYLOAD_BYTE_LIMIT):
    """Read authenticated metrics and bounded gzip/JSON; never evaluate AP.

    Caller authenticates the group, metrics reference and canonical ID sets.
    An existing sidecar requires its own pinned reference and exact content
    equality. Missing detections are valid; missing evaluation is not inferred.
    """
    cap = _limit(max_bytes)
    canonical = []
    for values in (image_ids, category_ids):
        check(type(values) in (list, tuple, set, frozenset) and
            all(type(v) is int and v > 0 for v in values) and
            len(set(values)) == len(values), 'Invalid canonical ID set')
        canonical.append(set(values))
    images, categories = canonical
    metrics_path = _payload_path(group_directory, 'metrics.json')
    check(plain(metrics_reference['file']) == metrics_path, 'Metrics reference belongs to another group')
    metrics = read.json(metrics_path, metrics_reference['sha256'])
    check(type(metrics) is dict and type(metrics.get('predictions_artifact')) is dict,
        'Missing predictions artifact')
    artifact = metrics['predictions_artifact']
    for key, expected in (('format', 'gzip'), ('archive_file', expected_relative_path),
            ('uncompressed_file', 'predictions.json')):
        check(key not in artifact or artifact[key] == expected,
            'Prediction artifact path or format differs')
    archive_size = _count(artifact.get('archive_bytes'), 'archive bytes', cap)
    raw_size = _count(artifact.get('uncompressed_bytes'), 'uncompressed bytes', cap)
    path = _payload_path(group_directory, expected_relative_path)
    compressed = read.bytes(path, artifact['archive_sha256'], cap)
    check(len(compressed) == archive_size, 'Compressed prediction byte count differs')
    raw = bytearray()
    # Read one byte past the independent cap to detect expansion, including
    # concatenated gzip members; reaching EOF also verifies gzip CRC/trailers.
    with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode='rb') as stream:
        while True:
            block = stream.read(min(1 << 20, cap - len(raw) + 1))
            if not block:
                break
            check(len(raw) + len(block) <= cap, 'Prediction decompression exceeds byte limit')
            raw.extend(block)
    check(len(raw) == raw_size and hashlib.sha256(raw).hexdigest() == artifact.get('uncompressed_sha256'),
        'Uncompressed prediction identity differs')
    if 'predictions_sha256' in metrics:
        check(metrics['predictions_sha256'] == artifact['uncompressed_sha256'], 'Metrics prediction hash differs')
    rows = json.loads(raw.decode('utf-8'), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    check(type(rows) is list, 'Predictions must be a JSON array')
    seen = set()
    for row in rows:
        check(type(row) is dict and set(row) == {'image_id', 'category_id', 'bbox', 'score'},
            'Prediction record schema differs')
        check(type(row['image_id']) is int and row['image_id'] in images and
            type(row['category_id']) is int and row['category_id'] in categories,
            'Prediction ID outside canonical sets')
        bbox = row['bbox']
        check(type(bbox) is list and len(bbox) == 4, 'Invalid prediction bbox')
        for value in bbox + [row['score']]:
            check(type(value) in (int, float), 'Boolean or nonnumeric prediction value')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            check(finite, 'Non-finite or overflowing prediction value')
        check(bbox[2] >= 0 and bbox[3] >= 0 and 0 <= row['score'] <= 1,
            'Prediction bbox extent or score out of range')
        seen.add(row['image_id'])
    check(_count(artifact.get('prediction_records'), 'prediction count') == len(rows),
        'Prediction artifact counts differ')
    if 'image_ids_with_detections' in artifact:
        check(_count(artifact['image_ids_with_detections'], 'detected image count') == len(seen),
            'Prediction detected image count differs')
    if 'detections' in metrics:
        check(_count(metrics['detections'], 'metrics detections') == len(rows), 'Metrics detection count differs')
    sidecar_path = plain(Path(group_directory) / 'predictions_artifact.json')
    sidecar_verified = False
    if sidecar_reference is not None or sidecar_path.exists():
        check(type(sidecar_reference) is dict, 'Existing sidecar requires a pinned reference')
        check(plain(sidecar_reference['file']) == sidecar_path, 'Sidecar belongs to another group')
        same(read.json(sidecar_path, sidecar_reference['sha256']), artifact, 'Prediction sidecar differs')
        sidecar_verified = True
    read.unchanged()
    return dict(status='prediction_payload_verified_not_independently_accepted',
        archive_sha256=hashlib.sha256(compressed).hexdigest(), archive_bytes=len(compressed),
        uncompressed_sha256=hashlib.sha256(raw).hexdigest(), uncompressed_bytes=len(raw),
        prediction_records=len(rows), image_ids_with_detections=len(seen),
        images_without_detections=len(images - seen), prediction_archive_verified=True,
        prediction_schema_verified=True, sidecar_verified=sidecar_verified,
        evaluated_image_coverage_verified=False, **_pending_flags())


def bind_target_to_generation(read, generation_reference, target_reference,
        projected_inputs, target, attack_id, parameters, logical_gradients):
    """Caller must authenticate references and projection before using this link.

    Uses the shared Evidence reader so byte drift is rejected before returning.
    Equality of recorded paths does not prove inference consumed those pixels.
    """
    from lgp.runners.vcsf_final_followup_records import (
        inspect_generation_metadata, inspect_target_record,
    )

    generation_path = plain(Path(generation_reference['file']))
    target_path = plain(Path(target_reference['file']))
    check(generation_path.name == 'run.json' and target_path.name == 'metrics.json',
        'Unexpected generation or target record filename')
    run = read.json(generation_path, generation_reference['sha256'])
    record = read.json(target_path, target_reference['sha256'])
    check(run.get('manifest') == 'manifest.jsonl', 'Unexpected generation manifest location')
    manifest_path = plain(generation_path.parent / 'manifest.jsonl')
    manifest_bytes = read.bytes(manifest_path, run['manifest_sha256'], 128 << 20)
    generation_check = inspect_generation_metadata(run, manifest_bytes, projected_inputs,
        attack_id, parameters, logical_gradients)
    target_check = inspect_target_record(record, projected_inputs, target, attack_id)
    check(isinstance(record.get('adversarial_run'), str) and
        plain(Path(record['adversarial_run'])).resolve() == generation_path.parent.resolve(),
        'Target points to another generation directory')
    check(record.get('gradient_evaluations_per_image') == run['gradient_evaluations_per_image'] and
        type(record.get('gradient_evaluations_per_image')) is int and
        record.get('attack_actual_gradient_evaluations_total') == run['actual_gradient_evaluations_total'] and
        type(record.get('attack_actual_gradient_evaluations_total')) is int,
        'Target inherited another generation gradient schedule')
    read.unchanged()
    return dict(status='record_files_linked_pending_payload_and_execution_acceptance',
        generation_reference=dict(generation_reference), target_reference=dict(target_reference),
        generation_metadata=generation_check, target_metadata=target_check,
        recorded_generation_path_matches=True, referenced_bytes_verified=True,
        input_authentication_verified=False, generated_pixels_verified=False,
        prediction_archive_verified=False, runtime_implementation_verified=False,
        independent_result_acceptance=False, reuse_accepted=False,
        formal_execution_admission=False)
