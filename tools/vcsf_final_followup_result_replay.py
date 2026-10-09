"""One formal follow-up cell replay; caller authenticates the outer input audit."""
from copy import deepcopy
from pathlib import Path
import sys
import time

from lgp.io import file_digest
from lgp.result_runtime import verify_result_runtime
from lgp.metrics import COCO_BBOX_METRICS
from lgp.reporting.vcsf_coco_replay import replay_coco_bbox
from lgp.reporting.vcsf_prediction_evidence import load_bound_predictions
from lgp.runners.vcsf_final_followup_dispatch import ALIAS
from lgp.runners.vcsf_final_followup_records import inspect_target_record
from lgp.runners.vcsf_research_plan import canonical_hash, require
from tools.audit_vcsf_cpu_analysis import Evidence, plain
from tools.preflight_vcsf_archived_bootstrap import verify_annotation_equivalence


def validate_cell_scope(group, projection, target):
    require(type(group.get('group_id')) is int and 2 <= group['group_id'] <= 16
        and group.get('disposition') == 'prospective_new', 'Require a new formal group')
    require(projection.get('binding_sha256') == canonical_hash(
        {k: v for k, v in projection.items() if k != 'binding_sha256'}), 'Projection seal changed')
    for key in ('dataset', 'split', 'source', 'seed', 'images', 'parameters_sha256'):
        require(canonical_hash(group.get(key)) == canonical_hash(projection.get(key)),
            'Group and projection differ: ' + key)
    expected = {'coco': 5000, 'voc': 4952}
    require(group['dataset'] in expected and type(group['images']) is int
        and group['images'] == expected[group['dataset']] and group['split'] == 'val',
        'Require full formal dataset')
    require(type(group['seed']) is int and group['seed'] in (42, 43, 44, 45, 46),
        'Invalid formal seed')
    require(group['seed'] == 42 or (group['dataset'] == 'coco'
        and group['source'] == 'faster_rcnn_r50'), 'Seed is outside the registered stability scope')
    targets, ids = projection['targets'], projection['ordered_image_ids']
    require(group['targets'] == targets and len(targets) == len(set(targets)) == 16
        and target in targets, 'Target panel differs')
    require(len(ids) == len(set(ids)) == group['images'] and ids == sorted(ids)
        and all(type(i) is int and i > 0 for i in ids)
        and canonical_hash(ids) == projection['image_ids_sha256'], 'Formal image identities differ')
    return ids


def replay_new_followup_cell(group, projection, target, *, run_root, checked_inventory,
        canonical_annotation, scratch_dir, max_uncompressed_bytes=4 << 30):
    """Recompute all twelve metrics from a caller-qualified full prediction cell.

    Inventory keys are paths relative to run_root, and values are SHA256 strings.
    The outer pipeline must authenticate its metadata/pixel audit before calling.
    This wrapper does not accept the matrix or the reused group1 observation.
    """
    verify_result_runtime(Path(__file__).resolve().parents[1])
    ids = validate_cell_scope(group, projection, target)
    root, scratch = plain(Path(run_root).absolute()), plain(Path(scratch_dir).absolute())
    require(root.is_dir() and scratch.is_dir() and root != scratch
        and root not in scratch.parents and scratch not in root.parents, 'Scratch overlaps immutable evidence')
    require(type(max_uncompressed_bytes) is int and 0 < max_uncompressed_bytes <= 8 << 30,
        'Replay cap must be explicit and at most 8 GiB; caller must qualify RAM separately')
    prefix = 'groups/{:06d}'.format(group['group_id'])
    reader = Evidence()

    def bound(relative):
        require(relative in checked_inventory, 'Unqualified input: ' + relative)
        return reader.json(root / relative, checked_inventory[relative])

    canonical_path = plain(Path(canonical_annotation['file']))
    canonical = reader.json(canonical_path, canonical_annotation['sha256'])
    require(canonical_annotation['sha256'] == projection['annotation']['sha256']
        and canonical_path == Path(projection['annotation']['file']), 'Canonical annotation changed')
    canonical_ids = sorted(row['id'] for row in canonical['images'])
    require(canonical_ids == ids, 'Canonical annotation does not cover the full group')
    annotation_name = prefix + '/attack/annotations.json'
    annotation = bound(annotation_name)
    verify_annotation_equivalence(canonical, annotation, ids)
    generation = bound(prefix + '/attack/run.json')
    require(generation['seed'] == group['seed'] and type(generation['seed']) is int
        and generation['manifest_sha256'] == checked_inventory.get(prefix + '/attack/manifest.jsonl')
        and generation['annotation_sha256'] == checked_inventory[annotation_name]
        and generation['image_ids_sha256'] == projection['image_ids_sha256']
        and generation['parameters_sha256'] == group['parameters_sha256']
        and generation['status'] == 'complete', 'Generation linkage differs')
    cell_prefix = prefix + '/evaluations/' + target
    metrics_name, artifact_name = cell_prefix + '/metrics.json', cell_prefix + '/predictions_artifact.json'
    metrics, artifact = bound(metrics_name), bound(artifact_name)
    inspect_target_record(metrics, projection, target, ALIAS)
    require(set(metrics['metrics']) == set(COCO_BBOX_METRICS), 'Require exactly twelve metrics')
    require(Path(metrics['annotation']) == root / annotation_name
        and Path(metrics['adversarial_run']) == root / prefix / 'attack', 'Foreign prediction source')
    archive_name = cell_prefix + '/predictions.json.gz'
    require(checked_inventory.get(archive_name) == artifact['archive_sha256'], 'Archive inventory differs')
    identity = dict(dataset=group['dataset'], split=group['split'], source=group['source'],
        attack=ALIAS, target=target, parameters_sha256=group['parameters_sha256'],
        checkpoint_sha256=projection['checkpoints'][target]['sha256'], code_commit=projection['base_commit'])
    started = time.monotonic()
    predictions, archive = load_bound_predictions(root / cell_prefix, immutable_root=root,
        metrics_sha256=checked_inventory[metrics_name], artifact_sha256=checked_inventory[artifact_name],
        expected_identity=identity, image_ids=ids,
        category_ids=sorted(c['id'] for c in annotation['categories']), scratch_dir=scratch,
        max_uncompressed_bytes=max_uncompressed_bytes)
    archive_seconds = time.monotonic() - started
    started = time.monotonic()
    evaluator, replay = replay_coco_bbox(annotation, predictions, metrics['metrics'], ids)
    del evaluator, predictions
    replay_seconds = time.monotonic() - started
    reader.unchanged()
    require(file_digest(root / archive_name) == checked_inventory[archive_name], 'Archive changed during replay')
    binding = dict(group=deepcopy(group), target=target, projected_binding_sha256=projection['binding_sha256'],
        canonical_annotation=deepcopy(canonical_annotation), checked_metadata_sha256=dict(reader.checked),
        archive_reference=dict(file=str(root / archive_name), sha256=checked_inventory[archive_name]))
    cell = dict(group_id=group['group_id'], dataset=group['dataset'], split=group['split'],
        source=group['source'], seed=group['seed'], target=target, images=len(ids),
        parameters_sha256=group['parameters_sha256'], image_ids_sha256=projection['image_ids_sha256'],
        max_images=None, diagnostic_only=False, metrics=deepcopy(replay['metrics']),
        input_evidence_sha256=canonical_hash(binding))
    return dict(status='followup_new_cell_official_replay_pending_outer_acceptance',
        normalized_cell=cell, input_binding=binding, archive_binding=archive, replay=replay,
        archive_seconds=archive_seconds, official_and_identity_replay_seconds=replay_seconds,
        model_calls=0, AP_replays=1, bootstrap_replicates_computed=0,
        outer_input_authority_verified=False, independent_result_acceptance=False,
        scientific_acceptance=False, formal_metrics_eligible=False)
