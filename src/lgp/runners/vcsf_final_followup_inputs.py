"""Project shared inputs into group views without claiming actual-asset acceptance."""
from copy import deepcopy
import hashlib
import json
import re

from .vcsf_final_followup_plan import PROTOCOL
from .vcsf_research_plan import canonical_hash, require


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _asset(value):
    require(isinstance(value, dict) and set(value) == {'file', 'bytes', 'sha256'} and
        isinstance(value['file'], str) and bool(value['file']) and
        type(value['bytes']) is int and value['bytes'] > 0 and _digest(value['sha256']),
        'Malformed asset reference; this check does not read its bytes')


def project_group_inputs(registry, group, shared, clean_rows):
    """Caller must separately authenticate inputs and independently read assets."""
    protocol = registry.protocols[PROTOCOL]
    dataset = group['dataset']
    require(dataset in protocol['datasets'], 'Dataset is outside the registered follow-up')
    scope = protocol['datasets'][dataset]
    sources, targets = registry.source_ids(), registry.target_ids()
    require(group['split'] == shared['split'] == scope['split'] and
        shared['dataset'] == dataset and type(group['images']) is int and
        type(shared['images']) is int and group['images'] == shared['images'] == scope['images'] and
        group['targets'] == shared['targets'] == targets and shared['sources'] == sources and
        group['source'] in sources, 'Group or shared input scope differs')
    seed = group['seed']
    allowed = protocol['stability_seeds'] if (dataset == protocol['stability_dataset'] and
        group['source'] == protocol['stability_source']) else [protocol['main_seed']]
    require(type(seed) is int and seed in allowed and
        group['seed_schedule'] == protocol['seed_schedule'] == 'selected_position',
        'Group seed or schedule differs from the registered scope')
    require(_digest(group['parameters_sha256']) and _digest(group['implementation_inventory_sha256']),
        'Missing group method identity')
    ids = shared['ordered_image_ids']
    require(isinstance(ids, list) and len(ids) == scope['images'] and
        all(type(v) is int and v > 0 for v in ids) and
        len(set(ids)) == len(ids) and ids == sorted(ids) and
        canonical_hash(ids) == shared['image_ids_sha256'], 'Incomplete or reordered full image IDs')
    require(isinstance(clean_rows, list) and len(clean_rows) == len(ids) and
        canonical_hash(clean_rows) == shared['clean_rows_content_sha256'],
        'Shared clean manifest content differs')
    seeded_rows = []
    for position, (image_id, row) in enumerate(zip(ids, clean_rows)):
        require(set(row) == {'image_id', 'position', 'input'} and
            type(row['image_id']) is int and row['image_id'] == image_id and
            type(row['position']) is int and row['position'] == position,
            'Shared clean-image row order differs')
        _asset(row['input'])
        seeded_rows.append(dict(deepcopy(row), attack_seed=seed + position))
    _asset(shared['annotation'])
    require(set(shared['checkpoints']) == set(targets), 'Incomplete checkpoint panel')
    for checkpoint in shared['checkpoints'].values():
        _asset(checkpoint)
    roles = [('source', model) for model in sources] + [('target', model) for model in targets]
    configs = shared['model_configs']
    require([(r['role'], r['model']) for r in configs] == roles,
        'Require six source and sixteen target role configs in canonical order')
    for row in configs:
        require(row['checkpoint_sha256'] == shared['checkpoints'][row['model']]['sha256'] and
            _digest(row['normalized_effective_sha256']), 'Role checkpoint or config identity differs')
        _asset(row['resolved_config'])
        _asset(row['effective_config'])
    require(isinstance(shared['packages'], dict) and
        {'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmyolo'} <= set(shared['packages']) and
        all(isinstance(k, str) and isinstance(v, str) and bool(v) for k, v in shared['packages'].items()),
        'Incomplete package identity')
    require(isinstance(shared['base_commit'], str) and
        re.fullmatch(r'[0-9a-f]{40}', shared['base_commit']), 'Missing base revision')
    selected = [r for r in configs if r['role'] == 'target' or r['model'] == group['source']]
    require([(r['role'], r['model']) for r in selected] ==
        [('source', group['source'])] + [('target', model) for model in targets],
        'Group role projection differs')
    value = dict(dataset=dataset, split=scope['split'], images=len(ids), seed=seed,
        source=group['source'], targets=deepcopy(targets), seed_schedule='selected_position',
        ordered_image_ids=deepcopy(ids), image_ids_sha256=shared['image_ids_sha256'],
        seed_offsets_sha256=canonical_hash([[v, i] for i, v in enumerate(ids)]),
        seed_mapping_sha256=canonical_hash([[v, i, seed+i] for i, v in enumerate(ids)]),
        seeded_clean_rows=seeded_rows, seeded_clean_rows_content_sha256=canonical_hash(seeded_rows),
        annotation=deepcopy(shared['annotation']), checkpoints=deepcopy(shared['checkpoints']),
        model_configs=deepcopy(selected), packages=deepcopy(shared['packages']),
        base_commit=shared['base_commit'], parameters_sha256=group['parameters_sha256'],
        producer_inventory_sha256=group['implementation_inventory_sha256'],
        shared_input_content_sha256=canonical_hash(shared),
        actual_assets_verified=False, independent_input_acceptance=False,
        implementation_bridge_accepted=False, reuse_accepted=False, formal_execution_admission=False)
    value['binding_sha256'] = canonical_hash(value)
    return value


def compare_seed42_metadata(original, original_clean_manifest_bytes, projected):
    """Compare pinned metadata only; neither matching hashes nor this result admits reuse."""
    require(projected['binding_sha256'] == canonical_hash(
        {k: v for k, v in projected.items() if k != 'binding_sha256'}),
        'Projected binding content changed')
    require(original['status'] == 'bound_final_attribution_full5000_runtime_inputs' and
        original['binding_sha256'] == canonical_hash(
            {k: v for k, v in original.items() if k != 'binding_sha256'}),
        'Original runtime input binding differs')
    manifest = original['clean_image_manifest']
    _asset(manifest)
    require(isinstance(original_clean_manifest_bytes, bytes) and
        len(original_clean_manifest_bytes) == manifest['bytes'] and
        hashlib.sha256(original_clean_manifest_bytes).hexdigest() == manifest['sha256'],
        'Original clean manifest bytes differ from the bound producer reference')
    original_clean_rows = json.loads(original_clean_manifest_bytes)
    require(original['sources'] == ['faster_rcnn_r50'] and
        projected['source'] == 'faster_rcnn_r50' and projected['dataset'] == 'coco' and
        type(projected['seed']) is int and projected['seed'] == 42 and projected['images'] == 5000,
        'Only the registered COCO Faster-RCNN seed42 observation can be compared')
    for key in ('dataset', 'split', 'images', 'seed', 'seed_schedule', 'ordered_image_ids',
            'image_ids_sha256', 'seed_mapping_sha256', 'targets', 'annotation', 'checkpoints',
            'packages', 'base_commit'):
        require(canonical_hash(original[key]) == canonical_hash(projected[key]),
            'Producer/destination metadata differs: ' + key)
    require(canonical_hash(original_clean_rows) == projected['seeded_clean_rows_content_sha256'] ==
        canonical_hash(projected['seeded_clean_rows']), 'Original per-image bytes/order/seed differs')
    expected = [('source', projected['source'])] + [('target', model) for model in projected['targets']]
    require([(r['role'], r['model']) for r in original['model_configs']] == expected and
        [(r['role'], r['model']) for r in original['model_config_comparisons']] == expected,
        'Original role-config panel is incomplete or reordered')
    require([dict(role=r['role'], model=r['model'], effective_sha256=r['normalized_effective_sha256'])
        for r in projected['model_configs']] == original['model_config_comparisons'],
        'Normalized detector/evaluator role config differs')
    return dict(status='seed42_input_metadata_matches_pending_asset_and_method_bridge_audits',
        original_binding_sha256=original['binding_sha256'],
        original_clean_manifest_sha256=manifest['sha256'],
        projected_binding_sha256=projected['binding_sha256'], metadata_matches=True,
        actual_assets_verified=False, independent_input_acceptance=False,
        implementation_bridge_accepted=False, reuse_accepted=False, formal_execution_admission=False,
        model_calls=0, AP_replays=0)
