"""Prepare dataset-aware call arguments; never execute or grant admission."""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import re

from .vcsf_final_followup_plan import PROTOCOL
from .vcsf_research_plan import canonical_hash, require
from .retention import PREDICTION_ARCHIVE_GZIP


ALIAS = 'vcsf_final_followup_isolated'


def prepare_worker_calls(registry, catalogue, worker_slot, projections, execution_root,
        *, max_images=None, execution_scope=None):
    """Bind registered whole-group lanes to a UUID-isolated worker's cuda:0.

    Catalogue and projections require external authentication. Physical device
    reservations and worker ownership must be checked before executing any call.
    """
    require(catalogue['protocol_id'] == PROTOCOL and
        catalogue['protocol_content_sha256'] == canonical_hash(registry.protocols[PROTOCOL]) and
        catalogue['groups_sha256'] == canonical_hash(catalogue['groups']),
        'Catalogue protocol or groups changed')
    devices = catalogue['devices']
    require(isinstance(devices, list) and len(devices) in (1, 2) and
        all(isinstance(d, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d)
            for d in devices) and len(set(devices)) == len(devices),
        'Require one or two distinct physical devices')
    require(type(worker_slot) is int and 0 <= worker_slot < len(devices),
        'Invalid follow-up worker slot')
    groups = catalogue['groups']
    ids = [g['group_id'] for g in groups]
    require(all(type(i) is int and i > 0 for i in ids) and len(ids) == len(set(ids)),
        'Catalogue contains invalid or duplicate groups')
    new_ids = [g['group_id'] for g in groups if g['disposition'] == 'prospective_new']
    require(new_ids and catalogue['prospective_new_group_ids'] == new_ids,
        'Prospective work differs from catalogue dispositions')
    lanes = [new_ids[i::len(devices)] for i in range(len(devices))]
    require(canonical_hash(catalogue['prospective_lanes']) == canonical_hash(lanes),
        'Worker lanes changed, omitted or duplicated whole groups')
    if execution_scope is not None:
        from .vcsf_final_followup_execution_contract import verify_execution_scope

        scope = verify_execution_scope(registry, catalogue, execution_scope)
        require(canonical_hash(scope['max_images']) == canonical_hash(max_images),
            'Worker image limit differs from the bound execution scope')
        lanes = scope['lanes']
    require(isinstance(projections, dict) and
        all(type(i) is int for i in projections) and set(projections) == set(lanes[worker_slot]),
        'Worker projections must match its exact group assignment')
    root = PurePosixPath(execution_root)
    category = 'experiments' if max_images is None else 'diagnostics'
    require(root.is_absolute() and '..' not in root.parts and len(root.parents) >= 3 and
        root.parent.name == 'vcsf_final_followup_execution' and
        root.parents[1].name == category and root.parents[2].name == 'outputs',
        'Worker root escaped follow-up execution namespace')
    calls = [prepare_group_calls(registry, catalogue, group_id, projections[group_id],
        str(root / 'groups' / '{:06d}'.format(group_id)), 'cuda:0', max_images=max_images)
        for group_id in lanes[worker_slot]]
    return dict(status='worker_call_arguments_prepared_not_admitted',
        worker_slot=worker_slot, physical_device=devices[worker_slot], local_device='cuda:0',
        group_ids=list(lanes[worker_slot]), calls=calls, formal_execution_admission=False,
        device_availability_verified=False, output_ownership_verified=False, model_calls=0)


def validate_attack_invocation(prepared, *, dataset_id, split, source_id, attack_id,
        seed, max_images, parameter_overrides, budget_profile, output_dir,
        image_ids=None, seed_offsets=None, input_transform=None):
    """Bind resolver arguments to a separately authenticated call preparation.

    This check does not authenticate the preparation or grant execution rights.
    It is intended to run after receipt/admission/worker-ownership validation.
    """
    require(prepared['status'] == 'group_call_arguments_prepared_not_admitted' and
        prepared['formal_execution_admission'] is False, 'Unexpected call preparation state')
    require(image_ids is None and seed_offsets is None and input_transform is None,
        'Follow-up invocation cannot override canonical image selection, seeds or inputs')
    expected = prepared['attack']
    actual = dict(dataset_id=dataset_id, split=split, source_id=source_id,
        attack_id=attack_id, seed=seed, max_images=max_images,
        parameter_overrides=parameter_overrides, budget_profile=budget_profile)
    require(isinstance(parameter_overrides, dict) and type(seed) is int and
        (max_images is None or type(max_images) is int), 'Malformed invocation parameters')
    for key, value in actual.items():
        require(canonical_hash(value) == canonical_hash(expected[key]),
            'Attack invocation differs from bound preparation: ' + key)
    require(expected['attack_id'] == ALIAS and output_dir is not None and
        isinstance(output_dir, Path) and output_dir == expected['output_dir'],
        'Attack output differs from bound group path')
    return dict(status='attack_invocation_matches_bound_preparation',
        group_id=prepared['group_id'], input_binding_sha256=prepared['input_binding_sha256'],
        formal_execution_admission=False, implementation_bridge_accepted=False)


def prepare_group_calls(registry, catalogue, group_id, projected, group_directory,
        device, *, max_images=None):
    """Caller must authenticate catalogue/projection and admit execution separately.

    Returned arguments deliberately lack the isolated execution descriptor.
    A future admitted consumer supplies that descriptor and owns execution.
    """
    protocol = registry.protocols[PROTOCOL]
    require(type(group_id) is int and group_id > 0, 'Invalid group identity')
    require(catalogue['protocol_id'] == PROTOCOL and
        catalogue['protocol_content_sha256'] == canonical_hash(protocol) and
        catalogue['groups_sha256'] == canonical_hash(catalogue['groups']),
        'Catalogue protocol or groups changed')
    matches = [g for g in catalogue['groups'] if g['group_id'] == group_id]
    require(len(matches) == 1, 'Require exactly one catalogue group')
    group = matches[0]
    require(group_id in catalogue['prospective_new_group_ids'] and
        group['disposition'] == 'prospective_new', 'Do not generate a reuse candidate')
    require(projected['binding_sha256'] == canonical_hash(
        {k: v for k, v in projected.items() if k != 'binding_sha256'}),
        'Projected input binding changed')
    for key in ('dataset', 'split', 'source', 'seed', 'images', 'targets',
            'parameters_sha256', 'seed_schedule'):
        require(canonical_hash(group[key]) == canonical_hash(projected[key]),
            'Group and projected inputs differ: ' + key)
    dataset = group['dataset']
    require(dataset in protocol['datasets'] and
        group['split'] == protocol['datasets'][dataset]['split'] and
        type(group['images']) is int and group['images'] == protocol['datasets'][dataset]['images'] and
        group['source'] in registry.source_ids() and group['targets'] == registry.target_ids(),
        'Dataset or canonical model panel differs')
    seeds = protocol['stability_seeds'] if (dataset == protocol['stability_dataset'] and
        group['source'] == protocol['stability_source']) else [protocol['main_seed']]
    require(type(group['seed']) is int and group['seed'] in seeds and
        group['seed_schedule'] == protocol['seed_schedule'], 'Unregistered group seed')
    require(canonical_hash(catalogue['parameters']) == group['parameters_sha256'] and
        canonical_hash(catalogue['producer_runtime_sha256']) ==
        group['implementation_inventory_sha256'] == projected['producer_inventory_sha256'],
        'Method parameters or producer implementation differ')
    require(isinstance(device, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', device),
        'Require one explicit worker CUDA device')
    require(max_images is None or (type(max_images) is int and
        0 < max_images < group['images']), 'Diagnostic image count must be a proper subset')
    root = PurePosixPath(group_directory)
    category = 'experiments' if max_images is None else 'diagnostics'
    require(root.is_absolute() and len(root.parents) >= 5 and '..' not in root.parts and
        root.name == '{:06d}'.format(group_id) and root.parent.name == 'groups' and
        root.parents[2].name == 'vcsf_final_followup_execution' and
        root.parents[3].name == category and root.parents[4].name == 'outputs',
        'Group output escaped its formal or diagnostic namespace')
    attack = dict(dataset_id=dataset, source_id=group['source'], attack_id=ALIAS,
        split=group['split'], output_dir=Path(str(root / 'attack')), max_images=max_images,
        seed=group['seed'], device=device, download_weights=False, keep_going=False,
        strict=True, parameter_overrides=deepcopy(catalogue['parameters']),
        budget_profile=protocol['budget_profile'])
    evaluations = [dict(dataset_id=dataset, target_id=target, split=group['split'],
        adversarial_run=Path(str(root / 'attack')), output_dir=Path(str(root / 'evaluations' / target)),
        max_images=max_images, device=device, download_weights=False, keep_going=False,
        save_visualizations=False, prediction_archive=PREDICTION_ARCHIVE_GZIP) for target in group['targets']]
    return dict(status='group_call_arguments_prepared_not_admitted', group_id=group_id,
        input_binding_sha256=projected['binding_sha256'], attack=attack,
        evaluations=evaluations, diagnostic_only=max_images is not None,
        expected_images=max_images or group['images'], formal_execution_admission=False,
        device_availability_verified=False, output_ownership_verified=False, model_calls=0)
