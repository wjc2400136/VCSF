"""Execute one admitted whole-group lane; leave scientific acceptance to audits."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import time

from ..io import atomic_json, file_digest
from ..attacks.vcsf_final_followup_execution_isolation import (
    invocation_metadata, verify_worker_request,
)
from .attack import run_attack
from .evaluate import run_evaluation
from .vcsf_efficacy_contract import child, verify_environment
from .vcsf_efficacy_runner import _cleanup, process_start_ticks
from .vcsf_final_followup_dispatch import ALIAS
from .vcsf_final_followup_execution_contract import MODE, read_execution_admission
from .vcsf_final_followup_records import inspect_generation_metadata
from .vcsf_gpu_context_owner import RegisteredGpuOwner
from .vcsf_research_plan import canonical_hash, require
from .vcsf_structure_workers import install_parent_death_guard, verify_coordinator


def _now():
    return datetime.now(timezone.utc).isoformat()


def execution_projection(projected, max_images):
    """Preserve the full input identity while describing the canonical smoke prefix."""
    require(projected['binding_sha256'] == canonical_hash(
        {k: v for k, v in projected.items() if k != 'binding_sha256'}), 'Input projection changed')
    value = deepcopy(projected)
    if max_images is None:
        return value
    require(type(max_images) is int and max_images == 1 and projected['images'] > 1,
        'Only one-image canonical follow-up diagnostics are registered')
    ids = value['ordered_image_ids'][:1]
    value.update(images=1, ordered_image_ids=ids, image_ids_sha256=canonical_hash(ids),
        seeded_clean_rows=value['seeded_clean_rows'][:1],
        seed_offsets_sha256=canonical_hash([[ids[0], 0]]),
        seed_mapping_sha256=canonical_hash([[ids[0], 0, value['seed']]]),
        full_input_binding_sha256=projected['binding_sha256'], diagnostic_only=True)
    value['seeded_clean_rows_content_sha256'] = canonical_hash(value['seeded_clean_rows'])
    value['binding_sha256'] = canonical_hash({k: v for k, v in value.items() if k != 'binding_sha256'})
    return value


def snapshot_assets(projected):
    from tools.audit_vcsf_cpu_analysis import Evidence, plain

    read = Evidence()
    assets = [projected['annotation']] + list(projected['checkpoints'].values())
    assets += [row['input'] for row in projected['seeded_clean_rows']]
    assets += [row[key] for row in projected['model_configs'] for key in ('resolved_config', 'effective_config')]
    for asset in assets:
        path = plain(Path(asset['file']))
        require(path.stat().st_size == asset['bytes'], 'Current asset byte count differs')
        read.bytes(path, asset['sha256'])
    read.unchanged()
    return read


def _reference(path):
    from tools.audit_vcsf_cpu_analysis import plain

    path = plain(Path(path))
    return dict(file=str(path), sha256=file_digest(path))


def execute_group(registry, bound, descriptor, ownership, progress, assets, *, on_group_created=None):
    from tools.audit_vcsf_cpu_analysis import Evidence, plain, no_duplicates, no_constant
    from tools.vcsf_final_followup_record_binding import bind_target_to_generation

    prepared, group = bound['prepared'], bound['group']
    require([row['target_id'] for row in prepared['evaluations']] == group['targets'],
        'Worker call list must retain the complete ordered target panel')
    limit = bound['admission']['scope']['max_images']
    projection = execution_projection(prepared['projected_inputs'], limit)
    root = child(registry.root, prepared['attack']['output_dir'].parent.relative_to(registry.root))
    require(not root.exists(), 'Do not resume or overwrite an existing group')
    root.mkdir(parents=True, exist_ok=False)
    if on_group_created is not None:
        on_group_created(root)
    parameters = prepared['attack']['parameter_overrides']
    gradients = parameters['iterations']
    times, targets, target_readers = {}, [], []

    def callback(phase, interval):
        def update(done, total, image_id, status):
            if done in (1, total) or done % interval == 0:
                ownership()
                progress(phase=phase, images_completed=done, images_total=total,
                    last_image_id=image_id, last_image_status=status)
        return update

    ownership()
    progress(phase='generation', images_completed=0, images_total=projection['images'], targets_completed=0)
    started = time.monotonic()
    try:
        generated = run_attack(registry, **prepared['attack'], isolated_research=descriptor,
            run_metadata=invocation_metadata(bound, descriptor['runtime'], descriptor['admission'], descriptor['worker_request']),
            progress_callback=callback('generation', 25))
        require(Path(generated) == root / 'attack', 'Generator returned another output root')
    finally:
        _cleanup()
    times['generation_seconds'] = time.monotonic() - started
    generation_reference = _reference(root / 'attack/run.json')
    started = time.monotonic()
    generated_read = Evidence()
    run = generated_read.json(root / 'attack/run.json', generation_reference['sha256'])
    require(run.get('annotation') == 'annotations.json' and run.get('image_root') == '.',
        'Generated annotation or image root escaped the attack directory')
    annotation = generated_read.json(root / 'attack/annotations.json', run['annotation_sha256'])
    images = annotation.get('images')
    expected_names = {image_id: 'images/{:012d}.png'.format(image_id)
        for image_id in projection['ordered_image_ids']}
    require(isinstance(images, list) and len(images) == len(expected_names) and
        all(isinstance(row, dict) and type(row.get('id')) is int for row in images)
        and {row['id']: row.get('file_name') for row in images} == expected_names,
        'Generated annotation image identities or filenames differ')
    manifest = generated_read.bytes(root / 'attack/manifest.jsonl', run['manifest_sha256'], 128 << 20)
    generation_metadata = inspect_generation_metadata(run, manifest, projection, ALIAS, parameters, gradients)
    for raw in manifest.splitlines():
        row = json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=no_constant)
        expected = 'images/{:012d}.png'.format(row['image_id'])
        require(row['output_file'] == expected and type(row['output_bytes']) is int,
            'Generated PNG location or byte count differs')
        image = child(root / 'attack', expected)
        require(image.stat().st_size == row['output_bytes'], 'Generated PNG byte count changed')
        generated_read.bytes(image, row['output_sha256'])
    generated_read.unchanged()
    times['generation_readback_seconds'] = time.monotonic() - started
    for position, call in enumerate(prepared['evaluations']):
        target = call['target_id']
        leaf = child(registry.root, call['output_dir'].relative_to(registry.root))
        require(not leaf.exists(), 'Do not resume or overwrite a target evaluation')
        ownership()
        progress(phase='target_evaluation', target=target, target_index=position + 1,
            targets_completed=position, images_completed=0, images_total=projection['images'])
        started = time.monotonic()
        try:
            evaluated = run_evaluation(registry, **call, progress_callback=callback('target_evaluation', 100))
            require(Path(evaluated) == leaf, 'Evaluator returned another output root')
        finally:
            _cleanup()
        times[target + '_evaluation_seconds'] = time.monotonic() - started
        reference = _reference(leaf / 'metrics.json')
        read = Evidence()
        metadata = bind_target_to_generation(read, generation_reference, reference,
            projection, target, ALIAS, parameters, gradients)
        record = read.json(leaf / 'metrics.json', reference['sha256'])
        archive = record['predictions_artifact']
        require(archive['status'] == 'verified_lossless_archive' and archive['format'] == 'gzip'
            and archive['archive_file'] == 'predictions.json.gz'
            and archive['uncompressed_sha256'] == record['predictions_sha256'],
            'Target prediction archive identity differs')
        archive_path = plain(leaf / archive['archive_file'])
        require(archive_path.stat().st_size == archive['archive_bytes'], 'Prediction archive size changed')
        read.bytes(archive_path, archive['archive_sha256'])
        read.unchanged()
        target_readers.append(read)
        targets.append(dict(target=target, record=reference,
            prediction_archive=dict(file=str(archive_path), sha256=archive['archive_sha256']), metadata_check=metadata))
        progress(targets_completed=position + 1)
    require([row['target'] for row in targets] == group['targets'], 'Worker omitted or reordered targets')
    ownership()
    assets.unchanged()
    generated_read.unchanged()
    for target_read in target_readers:
        target_read.unchanged()
    result = dict(status='generated_and_evaluated_pending_independent_acceptance',
        group_id=group['group_id'], runtime_reference=descriptor['runtime'],
        admission_reference=descriptor['admission'], scope_sha256=bound['admission']['scope']['scope_sha256'],
        input_binding_sha256=prepared['input_binding_sha256'],
        execution_input_binding_sha256=projection['binding_sha256'], generation=generation_reference,
        targets=targets, generation_metadata=generation_metadata,
        generated_payload_bytes_verified=True, generated_pixels_independently_verified=False,
        images=projection['images'], diagnostic_only=limit is not None,
        phase_seconds=times, payload_policy='keep_all_until_independent_acceptance',
        independent_result_acceptance=False, scientific_acceptance=False)
    atomic_json(root / 'worker_group.json', result)
    return result


def run_worker(registry, request_reference):
    from tools.audit_vcsf_cpu_analysis import Evidence, plain
    from tools.vcsf_final_followup_bound_dispatch import load_bound_execution_group

    read = Evidence()
    path = plain(Path(request_reference['file']))
    request = read.json(path, request_reference['sha256'])
    install_parent_death_guard(request['coordinator_pid'])
    require(isinstance(request['group_ids'], list) and request['group_ids'], 'Do not launch an empty worker lane')
    output = child(registry.root, request['execution_root'])
    scope_runtime = read.json(Path(request['runtime_reference']['file']), request['runtime_reference']['sha256'])
    limit = scope_runtime['scope']['max_images']
    descriptor = dict(mode=MODE, execution_alias=ALIAS, runtime=request['runtime_reference'],
        admission=request['admission_reference'], execution_root=request['execution_root'],
        worker_request=request_reference, run_binding_sha256=request['run_binding_sha256'], group_id=request['group_ids'][0])
    setattr(sys.modules['__main__'], '_FOLLOWUP_WORKER_REQUEST_SHA256', request_reference['sha256'])
    bound = load_bound_execution_group(read, registry, descriptor['runtime'], descriptor['admission'],
        descriptor['group_id'], str(output / 'groups' / '{:06d}'.format(descriptor['group_id'])), 'cuda:0', max_images=limit)
    verify_worker_request(read, registry, descriptor, bound, output)
    directory = path.parent
    require(not any((directory / name).exists() or (directory / name).is_symlink() for name in
        ('execution_state.json', 'completion.json', 'failure.json')), 'Do not restart an existing worker')
    state = dict(status='preparing', completed_groups=0, completed_evaluations=0,
        group_ids=request['group_ids'], worker_slot=request['worker_slot'], current=None,
        failed_records=0, request_reference=request_reference, started_at=_now())
    completed, owner, owned_group = [], None, None

    def claim_group(path):
        nonlocal owned_group
        require(owned_group is None, 'Worker cannot claim two current groups')
        expected = output / 'groups' / '{:06d}'.format(descriptor['group_id'])
        require(path == expected, 'Worker created a different group directory')
        info = path.stat()
        owned_group = (path, info.st_dev, info.st_ino)

    def progress(**fields):
        current = dict(state['current'] or {})
        current.update(fields)
        state.update(current=current, updated_at=_now())
        atomic_json(directory / 'execution_state.json', state)

    def ownership():
        verify_coordinator(request['coordinator_pid'])
        require(process_start_ticks(request['coordinator_pid']) == request['coordinator_start_ticks'],
            'Coordinator identity changed during execution')
        owner.verify()

    try:
        for group_id in request['group_ids']:
            owned_group = None
            descriptor = dict(descriptor, group_id=group_id)
            state['current'] = dict(group_id=group_id, phase='preflight')
            if group_id != request['group_ids'][0]:
                bound = load_bound_execution_group(read, registry, descriptor['runtime'], descriptor['admission'],
                    group_id, str(output / 'groups' / '{:06d}'.format(group_id)), 'cuda:0', max_images=limit)
            group_root = child(registry.root, output.relative_to(registry.root) / 'groups' / '{:06d}'.format(group_id))
            require(not group_root.exists(), 'Do not resume or overwrite an existing group')
            verify_environment(bound['prepared']['projected_inputs'])
            progress(phase='current_asset_readback', group_id=group_id)
            assets = snapshot_assets(execution_projection(bound['prepared']['projected_inputs'], limit))
            read_execution_admission(read, registry, bound['catalogue'], bound['admission']['scope'],
                descriptor['runtime'], descriptor['admission'])
            verify_worker_request(read, registry, descriptor, bound, output)
            if owner is None:
                owner = RegisteredGpuOwner(request)
                setattr(sys.modules['__main__'], '_FOLLOWUP_GPU_OWNER', owner)
                state['gpu_context_registration'] = owner.receipt
            state['status'] = 'running'
            result = execute_group(registry, bound, descriptor, ownership, progress, assets,
                on_group_created=claim_group)
            completed.append(result)
            state.update(completed_groups=len(completed),
                completed_evaluations=sum(len(row['targets']) for row in completed), current=None)
            atomic_json(directory / 'completed_groups.json', completed)
            progress(phase='between_groups')
        ownership()
        state.update(status='finished_pending_independent_acceptance', current=None, finished_at=_now())
        atomic_json(directory / 'execution_state.json', state)
        atomic_json(directory / 'completion.json', dict(state, groups=completed,
            independent_result_acceptance=False, scientific_acceptance=False))
        return state
    except BaseException as exc:
        state.update(status='failed', failed_records=1, finished_at=_now(), error_type=type(exc).__name__)
        atomic_json(directory / 'execution_state.json', state)
        if not (directory / 'failure.json').exists():
            atomic_json(directory / 'failure.json', state)
        if owned_group is not None:
            path, device, inode = owned_group
            try:
                failure = child(registry.root, (path / 'failure.json').relative_to(registry.root))
                info = path.stat()
            except (OSError, RuntimeError):
                failure = None
            if failure is not None and (info.st_dev, info.st_ino) == (device, inode) and not failure.exists():
                atomic_json(failure, dict(state, group_id=descriptor['group_id']))
        raise
