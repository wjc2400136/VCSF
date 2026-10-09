"""Owned radius execution overlay; the frozen attack implementation is inherited."""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
import os

from ..registry import AttackSpec, Registry
from ..runners.vcsf_research_plan import canonical_hash
from .factory import ATTACK_TYPES
from .vcsf_final_radius_isolated import build_radius_candidate, make_radius_config

MODE = 'final_budget_execution'
ALIAS = 'vcsf_final_budget_isolated'


def background_parameters(registry, parameters):
    result = deepcopy(parameters)
    spec = registry.protocols['vcsf_final_budget_refresh']
    result['eps'] = float(Fraction(spec['reference_epsilon']))
    if canonical_hash(result) != spec['parameters_sha256']:
        raise ValueError('Radius invocation changes the frozen background')
    return result


def build_final_budget(adapter, parameters):
    registry = Registry()
    return build_radius_candidate(adapter, registry,
        background_parameters(registry, parameters), parameters['eps'])


def validate_invocation(registry, binding, *, dataset_id, split, source_id,
                        attack_id, seed, max_images, parameter_overrides,
                        budget_profile, image_ids, seed_offsets, input_transform,
                        output_dir, expected_output, device):
    if ALIAS in registry.attacks or ALIAS in ATTACK_TYPES:
        raise ValueError('Radius overlay must not enter the global method registry')
    if (dataset_id != 'coco' or split != 'val' or source_id != binding['source']
            or attack_id != ALIAS or type(seed) is not int or seed != binding['seed']
            or device != 'cuda:0' or input_transform is not None):
        raise ValueError('Radius invocation identity or transform differs')
    if (type(max_images) is not type(binding['max_images'])
            or max_images != binding['max_images'] or budget_profile != binding['budget_profile']):
        raise ValueError('Radius invocation scope or budget differs')
    if (not isinstance(image_ids, (tuple, list))
            or any(type(i) is not int for i in image_ids)
            or list(image_ids) != binding['image_ids']):
        raise ValueError('Radius invocation image population differs')
    expected = {row['image_id']: row['offset'] for row in binding['seed_mapping']}
    if (not isinstance(seed_offsets, dict) or seed_offsets != expected
            or any(type(k) is not int or type(v) is not int for k, v in seed_offsets.items())):
        raise ValueError('Radius invocation seed offsets differ')
    if (not isinstance(parameter_overrides, dict)
            or canonical_hash(parameter_overrides) != binding['parameters_sha256']
            or canonical_hash(binding['parameters']) != binding['parameters_sha256']):
        raise ValueError('Radius invocation parameters differ')
    parsed = make_radius_config(registry, background_parameters(registry, parameter_overrides),
                                binding['epsilon'])
    if canonical_hash(asdict(parsed)) != binding['parameters_sha256']:
        raise ValueError('Radius label differs from the bound numerical parameters')
    if output_dir is None or Path(output_dir).resolve() != Path(expected_output).resolve():
        raise ValueError('Radius output escaped the assigned group')


def resolve_final_budget(registry, descriptor, *, run_metadata, output_dir,
                         device, gpu_owner, **arguments):
    from ..io import file_digest
    from ..runners.training_pair_processes import process_identity
    from ..runners.vcsf_final_budget import read_permit
    from ..runners.vcsf_oblivious_admission import bound_json
    from ..runners.vcsf_structure_workers import verify_reservation
    from ..runners.vcsf_gpu_context_owner import RegisteredGpuOwner
    fields = {'mode', 'execution_alias', 'worker_request', 'worker_request_sha256', 'group_index'}
    if (not isinstance(descriptor, dict) or set(descriptor) != fields
            or descriptor['mode'] != MODE or descriptor['execution_alias'] != ALIAS):
        raise ValueError('Unsupported radius worker descriptor')
    request = bound_json(descriptor['worker_request'], descriptor['worker_request_sha256'])
    ref = request['permit']
    permit, bindings = read_permit(registry, ref['file'], ref['sha256'])
    index = descriptor['group_index']
    selected = [row for row in bindings if row['group_index'] == index]
    if type(index) is not int or len(selected) != 1 or index not in request['group_indices']:
        raise ValueError('Radius group is not assigned to this worker')
    binding = selected[0]
    if (request['device'] != binding['device']
            or permit['gpu_uuids'][request['device']] != request['gpu_uuid']):
        raise ValueError('Radius worker device differs')
    own, parent = process_identity(os.getpid()), process_identity(os.getppid())
    coordinator = request['coordinator']
    if (parent['pid'] != coordinator['pid'] or parent['start_ticks'] != coordinator['start_ticks']
            or own['process_group'] != own['pid'] or own['session_id'] != own['pid']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != request['gpu_uuid']):
        raise ValueError('Radius worker process ownership differs')
    verify_reservation(request)
    if (type(gpu_owner) is not RegisteredGpuOwner
            or canonical_hash(gpu_owner._request) != canonical_hash(request)):
        raise ValueError('Radius GPU context ownership differs')
    gpu_owner.verify()
    output = Path(permit['output']).resolve()
    if output not in Path(descriptor['worker_request']).resolve().parents:
        raise ValueError('Radius request escaped execution root')
    expected_metadata = dict(protocol=binding['protocol'], group_index=index,
        epsilon=binding['epsilon'], permit_sha256=ref['sha256'],
        assets_sha256=permit['assets']['sha256'], diagnostic_only=permit['max_images'] is not None)
    if run_metadata != expected_metadata:
        raise ValueError('Radius run metadata differs')
    validate_invocation(registry, binding, output_dir=output_dir,
        expected_output=output / 'groups' / '{:06d}'.format(index) / 'attack',
        device=device, **arguments)
    from ..data.coco import CocoIndex, select_coco_images
    from ..modeling import checkpoint_path
    assets = bound_json(permit['assets']['file'], permit['assets']['sha256'])
    data = CocoIndex(registry.dataset('coco'), 'val')
    if (str(data.annotation_path) != assets['annotation']['file']
            or file_digest(data.annotation_path) != assets['annotation']['sha256']):
        raise ValueError('Radius annotation changed')
    clean = {row['image_id']: row['input'] for row in assets['clean']}
    for image in select_coco_images(data.images, image_ids=binding['image_ids']):
        path = data.image_path(image)
        expected = clean[image['id']]
        if (str(path) != expected['file'] or path.stat().st_size != expected['bytes']
                or file_digest(path) != expected['sha256']):
            raise ValueError('Radius clean image changed before generation')
    checkpoint = checkpoint_path(registry.model(binding['source']), registry.dataset('coco')).resolve()
    expected = assets['checkpoints'][binding['source']]
    if str(checkpoint) != expected['file'] or file_digest(checkpoint) != expected['sha256']:
        raise ValueError('Radius source checkpoint changed before generation')
    return AttackSpec(id=ALIAS, display_name='VCSF', executor='native',
        parameters=deepcopy(binding['parameters']), metadata=dict(protocol=binding['protocol'],
            group_index=index, epsilon=binding['epsilon'], stress_only=binding['stress_only'],
            parameters_sha256=binding['parameters_sha256'],
            fidelity_status='frozen_background_registered_radius_pending_result_acceptance',
            fidelity_basis='inherited_common_anchor_numerical_methods',
            semantic_contract='registered_radius_sensitivity_not_independent_confirmation',
            formal_AP_eligible=False, global_registry_mutated=False, global_factory_mutated=False))
