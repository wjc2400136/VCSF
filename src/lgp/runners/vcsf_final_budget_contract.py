"""Registered full-split radius task topology; grants no execution or reuse."""
from copy import deepcopy
from fractions import Fraction
import re

from .vcsf_final_training_state_contract import ids_digest
from .vcsf_research_plan import canonical_hash

PROTOCOL = 'vcsf_final_budget_refresh'


def build_plan(registry, devices):
    spec = registry.protocols[PROTOCOL]
    reference = registry.protocols['vcsf_final_oblivious_refresh']
    if (not isinstance(devices, (list, tuple)) or len(devices) not in (1, 2)
            or len(set(devices)) != len(devices)
            or any(type(d) is not str or not re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d)
                   for d in devices)):
        raise ValueError('Select one or two unique explicit CUDA devices')
    if (spec['sources'] != reference['sources']
            or spec['parameters_sha256'] != reference['parameters_sha256']
            or spec['images'] != reference['full_images']
            or spec['seed'] != reference['seed'] or spec['targets'] != 'all'):
        raise ValueError('Radius reference population or frozen identity differs')
    radii = spec['epsilon_order']
    if (not isinstance(radii, list) or not radii or len(set(radii)) != len(radii)
            or any(type(r) is not str or Fraction(r) <= 0 for r in radii)
            or len({Fraction(r) for r in radii}) != len(radii)
            or spec['reference_epsilon'] not in radii
            or not set(spec['stress_only']).issubset(radii)):
        raise ValueError('Invalid registered radius axis')
    groups, jobs = [], []
    for source_index, source in enumerate(spec['sources']):
        for radius in radii:
            reuse = radius == spec['reference_epsilon']
            group = dict(index=len(groups), source=source, epsilon=radius,
                images=spec['images'], device=devices[source_index % len(devices)],
                action='qualified_reuse_required' if reuse else 'generate_and_evaluate',
                stress_only=radius in spec['stress_only'])
            groups.append(group)
            jobs.extend(dict(group_index=group['index'], source=source, epsilon=radius,
                target=target, images=spec['images'], device=group['device'],
                action=group['action']) for target in registry.paper_order)
    generated = [g for g in groups if g['action'] == 'generate_and_evaluate']
    new_jobs = [j for j in jobs if j['action'] == 'generate_and_evaluate']
    reuse_jobs = [j for j in jobs if j['action'] == 'qualified_reuse_required']
    return dict(protocol=PROTOCOL, dataset=spec['dataset'], split=spec['split'],
        devices=list(devices), parameters_sha256=spec['parameters_sha256'],
        budget_profile=spec['budget_profile'], seed=spec['seed'],
        seed_schedule='base_seed_plus_full_numeric_image_position',
        full_image_ids_sha256=reference['full_image_ids_sha256'],
        annotation_sha256=reference['annotation_sha256'],
        epsilon_order=deepcopy(radii), reference_epsilon=spec['reference_epsilon'],
        step_size=spec['step_size'], groups=groups, canonical_jobs=jobs,
        generation_groups=generated, generated_evaluations=new_jobs,
        reference_reuse=reuse_jobs, new_generation_images=sum(g['images'] for g in generated),
        new_target_image_evaluations=sum(j['images'] for j in new_jobs),
        clean_jobs=0, baseline_jobs=0, formal_scope=True,
        reuse_qualified=False, formal_execution_admitted=False)


def generation_binding(registry, plan, prepared, group_index, max_images=None):
    """Bind preauthenticated full-source inputs; caller must verify their file hash."""
    from dataclasses import asdict
    from ..attacks.vcsf_final_radius_isolated import make_radius_config

    if plan != build_plan(registry, plan['devices']):
        raise ValueError('Budget plan differs from canonical topology')
    if type(group_index) is not int or not 0 <= group_index < len(plan['groups']):
        raise ValueError('Invalid canonical budget group')
    group = plan['groups'][group_index]
    if group['action'] != 'generate_and_evaluate':
        raise ValueError('Reference radius requires qualified reuse, not generation')
    paired = registry.protocols['vcsf_final_oblivious_refresh']
    inputs = prepared['inputs']
    source = group['source']
    if (prepared['source'] != source or inputs['source'] != source
            or type(prepared['seed']) is not int or prepared['seed'] != plan['seed']
            or prepared['parameters_sha256'] != plan['parameters_sha256']
            or canonical_hash(prepared['parameters']) != plan['parameters_sha256']
            or prepared['payload_run_sha256'] != paired['payload_bindings'][source]['payload_run_sha256']):
        raise ValueError('Prepared source or frozen parameters differ')
    full_ids = inputs['image_ids']
    if (not isinstance(full_ids, list) or len(full_ids) != group['images']
            or any(type(i) is not int for i in full_ids)
            or full_ids != sorted(set(full_ids))
            or ids_digest(full_ids) != plan['full_image_ids_sha256']):
        raise ValueError('Budget generation requires the exact full image population')
    if max_images is not None and (type(max_images) is not int or not 0 < max_images <= len(full_ids)):
        raise ValueError('Invalid explicit diagnostic limit')
    ids = list(full_ids if max_images is None else full_ids[:max_images])
    parameters = asdict(make_radius_config(registry, prepared['parameters'], group['epsilon']))
    return dict(protocol=PROTOCOL, group_index=group_index, source=source,
        epsilon=group['epsilon'], stress_only=group['stress_only'], device=group['device'],
        image_ids=ids, seed=plan['seed'], seed_mapping=[dict(image_id=i, offset=n,
            attack_seed=plan['seed']+n) for n, i in enumerate(ids)],
        parameters=parameters, parameters_sha256=canonical_hash(parameters),
        frozen_background_sha256=plan['parameters_sha256'], budget_profile=plan['budget_profile'],
        original_payload_run_sha256=prepared['payload_run_sha256'],
        max_images=max_images, formal_scope=max_images is None, execution_admitted=False)
