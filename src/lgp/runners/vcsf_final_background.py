"""Compile the fixed final-background request without execution or reuse admission."""
from copy import deepcopy
from dataclasses import asdict
import re

from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
from ..io import file_digest
from .vcsf_operator_execution_contract import balanced_lanes, read_bound
from .vcsf_research_plan import canonical_hash, require


PROTOCOL = 'vcsf_final_background_validation'


def planned_work(parameters):
    """Logical schedule only; reference slots are not separate full detector calls."""
    detector = int(parameters['initialization'] == 'detector')
    feature = parameters['iterations'] - detector
    terms = parameters['levels_per_stage'] * (2 if parameters['surface'] == 'cross_stage' else 1)
    return dict(logical_gradient_updates=parameters['iterations'],
        detector_backward_updates=detector, partial_feature_backward_updates=feature,
        clean_reference_image_slots=feature, differentiable_feature_image_slots=feature,
        feature_terms_per_update=terms, feature_terms=terms * feature,
        physical_detector_backward_equivalents=None, measured_seconds=None,
        measured_peak_memory_bytes=None)


def prepare_final_background_design(registry, devices, max_images=None):
    require(isinstance(devices, (list, tuple)) and len(devices) in (1, 2)
        and all(isinstance(d, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) for d in devices)
        and len(set(devices)) == len(devices), 'Select one or two distinct physical CUDA IDs')
    require(max_images is None or type(max_images) is int and max_images == 1,
        'Only full-scope preparation or a separately labelled one-image diagnostic is allowed')
    protocol = registry.protocols[PROTOCOL]
    require(protocol['preparation_only'] is True and protocol['execution_admitted'] is False,
        'This compiler cannot consume an armed protocol')
    request = read_bound(registry.root / protocol['request_file'], protocol['request_sha256'])
    study = registry.ablation_studies[protocol['ablation_study']]
    scope, decision = request['final_background_request'], request['decision']
    require(all(decision[k] is False for k in ('final_scientific_freeze',
        'public_implementation_promoted', 'formal_execution_authorized', 'formal_result_accepted')),
        'The request must remain a working nomination, not final acceptance')
    require(scope['formal_execution_authorized'] is False,
        'A preparation request cannot authorize execution')
    targets = list(registry.target_ids())
    require(targets == scope['targets'] and scope['source'] in targets,
        'Canonical target population differs')
    for name, digest in request['execution_identity']['numeric_source_sha256'].items():
        path = registry.root / name
        require(path == path.resolve() and registry.root in path.parents
            and file_digest(path) == digest, 'Registered numerical source changed: ' + name)
    controls = {row['id']: row for row in scope['controls']}
    require(len(controls) == len(scope['controls']), 'Duplicate declared control')
    groups, identities = [], set()
    for index, variant in enumerate(study['row_order'], 1):
        raw = deepcopy(study['base_parameters'])
        raw.update(study['variants'][variant]['parameters'])
        config = VCSFCommonAnchorConfig.from_mapping(raw)
        config.validate()
        parameters = asdict(config)
        parameters['feature_levels'] = list(parameters['feature_levels'])
        expected = decision['parameters'] if variant == study['anchor'] else controls[variant]['parameters']
        digest = canonical_hash(parameters)
        require(digest == canonical_hash(expected) and digest not in identities,
            'Changed or duplicate final-background configuration')
        work = planned_work(parameters)
        if variant != study['anchor']:
            control = controls[variant]
            require(digest == control['parameters_sha256'], 'Control parameter identity differs')
            changes = {key: dict(A10=decision['parameters'][key], control=value)
                for key, value in parameters.items() if value != decision['parameters'][key]}
            require(changes == control['exact_changes'], 'Control changes unregistered factors')
            for declared, computed in (
                ('planned_logical_gradients', 'logical_gradient_updates'),
                ('planned_detector_backward_updates', 'detector_backward_updates'),
                ('planned_partial_feature_backward_updates', 'partial_feature_backward_updates'),
                ('planned_clean_reference_image_slots', 'clean_reference_image_slots'),
                ('planned_feature_terms', 'feature_terms')):
                require(control[declared] == work[computed], 'Declared schedule differs from parameters')
            question, interpretation = control['question'], control['interpretation']
        else:
            require(digest == study['anchor_parameters_sha256'], 'Working candidate changed')
            question = 'Registered working candidate; exact original-result reuse is unresolved.'
            interpretation = 'Descriptive selection evidence is not final-method acceptance.'
        groups.append(dict(group_id=index, variant=variant, source=scope['source'],
            dataset=scope['dataset'], split=scope['split'], seed=scope['seed'],
            seed_schedule=protocol['seed_schedule'], images=max_images or scope['images_per_group'],
            targets=targets[:], parameters=parameters, parameters_sha256=digest,
            planned_work=work, question=question, interpretation=interpretation,
            diagnostic_only=max_images is not None, max_images=max_images,
            potential_reuse=variant == study['anchor'] and max_images is None,
            reuse_accepted=False))
        identities.add(digest)
    scheduled = [group for group in groups if not group['potential_reuse']]
    lanes = balanced_lanes(scheduled, protocol['sources'], devices)
    formal_slots = [dict(variant=group['variant'], target=target, status='NR', metrics=None)
        for group in groups for target in targets]
    diagnostic_slots = [dict(slot, images=max_images, diagnostic_only=True)
        for slot in formal_slots] if max_images is not None else []
    require(len(groups) == scope['logical_groups'] and len(formal_slots) == scope['logical_target_cells'],
        'Final-background logical coverage changed')
    if max_images is None:
        require(len(scheduled) == scope['new_groups_if_anchor_reuse_qualifies']
            and sum(group['images'] for group in scheduled) == scope['new_images_if_anchor_reuse_qualifies']
            and len(scheduled) * len(targets) == scope['new_target_evaluations_if_anchor_reuse_qualifies'],
            'Conditional new-work count changed')
    return dict(record_type='vcsf_final_background_prospective_design', protocol=PROTOCOL,
        status='prepared_pending_input_reuse_runtime_and_independent_admission',
        request_sha256=protocol['request_sha256'], request_content_sha256=canonical_hash(request),
        working_candidate=study['anchor'], candidate_parameters_sha256=study['anchor_parameters_sha256'],
        groups=groups, metrics=scope['metrics'], canonical_targets=targets,
        blackbox_targets=[target for target in targets if target != scope['source']],
        requested_devices=list(devices), device_availability_verified=False,
        prospective_lanes=[[group['group_id'] for group in lane] for lane in lanes],
        unresolved_reuse_group_ids=[group['group_id'] for group in groups if group['potential_reuse']],
        scheduled_group_ids=[group['group_id'] for group in scheduled],
        prospective_new_images=sum(group['images'] for group in scheduled),
        prospective_new_target_evaluations=len(scheduled) * len(targets),
        logical_groups=len(groups), logical_target_cells=len(formal_slots),
        formal_result_slots=formal_slots, diagnostic_result_slots=diagnostic_slots,
        diagnostic_only=max_images is not None, max_images=max_images,
        full_scope_reuse_partition_resolved=False,
        historical_anchor_run_reference=deepcopy(request['input_bindings']['original_A10_run']),
        anchor_reuse_failure_policy='stop_and_record_not_automatic_regeneration',
        numerical_source_sha256=deepcopy(request['execution_identity']['numeric_source_sha256']),
        payload_policy=scope['payload_policy'], independent_complete_jobs=True,
        device_count_changes_method=False, input_provenance_verified=False,
        runtime_qualified=False, reuse_accepted=False, runner_armed=False,
        formal_execution_admission=False, scientific_acceptance=False,
        final_method_promoted=False, model_calls=0, AP_replays=0)
