"""Normalize accepted observations through explicit, immutable reuse mappings."""
import hashlib
import json
import math

from ..metrics import COCO_BBOX_METRICS
from .vcsf_prediction_evidence import _require, _same_json, _sha


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def compile_input_slots(plan, semantic, plan_sha256):
    """Keep every registered slot, including NR jobs without any observation."""
    _require(_sha(plan_sha256) and semantic.get('plan_sha256') == plan_sha256
        and plan.get('protocol') == 'vcsf_fullval_retrospective_ablation'
        and plan.get('independent_confirmation') is False and plan.get('runner_armed') is False,
        'Wrong analysis plan identity or scientific scope')
    _require(semantic.get('status') == 'adjudicated_historical_semantic_reuse'
        and semantic.get('errors') == [] and semantic.get('scientific_promotion') is False
        and semantic.get('efficacy_runner_armed') is False
        and semantic.get('current_implementation_physical_cost_inheritance') is False,
        'The semantic efficacy-only reuse decision is not applicable')
    groups, jobs = plan['groups'], semantic['jobs']
    sources, targets = plan['sources'], plan['targets']
    _require(sources and targets and len(sources) == len(set(sources))
        and len(targets) == len(set(targets)) and set(sources) <= set(targets), 'Invalid canonical panel')
    _require(all(type(group['group_id']) is int for group in groups)
        and [group['group_id'] for group in groups] == list(range(1, len(groups) + 1))
        and len({(g['source'], g['variant'], g['seed']) for g in groups}) == len(groups),
        'Planned group identity/order differs')
    _require(type(semantic.get('registered_groups')) is int and semantic['registered_groups'] == len(groups)
        and len(jobs) == len(groups) and all(type(job['group_id']) is int for job in jobs)
        and len({job['group_id'] for job in jobs}) == len(groups)
        and semantic.get('partition_sha256') == canonical_hash(jobs), 'Semantic job partition changed')
    indexed = {job['group_id']: job for job in jobs}
    _require(set(indexed) == {group['group_id'] for group in groups}, 'Semantic partition has missing or extra jobs')
    identity_fields = ('group_id', 'source', 'variant', 'seed', 'images', 'targets', 'blackbox_targets', 'parameters_sha256')
    slots, counts = [], {'reuse_frozen_reference': 0, 'reuse_historical_semantic_equivalent': 0, 'new_registered_measurement': 0}
    for group in groups:
        job = indexed[group['group_id']]
        _require(group['source'] in sources and group['targets'] == targets
            and type(group['seed']) is int and type(group['images']) is int and group['images'] > 0
            and group['blackbox_targets'] == [target for target in targets if target != group['source']]
            and group['parameters_sha256'] == canonical_hash(group['parameters'])
            and all(_same_json(job.get(field), group[field]) for field in identity_fields),
            'Semantic mapping no longer identifies the planned group')
        disposition = job['disposition']
        _require(disposition in counts, 'Unknown input disposition')
        counts[disposition] += 1
        slot = {field: group[field] for field in identity_fields}
        slot.update(disposition=disposition, original_group=None, original_parameters_sha256=None,
            original_variant=None, original_run_sha256=None, original_attack=None,
            observation_status='NR', physical_cost_inheritance=False, trajectory_identity_claim=False)
        if disposition == 'reuse_frozen_reference':
            old = job['original_reference']
            _require(job.get('historical_observation') is None and old.get('status') == 'verified_frozen_reference_reuse'
                and _same_json(old.get('group_id'), group['group_id']) and old.get('variant') == group['variant'] == 'full'
                and _same_json(old.get('seed'), group['seed']) and old['seed'] == 42
                and old.get('source') == group['source'] and old.get('research_parameters_sha256') == group['parameters_sha256']
                and _same_json(old.get('images'), group['images']) and old.get('evaluations') == len(targets)
                and old.get('blackbox_cells') == len(targets) - 1 and _sha(old.get('old_parameters_sha256')),
                'Frozen-reference origin mapping differs')
            slot.update(original_group=old['old_group'], original_parameters_sha256=old['old_parameters_sha256'],
                original_attack='vcsf', observation_status='qualified_pending_current_metadata_binding')
        elif disposition == 'reuse_historical_semantic_equivalent':
            old, mapping = job['historical_observation'], job['semantic_mapping']
            _require(job.get('original_reference') is None and job.get('historical_implementation_inheritance') is True
                and job.get('inherited_evidence_scope') == 'accepted_efficacy_only_original_provenance_preserved'
                and all(job.get(field) is False for field in ('current_runtime_inheritance', 'current_memory_inheritance',
                    'scalar_diagnostic_inheritance', 'trajectory_identity_claim')),
                'Historical evidence cannot inherit new physical-cost or trajectory claims')
            _require(_same_json(old.get('research_group_id'), group['group_id']) and old.get('source') == group['source']
                and old.get('research_variant') == group['variant'] and old.get('research_parameters_sha256') == group['parameters_sha256']
                and _same_json(old.get('seed'), group['seed']) and old['seed'] == 42
                and _same_json(old.get('images'), group['images']) and old.get('attack_evaluations') == len(targets)
                and old.get('historical_observation_scope_verified') is True
                and old.get('generic_implementation_result_inheritance') is False,
                'Historical original observation differs')
            _require(all(_same_json(mapping.get(field), group[field]) for field in
                ('group_id', 'source', 'variant', 'seed', 'parameters_sha256'))
                and mapping.get('source_semantics_reviewed') is True
                and mapping.get('historical_run_sha256') == old.get('historical_run_sha256')
                and mapping.get('historical_parameters_sha256') == old.get('historical_parameters_sha256')
                and _sha(old.get('historical_run_sha256')) and _sha(old.get('historical_parameters_sha256')),
                'Historical semantic adjudication is not bound to this observation')
            slot.update(original_group=old['historical_group'], original_parameters_sha256=old['historical_parameters_sha256'],
                original_variant=old['historical_variant'], original_run_sha256=old['historical_run_sha256'],
                original_attack='vcsf', observation_status='qualified_pending_current_metadata_binding')
        else:
            _require(job.get('original_reference') is None and job.get('historical_observation') is None,
                'An NR job cannot silently discard an existing origin mapping')
        if disposition != 'new_registered_measurement':
            _require(type(slot['original_group']) is str and slot['original_group'], 'Missing original group location')
        slots.append(slot)
    _require(all(type(semantic.get(key)) is int for key in
        ('frozen_reference_groups', 'historical_semantic_reuse_groups', 'qualified_reuse_groups', 'new_groups'))
        and semantic.get('frozen_reference_groups') == counts['reuse_frozen_reference']
        and semantic.get('historical_semantic_reuse_groups') == counts['reuse_historical_semantic_equivalent']
        and semantic.get('qualified_reuse_groups') == counts['reuse_frozen_reference'] + counts['reuse_historical_semantic_equivalent']
        and semantic.get('new_groups') == counts['new_registered_measurement'], 'Semantic partition totals conflict')
    return slots


def normalize_original_cell(slot, row, *, input_evidence_sha256, image_ids_sha256):
    """Attach current logical identity while explicitly preserving original identity."""
    _require(slot['observation_status'] == 'qualified_pending_current_metadata_binding'
        and _sha(input_evidence_sha256) and _sha(image_ids_sha256), 'No qualified input evidence for this cell')
    _require(row.get('status') == 'complete' and row.get('failures') == []
        and row.get('dataset') == 'coco' and row.get('split') == 'val'
        and row.get('source') == slot['source'] and row.get('target') in slot['targets']
        and row.get('attack') == slot['original_attack'] and _same_json(row.get('images'), slot['images'])
        and row.get('adversarial_run') == slot['original_group']
        and row.get('parameters_sha256') == slot['original_parameters_sha256'],
        'Original cell does not match the qualified mapping')
    if slot['original_variant'] is not None:
        _require(row.get('variant') == slot['original_variant'], 'Historical variant was relabeled')
    _require(row.get('evaluated_image_ids_sha256') == row.get('expected_image_ids_sha256') == image_ids_sha256
        and row.get('evaluated_image_ids_match_expected') is True, 'Original cell image panel changed')
    commit = row.get('code_commit')
    _require(type(commit) is str and len(commit) == 40 and all(c in '0123456789abcdef' for c in commit),
        'Original cell Git provenance is missing or invalid')
    metrics = row.get('metrics')
    _require(isinstance(metrics, dict) and all(key in metrics and type(metrics[key]) in (int, float)
        and math.isfinite(metrics[key]) and (metrics[key] == -1 or 0 <= metrics[key] <= 1) for key in COCO_BBOX_METRICS)
        and 0 <= metrics['bbox_mAP'] <= 1, 'Original full COCO metrics are incomplete or invalid')
    result = {key: slot[key] for key in ('group_id', 'source', 'variant', 'seed', 'parameters_sha256', 'images')}
    result.update(target=row['target'], status='complete', failures=[], bbox_mAP=float(metrics['bbox_mAP']),
        metrics=dict(metrics), input_evidence_sha256=input_evidence_sha256, image_ids_sha256=image_ids_sha256,
        original_parameters_sha256=row['parameters_sha256'], original_record_sha256=canonical_hash(row),
        original_code_commit=row['code_commit'], reuse_disposition=slot['disposition'],
        physical_cost_inheritance=False, trajectory_identity_claim=False)
    return result
