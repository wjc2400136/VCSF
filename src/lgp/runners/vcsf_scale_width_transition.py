"""Bind the accepted operator stage before the registered remaining widths."""
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import subprocess

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS
from .vcsf_efficacy_contract import read_bound, require
from .vcsf_producer_verification import verify_producer_contract
from .vcsf_research_plan import canonical_hash
from .vcsf_scale_plan import compile_stage_plan
from .vcsf_scale_stage_transition import BoundEvidence, REVIEW_DIMENSIONS, scientific_plan, scope_flag


def verify_operator_scope(audit, cost, audit_hash):
    for key, value in dict(status='independently_verified_operator_point_analysis', errors=[],
            diagnostic=False, max_images=None, groups=5, cells=80, images_per_group=5000, contrasts=6,
            previous_cells_reused=32, new_official_metric_replays=48, all_twelve_metrics_verified=True,
            analysis_pipeline_accepted=True, bootstrap_replicates_computed=0, new_model_calls=0,
            original_roots_modified=False, detailed_cost_acceptance=False, candidate_selected=False,
            significance_claimed=False, scientific_acceptance=False, independent_confirmation=False).items():
        scope_flag(audit, key, value)
    matrix = audit.get('matrix_verification', {})
    for key, value in dict(status='independently_verified_scale_point_arithmetic', groups=5, cells=80,
            images=5000, contrasts=6, excluded_whitebox_cells=5, bootstrap_required=False,
            significance_claimed=False).items():
        scope_flag(matrix, key, value)
    for key, value in dict(status='independently_verified_operator_saved_cost_disclosure', errors=[],
            diagnostic=False, groups=5, images_per_group=5000, contrasts=6,
            saved_cost_disclosure_accepted=True, physical_cost_calibration_accepted=False,
            scientific_acceptance=False, independent_confirmation=False, new_model_calls=0,
            point_acceptance_sha256=audit_hash, execution_plan_sha256=audit['execution_plan_sha256']).items():
        scope_flag(cost, key, value)


def verify_operator_decision(decision, audit, audit_hash, cost, cost_hash, stage_hash):
    for key, value in dict(status='recorded_operator_scale_stage_decision',
            scope='operator_comparison_at_anchor_width', next_stage='remaining_widths',
            previous_analysis_audit_sha256=audit_hash, previous_cost_audit_sha256=cost_hash,
            prepared_stage_sha256=stage_hash, reviewed_analysis_sha256=audit['report_sha256'],
            reviewed_cost_report_sha256=cost['cost_report_sha256'], reviewed_dimensions=list(REVIEW_DIMENSIONS),
            next_stage_configurations_unchanged=True, no_early_efficacy_pruning=True,
            physical_cost_calibration_claimed=False, independent_confirmation=False,
            final_candidate_frozen=False, automatic_promotion=False).items():
        scope_flag(decision, key, value)
    require(decision.get('decision') in ('keep', 'change', 'reject')
        and isinstance(decision.get('rationale'), str) and decision['rationale'].strip(),
        'An operator-stage decision needs an explicit complete-evidence rationale')
    recorded = datetime.fromisoformat(decision['recorded_at'])
    times = [datetime.fromisoformat(item['completed_at']) for item in (audit, cost)]
    require(recorded.tzinfo is not None and all(t.tzinfo is not None and t <= recorded for t in times),
        'Operator decision predates full AP or cost acceptance')


def bind_operator_cost(read, cost, audit, audit_hash):
    path = Path(cost['cost_report_file'])
    report = read.json(path, cost['cost_report_sha256'])
    for key, value in dict(status='operator_saved_cost_pending_independent_acceptance',
            diagnostic=False, images_per_group=5000, point_acceptance_sha256=audit_hash,
            point_plan_sha256=audit['point_plan_sha256'], execution_plan_sha256=audit['execution_plan_sha256'],
            independent_cost_acceptance=False, physical_cost_calibration_accepted=False,
            efficacy_selection=False, independent_confirmation=False, significance_claimed=False,
            new_model_calls=0, bootstrap_replicates_computed=0).items():
        scope_flag(report, key, value)
    require(len(report['groups']) == 5 and len(report['contrasts']) == 6,
        'Accepted operator cost report lost its full scope')
    manifest = read.json(path.parent/'manifest.json')
    require(set(manifest) == {'cost.json', 'cost.csv', 'contrasts.csv', 'cost.tex'}
        and {p.name for p in path.parent.iterdir()} == set(manifest) | {'manifest.json'},
        'Operator cost export inventory changed')
    for name, digest in manifest.items():
        read.bind(path.parent/name, digest)
    return report


def bind_operator_point_stage(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash):
    audit, cost = read_bound(audit_path, audit_hash), read_bound(cost_path, cost_hash)
    verify_operator_scope(audit, cost, audit_hash)
    read, cost_read = BoundEvidence(audit['checked_input_sha256']), BoundEvidence(cost['checked_input_sha256'])
    cost_report = bind_operator_cost(cost_read, cost, audit, audit_hash)
    point = read.json(audit['point_plan_file'], audit['point_plan_sha256'])
    execution = read.json(point['execution_plan_file'], audit['execution_plan_sha256'])
    require(point['execution_plan_sha256'] == audit['execution_plan_sha256']
        and point['new_inputs_sha256'] == audit['new_inputs_sha256']
        and point['diagnostic'] is False and point['max_images'] is None
        and execution['stage'] == 'operators_at_anchor_width'
        and execution['analysis_mode'] == 'point_estimate', 'Not the accepted full operator execution')
    stage = compile_stage_plan(registry, 'remaining_widths', ['cuda:0', 'cuda:1'], analysis_mode='point_estimate')
    require(canonical_hash(execution['groups']) == canonical_hash(stage['previous_stage_groups'])
        and canonical_hash(scientific_plan(execution['prepared_plan']))
            == canonical_hash(scientific_plan(stage['prepared_plan'])),
        'Successor changed the registered operator or width configurations')
    from .vcsf_scale_execution_contract import UNCHANGED_SCIENTIFIC_FILES
    for name in UNCHANGED_SCIENTIFIC_FILES + ('src/lgp/attacks/vcsf_scale_isolated.py',):
        require(file_digest(registry.root/name) == execution['runtime_sha256'].get(name),
            'Previous scientific implementation changed: '+name)
    run = Path(point['run_root'])
    binding = read.json(run/'plan_binding.json')
    admissions = [p for p, h in read.mapping.items()
        if Path(p).name == 'admission.json' and h == binding['admission_sha256']]
    require(len(admissions) == 1 and binding['execution_plan_sha256'] == audit['execution_plan_sha256']
        and binding['max_images'] is None, 'Missing unique original formal producer admission')
    admission = read.bind(admissions[0], binding['admission_sha256'])
    proof = verify_producer_contract(Path(point['execution_worktree']), Path(point['execution_plan_file']),
        audit['execution_plan_sha256'], admission, binding['admission_sha256'], None)
    root = Path(audit['analysis_root'])
    completion = read.json(root/'completion.json', audit['analysis_completion_sha256'])
    for key, value in dict(status='complete_operator_points_pending_independent_acceptance', errors=[],
            point_plan_sha256=audit['point_plan_sha256'], execution_plan_sha256=audit['execution_plan_sha256'],
            diagnostic=False, max_images=None, new_cells=48, observation_cells=80, contrasts=6,
            previous_cells_reused=32, new_model_calls=0, bootstrap_replicates_computed=0).items():
        scope_flag(completion, key, value)
    inventory = read.json(root/'artifact_manifest.json', completion['artifact_manifest_sha256'])
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in inventory)
        and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(inventory) | {'artifact_manifest.json', 'completion.json'},
        'Accepted operator point output inventory changed')
    for name, digest in inventory.items():
        read.bind(root/name, digest)
    report = read.json(root/'reports/analysis.json', audit['report_sha256'])
    groups, cells = point['contract']['plan']['groups'], report['normalized_cells']
    require(report['status'] == 'point_matrix_pending_independent_acceptance'
        and report['groups'] == 5 and report['cells'] == 80 and len(groups) == 5 and len(cells) == 80
        and len(report['contrasts']) == 6
        and report['targets'] == stage['targets'] and report['sources'] == stage['sources']
        and cells == point['previous_cells'] + [c for g in point['groups'] for c in g['cells']]
        and [g['group_id'] for g in groups] == [29, 2, 3, 4, 5]
        and [g['variant'] for g in groups] == ['levels_two', 'A_h0', 'B_h2', 'C_h2', 'D_h2'],
        'Operator report lost original complete group identities')
    refs = deepcopy(point['contract']['previous_point_replay_refs'])
    require(len(refs) == 32, 'Original first-pair replay references are incomplete')
    for index, cell in enumerate(cells):
        group, target = groups[index//16], stage['targets'][index%16]
        require(cell['target'] == target and cell['status'] == 'complete' and cell['failures'] == []
            and all(type(cell[k]) is type(group[k]) and cell[k] == group[k] for k in
                ('group_id', 'variant', 'source', 'seed', 'images', 'parameters_sha256'))
            and cell['image_ids_sha256'] == execution['image_ids_sha256']
            and set(cell['metrics']) == set(COCO_BBOX_METRICS)
            and all(type(v) in (int, float) and math.isfinite(v) and (v == -1 or 0 <= v <= 1)
                for v in cell['metrics'].values()) and cell['bbox_mAP'] == cell['metrics']['bbox_mAP'] >= 0,
            'Accepted operator cell changed its original scientific identity')
        if index < 32:
            ref = refs[index]
            require(ref['group_id'] == cell['group_id'] and ref['target'] == target
                and ref['normalized_cell_sha256'] == canonical_hash(cell), 'Original first-pair replay was relabelled')
            read.bind(ref['cell_file'], ref['cell_sha256'])
            read.bind(ref['replay_file'], ref['replay_sha256'])
        else:
            path = root/'cells'/('g{:06d}'.format(cell['group_id']))/target/'cell.json'
            outer = read.json(path)
            require(outer['status'] == 'complete_operator_point_cell_pending_independent_acceptance'
                and outer['errors'] == [] and outer['normalized_cell'] == cell
                and outer['normalized_cell_sha256'] == canonical_hash(cell)
                and outer['point_plan_sha256'] == audit['point_plan_sha256']
                and outer['replay_file'] == str(path.parent/'input_replay.json'), 'Original operator replay changed')
            read.bind(outer['replay_file'], outer['replay_sha256'])
            refs.append(dict(group_id=cell['group_id'], target=target, cell_file=str(path),
                cell_sha256=read.checked[str(path)], replay_file=outer['replay_file'],
                replay_sha256=outer['replay_sha256'], normalized_cell_sha256=canonical_hash(cell)))
    require(len(refs) == 80 and len({(r['group_id'], r['target']) for r in refs}) == 80,
        'Operator replay reuse is incomplete or duplicated')
    decision = read_bound(decision_path, decision_hash)
    declared = read_bound(decision['prepared_stage_file'], decision['prepared_stage_sha256'])
    require(canonical_hash(declared) == canonical_hash(stage), 'Operator decision binds another width preparation')
    verify_operator_decision(decision, audit, audit_hash, cost, cost_hash, decision['prepared_stage_sha256'])
    require([c['id'] for c in report['contrasts']] == [c['name'] for c in cost_report['contrasts']],
        'Accepted AP and cost contrasts differ')
    read.unchanged()
    cost_read.unchanged()
    references = [dict(file=str(Path(p).resolve()), sha256=h, purpose=purpose) for p, h, purpose in
        ((audit_path, audit_hash, 'operator_point_acceptance'), (cost_path, cost_hash, 'operator_saved_cost_acceptance'),
         (decision_path, decision_hash, 'operator_stage_decision'),
         (decision['prepared_stage_file'], decision['prepared_stage_sha256'], 'remaining_width_preparation'))]
    return dict(status='bound_operator_point_stage_for_registered_width_continuation', contract_version=2,
        operator_execution_plan_file=point['execution_plan_file'],
        operator_execution_plan_sha256=audit['execution_plan_sha256'], producer_contract=proof,
        analysis_audit_sha256=audit_hash, cost_audit_sha256=cost_hash, decision_sha256=decision_hash,
        references=references, groups=deepcopy(groups), cells=deepcopy(cells), point_replay_refs=refs,
        checked_input_sha256=read.checked, checked_cost_input_sha256=cost_read.checked,
        stage_preparation_sha256=decision['prepared_stage_sha256'], decision=decision,
        original_cells_modified=False, bootstrap_required=False, previous_intervals_reused=False,
        physical_cost_calibration_claimed=False, independent_confirmation=False,
        final_candidate_frozen=False, automatic_promotion=False)


def compile_remaining_execution_plan(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash):
    from . import vcsf_scale_execution_contract as execution
    previous = bind_operator_point_stage(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash)
    operator = read_bound(previous['operator_execution_plan_file'], previous['operator_execution_plan_sha256'])
    stage = compile_stage_plan(registry, 'remaining_widths', ['cuda:0', 'cuda:1'], analysis_mode='point_estimate')
    historical = read_bound(operator['historical_audit_file'], operator['historical_audit_sha256'])
    configs = execution.bind_model_configs(registry, historical)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=registry.root, text=True).strip()
    require(head == operator['base_commit'], 'Width successor changed its scientific base revision')
    snapshot = execution.runtime_snapshot(registry.root)
    plan = deepcopy(operator)
    definition = registry.protocols[execution.EXECUTION_PROTOCOL]
    point_definition = registry.protocols['vcsf_scale_point_stages']
    plan.update(definition=deepcopy(definition), definition_sha256=canonical_hash(definition),
        point_stage_definition=deepcopy(point_definition), point_stage_definition_sha256=canonical_hash(point_definition),
        stage=stage['stage'], prepared_plan=stage['prepared_plan'], prepared_plan_sha256=stage['prepared_plan_sha256'],
        groups=stage['previous_stage_groups']+stage['groups'],
        new_group_ids=[g['group_id'] for g in stage['groups']],
        reused_group_ids=[g['group_id'] for g in stage['previous_stage_groups']], new_groups=len(stage['groups']),
        new_images=stage['scheduled_images_before_qualified_reuse'],
        new_evaluations=stage['target_evaluations_before_qualified_reuse'], model_config_bindings=configs,
        runtime_sha256=snapshot, runtime_snapshot_sha256=canonical_hash(snapshot),
        analysis=stage['analysis'], analysis_sha256=stage['analysis_sha256'], previous_stage=previous,
        previous_stage_sha256=canonical_hash(previous),
        previous_analysis_audit_file=str(Path(audit_path).resolve()), previous_analysis_audit_sha256=audit_hash,
        previous_cost_audit_file=str(Path(cost_path).resolve()), previous_cost_audit_sha256=cost_hash,
        previous_decision_file=str(Path(decision_path).resolve()), previous_decision_sha256=decision_hash)
    return plan
