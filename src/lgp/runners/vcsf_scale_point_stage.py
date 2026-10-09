"""Prospective stage transitions from accepted full-panel point estimates."""
from copy import deepcopy
import math
from pathlib import Path
import subprocess

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS
from .vcsf_efficacy_contract import read_bound, require
from .vcsf_producer_verification import verify_producer_contract
from .vcsf_research_plan import canonical_hash
from .vcsf_scale_plan import compile_stage_plan
from .vcsf_scale_stage_transition import (BoundEvidence, scientific_plan, scope_flag,
    verify_decision, verify_previous_group_identities)


def verify_point_scope(audit, cost):
    require(audit.get('status') == 'independently_verified_scale_point_analysis'
        and audit.get('errors') == [] and cost.get('status')
        == 'independently_verified_saved_cost_disclosure_not_physical_calibration'
        and cost.get('errors') == [], 'The point stage needs independent AP and saved-cost acceptance')
    for key, value in dict(groups=2, cells=32, images=5000, contrasts=1,
            diagnostic=False, all_twelve_metrics_verified=True, analysis_pipeline_accepted=True,
            bootstrap_replicates_computed=0, new_model_calls=0, original_roots_modified=False,
            scientific_acceptance=False, independent_confirmation=False, significance_claimed=False,
            detailed_cost_acceptance=False, candidate_selected=False).items():
        scope_flag(audit, key, value)
    matrix = audit.get('matrix_verification', {})
    require(matrix.get('status') == 'independently_verified_scale_point_arithmetic',
        'Point acceptance lacks independent arithmetic')
    for key, value in dict(groups=2, cells=32, images=5000, contrasts=1, excluded_whitebox_cells=2,
            bootstrap_required=False, significance_claimed=False).items():
        scope_flag(matrix, key, value)
    for key, value in dict(diagnostic=False, images_per_group=5000, groups=2,
            source_trace_steps_replayed=200000, saved_cost_disclosure_accepted=True,
            physical_cost_calibration_accepted=False, physical_full_detector_BE_measured=False,
            kernel_FLOPs_measured=False, efficacy_selection=False, independent_confirmation=False,
            scientific_acceptance=False, new_model_calls=0).items():
        scope_flag(cost, key, value)
    require(audit['legacy_plan_sha256'] == cost['analysis_plan_sha256']
        and audit['new_inputs_sha256'] == cost['full_group_input_sha256'],
        'Point AP and saved costs do not bind the same original inputs')


def bind_cost(cost, read):
    paths = [p for p, h in read.mapping.items()
        if Path(p).name == 'cost.json' and h == cost['source_report_sha256']]
    require(len(paths) == 1, 'Accepted cost report has no unique original binding')
    path = Path(paths[0])
    report = read.json(path, cost['source_report_sha256'])
    require(report.get('status') == 'source_and_trace_cost_disclosure_not_independent_acceptance'
        and report.get('diagnostic') is False and report.get('images') == 5000
        and report.get('contrast') == 'A_h2_minus_A_h0'
        and report.get('independent_cost_acceptance') is False, 'Original cost scope changed')
    manifest = read.json(path.parent / 'manifest.json')
    require(set(manifest) == {'cost.json', 'cost.csv', 'cost.tex'}
        and {p.name for p in path.parent.iterdir()} == set(manifest) | {'manifest.json'},
        'Accepted saved-cost export inventory changed')
    for name, digest in manifest.items():
        read.bind(path.parent / name, digest)
    return report


def verify_point_cells(first, plan, report):
    groups = plan['contract']['groups']
    cells = report['normalized_cells']
    require(report.get('status') == 'point_matrix_pending_independent_acceptance'
        and report.get('groups') == 2 and report.get('cells') == 32
        and cells == plan['historical_cells'] + plan['new_cells']
        and report.get('targets') == first['targets'] and report.get('sources') == first['sources']
        and report.get('bootstrap_replicates_computed') == 0
        and report.get('significance_claimed') is False
        and report.get('independent_confirmation') is False, 'Accepted point report changed scope')
    for index, cell in enumerate(cells):
        group, target = groups[index // 16], first['targets'][index % 16]
        require(cell['target'] == target and cell['status'] == 'complete' and cell['failures'] == []
            and all(type(cell[k]) is type(group[k]) and cell[k] == group[k] for k in
                ('group_id', 'variant', 'source', 'seed', 'images', 'parameters_sha256'))
            and cell['image_ids_sha256'] == first['image_ids_sha256']
            and set(cell['metrics']) == set(COCO_BBOX_METRICS)
            and all(type(v) in (int, float) and math.isfinite(v) and (v == -1 or 0 <= v <= 1)
                for v in cell['metrics'].values())
            and cell['bbox_mAP'] == cell['metrics']['bbox_mAP'] >= 0,
            'Accepted point cell changed its full original identity')
    return groups, cells


def bind_first_point_stage(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash):
    audit, cost = read_bound(audit_path, audit_hash), read_bound(cost_path, cost_hash)
    verify_point_scope(audit, cost)
    read, cost_read = BoundEvidence(audit['checked_input_sha256']), BoundEvidence(cost['checked_input_sha256'])
    bind_cost(cost, cost_read)
    point = read.json(audit['point_plan_file'], audit['point_plan_sha256'])
    require(point['legacy_plan_sha256'] == audit['legacy_plan_sha256']
        and point['new_inputs_sha256'] == audit['new_inputs_sha256'], 'Point plan and acceptance differ')
    first = read.json(point['execution_plan_file'], audit['execution_plan_sha256'])
    read.bind(point['legacy_plan_file'], audit['legacy_plan_sha256'])
    admission = read.bind(point['admission_file'])
    producer = Path(point['execution_worktree'])
    proof = verify_producer_contract(producer, Path(point['execution_plan_file']), audit['execution_plan_sha256'],
        admission, read.checked[str(admission)], None)
    stage = compile_stage_plan(registry, 'operators_at_anchor_width', ['cuda:0', 'cuda:1'],
        analysis_mode='point_estimate')
    require(first['stage'] == 'baseline_and_identity' and first['new_group_ids'] == [2]
        and first['reused_group_ids'] == [1] and point['execution_plan_sha256'] == audit['execution_plan_sha256']
        and canonical_hash(scientific_plan(first['prepared_plan']))
            == canonical_hash(scientific_plan(stage['prepared_plan'])),
        'Original first-stage scientific conditions differ from the successor')
    from .vcsf_scale_execution_contract import UNCHANGED_SCIENTIFIC_FILES
    for name in UNCHANGED_SCIENTIFIC_FILES + ('src/lgp/attacks/vcsf_scale_isolated.py',):
        require(file_digest(registry.root / name) == first['runtime_sha256'].get(name),
            'A previous scientific implementation changed: ' + name)
    root = Path(audit['analysis_root'])
    completion = read.json(root / 'completion.json', audit['analysis_completion_sha256'])
    require(completion['status'] == 'complete_scale_points_pending_independent_acceptance'
        and completion['errors'] == [] and completion['point_plan_sha256'] == audit['point_plan_sha256']
        and completion['execution_plan_sha256'] == audit['execution_plan_sha256'], 'Point terminal binding changed')
    inventory = read.json(root / 'artifact_manifest.json', completion['artifact_manifest_sha256'])
    require(all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in inventory)
        and {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
            == set(inventory) | {'artifact_manifest.json', 'completion.json'},
        'Accepted point output inventory changed')
    for name, digest in inventory.items():
        read.bind(root / name, digest)
    exports = read.json(root / 'reports/manifest.json')
    require(set(exports) == {'analysis.json', 'metrics.csv', 'configuration_seed.csv', 'contrasts.csv',
            'target_contrasts.csv', 'contrasts.tex', 'metrics.tex'}
        and all(inventory.get('reports/' + name) == digest for name, digest in exports.items()),
        'Accepted point report exports changed')
    report_path = root / 'reports/analysis.json'
    report = read.json(report_path, audit['report_sha256'])
    groups, cells = verify_point_cells(first, point, report)
    verify_previous_group_identities(first['groups'], stage['previous_stage_groups'], groups)
    refs = deepcopy(point['historical_replay_refs'])
    require(len(refs) == 16 and [r['target'] for r in refs] == first['targets'], 'Historical point references changed')
    for ref, cell in zip(refs, cells[:16]):
        require(ref['group_id'] == cell['group_id']
            and ref['normalized_cell_sha256'] == canonical_hash(cell), 'Historical replay identity changed')
        read.bind(ref['cell_file'], ref['cell_sha256'])
        read.bind(ref['replay_file'], ref['replay_sha256'])
    for cell in cells[16:]:
        path = root / 'cells' / cell['target'] / 'cell.json'
        outer = read.json(path)
        require(outer['status'] == 'complete_point_cell_pending_independent_acceptance'
            and outer['errors'] == [] and outer['normalized_cell'] == cell
            and outer['normalized_cell_sha256'] == canonical_hash(cell)
            and outer['point_plan_sha256'] == audit['point_plan_sha256'], 'Accepted new point cell changed')
        read.bind(outer['replay_file'], outer['replay_sha256'])
        refs.append(dict(group_id=cell['group_id'], target=cell['target'], cell_file=str(path),
            cell_sha256=read.checked[str(path)], replay_file=outer['replay_file'], replay_sha256=outer['replay_sha256'],
            normalized_cell_sha256=canonical_hash(cell)))
    decision = read_bound(decision_path, decision_hash)
    declared = read_bound(decision['prepared_stage_file'], decision['prepared_stage_sha256'])
    require(canonical_hash(declared) == canonical_hash(stage), 'Decision binds another point-stage preparation')
    verify_decision(decision, audit, audit_hash, cost, cost_hash, decision['prepared_stage_sha256'], audit['report_sha256'])
    read.unchanged()
    cost_read.unchanged()
    references = [dict(file=str(Path(p).resolve()), sha256=h, purpose=purpose) for p, h, purpose in
        ((audit_path, audit_hash, 'previous_stage_point_acceptance'), (cost_path, cost_hash, 'previous_stage_saved_cost'),
         (decision_path, decision_hash, 'previous_stage_decision'),
         (decision['prepared_stage_file'], decision['prepared_stage_sha256'], 'point_stage_preparation'))]
    return dict(status='bound_first_scale_point_stage_for_registered_operator_continuation', contract_version=2,
        first_execution_plan_file=point['execution_plan_file'], first_execution_plan_sha256=audit['execution_plan_sha256'],
        producer_contract=proof, analysis_audit_sha256=audit_hash, cost_audit_sha256=cost_hash,
        decision_sha256=decision_hash, references=references, groups=groups, cells=cells, point_replay_refs=refs,
        checked_input_sha256=read.checked, checked_cost_input_sha256=cost_read.checked,
        stage_preparation_sha256=decision['prepared_stage_sha256'], decision=decision,
        original_cells_modified=False, bootstrap_required=False, previous_intervals_reused=False,
        physical_cost_calibration_claimed=False, independent_confirmation=False,
        final_candidate_frozen=False, automatic_promotion=False)


def compile_point_execution_plan(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash):
    from . import vcsf_scale_execution_contract as execution
    previous = bind_first_point_stage(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash)
    first = read_bound(previous['first_execution_plan_file'], previous['first_execution_plan_sha256'])
    stage = compile_stage_plan(registry, 'operators_at_anchor_width', ['cuda:0', 'cuda:1'],
        analysis_mode='point_estimate')
    historical = read_bound(first['historical_audit_file'], first['historical_audit_sha256'])
    configs = execution.bind_model_configs(registry, historical)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=registry.root, text=True).strip()
    require(head == first['base_commit'], 'Point successor changed its scientific base revision')
    snapshot = execution.runtime_snapshot(registry.root)
    # Only execution and analysis identity changes; frozen data/checkpoint fields are retained.
    plan = deepcopy(first)
    definition = registry.protocols[execution.EXECUTION_PROTOCOL]
    point_definition = registry.protocols['vcsf_scale_point_stages']
    plan.update(schema_version=2, analysis_mode='point_estimate', definition=deepcopy(definition),
        definition_sha256=canonical_hash(definition), point_stage_definition=deepcopy(point_definition),
        point_stage_definition_sha256=canonical_hash(point_definition), stage=stage['stage'],
        prepared_plan=stage['prepared_plan'], prepared_plan_sha256=stage['prepared_plan_sha256'],
        groups=stage['previous_stage_groups'] + stage['groups'], new_group_ids=[3, 4, 5], reused_group_ids=[1, 2],
        new_groups=3, new_images=15000, new_evaluations=48, model_config_bindings=configs,
        runtime_sha256=snapshot, runtime_snapshot_sha256=canonical_hash(snapshot),
        analysis=stage['analysis'], analysis_sha256=stage['analysis_sha256'],
        previous_stage=previous, previous_stage_sha256=canonical_hash(previous),
        previous_analysis_audit_file=str(Path(audit_path).resolve()), previous_analysis_audit_sha256=audit_hash,
        previous_cost_audit_file=str(Path(cost_path).resolve()), previous_cost_audit_sha256=cost_hash,
        previous_decision_file=str(Path(decision_path).resolve()), previous_decision_sha256=decision_hash)
    require([g['variant'] for g in plan['groups']] == ['A_h2', 'A_h0', 'B_h2', 'C_h2', 'D_h2'],
        'Point successor lost its full operator stage')
    return plan
