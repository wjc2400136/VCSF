"""Bind an accepted first scale pair without rewriting its original cells."""
from copy import deepcopy
from datetime import datetime
import math
from pathlib import Path
import struct

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS
from .vcsf_efficacy_contract import read_bound, require
from .vcsf_research_plan import canonical_hash
from .vcsf_scale_plan import compile_stage_plan


NUMERICAL_FILES = tuple('src/lgp/reporting/' + name + '.py' for name in
    ('vcsf_vectorized_bootstrap', 'vcsf_bootstrap_cell', 'vcsf_image_bootstrap',
     'vcsf_coco_replay', 'vcsf_prediction_evidence', 'vcsf_resample_artifact'))
REVIEW_DIMENSIONS = ('full_sixteen_target_AP', 'paired_BB_change',
    'saved_cost_disclosure', 'implementation_complexity')


def scientific_plan(plan):
    result = deepcopy(plan)
    result.pop('source_sha256')
    return result


def scope_flag(document, key, expected):
    require(type(document.get(key)) is type(expected) and document[key] == expected,
        'Previous-stage scope differs: ' + key)


def verify_pair_scope(audit, cost):
    require(audit.get('status') == 'independently_verified_scale_pair_analysis'
        and audit.get('errors') == [] and cost.get('status')
        == 'independently_verified_saved_cost_disclosure_not_physical_calibration'
        and cost.get('errors') == [], 'The previous stage lacks independent full AP and saved-cost acceptance')
    for key, value in dict(diagnostic=False, cells=32, bootstrap_vectors=30, images=5000,
            analysis_pipeline_accepted=True, new_model_calls=0, prediction_bootstraps_recomputed=0,
            original_roots_modified=False, independent_confirmation=False, scientific_acceptance=False,
            detailed_cost_acceptance=False, candidate_selected=False).items():
        scope_flag(audit, key, value)
    matrix = audit.get('matrix_verification', {})
    require(matrix.get('status') == 'independently_verified_scale_pair_arithmetic',
        'Previous AP receipt lacks independent paired arithmetic')
    for key, value in dict(groups=2, observation_cells=32, blackbox_bootstrap_cells=30,
            whitebox_cells_excluded=2, replicates=2000, contrasts=1).items():
        scope_flag(matrix, key, value)
    for key, value in dict(diagnostic=False, images_per_group=5000, groups=2,
            source_trace_steps_replayed=200000, saved_cost_disclosure_accepted=True,
            physical_cost_calibration_accepted=False, physical_full_detector_BE_measured=False,
            kernel_FLOPs_measured=False, efficacy_selection=False,
            independent_confirmation=False, scientific_acceptance=False, new_model_calls=0).items():
        scope_flag(cost, key, value)
    require(audit['analysis_plan_sha256'] == cost['analysis_plan_sha256']
        and audit['new_inputs_sha256'] == cost['full_group_input_sha256'],
        'AP and cost acceptance refer to different first-stage evidence')


def verify_decision(decision, audit, audit_hash, cost, cost_hash, stage_hash, report_hash):
    require(decision.get('status') == 'recorded_first_scale_stage_decision'
        and decision.get('decision') in ('keep', 'change', 'reject')
        and decision.get('scope') == 'first_stage_scale_on_vs_identity'
        and decision.get('next_stage') == 'operators_at_anchor_width'
        and decision.get('previous_analysis_audit_sha256') == audit_hash
        and decision.get('previous_cost_audit_sha256') == cost_hash
        and decision.get('prepared_stage_sha256') == stage_hash
        and decision.get('reviewed_analysis_sha256') == report_hash
        and decision.get('reviewed_cost_report_sha256') == cost['source_report_sha256']
        and decision.get('reviewed_dimensions') == list(REVIEW_DIMENSIONS),
        'A recorded complete-evidence stage decision is required')
    require(isinstance(decision.get('rationale'), str) and decision['rationale'].strip(),
        'A stage decision needs an explicit rationale, not only a status')
    for key, value in dict(next_stage_configurations_unchanged=True, no_early_efficacy_pruning=True,
            physical_cost_calibration_claimed=False, independent_confirmation=False,
            final_candidate_frozen=False, automatic_promotion=False).items():
        scope_flag(decision, key, value)
    times = [datetime.fromisoformat(document['completed_at']) for document in (audit, cost)]
    recorded = datetime.fromisoformat(decision['recorded_at'])
    require(recorded.tzinfo is not None and all(t.tzinfo is not None and t <= recorded for t in times),
        'Stage decision predates complete AP or cost acceptance')


def verify_previous_group_identities(executed, prepared, analyzed):
    require(canonical_hash(executed) == canonical_hash(prepared),
        'Previous scale configurations changed')
    require(len(analyzed) == 2 and [g['variant'] for g in analyzed] == ['levels_two', 'A_h0']
        and all(type(g['group_id']) is int for g in analyzed)
        and [g['group_id'] for g in analyzed] == [29, 2]
        and canonical_hash(analyzed[1]['parameters']) == canonical_hash(executed[1]['parameters']),
        'Original first-stage group or parameter identity changed')


class BoundEvidence:
    def __init__(self, mapping):
        self.mapping = mapping
        self.checked = {}

    def bind(self, path, digest=None):
        path = Path(path).absolute()
        require(path == path.resolve() and path.is_file(), 'Previous evidence has moved or uses a symlink')
        expected = self.mapping.get(str(path))
        require(isinstance(expected, str) and len(expected) == 64
            and (digest is None or expected == digest) and file_digest(path) == expected,
            'Previous acceptance does not bind the exact evidence bytes: ' + path.name)
        self.checked[str(path)] = expected
        return path

    def json(self, path, digest=None):
        path = self.bind(path, digest)
        return read_bound(path, self.checked[str(path)])

    def unchanged(self):
        require(all(file_digest(Path(p)) == h for p, h in self.checked.items()),
            'Previous evidence changed during stage binding')


def bind_first_stage(registry, audit_path, audit_hash, cost_path, cost_hash, decision_path, decision_hash):
    audit = read_bound(audit_path, audit_hash)
    cost = read_bound(cost_path, cost_hash)
    verify_pair_scope(audit, cost)
    read = BoundEvidence(audit['checked_input_sha256'])
    cost_read = BoundEvidence(cost['checked_input_sha256'])
    cost_paths = [p for p, h in cost_read.mapping.items()
        if Path(p).name == 'cost.json' and h == cost['source_report_sha256']]
    require(len(cost_paths) == 1, 'The accepted saved-cost report has no unique original file binding')
    cost_report = cost_read.json(cost_paths[0], cost['source_report_sha256'])
    require(cost_report.get('status') == 'source_and_trace_cost_disclosure_not_independent_acceptance'
        and cost_report.get('diagnostic') is False and cost_report.get('images') == 5000
        and cost_report.get('contrast') == 'A_h2_minus_A_h0'
        and cost_report.get('independent_cost_acceptance') is False,
        'The original cost report scope differs from its independent acceptance')
    cost_root = Path(cost_paths[0]).parent
    inventory = cost_read.json(cost_root / 'manifest.json')
    require(set(inventory) == {'cost.json', 'cost.csv', 'cost.tex'}
        and {p.name for p in cost_root.iterdir()} == set(inventory) | {'manifest.json'},
        'The accepted saved-cost export inventory changed')
    for name, digest in inventory.items():
        cost_read.bind(cost_root / name, digest)
    plan = read.json(audit['analysis_plan_file'], audit['analysis_plan_sha256'])
    execution = read.json(plan['execution_plan_file'], audit['execution_plan_sha256'])
    stage = compile_stage_plan(registry, 'operators_at_anchor_width', ['cuda:0', 'cuda:1'])
    require(plan['execution_plan_sha256'] == audit['execution_plan_sha256']
        and execution['stage'] == 'baseline_and_identity'
        and execution['new_group_ids'] == [2] and execution['reused_group_ids'] == [1]
        and canonical_hash(scientific_plan(execution['prepared_plan']))
            == canonical_hash(scientific_plan(stage['prepared_plan'])),
        'Previous and successor scale study conditions differ')
    verify_previous_group_identities(execution['groups'], stage['previous_stage_groups'],
        plan['contract']['plan']['groups'])
    from .vcsf_scale_execution_contract import UNCHANGED_SCIENTIFIC_FILES
    for name in UNCHANGED_SCIENTIFIC_FILES + ('src/lgp/attacks/vcsf_scale_isolated.py',):
        require(file_digest(registry.root / name) == execution['runtime_sha256'].get(name),
            'A previous attack, detector or evaluator implementation changed: ' + name)
    for name in NUMERICAL_FILES:
        require(file_digest(registry.root / name) == plan['source_sha256'].get(name),
            'Previous bootstrap numerical qualification cannot be inherited: ' + name)
    root = Path(audit['analysis_root'])
    completion = read.json(root / 'completion.json', audit['analysis_completion_sha256'])
    require(completion['analysis_plan_sha256'] == audit['analysis_plan_sha256']
        and completion['execution_plan_sha256'] == audit['execution_plan_sha256']
        and completion['status'] == 'complete_scale_pair_analysis_pending_independent_acceptance'
        and completion['errors'] == [], 'Previous analysis terminal evidence differs')
    report_path = root / 'reports/analysis.json'
    report = read.json(report_path)
    cells = report['normalized_cells']
    groups = plan['contract']['plan']['groups']
    require(cells[:16] == plan['history']['cells'] and len(cells) == 32,
        'Original first-stage group or observation identities changed')
    shared = plan['contract']['shared_resamples']
    require(shared['shape'] == [2000, 5000] and shared['image_ids_sha256'] == execution['image_ids_sha256'],
        'Previous common image resamples differ')
    for key, digest_key in (('shared_metadata_file', 'metadata_sha256'), ('shared_payload_file', 'payload_sha256')):
        read.bind(plan['contract'][key], shared[digest_key])
    old_refs = {r['target']: r for r in plan['history']['bootstrap_refs']}
    require(len(old_refs) == 15 and list(old_refs) == groups[0]['blackbox_targets'],
        'Historical bootstrap panel is incomplete or reordered')
    refs = []
    for index, cell in enumerate(cells):
        group = groups[index // 16]
        target = execution['targets'][index % 16]
        require(cell['target'] == target and cell['status'] == 'complete' and cell['failures'] == []
            and all(type(cell[k]) is type(group[k]) and cell[k] == group[k]
                for k in ('group_id', 'variant', 'source', 'seed', 'images', 'parameters_sha256'))
            and cell['image_ids_sha256'] == execution['image_ids_sha256']
            and set(cell['metrics']) == set(COCO_BBOX_METRICS)
            and all(type(v) in (int, float) and math.isfinite(v) and (v == -1 or 0 <= v <= 1)
                for v in cell['metrics'].values())
            and cell['bbox_mAP'] == cell['metrics']['bbox_mAP'], 'Previous cell lost its exact complete identity')
        if index >= 16:
            outer = read.json(root / 'cells' / target / 'cell.json')
            require(outer['status'] == 'complete_scale_cell_pending_pipeline_acceptance'
                and outer['errors'] == [] and outer['diagnostic'] is False
                and outer['normalized_cell'] == cell and outer['normalized_cell_sha256'] == canonical_hash(cell),
                'Saved identity cell differs from its accepted report')
        if target == cell['source']:
            continue
        path = Path(old_refs[target]['bootstrap_receipt_file']) if index < 16 else root / 'cells' / target / 'bootstrap/bootstrap_cell.json'
        boot = read.json(path)
        require(boot['status'] == 'complete' and boot['failures'] == []
            and boot['replicates_completed'] == boot['replicates_expected'] == 2000
            and boot['normalized_cell_sha256'] == canonical_hash(cell)
            and boot['input_evidence_sha256'] == cell['input_evidence_sha256']
            and all(boot[k] == cell[k] for k in ('group_id', 'variant', 'source', 'target',
                'seed', 'images', 'parameters_sha256')),
            'Previous bootstrap no longer binds the original observation')
        for field, key in (('resample_metadata_sha256', 'metadata_sha256'),
                ('resample_stream_sha256', 'payload_sha256'), ('resample_definition_sha256', 'definition_sha256')):
            require(boot[field] == shared[key], 'Previous bootstrap used different paired draws')
        ap = read.bind(path.parent / 'bootstrap_ap.f8le', boot['ap_values_sha256'])
        raw = ap.read_bytes()
        require(len(raw) == 16000 and all(math.isfinite(v) and 0 <= v <= 1 for v in struct.unpack('<2000d', raw)),
            'Previous bootstrap vector is truncated or invalid')
        refs.append(dict(group_id=cell['group_id'], target=target, original_cell_sha256=canonical_hash(cell),
            bootstrap_receipt_file=str(path), bootstrap_receipt_sha256=read.checked[str(path)],
            ap_file=str(ap), ap_values_sha256=boot['ap_values_sha256']))
    require(len(refs) == 30, 'The previous paired bootstrap panel is incomplete')
    decision = read_bound(decision_path, decision_hash)
    declared_stage = read_bound(decision['prepared_stage_file'], decision['prepared_stage_sha256'])
    require(canonical_hash(declared_stage) == canonical_hash(stage), 'The decision binds another successor-stage preparation')
    verify_decision(decision, audit, audit_hash, cost, cost_hash,
        decision['prepared_stage_sha256'], read.checked[str(report_path)])
    read.unchanged()
    cost_read.unchanged()
    references = [dict(file=str(Path(p).resolve()), sha256=h, purpose=purpose)
        for p, h, purpose in ((audit_path, audit_hash, 'previous_stage_AP_acceptance'),
            (cost_path, cost_hash, 'previous_stage_saved_cost_acceptance'),
            (decision_path, decision_hash, 'previous_stage_scientific_decision'),
            (decision['prepared_stage_file'], decision['prepared_stage_sha256'], 'successor_stage_preparation'))]
    return dict(status='bound_first_scale_stage_for_registered_operator_continuation',
        first_execution_plan_file=plan['execution_plan_file'],
        first_execution_plan_sha256=audit['execution_plan_sha256'],
        analysis_audit_sha256=audit_hash, cost_audit_sha256=cost_hash, decision_sha256=decision_hash,
        references=references, groups=groups, cells=cells, bootstrap_refs=refs,
        shared_contract=plan['contract'], checked_input_sha256=read.checked,
        checked_cost_input_sha256=cost_read.checked,
        scientific_plan_sha256=canonical_hash(scientific_plan(stage['prepared_plan'])),
        stage_preparation_sha256=decision['prepared_stage_sha256'], decision=decision,
        original_cells_modified=False, previous_intervals_reused=False,
        physical_cost_calibration_claimed=False, independent_confirmation=False,
        final_candidate_frozen=False, automatic_promotion=False)
