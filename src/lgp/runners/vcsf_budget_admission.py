"""Prospective radius admission, separate from numerical method implementation."""
from pathlib import Path
import importlib.metadata
import shutil

from ..io import file_digest
from .vcsf_final_budget_contract import PROTOCOL, build_plan
from .vcsf_final_training_state import source_identity
from .vcsf_oblivious_admission import bound_json, isolate_output


CPU_ONLY_BRIDGE_FILES = {
    'src/lgp/runners/vcsf_budget_admission.py',
    'src/lgp/runners/vcsf_budget_lifecycle.py',
}
CPU_BRIDGE_TESTS = {
    'tests/test_vcsf_budget_lifecycle.py',
    'tests/test_vcsf_budget_group_audit.py',
    'tests/test_vcsf_budget_source_bridge.py',
}


def verify_diagnostic_source(permit, diagnostic):
    before, after = diagnostic['source_identity'], permit['source_identity']
    changes = {name: dict(before=before.get(name), after=after.get(name))
        for name in set(before) | set(after) if before.get(name) != after.get(name)}
    if not changes:
        return
    if (not set(changes).issubset(CPU_ONLY_BRIDGE_FILES)
            or changes != permit.get('reviewed_source_changes')):
        raise ValueError('Radius GPU path changed after its actual diagnostic')
    ref = permit['cpu_bridge_receipt']
    bridge = bound_json(ref['file'], ref['sha256'])
    if (bridge['status'] != 'qualified_cpu_only_radius_bridge'
            or bridge['changes'] != changes or bridge['model_calls'] != 0
            or bridge['independent_review_accepted'] is not True):
        raise ValueError('Radius CPU-only source bridge is not qualified')
    tests = bridge['tests_exit']
    qualification = bound_json(tests['file'], tests['sha256'])
    evaluator_ref = permit['evaluator_receipt']
    evaluator = bound_json(evaluator_ref['file'], evaluator_ref['sha256'])
    if (qualification.get('exit_code') != 0
            or qualification.get('status') != 'qualified_radius_cpu_bridge_tests'
            or qualification.get('source_identity') != after
            or qualification.get('environment_name') != 'oda'
            or qualification.get('packages') != evaluator['packages']
            or qualification.get('evaluator_receipt') != evaluator_ref
            or qualification.get('acceptance_tool') != permit['acceptance_tool']
            or type(qualification.get('passed')) is not int
            or qualification['passed'] <= 0
            or any(qualification.get(key) != 0 for key in ('failed', 'errors', 'skipped'))
            or set(qualification.get('test_files', {})) != CPU_BRIDGE_TESTS):
        raise ValueError('Radius CPU-only source bridge tests failed')
    tool = qualification['acceptance_tool']
    if file_digest(Path(tool['file'])) != tool['sha256']:
        raise ValueError('Radius CPU-only source bridge tested tool changed')
    for item in qualification['test_files'].values():
        if file_digest(Path(item['file'])) != item['sha256']:
            raise ValueError('Radius CPU-only source bridge test bytes changed')
    return bridge


def verify_formal_permit(registry, permit, *, recheck_evidence=False):
    if (permit.get('status') != 'approved_budget_formal'
            or permit.get('protocol') != PROTOCOL
            or permit.get('max_images', 'missing') is not None):
        raise ValueError('Exact formal radius permission is missing')
    plan = build_plan(registry, permit['devices'])
    indices = [row['index'] for row in plan['generation_groups']]
    if (permit['plan'] != plan or permit['group_indices'] != indices
            or any(type(i) is not int for i in permit['group_indices'])
            or permit['source_identity'] != source_identity(registry.root)):
        raise ValueError('Formal radius plan, groups or source snapshot differs')
    targets_ref = permit['target_receipt']
    targets = bound_json(targets_ref['file'], targets_ref['sha256'])
    if (targets['status'] != 'radius_target_bytes_bound_pending_evaluator_and_reuse'
            or targets['canonical_order'] != registry.paper_order
            or targets['targets'] != permit['checkpoints']):
        raise ValueError('Formal radius target checkpoint binding differs')
    evaluator_ref = permit['evaluator_receipt']
    evaluator = bound_json(evaluator_ref['file'], evaluator_ref['sha256'])
    if (evaluator['status'] != 'radius_target_configs_match_historical_pending_reuse'
            or [row['model'] for row in evaluator['comparisons']] != registry.paper_order
            or evaluator['evidence'].get(str(Path(targets_ref['file']).resolve())) != targets_ref['sha256']):
        raise ValueError('Formal radius evaluator binding differs')
    if {name: importlib.metadata.version(name) for name in evaluator['packages']} != evaluator['packages']:
        raise ValueError('Formal radius installed package versions changed')
    reuse_ref = permit['reference_reuse']
    reuse = bound_json(reuse_ref['file'], reuse_ref['sha256'])
    expected = [(row['source'], row['epsilon'], row['target']) for row in plan['reference_reuse']]
    if (reuse['status'] != 'radius_reference_destination_qualified'
            or reuse['destination_reuse_qualified'] is not True
            or reuse['parameters_sha256'] != plan['parameters_sha256']
            or reuse['image_ids_sha256'] != plan['full_image_ids_sha256']
            or [(row['source'], row['epsilon'], row['target']) for row in reuse['cells']] != expected
            or reuse['target_receipt'] != targets_ref or reuse['evaluator_receipt'] != evaluator_ref
            or reuse['assets'] != permit['assets']):
        raise ValueError('Formal radius reference reuse is not qualified for these inputs')
    diagnostic_ref = permit['diagnostic_receipt']
    diagnostic = bound_json(diagnostic_ref['file'], diagnostic_ref['sha256'])
    if (diagnostic['status'] != 'independently_verified_radius_execution_diagnostic'
            or diagnostic['sources'] != registry.protocols[PROTOCOL]['sources']
            or diagnostic['devices'] != permit['devices']
            or diagnostic['gpu_uuids'] != permit['gpu_uuids']
            or diagnostic['evaluator_receipt'] != evaluator_ref
            or diagnostic['groups'] != len(diagnostic['sources'])
            or diagnostic['cells'] != len(diagnostic['sources']) * len(registry.paper_order)
            or diagnostic['formal_result_eligible'] is not False):
        raise ValueError('Current owned radius execution lacks its bounded diagnostic acceptance')
    bridge = verify_diagnostic_source(permit, diagnostic)
    tool = permit['acceptance_tool']
    if diagnostic['acceptance_tool'] != tool or file_digest(Path(tool['file'])) != tool['sha256']:
        raise ValueError('Prospective radius acceptance tool changed')
    if permit['lifecycle'] != 'delete_new_images_after_group_acceptance_no_backup':
        raise ValueError('Formal radius requires the explicitly authorized per-group lifecycle')
    storage = permit['storage']
    for key in ['image_allowance_per_group', 'prediction_allowance_per_group', 'reserve_bytes']:
        if type(storage.get(key)) is not int or storage[key] <= 0:
            raise ValueError('Invalid radius capacity allowance')
    references = [targets_ref, evaluator_ref, reuse_ref, diagnostic_ref, tool,
                  permit['assets'], permit['prepared'], permit['checkpoint_bindings']]
    if bridge is not None:
        references += [permit['cpu_bridge_receipt'], bridge['tests_exit']]
    isolate_output(permit['output'], [Path(ref['file']).resolve().parent for ref in references])
    if recheck_evidence:
        for ref in references:
            if file_digest(Path(ref['file'])) != ref['sha256']:
                raise ValueError('Formal radius prerequisite changed')
        for item in [targets, evaluator, reuse, diagnostic]:
            for path, digest in item.get('evidence', {}).items():
                if file_digest(Path(path)) != digest:
                    raise ValueError('Formal radius bound evidence changed')
        require_group_space(registry.root, permit)
    result = dict(targets=targets, evaluator=evaluator, reuse=reuse, diagnostic=diagnostic)
    if bridge is not None:
        result['bridge'] = dict(bridge, evidence={
            str(Path(permit['cpu_bridge_receipt']['file']).resolve()): permit['cpu_bridge_receipt']['sha256'],
            str(Path(bridge['tests_exit']['file']).resolve()): bridge['tests_exit']['sha256']})
    return result


def require_group_space(root, permit):
    if permit['max_images'] is not None:
        return
    storage = permit['storage']
    allowance = storage['image_allowance_per_group'] + storage['prediction_allowance_per_group']
    required = storage['reserve_bytes'] + len(permit['devices']) * allowance
    path = Path(permit['output']).resolve()
    while not path.exists():
        path = path.parent
    if shutil.disk_usage(path).free < required:
        raise RuntimeError('Insufficient space for the next radius groups; preserve existing evidence')
