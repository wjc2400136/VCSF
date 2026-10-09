"""Qualify one original whole group by fresh collection and 16 official replays.

qualify_group_inputs(report_reference, *, output, registry=None,
max_uncompressed_bytes=4 << 30) owns one NEW direct audit leaf under
ROOT/outputs/audits/vcsf_final_followup_group_qualification. It retains inputs,
per-target replay artifacts and failures; it never resumes or edits evidence.
This is callable by the integrating parent, not a replacement experiment CLI.

Qualification concerns this original group's full inputs and reproduced saved
metrics only. It does not certify planned-stop cause, predecessor shutdown,
successor admission, dataset independence, or scientific/matrix acceptance.
Caller supplies a trusted report FILE digest; report self-seals are consistency
checks only. The unchanged collector reauthenticates original runtime/admission
on their original source root. No process or source-snapshot bypass is offered.
"""
from copy import deepcopy
from pathlib import Path, PurePosixPath
import sys

from lgp.io import atomic_json
from lgp.metrics import COCO_BBOX_METRICS
from lgp.runners.vcsf_research_plan import canonical_hash, require
from tools.audit_vcsf_cpu_analysis import child, plain, sha, same
from tools import vcsf_final_followup_group_inputs as collector
from tools import vcsf_final_followup_result_inputs as inputs
from tools.vcsf_final_followup_result_replay import replay_new_followup_cell


ROOT = Path(__file__).resolve().parents[1]
NAMESPACE = 'vcsf_final_followup_group_qualification'
STATUS = 'qualified_original_whole_group_after_input_readback_and_official_replay'


class GroupQualificationError(RuntimeError):
    """failure_report is never a successful C reference, even after 15 replays."""

    def __init__(self, report):
        self.failure_report = report
        super().__init__(report['error'])


def _server_oda():
    require(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        'Group qualification requires authorized server ODA')


def _semantic(report):
    require(type(report) is dict and type(report.get('schema_version')) is int
        and report['schema_version'] == 1 and report.get('status') == collector.STATUS,
        'Require successful original whole-group collector report')
    require(report.get('report_content_sha256') == canonical_hash(
        {k: v for k, v in report.items() if k != 'report_content_sha256'}), 'Collector seal differs')
    value = deepcopy(report)
    value.pop('report_content_sha256')
    prefix = value['completed_prefix_reference']
    path = collector._reference_path(prefix)
    request = collector._reference_path(value['worker_request_reference'])
    require(path == request.parent / 'completed_groups.json', 'Foreign completed-prefix path')
    controls = value['control_metadata_sha256']
    require(type(controls) is dict and controls.get(str(path)) == prefix['sha256'],
        'Completed-prefix control digest differs')
    # Only this appendable lane-prefix digest is observational. Fresh collection
    # still checks its canonical prefix and exact selected worker_group content.
    prefix['sha256'] = None
    controls[str(path)] = None
    return value


def _recollect(saved, registry):
    current = collector.collect_group_inputs(saved['runtime_reference'], saved['admission_reference'],
        saved['worker_group_reference'], saved['worker_request_reference'], registry=registry,
        planned_stop_failure_references=saved['planned_stop_failure_references'])
    same(_semantic(current), _semantic(saved), 'Fresh collector semantic contents differ')
    return current


def _inventory(report):
    gid = report['group_id']
    require(type(gid) is int and 2 <= gid <= 16 and report['group']['group_id'] == gid,
        'Require one original new group')
    inventory = report['raw_inventory']
    require(type(inventory) is dict and inventory, 'Missing complete group inventory')
    result = {}
    for name, digest in inventory.items():
        require(type(name) is str and bool(name) and '\\' not in name,
            'Invalid group-relative inventory key')
        relative = PurePosixPath(name)
        require(bool(relative.parts) and not relative.is_absolute() and relative.as_posix() == name
            and '..' not in relative.parts and ':' not in name and inputs.is_sha(digest),
            'Unsafe group inventory key or digest')
        result['groups/%06d/%s' % (gid, name)] = digest
    return result


def _validate_replay(result, report, target, inventory, annotation):
    require(type(result) is dict and result.get('status') ==
        'followup_new_cell_official_replay_pending_outer_acceptance', 'Unexpected replay result')
    for name, count in (('model_calls', 0), ('AP_replays', 1), ('bootstrap_replicates_computed', 0)):
        require(type(result.get(name)) is int and result[name] == count, 'Invalid replay count: ' + name)
    require(all(result.get(k) is False for k in ('outer_input_authority_verified',
        'independent_result_acceptance', 'scientific_acceptance', 'formal_metrics_eligible')),
        'Replay component exceeded its authority')
    group, projection = report['group'], report['projected_inputs']
    binding = result['input_binding']
    root = plain(report['execution_root'])
    prefix = 'groups/%06d' % group['group_id']
    names = [prefix + '/attack/annotations.json', prefix + '/attack/run.json',
        prefix + '/evaluations/' + target + '/metrics.json',
        prefix + '/evaluations/' + target + '/predictions_artifact.json']
    checked = {str(root / n): inventory[n] for n in names}
    checked[annotation['file']] = annotation['sha256']
    archive = prefix + '/evaluations/' + target + '/predictions.json.gz'
    expected_binding = dict(group=group, target=target,
        projected_binding_sha256=projection['binding_sha256'], canonical_annotation=annotation,
        checked_metadata_sha256=checked,
        archive_reference=dict(file=str(root / archive), sha256=inventory[archive]))
    same(binding, expected_binding, 'Replay did not bind the exact collected cell')
    normalized = result['normalized_cell']
    expected = {k: group[k] for k in ('group_id', 'dataset', 'split', 'source', 'seed',
        'images', 'parameters_sha256')}
    expected.update(target=target, image_ids_sha256=projection['image_ids_sha256'],
        max_images=None, diagnostic_only=False, input_evidence_sha256=canonical_hash(binding))
    require(type(normalized) is dict and set(normalized) == set(expected) | {'metrics'},
        'Unexpected normalized cell schema')
    same({k: normalized[k] for k in expected}, expected, 'Normalized replay identity differs')
    raw = next(c['metrics'] for c in report['cells'] if c['target'] == target)
    metrics = normalized['metrics']
    require(type(metrics) is dict and set(metrics) == set(raw) == set(COCO_BBOX_METRICS)
        and all(inputs._number(v) and (v == -1 or 0 <= v <= 1) for v in metrics.values())
        and metrics['bbox_mAP'] >= 0, 'Replay omitted or corrupted metrics')
    require(all(abs(metrics[k] - raw[k]) <= 1e-12 for k in metrics), 'Replayed raw metrics differ')
    proof = result['replay']
    require(proof['status'] == 'cell_replay_verified_pending_pipeline_acceptance'
        and proof['comparison_tolerance'] == 1e-12 and proof['root_acceptance'] is False
        and proof['scientific_acceptance'] is False and type(proof['model_calls']) is int
        and proof['model_calls'] == 0, 'Official replay proof differs')
    same(proof['metrics'], metrics, 'Official and normalized metrics differ')
    require(inputs._number(proof['maximum_absolute_metric_delta'])
        and 0 <= proof['maximum_absolute_metric_delta'] <= 1e-12
        and inputs._number(proof['identity_resample_bbox_mAP'])
        and abs(proof['identity_resample_bbox_mAP'] - raw['bbox_mAP']) <= 1e-12,
        'Official or identity replay comparison failed')


def _source_binding(read):
    names = ('tools/vcsf_final_followup_group_qualification.py',
        'tools/vcsf_final_followup_group_inputs.py', 'tools/vcsf_final_followup_result_inputs.py',
        'tools/vcsf_final_followup_result_replay.py', 'tools/audit_vcsf_cpu_analysis.py',
        'tools/preflight_vcsf_archived_bootstrap.py',
        'src/lgp/reporting/vcsf_coco_replay.py', 'src/lgp/reporting/vcsf_prediction_evidence.py',
        'src/lgp/reporting/vcsf_image_bootstrap.py',
        'src/lgp/result_runtime.py', 'requirements/locked-cu118.txt')
    sources = {}
    for name in names:
        path = child(ROOT, name)
        sources[name] = sha(path)
        read.bytes(path, sources[name])
    return sources


def _publish(read, path, value):
    require(not path.exists() and not path.is_symlink(), 'Do not overwrite qualification evidence')
    atomic_json(path, value)
    reference = dict(file=str(path), sha256=sha(path))
    same(inputs._reference(read, reference), value, 'Published qualification artifact differs')
    return reference


def qualify_group_inputs(report_reference, *, output, registry=None, max_uncompressed_bytes=4 << 30):
    """Execute serial CPU replays and publish a fresh, bounded technical receipt.

    The caller separately authorizes RAM/scratch cost. Official replay retains
    its own server-ODA guard. No arbitrary replayer or trust callback is accepted.
    Receipt consumers must reject a leaf containing failure.json and authenticate
    its sources/artifacts; a report_content_sha256 alone never qualifies C.
    """
    _server_oda()
    require(type(max_uncompressed_bytes) is int and 0 < max_uncompressed_bytes <= 8 << 30,
        'Require an explicit positive replay cap no larger than 8 GiB')
    leaf = plain(Path(output).absolute())
    require(leaf.parent == ROOT / 'outputs/audits' / NAMESPACE and not leaf.exists(),
        'Require a new direct group qualification audit leaf')
    reference_path = collector._reference_path(report_reference)
    require(leaf not in reference_path.parents, 'Require an external collector report file')
    read = inputs._Metadata()
    results, cell_refs, returned_refs = [], [], []
    context = dict(stage='input_readback', target=None)
    leaf.mkdir(parents=True, exist_ok=False)
    try:
        sources = _source_binding(read)
        saved = inputs._reference(read, report_reference)
        _semantic(saved)
        current = _recollect(saved, registry)
        group, projection = current['group'], current['projected_inputs']
        context['group_id'] = group['group_id']
        require(type(current['group_count']) is int and current['group_count'] == 1
            and type(current['target_cells']) is int and current['target_cells'] == 16
            and current['max_images'] is None and current['diagnostic_only'] is False,
            'Require a full single-group report')
        targets = group['targets']
        require(type(targets) is list and len(targets) == 16 and all(type(t) is str
            and t and '/' not in t and '\\' not in t and t not in ('.', '..') for t in targets)
            and len(set(targets)) == 16, 'Invalid complete canonical target panel')
        same([(c['group_id'], c['target']) for c in current['cells']],
            [(group['group_id'], t) for t in targets], 'Collected cell order differs')
        inventory = _inventory(current)
        root = plain(current['execution_root'])
        require(root != leaf and root not in leaf.parents and leaf not in root.parents,
            'Qualification output overlaps original execution')
        annotation = {k: projection['annotation'][k] for k in ('file', 'sha256')}
        collector._reference_path(annotation)
        inputs_ref = _publish(read, leaf / 'recollected_inputs.json', current)
        scratch = leaf / 'scratch'
        scratch.mkdir()
        for target in targets:
            context.update(stage='official_replay', target=target)
            result = replay_new_followup_cell(group, projection, target, run_root=root,
                checked_inventory=inventory, canonical_annotation=annotation, scratch_dir=scratch,
                max_uncompressed_bytes=max_uncompressed_bytes)
            # Preserve returned diagnostics even if subsequent binding checks fail.
            cell_ref = _publish(read, leaf / 'cells' / target / 'replay.json', result)
            returned_refs.append(dict(target=target, reference=cell_ref))
            _validate_replay(result, current, target, inventory, annotation)
            results.append(result['normalized_cell'])
            cell_refs.append(dict(target=target, reference=cell_ref))
        require(not any(scratch.iterdir()), 'Replay scratch contains unfinished payloads')
        context.update(stage='final_input_readback', target=None)
        final = _recollect(current, registry)
        same(_semantic(final), _semantic(saved), 'Collector semantics changed during replay')
        final_ref = _publish(read, leaf / 'final_inputs.json', final)
        normalized_ref = _publish(read, leaf / 'normalized_cells.json', results)
        read.unchanged()
        receipt = inputs._seal(dict(schema_version=1, status=STATUS, errors=[],
            qualification_scope='one_original_full_group_inputs_and_all_twelve_metrics_per_target',
            group_id=group['group_id'], group=deepcopy(group), group_content_sha256=canonical_hash(group),
            catalogue_content_sha256=current['catalogue_content_sha256'],
            scope_sha256=current['scope_sha256'], runtime_snapshot_sha256=current['runtime_snapshot_sha256'],
            runtime_reference=deepcopy(current['runtime_reference']),
            admission_reference=deepcopy(current['admission_reference']),
            worker_group_reference=deepcopy(current['worker_group_reference']),
            worker_request_reference=deepcopy(current['worker_request_reference']),
            collector_report_reference=deepcopy(report_reference),
            collector_semantic_sha256=canonical_hash(_semantic(current)),
            semantic_exclusions=['report_content_sha256', 'completed_prefix_reference.sha256',
                'control_metadata_sha256[completed_prefix_reference.file]'],
            recollected_inputs_reference=inputs_ref, final_inputs_reference=final_ref,
            replay_references=cell_refs, normalized_cells_reference=normalized_ref,
            group_count=1, images=group['images'], target_cells=16, metric_values=16 * 12,
            targets=deepcopy(targets), max_images=None, diagnostic_only=False,
            auditor_source_sha256=sources, max_uncompressed_bytes=max_uncompressed_bytes,
            complete_group_qualification_accepted=True, official_full_panel_replayed=True,
            original_runtime_admission_reauthenticated=True,
            preserved_ancestor_failures=deepcopy(final['preserved_ancestor_failures']),
            planned_stop_cause_authenticated=False, predecessor_stop_verified=False,
            successor_admitted=False, formal_execution_admission=False, reuse_accepted=False,
            independent_result_acceptance=False, scientific_acceptance=False,
            independent_confirmation=False, formal_metrics_eligible=False,
            model_calls=0, AP_replays=16, bootstrap_replicates_computed=0))
        receipt_ref = _publish(read, leaf / 'receipt.json', receipt)
        read.unchanged()
        return dict(status=STATUS, qualification_reference=receipt_ref,
            group_id=group['group_id'], group_content_sha256=receipt['group_content_sha256'],
            complete_group_qualification_accepted=True, successor_admitted=False,
            scientific_acceptance=False, model_calls=0, AP_replays=16)
    except Exception as error:
        failure = inputs._seal(dict(schema_version=1, status='whole_group_qualification_failed',
            collector_report_reference=deepcopy(report_reference), context=deepcopy(context),
            error_type=type(error).__name__, error=str(error), verified_replay_references=cell_refs,
            returned_replay_references=returned_refs,
            completed_target_count=len(results), nested_failure_report=getattr(error, 'failure_report', None),
            complete_group_qualification_accepted=False, partial_evidence_is_acceptance=False,
            successor_admitted=False, scientific_acceptance=False, model_calls=0))
        atomic_json(leaf / 'failure.json', failure)
        raise GroupQualificationError(failure) from error
