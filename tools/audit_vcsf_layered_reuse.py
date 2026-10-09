"""Server-only reference reuse audit; never edits an accepted experiment root.

This revalidates selected archived observations, not a new AP evaluation or the
entire historical campaign. Non-reference historical ablations need a separate
behavioral bridge before they can reduce the prepared research matrix.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def finite_float(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite JSON number')
    return result


def decode(raw):
    def invalid(value):
        raise RuntimeError('Invalid JSON constant: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite_float,
        parse_constant=invalid)


def parameter_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def bound(path, expected):
    path = Path(path)
    require(path.is_file(), 'Missing bound file: ' + str(path))
    require(path.stat().st_size <= 512 << 20, 'Metadata resource cap exceeded')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, 'Bound hash mismatch: ' + str(path))
    return decode(raw)


def child(root, name):
    root = Path(root).resolve(strict=True)
    value = Path(name)
    require(not value.is_absolute() and '..' not in value.parts, 'Escaping artifact path')
    path = root / value
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root),
        'Artifact symlink is not accepted')
    require(root in path.resolve().parents, 'Artifact must be below the evidence root')
    return path


class Inventory:
    def __init__(self, root, manifest):
        self.root = Path(root).resolve(strict=True)
        self.entries = {}
        self.checked = {}
        for row in manifest['files']:
            name = row['path']
            child(self.root, name)
            require(name not in self.entries, 'Duplicate artifact path')
            self.entries[name] = row

    def file(self, path):
        path = Path(path)
        name = path.relative_to(self.root).as_posix()
        require(name in self.entries, 'File is not in accepted inventory: ' + name)
        target = child(self.root, name)
        row = self.entries[name]
        require(target.is_file() and target.stat().st_size == row['bytes'],
            'Artifact missing or byte count changed: ' + name)
        require(digest(target) == row['sha256'], 'Artifact SHA-256 mismatch: ' + name)
        self.checked[name] = row['sha256']
        return target

    def json(self, path):
        path = self.file(path)
        return bound(path, self.entries[path.relative_to(self.root).as_posix()]['sha256'])


def verify_manifest_row(row, position, image_id, seed):
    require(row.get('status') == 'ok' and row.get('position') == position
        and row.get('image_id') == image_id and row.get('attack_seed') == seed + position,
        'Manifest status, order or seed mismatch')
    require(type(row.get('linf_pixel')) in (int, float)
        and math.isfinite(row['linf_pixel']) and 0 <= row['linf_pixel'] <= 4.0 + 1e-6,
        'Manifest L-infinity violation')
    require(type(row.get('actual_gradient_evaluations')) is int
        and row['actual_gradient_evaluations'] == 20,
        'Logical gradient accounting mismatch')
    require(row.get('auxiliary_forward_passes') == 0
        and row.get('auxiliary_backward_passes') == 0, 'Explicit auxiliary count mismatch')
    diagnostics = row.get('diagnostics', [])
    require(len(diagnostics) == 20, 'Missing full diagnostics')
    for index, item in enumerate(diagnostics):
        require(item.get('step') == index
            and item.get('phase') == ('risk_initialization' if index == 0 else 'feature')
            and item.get('uncertainty_gate_active') is False
            and item.get('risk_direction_carryover_applied') is False,
            'Reference phase schedule or state changed')


def verify_archive(path, artifact):
    require(artifact.get('status') == 'verified_lossless_archive'
        and artifact.get('format') == 'gzip'
        and artifact.get('archive_file') == 'predictions.json.gz', 'Wrong archive schema')
    require(path.stat().st_size == artifact['archive_bytes']
        and digest(path) == artifact['archive_sha256'], 'Compressed archive changed')
    h, size = hashlib.sha256(), 0
    with gzip.open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            size += len(block)
            require(size <= artifact['uncompressed_bytes'], 'Archive decompression overflow')
            h.update(block)
    require(size == artifact['uncompressed_bytes']
        and h.hexdigest() == artifact['uncompressed_sha256'], 'Uncompressed archive changed')
    require(digest(path) == artifact['archive_sha256'], 'Archive changed during read')
    return {'archive_sha256': artifact['archive_sha256'],
        'uncompressed_sha256': h.hexdigest(), 'uncompressed_bytes': size}


def verify_cell(row, metadata, artifact, ids_hash, checkpoint_hash):
    for key in ('status', 'dataset', 'split', 'source', 'attack', 'target', 'images',
            'parameters_sha256', 'checkpoint_sha256', 'code_commit', 'metrics',
            'predictions_sha256', 'failures'):
        require(key in row and key in metadata and row[key] == metadata[key],
            'Record/cell mismatch: ' + key)
    require(row['status'] == 'complete' and row['failures'] == [] and row['images'] == 5000
        and row['dataset'] == 'coco' and row['split'] == 'val', 'Incomplete or wrong-scope cell')
    require(row['checkpoint_sha256'] == checkpoint_hash, 'Cell checkpoint mismatch')
    for item in (row, metadata):
        require(item.get('evaluated_image_ids_sha256') == ids_hash
            and item.get('expected_image_ids_sha256') == ids_hash
            and item.get('evaluated_image_ids_match_expected') is True, 'Cell image set mismatch')
        require(item.get('predictions_artifact') == artifact
            and item.get('predictions_sha256') == artifact['uncompressed_sha256'],
            'Cell/archive binding mismatch')


def _git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), '--no-pager', *args], text=True).strip()


def audit(args):
    require(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        'Run this audit only in server ODA')
    worktree = args.worktree.resolve(strict=True)
    require(not _git(worktree, 'status', '--porcelain', '--untracked-files=no'), 'Tracked worktree is dirty')
    campaign = bound(args.campaign, args.campaign_sha256)
    require(campaign['status'] == 'accepted' and campaign['formal_eligible'] is True,
        'Campaign is not accepted')
    stage = next(s for s in campaign['stages'] if s['stage'] == 'coco_canonical')
    canonical = bound(stage['acceptance'], stage['acceptance_sha256'])
    reference = canonical['refreshed_reference']
    root = Path(reference['run']).resolve(strict=True)
    acceptance = bound(root / 'acceptance_audit.json', reference['acceptance_sha256'])
    independent = bound(reference['independent_acceptance'], reference['independent_acceptance_sha256'])
    require(acceptance['status'] == 'accepted' and acceptance['formal_eligible'] is True
        and independent['status'] == 'independently_verified'
        and all(value == 0 for value in independent['violations'].values()), 'Main acceptance failed')
    require(Path(independent['run']).resolve() == root
        and independent['acceptance_audit_sha256'] == reference['acceptance_sha256']
        and independent['records_sha256'] == reference['records_sha256'], 'Main acceptance identity conflict')
    inventory = Inventory(root, bound(root / 'artifact_manifest.json', independent['artifact_manifest_sha256']))
    rows = bound(root / 'records.json', reference['records_sha256'])
    require(len(rows) == 208 and all(row['status'] == 'complete' for row in rows), 'Main records changed')
    provenance = inventory.json(root / 'provenance.json')

    fixed = bound(args.reference_receipt, args.reference_sha256)
    require(fixed['status'] == 'independently_verified_fixed_state' and fixed['errors'] == [],
        'Current fixed-state gate has not passed')
    fixed_root = Path(fixed['run'])
    producer = bound(fixed_root / 'reference_receipt.json', fixed['receipt_sha256'])
    current = bound(fixed_root / 'plan.json', producer['plan_sha256'])
    require(current['git_head'] == _git(worktree, 'rev-parse', 'HEAD'), 'Current reference Git mismatch')
    for name, expected in current['file_sha256'].items():
        require(digest(child(worktree, name)) == expected, 'Reference snapshot changed: ' + name)
    plan = bound(args.research_plan, args.research_plan_sha256)
    require(plan['protocol'] == 'vcsf_fullval_retrospective_ablation'
        and plan['independent_confirmation'] is False, 'Wrong research namespace')
    for name, expected in plan['file_sha256'].items():
        require(digest(child(worktree, name)) == expected, 'Research snapshot changed: ' + name)

    # These are the shared model/data/evaluator dependencies, not all historical
    # reporting and scheduling files. Changed dispatch files remain explicit below.
    source_files = {r['path']: r['sha256'] for r in provenance['source_manifest']['files']}
    exact_paths = [name for name in source_files if name.startswith(('src/lgp/adapters/', 'src/lgp/data/'))]
    exact_paths += ['configs/models.yaml', 'configs/datasets/coco.yaml', 'src/lgp/modeling.py',
        'src/lgp/runtime_config.py', 'src/lgp/mmdet_compat.py', 'src/lgp/io.py', 'src/lgp/paths.py',
        'src/lgp/metrics.py', 'src/lgp/attacks/base.py', 'src/lgp/attacks/common.py',
        'src/lgp/attacks/vcsf_final_candidate.py', 'src/lgp/runners/evaluate.py',
        'src/lgp/runners/clean.py', 'src/lgp/runners/artifacts.py', 'src/lgp/runners/weights.py']
    exact = {}
    for name in exact_paths:
        actual = digest(child(worktree, name))
        require(actual == source_files[name], 'Shared dependency differs: ' + name)
        exact[name] = actual
    packages = {}
    for name, expected in provenance['packages'].items():
        actual = importlib.metadata.version(name)
        require(actual == expected, 'Package version changed: ' + name)
        packages[name] = actual
    require(sys.version.split()[0] == provenance['python'], 'Python version changed')
    require(current['targets'] == plan['targets'], 'Target order differs')
    checkpoints = {}
    for name in plan['targets']:
        item = current['checkpoints'][name]
        actual = digest(item['path'])
        require(actual == item['sha256'] == acceptance['target_checkpoint_sha256'][name],
            'Current checkpoint differs: ' + name)
        checkpoints[name] = actual
        print('[CHECKPOINT]', name, 'verified', flush=True)
    annotation = bound(current['annotation'], current['annotation_sha256'])
    image_ids = sorted(image['id'] for image in annotation['images'])
    require(len(image_ids) == 5000 and len(set(image_ids)) == 5000, 'Wrong COCO image panel')
    ids_hash = hashlib.sha256(json.dumps(image_ids, separators=(',', ':')).encode()).hexdigest()
    require(ids_hash == acceptance['image_ids_sha256'] == current['all_image_ids_sha256'], 'COCO IDs differ')

    sys.path.insert(0, str(worktree / 'src'))
    from lgp.attacks.vcsf_research_isolated import VCSFResearchConfig
    groups = [g for g in plan['groups'] if g['variant'] == 'full' and g['seed'] == 42]
    require(len(groups) == 2 and [g['source'] for g in groups] == plan['sources'],
        'Reference reuse must cover exactly Common-2 seed 42')
    qualified, cells = [], []
    for group in groups:
        config = VCSFResearchConfig.from_mapping(group['parameters'])
        require(config.is_reference(), 'A changed candidate cannot reuse the frozen reference')
        panel = [row for row in rows if row.get('attack') == 'vcsf' and row.get('source') == group['source']]
        require(len(panel) == 16 and {r['target'] for r in panel} == set(plan['targets']), 'Reference panel incomplete')
        group_root = Path(panel[0]['adversarial_run'])
        require(all(Path(row['adversarial_run']) == group_root for row in panel), 'Mixed generation roots')
        run = inventory.json(group_root / 'run.json')
        reference_parameters = json.loads(json.dumps(config.reference_parameters()))
        require(run['parameters'] == reference_parameters and run['parameters_sha256'] == parameter_digest(reference_parameters)
            and all(row['parameters_sha256'] == run['parameters_sha256'] for row in panel), 'Reference parameter mismatch')
        require(run['status'] == 'complete' and run['seed'] == 42 and run['seed_schedule'] == 'selected_position'
            and run['successful_images'] == 5000 and run['failed_images'] == 0
            and run['image_ids_sha256'] == ids_hash and run['requested_ordered_image_ids_sha256'] == ids_hash,
            'Reference generation identity mismatch')
        require(run['checkpoint_sha256'] == checkpoints[group['source']], 'Source checkpoint changed')
        manifest = inventory.file(group_root / run['manifest'])
        require(digest(manifest) == run['manifest_sha256'], 'Manifest/run hash conflict')
        png_count = 0
        with manifest.open('r', encoding='utf-8') as handle:
            for position, line in enumerate(handle):
                require(position < len(image_ids), 'Extra manifest image')
                row = decode(line)
                verify_manifest_row(row, position, image_ids[position], run['seed'])
                png = inventory.file(child(group_root, row['output_file']))
                require(digest(png) == row['output_sha256'] and png.stat().st_size == row['output_bytes'],
                    'PNG/manifest binding mismatch')
                png_count += 1
        require(png_count == 5000, 'Short manifest')
        require(len(list((group_root / 'images').glob('*.png'))) == 5000, 'PNG directory count mismatch')
        qualified.append({'group_id': group['group_id'], 'variant': group['variant'], 'seed': 42,
            'source': group['source'], 'status': 'verified_frozen_reference_reuse',
            'old_group': str(group_root), 'old_parameters_sha256': run['parameters_sha256'],
            'research_parameters_sha256': group['parameters_sha256'], 'images': png_count,
            'evaluations': 16, 'blackbox_cells': 15, 'manifest_sha256': run['manifest_sha256']})
        cells.extend(panel)
        print('[GROUP]', group['source'], '5000 PNG hashes and phase ledgers verified', flush=True)
    cells += [row for row in rows if row['attack'] == 'clean']
    require(len(cells) == 48, 'Expected 32 attack and 16 clean cells')
    archive_checks = []
    for row in cells:
        if row['attack'] == 'clean':
            directory = root / 'evaluations/coco/clean' / row['target']
        else:
            relative = Path(row['adversarial_run']).relative_to(root / 'attacks')
            directory = root / 'evaluations' / relative / row['target']
        metadata = inventory.json(directory / 'metrics.json')
        artifact = inventory.json(directory / 'predictions_artifact.json')
        verify_cell(row, metadata, artifact, ids_hash, checkpoints[row['target']])
        archive = inventory.file(directory / 'predictions.json.gz')
        checked = verify_archive(archive, artifact)
        archive_checks.append(dict(checked, path=str(archive), source=row['source'], target=row['target']))
        print('[ARCHIVE]', len(archive_checks), '/48', row['source'], row['target'], flush=True)

    old_stage = next(s for s in campaign['stages'] if s['stage'] == 'vcsf_submission_final_exact_ablation')
    old_rows = bound(Path(old_stage['root']) / 'records.json', old_stage['records_sha256'])
    pending = {}
    for row in old_rows:
        if row.get('variant') in ('neck_identity_reset', 'neck_scale_reset',
                'cross_stage_identity_reset', 'cross_stage_scale_feature_only'):
            key = (row['variant'], row['source'])
            pending[key] = {'variant': key[0], 'source': key[1], 'old_group': row['adversarial_run'],
                'status': 'pending_historical_behavior_bridge_do_not_subtract'}
    require(len(pending) == 8, 'Historical non-reference inventory differs')
    require(not _git(worktree, 'status', '--porcelain', '--untracked-files=no'), 'Worktree changed during audit')
    require(digest(args.campaign) == args.campaign_sha256
        and digest(root / 'artifact_manifest.json') == independent['artifact_manifest_sha256']
        and digest(root / 'records.json') == reference['records_sha256'], 'Evidence changed during audit')
    return {'schema_version': 1, 'status': 'verified_reference_reuse_and_shared_provenance', 'errors': [],
        'completed_at': datetime.now(timezone.utc).isoformat(), 'current_git': current['git_head'],
        'campaign_sha256': args.campaign_sha256, 'main_root': str(root),
        'main_records_sha256': reference['records_sha256'], 'main_acceptance_sha256': reference['acceptance_sha256'],
        'main_independent_sha256': reference['independent_acceptance_sha256'],
        'reference_receipt_sha256': args.reference_sha256, 'research_plan_sha256': args.research_plan_sha256,
        'unchanged_shared_dependencies': exact, 'packages': packages, 'target_checkpoint_sha256': checkpoints,
        'image_ids_sha256': ids_hash, 'annotation_sha256': current['annotation_sha256'],
        'qualified_reference_groups': qualified, 'pending_historical_groups': list(pending.values()),
        'verified_png_files': 10000, 'verified_prediction_archives': 48,
        'prediction_archive_checks': archive_checks, 'artifact_files_rehashed': len(inventory.checked),
        'artifact_subset_sha256': hashlib.sha256(json.dumps(inventory.checked, sort_keys=True).encode()).hexdigest(),
        'remaining_new_groups_upper_bound': len(plan['groups']) - len(qualified),
        'new_AP_evaluations': 0, 'efficacy_runner_armed': False, 'complete_reuse_adjudication': False,
        'scope': 'Exact frozen reference observations only; non-reference historical behavior remains unresolved.',
        'accounting': '20 logical gradients and zero explicitly charged auxiliary calls; reference-side work is not zero physical cost.',
        'limits': ['No new pixel-decoding L-infinity audit; accepted manifest bounds and identical PNG bytes verified.',
            'No new AP replay; original accepted metrics and lossless prediction bytes verified.',
            'Not a full historical-campaign reacceptance or a launch authorization.',
            'Shared evaluator files are identical; changed research dispatch still requires the new lifecycle gate.'],
        'auditor_sha256': digest(__file__)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('worktree', 'campaign', 'reference-receipt', 'research-plan', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('campaign-sha256', 'reference-sha256', 'research-plan-sha256'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(args.worktree.resolve() / 'outputs' / 'audits' in output.parents,
        'Audit output must be a new audits directory')
    require(not output.exists(), 'Never overwrite an audit output')
    output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args)
    except Exception as exc:
        result = {'status': 'failed', 'errors': [str(exc)], 'new_AP_evaluations': 0,
            'efficacy_runner_armed': False, 'auditor_sha256': digest(__file__)}
        (output / 'reuse_acceptance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        raise
    path = output / 'reuse_acceptance.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (output / 'reuse_acceptance.sha256').write_text(digest(path) + '  reuse_acceptance.json\n', encoding='ascii')
    print('[PASS]', result['status'], str(path), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
