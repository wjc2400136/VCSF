"""Replay a prespecified archived cell, then measure bounded CPU resamples.

This bounded preflight produces no model predictions, interval or matrix result.
Original accepted roots and the registered 2,000-resample definition are read-only.
"""
import argparse
from copy import copy, deepcopy
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
from io import StringIO
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

from audit_vcsf_layered_reuse import Inventory, bound, child, digest, require, verify_cell


def select_cell(plan, reuse, rows, selection='canonical_first_blackbox'):
    require(plan.get('protocol') == 'vcsf_fullval_retrospective_ablation'
        and plan.get('independent_confirmation') is False
        and plan['analysis']['uncertainty']['replicates'] == 2000, 'Wrong analysis namespace')
    require(reuse.get('status') == 'verified_reference_reuse_and_shared_provenance'
        and reuse.get('errors') == [], 'Reference reuse is not accepted')
    sources, targets = plan['sources'], plan['targets']
    require(sources and targets and len(set(sources)) == len(sources)
        and len(set(targets)) == len(targets), 'Missing or duplicate canonical panel')
    require(selection in ('canonical_first_blackbox', 'largest_reference_archive'),
        'Unknown outcome-independent resource selection rule')
    if selection == 'largest_reference_archive':
        candidates = []
        for source_index, source in enumerate(sources):
            qualified = [group for group in reuse['qualified_reference_groups']
                if group['source'] == source and group['variant'] == 'full' and group['seed'] == 42]
            require(len(qualified) == 1 and qualified[0]['status'] == 'verified_frozen_reference_reuse',
                'Resource inventory requires every qualified frozen-reference source')
            panel = [row for row in rows if row.get('attack') == 'vcsf' and row.get('source') == source]
            require(len(panel) == len(targets) and {row['target'] for row in panel} == set(targets),
                'Resource inventory requires a complete original target panel')
            for row in panel:
                require(row['parameters_sha256'] == qualified[0]['old_parameters_sha256']
                    and row['adversarial_run'] == qualified[0]['old_group'],
                    'Resource candidate is outside the accepted frozen reference')
                size = row['predictions_artifact']['uncompressed_bytes']
                require(type(size) is int and size >= 2, 'Invalid accepted archive size')
                if row['target'] != source:
                    candidates.append((-size, source_index, targets.index(row['target'])))
        require(candidates, 'No black-box resource candidate')
        _, source_index, target_index = min(candidates)
        source, target = sources[source_index], targets[target_index]
    else:
        source = sources[0]
        blackbox = [target for target in targets if target != source]
        require(blackbox, 'No canonical black-box target')
        target = blackbox[0]
    groups = [group for group in reuse['qualified_reference_groups']
        if group['source'] == source and group['variant'] == 'full' and group['seed'] == 42]
    require(len(groups) == 1 and groups[0]['status'] == 'verified_frozen_reference_reuse',
        'Selected source lacks one qualified frozen-reference group')
    selected = [row for row in rows if row.get('attack') == 'vcsf'
        and row.get('source') == source and row.get('target') == target]
    require(len(selected) == 1, 'Missing or duplicate canonical archived cell')
    row, group = selected[0], groups[0]
    require(row['parameters_sha256'] == group['old_parameters_sha256']
        and row['adversarial_run'] == group['old_group'], 'Cell does not belong to the accepted reference')
    return row, group


def verify_annotation_equivalence(canonical, archived, image_ids):
    for data in (canonical, archived):
        require(all(isinstance(data.get(key), list) for key in ('images', 'categories', 'annotations')),
            'Incomplete bound annotation structure')
        ids = [row['id'] for row in data['images']]
        require(all(type(value) is int for value in ids) and sorted(ids) == image_ids
            and len(ids) == len(set(ids)), 'Annotation image panel mismatch')
    require(canonical['categories'] == archived['categories']
        and canonical['annotations'] == archived['annotations'], 'Annotation matching semantics changed')
    def images(data):
        return sorted(({key: value for key, value in row.items() if key != 'file_name'}
            for row in data['images']), key=lambda row: row['id'])
    require(images(canonical) == images(archived), 'Image metadata changed beyond exported file names')


def verify_main_acceptance(reuse, acceptance, independent, root):
    require(acceptance.get('status') == 'accepted' and acceptance.get('formal_eligible') is True
        and independent.get('status') == 'independently_verified'
        and isinstance(independent.get('violations'), dict) and independent['violations']
        and all(type(value) is int and value == 0 for value in independent['violations'].values()),
        'Original main acceptance is incomplete or failed')
    require(Path(independent['run']).resolve() == root.resolve()
        and independent['acceptance_audit_sha256'] == reuse['main_acceptance_sha256']
        and independent['records_sha256'] == reuse['main_records_sha256']
        and acceptance['image_ids_sha256'] == reuse['image_ids_sha256'],
        'Original main acceptance identity conflict')


def load_vectorized_candidate(args, worktree):
    fields = ('vectorized_module', 'vectorized_sha256', 'vectorized_tests_receipt', 'vectorized_tests_sha256')
    supplied = [getattr(args, name) is not None for name in fields]
    require(not any(supplied) or all(supplied), 'Supply all four vectorized evidence arguments together')
    if not any(supplied):
        return None, {}
    module_path = args.vectorized_module.resolve(strict=True)
    receipt_path = args.vectorized_tests_receipt.resolve(strict=True)
    module_path.relative_to(worktree / 'outputs/audit_tools')
    receipt_path.relative_to(worktree / 'outputs/diagnostics')
    receipt = bound(receipt_path, args.vectorized_tests_sha256)
    require(receipt.get('status') == 'passed' and type(receipt.get('exit_code')) is int
        and receipt['exit_code'] == 0 and receipt.get('new_model_calls') == 0
        and receipt.get('scope') == 'isolated_vectorized_precision_exact_fresh_official_fixture_equivalence'
        and receipt.get('source_sha256', {}).get('src/lgp/reporting/vcsf_vectorized_bootstrap.py')
            == args.vectorized_sha256, 'Vectorized candidate lacks bound fixture evidence')
    require(digest(module_path) == args.vectorized_sha256, 'Vectorized candidate source changed')
    name = 'lgp.reporting._archived_probe_vectorized_candidate'
    spec = importlib.util.spec_from_file_location(name, module_path)
    require(spec is not None and spec.loader is not None, 'Cannot load bound vectorized candidate')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VectorizedCocoBootstrap, {module_path: args.vectorized_sha256,
        receipt_path: args.vectorized_tests_sha256}


def _official_resampled_precision(evaluator, positions):
    # The unchanged reference helper has already validated this exact resample.
    params = evaluator._paramsEval
    n, categories = len(params.imgIds), len(params.catIds)
    area = list(params.areaRngLbl).index('all')
    replica = copy(evaluator)
    replica.params = deepcopy(params)
    replica.params.imgIds = list(range(n))
    replica.params.areaRng = [deepcopy(params.areaRng[area])]
    replica.params.areaRngLbl = ['all']
    replica.params.maxDets = [100]
    replica._paramsEval = deepcopy(replica.params)
    replica.evalImgs = [evaluator.evalImgs[(category * len(params.areaRng) + area) * n + int(position)]
        for category in range(categories) for position in sorted(positions)]
    replica.eval = {}
    replica.accumulate()
    return replica.eval['precision'][:, :, :, 0, 0]


def paired_kernel_probe(evaluator, samples, candidate, progress):
    import numpy as np
    from lgp.reporting.vcsf_image_bootstrap import bootstrap_primary_ap

    timings, values, comparisons = [], [], []
    for index, positions in enumerate(samples):
        order = ['reference', 'vectorized'] if index % 2 == 0 else ['vectorized', 'reference']
        measured = {}
        for name in order:
            wall, cpu = time.perf_counter(), time.process_time()
            with redirect_stdout(StringIO()):
                value = bootstrap_primary_ap(evaluator, positions) if name == 'reference' else candidate.ap(positions)
            measured[name] = dict(ap=value, wall_seconds=time.perf_counter() - wall,
                cpu_seconds=time.process_time() - cpu)
        reference, fast = measured['reference'], measured['vectorized']
        require(all(math.isfinite(item['ap']) and 0 <= item['ap'] <= 1 for item in measured.values())
            and abs(reference['ap'] - fast['ap']) <= 1e-12, 'Paired bootstrap AP differs')
        with redirect_stdout(StringIO()):
            official = _official_resampled_precision(evaluator, positions)
            actual = candidate.precision(positions)
        require(np.array_equal(official, actual), 'Paired full precision tensor differs')
        require(float(np.mean(official[official != -1])) == reference['ap'],
            'Official precision oracle differs from the unchanged reference AP')
        timings.append({key: reference[key] for key in ('wall_seconds', 'cpu_seconds')})
        values.append(reference['ap'])
        comparisons.append(dict(index=index, order=order, reference=reference, vectorized=fast,
            absolute_ap_delta=abs(reference['ap'] - fast['ap']), full_precision_exact=True,
            precision_elements=int(official.size),
            precision_sha256=hashlib.sha256(official.astype('<f8', copy=False).tobytes(order='C')).hexdigest()))
        progress.update(stage='paired_timing_samples', completed_timing_samples=index + 1)
        print('[PAIRED]', index + 1, '/', len(samples),
            'reference=%.6f vectorized=%.6f precision_exact=True' %
            (reference['wall_seconds'], fast['wall_seconds']), flush=True)
    reference_mean = sum(item['reference']['wall_seconds'] for item in comparisons) / len(comparisons)
    fast_mean = sum(item['vectorized']['wall_seconds'] for item in comparisons) / len(comparisons)
    return timings, values, dict(status='verified_paired_samples_not_full_analysis', samples=comparisons,
        ap_tolerance=1e-12, full_precision_exact=True,
        reference_mean_wall_seconds=reference_mean, vectorized_mean_wall_seconds=fast_mean,
        ratio_of_mean_wall_seconds=reference_mean / fast_mean,
        timing_scope='AP calls only; candidate construction and extra precision-oracle checks excluded',
        formal_default_changed=False, independent_pipeline_acceptance=False)


def preflight(args, progress):
    require(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python"),
        'Server ODA is required')
    require(2 <= args.probe_replicates <= 32, 'Use two to thirty-two timing samples, never an interval')
    worktree = args.root.resolve(strict=True)
    plan = bound(args.research_plan, args.plan_sha256)
    reuse = bound(args.reuse_receipt, args.reuse_sha256)
    require(reuse['research_plan_sha256'] == args.plan_sha256, 'Reuse and analysis plans differ')
    source_hashes = {name: expected for name, expected in plan['file_sha256'].items()
        if name.startswith(('src/lgp/', 'configs/'))}
    require(source_hashes and all(digest(child(worktree, name)) == expected
        for name, expected in source_hashes.items()), 'Current analysis source snapshot drift')
    packages = {name: importlib.metadata.version(name) for name in reuse['packages']}
    require(packages == reuse['packages'], 'Accepted evaluator package versions changed')
    root = Path(reuse['main_root']).resolve(strict=True)
    acceptance = bound(root / 'acceptance_audit.json', reuse['main_acceptance_sha256'])
    independent = bound(args.main_independent, reuse['main_independent_sha256'])
    verify_main_acceptance(reuse, acceptance, independent, root)
    inventory = Inventory(root, bound(root / 'artifact_manifest.json', independent['artifact_manifest_sha256']))
    rows = bound(root / 'records.json', reuse['main_records_sha256'])
    row, group = select_cell(plan, reuse, rows, args.cell_selection)
    group_root = Path(group['old_group'])
    run = inventory.json(group_root / 'run.json')
    require(run['seed'] == 42 and run['parameters_sha256'] == group['old_parameters_sha256']
        and run['image_ids_sha256'] == reuse['image_ids_sha256'], 'Generation identity conflict')
    annotation_path = group_root / run['annotation']
    require(Path(row['annotation']) == annotation_path, 'Record/generation annotation conflict')
    archived = inventory.json(annotation_path)
    require(digest(annotation_path) == run['annotation_sha256'], 'Generation annotation hash conflict')
    canonical = bound(args.annotation, reuse['annotation_sha256'])
    image_ids = sorted(image['id'] for image in canonical['images'])
    require(len(image_ids) == 5000 and hashlib.sha256(json.dumps(image_ids,
        separators=(',', ':')).encode()).hexdigest() == reuse['image_ids_sha256'], 'Wrong full COCO panel')
    verify_annotation_equivalence(canonical, archived, image_ids)
    del canonical
    category_ids = sorted(category['id'] for category in archived['categories'])
    directory = root / 'evaluations' / group_root.relative_to(root / 'attacks') / row['target']
    metadata = inventory.json(directory / 'metrics.json')
    artifact = inventory.json(directory / 'predictions_artifact.json')
    verify_cell(row, metadata, artifact, reuse['image_ids_sha256'],
        reuse['target_checkpoint_sha256'][row['target']])
    metrics_hash = inventory.checked[(directory / 'metrics.json').relative_to(root).as_posix()]
    artifact_hash = inventory.checked[(directory / 'predictions_artifact.json').relative_to(root).as_posix()]
    identity = {key: row[key] for key in ('dataset', 'split', 'source', 'attack', 'target',
        'parameters_sha256', 'checkpoint_sha256', 'code_commit')}
    scratch = args.output / 'scratch'
    scratch.mkdir(exist_ok=False)
    sys.path.insert(0, str(worktree / 'src'))
    from lgp.metrics import COCO_BBOX_METRICS
    from lgp.reporting.vcsf_prediction_evidence import load_bound_predictions
    from lgp.reporting.vcsf_coco_replay import replay_coco_bbox
    from lgp.reporting.vcsf_image_bootstrap import bootstrap_primary_ap
    from lgp.reporting.vcsf_paired_statistics import image_resamples

    candidate_class, candidate_binding = load_vectorized_candidate(args, worktree)

    started = time.perf_counter()
    predictions, byte_receipt = load_bound_predictions(directory, immutable_root=root,
        metrics_sha256=metrics_hash, artifact_sha256=artifact_hash, expected_identity=identity,
        image_ids=image_ids, category_ids=category_ids, scratch_dir=scratch,
        max_uncompressed_bytes=args.max_uncompressed_bytes)
    archive_seconds = time.perf_counter() - started
    require(not list(scratch.iterdir()), 'Owned prediction scratch was not cleaned')
    progress.update(stage='bound_archive_verified', predictions=len(predictions), identity=identity)
    print('[BOUND]', identity['source'], identity['target'], len(predictions), flush=True)
    expected = {key: row['metrics'][key] for key in COCO_BBOX_METRICS}
    started = time.perf_counter()
    evaluator, replay_receipt = replay_coco_bbox(archived, predictions, expected, image_ids)
    replay_seconds = time.perf_counter() - started
    progress.update(stage='official_and_identity_replay_verified', real_archived_cell_replays=1)
    print('[REPLAY] all twelve metrics and identity resample verified', flush=True)
    del predictions, archived, rows
    candidate, candidate_setup = None, None
    if candidate_class is not None:
        wall, cpu = time.perf_counter(), time.process_time()
        with redirect_stdout(StringIO()):
            candidate = candidate_class(evaluator)
        candidate_setup = dict(wall_seconds=time.perf_counter() - wall,
            cpu_seconds=time.process_time() - cpu, cache_array_bytes=candidate.cache_array_bytes,
            identity_ap=candidate.reference_identity_ap, identity_precision_exact=True)
        print('[CANDIDATE] bound owned cache and exact identity precision verified', flush=True)
    stream = hashlib.sha256()
    samples, sample_hashes = [], []
    for index, positions in enumerate(image_resamples(len(image_ids), plan['analysis']['uncertainty'])):
        raw = positions.astype('<u4', copy=False).tobytes(order='C')
        stream.update(raw)
        if index < args.probe_replicates:
            samples.append(positions.copy())
            sample_hashes.append(hashlib.sha256(raw).hexdigest())
    timings, values, comparison = [], [], None
    if candidate is not None:
        timings, values, comparison = paired_kernel_probe(evaluator, samples, candidate, progress)
    else:
        for index, positions in enumerate(samples):
            wall, cpu = time.perf_counter(), time.process_time()
            with redirect_stdout(StringIO()):
                value = bootstrap_primary_ap(evaluator, positions)
            timings.append(dict(wall_seconds=time.perf_counter() - wall, cpu_seconds=time.process_time() - cpu))
            values.append(value)
            progress.update(stage='timing_samples', completed_timing_samples=index + 1)
            print('[PROBE]', index + 1, '/', len(samples), 'wall_seconds=%.6f' % timings[-1]['wall_seconds'], flush=True)
    expected_bound = {args.research_plan: args.plan_sha256, args.reuse_receipt: args.reuse_sha256,
        args.main_independent: reuse['main_independent_sha256'], args.annotation: reuse['annotation_sha256'],
        root / 'records.json': reuse['main_records_sha256'],
        root / 'acceptance_audit.json': reuse['main_acceptance_sha256'],
        root / 'artifact_manifest.json': independent['artifact_manifest_sha256'],
        directory / 'predictions.json.gz': artifact['archive_sha256']}
    expected_bound.update({child(root, name): expected for name, expected in inventory.checked.items()})
    expected_bound.update(candidate_binding)
    require(all(digest(path) == expected for path, expected in expected_bound.items())
        and all(digest(child(worktree, name)) == expected for name, expected in source_hashes.items()),
        'Inputs changed during real-cell preflight')
    mean_wall = sum(item['wall_seconds'] for item in timings) / len(timings)
    progress.update(stage='completed')
    resource_limits = {}
    for name in ('memory.max', 'cpu.max', 'memory/memory.limit_in_bytes', 'cpu/cpu.cfs_quota_us', 'cpu/cpu.cfs_period_us'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            resource_limits[name] = path.read_text().strip()
    status = 'verified_one_archived_cell_paired_vectorized_probe' if comparison else 'verified_one_archived_cell_replay_resource_probe'
    return dict(status=status, errors=[], vectorized_comparison=comparison, vectorized_setup=candidate_setup,
        selection_rule=args.cell_selection, selection_uses_ap=False,
        selection_scope='accepted Common-2 frozen-reference black-box cells only; not a bound for all future cells',
        identity=identity, original_group=group['old_group'], images=len(image_ids),
        annotation_sha256=reuse['annotation_sha256'], exported_annotation_sha256=run['annotation_sha256'],
        archive_binding=byte_receipt, replay=replay_receipt, archive_validation_seconds=archive_seconds,
        official_and_identity_replay_seconds=replay_seconds, measured_replicates=len(samples),
        registered_analysis_replicates=2000, full_resample_stream_sha256=stream.hexdigest(),
        measured_resample_sha256=sample_hashes, timing_sample_ap=values, timings=timings,
        mean_replicate_wall_seconds=mean_wall,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        available_cpu_affinity=len(os.sched_getaffinity(0)), cgroup_resource_limits=resource_limits,
        illustrative_serial_kernel_hours_1440_cells=mean_wall * 2000 * 1440 / 3600,
        packages=packages, source_sha256=source_hashes, research_plan_sha256=args.plan_sha256,
        reuse_receipt_sha256=args.reuse_sha256, main_independent_sha256=reuse['main_independent_sha256'],
        checked_input_sha256={str(path): expected for path, expected in expected_bound.items()},
        new_model_calls=0, new_detector_evaluations=0, real_archived_cell_replays=1,
        confidence_intervals_produced=0, analysis_pipeline_accepted=False,
        cpu_concurrency_accepted=False, efficacy_runner_armed=False,
        limits=['One accepted cell is replayed; this is not a whole-panel or full-analysis acceptance.',
            'Timing-sample AP is diagnostic only; it is not an interval, contrast, rank or selection result.',
            'The extrapolation describes only this cell and is not a full-workload resource upper bound.',
            'Process peak RSS includes provenance loading, archive materialization and the official cache.',
            'With comparison enabled, peak RSS also includes the owned candidate cache; it is not candidate-only RAM.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'research-plan', 'reuse-receipt', 'main-independent', 'annotation', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    for key in ('plan-sha256', 'reuse-sha256'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--probe-replicates', type=int, default=16)
    parser.add_argument('--cell-selection', choices=('canonical_first_blackbox', 'largest_reference_archive'),
        default='canonical_first_blackbox')
    parser.add_argument('--max-uncompressed-bytes', type=int, default=256 << 20)
    parser.add_argument('--vectorized-module', type=Path)
    parser.add_argument('--vectorized-sha256')
    parser.add_argument('--vectorized-tests-receipt', type=Path)
    parser.add_argument('--vectorized-tests-sha256')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(args.root.resolve() / 'outputs/diagnostics')
    args.output.mkdir(parents=True, exist_ok=False)
    progress = dict(stage='starting', real_archived_cell_replays=0, completed_timing_samples=0)
    result = dict(status='failed', errors=[], progress=progress,
        analysis_pipeline_accepted=False, efficacy_runner_armed=False)
    try:
        result = preflight(args, progress)
    except BaseException as exc:
        result['errors'] = [repr(exc)]
        raise
    finally:
        result.update(tool_sha256=digest(__file__), helper_sha256=digest(
            Path(__file__).with_name('audit_vcsf_layered_reuse.py')),
            completed_at=datetime.now(timezone.utc).isoformat())
        (args.output / 'preflight.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('[PASS]', result['status'], flush=True)


if __name__ == '__main__':
    main()
