"""Independently audit complete archive-analysis cells and paired matrix algebra.

This auditor does not import the production assembler or statistical helpers.
It consumes already computed AP bytes, never invokes a detector or resamples
predictions, and never promotes a candidate or accepts the original GPU root.
"""
import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np


TOLERANCE = 1e-12


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode('utf-8')).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def plain(path):
    path = Path(path)
    check(path.is_absolute() and '..' not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)), 'Unsafe evidence path')
    return path


def child(root, name):
    relative = Path(name)
    check(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe relative evidence path')
    return plain(plain(root) / relative)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        check(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def no_constant(value):
    raise RuntimeError('Non-finite JSON constant: ' + value)


def signature(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class Evidence:
    """Bind immutable input bytes and rehash them before publishing the audit."""

    def __init__(self):
        self.checked = {}
        self.signatures = {}

    def bytes(self, path, expected, cap=None):
        path = plain(path)
        name = str(path)
        check(is_sha(expected) and (name not in self.checked or self.checked[name] == expected),
            'Conflicting or invalid expected evidence hash')
        before = signature(path.stat())
        check(cap is None or before[2] <= cap, 'Evidence exceeds its byte limit')
        check(name not in self.signatures or before == self.signatures[name], 'Previously read evidence changed')
        digest, blocks = hashlib.sha256(), [] if cap is not None else None
        with path.open('rb') as handle:
            check(signature(os.fstat(handle.fileno())) == before, 'Evidence changed before reading')
            count = 0
            for block in iter(lambda: handle.read(8 << 20), b''):
                count += len(block)
                check(cap is None or count <= cap, 'Evidence grew beyond its byte limit')
                digest.update(block)
                if blocks is not None:
                    blocks.append(block)
            check(signature(os.fstat(handle.fileno())) == before, 'Evidence changed while reading')
        check(count == before[2] and digest.hexdigest() == expected
            and signature(path.stat()) == before, 'Evidence bytes differ: ' + path.name)
        self.checked[name], self.signatures[name] = expected, before
        return None if blocks is None else b''.join(blocks)

    def json(self, path, expected):
        return json.loads(self.bytes(path, expected, 128 << 20),
            object_pairs_hook=no_duplicates, parse_constant=no_constant)

    def bind_map(self, mapping):
        check(type(mapping) is dict and mapping, 'Missing immutable input map')
        for name, expected in mapping.items():
            if str(plain(name)) in self.checked:
                check(self.checked[name] == expected, 'Conflicting transitive input hash')
            else:
                self.bytes(name, expected)

    def unchanged(self):
        check(all(signature(plain(name).stat()) == expected and sha(name) == self.checked[name]
            for name, expected in self.signatures.items()),
            'An audited input changed before completion')


def same(left, right, message):
    check(value_sha(left) == value_sha(right), message)


def number(value, label, low=None, high=None):
    check(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric ' + label)
    check((low is None or value >= low) and (high is None or value <= high), 'Out-of-range ' + label)
    return float(value)


def near(actual, expected, label):
    actual = number(actual, label)
    expected = number(expected, label)
    difference = abs(actual - expected)
    check(difference <= TOLERANCE, 'Independent arithmetic differs: ' + label)
    return difference


def array(values, size, label):
    check(type(values) in (list, np.ndarray), 'Missing numeric vector: ' + label)
    check(not any(isinstance(value, (bool, np.bool_)) for value in values), 'Boolean numeric vector: ' + label)
    result = np.asarray(values)
    check(result.shape == (size,) and result.dtype.kind in 'fi' and np.isfinite(result).all()
        and ((result >= 0) & (result <= 1)).all(), 'Invalid complete raw-AP vector: ' + label)
    return result.astype(np.float64, copy=False)


def parse_arguments(argv):
    check(type(argv) is list and len(argv) % 2 == 0 and all(type(x) is str for x in argv),
        'Unexpected CPU argument vector')
    result = {}
    for flag, value in zip(argv[::2], argv[1::2]):
        check(flag.startswith('--') and flag not in result, 'Duplicate or positional CPU argument')
        result[flag] = value
    return result


def verify_invocation(read, row, task, plan, cpu_plan_hash, coordinator, expected_arguments):
    directory = plain(row['launch_root'])
    expected_dir = plain(plan['inputs']['execution_worktree']) / 'outputs/diagnostics/vcsf_cpu_analysis' / plan['grid']['output_id'] / 'workers' / task['task_id']
    check(directory == expected_dir, 'CPU task has another launch directory')
    claimed = row['invocation']
    result = read.json(directory / 'worker_result.json', claimed['receipt_sha256'])
    same(result, {key: value for key, value in claimed.items() if key != 'receipt_sha256'}, 'Invocation receipt was relabeled')
    request = read.json(directory / 'request.json', result['request_sha256'])
    launch = read.json(directory / 'launch.json', sha(directory / 'launch.json'))
    started = read.json(directory / 'worker_started.json', sha(directory / 'worker_started.json'))
    check(result.get('status') == 'complete_cpu_invocation_pending_cell_acceptance'
        and result.get('exit_code') == 0 and type(result.get('exit_code')) is int and result.get('failure') is None
        and request.get('status') == 'declared_cpu_tool_invocation'
        and request.get('execution_plan_sha256') == cpu_plan_hash
        and result.get('job_id') == request.get('job_id') == launch.get('job_id') == task['task_id'],
        'CPU invocation is incomplete or belongs to another task')
    owner = dict(coordinator_pid=coordinator['pid'], coordinator_start_ticks=coordinator['start_ticks'])
    check(all(result.get(key) == value and request.get(key) == value for key, value in owner.items())
        and launch.get('coordinator_pid') == coordinator['pid']
        and type(result.get('worker_pid')) is int and result['worker_pid'] > 1
        and type(result.get('worker_start_ticks')) is int and result['worker_start_ticks'] > 0
        and launch.get('pid') == result['worker_pid'] == result.get('process_group') == launch.get('process_group')
        and launch.get('start_ticks') == result['worker_start_ticks']
        and launch.get('request_sha256') == result['request_sha256'], 'CPU worker ownership differs')
    start_fields = {'schema_version', 'status', 'job_id', 'request_sha256', 'worker_pid', 'worker_start_ticks',
        'coordinator_pid', 'coordinator_start_ticks', 'process_group', 'worker_address_space_bytes',
        'entrypoint_sha256', 'wrapper_sha256', 'started_at', 'scientific_acceptance', 'analysis_pipeline_accepted'}
    check(set(started) == start_fields and started.get('schema_version') == 1
        and all(started[key] == result[key] for key in start_fields - {'status'})
        and started.get('status') == 'running', 'Exclusive worker-start claim differs')
    for flag in ('scientific_acceptance', 'analysis_pipeline_accepted'):
        check(result.get(flag) is False, 'Invocation claims unsupported acceptance')
    cap = plan['resource_contract']['worker_address_space_bytes']
    check(result.get('worker_address_space_bytes') == request.get('worker_address_space_bytes') == cap,
        'CPU worker resource declaration changed')
    check(request.get('reservation') == coordinator['reservation'], 'CPU task used another scope reservation')
    entry = {'original_cell': 'run_vcsf_indexed_bootstrap_cell.py', 'new_cell': 'run_vcsf_new_bootstrap_cell.py',
        'bind_new_group': 'build_vcsf_new_group_inputs.py'}[task['kind']]
    expected_entry = child(plain(plan['source_snapshot']), 'tools/' + entry)
    check(request.get('entrypoint') == str(expected_entry)
        and result.get('entrypoint_sha256') == request.get('entrypoint_sha256') == plan['implementation_sha256']['tools/' + entry]
        and result.get('wrapper_sha256') == request.get('wrapper_sha256')
            == plan['implementation_sha256']['src/lgp/reporting/vcsf_cpu_processes.py'], 'CPU invocation source differs')
    worktree = plan['inputs']['original_worktree' if task['kind'] == 'original_cell' else 'execution_worktree']
    check(request.get('worktree') == worktree and parse_arguments(request.get('argv')) == expected_arguments,
        'CPU invocation arguments differ from the prospective task')
    start = datetime.fromisoformat(result['started_at'])
    stop = datetime.fromisoformat(result['completed_at'])
    check(start.tzinfo is not None and stop.tzinfo is not None and stop > start, 'Invalid invocation time interval')
    for key in ('wall_seconds', 'cpu_seconds', 'process_peak_rss_bytes'):
        number(result.get(key), key, 0)
    process = Path('/proc') / str(result['worker_pid'])
    if process.exists():
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        check(int(fields[19]) != result['worker_start_ticks'], 'An accepted CPU worker is still present')
    return result


def verify_cell(read, task, row, plan, expected_cell, expected_index_hash, expected_tests_hash, expected_input, invocation=None):
    from lgp.metrics import COCO_BBOX_METRICS

    path = plain(row['receipt_file'])
    qualified = task.get('qualified_complete_reuse')
    if qualified is None:
        filename = 'new_cell.json' if task['kind'] == 'new_cell' else 'indexed_cell.json'
        check(path == plain(task['output']) / filename, 'Cell receipt left its declared output directory')
    else:
        check(path == plain(qualified['receipt_file']) and row['receipt_sha256'] == qualified['receipt_sha256'],
            'Qualified cell receipt differs from its original immutable location')
    outer = read.json(path, row['receipt_sha256'])
    new = task['kind'] == 'new_cell'
    status = 'complete_new_cell_pending_pipeline_acceptance' if new else 'complete_indexed_cell_pending_pipeline_acceptance'
    check(outer.get('status') == status and outer.get('errors') == []
        and all(outer.get(key) == task[key] for key in ('group_id', 'source', 'target', 'variant', 'seed'))
        and outer.get('index_sha256') == expected_index_hash and outer.get('tests_receipt_sha256') == expected_tests_hash,
        'Cell completion or index identity differs')
    for key in ('scientific_acceptance', 'analysis_pipeline_accepted', 'cpu_concurrency_accepted'):
        check(outer.get(key) is False, 'Cell claims unsupported acceptance')
    check(all(type(outer.get(key)) is int and outer[key] == 0 for key in
        ('new_model_calls', 'new_detector_evaluations', 'confidence_intervals_produced'))
        and outer.get('real_archived_cell_replays') == 1, 'Cell changed its numerical execution scope')
    same(outer['normalized_cell'], expected_cell, 'Normalized observation changed after input binding')
    cell = outer['normalized_cell']
    check(cell.get('status') == 'complete' and cell.get('failures') == []
        and all(cell.get(key) == task[key] for key in ('group_id', 'source', 'target', 'variant', 'seed', 'images', 'parameters_sha256'))
        and cell.get('image_ids_sha256') == plan['execution']['image_ids_sha256'], 'Normalized scientific identity differs')
    check(all(plan['implementation_sha256'].get(name) == digest for name, digest in
        outer['implementation_sha256'].items())
        and outer['implementation_sha256'], 'Cell implementation differs from the CPU snapshot')
    same(outer.get('packages'), plan['execution']['packages'], 'Cell environment differs')
    if new:
        check(outer.get('policy_lock_sha256') == plan['inputs']['evidence']['policy_lock']['sha256']
            and outer.get('execution_plan_sha256') == plan['grid']['execution_plan_sha256']
            and outer.get('runtime_snapshot_sha256') == plan['execution']['runtime_snapshot_sha256'],
            'New cell lost its policy or runtime binding')
    else:
        check(outer.get('paired_receipt_sha256') == plan['policy']['bound_evidence']['paired_kernel']['sha256'],
            'Original cell used another paired-kernel validation')
    replay = read.json(path.parent / 'input_replay.json', outer['input_replay_sha256'])
    check(replay.get('index_sha256') == expected_index_hash and replay.get('normalized_cell_sha256') == value_sha(cell)
        and replay.get('input_sha256') == cell['input_evidence_sha256'], 'Official replay has another input binding')
    same(replay['shared_validation'], plan['shared_validation'], 'Cell consumed another shared resampling stream')
    source = replay['replay']
    check(source.get('status') == 'cell_replay_verified_pending_pipeline_acceptance'
        and source.get('comparison_tolerance') == TOLERANCE
        and set(source.get('metrics', {})) == set(COCO_BBOX_METRICS)
        and source.get('model_calls') == 0 and source.get('root_acceptance') is False
        and source.get('scientific_acceptance') is False, 'Official twelve-metric replay is missing')
    check(number(source.get('maximum_absolute_metric_delta'), 'maximum metric delta', 0) <= TOLERANCE,
        'Official twelve-metric replay exceeds tolerance')
    for key in COCO_BBOX_METRICS:
        metric = number(cell['metrics'][key], key, -1, 1)
        check(metric == -1 or metric >= 0, 'Invalid undefined COCO metric')
        near(source['metrics'][key], metric, key)
    check(cell['bbox_mAP'] == cell['metrics']['bbox_mAP'], 'Normalized AP is not the original unrounded metric')
    near(source['identity_resample_bbox_mAP'], cell['bbox_mAP'], 'identity AP')
    archive = replay['archive_binding']
    check(archive.get('status') == 'cell_bytes_verified_pending_root_and_replay_acceptance'
        and archive.get('image_ids_sha256') == cell['image_ids_sha256']
        and archive.get('scientific_acceptance') is False and archive.get('root_acceptance') is False,
        'Prediction byte replay is not bound to the complete image set')
    same(archive['metrics']['metrics'], cell['metrics'], 'Archived metrics were not the replayed metrics')
    artifact = archive['artifact']
    check(artifact.get('status') == 'verified_lossless_archive'
        and artifact.get('uncompressed_sha256') == archive['metrics'].get('predictions_sha256')
        and artifact.get('prediction_records') == archive['metrics'].get('detections'), 'Archive byte provenance differs')
    hashes = outer['checked_input_sha256']
    check(expected_input['input_sha256'] == cell['input_evidence_sha256']
        and archive['metrics_sha256'] == expected_input['metadata_sha256']
        and archive['artifact_sha256'] == expected_input['artifact_sha256']
        and artifact['archive_sha256'] == expected_input['archive_sha256'], 'Replay consumed another sealed prediction input')
    same(archive['metrics'], expected_input['metadata'], 'Replay metadata differs from the sealed source JSON')
    same(artifact, expected_input['artifact'], 'Replay archive differs from the sealed source JSON')
    for name in ('input', 'metadata', 'artifact', 'archive'):
        check(hashes.get(expected_input[name + '_file']) == expected_input[name + '_sha256'],
            'Cell omitted the exact sealed prediction or input path binding')
    read.bind_map(hashes)
    bootstrap = None
    if task['bootstrap_replicates'] == 0:
        check(cell['source'] == cell['target'] and outer.get('bootstrap_status') == 'not_required_whitebox'
            and outer.get('bootstrap') is None and outer.get('bootstrap_receipt_sha256') is None,
            'White-box diagnostic entered the bootstrap family')
    else:
        check(cell['source'] != cell['target'] and outer.get('bootstrap_status') == 'complete', 'Black-box bootstrap is incomplete')
        boot = read.json(path.parent / 'bootstrap/bootstrap_cell.json', outer['bootstrap_receipt_sha256'])
        same(boot, outer['bootstrap'], 'Embedded bootstrap receipt differs from its actual file')
        count = plan['execution']['analysis']['uncertainty']['replicates']
        check(boot.get('status') == 'complete' and boot.get('failures') == []
            and type(boot.get('replicates_completed')) is int and type(boot.get('replicates_expected')) is int
            and boot['replicates_completed'] == boot['replicates_expected'] == task['bootstrap_replicates'] == count
            and boot.get('ap_dtype') == 'float64_little_endian' and boot.get('ap_file') == 'bootstrap_ap.f8le'
            and type(boot.get('ap_bytes')) is int and boot['ap_bytes'] == count * 8,
            'Black-box AP byte shape is incomplete or changed')
        check(all(boot.get(key) == cell[key] for key in
            ('group_id', 'source', 'target', 'variant', 'seed', 'images', 'parameters_sha256'))
            and boot.get('normalized_cell_sha256') == value_sha(cell)
            and boot.get('input_evidence_sha256') == cell['input_evidence_sha256'], 'Bootstrap observation identity differs')
        for actual, expected in (('resample_metadata_sha256', 'metadata_sha256'),
                ('resample_stream_sha256', 'payload_sha256'), ('resample_definition_sha256', 'definition_sha256')):
            check(boot.get(actual) == plan['shared_validation'][expected], 'Bootstrap paired stream differs')
        for actual, name in (('kernel_sha256', 'vcsf_vectorized_bootstrap.py'),
                ('worker_source_sha256', 'vcsf_bootstrap_cell.py'), ('reference_source_sha256', 'vcsf_image_bootstrap.py')):
            check(boot.get(actual) == plan['implementation_sha256']['src/lgp/reporting/' + name], 'Bootstrap source differs')
        check(boot.get('kernel') == 'vectorized_owned_cache' and boot.get('new_model_calls') == 0
            and all(boot.get(key) is False for key in ('scientific_acceptance', 'original_root_acceptance', 'analysis_pipeline_accepted')),
            'Bootstrap kernel or acceptance scope differs')
        near(boot.get('identity_ap'), cell['bbox_mAP'], 'bootstrap identity AP')
        check(number(boot.get('identity_absolute_delta'), 'bootstrap identity delta', 0) <= TOLERANCE,
            'Bootstrap identity replay failed')
        if invocation is not None:
            check(boot.get('worker_pid') == invocation['worker_pid'] and boot.get('parent_pid') == invocation['coordinator_pid'],
                'Bootstrap was not executed by its registered CPU worker')
            check(datetime.fromisoformat(invocation['started_at']) <= datetime.fromisoformat(boot['started_at'])
                <= datetime.fromisoformat(boot['completed_at']) <= datetime.fromisoformat(invocation['completed_at']),
                'Bootstrap execution leaves its owned invocation interval')
        raw = read.bytes(path.parent / 'bootstrap/bootstrap_ap.f8le', boot['ap_values_sha256'], count * 8)
        check(len(raw) == count * 8, 'Truncated AP bytes')
        values = np.frombuffer(raw, dtype='<f8')
        array(values, count, 'bootstrap AP')
        bootstrap = dict(boot, ap_values=values)
    return cell, bootstrap


def linear_quantile(values, probability):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def verify_matrix(plan, cells, bootstraps, shared, result):
    """Reconstruct cells, equal-cell means, seeded contrasts and max-error bands."""
    groups = sorted(plan['groups'], key=lambda group: group['group_id'])
    sources, targets = plan['sources'], plan['targets']
    check(groups and sources and targets and len(set(sources)) == len(sources)
        and len(set(targets)) == len(targets) and set(sources) <= set(targets), 'Invalid planned panel')
    count = plan['analysis']['uncertainty']['replicates']
    check(type(count) is int and count >= 2 and shared['shape'] == [count, groups[0]['images']]
        and shared['definition_sha256'] == value_sha(plan['analysis']['uncertainty']), 'Different shared sample definition')
    expected = [(group['group_id'], target) for group in groups for target in targets]
    lookup, draws = {}, {}
    for cell in cells:
        key = cell['group_id'], cell['target']
        check(type(key[0]) is int and key in expected and key not in lookup, 'Duplicate or unexpected normalized cell')
        lookup[key] = cell
    check(set(lookup) == set(expected), 'Incomplete observation matrix, including white-box diagnostics')
    for boot in bootstraps:
        key = boot['group_id'], boot['target']
        check(key in lookup and key not in draws and lookup[key]['source'] != key[1], 'Duplicate, unexpected or white-box bootstrap')
        check(boot['normalized_cell_sha256'] == value_sha(lookup[key])
            and boot['input_evidence_sha256'] == lookup[key]['input_evidence_sha256'], 'Matrix AP bytes are not bound to observations')
        for actual, name in (('resample_metadata_sha256', 'metadata_sha256'),
                ('resample_stream_sha256', 'payload_sha256'), ('resample_definition_sha256', 'definition_sha256')):
            check(boot[actual] == shared[name], 'Matrix has mixed shared streams')
        values = array(boot['ap_values'], count, 'matrix input AP')
        check(hashlib.sha256(values.astype('<f8', copy=False).tobytes()).hexdigest() == boot['ap_values_sha256'],
            'Matrix input AP bytes changed')
        draws[key] = values
    check(set(draws) == {key for key in expected if lookup[key]['source'] != key[1]}, 'Incomplete black-box bootstrap matrix')
    check(result.get('status') == 'assembled_pending_external_input_and_pipeline_acceptance'
        and all(result.get(key) is False for key in ('scientific_acceptance', 'independent_confirmation', 'efficacy_runner_armed')),
        'Matrix claims unsupported acceptance')
    same(result.get('shared_resamples'), shared, 'Reported matrix consumed another shared stream')
    same(result.get('normalized_cells'), [lookup[key] for key in expected], 'Reported cells differ or lost canonical order')
    same(result.get('sources'), sources, 'Reported source order differs')
    same(result.get('targets'), targets, 'Reported target order differs')
    pairs = {}
    for group in groups:
        check(type(group['group_id']) is int and type(group['seed']) is int and group['source'] in sources
            and group['targets'] == targets and group['source_matched_target'] == group['source']
            and group['blackbox_targets'] == [target for target in targets if target != group['source']], 'Wrong planned source exclusion')
        pair = group['variant'], group['seed']
        pairs.setdefault(pair, []).append(group)
        for target in targets:
            cell = lookup[group['group_id'], target]
            check(cell['status'] == 'complete' and cell['failures'] == []
                and all(type(cell[key]) is type(group[key]) and cell[key] == group[key] for key in
                    ('source', 'variant', 'seed', 'parameters_sha256', 'images')), 'Normalized cell differs from its planned group')
            number(cell['bbox_mAP'], 'observed AP', 0, 1)
    check(result.get('groups') == len(groups) and result.get('all_cells') == len(cells)
        and result.get('blackbox_bootstrap_cells') == len(draws) and result.get('configuration_seed_pairs') == len(pairs),
        'Reported matrix totals differ')
    check([(row['variant'], row['seed']) for row in result['configuration_seed']] == list(pairs),
        'Configuration-seed rows are missing, duplicated or reordered')
    observed, sampled, maximum_delta, ordered_bindings = {}, {}, 0.0, []
    boot_lookup = {(row['group_id'], row['target']): row for row in bootstraps}
    for pair, reported in zip(pairs, result['configuration_seed']):
        group_rows = pairs[pair]
        check([group['source'] for group in group_rows] == sources, 'Missing or duplicated source in configuration-seed pair')
        keys = [(group['group_id'], target) for group in group_rows for target in targets if target != group['source']]
        check(len(keys) == plan['blackbox_cells_per_configuration_seed'], 'Wrong equal-cell panel size')
        rows = [dict(source=lookup[key]['source'], target=key[1], value=lookup[key]['bbox_mAP']) for key in keys]
        same(reported['blackbox_cells'], rows, 'Reported black-box cell panel differs')
        observed[pair] = math.fsum(row['value'] for row in rows) / len(rows)
        maximum_delta = max(maximum_delta, near(reported['bbox_mAP'], observed[pair], 'equal-cell mean'))
        for field, order, column in (('source_means', sources, 'source'), ('target_means', targets, 'target')):
            check(set(reported[field]) == set(order), 'Missing source or target mean')
            for name in order:
                values = [row['value'] for row in rows if row[column] == name]
                maximum_delta = max(maximum_delta, near(reported[field][name], math.fsum(values) / len(values), field))
        sampled[pair] = np.mean(np.stack([draws[key] for key in keys]), axis=0, dtype=np.float64)
        published = array(reported['bootstrap_mean_ap'], count, 'reported paired mean AP')
        check(hashlib.sha256(published.astype('<f8', copy=False).tobytes()).hexdigest() == reported['bootstrap_mean_sha256'],
            'Reported paired-mean byte hash differs')
        delta = float(np.max(np.abs(published - sampled[pair])))
        check(delta <= TOLERANCE, 'Independent paired bootstrap mean differs')
        maximum_delta = max(maximum_delta, delta)
        for key in keys:
            ordered_bindings.append({name: boot_lookup[key][name] for name in ('group_id', 'target', 'normalized_cell_sha256',
                'input_evidence_sha256', 'ap_values_sha256', 'resample_metadata_sha256',
                'resample_stream_sha256', 'resample_definition_sha256', 'replicates_completed')})
    same(result['cell_bootstrap_bindings'], ordered_bindings, 'Matrix lost a cell-to-AP byte binding')
    analysis, definition = plan['analysis'], plan['analysis']['uncertainty']
    declarations = analysis['contrasts']
    check(analysis['metric'] == 'bbox_mAP' and analysis['family_size'] == len(declarations)
        and len({row['id'] for row in declarations}) == len(declarations)
        and definition['multiplicity'] == 'one_family_centered_max_absolute_bootstrap_error'
        and definition['seed_estimand'] == 'conditional_on_the_five_registered_seeds_not_resampled',
        'The registered contrast family or interpretation differs')
    estimates, replicated, used = [], [], set()
    for row in declarations:
        coefficients = {}
        for term in row['terms']:
            pair = term['variant'], term['seed']
            check(pair in pairs and pair not in coefficients, 'Missing or duplicated contrast endpoint')
            coefficients[pair] = Fraction(term['coefficient'])
        check(coefficients and sum(coefficients.values()) == 0 and any(coefficients.values()), 'Invalid signed contrast')
        used.update(coefficients)
        estimates.append(math.fsum(float(weight) * observed[pair] for pair, weight in coefficients.items()))
        replicated.append(sum((sampled[pair] * float(weight) for pair, weight in coefficients.items()), np.zeros(count)))
    check(used == set(pairs), 'A registered configuration-seed pair was excluded from inference')
    errors = np.max(np.abs(np.stack(replicated, axis=1) - np.asarray(estimates)[None, :]), axis=1)
    alpha = number(definition['alpha'], 'alpha', 0, 1)
    check(0 < alpha < 1, 'Invalid alpha')
    radius = sorted(float(value) for value in errors)[math.ceil((1 - alpha) * count) - 1]
    intervals = result['intervals']
    check(intervals.get('status') == 'computed_pending_input_and_pipeline_acceptance'
        and intervals.get('metric') == 'bbox_mAP' and intervals.get('family_size') == len(declarations)
        and intervals.get('bootstrap_replicates') == count and intervals.get('alpha') == alpha
        and all(intervals.get(key) is False for key in ('finite_sample_guarantee', 'independent_confirmation', 'scientific_acceptance')),
        'Reported interval family or scope differs')
    maximum_delta = max(maximum_delta, near(intervals['maximum_error_radius'], radius, 'family maximum-error radius'))
    check([row['id'] for row in intervals['contrasts']] == [row['id'] for row in declarations], 'Incomplete or reordered interval family')
    for index, reported in enumerate(intervals['contrasts']):
        expected_values = dict(estimate=estimates[index], simultaneous_lower=estimates[index] - radius,
            simultaneous_upper=estimates[index] + radius,
            pointwise_percentile_lower=linear_quantile(replicated[index], alpha / 2),
            pointwise_percentile_upper=linear_quantile(replicated[index], 1 - alpha / 2))
        for name, expected_value in expected_values.items():
            maximum_delta = max(maximum_delta, near(reported[name], expected_value, reported['id'] + ':' + name))
        check(type(reported['simultaneous_band_excludes_zero']) is bool
            and reported['simultaneous_band_excludes_zero'] == (reported['simultaneous_lower'] > 0 or reported['simultaneous_upper'] < 0),
            'Zero-exclusion label disagrees with the reported band')
    expected_stability = [variant for variant in dict.fromkeys(pair[0] for pair in pairs)
        if sum(pair[0] == variant for pair in pairs) > 1]
    check([row['variant'] for row in result['seed_stability']] == expected_stability, 'Seed-stability rows differ')
    for row in result['seed_stability']:
        seeds = sorted(seed for variant, seed in pairs if variant == row['variant'])
        check(len(seeds) == 5 and row['seeds'] == seeds and row['attack_seeds_resampled'] is False,
            'The five fixed attack seeds were changed or resampled')
        values = [observed[row['variant'], seed] for seed in seeds]
        deltas = [value - observed['full', seed] for seed, value in zip(seeds, values)]
        for name, numbers in (('bbox_mAP', values), ('paired_candidate_minus_full', deltas)):
            check(len(row[name]) == len(numbers), 'Seed vector length differs')
            for actual, expected_value in zip(row[name], numbers):
                maximum_delta = max(maximum_delta, near(actual, expected_value, name))
        for prefix, numbers in (('', values), ('paired_', deltas)):
            mean = math.fsum(numbers) / len(numbers)
            std = math.sqrt(math.fsum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1))
            for name, expected_value in (('mean', mean), ('sample_std', std), ('minimum', min(numbers)), ('maximum', max(numbers))):
                maximum_delta = max(maximum_delta, near(row[prefix + name], expected_value, prefix + name))
    return dict(groups=len(groups), observation_cells=len(cells), blackbox_bootstrap_cells=len(draws),
        whitebox_replay_cells=len(cells) - len(draws), configuration_seed_pairs=len(pairs),
        replicates=count, contrasts=len(declarations), stability_variants=len(expected_stability),
        maximum_independent_arithmetic_delta=maximum_delta, arithmetic_tolerance=TOLERANCE,
        independent_arithmetic_replayed=True, prediction_bootstraps_recomputed=0,
        independent_confirmation=False, scientific_acceptance=False)


def verify_grid(plan, index, qualified_cell):
    execution, grid = plan['execution'], plan['grid']
    groups, targets = execution['groups'], execution['targets']
    same(groups, execution['prepared_plan']['groups'], 'Execution changed the prepared group matrix')
    same(execution['analysis'], execution['prepared_plan']['analysis'], 'Execution changed the prepared analysis')
    same(index['groups'], execution['input_slots'], 'Original input-slot partition differs')
    check(index.get('status') == 'partial_bound_analysis_inputs' and index.get('errors') == []
        and index.get('plan_sha256') == execution['prepared_plan_sha256']
        and grid.get('execution_plan_sha256') == plan['inputs']['evidence']['execution_plan']['sha256']
        and grid.get('analysis_sha256') == execution['analysis_sha256'] == value_sha(execution['analysis']),
        'CPU grid has another execution, input or analysis identity')
    check([group['group_id'] for group in groups] == list(range(1, len(groups) + 1))
        and len({(g['variant'], g['source'], g['seed']) for g in groups}) == len(groups), 'Invalid planned group identities')
    new_ids = [slot['group_id'] for slot in execution['input_slots'] if slot['disposition'] == 'new_registered_measurement']
    check(execution['new_group_ids'] == new_ids
        and execution['reused_group_ids'] == [g['group_id'] for g in groups if g['group_id'] not in new_ids],
        'The qualified reuse partition changed')
    expected_ids, original, observed_keys, outputs = [], {}, [], []
    tasks = {task['task_id']: task for task in grid['tasks']}
    check(len(tasks) == len(grid['tasks']), 'Duplicate CPU task identity')
    check([(cell['group_id'], cell['target']) for cell in index['cells']]
        == [(group['group_id'], target) for group in groups for target in targets], 'Original input catalogue lost canonical cells')
    index_lookup = {(cell['group_id'], cell['target']): cell for cell in index['cells']}
    reused = []
    for group in groups:
        gid, new = group['group_id'], group['group_id'] in new_ids
        if new:
            identifier = 'bind_g{:06d}'.format(gid)
            expected_ids.append(identifier)
            task = tasks[identifier]
            root = plain(plan['inputs']['execution_worktree'])
            output = root / 'outputs/diagnostics/vcsf_cpu_analysis_inputs' / grid['output_id'] / identifier
            check(task['kind'] == 'bind_new_group' and task['target'] is None
                and task['bootstrap_replicates'] is None and task['output'] == str(output)
                and task['source_closure_required'] is True
                and all(type(task[key]) is type(group[key]) and task[key] == group[key]
                    for key in ('group_id', 'source', 'variant', 'seed')), 'New-group binding task differs')
            outputs.append(str(output))
        for target_index, target in enumerate(targets):
            identifier = 'cell_g{:06d}_t{:02d}'.format(gid, target_index)
            expected_ids.append(identifier)
            task, entry = tasks[identifier], index_lookup[gid, target]
            check(task['kind'] == ('new_cell' if new else 'original_cell') and task['target'] == target
                and task['target_index'] == target_index and type(task['target_index']) is int
                and all(type(task[key]) is type(group[key]) and task[key] == group[key] for key in
                    ('group_id', 'source', 'variant', 'seed', 'images', 'parameters_sha256')),
                'CPU cell differs from its exact prospective observation')
            expected_count = 0 if target == group['source'] else execution['analysis']['uncertainty']['replicates']
            check(type(task['bootstrap_replicates']) is int and task['bootstrap_replicates'] == expected_count
                and task['input_binding_required'] is new, 'CPU task changed the bootstrap or input-binding scope')
            observed_keys.append((gid, target))
            if new:
                check(entry['status'] == 'NR' and entry['normalized_cell'] is None and entry['input_sha256'] is None
                    and task['initial_normalized_cell_sha256'] is None, 'New task invented pre-existing evidence')
            else:
                cell = entry['normalized_cell']
                check(entry['status'] == 'metadata_bound_pending_archive_replay'
                    and entry['normalized_cell_sha256'] == task['initial_normalized_cell_sha256'] == value_sha(cell)
                    and entry['input_sha256'] == cell['input_evidence_sha256'], 'Original normalized input changed')
                original[gid, target] = entry
            reuse = task['qualified_complete_reuse']
            if reuse is not None:
                reference = plan['policy']['bound_evidence']['indexed_cell']
                check(not new and target != group['source'] and task['output'] is None
                    and reuse['receipt_file'] == reference['file'] and reuse['receipt_sha256'] == reference['sha256']
                    and (gid, target) == (qualified_cell['group_id'], qualified_cell['target'])
                    and reuse['normalized_cell_sha256'] == value_sha(qualified_cell['normalized_cell'])
                    and reuse['bootstrap_receipt_sha256'] == qualified_cell['bootstrap_receipt_sha256'],
                    'Completed-cell reuse is not the independently locked original cell')
                reused.append(identifier)
            else:
                root = plain(plan['inputs']['execution_worktree' if new else 'original_worktree'])
                output = root / 'outputs/diagnostics/vcsf_cpu_analysis_cells' / grid['output_id'] / identifier
                check(task['output'] == str(output), 'A CPU task output moved outside its prospective root')
                outputs.append(str(output))
    check([task['task_id'] for task in grid['tasks']] == expected_ids and len(outputs) == len(set(outputs))
        and len(reused) == 1, 'CPU grid lost, duplicated or reordered work')
    counts = dict(registered_groups=len(groups), observation_cells=len(observed_keys),
        original_metadata_cells=len(original), new_metadata_cells=len(observed_keys) - len(original),
        new_group_input_tasks=len(new_ids), whitebox_replay_cells=len(groups),
        blackbox_bootstrap_cells=len(observed_keys) - len(groups), qualified_complete_reuse_cells=len(reused),
        newly_executed_cell_tasks=len(observed_keys) - len(reused),
        registered_replicates=execution['analysis']['uncertainty']['replicates'])
    check(all(type(grid.get(name)) is int and grid[name] == value for name, value in counts.items()), 'CPU grid totals differ')
    return original, reused


def verify_completed_identity(row, tasks, reuse_ids):
    identifier = row.get('task_id')
    check(identifier in tasks and type(row.get('qualified_reuse')) is bool
        and row['qualified_reuse'] == (identifier in reuse_ids), 'Task claimed unauthorized completed-cell reuse')
    task = tasks[identifier]
    check(all(type(row.get(key)) is type(task[key]) and row[key] == task[key]
        for key in ('task_id', 'kind', 'group_id', 'target')), 'Completed task changed its prospective identity')
    if not row['qualified_reuse']:
        check(type(row.get('invocation')) is dict and type(row.get('launch_root')) is str,
            'Newly executed task lacks invocation evidence')


def sealed_prediction_input(read, document, input_path, input_hash, directory, entries):
    """Bind exact prediction paths and JSON values to an already sealed inventory."""
    metadata_path, artifact_path = directory / 'metrics.json', directory / 'predictions_artifact.json'
    archive_path = directory / 'predictions.json.gz'
    check(document.get('metadata_file') == str(metadata_path)
        and document.get('prediction_artifact_file') == str(artifact_path), 'Input prediction path differs from its sealed target')
    metadata_entry, artifact_entry, archive_entry = (entries[name] for name in ('metrics.json', 'predictions_artifact.json', 'predictions.json.gz'))
    check(document.get('metadata_sha256') == metadata_entry['sha256']
        and document.get('prediction_artifact_sha256') == artifact_entry['sha256'], 'Input prediction hash differs from its sealed target')
    for path, item in ((metadata_path, metadata_entry), (artifact_path, artifact_entry), (archive_path, archive_entry)):
        check(type(item.get('bytes')) is int and item['bytes'] == plain(path).stat().st_size, 'Sealed prediction byte count differs')
    metadata = read.json(metadata_path, metadata_entry['sha256'])
    artifact = read.json(artifact_path, artifact_entry['sha256'])
    check(artifact.get('status') == 'verified_lossless_archive' and artifact.get('format') == 'gzip'
        and artifact.get('archive_file') == 'predictions.json.gz' and artifact.get('archive_sha256') == archive_entry['sha256']
        and artifact.get('archive_bytes') == archive_entry['bytes']
        and artifact.get('uncompressed_sha256') == metadata.get('predictions_sha256')
        and artifact.get('prediction_records') == metadata.get('detections'), 'Sealed prediction archive disagrees with metrics')
    same(metadata.get('predictions_artifact'), artifact, 'Metrics embed another prediction archive')
    read.bind_map({str(archive_path): archive_entry['sha256']})
    return dict(input_file=str(input_path), input_sha256=input_hash,
        metadata_file=str(metadata_path), metadata_sha256=metadata_entry['sha256'], metadata=metadata,
        artifact_file=str(artifact_path), artifact_sha256=artifact_entry['sha256'], artifact=artifact,
        archive_file=str(archive_path), archive_sha256=archive_entry['sha256'])


def verify_original_input(read, entry, task, plan, source_cache):
    input_path = child(plain(plan['execution']['input_index_file']).parent, entry['input_file'])
    document = read.json(input_path, entry['input_sha256'])
    slot = plan['execution']['input_slots'][task['group_id'] - 1]
    same(document['slot'], slot, 'Original input document changed its qualified source mapping')
    provenance = document['original_provenance']
    root = plain(provenance['root'])
    cache_key = str(root)
    if cache_key not in source_cache:
        inventory = read.json(provenance['artifact_manifest'], provenance['artifact_manifest_sha256'])
        rows = read.json(provenance['records'], provenance['records_sha256'])
        entries = {row['path']: row for row in inventory['files']}
        check(len(entries) == len(inventory['files']), 'Original inventory duplicates a file path')
        source_cache[cache_key] = dict(provenance=provenance, entries=entries, records=rows)
    source = source_cache[cache_key]
    same(provenance, source['provenance'], 'Original cells disagree about accepted root provenance')
    selected = [row for row in source['records'] if row.get('adversarial_run') == slot['original_group']
        and row.get('source') == slot['source'] and row.get('target') == task['target'] and row.get('attack') == 'vcsf']
    check(selected == [document['original_record']], 'Original input document changed the accepted target record')
    record, cell = selected[0], entry['normalized_cell']
    check(record['parameters_sha256'] == slot['original_parameters_sha256']
        and cell['original_record_sha256'] == value_sha(record) and cell['original_code_commit'] == record['code_commit']
        and cell['original_parameters_sha256'] == record['parameters_sha256']
        and cell['metrics'] == record['metrics'] and cell['bbox_mAP'] == float(record['metrics']['bbox_mAP'])
        and cell['reuse_disposition'] == slot['disposition'], 'Original normalization changed accepted values or provenance')
    attack = plain(slot['original_group'])
    directory = root / 'evaluations' / attack.relative_to(root / 'attacks') / task['target']
    entries = {name: source['entries'][(directory / name).relative_to(root).as_posix()]
        for name in ('metrics.json', 'predictions_artifact.json', 'predictions.json.gz')}
    verified = sealed_prediction_input(read, document, input_path, entry['input_sha256'], directory, entries)
    for key in ('dataset', 'split', 'source', 'target', 'attack', 'parameters_sha256', 'checkpoint_sha256', 'code_commit', 'images', 'metrics'):
        same(verified['metadata'][key], record[key], 'Original prediction metadata changed: ' + key)
    check(document['archive']['path'] == verified['archive_file'] and document['archive']['sha256'] == verified['archive_sha256']
        and document['archive']['bytes'] == verified['artifact']['archive_bytes'], 'Original document points to another prediction archive')
    return verified


def verify_closure(read, cpu_root, plan, group_id):
    witness_path = cpu_root / 'closures' / 'group_{:06d}.json'.format(group_id)
    witness = read.json(witness_path, sha(witness_path))
    stable = witness['stable']
    check(witness.get('snapshot_is_mutable') is True and witness.get('source_root_modified') is False
        and witness.get('group_scientific_acceptance') is False and stable['group_id'] == group_id
        and stable['row_sha256'] == value_sha(stable['row']), 'Producer closure witness changed')
    gpu_root = plain(plan['inputs']['run'])
    worktree = plain(plan['inputs']['execution_worktree'])
    expected_receipt = gpu_root / 'groups' / '{:06d}'.format(group_id) / 'group_acceptance.json'
    check(stable['source_receipt_file'] == str(expected_receipt), 'Closure points to another source group')
    receipt = read.json(expected_receipt, stable['source_receipt_sha256'])
    group = plan['execution']['groups'][group_id - 1]
    check(receipt.get('status') == 'complete_group_pending_independent_acceptance' and receipt.get('errors') == []
        and receipt.get('failed_records') == 0 and receipt.get('payload_state') == 'retained_500'
        and receipt.get('evaluations') == len(plan['execution']['targets'])
        and receipt.get('execution_plan_sha256') == plan['grid']['execution_plan_sha256']
        and receipt.get('runtime_snapshot_sha256') == plan['execution']['runtime_snapshot_sha256']
        and all(receipt.get(key) == group[key] for key in ('group_id', 'source', 'variant', 'seed', 'images', 'targets', 'parameters_sha256')),
        'Producer group has not completed its full registered panel and retention gate')
    row = stable['row']
    check(all(row.get(key) == value for key, value in receipt.items())
        and row.get('receipt_sha256') == stable['source_receipt_sha256'], 'Closure row relabeled the producer receipt')
    assignment = read.json(stable['assignment_file'], stable['assignment_sha256'])
    stage = assignment['stage']
    check(stage in plan['binding']['stages'] and group_id in stage['group_ids']
        and plain(stable['assignment_file']) == gpu_root / 'stages' / stage['name'] / 'assignment.json',
        'Closure has another declared stage')
    entries = [entry for entry in assignment['workers'] if group_id in entry['group_ids']]
    check(len(entries) == 1, 'Source group has missing or duplicate worker assignment')
    entry = entries[0]
    directory = child(worktree, entry['directory'])
    check(directory == gpu_root / 'workers' / stage['name'] / str(entry['slot'])
        and stable['request_file'] == str(directory / 'request.json')
        and stable['request_sha256'] == entry['request_sha256'], 'Source worker directory or request differs')
    request = read.json(stable['request_file'], stable['request_sha256'])
    refs = plan['inputs']['evidence']
    descriptor = dict(candidate_id='vcsf_layered_research', execution_alias='vcsf_research_isolated', mode='layered_efficacy',
        execution_plan=plain(refs['execution_plan']['file']).relative_to(worktree).as_posix(),
        execution_plan_sha256=refs['execution_plan']['sha256'],
        admission=plain(refs['formal_admission']['file']).relative_to(worktree).as_posix(),
        admission_sha256=refs['formal_admission']['sha256'], execution_root=gpu_root.relative_to(worktree).as_posix(),
        run_binding_sha256=plan['inputs']['binding_sha256'], max_images=None)
    same(request['descriptor'], descriptor, 'Source worker has another execution descriptor')
    check(request.get('group_ids') == entry['group_ids'] and request.get('stage') == stage['name']
        and request.get('devices') == plan['binding']['devices']
        and request.get('physical_device') == entry['physical_device'] == plan['binding']['devices'][entry['slot']]
        and request.get('gpu_uuid') == entry['gpu_uuid'] and request.get('worker_slot') == entry['slot'],
        'Source worker assignment differs')
    owner = dict(worker_pid=entry['pid'], worker_pid_start_ticks=entry['pid_start_ticks'], worker_slot=entry['slot'],
        request_sha256=entry['request_sha256'], gpu_uuid=entry['gpu_uuid'], physical_device=entry['physical_device'],
        stage=stage['name'], **plan['source_owner'])
    check(all(row.get(key) == value for key, value in owner.items())
        and all(request.get(key) == value for key, value in plan['source_owner'].items()), 'Source group ownership differs')
    completed_file = plain(stable['completed_groups_file'])
    check(completed_file == directory / 'completed_groups.json', 'Source completion-prefix path differs')
    current = json.loads(completed_file.read_bytes(), object_pairs_hook=no_duplicates, parse_constant=no_constant)
    check([item['group_id'] for item in current] == entry['group_ids'][:len(current)]
        and [item for item in current if item['group_id'] == group_id] == [row], 'Previously sealed source row changed')
    return stable, receipt


def verify_new_binding(read, row, task, plan, stable, receipt):
    path = plain(row['receipt_file'])
    check(path == plain(task['output']) / 'new_group_inputs.json', 'New input index moved to another task')
    index = read.json(path, row['receipt_sha256'])
    group = plan['execution']['groups'][task['group_id'] - 1]
    check(index.get('status') == 'new_group_metadata_bound_pending_analysis_acceptance'
        and index.get('errors') == [] and index.get('group_id') == group['group_id']
        and index.get('execution_plan_sha256') == plan['grid']['execution_plan_sha256']
        and index.get('metadata_bound_cells') == len(plan['execution']['targets'])
        and index.get('complete_bootstrap_cells') == 0 and index.get('new_model_calls') == 0
        and all(index.get(key) is False for key in ('original_twelve_metrics_replayed', 'source_root_modified',
            'independent_confirmation', 'analysis_pipeline_accepted', 'scientific_acceptance')),
        'New group index lost its nonaccepting complete metadata scope')
    check([entry['target'] for entry in index['cells']] == plan['execution']['targets'], 'New group target panel differs')
    evidence = read.json(path.parent / 'group_evidence.json', index['group_evidence_sha256'])
    check(evidence.get('status') == 'independently_verified_new_group_artifacts_pending_metric_replay'
        and evidence.get('errors') == [] and evidence.get('source_group_receipt_sha256') == stable['source_receipt_sha256']
        and evidence.get('root_binding_sha256') == plan['inputs']['binding_sha256']
        and evidence.get('execution_plan_sha256') == plan['grid']['execution_plan_sha256']
        and evidence.get('runtime_snapshot_sha256') == plan['execution']['runtime_snapshot_sha256']
        and all(evidence.get(key) == group[key] for key in ('group_id', 'source', 'variant', 'seed', 'images', 'targets')),
        'Independent group-artifact evidence has another source binding')
    retained = plan['execution']['definition']['payload_retention']['retained_images']
    check(evidence.get('retained_png_pixels_verified') == retained
        and evidence.get('pruned_pngs_verified_absent') == group['images'] - retained
        and evidence.get('producer_pre_prune_decoded_pngs') == group['images']
        and type(evidence.get('retained_png_max_linf')) is int and 0 <= evidence['retained_png_max_linf'] <= 4
        and evidence.get('pruned_png_pixels_independently_replayed') is False,
        'New group retention evidence changed its pixel verification scope')
    check(all(evidence.get(key) is False for key in ('original_twelve_metrics_replayed', 'prediction_uncompressed_bytes_replayed',
        'group_scientific_acceptance', 'whole_run_acceptance', 'analysis_pipeline_accepted', 'scientific_acceptance', 'source_root_modified'))
        and evidence.get('new_model_calls') == 0, 'Artifact binder claims unsupported acceptance')
    check(evidence['checked_input_sha256'].get(stable['source_receipt_file']) == stable['source_receipt_sha256'],
        'Independent artifact binder checked another producer receipt')
    for name, digest in evidence['implementation_sha256'].items():
        check(plan['implementation_sha256'].get(name) == digest, 'Input builder source differs')
    read.bind_map(evidence['checked_input_sha256'])
    manifest_path = plain(stable['source_receipt_file']).parent / 'artifact_manifest.json'
    inventory = read.json(manifest_path, receipt['artifact_manifest_sha256'])
    records_path = manifest_path.parent / 'records.json'
    records = read.json(records_path, receipt['records_sha256'])
    check([item['target'] for item in records] == plan['execution']['targets']
        and inventory['artifacts']['records.json']['sha256'] == receipt['records_sha256'], 'Sealed source records differ')
    documents = {}
    for entry, record in zip(index['cells'], records):
        check(entry.get('group_id') == group['group_id'] and entry.get('status') == 'metadata_bound_pending_archive_replay'
            and entry.get('bootstrap_status') == 'NR' and entry['input_file'] == 'cells/' + entry['target'] + '/input.json',
            'New group input slot changed identity or execution state')
        input_path = child(path.parent, entry['input_file'])
        document = read.json(input_path, entry['input_sha256'])
        same(document['original_record'], record, 'New input document relabeled its sealed original record')
        same(document['group'], group, 'New input document changed scientific group identity')
        check(document['group_evidence_sha256'] == index['group_evidence_sha256']
            and document['source_group_receipt_sha256'] == stable['source_receipt_sha256']
            and document['root_binding_sha256'] == plan['inputs']['binding_sha256']
            and document['execution_plan_sha256'] == plan['grid']['execution_plan_sha256'], 'New input document lost source provenance')
        group_root, run_root = manifest_path.parent, plain(plan['inputs']['run'])
        expected_paths = dict(group_root=group_root, run_root=run_root, attack_root=group_root / 'attack',
            execution_plan_file=plain(plan['inputs']['evidence']['execution_plan']['file']),
            group_evidence_file=path.parent / 'group_evidence.json', source_group_receipt_file=plain(stable['source_receipt_file']),
            root_binding_file=run_root / 'plan_binding.json', annotation_file=group_root / 'attack/annotations.json',
            canonical_annotation_file=plain(plan['inputs']['canonical_annotation']).resolve(strict=True))
        check(all(document.get(key) == str(value) for key, value in expected_paths.items())
            and document.get('canonical_annotation_sha256') == plan['execution']['annotation_sha256']
            and document.get('annotation_sha256') == inventory['artifacts']['attack/annotations.json']['sha256']
            and document.get('image_ids_sha256') == plan['execution']['image_ids_sha256'], 'New input physical source mapping differs')
        prefix = 'evaluations/' + entry['target'] + '/'
        prediction_entries = {name: inventory['artifacts'][prefix + name]
            for name in ('metrics.json', 'predictions_artifact.json', 'predictions.json.gz')}
        verified = sealed_prediction_input(read, document, input_path, entry['input_sha256'], group_root / prefix, prediction_entries)
        same(document['metadata'], verified['metadata'], 'New document embeds another target metadata JSON')
        same(document['archive'], verified['artifact'], 'New document embeds another prediction archive JSON')
        metadata = verified['metadata']
        check(metadata.get('status') == 'complete' and metadata.get('failures') == []
            and metadata.get('source') == group['source'] and metadata.get('target') == entry['target']
            and metadata.get('dataset') == 'coco' and metadata.get('split') == 'val'
            and metadata.get('images') == group['images'] and metadata.get('attack') == 'vcsf_research_isolated'
            and metadata.get('adversarial_run') == str(group_root / 'attack')
            and metadata.get('parameters_sha256') == group['parameters_sha256']
            and metadata.get('checkpoint_sha256') == plan['execution']['checkpoint_sha256'][entry['target']]
            and metadata.get('code_commit') == plan['execution']['base_commit']
            and metadata.get('evaluated_image_ids_sha256') == metadata.get('expected_image_ids_sha256')
                == plan['execution']['image_ids_sha256'] and metadata.get('evaluated_image_ids_match_expected') is True,
            'Sealed new prediction metadata has another scientific identity')
        expected_record = dict(metadata, variant=group['variant'], seed=group['seed'], group_id=group['group_id'],
            execution_protocol=plan['execution']['protocol'], execution_plan_sha256=plan['grid']['execution_plan_sha256'],
            runtime_snapshot_sha256=plan['execution']['runtime_snapshot_sha256'], independent_confirmation=False, formal_metrics_eligible=False)
        same(record, expected_record, 'New group record is not the exact enriched sealed target metadata')
        documents[entry['target']] = verified
        cell = entry['normalized_cell']
        check(entry['normalized_cell_sha256'] == value_sha(cell) and cell['input_evidence_sha256'] == entry['input_sha256']
            and cell['original_record_sha256'] == value_sha(record) and cell['original_code_commit'] == record['code_commit']
            and cell['bbox_mAP'] == float(record['metrics']['bbox_mAP']) and cell['metrics'] == record['metrics']
            and cell['observation_origin'] == 'new_registered_measurement'
            and cell['group_evidence_sha256'] == index['group_evidence_sha256'], 'New normalized observation changed its original values')
    return index, documents


def arguments_for(task, plan, completed, stable=None):
    inputs, refs = plan['inputs'], plan['inputs']['evidence']
    values = dict(output=task['output'], group_id=task['group_id'])
    if task['kind'] == 'original_cell':
        values.update(root=inputs['original_worktree'], index=plan['execution']['input_index_file'],
            index_sha256=plan['execution']['input_index_sha256'], tests_receipt=refs['original_cell_tests']['file'],
            tests_sha256=refs['original_cell_tests']['sha256'], paired_receipt=plan['policy']['bound_evidence']['paired_kernel']['file'],
            paired_sha256=plan['policy']['bound_evidence']['paired_kernel']['sha256'])
    else:
        values.update(execution_worktree=inputs['execution_worktree'], run=inputs['run'],
            execution_plan=refs['execution_plan']['file'], plan_sha256=refs['execution_plan']['sha256'],
            admission=refs['formal_admission']['file'], binding_sha256=inputs['binding_sha256'])
        if task['kind'] == 'bind_new_group':
            values.update(group_receipt_sha256=stable['source_receipt_sha256'], canonical_annotation=inputs['canonical_annotation'],
                clean_images=inputs['clean_images'], tests_receipt=refs['builder_tests']['file'], tests_sha256=refs['builder_tests']['sha256'])
        else:
            bound = completed['bind_g{:06d}'.format(task['group_id'])]
            values.update(index=bound['receipt_file'], index_sha256=bound['receipt_sha256'],
                policy_lock=refs['policy_lock']['file'], policy_sha256=refs['policy_lock']['sha256'],
                tests_receipt=refs['new_cell_tests']['file'], tests_sha256=refs['new_cell_tests']['sha256'])
    if task['kind'] != 'bind_new_group':
        values.update(target=task['target'], kernel='vectorized_owned_cache',
            kernel_sha256=plan['implementation_sha256']['src/lgp/reporting/vcsf_vectorized_bootstrap.py'],
            max_uncompressed_bytes=plan['resource_contract']['max_uncompressed_bytes'], progress_every=25)
    return {'--' + key.replace('_', '-'): str(value) for key, value in values.items()}


def verify_reports(read, directory, result, expected_manifest_hash):
    manifest = read.json(directory / 'report_manifest.json', expected_manifest_hash)
    check(manifest.get('status') == 'reports_generated_pending_acceptance' and manifest.get('scientific_acceptance') is False
        and set(manifest['files']) == {'analysis.json', 'cells.csv', 'configuration_seed.csv', 'contrasts.csv', 'contrasts.tex'},
        'Analysis report inventory is incomplete')
    tables = {
        'cells.csv': (['group_id', 'variant', 'seed', 'source', 'target', 'bbox_mAP'], result['normalized_cells']),
        'configuration_seed.csv': (['variant', 'seed', 'bbox_mAP'], result['configuration_seed']),
        'contrasts.csv': (['id', 'estimate', 'simultaneous_lower', 'simultaneous_upper',
            'pointwise_percentile_lower', 'pointwise_percentile_upper'], result['intervals']['contrasts']),
    }
    for name, digest in manifest['files'].items():
        raw = read.bytes(directory / name, digest, 128 << 20)
        if name in tables:
            fields, records = tables[name]
            parsed = csv.DictReader(raw.decode('utf-8').splitlines())
            check(parsed.fieldnames == fields, 'CSV field order differs: ' + name)
            expected = [{field: format(row[field], '.4f') if type(row[field]) is float else str(row[field])
                for field in fields} for row in records]
            check(list(parsed) == expected, 'CSV does not reproduce from unrounded JSON: ' + name)
        elif name == 'analysis.json':
            same(json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=no_constant), result, 'Report JSON differs')
        else:
            text = raw.decode('utf-8').splitlines()
            check(text[1:4] == [r'\begin{longtable}{lrrr}',
                r'Contrast & Estimate & Simultaneous lower & Simultaneous upper \\', r'\hline']
                and text[-1] == r'\end{longtable}' and len(text) == len(result['intervals']['contrasts']) + 5,
                'TeX report is incomplete')
            for line, row in zip(text[4:-1], result['intervals']['contrasts']):
                check(line.endswith(' & %.4f & %.4f & %.4f \\\\' %
                    (row['estimate'], row['simultaneous_lower'], row['simultaneous_upper']))
                    and line.split(' & ', 1)[0] == row['id'].replace('_', r'\_').replace('&', r'\&'),
                    'TeX row differs from its declared contrast and raw JSON')
    return dict(files=len(manifest['files']), generated_values_verified=True)


def verify_tests(read, path, expected_hash):
    receipt = read.json(path, expected_hash)
    check(receipt.get('status') == 'passed' and receipt.get('scope') == 'independent_cpu_analysis_audit'
        and receipt.get('exit_code') == receipt.get('help_exit_code') == 0
        and receipt.get('new_model_calls') == 0 and receipt.get('analysis_pipeline_accepted') is False,
        'Independent CPU auditor lacks matching server ODA test evidence')
    snapshot = plain(receipt['source_snapshot'])
    check(receipt['source_sha256'].get('tools/audit_vcsf_cpu_analysis.py') == sha(__file__),
        'The running CPU auditor is not the tested version')
    read.bind_map({str(child(snapshot, name)): digest for name, digest in receipt['source_sha256'].items()})
    junit = child(plain(path).parent, receipt['junit_file'])
    raw = read.bytes(junit, receipt['junit_sha256'], 16 << 20)
    cases = ET.fromstring(raw).findall('.//testcase')
    check(cases and all(not case.findall('failure') and not case.findall('error') for case in cases), 'Auditor tests contain failures')
    executed = [case for case in cases if not case.findall('skipped')]
    required = {'test_independent_full_registered_matrix', 'test_independent_matrix_rejects_corruption',
        'test_cell_audit_rejects_changed_bindings', 'test_invocation_rejects_wrong_owner_or_arguments',
        'test_sealed_prediction_input_rejects_wrong_source', 'test_completed_task_reuse_rejects_forged_authority',
        'test_cell_audit_rejects_matching_hash_under_wrong_input_path', 'test_cell_audit_rejects_relocated_whitebox_receipt',
        'test_invocation_rejects_incomplete_worker_start', 'test_independent_joint_radius_has_an_analytic_noninterpolated_answer',
        'test_audit_output_guard_preserves_inputs'}
    check(type(receipt.get('passed')) is int and receipt['passed'] == len(executed)
        and required <= {case.attrib['name'].split('[', 1)[0] for case in executed}, 'Required independent-audit tests did not execute')
    return dict(passed=len(executed), required_cases=sorted(required))


def guard_output(output, cpu_root, execution_root):
    output, cpu_root, execution_root = plain(output), plain(cpu_root), plain(execution_root)
    parent = execution_root / 'outputs/audits/vcsf_cpu_analysis'
    check(output.parent == parent and not output.exists() and not output.is_symlink()
        and output != cpu_root and output not in cpu_root.parents and cpu_root not in output.parents,
        'Use a fresh, disjoint CPU-audit root; no overwrite or resume is allowed')
    return output


def audit(args):
    check(sys.platform == 'linux' and Path(sys.prefix).name == 'oda', 'Numerical audit requires server ODA')
    cpu_root, read = plain(args.cpu_root), Evidence()
    plan = read.json(cpu_root / 'cpu_plan.json', args.cpu_plan_sha256)
    execution_root = plain(plan['inputs']['execution_worktree'])
    check(cpu_root == execution_root / 'outputs/diagnostics/vcsf_cpu_analysis' / plan['grid']['output_id'], 'Wrong CPU root')
    tests = verify_tests(read, args.tests_receipt, args.tests_sha256)
    check(plan.get('status') == 'prepared_complete_cpu_grid_pending_execution'
        and all(plan.get(key) is False for key in ('automatic_resume', 'automatic_failure_retry',
            'scientific_acceptance', 'analysis_pipeline_accepted', 'source_root_modified')) and plan.get('new_model_calls') == 0,
        'CPU prospective plan changed scope')
    refs = plan['inputs']['evidence']
    same(read.json(refs['execution_plan']['file'], refs['execution_plan']['sha256']), plan['execution'], 'CPU embedded execution plan differs')
    same(read.json(refs['policy_lock']['file'], refs['policy_lock']['sha256']), plan['policy'], 'CPU embedded analysis policy differs')
    same(read.json(plain(plan['inputs']['run']) / 'plan_binding.json', plan['inputs']['binding_sha256']), plan['binding'], 'CPU source-root binding differs')
    check(plan['binding']['max_images'] is None and plan['binding']['group_ids'] == plan['execution']['new_group_ids'],
        'CPU input producer is a limited-image or changed group run')
    check(plan['policy'].get('status') == 'independently_verified_analysis_policy_lock' and plan['policy'].get('errors') == []
        and plan['policy']['execution_plan_sha256'] == refs['execution_plan']['sha256']
        and plan['policy']['analysis_sha256'] == plan['execution']['analysis_sha256']
        and plan['policy']['shared_resampling']['independent_rng_replay'] is True, 'The predeclared analysis lock is not applicable')
    reference = plan['policy']['bound_evidence']['indexed_cell']
    qualified = read.json(reference['file'], reference['sha256'])
    index = read.json(plan['execution']['input_index_file'], plan['execution']['input_index_sha256'])
    original, reuse_ids = verify_grid(plan, index, qualified)
    read.bind_map(plan['checked_input_sha256'])
    read.bind_map({str(child(plain(plan['source_snapshot']), name)): digest for name, digest in plan['implementation_sha256'].items()})
    coordinator_path = cpu_root / 'coordinator.json'
    coordinator = read.json(coordinator_path, sha(coordinator_path))
    check(coordinator.get('cpu_plan_sha256') == args.cpu_plan_sha256, 'CPU coordinator has another prospective plan')
    tasks = {task['task_id']: task for task in plan['grid']['tasks']}
    completed = {}
    completion = None
    if args.observed_prefix:
        check(not (cpu_root / 'cpu_analysis.json').exists(), 'Use the full audit for a terminal CPU root')
        process = Path('/proc') / str(coordinator['pid']) / 'stat'
        check(process.is_file() and int(process.read_text().rsplit(') ', 1)[1].split()[19]) == coordinator['start_ticks'],
            'CPU coordinator liveness is not verified for this prefix observation')
        for path in sorted((cpu_root / 'completed').glob('*.json')):
            row = read.json(path, sha(path))
            verify_completed_identity(row, tasks, reuse_ids)
            check(path.name == row['task_id'] + '.json' and row['task_id'] not in completed, 'Duplicated completed-task evidence')
            completed[row['task_id']] = row
    else:
        check(is_sha(args.completion_sha256) and is_sha(args.launch_completion_sha256), 'Full audit requires terminal receipt hashes')
        completion = read.json(cpu_root / 'cpu_analysis.json', args.completion_sha256)
        launch_path = execution_root / 'outputs/diagnostics/vcsf_cpu_analysis_launches' / cpu_root.name / 'completion.json'
        launch = read.json(launch_path, args.launch_completion_sha256)
        check(launch.get('exit_code') == 0 and launch.get('cpu_analysis_receipt_sha256') == args.completion_sha256
            and launch.get('coordinator_pid') == coordinator['pid'], 'CPU supervisor has not confirmed successful terminal exit')
        check(completion.get('status') == 'complete_cpu_matrix_pending_independent_pipeline_acceptance'
            and completion.get('errors') == [] and completion.get('failed_records') == 0
            and completion.get('cpu_plan_sha256') == args.cpu_plan_sha256
            and all(completion.get(key) is False for key in ('scientific_acceptance', 'analysis_pipeline_accepted',
                'independent_confirmation', 'source_root_modified', 'automatic_resume', 'automatic_failure_retry'))
            and completion.get('new_model_calls') == 0,
            'The CPU matrix is not complete and failure-free')
        for row in completion['completed']:
            verify_completed_identity(row, tasks, reuse_ids)
            check(row['task_id'] not in completed, 'Terminal CPU task was duplicated')
            completed[row['task_id']] = row
            if not row['qualified_reuse']:
                path = cpu_root / 'completed' / (row['task_id'] + '.json')
                same(read.json(path, sha(path)), row, 'Terminal completion differs from its write-once task receipt')
    check(set(completed) <= set(tasks), 'An unplanned CPU task entered the audit')
    for identifier in reuse_ids:
        task, item = tasks[identifier], tasks[identifier]['qualified_complete_reuse']
        reused = dict(task_id=identifier, kind=task['kind'], group_id=task['group_id'], target=task['target'],
            receipt_file=item['receipt_file'], receipt_sha256=item['receipt_sha256'], qualified_reuse=True)
        check(identifier not in completed or completed[identifier] == reused, 'Qualified reuse was relaunched or relabeled')
        completed[identifier] = reused
    if completion is not None:
        check(set(completed) == set(tasks) and completion.get('completed_tasks') == len(tasks)
            and completion.get('launched_tasks') == len(tasks) - len(reuse_ids)
            and completion.get('closure_groups') == plan['execution']['new_group_ids'], 'Incomplete terminal CPU grid')
    cells, bootstraps, invocations, indices, closures, documents, original_cache = [], [], [], {}, {}, {}, {}
    for task in plan['grid']['tasks']:
        identifier = task['task_id']
        if identifier not in completed:
            continue
        row = completed[identifier]
        verify_completed_identity(row, tasks, reuse_ids)
        stable = receipt = None
        if task['kind'] == 'bind_new_group':
            stable, receipt = verify_closure(read, cpu_root, plan, task['group_id'])
            closures[task['group_id']] = stable
        invocation = None
        if not row['qualified_reuse']:
            invocation = verify_invocation(read, row, task, plan, args.cpu_plan_sha256, coordinator,
                arguments_for(task, plan, completed, stable))
            invocations.append(invocation)
        if task['kind'] == 'bind_new_group':
            indices[task['group_id']], documents[task['group_id']] = verify_new_binding(read, row, task, plan, stable, receipt)
            continue
        if task['kind'] == 'original_cell':
            entry = original[task['group_id'], task['target']]
            expected_index = plan['execution']['input_index_sha256']
            expected_tests = refs['original_cell_tests']['sha256']
            expected_input = verify_original_input(read, entry, task, plan, original_cache)
        else:
            check(task['group_id'] in indices, 'A new cell completed without its sealed input binding')
            entry = indices[task['group_id']]['cells'][task['target_index']]
            expected_index = completed['bind_g{:06d}'.format(task['group_id'])]['receipt_sha256']
            expected_tests = refs['new_cell_tests']['sha256']
            expected_input = documents[task['group_id']][task['target']]
        cell, bootstrap = verify_cell(read, task, row, plan, entry['normalized_cell'], expected_index, expected_tests, expected_input, invocation)
        cells.append(cell)
        if bootstrap is not None:
            bootstraps.append(bootstrap)
    identities = [(row['worker_pid'], row['worker_start_ticks']) for row in invocations]
    check(len(identities) == len(set(identities)) == len(completed) - len(reuse_ids),
        'Tasks lost, duplicated or bypassed their worker invocations')
    if completion is not None:
        check(completion['launched_tasks'] == len(invocations), 'Declared CPU launches differ from independently verified invocations')
    points = sorted([(datetime.fromisoformat(row['started_at']), 1) for row in invocations]
        + [(datetime.fromisoformat(row['completed_at']), -1) for row in invocations])
    active, maximum = 0, 0
    for _, direction in points:
        active += direction
        check(active >= 0, 'Invalid worker execution intervals')
        maximum = max(maximum, active)
    check(active == 0 and maximum <= plan['resource_contract']['workers'], 'CPU worker execution exceeded its registered concurrency')
    result = dict(schema_version=1, status='observed_cpu_cells_verified_pending_complete_pipeline', errors=[],
        cpu_plan_sha256=args.cpu_plan_sha256, execution_plan_sha256=plan['grid']['execution_plan_sha256'],
        observed_completed_tasks=len(completed), registered_tasks=len(tasks), observation_cells=len(cells),
        blackbox_bootstrap_cells=len(bootstraps), whitebox_replay_cells=len(cells) - len(bootstraps),
        new_group_bindings=len(indices), qualified_complete_reuse_cells=len(reuse_ids), new_invocations=len(invocations),
        maximum_observed_concurrency=maximum, tests=tests, analysis_pipeline_accepted=False,
        original_gpu_root_accepted=False, scientific_acceptance=False, independent_confirmation=False,
        formal_metrics_eligible=False, source_root_modified=False, new_model_calls=0,
        prediction_bootstraps_recomputed=0, confidence_intervals_produced=0,
        limitations=['Independent implementation audit, not an independent investigator.',
            'Consumed AP bytes and existing full official replays are checked; predictions are not resampled again.',
            'A complete CPU audit does not replace final GPU-root acceptance, scientific judgment or author promotion.'])
    if completion is not None:
        path = cpu_root / 'matrix/analysis.json'
        check(completion.get('matrix_file') == str(path), 'Final matrix points outside this CPU root')
        matrix = read.json(path, completion['matrix_sha256'])
        result['matrix_verification'] = verify_matrix(plan['execution']['prepared_plan'], cells, bootstraps, plan['shared_validation'], matrix)
        result['reports'] = verify_reports(read, path.parent, matrix, completion['report_manifest_sha256'])
        result.update(status='independently_verified_cpu_pipeline_pending_formal_root_acceptance',
            analysis_pipeline_accepted=True, cpu_completion_sha256=args.completion_sha256)
    read.unchanged()
    result['checked_input_sha256'] = read.checked
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-root', type=Path, required=True)
    parser.add_argument('--cpu-plan-sha256', required=True)
    parser.add_argument('--tests-receipt', type=Path, required=True)
    parser.add_argument('--tests-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--observed-prefix', action='store_true')
    parser.add_argument('--completion-sha256')
    parser.add_argument('--launch-completion-sha256')
    args = parser.parse_args(argv)
    args.cpu_root, args.tests_receipt, args.output = map(lambda p: p.absolute(), (args.cpu_root, args.tests_receipt, args.output))
    check(sys.platform == 'linux' and Path(sys.prefix).name == 'oda', 'Numerical audit requires server ODA')
    plan = Evidence().json(args.cpu_root / 'cpu_plan.json', args.cpu_plan_sha256)
    guard_output(args.output, args.cpu_root, plain(plan['inputs']['execution_worktree']))
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args)
    except Exception as exc:
        result = dict(status='failed', errors=[repr(exc)[:2000]], analysis_pipeline_accepted=False,
            scientific_acceptance=False, original_gpu_root_accepted=False, new_model_calls=0)
    result.update(auditor_sha256=sha(__file__), completed_at=datetime.now(timezone.utc).isoformat())
    with (args.output / 'audit.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({key: result.get(key) for key in ('status', 'errors', 'observation_cells', 'new_group_bindings', 'matrix_verification')}))
    return 1 if result['status'] == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
