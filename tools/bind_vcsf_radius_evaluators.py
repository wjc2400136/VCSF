"""Compare current target test configs with pinned historical effective configs."""
import argparse
import importlib.metadata
import math
from pathlib import Path

from lgp.io import atomic_json, file_digest
from lgp.registry import Registry
from lgp.runtime_config import build_runtime_config
from lgp.runners.vcsf_research_plan import canonical_hash
from lgp.runners.vcsf_oblivious_admission import bound_json, isolate_output


def normalized_model_config(value):
    # Same declarative comparison, without importing a retired execution contract.
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError('Non-string configuration key')
        return {key: normalized_model_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized_model_config(item) for item in value]
    if value is not None and type(value) not in (str, bool, int, float):
        raise ValueError('Non-declarative configuration value')
    if type(value) is float and not math.isfinite(value):
        raise ValueError('Non-finite configuration value')
    if isinstance(value, str) and value.startswith('/'):
        return str(Path(value).resolve())
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['historical', 'targets']:
        parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--' + name + '-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    registry = Registry()
    historical = bound_json(args.historical, args.historical_sha256)
    targets = bound_json(args.targets, args.targets_sha256)
    if (targets['status'] != 'radius_target_bytes_bound_pending_evaluator_and_reuse'
            or targets['canonical_order'] != registry.paper_order
            or historical['targets'] != registry.paper_order):
        raise ValueError('Historical or destination target scope differs')
    rows = [row for row in historical['model_configs'] if row['role'] == 'target']
    if [row['model'] for row in rows] != registry.paper_order:
        raise ValueError('Historical target config panel is incomplete')
    output = args.output.resolve()
    protected = [args.historical.resolve().parent, args.targets.resolve().parent,
                 registry.root / 'data', registry.root / 'checkpoints']
    protected.extend(Path(row['effective_config']['file']).parent for row in rows)
    isolate_output(output, protected)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {str(args.historical.resolve()): args.historical_sha256,
                str(args.targets.resolve()): args.targets_sha256}
    try:
        packages = {name: importlib.metadata.version(name) for name in historical['packages']}
        if packages != historical['packages']:
            raise ValueError('Historical dependency versions differ')
        comparisons = []
        for row in rows:
            model = row['model']
            if row['checkpoint_sha256'] != targets['targets'][model]['sha256']:
                raise ValueError('Historical config checkpoint differs: ' + model)
            ref = row['effective_config']
            original = bound_json(ref['file'], ref['sha256'])
            evidence[str(Path(ref['file']).resolve())] = ref['sha256']
            current = build_runtime_config(registry.model(model), registry.dataset('coco'),
                output / 'configs' / model, mode='test', test_split='val',
                dump_config=False).to_dict()
            current.pop('work_dir', None)
            current = normalized_model_config(current)
            before = canonical_hash(normalized_model_config(original))
            after = canonical_hash(normalized_model_config(current))
            if before != after:
                atomic_json(output / ('mismatch_' + model + '.json'),
                            dict(original=original, current=current))
                raise ValueError('Historical model/preprocessing/evaluator differs: ' + model)
            atomic_json(output / 'configs' / model / 'effective_config.json', current)
            comparisons.append(dict(model=model, normalized_sha256=after,
                original=ref, checkpoint_sha256=row['checkpoint_sha256']))
        for path, digest in evidence.items():
            if file_digest(Path(path)) != digest:
                raise ValueError('Evaluator metadata changed during comparison')
        receipt = dict(status='radius_target_configs_match_historical_pending_reuse',
            comparisons=comparisons, packages=packages, evidence=evidence,
            normalization='exclude_work_dir_resolve_absolute_paths_json_sequences',
            model_calls=0, AP_replays=0, destination_reuse_qualified=False,
            formal_execution_admitted=False, tool_sha256=file_digest(Path(__file__)))
        atomic_json(output / 'receipt.json', receipt)
        atomic_json(output / 'terminal.json', dict(status=receipt['status'],
            targets=len(comparisons), receipt_sha256=file_digest(output / 'receipt.json')))
    except BaseException as exc:
        atomic_json(output / 'terminal.json', dict(status='failed', error=repr(exc),
                                                 formal_execution_admitted=False))
        raise


if __name__ == '__main__':
    main()
