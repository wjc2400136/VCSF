"""Read dataset-aware assets and rebuild configs in ODA; no detector inference."""
from copy import deepcopy
import importlib.metadata
from pathlib import Path
import sys

from tools.audit_vcsf_cpu_analysis import check, plain
from lgp.io import atomic_json
from lgp.runners.vcsf_ablation_preflight import required_packages
from lgp.runners.vcsf_ablation_runtime_inputs import _file, _head, _normal
from lgp.runners.vcsf_final_followup_plan import PROTOCOL
from lgp.runners.vcsf_research_plan import canonical_hash
from lgp.runners.vcsf_scale_execution_contract import normalized_model_config


def _server():
    check(sys.platform == 'linux' and Path(sys.prefix).name == 'oda' and sys.executable == str(Path(sys.prefix) / "bin" / "python")
        and sys.version_info[:3] == (3, 8, 20), 'Use the pinned server ODA interpreter')


def _asset(read, path):
    value = _file(Path(path).resolve(strict=True))
    read.bytes(plain(Path(value['file'])), value['sha256'])
    return value


def _collect_dataset_inputs(read, registry, dataset, directory, *, write):
    _server()
    from lgp.data.coco import CocoIndex
    from lgp.modeling import checkpoint_path
    from lgp.paths import project_root
    from lgp.runtime_config import build_runtime_config

    check(project_root().resolve() == registry.root.resolve(), 'Registry and active project roots differ')
    protocol = registry.protocols[PROTOCOL]
    check(dataset in protocol['datasets'], 'Dataset outside the final follow-up scope')
    scope = protocol['datasets'][dataset]
    directory = plain(Path(directory))
    check(directory.is_dir(), 'Caller must own an existing dataset output directory')
    if write:
        check(not any(directory.iterdir()), 'Dataset input output must be fresh and empty')
    packages = {name: importlib.metadata.version(name) for name in required_packages(registry.root)}
    check(packages == required_packages(registry.root), 'Installed core packages differ from pinned versions')
    specification = registry.dataset(dataset)
    index = CocoIndex(specification, scope['split'])
    ids = [row['id'] for row in index.images]
    check(len(ids) == scope['images'] and all(type(v) is int and v > 0 for v in ids) and
        len(ids) == len(set(ids)) and ids == sorted(ids), 'Full ordered dataset IDs differ')
    annotation = _asset(read, index.annotation_path)
    rows = [dict(image_id=image['id'], position=position,
        input=_asset(read, index.image_path(image))) for position, image in enumerate(index.images)]
    manifest_path = directory / 'clean_images.json'
    if write:
        atomic_json(manifest_path, rows)
    clean_manifest = _asset(read, manifest_path)
    check(read.json(plain(manifest_path), clean_manifest['sha256']) == rows,
        'Saved clean manifest differs from actual dataset bytes')
    sources, targets = list(registry.source_ids()), list(registry.target_ids())
    checkpoints = {model: _asset(read, checkpoint_path(registry.model(model), specification))
        for model in targets}
    roles = [('source', model) for model in sources] + [('target', model) for model in targets]
    configs = []
    for role, model in roles:
        destination = directory / 'configs' / role / model
        if write:
            destination.mkdir(parents=True, exist_ok=False)
        check(destination.is_dir(), 'Missing saved role config directory')
        config = build_runtime_config(registry.model(model), specification, destination,
            mode='test', test_split=scope['split'], dump_config=write)
        resolved = _asset(read, destination / 'runtime_config.py')
        raw = read.bytes(plain(Path(resolved['file'])), resolved['sha256'], 16 << 20)
        check(raw == config.pretty_text.encode('utf-8'), 'Saved role config differs from fresh reconstruction')
        effective = config.to_dict()
        effective.pop('work_dir', None)
        effective = _normal(effective)
        effective_path = destination / 'effective_config.json'
        if write:
            atomic_json(effective_path, effective)
        effective_ref = _asset(read, effective_path)
        check(read.json(plain(effective_path), effective_ref['sha256']) == effective,
            'Effective role config differs from fresh reconstruction')
        configs.append(dict(role=role, model=model, checkpoint_sha256=checkpoints[model]['sha256'],
            resolved_config=resolved, effective_config=effective_ref,
            normalized_effective_sha256=canonical_hash(normalized_model_config(effective))))
    result = dict(schema_version=1, status='actual_dataset_inputs_collected_pending_independent_acceptance',
        dataset=dataset, split=scope['split'], images=len(ids), sources=sources, targets=targets,
        ordered_image_ids=ids, image_ids_sha256=canonical_hash(ids), annotation=annotation,
        clean_image_manifest=clean_manifest, clean_rows_content_sha256=canonical_hash(rows),
        checkpoints=checkpoints, model_configs=configs, packages=packages, base_commit=_head(registry.root),
        dataset_output_directory=str(directory), actual_assets_read=True,
        independent_input_acceptance=False, checkpoint_qualification_accepted=False,
        implementation_bridge_accepted=False, reuse_accepted=False, formal_execution_admission=False,
        model_calls=0, AP_replays=0)
    read.unchanged()
    return result, rows


def collect_dataset_inputs(read, registry, dataset, owned_directory):
    """Writes metadata/config artifacts only inside the caller's fresh directory."""
    return _collect_dataset_inputs(read, registry, dataset, owned_directory, write=True)


def verify_saved_dataset_inputs(read, registry, saved):
    """Rebuild against saved paths without writing; caller binds source and receipt."""
    current, rows = _collect_dataset_inputs(read, registry, saved['dataset'],
        saved['dataset_output_directory'], write=False)
    check(canonical_hash(current) == canonical_hash(saved),
        'Published shared inputs differ from fresh asset/config reconstruction')
    return dict(status='dataset_asset_reconstruction_matches_pending_outer_acceptance',
        shared_input_content_sha256=canonical_hash(saved),
        clean_rows_content_sha256=canonical_hash(rows), dataset=saved['dataset'],
        images=saved['images'], checkpoints=len(saved['checkpoints']), model_roles=len(saved['model_configs']),
        model_calls=0, AP_replays=0, independent_input_acceptance=False,
        checkpoint_qualification_accepted=False, implementation_bridge_accepted=False,
        reuse_accepted=False, formal_execution_admission=False), deepcopy(rows)
