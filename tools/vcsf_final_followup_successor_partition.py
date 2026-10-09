"""Pure migration partition, not a stop proof, reuse qualification or admission.

The caller must authenticate the original catalogue before calling, rejecting
duplicate keys while decoding it. C maps integer whole-group IDs to exact
file/sha256 qualification references. References are opaque: this module never
reads them. The outer admission layer must authenticate their contents, bind
each qualification to the complete original group and authenticate predecessor
termination and independent physical device ownership before any execution.
Partial groups are absent from C and are scheduled again in full, in a fresh
successor namespace. Nothing here changes or stops the running predecessor.
"""
from copy import deepcopy
import re

from lgp.runners.vcsf_research_plan import canonical_hash, require


def _sha(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _reference(value):
    require(type(value) is dict and set(value) == {'file', 'sha256'},
        'Require exact file/sha256 reference')
    name = value['file']
    require(type(name) is str and bool(name.strip()) and name == name.strip()
        and not any(ord(c) < 32 for c in name) and _sha(value['sha256']),
        'Invalid reference file or sha256')


def _devices(value, counts):
    require(type(value) is list and len(value) in counts and
        all(type(d) is str and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d)
            for d in value) and len(set(value)) == len(value),
        'Require distinct explicit canonical CUDA devices')


def _json(value):
    """Reject coercible keys/types before hashing a caller-owned JSON tree."""
    if type(value) is dict:
        require(all(type(k) is str for k in value), 'Require string JSON keys')
        for item in value.values():
            _json(item)
    elif type(value) is list:
        for item in value:
            _json(item)
    else:
        require(value is None or type(value) in (str, int, float, bool),
            'Require plain JSON values')


def _catalogue(catalogue):
    require(type(catalogue) is dict, 'Require original catalogue object')
    _json(catalogue)
    canonical_hash(catalogue)
    groups = catalogue.get('groups')
    require(type(groups) is list and len(groups) == 16 and
        all(type(g) is dict for g in groups), 'Require complete 16-group catalogue')
    required = {'group_id', 'dataset', 'split', 'images', 'source', 'seed',
        'seed_schedule', 'targets', 'report_roles', 'parameters_sha256',
        'implementation_inventory_sha256', 'disposition', 'input_binding',
        'input_readiness_accepted', 'reuse_accepted'}
    require(all(required <= set(g) for g in groups), 'Incomplete original group identity')
    require(all(type(g['group_id']) is int and g['group_id'] == i
        for i, g in enumerate(groups, 1)), 'Noncanonical or duplicate group IDs')
    require(catalogue.get('groups_sha256') == canonical_hash(groups), 'Group identity hash differs')
    require(type(catalogue.get('parameters')) is dict and
        type(catalogue.get('producer_runtime_sha256')) is dict and
        bool(catalogue['producer_runtime_sha256']) and
        all(bool(k.strip()) and _sha(v) for k, v in catalogue['producer_runtime_sha256'].items()),
        'Incomplete method identity')
    parameters = canonical_hash(catalogue['parameters'])
    inventory = canonical_hash(catalogue['producer_runtime_sha256'])
    targets = groups[0]['targets']
    require(type(targets) is list and len(targets) == 16 and
        all(type(t) is str and bool(t.strip()) for t in targets) and len(set(targets)) == 16,
        'Incomplete or duplicate targets')
    sources = [groups[0]['source']] + [g['source'] for g in groups[5:10]]
    require(all(type(s) is str and s in targets for s in sources) and len(set(sources)) == 6,
        'Incomplete or duplicate sources')
    identities = [('coco', sources[0], seed) for seed in range(42, 47)]
    identities += [('coco', s, 42) for s in sources[1:]]
    identities += [('voc', s, 42) for s in sources]
    for g, identity in zip(groups, identities):
        gid = g['group_id']
        roles = (['final_stability'] if gid <= 5 else [])
        if gid == 1 or gid >= 6:
            roles += [identity[0] + '_main_vcsf']
        require(type(g['seed']) is int and
            (g['dataset'], g['source'], g['seed']) == identity and g['split'] == 'val' and
            type(g['images']) is int and g['images'] == (5000 if identity[0] == 'coco' else 4952)
            and g['targets'] == targets and g['report_roles'] == roles and
            g['seed_schedule'] == 'selected_position' and
            g['parameters_sha256'] == parameters and g['implementation_inventory_sha256'] == inventory,
            'Original complete group identity differs')
        require(g['disposition'] == ('reuse_candidate_unqualified' if gid == 1 else 'prospective_new')
            and g['input_binding'] is None and g['input_readiness_accepted'] is False
            and g['reuse_accepted'] is False, 'Original group disposition or readiness differs')
    universe = list(range(2, 17))
    require(canonical_hash(catalogue.get('prospective_new_group_ids')) == canonical_hash(universe),
        'Original new-group universe differs')
    counts = {'group_count': len(groups), 'target_cell_count': sum(len(g['targets']) for g in groups),
        'prospective_new_images': sum(g['images'] for g in groups[1:]),
        'prospective_new_target_cells': sum(len(g['targets']) for g in groups[1:])}
    require((len(universe), counts['prospective_new_target_cells'], counts['prospective_new_images'])
        == (15, 240, 74712), 'Original total coverage differs')
    require(all(type(catalogue.get(k)) is int and catalogue[k] == v for k, v in counts.items()),
        'Catalogue coverage counters differ')
    slots = [dict(group_id=g['group_id'], target=t, status='NR', metrics=None)
        for g in groups for t in targets]
    require(canonical_hash(catalogue.get('result_slots')) == canonical_hash(slots),
        'Original complete result slots differ')
    old_devices = catalogue.get('devices')
    _devices(old_devices, (1, 2))
    require(canonical_hash(catalogue.get('prospective_lanes')) ==
        canonical_hash([universe[i::len(old_devices)] for i in range(len(old_devices))]),
        'Original lanes differ')
    return groups, universe


def compile_successor_partition(catalogue, completed_group_references,
        predecessor_stop_reference, devices):
    """Return deterministic U-C on four lanes, optionally viewed as two pairs.

    Only already-authenticated catalogue content is accepted by contract; a
    matching self-hash is not authentication. C references must each describe
    one entire group, all original images and all targets, never partial work.
    This function validates reference syntax, NOT evidence or current liveness.
    """
    groups, universe = _catalogue(catalogue)
    _devices(devices, (4,))
    _reference(predecessor_stop_reference)
    require(type(completed_group_references) is dict and
        all(type(gid) is int and gid in universe for gid in completed_group_references),
        'C must be an integer-keyed whole-group reference mapping within U')
    seen_files, seen_hashes = set(), set()
    for ref in completed_group_references.values():
        _reference(ref)
        require(ref['file'] not in seen_files and ref['sha256'] not in seen_hashes,
            'Duplicate qualification reference')
        require(ref['file'] != predecessor_stop_reference['file'] and
            ref['sha256'] != predecessor_stop_reference['sha256'],
            'Stop reference cannot substitute for group qualification')
        seen_files.add(ref['file'])
        seen_hashes.add(ref['sha256'])
    completed = [gid for gid in universe if gid in completed_group_references]
    missing = [gid for gid in universe if gid not in completed_group_references]
    lanes = [missing[i::4] for i in range(4)]

    def coverage(ids):
        return dict(groups=len(ids), target_cells=sum(len(groups[i - 1]['targets']) for i in ids),
            images=sum(groups[i - 1]['images'] for i in ids))

    result = dict(schema_version=1, status='planned_not_admitted',
        catalogue=deepcopy(catalogue), catalogue_content_sha256=canonical_hash(catalogue),
        universe_group_ids=universe, completed_group_ids=completed, successor_group_ids=missing,
        completed_groups=[dict(group=deepcopy(groups[i - 1]),
            group_content_sha256=canonical_hash(groups[i - 1]),
            qualification_reference=deepcopy(completed_group_references[i])) for i in completed],
        successor_groups=[deepcopy(groups[i - 1]) for i in missing],
        predecessor_stop_reference=deepcopy(predecessor_stop_reference),
        devices=list(devices), lanes=lanes,
        dual_card_sets=[dict(lane_ids=[i, i + 1], devices=list(devices[i:i + 2]),
            lanes=deepcopy(lanes[i:i + 2])) for i in (0, 2)],
        total_coverage=coverage(universe), completed_coverage=coverage(completed),
        successor_coverage=coverage(missing), original_reuse_group_ids=[1],
        reuse_policy='whole_groups_only_partial_groups_rerun_in_full',
        outer_requirements=['authenticate_original_catalogue',
            'authenticate_each_complete_group_qualification_and_original_identity',
            'authenticate_predecessor_stop_evidence_and_current_ownership',
            'verify_four_independent_physical_devices_and_fresh_successor_namespace',
            'issue_separate_successor_admission'],
        catalogue_authenticated_here=False, qualification_references_authenticated=False,
        predecessor_stop_verified=False, device_availability_verified=False,
        reuse_accepted=False, formal_execution_admission=False, runner_armed=False,
        live_touched=False, model_calls=0, AP_replays=0)
    result['partition_sha256'] = canonical_hash(result)
    return result
