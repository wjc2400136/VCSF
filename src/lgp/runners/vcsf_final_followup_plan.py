"""Compile the frozen follow-up topology without admitting or executing work."""
from copy import deepcopy
import hashlib
import json
import re

from .vcsf_research_plan import canonical_hash, require


FREEZE_SHA256 = 'a697b1331cf54808633bcfc57d6f07adb630ef41beb7fea709402b749e1418d0'
PROTOCOL = 'vcsf_final_followup_preparation'


def _authenticated_json(payload, expected, label):
    require(isinstance(payload, bytes), label + ' must be original file bytes')
    require(hashlib.sha256(payload).hexdigest() == expected, label + ' hash differs')
    return json.loads(payload)


def _pointer(document, pointer):
    require(isinstance(pointer, str) and pointer.startswith('/'), 'Invalid JSON pointer')
    value = document
    for token in pointer[1:].split('/'):
        token = token.replace('~1', '/').replace('~0', '~')
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def prepare_final_followup_plan(registry, freeze_bytes, input_report_bytes, devices):
    """Return an unarmed design; bytes authentication is not live input readiness."""
    protocol = registry.protocols[PROTOCOL]
    freeze = _authenticated_json(freeze_bytes, FREEZE_SHA256, 'Frozen disposition')
    require(freeze['configuration_frozen'] is True and
        freeze['decision'] == 'KEEP_F_WITH_NARROWED_CLAIMS', 'Final identity is not frozen')
    require(freeze['runner_armed'] is False and
        freeze['formal_execution_admission'] is False, 'Unexpected freeze admission')
    report = _authenticated_json(input_report_bytes,
        freeze['authenticated_input_report']['sha256'], 'Authenticated input report')
    parameters = _pointer(report, freeze['candidate']['parameters_json_pointer'])
    inventory = _pointer(report, freeze['implementation']['inventory_json_pointer'])
    require(canonical_hash(parameters) == freeze['candidate']['parameters_sha256'],
        'Resolved parameter identity differs')
    require(isinstance(inventory, dict) and
        len(inventory) == freeze['implementation']['inventory_files'] and
        canonical_hash(inventory) == freeze['implementation']['inventory_sha256'],
        'Complete producer inventory differs')
    require(all(isinstance(path, str) and isinstance(digest, str) and
        re.fullmatch(r'[0-9a-f]{64}', digest) for path, digest in inventory.items()),
        'Malformed producer inventory')
    for path, digest in freeze['implementation']['critical_files'].items():
        require(inventory.get(path) == digest, 'Critical producer file differs: ' + path)
    require(freeze['next_scope']['final_stability_seeds'] == protocol['stability_seeds'],
        'Final seed scope differs')
    require(isinstance(devices, (list, tuple)) and len(devices) in protocol['device_counts'] and
        all(isinstance(d, str) and re.fullmatch(r'cuda:(0|[1-9][0-9]*)', d) for d in devices) and
        len(set(devices)) == len(devices), 'Require one or two distinct explicit devices')
    sources, targets = list(registry.source_ids()), list(registry.target_ids())
    require(len(sources) == len(set(sources)) == 6 and
        len(targets) == len(set(targets)) == 16 and set(sources) <= set(targets) and
        'faster_rcnn_r50' in sources, 'Canonical source/target panel differs')

    groups, lookup = [], {}

    def add(dataset, images, source, seed, role):
        key = (dataset, source, seed)
        if key in lookup:
            lookup[key]['report_roles'].append(role)
            return
        reuse = key == (protocol['stability_dataset'], protocol['stability_source'], protocol['main_seed'])
        row = dict(group_id=len(groups) + 1, dataset=dataset, split=protocol['datasets'][dataset]['split'],
            images=images, source=source, seed=seed, seed_schedule=protocol['seed_schedule'],
            targets=deepcopy(targets), report_roles=[role],
            parameters_sha256=freeze['candidate']['parameters_sha256'],
            implementation_inventory_sha256=freeze['implementation']['inventory_sha256'],
            disposition='reuse_candidate_unqualified' if reuse else 'prospective_new',
            input_binding=None, input_readiness_accepted=False, reuse_accepted=False)
        groups.append(row)
        lookup[key] = row

    for seed in freeze['next_scope']['final_stability_seeds']:
        dataset = protocol['stability_dataset']
        add(dataset, protocol['datasets'][dataset]['images'], protocol['stability_source'], seed, 'final_stability')
    for dataset, scope in protocol['datasets'].items():
        for source in sources:
            add(dataset, scope['images'], source, protocol['main_seed'], dataset + '_main_vcsf')
    new_ids = [g['group_id'] for g in groups if g['disposition'] == 'prospective_new']
    slots = [dict(group_id=g['group_id'], target=t, status='NR', metrics=None)
        for g in groups for t in targets]
    return dict(schema_version=1, status='prospective_final_followup_not_admitted',
        protocol_id=PROTOCOL, protocol_content_sha256=canonical_hash(protocol),
        freeze_sha256=FREEZE_SHA256,
        authenticated_input_report=deepcopy(freeze['authenticated_input_report']),
        parameters=deepcopy(parameters), producer_runtime_sha256=deepcopy(inventory),
        groups=groups, result_slots=slots, group_count=len(groups),
        target_cell_count=len(slots), prospective_new_group_ids=new_ids,
        unresolved_reuse_group_ids=[1], qualified_reuse_group_ids=[],
        prospective_lanes=[new_ids[i::len(devices)] for i in range(len(devices))],
        devices=list(devices), prospective_new_images=sum(g['images'] for g in groups
            if g['group_id'] in new_ids), prospective_new_target_cells=len(new_ids) * len(targets),
        groups_sha256=canonical_hash(groups), reuse_accepted=False,
        formal_execution_admission=False, runner_armed=False,
        live_inputs_verified=False, producer_files_readback_in_this_step=False,
        independent_result_acceptance=False, model_calls=0, AP_replays=0)
