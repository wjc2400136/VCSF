"""Authenticate successor invocation/ownership without changing numerical code.

The successor loader alone consumes admission; no predecessor fallback exists.
The admitted runtime source map must explicitly include the execution entry,
not merely the preparation entry. This module never creates a CUDA context.
"""
from dataclasses import asdict
from pathlib import Path
import sys

from ..registry import AttackSpec
from ..runners.vcsf_final_followup_dispatch import ALIAS, validate_attack_invocation
from ..runners.vcsf_final_followup_successor_contract import MODE, EXECUTION_PROTOCOL, verify_successor_scope
from ..runners.vcsf_final_followup_successor_coordinator import ENTRYPOINT
from ..runners.vcsf_research_plan import canonical_hash, require
from .factory import ATTACK_TYPES
from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
from .vcsf_final_followup_execution_isolation import invocation_metadata


def _descriptor(descriptor):
    fields = {'mode', 'execution_alias', 'runtime', 'admission', 'group_id',
        'execution_root', 'worker_request', 'run_binding_sha256'}
    require(type(descriptor) is dict and set(descriptor) == fields
        and descriptor['mode'] == MODE and descriptor['execution_alias'] == ALIAS
        and type(descriptor['group_id']) is int and descriptor['group_id'] > 0
        and descriptor['group_id'] not in (2, 3), 'Unsupported successor execution descriptor')


def verify_worker_request(read, registry, descriptor, bound, output):
    """Verify an already-admitted bound group against the actual four-lane worker."""
    from tools.audit_vcsf_cpu_analysis import plain, is_sha
    from ..runners.vcsf_efficacy_contract import child
    from ..runners.vcsf_ablation_execution_contract import _verify_process_binding
    from ..runners.vcsf_efficacy_runner import process_start_ticks
    from ..runners.vcsf_structure_workers import verify_coordinator, verify_reservation, physical_gpus

    _descriptor(descriptor)
    scope = bound['admission']['scope']
    scope = verify_successor_scope(registry, bound['catalogue'], scope['partition'], scope,
        partition_reference=scope['partition_reference'])
    require(len(scope['devices']) == len(scope['lanes']) == 4
        and {2, 3} <= set(scope['partition']['completed_group_ids']),
        'Require four successor lanes and protected completed groups 2/3')
    selected = [group for group in scope['groups'] if group['group_id'] == descriptor['group_id']]
    require(len(selected) == 1 and canonical_hash(selected[0]) == canonical_hash(bound['group']),
        'Bound group differs from the reconstructed successor identity')
    category = 'diagnostics' if scope['max_images'] is not None else 'experiments'
    require(output == child(registry.root, descriptor['execution_root'])
        and output.parent == registry.root / 'outputs' / category / EXECUTION_PROTOCOL,
        'Successor worker output escaped its namespace')
    reference = descriptor['worker_request']
    require(type(reference) is dict and set(reference) == {'file', 'sha256'}
        and type(reference['file']) is str and is_sha(reference['sha256']),
        'Require a pinned worker request')
    path = plain(Path(reference['file']))
    request = read.json(path, reference['sha256'])
    fields = {'runtime_reference', 'admission_reference', 'execution_root', 'scope_sha256',
        'worker_slot', 'physical_device', 'group_ids', 'coordinator_pid', 'coordinator_start_ticks',
        'reservation_fd', 'reservation_identity', 'gpu_uuid', 'run_binding_sha256'}
    require(type(request) is dict and set(request) == fields and type(request['worker_slot']) is int
        and 0 <= request['worker_slot'] < 4, 'Invalid successor worker request')
    require(type(request['reservation_fd']) is int and request['reservation_fd'] >= 0
        and type(request['reservation_identity']) is list and len(request['reservation_identity']) == 2
        and all(type(value) is int and value >= 0 for value in request['reservation_identity'])
        and type(request['gpu_uuid']) is str and bool(request['gpu_uuid']),
        'Malformed worker reservation identity')
    slot = request['worker_slot']
    require(path == output / 'workers' / str(slot) / 'request.json'
        and request['runtime_reference'] == descriptor['runtime']
        and request['admission_reference'] == descriptor['admission']
        and request['execution_root'] == descriptor['execution_root']
        and request['scope_sha256'] == scope['scope_sha256']
        and request['physical_device'] == scope['devices'][slot]
        and canonical_hash(request['group_ids']) == canonical_hash(scope['lanes'][slot])
        and descriptor['group_id'] in scope['lanes'][slot]
        and bound['group']['group_id'] == descriptor['group_id']
        and request['run_binding_sha256'] == descriptor['run_binding_sha256'],
        'Worker request differs from admitted runtime or whole-group assignment')
    binding = read.json(output / 'plan_binding.json', descriptor['run_binding_sha256'])
    expected = dict(runtime_reference=descriptor['runtime'], admission_reference=descriptor['admission'],
        scope_sha256=scope['scope_sha256'], execution_root=descriptor['execution_root'],
        devices=scope['devices'], lanes=scope['lanes'], payload_policy=scope['payload_policy'])
    require(canonical_hash(binding) == canonical_hash(expected), 'Execution root binding changed')
    sources = bound['admission']['runtime_sha256']
    require(type(sources) is dict and is_sha(sources.get(ENTRYPOINT)),
        'Admission must explicitly bind the successor execution entry')
    read.bytes(child(registry.root, ENTRYPOINT), sources[ENTRYPOINT])
    _verify_process_binding(registry)
    entry = sys.modules.get('__main__')
    require(getattr(entry, '__file__', None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, '_ENTRY_SOURCE_SHA256', None) == sources[ENTRYPOINT]
        and getattr(entry, '_FOLLOWUP_WORKER_REQUEST_SHA256', None) == reference['sha256'],
        'Use the source-bound admitted successor worker entry')
    uuids = physical_gpus(scope['devices'])
    require(len(uuids) == len(set(uuids)) == 4 and uuids[slot] == request['gpu_uuid'],
        'Worker GPU UUID differs from the four-device assignment')
    verify_coordinator(request['coordinator_pid'])
    require(type(request['coordinator_start_ticks']) is int
        and process_start_ticks(request['coordinator_pid']) == request['coordinator_start_ticks'],
        'Worker coordinator process identity changed')
    verify_reservation(request)
    read.unchanged()
    require(not any((root / name).exists() or (root / name).is_symlink()
        for root in (output, path.parent) for name in ('failure.json', 'completion.json')),
        'Execution root or worker is already terminal')
    return request


def _verify_worker_context(read, registry, descriptor, bound, output):
    from ..runners.vcsf_gpu_context_owner import RegisteredGpuOwner

    request = verify_worker_request(read, registry, descriptor, bound, output)
    owner = getattr(sys.modules['__main__'], '_FOLLOWUP_GPU_OWNER', None)
    require(type(owner) is RegisteredGpuOwner and canonical_hash(owner._request) == canonical_hash(request),
        'Worker has no matching registered GPU context owner')
    owner.verify()


def resolve_followup_successor_execution(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None, device=None):
    from tools.vcsf_successor_evidence import SuccessorEvidence as Evidence
    from tools.vcsf_final_followup_successor_bound_dispatch import load_bound_successor_execution_group
    from ..runners.vcsf_efficacy_contract import child

    _descriptor(descriptor)
    require(attack_id == ALIAS and ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        'Successor must remain a process-local overlay')
    require(device == 'cuda:0', 'Successor worker requires local cuda:0')
    output = child(registry.root, descriptor['execution_root'])
    category = 'diagnostics' if max_images is not None else 'experiments'
    require(output.parent == registry.root / 'outputs' / category / EXECUTION_PROTOCOL,
        'Successor execution root escaped its namespace')
    group_relative = output.relative_to(registry.root) / 'groups' / '{:06d}'.format(descriptor['group_id'])
    group_root = child(registry.root, group_relative)
    child(registry.root, group_relative / 'attack')
    read = Evidence()
    bound = load_bound_successor_execution_group(read, registry, descriptor['runtime'], descriptor['admission'],
        descriptor['group_id'], str(group_root), device, max_images=max_images)
    validate_attack_invocation(bound['prepared'], dataset_id=dataset_id, split=split,
        source_id=source_id, attack_id=attack_id, seed=seed, max_images=max_images,
        parameter_overrides=parameter_overrides, budget_profile=budget_profile, output_dir=output_dir,
        image_ids=image_ids, seed_offsets=seed_offsets, input_transform=input_transform)
    expected_metadata = invocation_metadata(bound, descriptor['runtime'], descriptor['admission'],
        descriptor['worker_request'])
    require(type(run_metadata) is dict and canonical_hash(run_metadata) == canonical_hash(expected_metadata),
        'Successor invocation metadata differs from the bound group')
    config = VCSFCommonAnchorConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == bound['group']['parameters_sha256'] and not config.is_reference(),
        'Successor configuration normalization changed')
    _verify_worker_context(read, registry, descriptor, bound, output)
    read.full_unchanged()
    _verify_worker_context(read, registry, descriptor, bound, output)
    attack_leaf = child(registry.root, group_relative / 'attack')
    require(not attack_leaf.exists() and not any((root / name).exists() or (root / name).is_symlink()
        for root in (output, Path(descriptor['worker_request']['file']).parent)
        for name in ('failure.json', 'completion.json')),
        'Do not overwrite an existing or terminal attack execution')
    return AttackSpec(id=ALIAS, display_name='VCSF final follow-up', executor='native',
        parameters=dict(parameter_overrides), metadata=dict(expected_metadata,
            fidelity_status='frozen_final_followup_pending_result_acceptance',
            semantic_contract='final_configuration_after_fullval_selection_not_independent_confirmation',
            global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False))
