"""Admitted final-followup invocation; numerical construction stays unchanged."""
from dataclasses import asdict
from pathlib import Path
import sys

from ..registry import AttackSpec
from ..runners.vcsf_final_followup_dispatch import ALIAS, validate_attack_invocation
from ..runners.vcsf_final_followup_execution_contract import (
    MODE, ENTRYPOINT, EXECUTION_PROTOCOL, read_execution_admission,
)
from ..runners.vcsf_research_plan import canonical_hash, require
from .factory import ATTACK_TYPES
from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig


def invocation_metadata(bound, runtime_reference, admission_reference, worker_reference):
    scope, group = bound['admission']['scope'], bound['group']
    return dict(protocol=EXECUTION_PROTOCOL, scope_sha256=scope['scope_sha256'],
        group_id=group['group_id'], dataset=group['dataset'], source=group['source'], seed=group['seed'],
        seed_schedule=group['seed_schedule'], parameters_sha256=group['parameters_sha256'],
        input_binding_sha256=bound['prepared']['input_binding_sha256'],
        runtime_reference=dict(runtime_reference), admission_reference=dict(admission_reference),
        worker_request_reference=dict(worker_reference), diagnostic_only=scope['diagnostic_only'])


def verify_worker_request(read, registry, descriptor, bound, output):
    from tools.audit_vcsf_cpu_analysis import plain
    from ..runners.vcsf_ablation_execution_contract import _verify_process_binding
    from ..runners.vcsf_efficacy_runner import process_start_ticks
    from ..runners.vcsf_structure_workers import verify_coordinator, verify_reservation, physical_gpus

    scope = bound['admission']['scope']
    reference = descriptor['worker_request']
    require(isinstance(reference, dict) and set(reference) == {'file', 'sha256'},
        'Require a pinned worker request')
    path = plain(Path(reference['file']))
    request = read.json(path, reference['sha256'])
    fields = {'runtime_reference', 'admission_reference', 'execution_root', 'scope_sha256',
        'worker_slot', 'physical_device', 'group_ids', 'coordinator_pid', 'coordinator_start_ticks',
        'reservation_fd', 'reservation_identity', 'gpu_uuid', 'run_binding_sha256'}
    require(set(request) == fields and type(request['worker_slot']) is int and
        0 <= request['worker_slot'] < len(scope['devices']), 'Invalid final-followup worker request')
    slot = request['worker_slot']
    require(path == output / 'workers' / str(slot) / 'request.json'
        and request['runtime_reference'] == descriptor['runtime']
        and request['admission_reference'] == descriptor['admission']
        and request['execution_root'] == descriptor['execution_root']
        and request['scope_sha256'] == scope['scope_sha256']
        and request['physical_device'] == scope['devices'][slot]
        and request['group_ids'] == scope['lanes'][slot]
        and descriptor['group_id'] in request['group_ids']
        and request['run_binding_sha256'] == descriptor['run_binding_sha256'],
        'Worker request differs from admitted runtime or whole-group assignment')
    binding = read.json(output / 'plan_binding.json', descriptor['run_binding_sha256'])
    expected = dict(runtime_reference=descriptor['runtime'], admission_reference=descriptor['admission'],
        scope_sha256=scope['scope_sha256'], execution_root=descriptor['execution_root'],
        devices=scope['devices'], lanes=scope['lanes'], payload_policy=scope['payload_policy'])
    require(canonical_hash(binding) == canonical_hash(expected), 'Execution root binding changed')
    _verify_process_binding(registry)
    entry = sys.modules.get('__main__')
    require(getattr(entry, '__file__', None) == str(registry.root / ENTRYPOINT) and
        getattr(entry, '_ENTRY_SOURCE_SHA256', None) == bound['admission']['runtime_sha256'][ENTRYPOINT]
        and getattr(entry, '_FOLLOWUP_WORKER_REQUEST_SHA256', None) == reference['sha256'],
        'Use the source-bound admitted worker entry')
    require(physical_gpus(scope['devices'])[slot] == request['gpu_uuid'],
        'Worker GPU UUID differs from the selected physical device')
    verify_coordinator(request['coordinator_pid'])
    require(type(request['coordinator_start_ticks']) is int and
        process_start_ticks(request['coordinator_pid']) == request['coordinator_start_ticks'],
        'Worker coordinator process identity changed')
    verify_reservation(request)
    read.unchanged()
    require(not (output / 'failure.json').exists() and not (output / 'completion.json').exists()
        and not (path.parent / 'failure.json').exists(), 'Execution root or worker is already terminal')
    return request


def _verify_worker_context(read, registry, descriptor, bound, output):
    from ..runners.vcsf_gpu_context_owner import RegisteredGpuOwner

    request = verify_worker_request(read, registry, descriptor, bound, output)
    owner = getattr(sys.modules['__main__'], '_FOLLOWUP_GPU_OWNER', None)
    require(type(owner) is RegisteredGpuOwner and canonical_hash(owner._request) == canonical_hash(request),
        'Worker has no matching registered GPU context owner')
    owner.verify()


def resolve_followup_execution(registry, descriptor, *, dataset_id, split, source_id,
        attack_id, seed, max_images, parameter_overrides, run_metadata, budget_profile,
        image_ids=None, seed_offsets=None, input_transform=None, output_dir=None, device=None):
    from tools.audit_vcsf_cpu_analysis import Evidence
    from tools.vcsf_final_followup_bound_dispatch import load_bound_execution_group
    from ..runners.vcsf_efficacy_contract import child

    fields = {'mode', 'execution_alias', 'runtime', 'admission', 'group_id',
        'execution_root', 'worker_request', 'run_binding_sha256'}
    require(isinstance(descriptor, dict) and set(descriptor) == fields and
        descriptor['mode'] == MODE and descriptor['execution_alias'] == attack_id == ALIAS,
        'Unsupported final-followup execution descriptor')
    require(ALIAS not in registry.attacks and ALIAS not in ATTACK_TYPES,
        'Final followup must remain a process-local overlay')
    require(device == 'cuda:0', 'Final-followup worker requires local cuda:0')
    output = child(registry.root, descriptor['execution_root'])
    category = 'diagnostics' if max_images is not None else 'experiments'
    require(output.parent == registry.root / 'outputs' / category / EXECUTION_PROTOCOL,
        'Final-followup execution root escaped its namespace')
    require(type(descriptor['group_id']) is int and descriptor['group_id'] > 0,
        'Invalid final-followup group identity')
    group_relative = output.relative_to(registry.root) / 'groups' / '{:06d}'.format(descriptor['group_id'])
    group_root = child(registry.root, group_relative)
    child(registry.root, group_relative / 'attack')
    read = Evidence()
    bound = load_bound_execution_group(read, registry, descriptor['runtime'], descriptor['admission'],
        descriptor['group_id'], str(group_root), device, max_images=max_images)
    validate_attack_invocation(bound['prepared'], dataset_id=dataset_id, split=split,
        source_id=source_id, attack_id=attack_id, seed=seed, max_images=max_images,
        parameter_overrides=parameter_overrides, budget_profile=budget_profile, output_dir=output_dir,
        image_ids=image_ids, seed_offsets=seed_offsets, input_transform=input_transform)
    expected_metadata = invocation_metadata(bound, descriptor['runtime'], descriptor['admission'],
        descriptor['worker_request'])
    require(isinstance(run_metadata, dict) and canonical_hash(run_metadata) == canonical_hash(expected_metadata),
        'Final-followup invocation metadata differs from the bound group')
    config = VCSFCommonAnchorConfig.from_mapping(dict(parameter_overrides))
    config.validate()
    require(canonical_hash(asdict(config)) == bound['group']['parameters_sha256'] and not config.is_reference(),
        'Final-followup configuration normalization changed')
    _verify_worker_context(read, registry, descriptor, bound, output)
    read_execution_admission(read, registry, bound['catalogue'], bound['admission']['scope'],
        descriptor['runtime'], descriptor['admission'])
    attack_leaf = child(registry.root, group_relative / 'attack')
    require(not attack_leaf.exists() and not (output / 'failure.json').exists()
        and not (output / 'completion.json').exists()
        and not (Path(descriptor['worker_request']['file']).parent / 'failure.json').exists(),
        'Do not overwrite an existing or terminal attack execution')
    return AttackSpec(id=ALIAS, display_name='VCSF final follow-up', executor='native',
        parameters=dict(parameter_overrides), metadata=dict(expected_metadata,
            fidelity_status='frozen_final_followup_pending_result_acceptance',
            semantic_contract='final_configuration_after_fullval_selection_not_independent_confirmation',
            global_registry_mutated=False, global_factory_mutated=False, formal_AP_eligible=False))
