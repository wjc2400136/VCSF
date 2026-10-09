"""Execute four reserved whole-group lanes; completion is not acceptance.

The separately implemented successor loader must authenticate admission before
dispatch. Missing loader or public entry is fatal before output or worker launch.
This coordinator never stops the predecessor or qualifies reusable groups.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time

from ..io import atomic_json, file_digest
from .vcsf_efficacy_contract import child
from .vcsf_efficacy_runner import process_start_ticks
from .vcsf_final_followup_successor_contract import EXECUTION_PROTOCOL, verify_successor_scope
from .vcsf_research_plan import canonical_hash, require
from .vcsf_structure_workers import (physical_gpus, gpu_reservations,
    verify_device_owner, worker_environment)


ENTRYPOINT = 'experiments/vcsf_final_followup_successor.py'


def inspect_execution(registry, runtime_reference, admission_reference, devices, max_images):
    from tools.audit_vcsf_cpu_analysis import plain
    from tools.vcsf_successor_evidence import SuccessorEvidence as Evidence
    from tools.vcsf_final_followup_successor_bound_dispatch import load_bound_successor_execution_group

    file_digest(child(registry.root, ENTRYPOINT))
    read = Evidence()
    runtime = read.json(plain(Path(runtime_reference['file'])), runtime_reference['sha256'])
    raw_scope = runtime['scope']
    require(raw_scope['group_ids'], 'Execution scope is empty')
    category = 'diagnostics' if max_images is not None else 'experiments'
    group_id = raw_scope['group_ids'][0]
    directory = registry.root / 'outputs' / category / EXECUTION_PROTOCOL / 'inspection' / 'groups' / '{:06d}'.format(group_id)
    bound = load_bound_successor_execution_group(read, registry, runtime_reference, admission_reference,
        group_id, str(directory), 'cuda:0', max_images=max_images)
    scope = bound['admission']['scope']
    scope = verify_successor_scope(registry, bound['catalogue'], scope['partition'], scope,
        partition_reference=scope['partition_reference'])
    require(canonical_hash(raw_scope) == canonical_hash(scope), 'Runtime and admitted scopes differ')
    require(devices is None or canonical_hash(devices) == canonical_hash(scope['devices']),
        'Requested devices differ from the admitted scope')
    require(len(scope['devices']) == len(scope['lanes']) == 4,
        'Successor requires exactly four registered devices and lanes')
    require(canonical_hash(max_images) == canonical_hash(scope['max_images']),
        'Requested image limit differs from the admitted scope')
    require({2, 3} <= set(scope['partition']['completed_group_ids']),
        'Current groups 2/3 cannot be successor recomputation')
    read.full_unchanged()
    return scope


def execute(registry, runtime_reference, admission_reference, devices, max_images, output):
    from tools.audit_vcsf_cpu_analysis import Evidence
    from .vcsf_ablation_execution_contract import _verify_process_binding

    scope = inspect_execution(registry, runtime_reference, admission_reference, devices, max_images)
    _verify_process_binding(registry)
    entry = sys.modules['__main__']
    require(getattr(entry, '__file__', None) == str(registry.root / ENTRYPOINT)
        and getattr(entry, '_ENTRY_SOURCE_SHA256', None) == file_digest(registry.root / ENTRYPOINT),
        'Use the source-bound public successor coordinator entry')
    output = Path(output).absolute()
    output = child(registry.root, output.relative_to(registry.root))
    category = 'diagnostics' if max_images is not None else 'experiments'
    require(output.parent == registry.root / 'outputs' / category / EXECUTION_PROTOCOL
        and not output.exists(), 'Use a fresh direct execution leaf; no resume or overwrite')
    uuids = physical_gpus(scope['devices'])
    require(len(uuids) == len(set(uuids)) == 4, 'Require four distinct physical GPU UUIDs')
    with gpu_reservations(uuids) as reservations:
        for uuid in uuids:
            verify_device_owner(uuid, os.getpid())
        require(canonical_hash(inspect_execution(registry, runtime_reference, admission_reference,
            devices, max_images)) == canonical_hash(scope), 'Admission changed before dispatch')
        output.mkdir(parents=True, exist_ok=False)
        processes, handles, entries = [], [], []
        read = Evidence()
        relative = output.relative_to(registry.root).as_posix()
        binding = dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
            scope_sha256=scope['scope_sha256'], execution_root=relative, devices=scope['devices'],
            lanes=scope['lanes'], payload_policy=scope['payload_policy'])
        try:
            atomic_json(output / 'plan_binding.json', binding)
            digest = file_digest(output / 'plan_binding.json')
            read.json(output / 'plan_binding.json', digest)
            for slot, lane in enumerate(scope['lanes']):
                if not lane:
                    continue
                directory = output / 'workers' / str(slot)
                directory.mkdir(parents=True, exist_ok=False)
                fd = reservations[uuids[slot]]
                info = os.fstat(fd)
                request = dict(runtime_reference=runtime_reference, admission_reference=admission_reference,
                    execution_root=relative, scope_sha256=scope['scope_sha256'], worker_slot=slot,
                    physical_device=scope['devices'][slot], group_ids=lane, coordinator_pid=os.getpid(),
                    coordinator_start_ticks=process_start_ticks(os.getpid()), reservation_fd=fd,
                    reservation_identity=[info.st_dev, info.st_ino], gpu_uuid=uuids[slot],
                    run_binding_sha256=digest)
                path = directory / 'request.json'
                atomic_json(path, request)
                reference = dict(file=str(path), sha256=file_digest(path))
                read.json(path, reference['sha256'])
                stdout = (directory / 'stdout.log').open('x', encoding='utf-8')
                handles.append(stdout)
                stderr = (directory / 'stderr.log').open('x', encoding='utf-8')
                handles.append(stderr)
                process = subprocess.Popen([sys.executable, '-u', ENTRYPOINT,
                    '--worker-request', str(path), '--worker-request-sha256', reference['sha256']],
                    cwd=registry.root, env=worker_environment(uuids[slot]), pass_fds=(fd,),
                    stdout=stdout, stderr=stderr)
                processes.append(process)
                entries.append(dict(slot=slot, pid=process.pid, group_ids=lane, request_reference=reference))
            atomic_json(output / 'execution_assignment.json', dict(workers=entries, lanes=scope['lanes']))
            while True:
                exits = [p.poll() for p in processes]
                atomic_json(output / 'execution_state.json', dict(status='running', workers=entries,
                    exit_codes=exits, updated_at=datetime.now(timezone.utc).isoformat()))
                require(all(code in (None, 0) for code in exits), 'Worker failure; preserve every lane')
                if all(code == 0 for code in exits):
                    break
                time.sleep(1)
            groups = []
            for entry in entries:
                directory = output / 'workers' / str(entry['slot'])
                require(not (directory / 'failure.json').exists(), 'Worker has failure evidence')
                completion = read.json(directory / 'completion.json', file_digest(directory / 'completion.json'))
                require(completion['status'] == 'finished_pending_independent_acceptance'
                    and completion['request_reference'] == entry['request_reference']
                    and completion['worker_slot'] == entry['slot'] and completion['failed_records'] == 0
                    and completion['group_ids'] == entry['group_ids']
                    and completion['completed_groups'] == len(entry['group_ids'])
                    and completion['completed_evaluations'] == 16 * len(entry['group_ids'])
                    and completion['independent_result_acceptance'] is False
                    and completion['scientific_acceptance'] is False
                    and [g['group_id'] for g in completion['groups']] == entry['group_ids'],
                    'Incomplete or mismatched worker completion')
                groups.extend(completion['groups'])
            require(sorted(g['group_id'] for g in groups) == sorted(scope['group_ids']),
                'Completed group set differs from admitted scope')
            read.unchanged()
            require(canonical_hash(inspect_execution(registry, runtime_reference, admission_reference,
                devices, max_images)) == canonical_hash(scope), 'Admission changed before completion')
            result = dict(status='finished_pending_independent_acceptance', execution_root=relative,
                scope_sha256=scope['scope_sha256'], group_ids=scope['group_ids'],
                worker_count=len(entries), requested_device_count=4,
                diagnostic_only=scope['diagnostic_only'], independent_result_acceptance=False,
                scientific_acceptance=False, payload_policy=scope['payload_policy'])
            atomic_json(output / 'completion.json', result)
            atomic_json(output / 'execution_state.json', result)
            return result
        except BaseException as exc:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            failure = dict(status='failed', error_type=type(exc).__name__,
                workers=entries, exit_codes=[p.returncode for p in processes],
                finished_at=datetime.now(timezone.utc).isoformat(),
                partial_evidence_preserved=True, scientific_acceptance=False)
            atomic_json(output / 'failure.json', failure)
            atomic_json(output / 'execution_state.json', failure)
            raise
        finally:
            for handle in handles:
                handle.close()
