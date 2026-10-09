"""Explicit NVML context registration for an exclusively reserved worker GPU."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

from .vcsf_structure_workers import require, verify_reservation


def _identity():
    pid = os.getpid()
    raw = (Path('/proc') / str(pid) / 'stat').read_text()
    return pid, int(raw.rsplit(') ', 1)[1].split()[19])


def _compute_pids(gpu_uuid):
    output = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        text=True, timeout=10)
    result = set()
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        require(len(row) == 2, 'Malformed NVML compute-process row')
        if row[0].strip() == gpu_uuid:
            value = row[1].strip()
            require(value.isdigit() and int(value) > 0, 'Invalid NVML compute-process PID')
            result.add(int(value))
    return result


def _create_context():
    import torch

    require(not torch.cuda.is_initialized(), 'Worker CUDA was initialized before ownership registration')
    require(torch.cuda.device_count() == 1, 'Ownership registration requires exactly one visible GPU')
    # An empty allocation creates a context without consuming the attack RNG stream.
    token = torch.empty(1, device='cuda:0')
    torch.cuda.synchronize(0)
    return token


class RegisteredGpuOwner:
    """Pin the idle-to-owned transition; never relearn a PID after registration.

    This relies on NVML enumerating every compute context on the selected GPU.
    It is a context-registration protocol, not a host/container PID mapping API.
    """

    def __init__(self, request):
        self._request = dict(request)
        self._uuid = request['gpu_uuid']
        self._identity = _identity()
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == self._uuid,
            'Worker GPU visibility changed before registration')
        verify_reservation(self._request)
        require(not _compute_pids(self._uuid), 'GPU is not idle before context registration')
        self._token = _create_context()
        observed = _compute_pids(self._uuid)
        require(len(observed) == 1, 'Ambiguous or missing own NVML context after registration')
        self._nvml_pid = next(iter(observed))
        self.verify()
        self.receipt = {
            'schema': 'vcsf_gpu_context_registration_v1',
            'gpu_uuid': self._uuid,
            'worker_pid': self._identity[0],
            'worker_pid_start_ticks': self._identity[1],
            'nvml_pid': self._nvml_pid,
            'before_context_pids': [],
            'registered_context_pids': sorted(observed),
            'assumption': 'NVML_enumerates_all_active_compute_context_processes_on_selected_device',
            'registered_at': datetime.now(timezone.utc).isoformat(),
        }

    def verify(self):
        require(_identity() == self._identity, 'Registered worker process identity changed')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == self._uuid,
            'Registered worker GPU visibility changed')
        verify_reservation(self._request)
        observed = _compute_pids(self._uuid)
        require(observed == {self._nvml_pid},
            'Registered GPU context disappeared, changed, or has foreign compute processes: {}'.format(
                sorted(observed)))
