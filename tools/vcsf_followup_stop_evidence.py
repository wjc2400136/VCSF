"""Validate supplied shutdown observations; never signal or authenticate them."""
from lgp.runners.vcsf_research_plan import canonical_hash, require


def validate_stop_observations(expected_processes, observations):
    require(type(expected_processes) is list and len(expected_processes) >= 3,
        'Require supervisor, coordinator and worker identities')
    expected = {}
    for row in expected_processes:
        require(type(row) is dict and set(row) == {'pid', 'start_ticks'},
            'Unexpected expected-process schema')
        require(all(type(row[k]) is int and row[k] > 0 for k in row),
            'Invalid expected process identity')
        require(row['pid'] not in expected, 'Duplicate expected PID')
        expected[row['pid']] = row['start_ticks']
    require(type(observations) is list and len(observations) == 2,
        'Require two separately captured shutdown observations')
    previous = None
    for observation in observations:
        require(type(observation) is dict and set(observation) == {
            'monotonic_ns', 'boot_id', 'processes', 'registered_gpu_contexts_absent',
            'device_reservations_released'}, 'Unexpected observation schema')
        stamp = observation['monotonic_ns']
        require(type(stamp) is int and stamp > 0 and (previous is None or stamp > previous),
            'Observations must have increasing monotonic times')
        previous = stamp
        require(type(observation['boot_id']) is str and observation['boot_id']
            and observation['boot_id'] == observations[0]['boot_id'],
            'Observations must be from the same host boot')
        require(observation['registered_gpu_contexts_absent'] is True
            and observation['device_reservations_released'] is True,
            'GPU contexts or reservations are not released')
        rows = observation['processes']
        require(type(rows) is list and len(rows) == len(expected), 'Incomplete process observations')
        seen = set()
        for row in rows:
            require(type(row) is dict and set(row) == {'pid', 'present', 'start_ticks'},
                'Unexpected observed-process schema')
            pid = row['pid']
            require(type(pid) is int and pid in expected and pid not in seen,
                'Unknown or repeated observed PID')
            seen.add(pid)
            require(type(row['present']) is bool, 'Presence must be explicit')
            if row['present']:
                require(type(row['start_ticks']) is int and row['start_ticks'] > 0
                    and row['start_ticks'] != expected[pid], 'Original process still exists')
            else:
                require(row['start_ticks'] is None, 'Absent process cannot have observed start ticks')
    return dict(status='shutdown_observations_consistent_pending_authentication',
        expected_processes_sha256=canonical_hash(expected_processes),
        observations_sha256=canonical_hash(observations),
        external_observations_authenticated=False, successor_admitted=False,
        signals_sent=0)
