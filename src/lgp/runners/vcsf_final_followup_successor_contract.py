"""Pure successor scope reconstruction; no evidence authentication or admission.

Callers must authenticate catalogue, partition bytes and external references,
rejecting duplicate JSON keys before decoding. A reference SHA256 binds file
bytes externally; it is not the partition's canonical content hash. Compilation
does not read either reference or certify stop, qualifications or input assets.
"""
from copy import deepcopy
import re

from tools.vcsf_final_followup_successor_partition import compile_successor_partition

from .vcsf_final_followup_execution_contract import EXECUTION_PROTOCOL, compile_execution_scope
from .vcsf_research_plan import canonical_hash, require


MODE = 'final_followup_successor_execution'


def compile_successor_scope(registry, catalogue, partition, *, partition_reference,
        max_images=None):
    """Reconstruct all remaining whole groups, on the partition's four lanes.

    max_images=1 describes a diagnostic prefix without changing group identity.
    An empty remainder is a valid unarmed plan, not permission to launch workers.
    Original catalogue device assignments and input placeholders stay unchanged.
    """
    require(type(partition_reference) is dict and
        set(partition_reference) == {'file', 'sha256'}, 'Require pinned partition reference')
    name, digest = partition_reference['file'], partition_reference['sha256']
    require(type(name) is str and bool(name.strip()) and name == name.strip()
        and not any(ord(c) < 32 for c in name) and type(digest) is str
        and re.fullmatch(r'[0-9a-f]{64}', digest) is not None,
        'Invalid partition reference')
    require(type(partition) is dict and type(partition.get('completed_groups')) is list,
        'Require complete successor partition')
    references = {}
    for row in partition['completed_groups']:
        require(type(row) is dict and type(row.get('group')) is dict,
            'Malformed completed group binding')
        gid = row['group'].get('group_id')
        require(type(gid) is int and gid not in references and 'qualification_reference' in row,
            'Invalid or duplicate completed group binding')
        references[gid] = row['qualification_reference']
    rebuilt = compile_successor_partition(catalogue, references,
        partition.get('predecessor_stop_reference'), partition.get('devices'))
    require(canonical_hash(rebuilt) == canonical_hash(partition),
        'Partition differs from reconstruction')

    # Validate original registry/method identity without admitting a subset via
    # the old contract. All successor membership and lanes come from partition.
    scope = compile_execution_scope(registry, catalogue,
        group_ids=catalogue['prospective_new_group_ids'], max_images=max_images)
    scope.pop('scope_sha256')
    scope.update(mode=MODE, protocol=EXECUTION_PROTOCOL, status='planned_not_admitted',
        partition=deepcopy(rebuilt), partition_reference=deepcopy(partition_reference),
        partition_sha256=rebuilt['partition_sha256'],
        partition_content_sha256=canonical_hash(rebuilt),
        group_ids=list(rebuilt['successor_group_ids']), groups=deepcopy(rebuilt['successor_groups']),
        devices=list(rebuilt['devices']), lanes=deepcopy(rebuilt['lanes']),
        images=(rebuilt['successor_coverage']['images'] if max_images is None
            else rebuilt['successor_coverage']['groups']),
        target_cells=rebuilt['successor_coverage']['target_cells'],
        catalogue_authenticated_here=False, partition_reference_authenticated=False,
        qualification_references_authenticated=False, predecessor_stop_verified=False,
        input_identity_authenticated=False, device_availability_verified=False,
        runner_armed=False, live_touched=False, scientific_acceptance=False, AP_replays=0)
    scope['scope_sha256'] = canonical_hash(scope)
    return scope


def verify_successor_scope(registry, catalogue, partition, scope, *, partition_reference):
    """Reconstruct against caller-supplied bindings, never trust a scope self-hash.

    This verifies structure only. Even matching references remain unauthenticated.
    """
    require(type(scope) is dict and 'max_images' in scope, 'Require successor scope object')
    expected = compile_successor_scope(registry, catalogue, partition,
        partition_reference=partition_reference, max_images=scope['max_images'])
    require(canonical_hash(expected) == canonical_hash(scope),
        'Successor scope differs from reconstruction')
    return expected
