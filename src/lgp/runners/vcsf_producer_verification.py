"""Verify frozen producer plans in an isolated source-only subprocess."""
import json
import os
from pathlib import Path
import subprocess
import sys

from .vcsf_efficacy_contract import require


def no_duplicates(items):
    result = {}
    for key, value in items:
        require(key not in result, 'Duplicate producer receipt key')
        result[key] = value
    return result


def no_constant(value):
    raise RuntimeError('Non-finite producer receipt value: ' + value)


PRODUCER_PROBE = r'''
import hashlib, importlib.abc, importlib.machinery, importlib.util, json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve(strict=True)
sys.path[:0] = [str(root / 'src'), str(root / 'tools')]
raw_plan = Path(sys.argv[2]).read_bytes()
assert hashlib.sha256(raw_plan).hexdigest() == sys.argv[3], 'Producer plan bytes changed'
source_hashes = json.loads(raw_plan)['runtime_sha256']
class SourceOnly(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        path = Path(self.path)
        assert not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink producer source'
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == source_hashes.get(relative), 'Unbound producer source: ' + fullname
        return compile(raw, str(path), 'exec', dont_inherit=True)
class ProducerSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'lgp' and not fullname.startswith('lgp.'):
            return None
        base = root.joinpath('src', *fullname.split('.'))
        package = (base / '__init__.py').is_file()
        source = base / '__init__.py' if package else base.with_suffix('.py')
        return importlib.util.spec_from_file_location(fullname, str(source), loader=SourceOnly(fullname, str(source)),
            submodule_search_locations=[str(base)] if package else None)
sys.meta_path.insert(0, ProducerSources())
from lgp.registry import Registry
from lgp.io import file_digest
from lgp.runners import vcsf_scale_execution_contract as contract
plan = contract.read_bound(sys.argv[2], sys.argv[3])
registry = Registry(root)
contract.verify_execution_plan(registry, plan)
contract.verify_admission(registry, plan, sys.argv[3], Path(sys.argv[4]), sys.argv[5], json.loads(sys.argv[6]))
modules = {}
for name, module in list(sys.modules.items()):
    if name == 'lgp' or name.startswith('lgp.'):
        path = Path(module.__file__).resolve(strict=True)
        relative = path.relative_to(root).as_posix()
        digest = file_digest(path)
        assert plan['runtime_sha256'].get(relative) == digest, 'Unbound producer import: ' + name
        modules[name] = dict(file=relative, sha256=digest)
print('LGP_PRODUCER_CONTRACT=' + json.dumps(dict(status='verified_original_producer_contract',
    execution_plan_sha256=sys.argv[3], admission_sha256=sys.argv[5], modules=modules, new_model_calls=0)))
'''


def verify_producer_contract(producer, plan_path, plan_hash, admission_path, admission_hash, max_images):
    command = [sys.executable, '-I', '-B', '-c', PRODUCER_PROBE, str(producer), str(plan_path), plan_hash,
        str(admission_path), admission_hash, json.dumps(max_images)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run(command, cwd=producer, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    require(result.returncode == 0, 'Original producer contract rejected its evidence: ' + result.stderr[-2000:])
    markers = [line.split('=', 1)[1] for line in result.stdout.splitlines() if line.startswith('LGP_PRODUCER_CONTRACT=')]
    require(len(markers) == 1, 'Producer validation did not return one scoped result')
    receipt = json.loads(markers[0], object_pairs_hook=no_duplicates, parse_constant=no_constant)
    require(receipt.get('status') == 'verified_original_producer_contract'
        and receipt.get('execution_plan_sha256') == plan_hash and receipt.get('admission_sha256') == admission_hash
        and receipt.get('new_model_calls') == 0 and receipt.get('modules'), 'Producer contract receipt changed scope')
    return receipt


