"""Installation-path-independent checks for the pinned CPU result consumer."""
import hashlib
import importlib.util
from importlib import metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement


CORE_AND_EVALUATOR = frozenset((
    'torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmyolo',
    'numpy', 'scipy', 'pycocotools', 'Pillow',
))
APPROVED_LOCK_SHA256 = 'a8d35a3d692129e7900df30bb99e4b40d0a2f783fffe87170c9e47282504e2cf'


def _module_origin(name):
    module = sys.modules.get(name)
    if module is not None:
        return getattr(module, '__file__', None)
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin


def _loaded_modules():
    return tuple(sys.modules.items())


def verify_result_runtime(root):
    if (sys.platform != 'linux' or sys.version_info[:3] != (3, 8, 20)
            or Path(sys.prefix).name != 'oda'):
        raise ValueError('Use Linux with the pinned Python 3.8.20 oda environment')
    lock = Path(root) / 'requirements' / 'locked-cu118.txt'
    raw = lock.read_bytes()
    lock_sha256 = hashlib.sha256(raw).hexdigest()
    if lock_sha256 != APPROVED_LOCK_SHA256:
        raise ValueError('Requirements lock differs from the approved runtime')
    expected = {}
    for line in raw.decode('utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '--')):
            continue
        requirement = Requirement(line)
        if requirement.name not in CORE_AND_EVALUATOR:
            continue
        specs = list(requirement.specifier)
        if (requirement.name in expected or len(specs) != 1
                or specs[0].operator != '==' or requirement.marker is not None
                or requirement.url is not None or requirement.extras):
            raise ValueError('Ambiguous core/evaluator package lock')
        expected[requirement.name] = specs[0].version
    if set(expected) != CORE_AND_EVALUATOR:
        raise ValueError('Incomplete core/evaluator package lock')
    actual = {name: metadata.version(name) for name in sorted(expected)}
    if actual != expected:
        raise ValueError('Installed core/evaluator versions differ from the lock')
    origins = {}
    prefix = Path(sys.prefix).resolve()
    for name in sorted(expected):
        module_name = 'PIL' if name == 'Pillow' else name
        distribution = metadata.distribution(name)
        package = Path(distribution.locate_file(module_name)).resolve()
        origin = _module_origin(module_name)
        if origin is None:
            raise ValueError('Missing import origin: ' + module_name)
        origin = Path(origin).resolve()
        if prefix not in package.parents or package not in origin.parents:
            raise ValueError('Module origin differs from pinned installation: ' + module_name)
        for loaded_name, module in _loaded_modules():
            if loaded_name.startswith(module_name + '.'):
                loaded_origin = getattr(module, '__file__', None)
                if loaded_origin and package not in Path(loaded_origin).resolve().parents:
                    raise ValueError('Shadowed loaded submodule: ' + loaded_name)
        origins[module_name] = str(origin)
    return {'python': '3.8.20', 'packages': actual,
            'requirements_sha256': lock_sha256, 'module_origins': origins,
            'execution_admission': False, 'scientific_acceptance': False}
