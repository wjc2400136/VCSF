"""Versioned source-package closure guard, separate from the preserved author freeze."""
from __future__ import annotations

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path, PurePosixPath
import re

import yaml

MANIFEST = "public-package.json"
FREEZE = "docs/research/vcsf-a10-public-package-freeze-v1.json"
SCHEMA = "lgp_public_source_closure_v1"
DISTRIBUTION_SCHEMA = "lgp_public_source_distribution_v2"
GPL_TEXT_SHA256 = "3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986"
UPSTREAM_LICENSE_SHA256 = {'docs/third_party/licenses/NumbOD-MIT.txt': 'bc14dbcc12b4c43ad1e382e8e315450784332208ddf75069834bdc18484a4277', 'docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt': '874b8b2e6f12306a1ff7812c068a0dbf880418b8ac1f7190382bd6c6da9f23a1', 'docs/third_party/licenses/GPL-3.0-upstream.txt': '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986'}
BASE_FREEZE = "docs/research/vcsf-a10-public-source-freeze.json"
BASE_FREEZE_SHA256 = "835ecf45606670c9f92e34caf7a8c42e1e5b89119c0adc273ec1afff445fbb2f"
PROVENANCE = (
    "docs/research/vcsf-a10-configuration-recommendation-provenance.md",
    "docs/research/vcsf-a10-main-audit-contract.json",
    "docs/research/vcsf-a10-main-producer-schema.json",
    BASE_FREEZE,
    "docs/research/vcsf-a10-recorded-configuration-choice.json",
    "docs/research/vcsf-a10-selection-protocol.json",
)
HISTORICAL_MISSING_AUTHORITY = "docs/research/vcsf-final-only-coverage-amendment-20260920.md"
HISTORICAL_MISSING_PROTOCOLS = ("vcsf_final_budget_refresh", "vcsf_final_training_state_refresh")
SHIPPED_TREES = ("src", "tools", "experiments", "configs", "docs", "requirements", "tests")
RUNTIME_TREES = ("data", "checkpoints", "outputs")
EXECUTABLE_SUFFIXES = (".py", ".yaml", ".yml", ".sh", ".exe", ".so", ".dll")
EXTERNAL_ROOTS = frozenset((
    "__future__", "argparse", "ast", "atexit", "codecs", "numbers", "queue", "stat", "struct",
    "collections", "concurrent", "contextlib",
    "copy", "csv", "ctypes", "dataclasses", "datetime", "decimal", "enum", "fcntl",
    "fractions", "functools", "gc", "glob", "gzip", "hashlib", "html", "http",
    "importlib", "inspect", "io", "itertools", "json", "logging", "math", "multiprocessing",
    "operator", "os", "pathlib", "pickle", "platform", "random", "re", "resource",
    "shlex", "shutil", "signal", "socket", "statistics", "subprocess", "sys", "tarfile",
    "tempfile", "threading", "time", "traceback", "types", "typing", "urllib", "uuid",
    "warnings", "weakref", "xml", "zipfile", "zlib", "torch", "torchvision", "numpy",
    "pandas", "PIL", "requests", "rich", "tqdm", "matplotlib", "scipy", "skimage",
    "jsonschema", "yaml", "filelock", "packaging", "psutil", "cv2", "lpips", "pytest",
    "mmcv", "mmengine", "mmdet", "mmyolo", "setuptools", "urllib3", "pycocotools",
))
MANDATORY_ENTRIES = (
    "experiments/all_method_ablations.py", "experiments/coco_all_methods.py",
    "experiments/extended_transfer.py", "experiments/naa_experiments.py",
    "experiments/visualize_predictions.py", "experiments/voc_transfer.py",
    "experiments/main_transfer.py", "experiments/coco_training_state_pair.py",
)
DYNAMIC_FILES = {
    "src/lgp/runners/finetune.py": ("tools/run_finetune_dual_queue.py", "tools/run_finetune_queue.py"),
    "tools/run_finetune_dual_queue.py": ("tools/run_finetune_queue.py",),
    "src/lgp/runners/prediction_visualization.py": ("src/lgp/runners/formal_parallel.py",),
    "src/lgp/runtime_config.py": ("src/lgp/training_hooks.py",),
}


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def plain_file(root, relative):
    require(type(relative) is str and relative and chr(92) not in relative,
            "Package path must be a POSIX relative path")
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and str(rel) == relative
            and not any(part in ("", ".", "..") for part in rel.parts),
            "Package path traversal or absolute path")
    path = root / relative
    require(root in path.parents and not any(p.is_symlink() for p in (path, *path.parents))
            and path.is_file() and path.resolve() == path, "Package requires a plain file: " + relative)
    require(path.stat().st_size <= 2 * 1024 * 1024, "Source file exceeds lightweight cap")
    return path


def read_json(root, relative):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate package JSON key")
            value[key] = item
        return value
    value = json.loads(plain_file(root, relative).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs)
    require(type(value) is dict, "Package JSON root must be an object")
    return value


def shipped_files(root):
    found = set()
    for directory in SHIPPED_TREES:
        parent = root / directory
        require(not parent.is_symlink(), "Shipped tree cannot be a link")
        if not parent.exists():
            continue
        for path in parent.rglob("*"):
            require(not path.is_symlink(), "Shipped tree contains a link")
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                require(not path.is_symlink(), "Bytecode path cannot be a link")
                continue
            if path.is_file():
                found.add(path.relative_to(root).as_posix())
    for path in root.iterdir():
        require(not path.is_symlink(), "Package root contains a link")
        if path.name == ".git":
            require(path.is_file() or path.is_dir(), "Git metadata must be a plain file or directory")
            continue
        if path.is_file():
            found.add(path.name)
        elif path.name not in SHIPPED_TREES + RUNTIME_TREES + ("__pycache__", ".pytest_cache"):
            raise RuntimeError("Unbound top-level directory: " + path.name)
    for directory in RUNTIME_TREES:
        parent = root / directory
        if not parent.exists():
            continue
        require(not parent.is_symlink(), "Package runtime tree cannot be a link")
        # Runtime payloads are outside shipped source scope; never enumerate them here.
        for name in ("README.md", "README.zh-CN.md"):
            if (parent / name).exists():
                found.add(directory + "/" + name)
    return found - {MANIFEST, FREEZE}


def module_path(name, owner, files):
    if name.startswith("lgp."):
        prefix = "src/" + name.replace(".", "/")
        candidates = (prefix + ".py", prefix + "/__init__.py")
        for rel in candidates:
            if rel in files:
                return rel
        raise RuntimeError("Omitted local import dependency: " + name + " in " + owner)
    if name == "lgp":
        require("src/lgp/__init__.py" in files, "Omitted lgp package")
        return "src/lgp/__init__.py"
    if name.startswith(("tools.", "experiments.")):
        prefix = name.replace(".", "/")
        for rel in (prefix + ".py", prefix + "/__init__.py"):
            if rel in files:
                return rel
        raise RuntimeError("Omitted local helper dependency: " + name)
    if "." not in name:
        for rel in ((Path(owner).parent / (name + ".py")).as_posix(),
                    "tools/" + name + ".py", "experiments/" + name + ".py"):
            if rel in files:
                return rel
    require(name.split(".")[0] in EXTERNAL_ROOTS, "Unresolved import dependency: " + name)
    return None


@lru_cache(maxsize=512)
def parsed_python(owner, raw):
    return ast.parse(raw, filename=owner)


@lru_cache(maxsize=64)
def parsed_yaml(raw):
    return yaml.safe_load(raw)


def verify_imports(root, files, runtime_dependencies):
    require(type(runtime_dependencies) is dict, "Missing runtime dependency map")
    for owner, dependencies in runtime_dependencies.items():
        require(owner in files and type(dependencies) is list
                and len(set(dependencies)) == len(dependencies), "Invalid runtime dependency owner")
        require(all(d in files for d in dependencies), "Omitted runtime/AST helper dependency: " + owner)
    for owner, dependencies in DYNAMIC_FILES.items():
        if owner in files:
            require(set(dependencies) <= set(runtime_dependencies.get(owner, ())),
                    "Unbound dynamic loader dependency: " + owner)
    def configuration_references(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from configuration_references(item)
        elif isinstance(value, list):
            for item in value:
                yield from configuration_references(item)
        elif isinstance(value, str) and value.startswith(("docs/", "src/")):
            if value.endswith((".py", ".json", ".md")):
                yield value
    for owner in sorted(files):
        if owner.startswith("configs/"):
            declared = set(configuration_references(parsed_yaml(
                plain_file(root, owner).read_text(encoding="utf-8"))))
            if owner == "configs/experiments/protocols.yaml":
                definitions = parsed_yaml(plain_file(root, owner).read_text(encoding="utf-8"))["protocols"]
                require(all(definitions[p].get("authority") == HISTORICAL_MISSING_AUTHORITY
                            for p in HISTORICAL_MISSING_PROTOCOLS),
                        "Historical missing authority reference drifted")
                declared.discard(HISTORICAL_MISSING_AUTHORITY)
            require(declared <= set(runtime_dependencies.get(owner, ())) and declared <= set(files),
                    "Omitted configuration helper/provenance dependency: " + owner)
        if not owner.endswith(".py"):
            continue
        raw = plain_file(root, owner).read_text(encoding="utf-8")
        tree = parsed_python(owner, raw)
        parts = list(Path(owner).with_suffix("").parts)
        if parts[0] == "src":
            parts.pop(0)
        parts.pop()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_path(alias.name, owner, files)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = ".".join(parts[:len(parts) - node.level + 1] + ([name] if name else []))
                if name in ("tools", "experiments"):
                    for alias in node.names:
                        module_path(name + "." + alias.name, owner, files)
                elif name and node.module:
                    module_path(name, owner, files)
                else:
                    for alias in node.names:
                        module_path(name + "." + alias.name, owner, files)
            elif isinstance(node, ast.Call):
                func = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if func in ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant):
                    module_path(node.args[0].value, owner, files)
        if owner.startswith("src/"):
            parent = Path(owner).parent
            while parent.as_posix() != "src":
                require((parent / "__init__.py").as_posix() in files, "Omitted package initializer")
                parent = parent.parent


def verify_package(root):
    root = Path(root).absolute()
    require(root.resolve() == root and not any(p.is_symlink() for p in (root, *root.parents)),
            "Package root must be plain")
    manifest = read_json(root, MANIFEST)
    expected_keys = {"schema", "candidate_only", "science", "formal", "release", "rights_status",
                     "source_file_sha256", "active_entrypoints", "runtime_file_dependencies",
                     "numeric_source_sha256", "configuration_sha256", "provenance_sha256",
                     "base_freeze_sha256", "whole_old_freeze_equivalence_claimed",
                     "historical_missing_metadata"}
    distribution = manifest.get("schema") == DISTRIBUTION_SCHEMA
    if distribution:
        expected_keys.add("licensing")
        require(manifest.get("candidate_only") is False and manifest.get("release") is True
                and manifest.get("rights_status") == "recorded_source_distribution_conditions_addressed"
                and manifest.get("licensing") == {
                    "project_license": "GPL-3.0-only", "grant_scope": "project_owned_source",
                    "license_file": "LICENSE", "license_text_sha256": GPL_TEXT_SHA256,
                    "third_party_terms_retained": True, "unconditional_legal_clearance_claimed": False,
                    "media_datasets_checkpoints_covered": False,
                }, "Source distribution licence boundary drifted")
    else:
        require(manifest.get("schema") == SCHEMA and manifest.get("candidate_only") is True
                and manifest.get("release") is False
                and manifest.get("rights_status") == "pending_not_cleared",
                "Historical candidate boundary drifted")
    require(set(manifest) == expected_keys and manifest["science"] is False
            and manifest["formal"] is False
            and manifest["whole_old_freeze_equivalence_claimed"] is False
            and manifest["base_freeze_sha256"] == BASE_FREEZE_SHA256,
            "Package schema or acceptance boundary drifted")
    require(manifest["historical_missing_metadata"] == [{
        "path": HISTORICAL_MISSING_AUTHORITY, "protocols": list(HISTORICAL_MISSING_PROTOCOLS),
        "status": "absent_in_original_not_current_runtime_input_not_repaired"}],
        "Historical missing metadata boundary changed")
    files = manifest["source_file_sha256"]
    require(type(files) is dict and bool(files), "Empty package manifest")
    require(set(files) == shipped_files(root), "Missing or extra unbound shipped file")
    for relative, expected in files.items():
        require(type(expected) is str and re.fullmatch("[0-9a-f]{64}", expected),
                "Invalid source digest")
        require(digest(plain_file(root, relative)) == expected, "Package source tampered: " + relative)
    if distribution:
        required = {"LICENSE", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.zh-CN.md",
                    "docs/licensing/license-scope.md", "docs/licensing/license-scope.zh-CN.md",
                    "docs/licensing/source-distribution-decision.json"}
        require(required <= set(files), "Missing distribution licence or notice document")
        require(files["LICENSE"] == GPL_TEXT_SHA256, "Project GPL text changed")
        require(all(files.get(p) == h for p, h in UPSTREAM_LICENSE_SHA256.items()),
                "Retained upstream licence text changed")
        decision = read_json(root, "docs/licensing/source-distribution-decision.json")
        required_decision = {
            "schema": "vcsf_source_licence_selection_v1",
            "project_license": "GPL-3.0-only",
            "selection_authority": "project_owner_delegated_common_and_appropriate_licence_choice",
            "grant_scope": "project_owned_source_and_accompanying_project_owned_source_guides",
            "integrated_source_distribution_license": "GPL-3.0-only",
            "retained_third_party_terms": ["MIT", "Apache-2.0", "GNU GPL version 3 as supplied upstream"],
            "GPL_original_text_sha256": GPL_TEXT_SHA256,
            "retained_upstream_license_sha256": UPSTREAM_LICENSE_SHA256,
            "upstream_later_version_option_inferred": False,
            "all_source_expressive_origin_established": False,
            "unconditional_legal_clearance_claimed": False,
            "new_method_or_numerical_acceptance": False,
            "manuscript_media_datasets_and_checkpoints_included": False,
        }
        require(all(type(decision.get(k)) is type(v) and decision[k] == v
                    for k, v in required_decision.items()),
                "Distribution licence decision drifted")
    require(digest(plain_file(root, BASE_FREEZE)) == BASE_FREEZE_SHA256,
            "Original source freeze was changed")
    original = read_json(root, BASE_FREEZE)["source_file_sha256"]
    protocol = read_json(root, PROVENANCE[-1])
    numeric = protocol["execution_identity"]["numeric_source_sha256"]
    require(len(numeric) == 9 and manifest["numeric_source_sha256"] == numeric
            and all(files.get(p) == h for p, h in numeric.items()), "Closed numerical core changed")
    provenance = {p: BASE_FREEZE_SHA256 if p == BASE_FREEZE else original[p] for p in PROVENANCE}
    require(manifest["provenance_sha256"] == provenance
            and all(files.get(p) == h for p, h in provenance.items()), "Original provenance bytes changed")
    configuration = {p: h for p, h in original.items() if p.startswith("configs/")}
    require(manifest["configuration_sha256"] == configuration
            and {p: h for p, h in files.items() if p.startswith("configs/")} == configuration,
            "Original configuration meanings or bytes changed")
    entries = manifest["active_entrypoints"]
    require(type(entries) is list and len(entries) == len(set(entries))
            and set(MANDATORY_ENTRIES) <= set(entries) and all(e in files for e in entries),
            "Maintained entry set was reduced")
    verify_imports(root, files, manifest["runtime_file_dependencies"])
    freeze = read_json(root, FREEZE)
    expected = dict(schema=manifest["schema"], source_manifest_sha256=canonical(files),
                    manifest_file_sha256=digest(root / MANIFEST),
                    base_freeze_sha256=BASE_FREEZE_SHA256,
                    numerical_source_map_sha256=canonical(numeric),
                    whole_old_freeze_equivalence_claimed=False,
                    candidate_only=manifest["candidate_only"], rights_status=manifest["rights_status"],
                    science=False, formal=False, release=manifest["release"])
    if distribution:
        expected["licensing_sha256"] = canonical(manifest["licensing"])
    require(freeze == expected, "New package freeze changed or mismatches complete closure")
    return files
