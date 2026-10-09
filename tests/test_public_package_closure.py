"""Focused package failures and byte/AST preservation, not efficacy tests."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from lgp.public_package import MANIFEST, FREEZE, BASE_FREEZE, canonical, verify_package
from lgp.attacks import vcsf_public
from lgp.registry import Registry

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def package(tmp_path):
    target = tmp_path / "source"
    manifest = json.loads((ROOT / MANIFEST).read_text())
    for relative in list(manifest["source_file_sha256"]) + [MANIFEST, FREEZE]:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    verify_package(target)
    return target

def repin(root):
    manifest = json.loads((root / MANIFEST).read_text())
    files = manifest["source_file_sha256"]
    for p in list(files):
        if not (root / p).exists():
            del files[p]
        else:
            files[p] = hashlib.sha256((root / p).read_bytes()).hexdigest()
    (root / MANIFEST).write_text(json.dumps(manifest, sort_keys=True, indent=2) + chr(10))
    freeze = json.loads((root / FREEZE).read_text())
    freeze["source_manifest_sha256"] = canonical(files)
    freeze["manifest_file_sha256"] = hashlib.sha256((root / MANIFEST).read_bytes()).hexdigest()
    (root / FREEZE).write_text(json.dumps(freeze, sort_keys=True, indent=2) + chr(10))

def test_complete_current_package_and_original_provenance():
    registry = Registry(ROOT)
    identity = vcsf_public.verify_public_identity(ROOT, registry.attack("vcsf"))
    assert len(identity["numeric_source_sha256"]) == 9
    assert identity["source_identity_schema"] == json.loads((ROOT / MANIFEST).read_text())["schema"]
    assert identity["parameters_sha256"] == vcsf_public.PARAMETERS_SHA256
    assert identity["original_source_freeze_sha256"] == hashlib.sha256((ROOT / BASE_FREEZE).read_bytes()).hexdigest()
    assert not identity["old_full_freeze_equivalence_claimed"]
    assert all(identity[k] is False for k in ("science", "formal", "release"))
    from lgp.attacks.factory import build_attack
    attack = build_attack("vcsf", object(), registry.attack("vcsf").parameters)
    assert asdict(attack.config) == identity["parameters"]
    assert [c.__module__ + "." + c.__name__ for c in type(attack).__mro__] == identity["actual_class_mro"]
    import torch
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("relative", [
    "src/lgp/runners/public_observation.py", "experiments/main_transfer.py",
    "configs/attacks/vcsf.yaml", "docs/research/vcsf-a10-selection-protocol.json",
])
def test_missing_bound_file(package, relative):
    original = package / relative
    original.rename(package.parent / (Path(relative).name + ".missing"))
    with pytest.raises(RuntimeError):
        verify_package(package)

@pytest.mark.parametrize("relative", [
    "src/lgp/runners/public_config_helpers.py", "experiments/visualize_predictions.py",
])
def test_tampered_bound_file(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\n# changed source\n")
    with pytest.raises(RuntimeError, match="tampered"):
        verify_package(package)

def test_nonplain_file(package):
    relative = "src/lgp/runners/public_observation.py"
    path = package / relative
    saved = package.parent / "observer.saved"
    path.rename(saved)
    path.symlink_to(saved)
    with pytest.raises(RuntimeError, match="link|plain"):
        verify_package(package)

def test_extra_unbound_executable(package):
    (package / "tools/extra_unbound.py").write_text("raise RuntimeError('unbound')\n")
    with pytest.raises(RuntimeError, match="unbound"):
        verify_package(package)

@pytest.mark.parametrize("kind", ["directory", "worktree_pointer"])
def test_git_metadata_is_excluded_without_reading_or_hashing(package, kind):
    git = package / ".git"
    if kind == "directory":
        git.mkdir()
        (git / "unbound_metadata.py").write_text("not a shipped project source")
        (git / "objects").symlink_to(package.parent / "absent_objects")
    else:
        git.write_text("gitdir: ../external-worktree-metadata\n")
    expected = json.loads((package / MANIFEST).read_text())["source_file_sha256"]
    assert verify_package(package) == expected
    assert ".git" not in expected

def test_git_metadata_symlink_is_not_a_plain_pointer(package):
    (package / ".git").symlink_to(package.parent / "external-git-metadata")
    with pytest.raises(RuntimeError, match="link"):
        verify_package(package)

def test_omitted_active_import_even_after_control_documents_repin(package):
    path = package / "src/lgp/runners/public_observation.py"
    path.rename(package.parent / "omitted-helper.saved")
    manifest = json.loads((package / MANIFEST).read_text())
    for owner, deps in manifest["runtime_file_dependencies"].items():
        if "src/lgp/runners/public_observation.py" in deps:
            deps.remove("src/lgp/runners/public_observation.py")
    manifest["runtime_file_dependencies"].pop("src/lgp/runners/public_observation.py", None)
    (package / MANIFEST).write_text(json.dumps(manifest))
    repin(package)
    with pytest.raises(RuntimeError, match="Omitted local import"):
        verify_package(package)

@pytest.mark.parametrize("relative", [
    "src/lgp/attacks/vcsf_common_anchor_execution_isolation.py",
    "src/lgp/attacks/vcsf_common_anchor_isolated.py",
])
def test_closed_numeric_core_cannot_be_redefined_by_new_package_metadata(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\n# numerical drift\n")
    repin(package)
    with pytest.raises(RuntimeError, match="numerical"):
        verify_package(package)

def test_original_config_cannot_be_redefined_by_new_manifest(package):
    path = package / "configs/attacks/vcsf.yaml"
    path.write_bytes(path.read_bytes() + b"\n# config drift\n")
    repin(package)
    with pytest.raises(RuntimeError, match="configuration"):
        verify_package(package)

def test_path_traversal_rejected(package):
    manifest = json.loads((package / MANIFEST).read_text())
    digest = next(iter(manifest["source_file_sha256"].values()))
    manifest["source_file_sha256"]["../escaped.py"] = digest
    (package / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError):
        verify_package(package)


def test_licensed_source_boundary_is_separate_from_scientific_acceptance():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["schema"] == "lgp_public_source_distribution_v2"
    assert manifest["release"] is True and manifest["candidate_only"] is False
    assert manifest["science"] is False and manifest["formal"] is False
    assert manifest["licensing"]["project_license"] == "GPL-3.0-only"
    assert manifest["licensing"]["media_datasets_checkpoints_covered"] is False
    assert manifest["source_file_sha256"]["LICENSE"] == manifest["licensing"]["license_text_sha256"]
    verify_package(ROOT)


def test_v1_candidate_remains_candidate_when_read_by_current_guard(package):
    manifest = json.loads((package / MANIFEST).read_text(encoding="utf-8"))
    manifest.update(schema="lgp_public_source_closure_v1", candidate_only=True,
                    release=False, rights_status="pending_not_cleared")
    del manifest["licensing"]
    (package / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    freeze = json.loads((package / FREEZE).read_text(encoding="utf-8"))
    freeze.update(schema=manifest["schema"], candidate_only=True,
                  release=False, rights_status="pending_not_cleared")
    del freeze["licensing_sha256"]
    (package / FREEZE).write_text(json.dumps(freeze), encoding="utf-8")
    repin(package)
    assert verify_package(package) == manifest["source_file_sha256"]
    identity = vcsf_public.verify_public_identity(package)
    assert identity["source_identity_schema"] == manifest["schema"]
    assert all(identity[k] is False for k in ("science", "formal", "release"))


@pytest.mark.parametrize("field,value", [
    ("project_license", "MIT"), ("grant_scope", "all_third_party_material"),
    ("third_party_terms_retained", False), ("unconditional_legal_clearance_claimed", True),
    ("media_datasets_checkpoints_covered", True), ("project_license", "GPL-3.0-or-later"),
])
def test_source_licence_boundary_cannot_be_redefined(package, field, value):
    manifest = json.loads((package / MANIFEST).read_text(encoding="utf-8"))
    manifest["licensing"][field] = value
    (package / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    repin(package)
    with pytest.raises(RuntimeError, match="licence boundary"):
        verify_package(package)


@pytest.mark.parametrize("field,value", [
    ("science", True), ("formal", True), ("candidate_only", True),
    ("release", False), ("rights_status", "unconditional_legal_clearance"),
])
def test_source_publication_cannot_promote_other_acceptance(package, field, value):
    manifest = json.loads((package / MANIFEST).read_text(encoding="utf-8"))
    manifest[field] = value
    (package / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    repin(package)
    with pytest.raises(RuntimeError, match="boundary"):
        verify_package(package)


@pytest.mark.parametrize("relative,match", [
    ("LICENSE", "Project GPL text"),
    ("docs/third_party/licenses/NumbOD-MIT.txt", "upstream licence"),
    ("docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt", "upstream licence"),
    ("docs/third_party/licenses/GPL-3.0-upstream.txt", "upstream licence"),
])
def test_original_licence_text_cannot_be_silently_rebound(package, relative, match):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged licence text\n")
    repin(package)
    with pytest.raises(RuntimeError, match=match):
        verify_package(package)


@pytest.mark.parametrize("relative", [
    "LICENSE", "THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.zh-CN.md",
    "docs/licensing/license-scope.md", "docs/licensing/license-scope.zh-CN.md",
])
def test_distribution_notice_cannot_be_omitted_by_repinning(package, relative):
    (package / relative).unlink()
    repin(package)
    with pytest.raises(RuntimeError, match="Missing distribution"):
        verify_package(package)


def test_licence_decision_cannot_promote_numerical_acceptance(package):
    path = package / "docs/licensing/source-distribution-decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision["new_method_or_numerical_acceptance"] = True
    path.write_text(json.dumps(decision), encoding="utf-8")
    repin(package)
    with pytest.raises(RuntimeError, match="licence decision"):
        verify_package(package)


def test_source_guard_cli_reports_current_distribution_not_old_pending_state():
    import subprocess
    result = subprocess.run([sys.executable, "-B", "experiments/verify_public_package.py", "--plan-only"],
                            cwd=str(ROOT), check=True, capture_output=True, text=True)
    report = json.loads(result.stdout)
    assert report["schema"] == "lgp_public_source_distribution_v2"
    assert report["project_license"] == "GPL-3.0-only"
    assert report["release"] is True and report["candidate_only"] is False
    assert report["rights_status"] == "recorded_source_distribution_conditions_addressed"
    assert all(report[key] == 0 for key in ("model_calls", "CUDA_calls", "inference_calls", "attack_calls", "AP_calls"))


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("grant_scope", "all_material"),
    ("integrated_source_distribution_license", "GPL-3.0-or-later"),
    ("manuscript_media_datasets_and_checkpoints_included", True),
    ("manuscript_media_datasets_and_checkpoints_included", 0),
    ("upstream_later_version_option_inferred", True),
    ("upstream_later_version_option_inferred", 0),
    ("all_source_expressive_origin_established", True),
    ("selection_authority", "not_delegated"),
    ("retained_third_party_terms", ["GPL-3.0-only"]),
    ("unconditional_legal_clearance_claimed", 0),
    ("new_method_or_numerical_acceptance", 0),
    ("grant_scope", None),
])
def test_rebound_licence_decision_preserves_all_recorded_policy_boundaries(package, field, value):
    path = package / "docs/licensing/source-distribution-decision.json"
    decision = json.loads(path.read_text(encoding="utf-8"))
    decision[field] = value
    path.write_text(json.dumps(decision), encoding="utf-8")
    repin(package)
    with pytest.raises(RuntimeError, match="licence decision"):
        verify_package(package)
