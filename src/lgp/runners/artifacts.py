from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ..attacks.common import parse_fraction
from ..io import atomic_json, file_digest
from ..paths import project_root
from ..registry import Registry


def import_external_artifact(
    registry: Registry,
    dataset_id: str,
    source_id: str,
    attack_id: str,
    annotation: Path,
    image_root: Path,
    clean_root: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    expected_eps: Optional[float] = None,
    budget_profile: Optional[str] = None,
    gradient_evaluations_per_image: Optional[int] = None,
    attack_seconds_per_image: Optional[float] = None,
    peak_memory_mb: Optional[float] = None,
    parameters_json: Optional[Path] = None,
    code_commit: Optional[str] = None,
) -> Path:
    if expected_eps is not None:
        expected_eps = parse_fraction(expected_eps)
        if not isinstance(expected_eps, (int, float)):
            raise ValueError("expected_eps must be a number or fraction such as 4/255")
        expected_eps = float(expected_eps)
    compatibility = registry.compatibility_status(attack_id, source_id)
    if compatibility["status"] == "unsupported":
        raise ValueError(compatibility["reason"])
    dataset = registry.dataset(dataset_id)
    annotation = annotation.resolve()
    image_root = image_root.resolve()
    clean_root = clean_root.resolve() if clean_root is not None else None
    parameters_json = parameters_json.resolve() if parameters_json is not None else None
    if not annotation.is_file() or not image_root.is_dir():
        raise FileNotFoundError("Artifact annotation and image root must exist")
    if clean_root is not None and not clean_root.is_dir():
        raise FileNotFoundError("Clean image root must exist: {}".format(clean_root))
    if expected_eps is not None:
        if not 0.0 <= expected_eps <= 1.0:
            raise ValueError("expected_eps must be in normalized [0, 1] units")
        if clean_root is None:
            raise ValueError("--clean-root is required when --expected-eps is provided")
    if budget_profile is not None and budget_profile not in registry.budget_profiles:
        raise ValueError("Unknown budget profile: {}".format(budget_profile))
    if gradient_evaluations_per_image is not None and gradient_evaluations_per_image <= 0:
        raise ValueError("gradient_evaluations_per_image must be positive")
    for name, value in (
        ("attack_seconds_per_image", attack_seconds_per_image),
        ("peak_memory_mb", peak_memory_mb),
    ):
        if value is not None and value < 0.0:
            raise ValueError("{} must be non-negative".format(name))
    parameters: Optional[Dict[str, Any]] = None
    parameters_sha256: Optional[str] = None
    if parameters_json is not None:
        if not parameters_json.is_file():
            raise FileNotFoundError("Missing parameter manifest: {}".format(parameters_json))
        parameters = json.loads(parameters_json.read_text(encoding="utf-8"))
        if not isinstance(parameters, dict):
            raise ValueError("Parameter manifest must contain a JSON object")
        parameters_sha256 = file_digest(parameters_json)
    if budget_profile == "compute_matched":
        if attack_id not in registry.baseline_profiles:
            raise ValueError(
                "No audited compute-matched baseline profile exists for {}".format(
                    attack_id
                )
            )
        profile = registry.budget_profiles[budget_profile]
        required_eps = parse_fraction(profile["epsilon"])
        if not isinstance(required_eps, (int, float)):
            raise ValueError("compute_matched epsilon is not numeric")
        required_eps = float(required_eps)
        max_gradients = int(profile["max_gradient_evaluations_per_image"])
        if expected_eps is None or abs(expected_eps - required_eps) > 1.0e-12:
            raise ValueError(
                "compute_matched artifacts require expected_eps={}"
                .format(required_eps)
            )
        if gradient_evaluations_per_image is None:
            raise ValueError(
                "compute_matched artifacts require gradient_evaluations_per_image"
            )
        if gradient_evaluations_per_image > max_gradients:
            raise ValueError(
                "compute_matched artifact uses {} gradient evaluations; maximum is {}"
                .format(gradient_evaluations_per_image, max_gradients)
            )
        required_fields = {
            "parameters_json": parameters_json,
            "code_commit": code_commit.strip() if code_commit else None,
            "attack_seconds_per_image": attack_seconds_per_image,
            "peak_memory_mb": peak_memory_mb,
        }
        missing_fields = sorted(
            name for name, value in required_fields.items() if value is None
        )
        if missing_fields:
            raise ValueError(
                "compute_matched artifacts require: {}".format(
                    ", ".join(missing_fields)
                )
            )
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    categories = payload.get("categories", [])
    if len(categories) != dataset.num_classes:
        raise ValueError(
            "Artifact category count {} does not match {} ({})".format(
                len(categories), dataset_id, dataset.num_classes
            )
        )
    images = payload.get("images", [])
    if not images:
        raise ValueError("Artifact annotation contains no images")
    missing = []
    unreadable = []
    max_linf_pixels = 0.0

    def checked_member(root: Path, filename: str) -> Path:
        candidate = (root / filename).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("Artifact file escapes its image root: {}".format(filename)) from exc
        return candidate

    for record in images:
        filename = str(record["file_name"])
        path = checked_member(image_root, filename)
        if not path.is_file():
            missing.append(str(path))
            continue
        try:
            with Image.open(str(path)) as image:
                image.verify()
        except Exception as exc:
            unreadable.append("{}: {}".format(path, exc))
            continue
        if clean_root is not None:
            clean_path = checked_member(clean_root, filename)
            if not clean_path.is_file():
                missing.append(str(clean_path))
                continue
            try:
                with Image.open(str(path)) as adversarial_image, Image.open(
                    str(clean_path)
                ) as clean_image:
                    adversarial = np.asarray(adversarial_image.convert("RGB"), dtype=np.int16)
                    clean = np.asarray(clean_image.convert("RGB"), dtype=np.int16)
                if adversarial.shape != clean.shape:
                    raise ValueError(
                        "Image shape mismatch for {}: {} != {}".format(
                            filename, adversarial.shape, clean.shape
                        )
                    )
                image_linf = float(np.abs(adversarial - clean).max(initial=0))
                max_linf_pixels = max(max_linf_pixels, image_linf)
            except Exception as exc:
                unreadable.append("{} / {}: {}".format(path, clean_path, exc))
    if missing or unreadable:
        raise RuntimeError(
            "Artifact validation failed: {} missing, {} unreadable; examples: {}".format(
                len(missing), len(unreadable), (missing + unreadable)[:5]
            )
        )
    if expected_eps is not None:
        allowed_pixels = expected_eps * 255.0
        if max_linf_pixels > allowed_pixels + 0.5 + 1e-9:
            raise RuntimeError(
                "Artifact exceeds L-infinity budget: {:.3f}/255 > {:.3f}/255".format(
                    max_linf_pixels, allowed_pixels
                )
            )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        output_dir
        or project_root()
        / "outputs"
        / "artifacts"
        / dataset_id
        / source_id
        / attack_id
        / stamp
    ).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty artifact run: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    if expected_eps is None:
        warnings.append(
            "No perturbation budget was claimed; provide --clean-root and --expected-eps before publication."
        )
    if budget_profile is None:
        warnings.append("No budget profile was declared.")
    if gradient_evaluations_per_image is None:
        warnings.append("No source-model gradient-evaluation count was declared.")
    if parameters_json is None:
        warnings.append("No exact parameter JSON manifest was archived.")
    metadata: Dict[str, Any] = {
        "status": "complete",
        "external_artifact": True,
        "dataset": dataset_id,
        "source": source_id,
        "attack": attack_id,
        "images": len(images),
        "annotation": str(annotation),
        "annotation_sha256": file_digest(annotation),
        "image_root": str(image_root),
        "clean_root": str(clean_root) if clean_root is not None else None,
        "expected_eps": expected_eps,
        "max_linf_pixels": max_linf_pixels if clean_root is not None else None,
        "max_linf_normalized": max_linf_pixels / 255.0 if clean_root is not None else None,
        "budget_verified": expected_eps is not None,
        "budget_profile": budget_profile,
        "budget_profile_definition": (
            registry.budget_profiles.get(budget_profile)
            if budget_profile is not None
            else None
        ),
        "baseline_profile_definition": registry.baseline_profiles.get(attack_id),
        "gradient_evaluations_per_image": gradient_evaluations_per_image,
        "attack_seconds_per_image": attack_seconds_per_image,
        "peak_memory_mb": peak_memory_mb,
        "parameters": parameters,
        "parameters_json": str(parameters_json) if parameters_json is not None else None,
        "parameters_sha256": parameters_sha256,
        "code_commit": code_commit,
        "compatibility": compatibility,
        "warning": "; ".join(warnings) if warnings else None,
    }
    atomic_json(run_dir / "run.json", metadata)
    return run_dir
