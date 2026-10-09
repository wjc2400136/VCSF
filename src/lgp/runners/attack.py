from __future__ import annotations

import math
import time
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from skimage.metrics import structural_similarity

from ..adapters import OpenMMLabAdapter
from ..attacks import build_attack
from ..attacks.common import parse_fraction
from ..attacks.isolated import build_isolated_vcsf, resolve_isolated_vcsf
from ..data.coco import CocoIndex, select_coco_images
from ..io import append_jsonl, atomic_json, file_digest
from ..modeling import load_detector
from ..paths import project_root
from ..registry import Registry


PSNR_IDENTICAL_CAP_DB = 100.0


def _default_run_dir(dataset: str, source: str, attack: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root() / "outputs" / "attacks" / dataset / source / attack / stamp


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root()),
            check=True,
            capture_output=True,
            text=True,
        )
        candidate = completed.stdout.strip().lower()
        if len(candidate) == 40 and all(
            character in "0123456789abcdef" for character in candidate
        ):
            return candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return "unavailable"


def _sequence_digest(values: Any) -> str:
    encoded = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_bgr(path: Path) -> torch.Tensor:
    with Image.open(str(path)) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return torch.from_numpy(bgr).permute(2, 0, 1).unsqueeze(0)


def _save_bgr_png(tensor: torch.Tensor, path: Path) -> None:
    bgr = tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb = np.rint(bgr[:, :, ::-1]).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(str(path), format="PNG", compress_level=1)


def _quality(clean: torch.Tensor, adversarial: torch.Tensor) -> Dict[str, float]:
    clean_np = clean.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = adversarial.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    difference = adv_np - clean_np
    normalized = difference / 255.0
    mse = float(np.mean(difference ** 2))
    # Exact identity has mathematically infinite PSNR, which is not a valid
    # portable JSON number and would poison an otherwise finite 5,000-image
    # aggregate.  Use a declared 100 dB ceiling for both identity and any
    # smaller positive MSE; MSE itself remains the exact, uncapped signal.
    psnr = (
        PSNR_IDENTICAL_CAP_DB
        if mse == 0.0
        else min(
            PSNR_IDENTICAL_CAP_DB,
            20.0 * math.log10(255.0 / math.sqrt(mse)),
        )
    )
    ssim = float(
        structural_similarity(
            clean_np,
            adv_np,
            data_range=255.0,
            channel_axis=2,
        )
    )
    denominator = float(np.sum(clean_np ** 2))
    nmse = (
        float(np.sum(difference ** 2) / denominator)
        if denominator > 0.0
        else float("inf")
    )
    adv_normalized = adv_np / 255.0
    total_variation = float(
        np.mean(np.abs(np.diff(adv_normalized, axis=0)))
        + np.mean(np.abs(np.diff(adv_normalized, axis=1)))
    )
    return {
        "mse_pixel": mse,
        "psnr": psnr,
        "ssim": ssim,
        "linf_normalized": float(np.max(np.abs(normalized))),
        "l2_rms_normalized": float(np.sqrt(np.mean(normalized ** 2))),
        "l0_fraction_gt_half_pixel": float(np.mean(np.abs(difference) > 0.5)),
        "mean_abs_normalized": float(np.mean(np.abs(normalized))),
        "nmse": nmse,
        "total_variation_normalized": total_variation,
    }


def _validate_budget_profile(
    registry: Registry, budget_profile: Optional[str], attack: Any
) -> None:
    if budget_profile is None:
        return
    if budget_profile not in registry.budget_profiles:
        raise ValueError("Unknown budget profile: {}".format(budget_profile))
    profile = registry.budget_profiles[budget_profile]
    if profile.get("epsilon") == "registered_axis":
        allowed = profile.get("allowed_epsilons")
        if not isinstance(allowed, list) or not allowed:
            raise ValueError("Registered-axis budget requires an explicit radius list")
        radii = [parse_fraction(value) for value in allowed]
        actual_eps = getattr(attack.config, "eps", None)
        if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in radii)
                or type(actual_eps) not in (int, float) or not math.isfinite(actual_eps)
                or actual_eps not in radii):
            raise ValueError("Attack radius is outside the registered budget axis")
        if (getattr(attack.config, "step_size", None) != parse_fraction(profile.get("step_size"))
                or type(getattr(attack.config, "iterations", None)) is not int
                or attack.config.iterations != profile.get("required_iterations")):
            raise ValueError("Registered-axis budget step or update schedule differs")
    required_eps = parse_fraction(profile.get("epsilon"))
    if isinstance(required_eps, (int, float)):
        actual_eps = getattr(attack.config, "eps", None)
        if actual_eps is None or abs(float(actual_eps) - float(required_eps)) > 1.0e-12:
            raise ValueError(
                "{} requires epsilon={}, but the attack resolves to {}".format(
                    budget_profile, required_eps, actual_eps
                )
            )
    max_gradients = parse_fraction(
        profile.get("max_gradient_evaluations_per_image")
    )
    if isinstance(max_gradients, (int, float)):
        actual_gradients = int(attack.gradient_evaluations_per_image)
        if actual_gradients > int(max_gradients):
            raise ValueError(
                "{} permits at most {} gradient image-equivalents, but the attack "
                "resolves to {}".format(
                    budget_profile, int(max_gradients), actual_gradients
                )
            )


def _research_components(mode):
    if mode == "a10_f03_swin_fullval_execution":
        from ..attacks.vcsf_a10_f03_swin_fullval_isolation import (
            build_swin_fullval, resolve_swin_fullval,
        )

        return resolve_swin_fullval, build_swin_fullval
    if mode == "final_background_execution":
        from functools import partial
        from ..attacks.vcsf_common_anchor_execution_isolation import resolve_common_anchor_execution, build_common_anchor_execution
        from . import vcsf_final_background_execution_contract

        return partial(resolve_common_anchor_execution,
            contract=vcsf_final_background_execution_contract), build_common_anchor_execution
    if mode == "final_budget_execution":
        from ..attacks.vcsf_final_budget_isolation import resolve_final_budget, build_final_budget

        return resolve_final_budget, build_final_budget
    if mode == "final_adaptive_execution":
        from ..attacks.vcsf_final_adaptive_isolation import resolve_final_adaptive, build_final_adaptive

        return resolve_final_adaptive, build_final_adaptive
    if mode == "final_followup_successor_execution":
        from ..attacks.vcsf_final_followup_successor_isolation import resolve_followup_successor_execution
        from ..attacks.vcsf_common_anchor_execution_isolation import build_common_anchor_execution

        return resolve_followup_successor_execution, build_common_anchor_execution
    if mode == "final_followup_execution":
        from ..attacks.vcsf_final_followup_execution_isolation import resolve_followup_execution
        from ..attacks.vcsf_common_anchor_execution_isolation import build_common_anchor_execution

        return resolve_followup_execution, build_common_anchor_execution
    if mode == "final_attribution_execution":
        from functools import partial
        from ..attacks.vcsf_common_anchor_execution_isolation import resolve_common_anchor_execution, build_common_anchor_execution
        from . import vcsf_final_attribution_execution_contract

        return partial(resolve_common_anchor_execution,
            contract=vcsf_final_attribution_execution_contract), build_common_anchor_execution
    if mode == "single_source_common_anchor_execution":
        from ..attacks.vcsf_common_anchor_execution_isolation import resolve_common_anchor_execution, build_common_anchor_execution

        return resolve_common_anchor_execution, build_common_anchor_execution
    if mode == "single_source_operator_factorial_execution":
        from ..attacks.vcsf_operator_execution_isolation import resolve_operator_execution, build_operator_execution

        return resolve_operator_execution, build_operator_execution
    if mode == "single_source_ablation_executor_smoke":
        from ..attacks.vcsf_ablation_executor_isolation import resolve_executor_smoke
        from ..attacks.vcsf_ablation_preflight_isolation import build_ablation_preflight

        return resolve_executor_smoke, build_ablation_preflight
    if mode == "single_source_ablation_preflight":
        from ..attacks.vcsf_ablation_preflight_isolation import (
            resolve_ablation_preflight, build_ablation_preflight,
        )

        return resolve_ablation_preflight, build_ablation_preflight
    if mode == "single_source_scale_efficacy":
        from ..attacks.vcsf_scale_execution_isolation import resolve_scale_efficacy
        from ..attacks.vcsf_scale_isolation import build_scale_preflight

        return resolve_scale_efficacy, build_scale_preflight
    if mode == "scale_structure_preflight":
        from ..attacks.vcsf_scale_isolation import resolve_scale_preflight, build_scale_preflight

        return resolve_scale_preflight, build_scale_preflight
    from ..attacks.vcsf_research_isolation import resolve_research_preflight, build_research_preflight

    if mode == "layered_efficacy":
        from ..attacks.vcsf_efficacy_isolation import resolve_research_efficacy

        return resolve_research_efficacy, build_research_preflight
    return resolve_research_preflight, build_research_preflight


def run_attack(
    registry: Registry,
    dataset_id: str,
    source_id: str,
    attack_id: str,
    split: str = "val",
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    seed: int = 42,
    device: str = "cuda:0",
    download_weights: bool = False,
    keep_going: bool = False,
    strict: bool = False,
    parameter_overrides: Optional[Mapping[str, Any]] = None,
    budget_profile: Optional[str] = None,
    image_ids: Optional[Sequence[int]] = None,
    input_transform: Any = None,
    run_metadata: Optional[Mapping[str, Any]] = None,
    progress_callback: Any = None,
    seed_offsets: Optional[Mapping[int, int]] = None,
    isolated_candidate: Optional[Mapping[str, Any]] = None,
    isolated_research: Optional[Mapping[str, Any]] = None,
    isolated_research_owner: Any = None,
    public_execution_context: Any = None,
) -> Path:
    fresh_public_reproduction = attack_id == "vcsf" and public_execution_context is None
    fresh_public_radius = fresh_public_reproduction and budget_profile == "vcsf_final_radius_observed_cost"
    fresh_public_background = fresh_public_reproduction and (run_metadata or {}).get(
        "public_protocol") == "current_vcsf_final_background"
    if public_execution_context is not None and attack_id != "vcsf":
        raise RuntimeError("Native A10 main context cannot authorize another attack")
    if attack_id == "vcsf":
        if isolated_candidate is not None or isolated_research is not None:
            raise RuntimeError("Public vcsf cannot be replaced by a historical or research overlay")
        from ..attacks.vcsf_public import guard_public_execution

        guard_public_execution(registry, parameter_overrides, public_execution_context,
            dict(dataset_id=dataset_id, source_id=source_id, attack_id=attack_id,
                split=split, output_dir=output_dir, max_images=max_images, seed=seed,
                device=device, strict=strict, keep_going=keep_going, download_weights=download_weights,
                budget_profile=budget_profile, image_ids=image_ids, seed_offsets=seed_offsets,
                parameter_overrides=parameter_overrides, input_transform=input_transform,
                run_metadata=run_metadata))
    dataset = registry.dataset(dataset_id)
    source = registry.model(source_id)
    use_isolated_candidate = bool(
        isolated_candidate
        and attack_id == str(isolated_candidate.get("execution_alias", ""))
    )
    if isolated_research is not None:
        if isolated_candidate is not None:
            raise RuntimeError("Frozen-candidate and research overlays are mutually exclusive")
        if not strict or keep_going or download_weights:
            raise RuntimeError("Research preflight requires strict failure handling and local weights")
        research_resolver, research_builder = _research_components(isolated_research.get("mode"))
        research_execution_arguments = {}
        if isolated_research.get("mode") in ("final_adaptive_execution", "final_budget_execution"):
            research_execution_arguments["gpu_owner"] = isolated_research_owner
        if isolated_research.get("mode") in (
            "final_followup_execution", "final_followup_successor_execution", "final_adaptive_execution", "final_budget_execution"
        ):
            research_execution_arguments.update(output_dir=output_dir, device=device)
        if isolated_research.get("mode") in ("layered_efficacy", "single_source_scale_efficacy",
                "single_source_ablation_preflight", "single_source_ablation_executor_smoke",
                "single_source_operator_factorial_execution", "single_source_common_anchor_execution", "final_attribution_execution",
                "final_background_execution", "a10_f03_swin_fullval_execution"):
            research_execution_arguments["output_dir"] = output_dir
        attack_spec = research_resolver(
            registry, isolated_research, dataset_id=dataset_id, split=split,
            source_id=source_id, attack_id=attack_id, seed=seed, max_images=max_images,
            parameter_overrides=dict(parameter_overrides or {}),
            run_metadata=dict(run_metadata or {}), budget_profile=budget_profile,
            image_ids=image_ids, seed_offsets=seed_offsets, input_transform=input_transform,
            **research_execution_arguments,
        )
    else:
        attack_spec = (
            resolve_isolated_vcsf(
                registry,
                attack_id,
                source_id,
                isolated_candidate or {},
            )
            if use_isolated_candidate
            else registry.attack(attack_id)
        )
    if not source.source:
        raise ValueError("{} is a target-only detector, not one of the six source models".format(source_id))
    run_dir = (output_dir or _default_run_dir(dataset_id, source_id, attack_id)).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty attack run: {}".format(run_dir))
    run_dir.mkdir(parents=True, exist_ok=True)

    compatibility = (
        {
            "status": "native",
            "reason": (
                "Hash-bound isolated retrospective efficacy execution."
                if isolated_research is not None and isolated_research.get("mode") in ("layered_efficacy", "single_source_scale_efficacy", "single_source_operator_factorial_execution", "single_source_common_anchor_execution", "final_attribution_execution", "final_background_execution", "a10_f03_swin_fullval_execution")
                else "Plan-bound isolated research structure preflight, not formal efficacy."
                if isolated_research is not None
                else "Verified process-local VCSF execution overlay."
            ),
            "requires": [],
        }
        if use_isolated_candidate or isolated_research is not None
        else registry.compatibility_status(attack_id, source_id)
    )
    if compatibility["status"] != "native":
        run_status = {
            "external": "artifact_required",
            "adaptable": "adapter_required",
            "unsupported": "skipped",
        }[compatibility["status"]]
        payload = {
            "status": run_status,
            "dataset": dataset_id,
            "source": source_id,
            "attack": attack_id,
            "compatibility": compatibility,
            "reason": compatibility["reason"],
        }
        atomic_json(run_dir / "run.json", payload)
        if strict:
            raise RuntimeError(payload["reason"])
        return run_dir
    index = CocoIndex(dataset, split)
    selected_images = select_coco_images(
        index.images,
        max_images=max_images,
        image_ids=image_ids,
    )
    normalized_seed_offsets = (
        {int(key): int(value) for key, value in seed_offsets.items()}
        if seed_offsets is not None
        else {}
    )
    resolved_seed_offsets = {
        int(image["id"]): (
            normalized_seed_offsets[int(image["id"])]
            if seed_offsets is not None
            and int(image["id"]) in normalized_seed_offsets
            else position
        )
        for position, image in enumerate(selected_images)
    }
    if seed_offsets is not None:
        missing_seed_offsets = [
            int(image["id"])
            for image in selected_images
            if int(image["id"]) not in normalized_seed_offsets
        ]
        if missing_seed_offsets:
            raise ValueError(
                "seed_offsets omits selected image IDs: {}".format(
                    missing_seed_offsets[:10]
                )
            )
    if any(value < 0 for value in resolved_seed_offsets.values()):
        raise ValueError("seed_offsets must be non-negative")
    if fresh_public_reproduction:
        from .vcsf_public_execution import PublicAttackExecutionContext

        public_execution_context = PublicAttackExecutionContext(
            registry, dataset_id, source_id, split, run_dir, selected_images,
            seed, resolved_seed_offsets, device,
            **({"parameter_overrides": parameter_overrides} if fresh_public_radius else
               {"parameter_overrides": parameter_overrides,
                "background_variant": run_metadata["background_variant"]}
               if fresh_public_background else {}),
        )
    model, loaded_cfg, checkpoint = load_detector(
        source,
        dataset,
        run_dir / "model",
        device=device,
        download=download_weights,
        split=split,
    )
    if isolated_research is not None and isolated_research.get("mode") == "a10_f03_swin_fullval_execution":
        from ..attacks.vcsf_a10_f03_swin_fullval_isolation import verify_source_checkpoint

        verify_source_checkpoint(checkpoint, attack_spec.metadata["source_checkpoint_sha256"])
    adapter = OpenMMLabAdapter(model, input_transform=input_transform)
    if public_execution_context is not None:
        public_execution_context.loaded(model, loaded_cfg, checkpoint, source_id, run_dir / "model")
        adapter = public_execution_context.observe_adapter(adapter)
    parameters = dict(attack_spec.parameters)
    parameters.update(dict(parameter_overrides or {}))
    if fresh_public_radius:
        from ..attacks.current_vcsf_radius import build_radius

        attack = build_radius(registry, adapter, parameters)
    elif fresh_public_background:
        from ..attacks.current_vcsf_background import build_background

        attack = build_background(registry, adapter, parameters, run_metadata["background_variant"])
    elif isolated_research is not None:
        attack = research_builder(adapter, parameters)
    else:
        attack = (
            build_isolated_vcsf(adapter, parameters, isolated_candidate or {})
            if use_isolated_candidate
            else build_attack(attack_id, adapter, parameters)
        )
    _validate_budget_profile(registry, budget_profile, attack)
    config = attack.config
    resolved_parameters = asdict(config)
    parameter_bytes = json.dumps(
        resolved_parameters, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    images_dir = run_dir / "images"
    manifest = run_dir / "manifest.jsonl"
    filenames: Dict[int, str] = {}
    successful = 0
    failed = 0
    quality_sum = {
        "mse_pixel": 0.0,
        "psnr": 0.0,
        "ssim": 0.0,
        "linf_normalized": 0.0,
        "l2_rms_normalized": 0.0,
        "l0_fraction_gt_half_pixel": 0.0,
        "mean_abs_normalized": 0.0,
        "nmse": 0.0,
        "total_variation_normalized": 0.0,
    }
    quality_finite_count = {key: 0 for key in quality_sum}
    quality_nonfinite_count = {key: 0 for key in quality_sum}
    runtime_seconds = 0.0
    pixel_identical_images = 0
    actual_gradient_evaluations_total = 0
    auxiliary_forward_passes = 0
    auxiliary_backward_passes = 0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    for position, image in enumerate(selected_images):
        image_id = int(image["id"])
        source_path = index.image_path(image)
        try:
            if public_execution_context is not None:
                public_execution_context.before_image(position, image_id, source_path)
            clean = _load_bgr(source_path)
            box_values, label_values = index.instances(image_id)
            boxes = torch.tensor(box_values, dtype=torch.float32).reshape(-1, 4)
            labels = torch.tensor(label_values, dtype=torch.long)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            image_seed = seed + resolved_seed_offsets[image_id]
            result = attack(clean, boxes, labels, seed=image_seed)
            if not bool(torch.isfinite(result.adversarial_bgr).all().item()):
                raise RuntimeError("Attack produced a non-finite adversarial image")
            if not bool(torch.isfinite(result.perturbation).all().item()):
                raise RuntimeError("Attack produced a non-finite perturbation")
            if not math.isfinite(float(result.linf_pixel)):
                raise RuntimeError("Attack produced a non-finite L-infinity norm")
            try:
                json.dumps(result.diagnostics, ensure_ascii=False, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Attack diagnostics are not strict finite JSON: {}".format(exc)
                ) from exc
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            if result.linf_pixel > config.eps * 255.0 + 1.0e-4:
                raise RuntimeError(
                    "L-infinity budget violated: {:.6f} pixels".format(result.linf_pixel)
                )
            if public_execution_context is not None:
                public_execution_context.after_image(result, elapsed)
            metrics = _quality(clean.to(result.adversarial_bgr.device), result.adversarial_bgr)
            pixel_identical = bool(
                torch.equal(clean.to(result.adversarial_bgr.device), result.adversarial_bgr)
            )
            identity_diagnostics = [
                item
                for item in result.diagnostics
                if item.get("empty_target_behavior") == "identity_output"
            ]
            if identity_diagnostics and not pixel_identical:
                raise RuntimeError(
                    "identity_output diagnostic does not preserve the input exactly"
                )
            image_gradient_evaluations = int(
                identity_diagnostics[0].get("actual_gradient_evaluations", 0)
                if identity_diagnostics
                else (
                    attack.gradient_evaluations_for_output(result)
                    if callable(getattr(attack, 'gradient_evaluations_for_output', None))
                    else attack.gradient_evaluations_per_image
                )
            )
            if not 0 <= image_gradient_evaluations <= int(
                attack.gradient_evaluations_per_image
            ):
                raise RuntimeError("Invalid per-image gradient accounting")
            image_auxiliary_forwards = int(
                attack.auxiliary_forward_passes_for_output(result)
                if callable(getattr(attack, "auxiliary_forward_passes_for_output", None))
                else getattr(attack, "auxiliary_forward_passes_per_image", 0)
            )
            image_auxiliary_backwards = int(
                getattr(attack, "auxiliary_backward_passes_per_image", 0)
            )
            if image_auxiliary_forwards < 0 or image_auxiliary_backwards < 0:
                raise RuntimeError("Invalid per-image auxiliary accounting")
            relative_name = "images/{:012d}.png".format(image_id)
            destination = run_dir / relative_name
            _save_bgr_png(result.adversarial_bgr, destination)
            output_sha256 = file_digest(destination)
            output_bytes = destination.stat().st_size
            append_jsonl(
                manifest,
                {
                    "status": "ok",
                    "position": position,
                    "image_id": image_id,
                    "attack_seed": image_seed,
                    "source_file": str(source_path),
                    "output_file": relative_name,
                    "output_sha256": output_sha256,
                    "output_bytes": output_bytes,
                    "linf_pixel": result.linf_pixel,
                    "runtime_seconds": elapsed,
                    "quality": metrics,
                    "pixel_identical": pixel_identical,
                    "actual_gradient_evaluations": image_gradient_evaluations,
                    "auxiliary_forward_passes": image_auxiliary_forwards,
                    "auxiliary_backward_passes": image_auxiliary_backwards,
                    "diagnostics": result.diagnostics,
                },
            )
            pixel_identical_images += int(pixel_identical)
            actual_gradient_evaluations_total += image_gradient_evaluations
            for key, value in metrics.items():
                if math.isfinite(value):
                    quality_sum[key] += value
                    quality_finite_count[key] += 1
                else:
                    quality_nonfinite_count[key] += 1
            filenames[image_id] = relative_name
            successful += 1
            runtime_seconds += elapsed
            auxiliary_forward_passes += image_auxiliary_forwards
            auxiliary_backward_passes += image_auxiliary_backwards
        except Exception as exc:
            if public_execution_context is not None and public_execution_context.observer is not None:
                public_execution_context.observer.close()
            failed += 1
            append_jsonl(
                manifest,
                {
                    "status": "failed",
                    "position": position,
                    "image_id": image_id,
                    "source_file": str(source_path),
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                },
            )
            if progress_callback is not None:
                progress_callback(
                    position + 1, len(selected_images), image_id, "failed"
                )
            if not keep_going:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            if progress_callback is not None:
                progress_callback(position + 1, len(selected_images), image_id, "ok")

    annotation = index.adversarial_annotation(filenames)
    atomic_json(run_dir / "annotations.json", annotation)
    divisor = max(successful, 1)
    summary = {
        "status": "complete" if failed == 0 else "complete_with_failures",
        "dataset": dataset_id,
        "split": split,
        "source": source_id,
        "attack": attack_id,
        "seed": seed,
        "requested_images": len(selected_images),
        "successful_images": successful,
        "failed_images": failed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_digest(checkpoint),
        "code_commit": _git_commit(),
        "parameters": resolved_parameters,
        "parameters_declared": parameters,
        "parameters_sha256": hashlib.sha256(parameter_bytes).hexdigest(),
        "budget_profile": budget_profile,
        "budget_profile_definition": (
            registry.budget_profiles.get(budget_profile)
            if budget_profile is not None
            else None
        ),
        "baseline_profile_definition": registry.baseline_profiles.get(attack_id),
        "gradient_evaluations_per_image": int(
            attack.gradient_evaluations_per_image
        ),
        "actual_gradient_evaluations_total": actual_gradient_evaluations_total,
        "actual_gradient_evaluations_mean": (
            actual_gradient_evaluations_total / divisor
        ),
        "auxiliary_passes": {
            "forward_total": auxiliary_forward_passes,
            "forward_mean": auxiliary_forward_passes / divisor,
            "backward_total": auxiliary_backward_passes,
            "backward_mean": auxiliary_backward_passes / divisor,
            "note": "Method-defined one-time partial passes excluded from the dominant full-image backward cap.",
        },
        "implementation": (
            str(attack_spec.metadata.get("implementation"))
            if use_isolated_candidate or attack_id == "vcsf"
            else "src/lgp/attacks/{}.py".format(
                "sfim" if attack_id == "sfim_b" else attack_id
            )
        ),
        "runtime_compatibility": dict(
            getattr(attack, "runtime_compatibility", {})
        ),
        "fidelity_status": attack_spec.metadata.get("fidelity_status"),
        "fidelity_basis": attack_spec.metadata.get("fidelity_basis"),
        "semantic_contract": attack_spec.metadata.get("semantic_contract"),
        "paper": attack_spec.metadata.get("paper"),
        "audited_repository": attack_spec.metadata.get("repository"),
        "audit_commit": attack_spec.metadata.get("audit_commit"),
        "implementation_adjudication": (
            attack_spec.metadata.get("discrepancy_resolution")
            or attack_spec.metadata.get("paper_discrepancy_resolution")
            or attack_spec.metadata.get("deviation")
            or attack_spec.metadata.get("repair")
            or attack_spec.metadata.get("reconstruction_basis")
            or attack_spec.metadata.get("implementation_note")
        ),
        "implementation_adjudication_details": {
            key: attack_spec.metadata.get(key)
            for key in (
                "discrepancy_resolution",
                "paper_discrepancy_resolution",
                "deviation",
                "repair",
                "reconstruction_basis",
                "repository_status",
                "implementation_note",
            )
            if attack_spec.metadata.get(key) is not None
        },
        "quality_mean": {
            key: (
                value / quality_finite_count[key]
                if quality_finite_count[key] > 0
                and quality_nonfinite_count[key] == 0
                else None
            )
            for key, value in quality_sum.items()
        },
        "quality_finite_count": quality_finite_count,
        "quality_nonfinite_count": quality_nonfinite_count,
        "quality_definition": {
            "psnr_identical_cap_db": PSNR_IDENTICAL_CAP_DB,
            "psnr_note": (
                "PSNR is capped at 100 dB; exact MSE remains available per image."
            ),
        },
        "pixel_identical_images": pixel_identical_images,
        "runtime_seconds_total": runtime_seconds,
        "runtime_seconds_mean": runtime_seconds / divisor,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if torch.cuda.is_available() and str(device).startswith("cuda")
            else None
        ),
        "annotation": "annotations.json",
        "annotation_sha256": file_digest(run_dir / "annotations.json"),
        "manifest": "manifest.jsonl",
        "manifest_sha256": file_digest(manifest),
        "image_ids_sha256": _sequence_digest(sorted(filenames)),
        "requested_ordered_image_ids_sha256": _sequence_digest(
            [int(image["id"]) for image in selected_images]
        ),
        "seed_schedule": (
            "explicit_offsets" if seed_offsets is not None else "selected_position"
        ),
        "seed_offsets_sha256": _sequence_digest(
            [
                [int(image["id"]), resolved_seed_offsets[int(image["id"])]]
                for image in selected_images
            ]
        ),
        "run_metadata": dict(run_metadata or {}),
        "image_root": ".",
        "payload_state": "full_generated_payload",
        "full_payload_available": failed == 0,
    }
    if isolated_research is not None:
        summary["research_execution"] = dict(attack_spec.metadata)
        summary["formal_AP_eligible"] = False
        if isolated_research.get("mode") == "a10_f03_swin_fullval_execution":
            summary["implementation"] = attack_spec.metadata["implementation"]
        summary["auxiliary_passes"]["note"] = (
            "Explicit auxiliary calls only; zero does not exclude detached clean-reference "
            "work inside paired feature batches or imply equal physical computation."
        )
    if public_execution_context is not None:
        public_execution_context.close_observer()
        native_peaks = public_execution_context.whole_run_cuda_peaks()
        peak_allocated = native_peaks["peak_allocated_bytes"]
        summary["peak_cuda_memory_mb"] = (
            peak_allocated / (1024.0 * 1024.0) if peak_allocated is not None else None
        )
        summary["native_run_peak_cuda_memory_bytes"] = native_peaks
        execution_key = "public_reproduction_execution" if fresh_public_reproduction else "native_main_execution"
        summary[execution_key] = public_execution_context.metadata()
        summary["auxiliary_passes"]["note"] = (
            "Explicit auxiliary calls only; paired clean-reference rows are recorded separately "
            "in native_cost.jsonl and are not free or calibrated complete-detector BE."
        )
    atomic_json(run_dir / "run.json", summary)
    return run_dir
