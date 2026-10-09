"""Fresh public VCSF cost observations, not historical main-run admission."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from ..io import append_jsonl, atomic_json, file_digest


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _content_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class PublicAttackExecutionContext:
    """Bind one ordinary attack run and observe its unchanged adapter/gradient path."""

    def __init__(self, registry, dataset_id, source_id, split, run_dir,
                 selected_images, seed, seed_offsets, device, parameter_overrides=None,
                 background_variant=None):
        import torch

        self.registry = registry
        self.root = registry.root
        self.dataset_id = dataset_id
        self.source_id = source_id
        self.split = split
        self.run_dir = Path(run_dir).resolve()
        self.device = torch.device(device)
        _require(self.device.type in ("cpu", "cuda")
                 and (self.device.type != "cuda" or self.device.index is not None),
                 "Public observation requires CPU or an explicit CUDA device ordinal")
        _require(type(seed) is int, "Attack seed must be an integer")
        self.seed = seed
        self.registered_parameters = deepcopy(dict(registry.attack("vcsf").parameters))
        self.parameters = deepcopy(self.registered_parameters)
        self.radius_label = None
        self.background_variant = background_variant
        if background_variant is not None:
            from ..attacks.current_vcsf_background import validate_parameters

            self.parameters.update(parameter_overrides or {})
            validate_parameters(registry, self.parameters, background_variant)
        elif parameter_overrides is not None:
            from ..attacks.current_vcsf_radius import validate_parameters

            self.parameters.update(parameter_overrides)
            _, self.radius_label, _ = validate_parameters(registry, self.parameters)
        for name in ("iterations", "levels_per_stage"):
            _require(type(self.parameters.get(name)) is int and self.parameters[name] > 0,
                     "Registered VCSF {} must be a positive integer".format(name))
        from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorConfig

        resolved_config = VCSFCommonAnchorConfig.from_mapping(self.parameters)
        if self.radius_label is None:
            resolved_config.validate()
        self._parameters_hash = _content_hash(asdict(resolved_config))
        self._parameters_declared_hash = _content_hash(self.parameters)
        self._background_parameters_hash = _content_hash(asdict(
            VCSFCommonAnchorConfig.from_mapping(self.registered_parameters)))
        dataset = registry.dataset(dataset_id)
        prefix = dataset.root / dataset.split(split).image_prefix
        self.selected = []
        self.offsets = {}
        for position, image in enumerate(selected_images):
            image_id = int(image["id"])
            _require(image_id not in self.offsets, "Selected image IDs must be unique")
            source_path = (prefix / str(image["file_name"])).resolve()
            offset = position if seed_offsets is None else seed_offsets.get(image_id)
            _require(type(offset) is int and offset >= 0,
                     "Seed offsets must cover selected IDs with non-negative integers")
            self.offsets[image_id] = offset
            self.selected.append(dict(image_id=image_id, source_path=source_path,
                                      sha256=file_digest(source_path)))
        self.ids = [row["image_id"] for row in self.selected]
        self._input_hash = _content_hash([
            dict(image_id=row["image_id"], sha256=row["sha256"],
                 seed_offset=self.offsets[row["image_id"]]) for row in self.selected])
        self._loading = None
        self._image = None
        self._attempted_positions = set()
        self.observer = None
        self._peak_allocated_bytes = None
        self._peak_reserved_bytes = None

    def metadata(self):
        result = dict(
            schema="vcsf_public_cost_observation_v1",
            execution_scope="fresh_user_reproduction",
            fresh_user_reproduction=True,
            # max_images is not supplied here; selected count cannot establish run scope.
            diagnostic_only=None,
            historical_result_inheritance=False,
            main_admission_claimed=False,
            formal_result_eligible=False, scientific=False, science=False,
            formal=False, release=False,
            dataset=self.dataset_id, split=self.split, source=self.source_id,
            device=str(self.device), seed=self.seed,
            selected_images=len(self.selected),
            ordered_image_ids_sha256=_content_hash(self.ids),
            seed_offsets_sha256=_content_hash([
                [image_id, self.offsets[image_id]] for image_id in self.ids]),
            input_binding_sha256=self._input_hash,
            parameters_sha256=self._parameters_hash,
            parameters_declared_sha256=self._parameters_declared_hash,
            cost_observation_is_scientific_result=False,
            cost_observation_is_source_promotion=False,
        )
        if self.radius_label is not None:
            result.update(radius_sensitivity=True, epsilon=self.radius_label,
                registered_A10_background_parameters_sha256=self._background_parameters_hash)
        if self.background_variant is not None:
            result.update(final_background_control=True, background_variant=self.background_variant,
                registered_A10_background_parameters_sha256=self._background_parameters_hash)
        return result

    def loaded(self, model, config, checkpoint, source_id, directory):
        # The actual modeling loader owns strict loading; no second torch.load/tensor audit.
        _require(self._loading is None, "Source loading is already bound")
        _require(source_id == self.source_id, "Loaded source differs from this attack run")
        _require(getattr(model, "training", None) is False, "Loaded source must be in eval mode")
        directory = Path(directory).resolve()
        _require(self.run_dir in directory.parents, "Loading evidence must stay inside the run")
        path = directory / "native_loading.json"
        _require(not path.exists(), "Refusing to overwrite native loading evidence")
        from .vcsf_a10_main_context import normalized_config_hash

        binding = dict(
            checkpoint_sha256=file_digest(Path(checkpoint)),
            actual_loaded_effective_sha256=normalized_config_hash(config),
            loading_verification_owner="lgp.modeling.load_detector",
            tensor_reaudit_performed=False,
            historical_checkpoint_binding_required=False,
            **self.metadata(),
        )
        atomic_json(path, binding)
        self._loading = binding

    def observe_adapter(self, adapter):
        _require(self._loading is not None, "Bind the loaded source before observing its adapter")
        _require(self._image is None, "Cannot replace the observer during an image")
        self.close_observer()
        from .public_observation import ObservedAdapter

        self.observer = ObservedAdapter(adapter)
        return self.observer

    def before_image(self, position, image_id, source_path):
        _require(self.observer is not None, "Public cost observer is absent")
        self.observer.close()
        self.observer.calls.clear()
        self.observer.backward_events.clear()
        self._image = None
        _require(type(position) is int and 0 <= position < len(self.selected),
                 "Selected image position is out of range")
        _require(position not in self._attempted_positions, "Image position was already attempted")
        row = self.selected[position]
        _require(image_id == row["image_id"]
                 and Path(source_path).resolve() == row["source_path"]
                 and file_digest(source_path) == row["sha256"],
                 "Selected image position, ID, source path or current bytes changed")
        self._base_allocated = None
        if self.device.type == "cuda":
            import torch

            torch.cuda.synchronize(self.device)
            self._base_allocated = torch.cuda.memory_allocated(self.device)
            self.whole_run_cuda_peaks()
            torch.cuda.reset_peak_memory_stats(self.device)
        self._attempted_positions.add(position)
        self._image = dict(position=position, **row)

    def whole_run_cuda_peaks(self):
        if self.device.type == "cuda":
            import torch

            allocated = torch.cuda.max_memory_allocated(self.device)
            reserved = torch.cuda.max_memory_reserved(self.device)
            self._peak_allocated_bytes = max(self._peak_allocated_bytes or 0, allocated)
            self._peak_reserved_bytes = max(self._peak_reserved_bytes or 0, reserved)
        return dict(
            peak_allocated_bytes=self._peak_allocated_bytes,
            peak_reserved_bytes=self._peak_reserved_bytes,
            device=str(self.device),
            scope="whole_native_generation_including_inter_image_postprocessing",
        )

    def _validate_observation(self, diagnostics, calls, backwards):
        _require(dict(self.registry.attack("vcsf").parameters) == self.registered_parameters,
                 "Registered VCSF parameters changed during this run")
        _require(isinstance(diagnostics, list), "Native trajectory must be a list")
        _require(all(isinstance(row, dict) for row in diagnostics),
                 "Native trajectory rows must be mappings")
        _require(all(row.get("kind") in ("feature", "detector_initialization_attempt")
                     for row in calls + backwards), "Unknown native observation kind")
        _require(all(row.get("finite") is True for row in backwards),
                 "Non-finite observed input gradient")
        feature = [row for row in calls if row["kind"] == "feature"]
        risk = [row for row in calls if row["kind"] != "feature"]
        feature_backward = [row for row in backwards if row["kind"] == "feature"]
        risk_backward = [row for row in backwards if row["kind"] != "feature"]
        terms = sum(row.get("feature_surface_level_count", 0)
                    for row in diagnostics if row.get("phase") == "feature")
        identity = bool(diagnostics
                        and diagnostics[0].get("empty_target_behavior") == "identity_output")
        if identity:
            declared = diagnostics[0].get("actual_gradient_evaluations")
            _require(len(diagnostics) == 1 and type(declared) is int and declared == 0
                     and not any(row.get("phase") == "feature" for row in diagnostics)
                     and not calls and not backwards and terms == 0,
                     "Identity output performed undeclared differentiable work")
        else:
            iterations = self.parameters["iterations"]
            detector = (int(self.parameters["initialization"] == "detector")
                        if self.background_variant is not None else 1)
            refine = iterations - detector
            stages = (2 if self.parameters["surface"] == "cross_stage" else 1
                      if self.background_variant is not None else 2)
            levels = stages * self.parameters["levels_per_stage"]
            expected_phases = ["risk_initialization"] * detector + ["feature"] * refine
            _require(
                len(diagnostics) == len(backwards) == iterations
                and [row.get("step") for row in diagnostics] == list(range(iterations))
                and [row.get("phase") for row in diagnostics] == expected_phases
                and all(row.get("phase") == "feature"
                        and row.get("feature_surface_level_count") == levels
                        for row in diagnostics[detector:])
                and len(feature) == len(feature_backward) == refine
                and len(risk_backward) == detector
                and len(risk) in ((1, 2) if detector else (0,))
                and all(row.get("batch_rows") == 2 and row.get("requires_grad") is True
                        for row in feature)
                and all(row.get("batch_rows") == 1 and row.get("requires_grad") is True
                        for row in risk)
                and all(row.get("batch_rows") == 2 for row in feature_backward)
                and all(row.get("batch_rows") == 1 for row in risk_backward)
                and [row["kind"] for row in calls] == (
                    ["detector_initialization_attempt"] * len(risk) + ["feature"] * refine)
                and [row["kind"] for row in backwards] == (
                    ["detector_initialization_attempt"] * detector + ["feature"] * refine)
                and terms == refine * levels,
                "Native initialization/refinement trajectory or cost schedule changed",
            )
        return feature, risk, feature_backward, risk_backward, terms, identity

    def after_image(self, result, elapsed):
        _require(self.observer is not None and self._image is not None,
                 "No active image observation")
        try:
            _require(isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
                     and math.isfinite(elapsed) and elapsed >= 0,
                     "Instrumented elapsed time must be finite and non-negative")
            calls = deepcopy(self.observer.calls)
            backwards = deepcopy(self.observer.backward_events)
            feature, risk, feature_backward, risk_backward, terms, identity = (
                self._validate_observation(result.diagnostics, calls, backwards))
            allocated = reserved = None
            if self.device.type == "cuda":
                import torch

                torch.cuda.synchronize(self.device)
                allocated = torch.cuda.max_memory_allocated(self.device)
                reserved = torch.cuda.max_memory_reserved(self.device)
            cost = dict(
                position=self._image["position"], image_id=self._image["image_id"],
                source_sha256=self._image["sha256"],
                checkpoint_sha256=self._loading["checkpoint_sha256"],
                actual_loaded_effective_sha256=self._loading["actual_loaded_effective_sha256"],
                attack_seed=self.seed + self.offsets[self._image["image_id"]],
                adapter_forward_events=calls, input_gradient_events=backwards,
                logical_gradients=len(backwards), actual_logical_updates=len(backwards),
                registered_iterations=self.parameters["iterations"],
                detector_initialization_forward_attempts=len(risk),
                detector_input_backwards=len(risk_backward),
                feature_partial_input_backwards=len(feature_backward),
                clean_reference_image_rows=len(feature),
                adversarial_differentiable_views=len(feature),
                batched_feature_image_rows=sum(row["batch_rows"] for row in feature),
                feature_terms_from_trajectory=terms, identity_output=identity,
                feature_backward_depth=("backbone_not_neck_or_detection_head"
                    if self.background_variant is not None and self.parameters["surface"] == "backbone"
                    else "backbone_plus_neck_not_detection_head"),
                row_role_assignment="frozen_code_cat_clean_detached_then_adversarial",
                synchronized_instrumented_attack_seconds=elapsed,
                timing_scope="runner_timed_attack_with_read_only_observation_overhead",
                elapsed_synchronization_owner="attack_runner_selected_device",
                base_allocated_bytes=self._base_allocated,
                peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                observation_scope="read_only_adapter_hooks_not_calibrated_kernel_or_FLOP_count",
                time_is_paired_performance=False,
                calibrated_whole_detector_BE=None, FLOPs=None,
                **self.metadata(),
            )
            append_jsonl(self.run_dir / "native_cost.jsonl", cost)
            self.whole_run_cuda_peaks()
        finally:
            self.observer.close()
            self._image = None

    def close_observer(self):
        if self.observer is not None:
            self.observer.close()
        self.observer = None
        self._image = None
