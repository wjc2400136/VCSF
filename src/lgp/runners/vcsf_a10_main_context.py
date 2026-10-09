"""Explicit per-wave worker capability for the unchanged native VCSF builder."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time

from ..io import append_jsonl, atomic_json, file_digest
from .vcsf_a10_main_contract import (
    canonical, no_acceptance, read_json, reference, require, safe_child, verify_admission)
from .vcsf_gpu_context_owner import RegisteredGpuOwner


def process_birth(pid):
    base = Path("/proc") / str(pid)
    fields = (base / "stat").read_text().rsplit(") ", 1)[1].split()
    raw = (base / "cmdline").read_bytes()
    return dict(pid=pid, ppid=int(fields[1]), pgid=int(fields[2]),
        start_ticks=int(fields[19]), command_sha256=hashlib.sha256(raw).hexdigest())


def loaded_state_equality(model, checkpoint, dataset, model_id):
    """Check actual loaded tensors, including the exact released buffer/head exceptions."""
    import torch
    from ..modeling import finite_state_dict_audit
    from .cost_calibration import validate_mask_exclusions
    raw = torch.load(str(checkpoint), map_location="cpu")
    expected = raw.get("state_dict", raw)
    finite = finite_state_dict_audit(expected)
    actual = model.state_dict()
    excluded = set(expected) - set(actual)
    mask = {key for key in excluded if key.startswith("roi_head.mask_head.")}
    if mask:
        require(dataset == "coco" and model_id == "mask_rcnn_swin_t",
                "undeclared unused checkpoint head")
        validate_mask_exclusions(sorted(mask), model_id)
    buffers = excluded - mask
    require(buffers <= {"data_preprocessor.mean", "data_preprocessor.std"},
            "unexpected checkpoint tensor keys")
    for key in sorted(buffers):
        name = key.rsplit(".", 1)[1]
        module = model.data_preprocessor
        value = getattr(module, name)
        require(name in module._non_persistent_buffers_set
            and value.dtype == expected[key].dtype and value.shape == expected[key].shape
            and torch.equal(value.detach().cpu(), expected[key]), "nonpersistent preprocessing buffer differs")
    require(set(actual) == set(expected) - excluded, "missing loaded detector tensor")
    for key in sorted(actual):
        value = actual[key].detach().cpu()
        require(value.dtype == expected[key].dtype and value.shape == expected[key].shape
            and torch.equal(value, expected[key]), "loaded detector tensor differs from registered checkpoint")
    return dict(exact_loaded_tensor_equality=True, finite=finite,
        excluded_mask_head_keys=sorted(mask), separately_verified_nonpersistent_buffers=sorted(buffers))


def normalized_config_hash(config):
    from copy import deepcopy
    from .public_config_helpers import _normal
    from .public_config_helpers import normalized_model_config
    value = deepcopy(config.to_dict())
    value.pop("work_dir", None)
    return canonical(normalized_model_config(_normal(value)))


class MainExecutionContext:
    """No context is ambient; every native call carries this exact checked capability."""

    def __init__(self, registry, request_ref, owner):
        require(type(owner) is RegisteredGpuOwner, "worker needs the registered GPU-owner protocol")
        self.registry = registry
        self.root = registry.root
        self.request_ref = dict(request_ref)
        self.request = reference(self.request_ref)
        self.plan_ref = self.request["plan_reference"]
        self.admission_ref = self.request["admission_reference"]
        self.plan = reference(self.plan_ref)
        self.admission, self.inputs = verify_admission(registry, self.plan, self.plan_ref,
            self.admission_ref, current=True)
        self.owner = owner
        self.birth = process_birth(os.getpid())
        self.output = Path(self.request["output"]).absolute()
        self.slot = self.request["worker_slot"]
        self.assigned = self.plan["lanes"][self.slot]
        self.jobs = {job["job_id"]: job for job in self.plan["jobs"]}
        require(self.request["schema"] == "vcsf_a10_native_main_worker_v1"
            and self.request["job_ids"] == self.assigned
            and self.request["coordinator_pid"] == os.getppid()
            and self.request["coordinator_start_ticks"] == process_birth(os.getppid())["start_ticks"]
            and self.request["gpu_uuid"] == self.admission["physical_gpu_uuids"][self.slot]
            and self.request["physical_device"] == self.plan["devices"][self.slot]
            and owner._request == self.request
            and owner.receipt["worker_pid"] == self.birth["pid"]
            and owner.receipt["worker_pid_start_ticks"] == self.birth["start_ticks"],
            "worker/source assignment or parent/GPU ownership differs")
        category = "diagnostics" if self.plan["diagnostic_only"] else "experiments"
        expected_output = self.root / "outputs" / category / self.plan["protocol"] / self.plan["wave_id"]
        require(self.output == expected_output and self.output.resolve() == self.output,
                "worker escaped the immutable wave leaf")
        self.selected = self.inputs["images"][:self.plan["images_per_job"]]
        self.ids = [r["image_id"] for r in self.selected]
        self.rows = {r["image_id"]: r for r in self.selected}
        self.offsets = {r["image_id"]: i for i, r in enumerate(self.selected)}
        self.job = None
        self.observer = None
        self.check_runtime()

    def check_runtime(self):
        require(process_birth(os.getpid()) == self.birth
            and process_birth(os.getppid())["start_ticks"] == self.request["coordinator_start_ticks"],
            "worker or parent process identity changed")
        require(reference(self.request_ref) == self.request and reference(self.plan_ref) == self.plan,
                "worker request or wave plan changed")
        admission = reference(self.admission_ref)
        require(admission == self.admission, "wave admission changed")
        from datetime import datetime, timezone
        require(datetime.now(timezone.utc) < datetime.fromisoformat(admission["expires_utc"]),
                "wave admission expired")
        self.owner.verify()

    def select_job(self, job_id):
        self.check_runtime()
        require(type(job_id) is int and job_id in self.assigned, "job is not owned by this worker")
        from ..attacks.vcsf_public import verify_public_identity
        identity = verify_public_identity(self.root, self.registry.attack("vcsf"))
        require(identity["source_manifest_sha256"] == self.plan["source_manifest_sha256"],
                "execution source changed between complete jobs")
        self.job = self.jobs[job_id]
        self.group = safe_child(self.output, "groups/{:06d}".format(job_id))
        require(not self.group.exists(), "complete source job already has an immutable leaf")
        self.group.mkdir(parents=True, exist_ok=False)
        self.attack_dir = self.group / "attack"
        self.generated = None
        self._peak_allocated_bytes = 0
        self._peak_reserved_bytes = 0
        return self.group

    def guard_attack(self, arguments):
        self.check_runtime()
        require(self.job is not None, "worker has no selected complete source job")
        expected = dict(dataset_id=self.plan["dataset"], source_id=self.job["source"], attack_id="vcsf",
            split=self.plan["split"], output_dir=self.attack_dir, max_images=self.plan["max_images"],
            seed=self.job["seed"], device="cuda:0", strict=True, keep_going=False, download_weights=False,
            budget_profile=self.plan["budget_profile"], image_ids=self.ids, seed_offsets=self.offsets,
            parameter_overrides=None, input_transform=None,
            run_metadata=self.metadata())
        require(set(arguments) == set(expected)
            and all(type(arguments[k]) is type(v) and arguments[k] == v for k, v in expected.items()),
            "native invocation differs from its exact admitted complete source job")

    def metadata(self):
        return dict(protocol=self.plan["protocol"], wave_id=self.plan["wave_id"],
            job_id=self.job["job_id"], plan_sha256=self.plan_ref["sha256"],
            admission_sha256=self.admission_ref["sha256"], request_sha256=self.request_ref["sha256"],
            source_manifest_sha256=self.plan["source_manifest_sha256"],
            parameters_sha256=self.plan["parameters_sha256"], worker_slot=self.slot, **no_acceptance())

    def loaded(self, model, config, checkpoint, target, directory):
        self.check_runtime()
        expected = self.inputs["current_clean_inputs"]["target_checkpoint_sha256"][target]
        require(file_digest(checkpoint) == expected, "loader checkpoint bytes differ")
        from ..runtime_config import build_runtime_config
        from .public_config_helpers import verify_loaded_config
        prepared = build_runtime_config(self.registry.model(target), self.registry.dataset(self.plan["dataset"]),
            directory, mode="test", test_split=self.plan["split"], dump_config=False)
        expected_config = self.inputs["upstream_configs"][target]["normalized_effective_sha256"]
        require(normalized_config_hash(prepared) == expected_config,
                "prepared source/target effective configuration differs")
        effective = verify_loaded_config(config, prepared)
        loading = loaded_state_equality(model, checkpoint, self.plan["dataset"], target)
        path = directory / "native_loading.json"
        require(not path.exists(), "loading evidence already exists")
        atomic_json(path, dict(target=target, checkpoint_sha256=expected,
            prepared_normalized_effective_sha256=expected_config,
            actual_loaded_effective_sha256=effective, loading=loading, **self.metadata()))
        self.check_runtime()

    def observe_adapter(self, adapter):
        from .public_observation import ObservedAdapter
        self.observer = ObservedAdapter(adapter)
        return self.observer

    def before_image(self, position, image_id, source_path):
        self.check_runtime()
        row = self.selected[position]
        require(image_id == row["image_id"] and file_digest(source_path) == row["sha256"],
                "generation clean image identity changed")
        require(self.observer is not None, "native cost observer is absent")
        self.observer.close()
        self.observer.calls.clear()
        self.observer.backward_events.clear()
        self._image = image_id
        import torch
        torch.cuda.synchronize(0)
        self._base_allocated = torch.cuda.memory_allocated(0)
        self.whole_run_cuda_peaks()
        torch.cuda.reset_peak_memory_stats(0)

    def whole_run_cuda_peaks(self):
        import torch
        self._peak_allocated_bytes = max(getattr(self, "_peak_allocated_bytes", 0),
            torch.cuda.max_memory_allocated(0))
        self._peak_reserved_bytes = max(getattr(self, "_peak_reserved_bytes", 0),
            torch.cuda.max_memory_reserved(0))
        return dict(peak_allocated_bytes=self._peak_allocated_bytes,
            peak_reserved_bytes=self._peak_reserved_bytes,
            scope="whole_native_generation_including_inter_image_postprocessing")

    def after_image(self, result, elapsed):
        import torch
        self.check_runtime()
        calls = list(self.observer.calls)
        backwards = list(self.observer.backward_events)
        feature = [r for r in calls if r["kind"] == "feature"]
        risk = [r for r in calls if r["kind"] != "feature"]
        feature_backward = [r for r in backwards if r["kind"] == "feature"]
        risk_backward = [r for r in backwards if r["kind"] != "feature"]
        require(all(r["finite"] for r in backwards), "non-finite observed input gradient")
        declared = int(result.diagnostics[0].get("actual_gradient_evaluations", 0)) if (
            result.diagnostics and result.diagnostics[0].get("empty_target_behavior") == "identity_output"
        ) else int(self.registry.attack("vcsf").parameters["iterations"])
        require(len(backwards) == declared and len(feature_backward) == len(feature)
            and all(r["batch_rows"] == 2 for r in feature)
            and len(risk_backward) <= len(risk),
            "native observed cost and logical-gradient schedule differ")
        terms = sum(r.get("feature_surface_level_count", 0)
            for r in result.diagnostics if r.get("phase") == "feature")
        if declared:
            refine = self.registry.attack("vcsf").parameters["iterations"] - 1
            require(len(result.diagnostics) == declared and len(feature) == refine
                and len(risk_backward) == 1 and len(risk) in (1, 2)
                and all(r["batch_rows"] == 1 and r["requires_grad"] for r in risk)
                and all(r["requires_grad"] for r in feature)
                and terms == refine * 2 * self.registry.attack("vcsf").parameters["levels_per_stage"],
                "native initialization/refinement trajectory or cost schedule changed")
        else:
            require(not calls and not backwards and terms == 0,
                    "identity output performed undeclared differentiable work")
        cost = dict(dataset=self.plan["dataset"], split=self.plan["split"], source=self.job["source"],
            image_id=self._image, attack_seed=self.job["seed"] + self.offsets[self._image],
            adapter_forward_events=calls, input_gradient_events=backwards,
            logical_gradients=declared, detector_initialization_forward_attempts=len(risk),
            detector_input_backwards=len(risk_backward), feature_partial_input_backwards=len(feature_backward),
            clean_reference_image_rows=len(feature), adversarial_differentiable_views=len(feature),
            batched_feature_image_rows=sum(r["batch_rows"] for r in feature),
            feature_terms_from_trajectory=terms,
            feature_backward_depth="backbone_plus_neck_not_detection_head",
            synchronized_instrumented_attack_seconds=elapsed,
            base_allocated_bytes=self._base_allocated,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
            observation_scope="read_only_adapter_hooks_not_calibrated_kernel_or_FLOP_count",
            time_is_paired_performance=False, calibrated_whole_detector_BE=None, FLOPs=None,
            **self.metadata())
        append_jsonl(self.attack_dir / "native_cost.jsonl", cost)
        self.whole_run_cuda_peaks()
        self.observer.close()

    def close_observer(self):
        if self.observer is not None:
            self.observer.close()
        self.observer = None

    def guard_evaluation(self, registry, dataset, split, target, adversarial_run, output_dir, image_ids, device,
                         max_images, download_weights, keep_going, save_visualizations,
                         prediction_archive, target_checkpoint):
        self.check_runtime()
        require(registry is self.registry and dataset == self.plan["dataset"] and split == self.plan["split"]
            and self.generated is not None and target in self.job["targets"]
            and adversarial_run == self.attack_dir and output_dir == self.group / "evaluations" / target
            and image_ids == self.ids and device == "cuda:0" and max_images is None
            and download_weights is False and keep_going is False and save_visualizations is False
            and prediction_archive == "gzip" and target_checkpoint is None,
            "native target evaluation differs from complete registered source job")
        require(file_digest(self.attack_dir / "run.json") == self.generated["run_sha256"],
                "completed source generation changed")
        for filename, digest in self.generated["sealed_generation_files"].items():
            require(file_digest(safe_child(self.attack_dir, filename, True)) == digest,
                    "generation seal changed before target evaluation")

    def before_evaluation_image(self, image_id, image_path):
        self.check_runtime()
        require(image_id in self.generated["pixels"]
            and file_digest(image_path) == self.generated["pixels"][image_id],
            "adversarial PNG changed before evaluation")

    def progress(self, completed, total, image_id, status):
        self.check_runtime()
        require(status == "ok", "source generation or target prediction failed")
        atomic_json(self.group / "progress.json", dict(completed_images=completed, total_images=total, last_image_id=image_id,
            updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            **self.metadata()))
