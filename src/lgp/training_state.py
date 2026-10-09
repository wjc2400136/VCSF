"""Training primitives for the controlled victim-state checkpoint pair.

The public training entry point imports this module explicitly before MMEngine
builds its runtime config.  Nothing here changes the maintained attack
implementations or the detector registry.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Set

import torch
from mmdet.models.detectors import FasterRCNN
from mmdet.registry import MODELS
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from .io import atomic_json


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _shape(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    return value


@MODELS.register_module()
class LGPTrainingStateFasterRCNN(FasterRCNN):
    """Faster R-CNN with one frozen paired continuation training step.

    Both branches preprocess a minibatch once and make the same number of
    weight updates.  The standard control reuses the clean tensor; the
    adversarial branch uses a zero-initialized, per-minibatch L-infinity
    perturbation updated from total detector loss.  The input perturbation is
    never retained as benchmark evidence and never touches val2017 while
    training.
    """

    def __init__(
        self,
        training_state_mode: str,
        replay_steps_per_minibatch: int,
        epsilon_pixel_255: float,
        step_size_pixel_255: float,
        padding_policy: str,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if training_state_mode not in {
            "standard_control",
            "adversarial_training",
        }:
            raise ValueError("Unsupported training_state_mode")
        if int(replay_steps_per_minibatch) <= 0:
            raise ValueError("replay_steps_per_minibatch must be positive")
        if float(epsilon_pixel_255) <= 0 or float(step_size_pixel_255) <= 0:
            raise ValueError("Adversarial pixel budgets must be positive")
        if padding_policy != "zero_outside_img_shape":
            raise ValueError("Unsupported adversarial padding policy")
        self.training_state_mode = str(training_state_mode)
        self.replay_steps_per_minibatch = int(replay_steps_per_minibatch)
        self.epsilon_pixel_255 = float(epsilon_pixel_255)
        self.step_size_pixel_255 = float(step_size_pixel_255)
        self.padding_policy = str(padding_policy)
        self._sample_trace = hashlib.sha256()
        self._outer_minibatches = 0
        self._sample_count = 0
        self._optimizer_steps = 0
        self._last_image_ids = []
        self._seen_image_ids: Set[int] = set()
        self._duplicate_samples = 0

    def _trace_samples(self, data_samples: Sequence[Any]) -> None:
        rows = []
        for position, sample in enumerate(data_samples):
            meta = dict(sample.metainfo)
            row = {
                "position": position,
                "img_id": int(meta["img_id"]),
                "ori_shape": _shape(meta.get("ori_shape")),
                "img_shape": _shape(meta.get("img_shape")),
                "scale_factor": _shape(meta.get("scale_factor")),
                "flip": bool(meta.get("flip", False)),
                "flip_direction": meta.get("flip_direction"),
            }
            rows.append(row)
            if row["img_id"] in self._seen_image_ids:
                self._duplicate_samples += 1
            else:
                self._seen_image_ids.add(row["img_id"])
        self._sample_trace.update(_canonical_bytes(rows) + b"\n")
        self._outer_minibatches += 1
        self._sample_count += len(rows)
        self._last_image_ids = [row["img_id"] for row in rows]

    @staticmethod
    def _valid_pixel_mask(
        inputs: torch.Tensor, data_samples: Sequence[Any]
    ) -> torch.Tensor:
        if inputs.ndim != 4 or len(data_samples) != int(inputs.shape[0]):
            raise RuntimeError("Training batch and sample metadata do not align")
        mask = torch.zeros(
            (int(inputs.shape[0]), 1, int(inputs.shape[-2]), int(inputs.shape[-1])),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        for index, sample in enumerate(data_samples):
            shape = sample.metainfo.get("img_shape")
            if isinstance(shape, torch.Tensor):
                shape = shape.tolist()
            if not isinstance(shape, (list, tuple)) or len(shape) < 2:
                raise RuntimeError("Training sample has no valid img_shape")
            height, width = int(shape[0]), int(shape[1])
            if (
                height <= 0
                or width <= 0
                or height > int(inputs.shape[-2])
                or width > int(inputs.shape[-1])
            ):
                raise RuntimeError("Training sample img_shape exceeds the padded batch")
            mask[index, :, :height, :width] = 1
        return mask

    def _loss_step(
        self,
        inputs: torch.Tensor,
        data_samples: Sequence[Any],
        optim_wrapper: Any,
    ) -> Dict[str, torch.Tensor]:
        with optim_wrapper.optim_context(self):
            losses = self._run_forward(
                {"inputs": inputs, "data_samples": data_samples},
                mode="loss",
            )
        parsed_loss, log_vars = self.parse_losses(losses)
        optim_wrapper.update_params(parsed_loss)
        self._optimizer_steps += 1
        return log_vars

    def _adversarial_loss_step(
        self,
        base_pixel: torch.Tensor,
        delta_pixel: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        valid_pixel_mask: torch.Tensor,
        data_samples: Sequence[Any],
        optim_wrapper: Any,
    ) -> Any:
        delta_pixel = delta_pixel * valid_pixel_mask
        adv_pixel = torch.clamp(base_pixel + delta_pixel, 0.0, 255.0)
        adv_pixel = adv_pixel.detach().requires_grad_(True)
        adv_inputs = (adv_pixel - mean) / std
        with optim_wrapper.optim_context(self):
            losses = self._run_forward(
                {"inputs": adv_inputs, "data_samples": data_samples},
                mode="loss",
            )
        parsed_loss, log_vars = self.parse_losses(losses)
        # One backward pass supplies both the detector-parameter gradients and
        # the leaf input gradient.  ``update_params`` only clears optimizer
        # parameter gradients; it does not clear ``adv_pixel.grad``.
        optim_wrapper.update_params(parsed_loss)
        self._optimizer_steps += 1
        if adv_pixel.grad is None:
            raise RuntimeError("Adversarial training produced no input gradient")
        input_grad = adv_pixel.grad.detach() * valid_pixel_mask
        next_delta = delta_pixel + self.step_size_pixel_255 * input_grad.sign()
        next_delta = torch.clamp(
            next_delta, -self.epsilon_pixel_255, self.epsilon_pixel_255
        )
        next_delta = torch.clamp(base_pixel + next_delta, 0.0, 255.0) - base_pixel
        next_delta = next_delta * valid_pixel_mask
        return log_vars, next_delta.detach()

    def train_step(self, data: Any, optim_wrapper: Any) -> Dict[str, torch.Tensor]:
        processed = self.data_preprocessor(data, training=True)
        inputs = processed["inputs"]
        data_samples = processed["data_samples"]
        self._trace_samples(data_samples)
        log_vars: Dict[str, torch.Tensor] = {}
        if self.training_state_mode == "standard_control":
            for _ in range(self.replay_steps_per_minibatch):
                log_vars = self._loss_step(inputs, data_samples, optim_wrapper)
            return log_vars

        mean = self.data_preprocessor.mean.to(inputs)
        std = self.data_preprocessor.std.to(inputs)
        base_pixel = (inputs.detach() * std + mean).detach()
        delta_pixel = torch.zeros_like(base_pixel)
        valid_pixel_mask = self._valid_pixel_mask(inputs, data_samples)
        for _ in range(self.replay_steps_per_minibatch):
            log_vars, delta_pixel = self._adversarial_loss_step(
                base_pixel,
                delta_pixel,
                mean,
                std,
                valid_pixel_mask,
                data_samples,
                optim_wrapper,
            )
        return log_vars

    def training_state_audit(self, final: bool = False) -> Dict[str, Any]:
        payload = {
            "training_state": self.training_state_mode,
            "processed_sample_trace_sha256": self._sample_trace.copy().hexdigest(),
            "outer_minibatches": int(self._outer_minibatches),
            "samples": int(self._sample_count),
            "unique_samples": len(self._seen_image_ids),
            "duplicate_samples": int(self._duplicate_samples),
            "optimizer_steps": int(self._optimizer_steps),
            "last_image_ids": list(self._last_image_ids),
            "replay_steps_per_minibatch": self.replay_steps_per_minibatch,
            "epsilon_pixel_255": self.epsilon_pixel_255,
            "step_size_pixel_255": self.step_size_pixel_255,
            "padding_policy": self.padding_policy,
        }
        if final:
            payload["processed_unique_image_ids_sha256"] = hashlib.sha256(
                _canonical_bytes(sorted(self._seen_image_ids))
            ).hexdigest()
        return payload


@HOOKS.register_module()
class LGPTrainingStateProgressHook(Hook):
    """Persist monitorable state without selecting or evaluating checkpoints."""

    priority = "VERY_LOW"

    def __init__(self, output: str, interval: int = 50) -> None:
        self.output = Path(output).resolve()
        self.interval = max(1, int(interval))

    @staticmethod
    def _model(runner: Any) -> LGPTrainingStateFasterRCNN:
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        if not hasattr(model, "training_state_audit"):
            raise RuntimeError("Training-state progress hook received the wrong model")
        return model

    def _write(self, runner: Any, status: str, phase: str) -> None:
        model = self._model(runner)
        payload = {
            "schema_version": 1,
            "status": status,
            "phase": phase,
            "epoch_zero_based": int(runner.epoch),
            "outer_iteration_zero_based": int(runner.iter),
            **model.training_state_audit(final=status == "complete"),
        }
        atomic_json(self.output, payload)

    def before_train(self, runner: Any) -> None:
        self._write(runner, "running", "training")

    def after_train_iter(
        self,
        runner: Any,
        batch_idx: int,
        data_batch: Any = None,
        outputs: Any = None,
    ) -> None:
        if self.every_n_train_iters(runner, self.interval):
            self._write(runner, "running", "training")

    def after_train_epoch(self, runner: Any) -> None:
        self._write(runner, "running", "checkpointing_fixed_final_epoch")

    def after_train(self, runner: Any) -> None:
        self._write(runner, "complete", "training_complete")
