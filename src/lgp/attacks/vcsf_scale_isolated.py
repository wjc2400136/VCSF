"""Single-source scale-operator research; frozen and retired classes stay intact."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .vcsf_research_isolated import VCSFResearchCandidate, VCSFResearchConfig, reflect_indices


@dataclass(frozen=True)
class VCSFScaleConfig(VCSFResearchConfig):
    levels_per_stage: int = 2
    image_interpolation: str = "bilinear"
    placement: str = "random"
    image_padding: str = "reflect"

    def is_reference(self):
        # The parent's default shortcut constructs the frozen six-level method.
        return False

    def validate(self):
        super().validate()
        reference = asdict(VCSFResearchConfig(levels_per_stage=2))
        current = asdict(self)
        for key, expected in reference.items():
            if key not in ("scale_min", "scale_max") and current[key] != expected:
                raise ValueError("Scale study fixes the four-level working baseline: " + key)
        combinations = {
            ("bilinear", "random", "reflect"),
            ("bilinear", "center", "reflect"),
            ("bilinear", "random", "zero"),
            ("nearest", "random", "reflect"),
        }
        if (self.image_interpolation, self.placement, self.image_padding) not in combinations:
            raise ValueError("Unregistered joint scale-operator replacement")


class VCSFScaleCandidate(VCSFResearchCandidate):
    implementation_path = "src/lgp/attacks/vcsf_scale_isolated.py"
    method_name = "VCSF isolated four-level scale study"

    def _resize_crop_pad(self, tensor, new_height, new_width, crop_top, crop_left,
            pad_left, pad_right, pad_top, pad_bottom, *, is_mask=False):
        arguments = (new_height, new_width, crop_top, crop_left,
            pad_left, pad_right, pad_top, pad_bottom)
        if is_mask or (self.config.image_interpolation == "bilinear"
                and self.config.image_padding == "reflect"):
            return VCSFResearchCandidate._resize_crop_pad(tensor, *arguments, is_mask=is_mask)
        mode = self.config.image_interpolation
        options = {"align_corners": False} if mode == "bilinear" else {}
        output = F.interpolate(tensor, size=(new_height, new_width), mode=mode, **options)
        if new_height >= tensor.shape[-2]:
            output = output[..., crop_top:crop_top + tensor.shape[-2], :]
        if new_width >= tensor.shape[-1]:
            output = output[..., crop_left:crop_left + tensor.shape[-1]]
        if pad_left or pad_right or pad_top or pad_bottom:
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            if self.config.image_padding == "zero":
                output = F.pad(output, padding, mode="constant", value=0.)
            elif max(pad_top, pad_bottom) < output.shape[-2] and max(pad_left, pad_right) < output.shape[-1]:
                output = F.pad(output, padding, mode="reflect")
            else:
                rows = reflect_indices(output.shape[-2], pad_top, pad_bottom, output.device)
                columns = reflect_indices(output.shape[-1], pad_left, pad_right, output.device)
                output = output.index_select(-2, rows).index_select(-1, columns)
        return output

    def _same_scale_view(self, clean, adversarial, mask, generator, clean_generator=None):
        if self.config.placement == "random":
            return super()._same_scale_view(clean, adversarial, mask, generator, clean_generator)
        # Consume the same scale and four offset draws before fixing the offsets.
        _, sampled = self._geometry(clean, generator)
        new_height, new_width = sampled[:2]
        height, width = clean.shape[-2:]
        pad_height, pad_width = max(height - new_height, 0), max(width - new_width, 0)
        pad_top, pad_left = pad_height // 2, pad_width // 2
        geometry = (new_height, new_width, max(new_height - height, 0) // 2,
            max(new_width - width, 0) // 2, pad_left, pad_width - pad_left,
            pad_top, pad_height - pad_top)
        return (self._resize_crop_pad(clean, *geometry).clamp(0., 255.),
            self._resize_crop_pad(adversarial, *geometry).clamp(0., 255.),
            self._resize_crop_pad(mask, *geometry, is_mask=True).clamp(0., 1.))

    def _feature_loss(self, clean, adversarial, global_mask, generator, clean_generator=None):
        loss, stats = super()._feature_loss(clean, adversarial, global_mask, generator, clean_generator)
        stats.update(image_interpolation=self.config.image_interpolation,
            placement=self.config.placement, image_padding=self.config.image_padding,
            mask_interpolation="nearest", mask_padding="zero",
            offset_random_draws_preserved=True)
        return loss, stats
