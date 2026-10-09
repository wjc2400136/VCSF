# Recorded LGP/OpenMMLab-derived portions: Copyright 2018-2023 OpenMMLab.
# Apache-2.0 text: docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt.
# Source subtree: liguopeng0923/LGP@fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet.
# Modified for the local benchmark; prominent notice added 2026-10-09.
# Earlier adaptations predate this notice; execution logic is unchanged.
# Attribution covers recorded derived portions, not every line of this file.

"""LGP-private formula components, isolated pending full attack qualification."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LGPImageCoordinates:
    mean: torch.Tensor
    std: torch.Tensor
    channel_conversion: bool

    @classmethod
    def from_preprocessor(cls, preprocessor, reference):
        if reference.ndim != 4 or reference.shape[1] != 3:
            raise ValueError("LGP requires Bx3xHxW raw BGR input")
        if not hasattr(preprocessor, '_enable_normalize') or not hasattr(preprocessor, '_channel_conversion'):
            raise ValueError("LGP requires an explicit detector normalization contract")
        if bool(preprocessor._enable_normalize):
            mean = preprocessor.mean.detach().to(reference).reshape(1, 3, 1, 1)
            std = preprocessor.std.detach().to(reference).reshape(1, 3, 1, 1)
        else:
            mean = reference.new_zeros((1, 3, 1, 1))
            std = reference.new_ones((1, 3, 1, 1))
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all()):
            raise ValueError("LGP normalization must be finite")
        if not bool((std > 0).all()):
            raise ValueError("LGP normalization standard deviations must be positive")
        return cls(mean.clone(), std.clone(), bool(preprocessor._channel_conversion))

    def normalize(self, raw_bgr):
        ordered = raw_bgr[:, [2, 1, 0]] if self.channel_conversion else raw_bgr
        return (ordered - self.mean) / self.std

    def pixels(self, normalized):
        ordered = normalized * self.std + self.mean
        return ordered[:, [2, 1, 0]] if self.channel_conversion else ordered


def lgp_shape_loss(widths_heights, shape_scale):
    if widths_heights.ndim != 2 or widths_heights.shape[1] != 2:
        raise ValueError("LGP shape terms require Nx2 widths and heights")
    if widths_heights.shape[0] == 0:
        raise ValueError("LGP shape terms require at least one tracked proposal")
    # Preserve the current target gradient while correcting only the reduction.
    terms = F.smooth_l1_loss(
        widths_heights, widths_heights * shape_scale, reduction='none'
    )
    return terms.sum(dim=1).mean()


def lgp_perceptibility_loss(delta, background, failed_foreground, heatmap, mode):
    if delta.ndim != 4 or delta.shape[0] != 1:
        raise ValueError("LGP perceptibility requires one BxCxHxW image")
    for weight in (background, failed_foreground, heatmap):
        if weight.shape != (1, 1, delta.shape[2], delta.shape[3]):
            raise ValueError("LGP heatmaps must match the valid model-image region")
        if not bool(torch.isfinite(weight).all() and (weight >= 0).all()):
            raise ValueError("LGP heatmaps must be finite and nonnegative")
    if mode == 'image':
        return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='sum')
    if mode == 'image_l2':
        return F.smooth_l1_loss(delta, torch.zeros_like(delta), reduction='sum') + 0.1 * torch.linalg.vector_norm(delta)
    if mode not in {'static_fbs', 'adaptive_fbs'}:
        raise ValueError("Unknown LGP perceptibility mode")
    bg_delta = delta * background
    fg_delta = delta * failed_foreground
    background_distance = F.smooth_l1_loss(bg_delta, torch.zeros_like(bg_delta), reduction='sum')
    foreground_distance = F.smooth_l1_loss(fg_delta, torch.zeros_like(fg_delta), reduction='sum')
    # Eq.8 uses the complete Eq.7 heatmap, without an extra foreground mask.
    return background_distance + foreground_distance + 0.1 * torch.linalg.vector_norm(delta * heatmap)
