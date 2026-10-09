from __future__ import annotations

import io
import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from ..attacks.common import parse_fraction


def _number(value: Any) -> float:
    parsed = parse_fraction(value)
    if not isinstance(parsed, (int, float)) or not math.isfinite(float(parsed)):
        raise ValueError("Preprocessing parameter must be finite")
    return float(parsed)


def _pil_forward(rgb: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """Exact uint8 Pillow forward used by the deployed preprocessing path."""
    rows = []
    kind = str(spec["kind"])
    for image in rgb:
        array = (
            image.detach()
            .permute(1, 2, 0)
            .round()
            .clamp(0.0, 255.0)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        pil = Image.fromarray(array, mode="RGB")
        if kind == "jpeg":
            buffer = io.BytesIO()
            pil.save(
                buffer,
                format="JPEG",
                quality=int(spec["quality"]),
                subsampling=0,
                optimize=False,
                progressive=False,
            )
            buffer.seek(0)
            with Image.open(buffer) as encoded:
                output = encoded.convert("RGB").copy()
        elif kind == "median":
            output = pil.filter(ImageFilter.MedianFilter(size=int(spec["size"])))
        elif kind == "gaussian":
            output = pil.filter(
                ImageFilter.GaussianBlur(radius=_number(spec["radius"]))
            )
        elif kind == "resize_roundtrip":
            scale = _number(spec["scale"])
            width, height = pil.size
            scaled_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            output = pil.resize(
                scaled_size, resample=Image.Resampling.BILINEAR
            ).resize((width, height), resample=Image.Resampling.BILINEAR)
        else:
            raise ValueError("No exact Pillow forward for {}".format(kind))
        values = np.asarray(output, dtype=np.float32).copy()
        rows.append(torch.from_numpy(values).permute(2, 0, 1))
    return torch.stack(rows).to(device=rgb.device, dtype=rgb.dtype)


def _gaussian_kernel(radius: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # Pillow's GaussianBlur is the frozen deployment transform.  This smooth,
    # normalized Gaussian is its declared backward surrogate only.
    sigma = max(float(radius), 1.0e-6)
    half = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-half, half + 1, device=device, dtype=dtype)
    kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    return kernel[:, None] * kernel[None, :]


def _surrogate(rgb: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    kind = str(spec["kind"])
    if kind in {"identity", "jpeg", "bit_depth", "median"}:
        return rgb
    if kind == "gaussian":
        kernel = _gaussian_kernel(_number(spec["radius"]), rgb.device, rgb.dtype)
        weight = kernel.expand(3, 1, *kernel.shape)
        pad = kernel.shape[-1] // 2
        return F.conv2d(F.pad(rgb, (pad, pad, pad, pad), mode="reflect"), weight, groups=3)
    if kind == "resize_roundtrip":
        scale = _number(spec["scale"])
        height, width = rgb.shape[-2:]
        scaled = F.interpolate(
            rgb,
            size=(max(1, round(height * scale)), max(1, round(width * scale))),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(scaled, size=(height, width), mode="bilinear", align_corners=False)
    raise ValueError("Unknown preprocessing kind: {}".format(kind))


def defense_forward(raw_bgr: torch.Tensor, spec: Mapping[str, Any]) -> torch.Tensor:
    """Differentiable defended-source forward with an exact frozen forward path.

    The tensor is BGR in [0, 255].  Non-differentiable variants use BPDA:
    exact uint8 deployment values in the forward pass and the predeclared
    surrogate gradient in the backward pass.  No result-dependent rule is
    selected at runtime.
    """
    if raw_bgr.ndim != 4 or raw_bgr.shape[1] != 3:
        raise ValueError("Expected Bx3xHxW raw BGR tensor")
    kind = str(spec["kind"])
    rgb = raw_bgr[:, [2, 1, 0], :, :]
    if kind == "identity":
        exact = rgb
        surrogate = rgb
    elif kind == "bit_depth":
        levels = float((1 << int(spec["bits"])) - 1)
        # The deployment path writes an 8-bit PNG after uniform-level
        # quantization, so the outer round is part of the exact forward too.
        exact = torch.round(
            torch.round(rgb * (levels / 255.0)) * (255.0 / levels)
        )
        surrogate = rgb
    elif kind in {"jpeg", "median"}:
        exact = _pil_forward(rgb, spec)
        surrogate = rgb
    elif kind in {"gaussian", "resize_roundtrip"}:
        exact = _pil_forward(rgb, spec)
        surrogate = _surrogate(rgb, spec)
    else:
        raise ValueError("Unknown preprocessing kind: {}".format(kind))
    # This algebraic order preserves the exact forward bits.  Writing
    # ``exact + surrogate - surrogate.detach()`` can introduce cancellation
    # noise even though the mathematical value is exact.
    defended = surrogate + (exact - surrogate).detach()
    return defended[:, [2, 1, 0], :, :].clamp(0.0, 255.0)


class AdaptivePreprocessor:
    def __init__(self, spec: Mapping[str, Any]) -> None:
        self.spec = dict(spec)

    def __call__(self, raw_bgr: torch.Tensor) -> torch.Tensor:
        return defense_forward(raw_bgr, self.spec)
