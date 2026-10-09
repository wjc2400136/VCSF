from __future__ import annotations

from dataclasses import fields
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar

import torch
import torch.nn.functional as F

from ..adapters.openmmlab import CandidateSet


ConfigT = TypeVar("ConfigT")


ADAMAX_RUNTIME_COMPATIBILITY: Dict[str, Any] = {
    "optimizer": "Adamax",
    "foreach": False,
    "implementation": "single_tensor",
    "semantic_change": False,
}


def build_adamax(
    parameters: Iterable[torch.Tensor],
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Adamax:
    """Build the method-prescribed Adamax with a stable execution backend.

    PyTorch 2.0 automatically selects its CUDA ``foreach`` implementation
    when the option is unspecified.  That implementation rejects the
    non-contiguous NCHW strides produced by the formal image loader during an
    internal reduction.  The single-tensor implementation applies the same
    Adamax equations and hyperparameters without imposing that stride
    restriction, so this is a runtime-compatibility choice rather than an
    attack parameter.
    """
    return torch.optim.Adamax(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        foreach=False,
    )


def parse_fraction(value: Any) -> Any:
    """Parse readable YAML numbers such as ``4/255`` without using eval."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    try:
        if "/" in stripped:
            numerator, denominator = stripped.split("/", 1)
            if not numerator.strip() or not denominator.strip():
                return value
            return float(Fraction(numerator.strip()) / Fraction(denominator.strip()))
        # PyYAML 6 treats some scientific-notation literals as strings.
        if any(character.isdigit() for character in stripped):
            return float(stripped)
    except (ValueError, ZeroDivisionError):
        return value
    return value


def config_from_mapping(config_type: Type[ConfigT], raw: Mapping[str, Any]) -> ConfigT:
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(
            "Unknown {} parameter(s): {}".format(
                config_type.__name__, ", ".join(unknown)
            )
        )
    converted = {key: parse_fraction(value) for key, value in raw.items()}
    return config_type(**converted)


def validate_linf_constraint(eps: float, iterations: int) -> None:
    if not 0.0 < eps <= 1.0:
        raise ValueError("eps must be expressed in normalized [0, 1] units")
    if iterations <= 0:
        raise ValueError("iterations must be positive")


def validate_linf(eps: float, iterations: int, step_size: float) -> None:
    validate_linf_constraint(eps, iterations)
    if not 0.0 < step_size <= eps:
        raise ValueError("step_size must be positive and no larger than eps")


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def random_linf_start(
    clean: torch.Tensor, eps: float, generator: torch.Generator
) -> torch.Tensor:
    radius = float(eps) * 255.0
    noise = torch.empty_like(clean).uniform_(-radius, radius, generator=generator)
    return (clean + noise).clamp(0.0, 255.0).detach()


def project_linf(
    clean: torch.Tensor, adversarial: torch.Tensor, eps: float
) -> torch.Tensor:
    radius = float(eps) * 255.0
    delta = (adversarial - clean).clamp(-radius, radius)
    return (clean + delta).clamp(0.0, 255.0)


def normalized_gradient(gradient: torch.Tensor) -> torch.Tensor:
    denominator = gradient.abs().mean(dim=(1, 2, 3), keepdim=True).clamp(min=1.0e-12)
    return gradient / denominator


def sign_step(
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    eps: float,
    minimize: bool,
) -> torch.Tensor:
    direction = -gradient.sign() if minimize else gradient.sign()
    updated = adversarial + float(step_size) * 255.0 * direction
    return project_linf(clean, updated, eps).detach()


def pairwise_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 or second.numel() == 0:
        return first.new_zeros((first.shape[0], second.shape[0]))
    left_top = torch.maximum(first[:, None, :2], second[None, :, :2])
    right_bottom = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0.0).prod(dim=-1)
    first_area = (first[:, 2:] - first[:, :2]).clamp(min=0.0).prod(dim=-1)
    second_area = (second[:, 2:] - second[:, :2]).clamp(min=0.0).prod(dim=-1)
    union = first_area[:, None] + second_area[None, :] - intersection
    return intersection / union.clamp(min=1.0e-8)


def background_log_probability(
    candidates: CandidateSet,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return the detector-native log probability of the no-object event.

    Softmax heads expose a real background class.  Sigmoid heads do not; for
    those heads the mathematically corresponding no-object event is that every
    independent foreground class is absent.  Dense objectness heads use their
    explicit objectness branch when it is available.  The final ``1-score``
    branch is only for heads whose public prediction surface exposes neither a
    class vector nor objectness.
    """
    if candidates.background_is_explicit:
        if candidates.background_scores is None:
            raise RuntimeError("Explicit-background candidates omitted background scores")
        return candidates.background_scores[indices].clamp(min=1.0e-8).log()
    if candidates.objectness is not None:
        objectness = candidates.objectness[indices].clamp(
            min=1.0e-8, max=1.0 - 1.0e-8
        )
        return torch.log1p(-objectness)
    if candidates.class_scores is not None:
        foreground = candidates.class_scores[indices].clamp(
            min=1.0e-8, max=1.0 - 1.0e-8
        )
        return torch.log1p(-foreground).sum(dim=-1)
    if candidates.background_scores is not None:
        return candidates.background_scores[indices].clamp(min=1.0e-8).log()
    foreground = candidates.scores[indices].clamp(
        min=1.0e-8, max=1.0 - 1.0e-8
    )
    return torch.log1p(-foreground)


def background_classification_loss(
    candidates: CandidateSet,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy/BCE no-object loss on a selected candidate set."""
    return -background_log_probability(candidates, indices).mean()


def lgp_background_semantic_loss(
    candidates: CandidateSet,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Return LGP's summed semantic objective for tracked proposals.

    Equation (5) and the pinned author release minimize the negative
    background *score* when a detector exposes an explicit background class.
    They use cross-entropy only for detectors without such a class.  Keeping
    these branches separate matters: ``-p_bg`` and ``-log(p_bg)`` have
    different gradients and are not interchangeable implementation details.
    """
    if candidates.background_is_explicit:
        if candidates.background_scores is None:
            raise RuntimeError(
                "Explicit-background candidates omitted background scores"
            )
        return -candidates.background_scores[indices].sum()
    return -background_log_probability(candidates, indices).sum()


def high_quality_indices(
    candidates: CandidateSet,
    gt_boxes: torch.Tensor,
    gt_labels: Optional[torch.Tensor] = None,
    by_iou: int = 5,
    by_score: int = 5,
) -> torch.Tensor:
    """Select the union of IoU- and class-score-ranked candidates per object."""
    if candidates.scores.numel() == 0:
        return candidates.labels.new_zeros((0,), dtype=torch.long)
    if gt_boxes.numel() == 0:
        count = min(max(by_iou + by_score, 1), candidates.scores.numel())
        return candidates.scores.topk(count, sorted=False).indices
    overlaps = pairwise_iou(gt_boxes, candidates.bboxes)
    selected: List[torch.Tensor] = []
    for object_index in range(gt_boxes.shape[0]):
        iou_count = min(max(int(by_iou), 0), candidates.scores.numel())
        if iou_count:
            selected.append(overlaps[object_index].topk(iou_count, sorted=False).indices)
        score_count = min(max(int(by_score), 0), candidates.scores.numel())
        if score_count:
            scores = candidates.scores
            if gt_labels is not None and object_index < gt_labels.numel():
                matching = candidates.labels == gt_labels[object_index]
                if matching.any():
                    scores = scores.masked_fill(~matching, -1.0)
            selected.append(scores.topk(score_count, sorted=False).indices)
    if not selected:
        return candidates.labels.new_zeros((0,), dtype=torch.long)
    return torch.unique(torch.cat(selected))


def positive_proposal_classification_scores(
    candidates: CandidateSet,
    gt_boxes: torch.Tensor,
    gt_labels: Optional[torch.Tensor],
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Return scores of all clean positive proposals for NAA/MLFAdv.

    The papers define positives by proposal IoU, not by an arbitrary top-k
    score subset.  ROI candidates expose the complete class distribution, so
    the score for the matched ground-truth class is used.  Dense heads whose
    public decoder exposes only the winning class score additionally require
    the winning label to match the ground truth; inventing missing logits would
    change the attack objective.
    """
    if candidates.scores.numel() == 0 or gt_boxes.numel() == 0:
        return candidates.scores.new_zeros((0,))
    if not 0.0 <= float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must lie in [0, 1]")
    overlaps = pairwise_iou(gt_boxes, candidates.bboxes)
    best_iou, best_gt = overlaps.max(dim=0)
    keep = best_iou >= float(iou_threshold)
    if gt_labels is not None and gt_labels.numel():
        if candidates.class_scores is not None:
            class_scores = candidates.class_scores
            foreground_count = class_scores.shape[-1]
            if candidates.background_is_explicit:
                foreground_count = max(foreground_count - 1, 1)
            labels = gt_labels[best_gt].clamp(min=0, max=foreground_count - 1)
            values = class_scores[:, :foreground_count].gather(
                1, labels[:, None]
            ).squeeze(1)
        else:
            keep = keep & (candidates.labels == gt_labels[best_gt])
            values = candidates.scores
    else:
        values = candidates.scores
    return values[keep]


def per_object_iou_assignments(
    candidates: CandidateSet,
    gt_boxes: torch.Tensor,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return top-IoU candidate and GT indices without collapsing duplicates."""
    if count <= 0 or candidates.scores.numel() == 0 or gt_boxes.numel() == 0:
        empty = candidates.labels.new_zeros((0,), dtype=torch.long)
        return empty, empty
    overlaps = pairwise_iou(gt_boxes, candidates.bboxes)
    limit = min(int(count), candidates.scores.numel())
    candidate_indices = overlaps.topk(limit, dim=1, sorted=True).indices.reshape(-1)
    gt_indices = torch.arange(
        gt_boxes.shape[0], device=gt_boxes.device, dtype=torch.long
    ).repeat_interleave(limit)
    return candidate_indices, gt_indices


def per_object_matching_label_assignments(
    candidates: CandidateSet,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NumbOD classification track: top-IoU boxes whose label matches GT."""
    if count <= 0 or candidates.scores.numel() == 0 or gt_boxes.numel() == 0:
        empty = candidates.labels.new_zeros((0,), dtype=torch.long)
        return empty, empty
    overlaps = pairwise_iou(gt_boxes, candidates.bboxes)
    selected_candidates: List[torch.Tensor] = []
    selected_gt: List[torch.Tensor] = []
    for gt_index in range(gt_boxes.shape[0]):
        matching = torch.nonzero(
            candidates.labels == gt_labels[gt_index], as_tuple=False
        ).flatten()
        if matching.numel() == 0:
            continue
        limit = min(int(count), matching.numel())
        local = overlaps[gt_index, matching].topk(limit, sorted=True).indices
        selected_candidates.append(matching[local])
        selected_gt.append(
            torch.full(
                (limit,), gt_index, dtype=torch.long, device=gt_boxes.device
            )
        )
    if not selected_candidates:
        empty = candidates.labels.new_zeros((0,), dtype=torch.long)
        return empty, empty
    return torch.cat(selected_candidates), torch.cat(selected_gt)


def lgp_score_assignments(
    candidates: CandidateSet,
    gt_boxes: torch.Tensor,
    count: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LGP score track from the released object-wise assigner.

    Foreground candidates are sorted by confidence, assigned to the GT with
    maximum IoU, and the first ``count`` candidates for every GT are retained.
    """
    if count <= 0 or candidates.scores.numel() == 0 or gt_boxes.numel() == 0:
        empty = candidates.labels.new_zeros((0,), dtype=torch.long)
        return empty, empty
    order = candidates.scores.argsort(descending=True)
    overlaps = pairwise_iou(gt_boxes, candidates.bboxes[order])
    owner = overlaps.argmax(dim=0)
    selected_candidates: List[torch.Tensor] = []
    selected_gt: List[torch.Tensor] = []
    for gt_index in range(gt_boxes.shape[0]):
        owned = order[owner == gt_index]
        if owned.numel() == 0:
            continue
        limit = min(int(count), owned.numel())
        selected_candidates.append(owned[:limit])
        selected_gt.append(
            torch.full(
                (limit,), gt_index, dtype=torch.long, device=gt_boxes.device
            )
        )
    if not selected_candidates:
        empty = candidates.labels.new_zeros((0,), dtype=torch.long)
        return empty, empty
    return torch.cat(selected_candidates), torch.cat(selected_gt)


def object_mask(
    image: torch.Tensor, boxes: torch.Tensor, dilation: float = 0.0
) -> torch.Tensor:
    batch, _, height, width = image.shape
    if batch != 1:
        raise ValueError("Object-aware attacks currently use batch size 1")
    mask = image.new_zeros((1, 1, height, width))
    if boxes.numel() == 0:
        return mask.fill_(1.0)
    for box in boxes.detach():
        box_width = max(float(box[2] - box[0]), 1.0)
        box_height = max(float(box[3] - box[1]), 1.0)
        left = max(0, int(torch.floor(box[0] - dilation * box_width).item()))
        top = max(0, int(torch.floor(box[1] - dilation * box_height).item()))
        right = min(width, int(torch.ceil(box[2] + dilation * box_width).item()))
        bottom = min(height, int(torch.ceil(box[3] + dilation * box_height).item()))
        if right > left and bottom > top:
            mask[:, :, top:bottom, left:right] = 1.0
    return mask


def object_region_weights(
    image: torch.Tensor,
    boxes: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return LGP background and radial foreground weights in image space."""
    if image.shape[0] != 1:
        raise ValueError("Object-wise weights currently use batch size 1")
    _, _, height, width = image.shape
    foreground = image.new_zeros((1, 1, height, width))
    radial = image.new_zeros((1, 1, height, width))
    if boxes.numel() == 0:
        return torch.ones_like(foreground), radial
    yy = torch.arange(height, device=image.device, dtype=image.dtype).view(1, 1, height, 1)
    xx = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, 1, width)
    for box in boxes.detach():
        center_x = (box[0] + box[2]) * 0.5
        center_y = (box[1] + box[3]) * 0.5
        half_width = (box[2] - box[0]).clamp(min=1.0) * float(scale) * 0.5
        half_height = (box[3] - box[1]).clamp(min=1.0) * float(scale) * 0.5
        inside = (
            (xx >= center_x - half_width)
            & (xx <= center_x + half_width)
            & (yy >= center_y - half_height)
            & (yy <= center_y + half_height)
        )
        radius = torch.sqrt(half_width.square() + half_height.square()).clamp(min=1.0)
        distance = torch.sqrt((xx - center_x).square() + (yy - center_y).square()) / radius
        previous_foreground = foreground
        # The release takes the minimum radial distance where objects overlap.
        value = torch.where(inside, distance, torch.ones_like(distance))
        existing = torch.where(
            previous_foreground > 0, radial, torch.ones_like(radial)
        )
        radial = torch.where(inside, torch.minimum(existing, value), radial)
        foreground = torch.maximum(previous_foreground, inside.to(image.dtype))
    background = 1.0 - foreground
    return background, radial * foreground


def feature_distance(
    clean_features: Sequence[torch.Tensor],
    adversarial_features: Sequence[torch.Tensor],
    levels: Sequence[int],
    mask: Optional[torch.Tensor] = None,
    amplify: float = 1.0,
) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for level in levels:
        if level < 0 or level >= min(len(clean_features), len(adversarial_features)):
            continue
        clean = clean_features[level].detach()
        adversarial = adversarial_features[level]
        difference = (float(amplify) * clean - adversarial).square()
        if mask is not None:
            resized = F.interpolate(
                mask, size=difference.shape[-2:], mode="bilinear", align_corners=False
            )
            difference = difference * resized
            losses.append(difference.sum() / resized.sum().clamp(min=1.0))
        else:
            losses.append(difference.mean())
    if not losses:
        raise RuntimeError("No requested detector feature level is available")
    return torch.stack(losses).mean()


def haar_components(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Orthonormal one-level Haar DWT implemented with tensor slicing."""
    pad_bottom = image.shape[-2] % 2
    pad_right = image.shape[-1] % 2
    value = F.pad(image, (0, pad_right, 0, pad_bottom), mode="reflect")
    x00 = value[..., 0::2, 0::2]
    x01 = value[..., 0::2, 1::2]
    x10 = value[..., 1::2, 0::2]
    x11 = value[..., 1::2, 1::2]
    ll = (x00 + x01 + x10 + x11) * 0.5
    lh = (-x00 - x01 + x10 + x11) * 0.5
    hl = (-x00 + x01 - x10 + x11) * 0.5
    hh = (x00 - x01 - x10 + x11) * 0.5
    return ll, lh, hl, hh


def haar_reconstructed_components(
    image: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct full-size LL-only and HH-only Haar images for NumbOD."""
    original_height, original_width = image.shape[-2:]
    ll, _lh, _hl, hh = haar_components(image)

    def reconstruct(
        low: torch.Tensor, high: torch.Tensor
    ) -> torch.Tensor:
        zero = torch.zeros_like(low)
        x00 = (low + high) * 0.5
        x01 = (low - high) * 0.5
        x10 = (low - high) * 0.5
        x11 = (low + high) * 0.5
        output = image.new_empty(
            (*image.shape[:-2], low.shape[-2] * 2, low.shape[-1] * 2)
        )
        output[..., 0::2, 0::2] = x00
        output[..., 0::2, 1::2] = x01
        output[..., 1::2, 0::2] = x10
        output[..., 1::2, 1::2] = x11
        return output[..., :original_height, :original_width]

    return reconstruct(ll, torch.zeros_like(hh)), reconstruct(
        torch.zeros_like(ll), hh
    )


def total_variation(value: torch.Tensor) -> torch.Tensor:
    horizontal = (value[..., :, 1:] - value[..., :, :-1]).abs().mean()
    vertical = (value[..., 1:, :] - value[..., :-1, :]).abs().mean()
    return horizontal + vertical


def diagnostics_record(step: int, loss: torch.Tensor, **values: Any) -> Dict[str, Any]:
    record: Dict[str, Any] = {"step": int(step), "loss": float(loss.detach().item())}
    for key, value in values.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            record[key] = float(value.detach().item())
        else:
            record[key] = value
    return record
