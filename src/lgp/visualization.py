from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

from .reporting.formatting import format_metric


_COLORS = (
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
    "#2E8B57",
    "#8A2BE2",
)


def _font(image: Image.Image, font_size: Optional[int] = None) -> ImageFont.ImageFont:
    if font_size is not None and (type(font_size) is not int or font_size <= 0):
        raise ValueError("font_size must be a positive integer")
    size = (font_size if font_size is not None
            else max(12, min(24, int(round(min(image.size) / 35.0)))))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _caption_metrics(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    *,
    image_size: tuple[int, int],
    limitations: Optional[list] = None,
    context: Optional[dict] = None,
) -> Optional[dict]:
    padding = 2
    width, height = image_size
    # Preserve the original bearing-aware fit policy and its 8 px floor.
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    def fits() -> bool:
        return (right - left + 2 * padding + 1 <= width
                and bottom - top + 2 * padding + 1 <= height)
    if not fits() and hasattr(font, "font_variant"):
        for size in range(getattr(font, "size", 8) - 1, 7, -1):
            candidate = font.font_variant(size=size)
            left, top, right, bottom = draw.textbbox((0, 0), text, font=candidate)
            font = candidate
            if fits():
                break
    if not fits():
        if limitations is not None:
            limitations.append({
                "text": text,
                "reason": "caption_does_not_fit_image_at_readable_font_size",
                "action": "caption_not_drawn_detection_unchanged",
                "required_width": right - left + 2 * padding + 1,
                "required_height": bottom - top + 2 * padding + 1,
                "image_width": width,
                "image_height": height,
                **(context or {}),
            })
        return None
    return {"text": text, "font": font, "left": left, "top": top,
            "width": right - left + 2 * padding + 1,
            "height": bottom - top + 2 * padding + 1}


def _caption_rectangle(position, caption, image_size):
    x, y = position
    width, height = image_size
    background_x = max(0, min(x + caption["left"] - 2, width - caption["width"]))
    background_y = max(0, min(y + caption["top"] - 2, height - caption["height"]))
    return (background_x, background_y,
            background_x + caption["width"], background_y + caption["height"])


def _draw_caption(draw, rectangle, caption, fill):
    left, top, right, bottom = rectangle
    draw.rectangle((left, top, right - 1, bottom - 1), fill=fill)
    draw.text((left + 2 - caption["left"], top + 2 - caption["top"]),
              caption["text"], fill="white", font=caption["font"])


def _text_box(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: str,
    *,
    image_size: Optional[tuple[int, int]] = None,
    limitations: Optional[list] = None,
) -> None:
    if image_size is not None:
        caption = _caption_metrics(draw, text, font, image_size=image_size, limitations=limitations)
        if caption is not None:
            _draw_caption(draw, _caption_rectangle(position, caption, image_size), caption, fill)
        return
    x, y = position
    padding = 2
    left, top, right, bottom = draw.textbbox((x, y), text, font=font)
    draw.rectangle(
        (
            left - padding,
            top - padding,
            right + padding,
            bottom + padding,
        ),
        fill=fill,
    )
    draw.text((x, y), text, fill="white", font=font)


def _overlaps(rectangle, occupied, gap=2):
    return any(rectangle[0] < other[2] + gap and rectangle[2] + gap > other[0]
               and rectangle[1] < other[3] + gap and rectangle[3] + gap > other[1]
               for other in occupied)


def _place_caption(preferred, caption, image_size, occupied):
    width, height = image_size
    max_x, max_y = width - caption["width"], height - caption["height"]
    def rectangle(x, y):
        x, y = max(0, min(x, max_x)), max(0, min(y, max_y))
        return (x, y, x + caption["width"], y + caption["height"])
    def distance(rect):
        return ((rect[0] - preferred[0]) ** 2 + (rect[1] - preferred[1]) ** 2, rect[1], rect[0])
    local = {rectangle(preferred[0] + dx, preferred[1] + dy)
             for dx in (0, -caption["width"] - 2, caption["width"] + 2)
             for dy in (0, -caption["height"] - 2, caption["height"] + 2,
                        -2 * (caption["height"] + 2), 2 * (caption["height"] + 2))}
    for candidate in sorted(local, key=distance):
        if not _overlaps(candidate, occupied):
            return candidate
    # At most 64 x 64 image-grid origins; do not claim exhaustive packing.
    xs = {int(round(max_x * i / 63.0)) for i in range(64)}
    ys = {int(round(max_y * i / 63.0)) for i in range(64)}
    for candidate in sorted((rectangle(x, y) for x in xs for y in ys), key=distance):
        if not _overlaps(candidate, occupied):
            return candidate
    return None


def _layout_record(kind, rectangle, caption, **details):
    return {"kind": kind, "text": caption["text"], "rectangle": list(rectangle),
            "effective_font_size": getattr(caption["font"], "size", None), **details}


def render_detection_overlay(
    image_path: Path,
    output_path: Path,
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    labels: Sequence[int],
    class_names: Sequence[str],
    *,
    score_threshold: float = 0.3,
    max_detections: int = 100,
    line_width: int = 3,
    title: Optional[str] = None,
    font_size: Optional[int] = None,
    include_layout_rectangles: bool = False,
) -> Dict[str, Any]:
    """Draw post-NMS boxes, class names and four-decimal confidence scores.

    Captions are moved inside the image and, if needed, fitted down to 8 px
    with scalable fonts. A caption still too large is not drawn and is recorded
    in ``caption_layout_limitations``; its detection is never removed. Caption
    collisions are avoided by deterministic nearby placement and a bounded
    image-grid search. An unavailable caption slot is explicitly recorded;
    no detections are filtered to improve layout. The title reserves its area.
    All boxes and optional thin leaders are drawn before text. Optional
    ``layout_rectangles`` use half-open source-image pixel coordinates and
    include the title, permitting bounds and non-overlap checks.
    ``font_size`` optionally requests a positive integer source-image pixel size;
    None preserves automatic sizing. Fitting may reduce that requested size.
    """
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must lie in [0, 1]")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    if line_width <= 0:
        raise ValueError("line_width must be positive")
    if font_size is not None and (type(font_size) is not int or font_size <= 0):
        raise ValueError("font_size must be a positive integer")

    image_path = image_path.resolve()
    output_path = output_path.resolve()
    with Image.open(str(image_path)) as source:
        image = source.convert("RGB")
    width, height = image.size
    font = _font(image, font_size=font_size)
    draw = ImageDraw.Draw(image)
    caption_layout_limitations = []

    detections = []
    for raw_box, raw_score, raw_label in zip(boxes, scores, labels):
        score = float(raw_score)
        label = int(raw_label)
        if not math.isfinite(score) or score < score_threshold:
            continue
        if label < 0 or label >= len(class_names) or len(raw_box) < 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in raw_box[:4]]
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        x1 = max(0.0, min(float(width - 1), x1))
        y1 = max(0.0, min(float(height - 1), y1))
        x2 = max(0.0, min(float(width - 1), x2))
        y2 = max(0.0, min(float(height - 1), y2))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append((score, label, (x1, y1, x2, y2)))

    detections.sort(key=lambda item: item[0], reverse=True)
    detections = detections[:max_detections]
    # Preserve box draw order, but finish every outline before adding text.
    for score, label, box in reversed(detections):
        color = _COLORS[label % len(_COLORS)]
        x1, y1, x2, y2 = box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)

    occupied, captions, layout_rectangles = [], [], []
    heading = title or ("No detections >= {}".format(format_metric(score_threshold)) if not detections else None)
    header = None
    if heading:
        kind = "title" if title else "empty"
        caption = _caption_metrics(draw, heading, font, image_size=image.size,
                                   limitations=caption_layout_limitations, context={"kind": kind})
        if caption is not None:
            rectangle = _caption_rectangle((6, 6), caption, image.size)
            occupied.append(rectangle)
            header = (rectangle, caption)
            layout_rectangles.append(_layout_record(kind, rectangle, caption))

    for index, (score, label, box) in enumerate(detections):
        color = _COLORS[label % len(_COLORS)]
        x1, y1, x2, y2 = box
        text = "{} {}".format(class_names[label], format_metric(score))
        text_bounds = draw.textbbox((0, 0), text, font=font)
        text_height = text_bounds[3] - text_bounds[1] + 4
        text_y = int(y1) - text_height if y1 >= text_height else int(y1) + 2
        context = {"kind": "detection", "detection_index": index}
        caption = _caption_metrics(draw, text, font, image_size=image.size,
                                   limitations=caption_layout_limitations, context=context)
        if caption is None:
            continue
        preferred = _caption_rectangle((int(x1) + 2, text_y), caption, image.size)
        rectangle = _place_caption(preferred, caption, image.size, occupied)
        if rectangle is None:
            caption_layout_limitations.append({
                **context, "text": text, "reason": "no_collision_free_caption_slot_in_bounded_search",
                "action": "caption_not_drawn_detection_unchanged", "grid_axis_samples": 64,
                "required_width": caption["width"], "required_height": caption["height"],
                "image_width": width, "image_height": height,
            })
            continue
        occupied.append(rectangle)
        leader = None
        moved = math.hypot(rectangle[0] - preferred[0], rectangle[1] - preferred[1])
        if moved > max(12, caption["height"]):
            anchor = (int(x1), int(y1))
            endpoint = (max(rectangle[0], min(anchor[0], rectangle[2] - 1)),
                        max(rectangle[1], min(anchor[1], rectangle[3] - 1)))
            leader = [list(anchor), list(endpoint)]
            draw.line((anchor, endpoint), fill=color, width=1)
        captions.append((rectangle, caption, color))
        layout_rectangles.append(_layout_record("detection", rectangle, caption,
            detection_index=index, class_label=label, score=score, box=list(box),
            anchor=[int(x1), int(y1)], leader_line=leader))

    for rectangle, caption, color in captions:
        _draw_caption(draw, rectangle, caption, color)
    if header is not None:
        _draw_caption(draw, header[0], header[1], "#202020")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path), format="PNG")
    record = {
        "input_path": str(image_path),
        "overlay_path": str(output_path),
        "image_width": width,
        "image_height": height,
        "score_threshold": float(score_threshold),
        "detections_available": len(scores),
        "detections_drawn": len(detections),
        "caption_layout_limitations": caption_layout_limitations,
    }
    if include_layout_rectangles:
        record["layout_rectangles"] = layout_rectangles
    return record
