"""CPU-only caption geometry, legacy fitting and unchanged detection selection."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import pytest

from lgp import visualization as vis


def picture(tmp_path, size=(640, 426)):
    path = tmp_path / "input.png"
    Image.new("RGB", size, "#777777").save(path)
    return path


def render(tmp_path, boxes, scores, labels=None, classes=None, size=(640, 426), **options):
    defaults = dict(score_threshold=0.5, include_layout_rectangles=True)
    defaults.update(options)
    return vis.render_detection_overlay(picture(tmp_path, size), tmp_path / "overlay.png",
        boxes, scores, labels if labels is not None else [0] * len(scores),
        classes if classes is not None else ["person"], **defaults)


def assert_geometry(record):
    rectangles = record["layout_rectangles"]
    for row in rectangles:
        left, top, right, bottom = row["rectangle"]
        assert 0 <= left < right <= record["image_width"]
        assert 0 <= top < bottom <= record["image_height"]
        assert all(type(value) is int for value in row["rectangle"])
    for index, row in enumerate(rectangles):
        a = row["rectangle"]
        for other in rectangles[index + 1:]:
            b = other["rectangle"]
            assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
    labels = [row for row in rectangles if row["kind"] == "detection"]
    missing = [row for row in record["caption_layout_limitations"] if row["kind"] == "detection"]
    assert len(labels) + len(missing) == record["detections_drawn"]
    assert len({row["detection_index"] for row in labels + missing}) == record["detections_drawn"]


@pytest.mark.parametrize("count", [20, 30])
def test_dense_labels_are_placed_without_changing_selected_boxes(tmp_path, count):
    boxes = [[180, 120, 320, 300]] * count
    scores = [0.99 - i * 0.01 for i in range(count)]
    before = deepcopy((boxes, scores))
    record = render(tmp_path, boxes, scores, title="Detector | clean | score >= 0.5000")
    assert record["detections_available"] == record["detections_drawn"] == count
    assert record["caption_layout_limitations"] == []
    assert record["image_width"] == 640 and record["image_height"] == 426
    assert_geometry(record)
    labels = [row for row in record["layout_rectangles"] if row["kind"] == "detection"]
    assert [row["text"] for row in labels] == ["person {:.4f}".format(s) for s in scores]
    assert all(row["box"] == box for row, box in zip(labels, boxes))
    assert any(row["leader_line"] is not None for row in labels)
    assert (boxes, scores) == before
    original = Path(record["overlay_path"]).read_bytes()
    again = vis.render_detection_overlay(tmp_path / "input.png", tmp_path / "again.png", boxes, scores,
        [0] * count, ["person"], score_threshold=0.5, title="Detector | clean | score >= 0.5000",
        include_layout_rectangles=True)
    assert Path(again["overlay_path"]).read_bytes() == original
    assert again["layout_rectangles"] == record["layout_rectangles"]


@pytest.mark.parametrize("box", [[0, 0, 120, 190], [233, 0, 352, 200], [0, 430, 200, 499]])
def test_edge_labels_respect_reserved_title(tmp_path, box):
    record = render(tmp_path, [box], [0.9981541037559509], size=(353, 500),
                    title="Faster R-CNN R50 | clean | score >= 0.5000")
    assert record["detections_drawn"] == 1 and not record["caption_layout_limitations"]
    assert {row["kind"] for row in record["layout_rectangles"]} == {"title", "detection"}
    assert_geometry(record)


def test_every_outline_and_leader_is_drawn_before_any_text(tmp_path, monkeypatch):
    events = []
    rectangle, line, text = ImageDraw.ImageDraw.rectangle, ImageDraw.ImageDraw.line, ImageDraw.ImageDraw.text
    def traced_rectangle(self, xy, *args, **kwargs):
        events.append("outline" if kwargs.get("outline") is not None else "background")
        return rectangle(self, xy, *args, **kwargs)
    def traced_line(self, xy, *args, **kwargs):
        events.append("leader")
        assert kwargs["width"] == 1
        return line(self, xy, *args, **kwargs)
    def traced_text(self, xy, message, *args, **kwargs):
        events.append("text:" + message)
        return text(self, xy, message, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", traced_rectangle)
    monkeypatch.setattr(ImageDraw.ImageDraw, "line", traced_line)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", traced_text)
    render(tmp_path, [[100, 100, 200, 300]] * 20, [0.9] * 20, title="Reserved title")
    first_text = next(i for i, event in enumerate(events) if event.startswith("text:"))
    assert events.count("outline") == 20 and "leader" in events
    assert all(i < first_text for i, event in enumerate(events) if event in {"outline", "leader"})
    assert events[-1] == "text:Reserved title"


def test_existing_readable_font_fit_floor_is_retained(tmp_path):
    message = "long category 0.9000"
    font = ImageFont.truetype("DejaVuSans.ttf", size=8)
    sample = ImageDraw.Draw(Image.new("RGB", (300, 100)))
    left, top, right, bottom = sample.textbbox((0, 0), message, font=font)
    size = (right - left + 5, bottom - top + 5)
    record = render(tmp_path, [[0, 0, size[0] - 1, size[1] - 1]], [0.9],
                    classes=["long category"], size=size, font_size=24)
    assert not record["caption_layout_limitations"] and record["detections_drawn"] == 1
    assert record["layout_rectangles"][0]["effective_font_size"] == 8
    assert_geometry(record)


def test_tiny_image_preserves_box_and_records_fit_failure(tmp_path):
    record = render(tmp_path, [[0, 0, 7, 7]], [0.9], size=(8, 8), title="Long title")
    assert record["detections_drawn"] == 1 and record["layout_rectangles"] == []
    assert {row["kind"] for row in record["caption_layout_limitations"]} == {"title", "detection"}
    assert all(row["reason"] == "caption_does_not_fit_image_at_readable_font_size"
               and row["action"] == "caption_not_drawn_detection_unchanged"
               for row in record["caption_layout_limitations"])
    assert Image.open(record["overlay_path"]).size == (8, 8)
    assert_geometry(record)


def test_exhausted_layout_is_honest_and_never_filters_detections(tmp_path):
    record = render(tmp_path, [[0, 0, 99, 39]] * 12, [0.9] * 12,
                    size=(100, 40), title="Title")
    assert record["detections_drawn"] == 12
    assert any(row["reason"] == "no_collision_free_caption_slot_in_bounded_search"
               and row["grid_axis_samples"] == 64 for row in record["caption_layout_limitations"])
    assert_geometry(record)


@pytest.mark.parametrize("position", [(-30, -20), (0, 0), (350, 490)])
def test_legacy_text_box_fitting_keeps_bearing_and_padding_inside_image(position):
    image = Image.new("RGB", (353, 500), "black")
    draw = ImageDraw.Draw(image)
    font = vis._font(image, 12)
    limits = []
    assert vis._text_box(draw, position, "person 0.9000", font, "#E64B35",
                         image_size=image.size, limitations=limits) is None
    assert limits == []
    changed = image.getbbox()
    assert changed is not None and 0 <= changed[0] < changed[2] <= 353
    assert 0 <= changed[1] < changed[3] <= 500


@pytest.mark.parametrize("font_size", [0, -1, True, 1.2, "12"])
def test_invalid_font_size_is_rejected_before_output(tmp_path, font_size):
    with pytest.raises(ValueError, match="font_size"):
        render(tmp_path, [[0, 0, 10, 10]], [0.9], font_size=font_size)
    assert not (tmp_path / "overlay.png").exists()


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf])
def test_invalid_threshold_is_rejected_before_output(tmp_path, threshold):
    with pytest.raises(ValueError, match="score_threshold"):
        render(tmp_path, [], [], score_threshold=threshold)
    assert not (tmp_path / "overlay.png").exists()


@pytest.mark.parametrize("title", [None, "Detector title"])
def test_empty_case_keeps_banner_or_title_with_explicit_geometry(tmp_path, title):
    record = render(tmp_path, [], [], title=title)
    assert record["detections_drawn"] == 0 and record["caption_layout_limitations"] == []
    assert len(record["layout_rectangles"]) == 1
    assert record["layout_rectangles"][0]["kind"] == ("title" if title else "empty")
    assert_geometry(record)


def test_clipping_sort_threshold_ties_and_max_selection_are_unchanged(tmp_path):
    boxes = [[-10, -5, 2000, 2000], [20, 20, 80, 80], [100, 100, 150, 150],
             [20, 20, 10, 10], [math.nan, 0, 10, 10], [1, 1, 3, 3], [1, 1, 3, 3]]
    scores = [0.8, 0.9, 0.9, 0.99, 0.99, math.nan, 0.49]
    labels = [0, 1, 0, 0, 0, 0, 0]
    record = render(tmp_path, boxes, scores, labels=labels, classes=["person", "chair"], max_detections=2)
    chosen = [row for row in record["layout_rectangles"] if row["kind"] == "detection"]
    assert record["detections_available"] == 7 and record["detections_drawn"] == 2
    assert [row["class_label"] for row in chosen] == [1, 0]
    assert [row["box"] for row in chosen] == [[20.0, 20.0, 80.0, 80.0], [100.0, 100.0, 150.0, 150.0]]
    full = render(tmp_path, boxes, scores, labels=labels, classes=["person", "chair"], max_detections=100)
    chosen = [row for row in full["layout_rectangles"] if row["kind"] == "detection"]
    assert full["detections_drawn"] == 3 and chosen[-1]["box"] == [0.0, 0.0, 639.0, 425.0]
    assert_geometry(full)


def test_optional_geometry_preserves_required_manifest_fields(tmp_path):
    record = render(tmp_path, [[10, 20, 80, 100]], [0.9], include_layout_rectangles=False)
    assert set(record) == {"input_path", "overlay_path", "image_width", "image_height", "score_threshold",
                           "detections_available", "detections_drawn", "caption_layout_limitations"}


def test_grid_fallback_search_is_bounded_and_returns_no_false_slot(monkeypatch):
    calls = []
    def blocked(rectangle, occupied, gap=2):
        calls.append(rectangle)
        return True
    monkeypatch.setattr(vis, "_overlaps", blocked)
    assert vis._place_caption((10, 10, 50, 25), {"width": 40, "height": 15}, (640, 426), []) is None
    assert 0 < len(calls) <= 15 + 64 * 64
