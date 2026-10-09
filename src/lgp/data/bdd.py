from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

from ..io import atomic_json
from ..registry import DatasetSpec


_ALIASES = {
    "person": "pedestrian",
    "motor": "motorcycle",
    "bike": "bicycle",
}


def _find_label_source(root: Path, split: str) -> Path:
    candidates = [
        root / "labels" / split,
        root / "labels" / "det_20" / ("det_{}.json".format(split)),
        root / "labels" / ("bdd100k_labels_images_{}.json".format(split)),
        root / ("bdd100k_labels_images_{}.json".format(split)),
    ]
    for candidate in candidates:
        if candidate.is_file() or candidate.is_dir():
            return candidate
    matches = list(root.rglob("*{}*.json".format(split)))
    for match in matches:
        lowered = match.name.lower()
        if "det" in lowered or "labels_images" in lowered:
            return match
    raise FileNotFoundError("Cannot locate BDD100K detection labels for '{}'".format(split))


def _load_records(source: Path) -> List[Dict[str, Any]]:
    if source.is_dir():
        records: List[Dict[str, Any]] = []
        for path in sorted(source.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if not isinstance(record, dict):
                raise ValueError("Per-image BDD100K label must be an object: {}".format(path))
            records.append(record)
        return records
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("BDD100K aggregate label file must contain a list: {}".format(source))
    return payload


def _image_size(path: Path) -> List[int]:
    with Image.open(str(path)) as image:
        return [int(image.width), int(image.height)]


def convert_bdd_split(dataset: DatasetSpec, split: str) -> Path:
    label_path = _find_label_source(dataset.root, split)
    records = _load_records(label_path)

    category_ids = {name: index + 1 for index, name in enumerate(dataset.classes)}
    categories = [
        {"id": category_id, "name": name, "supercategory": "object"}
        for name, category_id in category_ids.items()
    ]
    image_root = dataset.root / "images" / "100k" / split
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    annotation_id = 1
    for image_id, record in enumerate(records, start=1):
        filename = str(record.get("name") or record.get("file_name") or "")
        if not filename:
            raise ValueError("BDD100K label record has no image name")
        if not Path(filename).suffix:
            filename += ".jpg"
        image_path = image_root / filename
        width, height = _image_size(image_path)
        images.append(
            {"id": image_id, "file_name": filename, "width": width, "height": height}
        )
        labels = record.get("labels") or record.get("objects") or []
        if not labels and isinstance(record.get("frames"), list) and record["frames"]:
            labels = record["frames"][0].get("objects") or []
        for label in labels:
            raw_category = str(label.get("category", ""))
            category = _ALIASES.get(raw_category, raw_category)
            if category not in category_ids:
                continue
            box = label.get("box2d") or label.get("bbox")
            if not isinstance(box, dict):
                continue
            x1 = float(box.get("x1", box.get("xmin", 0.0)))
            y1 = float(box.get("y1", box.get("ymin", 0.0)))
            x2 = float(box.get("x2", box.get("xmax", 0.0)))
            y2 = float(box.get("y2", box.get("ymax", 0.0)))
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            if box_width <= 0 or box_height <= 0:
                continue
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[category],
                    "bbox": [x1, y1, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [],
                    "attributes": dict(label.get("attributes", {})),
                }
            )
            annotation_id += 1

    output = dataset.root / "annotations" / ("det_{}_coco.json".format(split))
    atomic_json(
        output,
        {
            "info": {"description": "BDD100K Detection 2020 converted by lgp-benchmark"},
            "licenses": [],
            "images": images,
            "annotations": annotations,
            "categories": categories,
        },
    )
    return output


def prepare_bdd100k(dataset: DatasetSpec) -> List[Path]:
    prepared = [
        dataset.root / "annotations" / "det_train_coco.json",
        dataset.root / "annotations" / "det_val_coco.json",
    ]
    try:
        sources_present = all(
            _find_label_source(dataset.root, split).exists()
            for split in ("train", "val")
        )
    except FileNotFoundError:
        sources_present = False
    if not sources_present and all(path.is_file() for path in prepared):
        return prepared
    converted: List[Path] = []
    for split in ("train", "val"):
        converted.append(convert_bdd_split(dataset, split))
    return converted
