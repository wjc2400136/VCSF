from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..io import atomic_json
from ..registry import DatasetSpec


def _read_ids(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError("Missing VOC split file: {}".format(path))
    return [line.strip().split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_object(
    node: ET.Element,
    category_ids: Dict[str, int],
    annotation_id: int,
    image_id: int,
) -> Dict[str, Any]:
    name = (node.findtext("name") or "").strip()
    if name not in category_ids:
        raise ValueError("Unexpected VOC category: {}".format(name))
    box = node.find("bndbox")
    if box is None:
        raise ValueError("VOC object has no bndbox")
    xmin = float(box.findtext("xmin")) - 1.0
    ymin = float(box.findtext("ymin")) - 1.0
    xmax = float(box.findtext("xmax")) - 1.0
    ymax = float(box.findtext("ymax")) - 1.0
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    difficult = int(node.findtext("difficult") or 0)
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_ids[name],
        "bbox": [xmin, ymin, width, height],
        "area": width * height,
        "iscrowd": difficult,
        "ignore": difficult,
        "segmentation": [],
    }


def convert_voc_split(
    dataset: DatasetSpec,
    sources: Sequence[Tuple[str, str]],
    output_name: str,
) -> Path:
    categories = [
        {"id": index + 1, "name": name, "supercategory": "object"}
        for index, name in enumerate(dataset.classes)
    ]
    category_ids = {item["name"]: item["id"] for item in categories}
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    image_id = 1
    annotation_id = 1
    for year, split_name in sources:
        year_root = dataset.root / ("VOC" + year)
        split_file = year_root / "ImageSets" / "Main" / (split_name + ".txt")
        for identifier in _read_ids(split_file):
            xml_path = year_root / "Annotations" / (identifier + ".xml")
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            size = root.find("size")
            if size is None:
                raise ValueError("VOC annotation has no size: {}".format(xml_path))
            width = int(float(size.findtext("width")))
            height = int(float(size.findtext("height")))
            images.append(
                {
                    "id": image_id,
                    "file_name": "VOC{}/JPEGImages/{}.jpg".format(year, identifier),
                    "width": width,
                    "height": height,
                    "year": int(year),
                    "voc_id": identifier,
                }
            )
            for obj in root.findall("object"):
                annotation = _parse_object(
                    obj, category_ids, annotation_id, image_id
                )
                if annotation["area"] > 0:
                    annotations.append(annotation)
                    annotation_id += 1
            image_id += 1

    payload = {
        "info": {"description": "PASCAL VOC converted by lgp-benchmark"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output = dataset.root / "annotations" / output_name
    atomic_json(output, payload)
    return output


def prepare_voc(dataset: DatasetSpec) -> List[Path]:
    prepared = [
        dataset.root / "annotations" / "voc0712_trainval.json",
        dataset.root / "annotations" / "voc2007_test.json",
    ]
    source_roots = [
        dataset.root / "VOC2007" / "Annotations",
        dataset.root / "VOC2007" / "ImageSets" / "Main",
        dataset.root / "VOC2012" / "Annotations",
        dataset.root / "VOC2012" / "ImageSets" / "Main",
    ]
    if not all(path.is_dir() for path in source_roots) and all(
        path.is_file() for path in prepared
    ):
        return prepared
    train = convert_voc_split(
        dataset,
        [("2007", "trainval"), ("2012", "trainval")],
        "voc0712_trainval.json",
    )
    test = convert_voc_split(
        dataset,
        [("2007", "test")],
        "voc2007_test.json",
    )
    return [train, test]
