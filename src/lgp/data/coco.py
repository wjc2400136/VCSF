from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..io import atomic_json, iter_download
from ..paths import project_root
from ..registry import DatasetSpec


def select_coco_images(
    images: Sequence[Mapping[str, Any]],
    max_images: Optional[int] = None,
    image_ids: Optional[Sequence[int]] = None,
) -> List[Mapping[str, Any]]:
    """Select an ordered COCO image subset with strict ID validation."""
    ordered = list(images)
    if image_ids is None:
        return ordered[:max_images] if max_images else ordered

    requested_ids = [int(value) for value in image_ids]
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("image_ids contains duplicates")
    by_id = {int(image["id"]): image for image in ordered}
    missing = [image_id for image_id in requested_ids if image_id not in by_id]
    if missing:
        raise ValueError(
            "Requested image IDs are absent from the COCO index: {}".format(
                missing[:10]
            )
        )
    if max_images is not None:
        requested_ids = requested_ids[:max_images]
    return [by_id[image_id] for image_id in requested_ids]


class CocoIndex:
    def __init__(self, dataset: DatasetSpec, split: str) -> None:
        self.dataset = dataset
        self.split = dataset.split(split)
        self.annotation_path = (dataset.root / self.split.annotation).resolve()
        if not self.annotation_path.is_file():
            raise FileNotFoundError("Missing annotation file: {}".format(self.annotation_path))
        with self.annotation_path.open("r", encoding="utf-8") as handle:
            self.payload: Dict[str, Any] = json.load(handle)
        self.images = sorted(self.payload.get("images", []), key=lambda item: int(item["id"]))
        self.categories = list(self.payload.get("categories", []))
        class_to_label = {name: index for index, name in enumerate(dataset.classes)}
        self._category_to_label = {}
        for category in self.categories:
            name = str(category.get("name", ""))
            if name not in class_to_label:
                raise ValueError(
                    "Annotation category '{}' is absent from dataset '{}' class order".format(
                        name, dataset.id
                    )
                )
            self._category_to_label[int(category["id"])] = class_to_label[name]
        self._annotations = defaultdict(list)
        for annotation in self.payload.get("annotations", []):
            self._annotations[int(annotation["image_id"])].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def image_path(self, image: Mapping[str, Any]) -> Path:
        prefix = self.dataset.root / self.split.image_prefix
        return (prefix / str(image["file_name"])).resolve()

    def annotations(self, image_id: int) -> List[Dict[str, Any]]:
        return list(self._annotations.get(int(image_id), []))

    def xyxy_boxes(self, image_id: int, include_crowd: bool = False) -> List[List[float]]:
        boxes, _ = self.instances(image_id, include_crowd=include_crowd)
        return boxes

    def instances(
        self, image_id: int, include_crowd: bool = False
    ) -> Tuple[List[List[float]], List[int]]:
        """Return aligned boxes and contiguous detector labels."""
        boxes: List[List[float]] = []
        labels: List[int] = []
        for annotation in self.annotations(image_id):
            if not include_crowd and int(annotation.get("iscrowd", 0)):
                continue
            x, y, width, height = [float(value) for value in annotation["bbox"]]
            if width <= 0 or height <= 0:
                continue
            boxes.append([x, y, x + width, y + height])
            category_id = int(annotation["category_id"])
            try:
                labels.append(self._category_to_label[category_id])
            except KeyError as exc:
                raise ValueError(
                    "Unknown category id {} in {}".format(
                        category_id, self.annotation_path
                    )
                ) from exc
        return boxes, labels

    def adversarial_annotation(self, filenames: Mapping[int, str]) -> Dict[str, Any]:
        selected_ids = set(int(value) for value in filenames)
        images = []
        for image in self.images:
            image_id = int(image["id"])
            if image_id not in selected_ids:
                continue
            copied = dict(image)
            copied["file_name"] = filenames[image_id]
            images.append(copied)
        annotations = [
            dict(annotation)
            for annotation in self.payload.get("annotations", [])
            if int(annotation["image_id"]) in selected_ids
        ]
        return {
            "info": dict(self.payload.get("info", {})),
            "licenses": list(self.payload.get("licenses", [])),
            "images": images,
            "annotations": annotations,
            "categories": list(self.payload.get("categories", [])),
        }


def prepare_coco_dev(
    dataset: DatasetSpec,
    count: int = 1000,
    seed: int = 20260722,
) -> List[Path]:
    """Create the frozen train-only development subset used for tuning."""
    output_annotation = dataset.root / "annotations" / "instances_dev2017.json"
    manifest = dataset.root / "annotations" / "dev2017_selection.json"
    destination_root = dataset.root / "dev2017"
    source_annotation = dataset.root / "annotations" / "instances_train2017.json"
    if not source_annotation.is_file():
        if output_annotation.is_file() and manifest.is_file():
            with output_annotation.open("r", encoding="utf-8") as handle:
                prepared = json.load(handle)
            prepared_images = list(prepared.get("images", []))
            missing = [
                destination_root / str(image["file_name"])
                for image in prepared_images
                if not (destination_root / str(image["file_name"])).is_file()
            ]
            if len(prepared_images) == count and not missing:
                return [output_annotation, manifest, destination_root]
        raise FileNotFoundError(
            "COCO development selection needs the official train annotation: {}".format(
                source_annotation
            )
        )
    with source_annotation.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = list(payload.get("images", []))
    if count <= 0 or count > len(images):
        raise ValueError("COCO dev count must lie in [1, {}]".format(len(images)))
    annotations = list(payload.get("annotations", []))
    image_by_id = {int(image["id"]): image for image in images}
    category_images: Dict[int, List[int]] = defaultdict(list)
    for annotation in annotations:
        category_images[int(annotation["category_id"])].append(
            int(annotation["image_id"])
        )
    rng = random.Random(seed)
    selected_ids = set()
    for category in sorted(
        payload.get("categories", []), key=lambda item: int(item["id"])
    ):
        candidates = sorted(set(category_images.get(int(category["id"]), [])))
        if not candidates:
            raise ValueError(
                "COCO train has no image for category {}".format(category["id"])
            )
        selected_ids.add(rng.choice(candidates))
    remaining = [
        image_id for image_id in sorted(image_by_id) if image_id not in selected_ids
    ]
    rng.shuffle(remaining)
    selected_ids.update(remaining[: count - len(selected_ids)])
    if len(selected_ids) != count:
        raise RuntimeError("COCO dev selection produced {} images".format(len(selected_ids)))
    selected_images = [image_by_id[image_id] for image_id in sorted(selected_ids)]
    selected_annotations = [
        annotation
        for annotation in annotations
        if int(annotation["image_id"]) in selected_ids
    ]
    atomic_json(
        output_annotation,
        {
            "info": {
                **dict(payload.get("info", {})),
                "description": "Frozen COCO train2017 development subset for LGP",
                "selection_seed": seed,
                "selection_count": count,
                "selection_algorithm": "one seeded image per category, then seeded random fill",
                "formal_evaluation_split": "val2017 (disjoint)",
            },
            "licenses": list(payload.get("licenses", [])),
            "images": selected_images,
            "annotations": selected_annotations,
            "categories": list(payload.get("categories", [])),
        },
    )
    atomic_json(
        manifest,
        {
            "seed": seed,
            "count": count,
            "image_ids": sorted(selected_ids),
            "annotation": str(output_annotation.relative_to(project_root())),
        },
    )
    source_root = dataset.root / "train2017"
    destination_root.mkdir(parents=True, exist_ok=True)
    for image in selected_images:
        filename = str(image["file_name"])
        destination = destination_root / filename
        if destination.is_file() and destination.stat().st_size > 0:
            continue
        source = source_root / filename
        if source.is_file():
            shutil.copy2(str(source), str(destination))
            continue
        url = image.get("coco_url")
        if not url:
            raise FileNotFoundError(
                "Missing {} and annotation has no coco_url".format(source)
            )
        for _ in iter_download(str(url), destination):
            pass
    return [output_annotation, manifest, destination_root]
