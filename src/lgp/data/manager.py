from __future__ import annotations

import json
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from ..io import file_digest, iter_download
from ..paths import project_root
from ..registry import DatasetSpec, Registry
from .bdd import prepare_bdd100k
from .coco import CocoIndex, prepare_coco_dev
from .voc import prepare_voc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def validate_dataset(
    dataset: DatasetSpec, deep: bool = False, full: bool = False
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "dataset": dataset.id,
        "root": str(dataset.root),
        "status": "valid",
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    if not dataset.root.is_dir():
        report["status"] = "missing"
        report["errors"].append("Dataset root does not exist")
        return report

    referenced_images = set()
    split_references: Dict[str, set] = {}
    inventory_complete = True
    for name, split in dataset.splits.items():
        annotation = dataset.root / split.annotation
        image_root = dataset.root / split.image_prefix
        split_report: Dict[str, Any] = {
            "annotation": str(annotation),
            "image_root": str(image_root),
            "expected_images": split.expected_images,
            "benchmark_required": split.benchmark_required,
        }
        if not full and not split.benchmark_required:
            split_report["status"] = "optional_not_checked"
            report["splits"][name] = split_report
            continue
        if not image_root.is_dir():
            inventory_complete = False
            split_report["status"] = "missing_images"
            report["errors"].append("{}: missing image directory".format(name))
            report["splits"][name] = split_report
            continue
        if not annotation.is_file():
            inventory_complete = False
            if split.annotations_optional:
                split_report["status"] = "images_only"
                report["warnings"].append("{}: optional annotation file is absent".format(name))
            else:
                split_report["status"] = "missing_annotation"
                report["errors"].append("{}: missing annotation file".format(name))
            if deep:
                count = sum(
                    1
                    for path in image_root.rglob("*")
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                )
                split_report["physical_images"] = count
            report["splits"][name] = split_report
            continue
        try:
            index = CocoIndex(dataset, name)
            split_report["annotation_images"] = len(index)
            split_report["annotation_objects"] = len(index.payload.get("annotations", []))
            split_report["categories"] = len(index.categories)
            split_report["status"] = "valid"
            if len(index) != split.expected_images:
                split_report["status"] = "count_mismatch"
                report["errors"].append(
                    "{}: expected {} annotation images, found {}".format(
                        name, split.expected_images, len(index)
                    )
                )
            if index.categories and len(index.categories) != dataset.num_classes:
                report["errors"].append(
                    "{}: expected {} categories, found {}".format(
                        name, dataset.num_classes, len(index.categories)
                    )
                )
            if deep:
                split_paths = [
                    index.image_path(image).resolve() for image in index.images
                ]
                split_references[name] = set(split_paths)
                duplicate_references = len(split_paths) - len(
                    split_references[name]
                )
                split_report["duplicate_image_references"] = duplicate_references
                if duplicate_references:
                    report["errors"].append(
                        "{}: {} duplicate image references".format(
                            name, duplicate_references
                        )
                    )
                escaped = [
                    str(path)
                    for path in split_paths
                    if path != dataset.root and dataset.root not in path.parents
                ]
                if escaped:
                    inventory_complete = False
                    report["errors"].append(
                        "{}: {} image paths escape the dataset root".format(
                            name, len(escaped)
                        )
                    )
                    split_report["escaped_examples"] = escaped[:10]
                referenced_images.update(
                    path
                    for path in split_paths
                    if path == dataset.root or dataset.root in path.parents
                )
                missing = [str(path) for path in split_paths if not path.is_file()]
                split_report["missing_referenced_images"] = len(missing)
                split_report["missing_examples"] = missing[:10]
                if missing:
                    report["errors"].append(
                        "{}: {} referenced images are missing".format(name, len(missing))
                    )
        except Exception as exc:
            inventory_complete = False
            split_report["status"] = "invalid_annotation"
            split_report["error"] = str(exc)
            report["errors"].append("{}: {}".format(name, exc))
        report["splits"][name] = split_report

    if deep and "train" in split_references and "val" in split_references:
        overlap = sorted(
            split_references["train"].intersection(split_references["val"])
        )
        report["split_overlap"] = {
            "train_val_images": len(overlap),
            "examples": [str(path) for path in overlap[:10]],
        }
        if overlap:
            report["errors"].append(
                "train/val split leakage: {} shared image paths".format(
                    len(overlap)
                )
            )

    if deep and inventory_complete:
        physical_images = {
            path.resolve()
            for path in dataset.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        unreferenced = sorted(physical_images - referenced_images)
        report["inventory"] = {
            "physical_images": len(physical_images),
            "referenced_images": len(referenced_images),
            "unreferenced_images": len(unreferenced),
            "unreferenced_bytes": sum(path.stat().st_size for path in unreferenced),
            "unreferenced_examples": [str(path) for path in unreferenced[:10]],
        }
        if unreferenced:
            report["warnings"].append(
                "{} image files are not referenced by any registered split".format(
                    len(unreferenced)
                )
            )

    if report["errors"]:
        report["status"] = "invalid"
    elif report["warnings"]:
        report["status"] = "valid_with_warnings"
    return report


def download_dataset(dataset: DatasetSpec, full: bool = False) -> List[Path]:
    records = dataset.download.get("archives", {})
    if dataset.id == "bdd100k":
        records = {"mirror": dataset.download.get("mirror", {})}
    downloaded: List[Path] = []
    for name, record in records.items():
        if not isinstance(record, Mapping) or not record.get("url"):
            continue
        if not full and not bool(record.get("benchmark_required", True)):
            continue
        destination = (project_root() / "data" / str(record["target"])).resolve()
        expected = record.get("md5")
        if destination.is_file() and (
            not expected or file_digest(destination, "md5").lower() == str(expected).lower()
        ):
            downloaded.append(destination)
            continue
        for _ in iter_download(
            str(record["url"]),
            destination,
            expected_digest=str(expected) if expected else None,
            digest_algorithm="md5",
        ):
            pass
        downloaded.append(destination)
    return downloaded


def _safe_members(destination: Path, names: Iterable[str]) -> None:
    root = destination.resolve()
    for name in names:
        target = (destination / name).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError("Archive member escapes destination: {}".format(name))


def extract_dataset(dataset: DatasetSpec) -> List[Path]:
    raw_root = project_root() / "data" / "raw"
    extracted: List[Path] = []
    if dataset.id == "voc":
        destination = project_root() / "data"
        records = dataset.download.get("archives", {})
        for record in records.values():
            archive = project_root() / "data" / str(record["target"])
            if not archive.is_file():
                raise FileNotFoundError("Missing VOC archive: {}".format(archive))
            with tarfile.open(str(archive), "r") as handle:
                names = handle.getnames()
                _safe_members(destination, names)
                handle.extractall(str(destination))
            extracted.append(archive)
    elif dataset.id == "coco":
        destination = dataset.root
        records = dataset.download.get("archives", {})
        for record in records.values():
            archive = project_root() / "data" / str(record["target"])
            if not archive.is_file():
                continue
            with zipfile.ZipFile(str(archive)) as handle:
                _safe_members(destination, handle.namelist())
                handle.extractall(str(destination))
            extracted.append(archive)
    elif dataset.id == "bdd100k":
        record = dataset.download.get("mirror", {})
        archive = project_root() / "data" / str(record.get("target", ""))
        if not archive.is_file():
            raise FileNotFoundError("Missing BDD100K mirror archive: {}".format(archive))
        # The verified mirror stores every member below a single
        # ``bdd100k/`` prefix. Extract beside dataset.root to avoid creating
        # data/bdd100k/bdd100k/... . Older mirrors without that prefix still
        # extract directly into dataset.root.
        with zipfile.ZipFile(str(archive)) as handle:
            names = handle.namelist()
            prefixed = bool(names) and all(
                name == "bdd100k" or name.startswith("bdd100k/") for name in names
            )
            destination = dataset.root.parent if prefixed else dataset.root
            destination.mkdir(parents=True, exist_ok=True)
            _safe_members(destination, names)
            handle.extractall(str(destination))
        extracted.append(archive)
    return extracted


def prepare_dataset(dataset: DatasetSpec) -> List[Path]:
    if dataset.id == "voc":
        return prepare_voc(dataset)
    if dataset.id == "bdd100k":
        return prepare_bdd100k(dataset)
    if dataset.id == "coco":
        return prepare_coco_dev(dataset)
    raise ValueError("No preparation handler for {}".format(dataset.id))


def minimize_dataset_storage(
    registry: Registry, apply: bool = False
) -> Dict[str, Any]:
    """Delete only paths that no registered experiment reads."""
    data_root = (project_root() / "data").resolve()
    prepare_coco_dev(registry.dataset("coco"))
    # A registered optional split is still an experiment dependency when it
    # exists.  In particular, the paired victim-training-state protocol uses
    # COCO train2017 but ordinary benchmark setup is allowed to omit it.  Do
    # not let the generic minimizer silently erase a present registered split.
    coco = registry.dataset("coco")
    keep_coco_annotations = {
        Path(split.annotation).name for split in coco.splits.values()
    }
    keep_coco_annotations.add("dev2017_selection.json")
    candidates: List[Path] = [
        data_root / "raw",
        data_root / "coco" / "test2017",
        data_root / "bdd100k" / "images" / "100k" / "test",
        data_root / "bdd100k" / "labels",
        data_root / "bdd100k" / "annotations" / "det_test_coco.json",
    ]
    coco_annotations = data_root / "coco" / "annotations"
    if coco_annotations.is_dir():
        candidates.extend(
            path
            for path in coco_annotations.iterdir()
            if path.name not in keep_coco_annotations
        )
    for year in ("VOC2007", "VOC2012"):
        root = data_root / "VOCdevkit" / year
        candidates.extend(
            root / name
            for name in (
                "Annotations",
                "ImageSets",
                "SegmentationClass",
                "SegmentationObject",
            )
        )
    # Registered COCO annotations are the source of truth for every retained
    # image. This catches individual leftovers (for example VOC2012 test
    # images) that cannot be represented by a removable top-level directory.
    for dataset_id in registry.datasets:
        dataset = registry.dataset(dataset_id)
        referenced = set()
        for split_name, split in dataset.splits.items():
            annotation = dataset.root / split.annotation
            image_root = dataset.root / split.image_prefix
            if (
                not split.benchmark_required
                and (not annotation.is_file() or not image_root.is_dir())
            ):
                continue
            index = CocoIndex(dataset, split_name)
            referenced.update(index.image_path(image) for image in index.images)
        candidates.extend(
            path
            for path in dataset.root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_SUFFIXES
            and path.resolve() not in referenced
        )
    candidates = list(dict.fromkeys(path.resolve() for path in candidates))
    targets: List[Dict[str, Any]] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if resolved == data_root or data_root not in resolved.parents:
            raise RuntimeError("Unsafe data-minimization target: {}".format(resolved))
        if resolved.is_dir():
            files = list(resolved.rglob("*"))
            bytes_total = sum(path.stat().st_size for path in files if path.is_file())
            file_count = sum(path.is_file() for path in files)
        else:
            bytes_total = resolved.stat().st_size
            file_count = 1
        targets.append(
            {
                "path": str(resolved),
                "bytes": int(bytes_total),
                "files": int(file_count),
            }
        )
    before = {
        dataset_id: validate_dataset(registry.dataset(dataset_id), deep=True)
        for dataset_id in registry.datasets
    }
    if any(not report["status"].startswith("valid") for report in before.values()):
        raise RuntimeError("Refusing to prune because retained data failed deep validation")
    if apply:
        for target in targets:
            path = Path(target["path"])
            if path.is_dir():
                shutil.rmtree(str(path))
            elif path.exists():
                path.unlink()
    after = (
        {
            dataset_id: validate_dataset(registry.dataset(dataset_id), deep=True)
            for dataset_id in registry.datasets
        }
        if apply
        else None
    )
    if after is not None and any(
        not report["status"].startswith("valid") for report in after.values()
    ):
        raise RuntimeError("Retained data failed validation after minimization")
    return {
        "status": "applied" if apply else "planned",
        "targets": targets,
        "bytes_reclaimed": sum(target["bytes"] for target in targets) if apply else 0,
        "bytes_reclaimable": sum(target["bytes"] for target in targets),
        "validation_before": before,
        "validation_after": after,
        "retained": {
            "coco": ["dev2017 (1000)", "val2017 (5000)"],
            "voc": [
                "VOC2007 trainval/test JPEGImages",
                "VOC2012 trainval JPEGImages",
                "COCO-format train/val annotations",
            ],
            "bdd100k": ["train images/annotations", "val images/annotations"],
        },
    }
