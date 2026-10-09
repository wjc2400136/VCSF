"""Metadata-only, isolated VOC training subsets; never a training recipe."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .registry import DatasetSpec


_SUBSET_FILENAME = "diagnostic_train_annotation.json"
_SELECTION_POLICY = "first_filtered_leaf_order_intersection_of_all_leaves_exact_n"
_SOURCE_KEYS = {"dataset", "datasets", "ann_file"}


def _image_id(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Diagnostic image IDs must be integers")
    return int(value)


def _unique_ids(values: Sequence[Any], context: str) -> List[int]:
    result = [_image_id(value) for value in values]
    if not result:
        raise ValueError("{} is empty".format(context))
    if len(set(result)) != len(result):
        raise ValueError("{} contains duplicate image IDs".format(context))
    return result


def _local_path(value: Any, root: Any = "") -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise ValueError("Dataset annotation/image path must be a nonempty local path")
    if not isinstance(root, (str, Path)):
        raise ValueError("Dataset data_root must be a local path")
    if "://" in str(value) or "://" in str(root):
        raise ValueError("Diagnostic datasets require local files")
    path = Path(value)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def _reject_hidden_config_sources(value: Any, active: Optional[set] = None) -> None:
    if not isinstance(value, (Mapping, list, tuple)):
        return
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("Cyclic dataset configuration")
    active.add(id(value))
    try:
        if isinstance(value, Mapping):
            if _SOURCE_KEYS.intersection(value):
                raise ValueError("Hidden dataset or annotation in dataset options")
            children = value.values()
        else:
            children = value
        for child in children:
            _reject_hidden_config_sources(child, active)
    finally:
        active.remove(id(value))


def _config_children(node: Any) -> List[Tuple[str, MutableMapping[str, Any]]]:
    if not isinstance(node, MutableMapping):
        raise ValueError("Every dataset config node must be a mutable mapping")
    if not node.get("type"):
        raise ValueError("Every dataset config node requires a registered type")
    children = []
    if "dataset" in node:
        if not isinstance(node["dataset"], MutableMapping):
            raise ValueError("Dataset wrapper .dataset must be a config mapping")
        children.append((".dataset", node["dataset"]))
    if "datasets" in node:
        if not isinstance(node["datasets"], (list, tuple)) or not node["datasets"]:
            raise ValueError("Dataset wrapper .datasets must be a nonempty sequence")
        for index, child in enumerate(node["datasets"]):
            if not isinstance(child, MutableMapping):
                raise ValueError("Every .datasets child must be a config mapping")
            children.append((".datasets[{}]".format(index), child))
    if children and "ann_file" in node:
        raise ValueError("Wrapper annotation cannot be proven to be a leaf source")
    for key, value in node.items():
        if key not in _SOURCE_KEYS:
            _reject_hidden_config_sources(value)
    return children


def _config_leaves(node: Any) -> List[MutableMapping[str, Any]]:
    leaves = []
    active = set()

    def visit(current: Any) -> None:
        if id(current) in active:
            raise ValueError("Cyclic dataset configuration")
        active.add(id(current))
        try:
            children = _config_children(current)
            if children:
                for _, child in children:
                    visit(child)
            else:
                leaves.append(current)
        finally:
            active.remove(id(current))

    visit(node)
    return leaves


def _has_dataset_object(value: Any, seen: Optional[set] = None) -> bool:
    if callable(getattr(value, "get_data_info", None)) or callable(
        getattr(value, "full_init", None)
    ):
        return True
    if not isinstance(value, (Mapping, list, tuple)):
        return False
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    children = value.values() if isinstance(value, Mapping) else value
    return any(_has_dataset_object(child, seen) for child in children)


def _runtime_children(node: Any) -> List[Tuple[str, Any]]:
    if isinstance(node, (Mapping, list, tuple)) or node is None:
        raise ValueError("Unverifiable runtime dataset structure")
    children = []
    if hasattr(node, "dataset"):
        child = node.dataset
        if child is None:
            raise ValueError("Runtime wrapper .dataset is empty")
        children.append((".dataset", child))
    if hasattr(node, "datasets"):
        datasets = node.datasets
        if not isinstance(datasets, (list, tuple)) or not datasets:
            raise ValueError("Runtime wrapper .datasets is empty or unverifiable")
        children.extend(
            (".datasets[{}]".format(index), child)
            for index, child in enumerate(datasets)
        )
    for key, value in getattr(node, "__dict__", {}).items():
        if key not in {"dataset", "datasets"} and _has_dataset_object(value):
            raise ValueError("Hidden runtime dataset outside .dataset/.datasets")
    if children and getattr(node, "ann_file", None):
        raise ValueError("Runtime wrapper exposes an unverified annotation source")
    return children


def _inspect_runtime(
    runtime_dataset: Any,
    config: Optional[MutableMapping[str, Any]] = None,
    annotation: Optional[Path] = None,
    image_paths: Optional[Mapping[int, Path]] = None,
) -> List[Dict[str, Any]]:
    leaves = []
    nodes = []
    active = set()

    def visit(node: Any, cfg_node: Any, location: str) -> None:
        if id(node) in active:
            raise ValueError("Cyclic runtime dataset structure")
        active.add(id(node))
        try:
            children = _runtime_children(node)
            cfg_children = _config_children(cfg_node) if cfg_node is not None else []
            if cfg_node is not None and [key for key, _ in children] != [
                key for key, _ in cfg_children
            ]:
                raise ValueError("Config/runtime dataset wrapper topology mismatch")
            if not callable(getattr(node, "full_init", None)):
                raise ValueError("Runtime dataset lacks metadata-only full_init")
            if not children:
                if not callable(getattr(node, "get_data_info", None)):
                    raise ValueError("Cannot prove runtime dataset leaf: missing get_data_info")
                check_annotation(node)
            if children:
                for index, (key, child) in enumerate(children):
                    child_cfg = cfg_children[index][1] if cfg_node is not None else None
                    visit(child, child_cfg, location + key)
            nodes.append((node, location, children))
        finally:
            active.remove(id(node))

    def check_annotation(node: Any) -> None:
        ann_path = _local_path(
            getattr(node, "ann_file", None), getattr(node, "data_root", "")
        )
        if not ann_path.is_file():
            raise ValueError("Runtime dataset leaf annotation is not a local file")
        if annotation is not None and ann_path != annotation:
            raise ValueError("Runtime leaf annotation differs from declared source")

    # Reject cycles before full_init, then read metadata only after all wrappers
    # have initialized. Wrapper initialization must not invalidate the tree audit.
    visit(runtime_dataset, config, "train_dataloader.dataset")
    initialized = set()
    for node, _, _ in nodes:
        if id(node) not in initialized:
            node.full_init()
            initialized.add(id(node))
    for node, location, children in nodes:
        if [(key, id(child)) for key, child in _runtime_children(node)] != [
            (key, id(child)) for key, child in children
        ]:
            raise ValueError("Runtime dataset topology changed during full_init")
        count = len(node)
        if count <= 0:
            raise ValueError("Runtime dataset is empty after filtering")
        if children:
            continue
        check_annotation(node)
        ids = []
        for index in range(count):
            info = node.get_data_info(index)
            if not isinstance(info, Mapping) or "img_id" not in info:
                raise ValueError("Runtime leaf metadata is missing img_id")
            image_id = _image_id(info["img_id"])
            if image_paths is not None:
                if image_id not in image_paths:
                    raise ValueError("Runtime image ID is absent from source annotation")
                if _local_path(info.get("img_path")) != image_paths[image_id]:
                    raise ValueError("Runtime image path differs from source image path")
            ids.append(image_id)
        ids = _unique_ids(ids, "Runtime leaf")
        leaves.append({"leaf": location, "image_ids": ids, "images": len(ids)})
    if not leaves:
        raise ValueError("No verifiable runtime dataset leaves")
    return leaves


def _build_dataset(node: MutableMapping[str, Any], scope: Optional[str]) -> Any:
    from mmengine.registry import DATASETS

    # build_runtime_config registers modules without setting the global scope.
    build_cfg = deepcopy(node)
    if scope is not None:
        build_cfg.setdefault("_scope_", scope)
    return DATASETS.build(build_cfg)


def _source_payload(raw: bytes) -> Tuple[Dict[str, Any], List[int]]:
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Source annotation must be a COCO JSON object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list) or not all(
            isinstance(item, dict) for item in payload[key]
        ):
            raise ValueError("Source annotation requires COCO {} records".format(key))
    ids = _unique_ids([item.get("id") for item in payload["images"]], "Source images")
    image_ids = set(ids)
    category_ids = set()
    for item in payload["categories"]:
        category_id = _image_id(item.get("id"))
        if category_id in category_ids:
            raise ValueError("Source annotation contains duplicate category IDs")
        category_ids.add(category_id)
    annotation_ids = set()
    for item in payload["annotations"]:
        annotation_id = _image_id(item.get("id"))
        if annotation_id in annotation_ids:
            raise ValueError("Source annotation contains duplicate annotation IDs")
        annotation_ids.add(annotation_id)
        if _image_id(item.get("image_id")) not in image_ids:
            raise ValueError("Source annotation references an unknown image ID")
        if _image_id(item.get("category_id")) not in category_ids:
            raise ValueError("Source annotation references an unknown category ID")
    return payload, ids


def _verified_receipt(leaves: List[Dict[str, Any]], ids: List[int]) -> Dict[str, Any]:
    expected = set(ids)
    for leaf in leaves:
        if set(leaf["image_ids"]) != expected:
            raise ValueError("Runtime leaf image ID set does not equal diagnostic subset")
    return {
        "image_ids": list(ids),
        "images": len(ids),
        "metadata_only": True,
        "leaves": [dict(leaf, exact_id_set=True) for leaf in leaves],
    }


def verify_diagnostic_dataset(
    runtime_dataset: Any, expected_ids: Sequence[int]
) -> Dict[str, Any]:
    """Recheck Runner's actual dataset leaves without calling __getitem__.

    Call after Runner has built train_dataloader.dataset and before training.
    Repeated wrapper visits are allowed; duplicate IDs within a true leaf are not.
    This does not execute or certify image transforms.
    """
    ids = _unique_ids(expected_ids, "Expected diagnostic IDs")
    return _verified_receipt(_inspect_runtime(runtime_dataset), ids)


def prepare_diagnostic_dataset(
    cfg: Any, dataset: DatasetSpec, work_dir: Path, max_images: int
) -> Tuple[DatasetSpec, Dict[str, Any]]:
    """Prepare an exact-N VOC subset after runtime config construction.

    Call before dumping protocol_config.py. Framework modules must already be
    registered. Only cfg training-leaf ann_file values are changed, on success;
    all recipe settings and the original DatasetSpec stay intact. The returned
    DatasetSpec's train split must also be used for checkpoint verification.

    The one output file is created exclusively in a fresh run's work directory.
    Failed preparation retains any written evidence; retry in a new directory.
    Receipts contain IDs/hashes and structural locations, never filesystem paths.
    """
    if dataset.id != "voc":
        raise ValueError("Training diagnostics are authorized for VOC only")
    if (
        isinstance(max_images, bool)
        or not isinstance(max_images, Integral)
        or max_images <= 0
    ):
        raise ValueError("max_images must be a positive integer")
    max_images = int(max_images)
    work_dir = Path(work_dir).resolve()
    subset_path = work_dir / _SUBSET_FILENAME
    if subset_path.exists() or subset_path.is_symlink():
        raise FileExistsError("Diagnostic subset already exists; use a new work directory")
    train_split = dataset.split("train")
    source_path = _local_path(train_split.annotation, dataset.root)
    if source_path == subset_path:
        raise ValueError("Diagnostic output cannot replace its source annotation")
    source_raw = source_path.read_bytes()
    source_hash = hashlib.sha256(source_raw).hexdigest()

    def check_source() -> None:
        if (
            _local_path(train_split.annotation, dataset.root) != source_path
            or source_path.read_bytes() != source_raw
        ):
            raise ValueError("Source annotation changed during diagnostic preparation")

    payload, _ = _source_payload(source_raw)
    image_root = _local_path(train_split.image_prefix or ".", dataset.root)
    image_paths = {
        item["id"]: _local_path(item.get("file_name"), image_root)
        for item in payload["images"]
    }
    loader = cfg.get("train_dataloader")
    if not isinstance(loader, MutableMapping):
        raise ValueError("Training config requires a train_dataloader mapping")
    original_node = loader.get("dataset")
    original_leaves = _config_leaves(original_node)
    for leaf in original_leaves:
        root = leaf.get("data_root", "")
        if _local_path(leaf.get("ann_file"), root) != source_path:
            raise ValueError("Every config leaf must use the declared train annotation")
        prefix = leaf.get("data_prefix")
        if not isinstance(prefix, Mapping) or "img" not in prefix:
            raise ValueError("Every config leaf requires a verifiable img data_prefix")
        if _local_path(prefix["img"] or ".", root) != image_root:
            raise ValueError("Config image prefix differs from declared train image prefix")
        if leaf.get("backend_args") or leaf.get("file_client_args"):
            raise ValueError("External or remapped annotation backends cannot be verified")
    scope = cfg.get("default_scope")
    original_runtime = _build_dataset(original_node, scope)
    original_checks = _inspect_runtime(
        original_runtime, original_node, source_path, image_paths
    )
    del original_runtime
    common = set(original_checks[0]["image_ids"])
    for leaf in original_checks[1:]:
        common.intersection_update(leaf["image_ids"])
    selected = [value for value in original_checks[0]["image_ids"] if value in common][
        :max_images
    ]
    if len(selected) != max_images:
        raise ValueError("Filtered leaf intersection has fewer than max_images unique IDs")
    selected_set = set(selected)
    subset_payload = dict(payload)
    subset_payload["images"] = [
        item for item in payload["images"] if item["id"] in selected_set
    ]
    subset_payload["annotations"] = [
        item for item in payload["annotations"] if item["image_id"] in selected_set
    ]
    subset_raw = (
        json.dumps(subset_payload, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    check_source()
    work_dir.mkdir(parents=True, exist_ok=True)
    with subset_path.open("xb") as handle:
        handle.write(subset_raw)
    subset_node = deepcopy(original_node)
    for leaf in _config_leaves(subset_node):
        leaf["ann_file"] = str(subset_path)
    subset_runtime = _build_dataset(subset_node, scope)
    subset_checks = _inspect_runtime(
        subset_runtime, subset_node, subset_path, image_paths
    )
    receipt = _verified_receipt(subset_checks, selected)
    receipt.update(
        source_annotation_sha256=source_hash,
        subset_annotation_sha256=hashlib.sha256(subset_raw).hexdigest(),
        selection_policy=_SELECTION_POLICY,
        source_leaves=original_checks,
    )
    new_splits = dict(dataset.splits)
    new_splits["train"] = replace(
        train_split, annotation=str(subset_path), expected_images=max_images
    )
    diagnostic_dataset = replace(dataset, splits=new_splits)
    # Bind both raw byte identities after builders/full_init have finished.
    if subset_path.read_bytes() != subset_raw:
        raise ValueError("Subset annotation changed during diagnostic preparation")
    check_source()
    for leaf in original_leaves:
        leaf["ann_file"] = str(subset_path)
    return diagnostic_dataset, receipt
