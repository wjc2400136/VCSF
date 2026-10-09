from __future__ import annotations

import importlib.util
from importlib import import_module
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

from mmengine.config import Config

from .registry import DatasetSpec, ModelSpec


def register_framework(framework: str) -> None:
    """Register modules without mutating MMEngine's process-global default scope.

    Every generated config carries its own ``default_scope``.  Leaving global scope
    selection to ``Runner``/``init_detector`` avoids order-dependent mmdet↔mmyolo
    warnings when a benchmark process resolves models from both registries.
    """
    if framework == "mmyolo":
        from mmyolo.utils import register_all_modules

        register_all_modules(init_default_scope=False)
    elif framework == "mmdet":
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=False)
    else:
        raise ValueError("Unsupported framework: {}".format(framework))


def package_root(framework: str) -> Path:
    spec = importlib.util.find_spec(framework)
    if spec is None or not spec.submodule_search_locations:
        raise FileNotFoundError("Python package is not installed: {}".format(framework))
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def resolve_upstream_config(model: ModelSpec) -> Path:
    root = package_root(model.framework)
    candidate = (root / ".mim" / model.config).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            "Pinned upstream config is missing for {}: {}".format(model.id, candidate)
        )
    return candidate


def _walk_patch_num_classes(value: Any, num_classes: int) -> None:
    if isinstance(value, MutableMapping):
        for key, child in list(value.items()):
            if key == "num_classes" and isinstance(child, int):
                value[key] = num_classes
            else:
                _walk_patch_num_classes(child, num_classes)
    elif isinstance(value, list):
        for child in value:
            _walk_patch_num_classes(child, num_classes)


def _patch_class_dependent_loss_weights(cfg: Config, num_classes: int) -> None:
    """Recompute values that upstream configs derive before Config resolution.

    The YOLOv5 config evaluates its classification-loss expression while
    loading the 80-class COCO config. Recursively replacing ``num_classes``
    afterwards is therefore insufficient: the already-resolved scalar must be
    scaled by the new-to-old class-count ratio as well.
    """
    model = cfg.get("model")
    if not isinstance(model, MutableMapping):
        return
    bbox_head = model.get("bbox_head")
    if not isinstance(bbox_head, MutableMapping):
        return
    head_module = bbox_head.get("head_module")
    loss_cls = bbox_head.get("loss_cls")
    if not isinstance(head_module, MutableMapping) or not isinstance(
        loss_cls, MutableMapping
    ):
        return
    if head_module.get("type") != "YOLOv5HeadModule":
        return
    old_classes = head_module.get("num_classes")
    old_weight = loss_cls.get("loss_weight")
    if (
        not isinstance(old_classes, int)
        or old_classes <= 0
        or not isinstance(old_weight, (int, float))
    ):
        raise RuntimeError(
            "Resolved YOLOv5 config is missing its class-dependent loss metadata"
        )
    loss_cls["loss_weight"] = float(old_weight) * (
        float(num_classes) / float(old_classes)
    )


def _patch_bbox_only(cfg: Config) -> None:
    """Turn an instance-segmentation config into its bbox-only detector form."""
    model = cfg.get("model")
    if isinstance(model, MutableMapping):
        roi_head = model.get("roi_head")
        if isinstance(roi_head, MutableMapping):
            roi_head.pop("mask_head", None)
            roi_head.pop("mask_roi_extractor", None)

    def patch_pipeline(value: Any) -> None:
        if isinstance(value, MutableMapping):
            if value.get("type") == "LoadAnnotations":
                value["with_bbox"] = True
                value["with_mask"] = False
                value["with_seg"] = False
            for child in value.values():
                patch_pipeline(child)
        elif isinstance(value, list):
            for child in value:
                patch_pipeline(child)

    patch_pipeline(cfg._cfg_dict)


def _leaf_dataset(node: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    current = node
    while isinstance(current, MutableMapping) and "dataset" in current:
        child = current["dataset"]
        if not isinstance(child, MutableMapping):
            break
        current = child
    return current


def _patch_dataset_node(
    node: MutableMapping[str, Any],
    dataset: DatasetSpec,
    split_name: str,
    test_mode: bool,
) -> None:
    split = dataset.split(split_name)
    if "datasets" in node and isinstance(node["datasets"], list):
        for child in node["datasets"]:
            if isinstance(child, MutableMapping):
                _patch_dataset_node(child, dataset, split_name, test_mode)
        return
    leaf = _leaf_dataset(node)
    leaf["data_root"] = str(dataset.root) + "/"
    leaf["ann_file"] = split.annotation
    leaf["data_prefix"] = {"img": split.image_prefix}
    leaf["metainfo"] = {"classes": tuple(dataset.classes)}
    leaf["test_mode"] = test_mode


def _patch_dataloader(
    cfg: Config,
    key: str,
    dataset: DatasetSpec,
    split_name: str,
    test_mode: bool,
    batch_size: Optional[int] = None,
) -> None:
    loader = cfg.get(key)
    if not isinstance(loader, MutableMapping):
        return
    if batch_size is not None:
        loader["batch_size"] = batch_size
    loader["num_workers"] = min(int(loader.get("num_workers", 2)), 2)
    loader["persistent_workers"] = False
    node = loader.get("dataset")
    if isinstance(node, MutableMapping):
        _patch_dataset_node(node, dataset, split_name, test_mode)


def _patch_evaluator(cfg: Config, key: str, dataset: DatasetSpec, split_name: str) -> None:
    evaluator = cfg.get(key)
    if evaluator is None:
        return
    evaluators = evaluator if isinstance(evaluator, list) else [evaluator]
    split = dataset.split(split_name)
    ann_file = str((dataset.root / split.annotation).resolve())
    for item in evaluators:
        if isinstance(item, MutableMapping) and "ann_file" in item:
            item["ann_file"] = ann_file
            if item.get("metric") != "bbox":
                item["metric"] = "bbox"


def _scale_epoch_scheduler(cfg: Config, old_epochs: int, new_epochs: int) -> None:
    schedulers = cfg.get("param_scheduler")
    if not isinstance(schedulers, list) or old_epochs <= 0:
        return
    ratio = float(new_epochs) / float(old_epochs)
    for scheduler in schedulers:
        if not isinstance(scheduler, MutableMapping):
            continue
        if scheduler.get("by_epoch", True) is False:
            continue
        for key in ("begin", "end", "T_max"):
            if key in scheduler and isinstance(scheduler[key], int):
                scheduler[key] = max(1 if key != "begin" else 0, int(round(scheduler[key] * ratio)))
        if isinstance(scheduler.get("milestones"), list):
            scheduler["milestones"] = [
                max(1, int(round(value * ratio))) for value in scheduler["milestones"]
            ]


def _scaled_epoch(value: int, ratio: float, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    return max(minimum, int(round(int(value) * ratio)))


def _scale_epoch_coupled_settings(
    cfg: Config, old_epochs: int, new_epochs: int
) -> None:
    """Scale resolved MMYOLO settings that remain tied to the upstream schedule."""
    if old_epochs <= 0:
        return
    ratio = float(new_epochs) / float(old_epochs)
    train_cfg = cfg.get("train_cfg")
    if isinstance(train_cfg, MutableMapping):
        dynamic_intervals = train_cfg.get("dynamic_intervals")
        if isinstance(dynamic_intervals, list):
            scaled_intervals = []
            for item in dynamic_intervals:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], int)
                ):
                    scaled_intervals.append(
                        (_scaled_epoch(item[0], ratio), item[1])
                    )
                else:
                    scaled_intervals.append(item)
            train_cfg["dynamic_intervals"] = scaled_intervals

    default_hooks = cfg.get("default_hooks")
    if isinstance(default_hooks, MutableMapping):
        scheduler_hook = default_hooks.get("param_scheduler")
        if (
            isinstance(scheduler_hook, MutableMapping)
            and isinstance(scheduler_hook.get("max_epochs"), int)
        ):
            scheduler_hook["max_epochs"] = new_epochs

    custom_hooks = cfg.get("custom_hooks")
    if isinstance(custom_hooks, list):
        for hook in custom_hooks:
            if not isinstance(hook, MutableMapping):
                continue
            if isinstance(hook.get("switch_epoch"), int):
                hook["switch_epoch"] = _scaled_epoch(
                    hook["switch_epoch"], ratio
                )
            if isinstance(hook.get("num_last_epochs"), int):
                hook["num_last_epochs"] = _scaled_epoch(
                    hook["num_last_epochs"], ratio
                )
            if isinstance(hook.get("max_epochs"), int):
                hook["max_epochs"] = new_epochs

    for key in ("num_last_epochs", "save_epoch_intervals"):
        if isinstance(cfg.get(key), int):
            cfg[key] = _scaled_epoch(cfg[key], ratio)
    if isinstance(cfg.get("max_epochs"), int):
        cfg["max_epochs"] = new_epochs


def _patch_optimizer_batch_size(cfg: Config, batch_size: int) -> None:
    optim_wrapper = cfg.get("optim_wrapper")
    if not isinstance(optim_wrapper, MutableMapping):
        return
    optimizer = optim_wrapper.get("optimizer")
    if (
        isinstance(optimizer, MutableMapping)
        and isinstance(optimizer.get("batch_size_per_gpu"), int)
    ):
        optimizer["batch_size_per_gpu"] = batch_size


def _ensure_custom_import(cfg: Config, module: str) -> None:
    custom_imports = cfg.get("custom_imports")
    if custom_imports is None:
        cfg.custom_imports = {
            "imports": [],
            "allow_failed_imports": False,
        }
        custom_imports = cfg.get("custom_imports")
    if not isinstance(custom_imports, MutableMapping):
        raise RuntimeError("Resolved config custom_imports must be a mapping")
    imports = custom_imports.get("imports", [])
    if not isinstance(imports, (list, tuple)):
        raise RuntimeError("Resolved config custom_imports.imports must be a list")
    imports = list(imports)
    if module not in imports:
        imports.append(module)
    custom_imports["imports"] = imports
    custom_imports["allow_failed_imports"] = False


def _patch_reppoints_point_assigner(cfg: Config, model: ModelSpec) -> None:
    """Apply the audited MMDetection 3.0.0 RepPoints device correction."""
    if model.id != "reppoints_r50":
        return
    if model.framework != "mmdet":
        raise RuntimeError("RepPoints compatibility requires the mmdet framework")

    model_cfg = cfg.get("model")
    train_cfg = (
        model_cfg.get("train_cfg")
        if isinstance(model_cfg, MutableMapping)
        else None
    )
    init_cfg = (
        train_cfg.get("init")
        if isinstance(train_cfg, MutableMapping)
        else None
    )
    assigner = (
        init_cfg.get("assigner")
        if isinstance(init_cfg, MutableMapping)
        else None
    )
    if not isinstance(assigner, MutableMapping):
        raise RuntimeError(
            "Resolved RepPoints config is missing model.train_cfg.init.assigner"
        )
    if assigner.get("type") != "PointAssigner":
        raise RuntimeError(
            "Resolved RepPoints init assigner changed unexpectedly: {}".format(
                assigner.get("type")
            )
        )

    from .mmdet_compat import validate_point_assigner_compatibility

    audit = validate_point_assigner_compatibility()
    _ensure_custom_import(cfg, "lgp.mmdet_compat")

    cfg.lgp_framework_compatibility = {
        "reppoints_point_assigner": {
            **audit,
            "config_type": "PointAssigner",
            "registry_binding": "LGPDeviceSafePointAssigner",
            "semantic_change": "points_range allocated on points.device",
        }
    }


def _patch_model_gradient_clipping(cfg: Config, model: ModelSpec) -> None:
    recipe = model.train_grad_clip
    if recipe is None:
        return
    optim_wrapper = cfg.get("optim_wrapper")
    if not isinstance(optim_wrapper, MutableMapping):
        raise RuntimeError(
            "{} training config has no optim_wrapper for gradient clipping".format(
                model.id
            )
        )
    expected = {
        "max_norm": float(recipe["max_norm"]),
        "norm_type": int(recipe["norm_type"]),
    }
    existing = optim_wrapper.get("clip_grad")
    if existing is not None:
        if not isinstance(existing, MutableMapping):
            raise RuntimeError(
                "{} upstream clip_grad must be a mapping".format(model.id)
            )
        normalized = {
            "max_norm": float(existing.get("max_norm")),
            "norm_type": int(existing.get("norm_type")),
        }
        if normalized != expected:
            raise RuntimeError(
                "{} upstream gradient clipping conflicts with the registered "
                "author recipe: expected {!r}, found {!r}".format(
                    model.id, expected, normalized
                )
            )
    optim_wrapper["clip_grad"] = expected
    cfg.lgp_model_training_recipe = {
        "gradient_clipping": dict(recipe),
    }


def _patch_training_safety(
    cfg: Config, training_safety: Mapping[str, Any]
) -> None:
    interval = int(training_safety["invalid_loss_check_interval"])
    if interval <= 0:
        raise ValueError("invalid_loss_check_interval must be positive")
    if training_safety["require_finite_checkpoint"] is not True:
        raise ValueError("require_finite_checkpoint must remain true")
    # ``custom_imports`` is consumed when a config is loaded from a file, but
    # LGP constructs this runtime Config in memory and passes it directly to
    # Runner.from_cfg.  Register the hook in this process as well as retaining
    # the declaration in the dumped config for standalone reproducibility.
    import_module("lgp.training_hooks")
    _ensure_custom_import(cfg, "lgp.training_hooks")

    hooks = cfg.get("custom_hooks")
    if hooks is None:
        cfg.custom_hooks = []
        hooks = cfg.get("custom_hooks")
    if not isinstance(hooks, list):
        raise RuntimeError("Resolved config custom_hooks must be a list")
    matching = [
        hook
        for hook in hooks
        if isinstance(hook, MutableMapping)
        and hook.get("type") == "LGPCheckFiniteLossHook"
    ]
    if len(matching) > 1:
        raise RuntimeError("Resolved config duplicates LGPCheckFiniteLossHook")
    if matching:
        matching[0]["interval"] = interval
    else:
        hooks.append(
            {
                "type": "LGPCheckFiniteLossHook",
                "interval": interval,
            }
        )
    cfg.lgp_training_safety = dict(training_safety)


def _patch_semantic_head_initialization(
    cfg: Config,
    model: ModelSpec,
    dataset: DatasetSpec,
    source_dataset: Optional[DatasetSpec],
    work_dir: Path,
) -> None:
    declared = model.finetune_head_init.get(dataset.id)
    if declared is None:
        return
    policy = dict(declared)
    if source_dataset is None:
        raise ValueError(
            "{} training on {} requires its declared warm-start source "
            "dataset".format(model.id, dataset.id)
        )
    if str(policy["source_dataset"]) != source_dataset.id:
        raise ValueError(
            "{} head initialization expects source dataset {}, got {}".format(
                model.id, policy["source_dataset"], source_dataset.id
            )
        )
    hooks = cfg.get("custom_hooks")
    if not isinstance(hooks, list):
        raise RuntimeError("Resolved config custom_hooks must be a list")
    matching = [
        hook
        for hook in hooks
        if isinstance(hook, MutableMapping)
        and hook.get("type") == "LGPSemanticClassHeadInitHook"
    ]
    if matching:
        raise RuntimeError("Resolved config duplicates LGPSemanticClassHeadInitHook")
    hooks.insert(
        0,
        {
            "type": "LGPSemanticClassHeadInitHook",
            "priority": "VERY_HIGH",
            "policy": str(policy["policy"]),
            "source_dataset": source_dataset.id,
            "target_dataset": dataset.id,
            "parameter_stem": str(policy["parameter_stem"]),
            "source_classes": list(source_dataset.classes),
            "target_classes": list(dataset.classes),
            "target_to_source": dict(policy["target_to_source"]),
            "audit_path": str(
                (work_dir / "warm_start_load_audit.json").resolve()
            ),
        },
    )
    cfg.lgp_head_initialization = {
        **policy,
        "source_classes": list(source_dataset.classes),
        "target_classes": list(dataset.classes),
        "audit_file": "warm_start_load_audit.json",
    }


def _disable_redundant_pretrained_initializers(value: Any) -> None:
    """Avoid network warm-starts when a complete detector checkpoint follows."""
    if isinstance(value, MutableMapping):
        init_cfg = value.get("init_cfg")
        if isinstance(init_cfg, MutableMapping):
            if str(init_cfg.get("type", "")).lower() == "pretrained":
                value["init_cfg"] = None
        elif isinstance(init_cfg, list):
            retained = [
                item
                for item in init_cfg
                if not (
                    isinstance(item, MutableMapping)
                    and str(item.get("type", "")).lower() == "pretrained"
                )
            ]
            value["init_cfg"] = retained or None
        for child in value.values():
            _disable_redundant_pretrained_initializers(child)
    elif isinstance(value, list):
        for child in value:
            _disable_redundant_pretrained_initializers(child)


def resume_phase_switch_boundary(cfg: Config) -> Optional[int]:
    """Return the first one-based checkpoint epoch unsafe for naive resume.

    MMYOLO's pipeline/mode switch hooks are equality-triggered and do not
    serialize their mutated dataloader/model flags. A checkpoint saved after
    such a switch cannot be resumed by simply rebuilding the stage-one config.
    """
    train_cfg = cfg.get("train_cfg")
    max_epochs = (
        int(train_cfg.get("max_epochs"))
        if isinstance(train_cfg, MutableMapping)
        and isinstance(train_cfg.get("max_epochs"), int)
        else None
    )
    boundaries = []
    hooks = cfg.get("custom_hooks")
    if not isinstance(hooks, list):
        return None
    for hook in hooks:
        if not isinstance(hook, MutableMapping):
            continue
        hook_type = str(hook.get("type", ""))
        if (
            hook_type == "PipelineSwitchHook"
            and isinstance(hook.get("switch_epoch"), int)
        ):
            boundaries.append(int(hook["switch_epoch"]) + 1)
        elif (
            hook_type == "YOLOXModeSwitchHook"
            and max_epochs is not None
            and isinstance(hook.get("num_last_epochs"), int)
        ):
            boundaries.append(max_epochs - int(hook["num_last_epochs"]))
    return min(boundaries) if boundaries else None


def build_runtime_config(
    model: ModelSpec,
    dataset: DatasetSpec,
    work_dir: Path,
    mode: str = "test",
    test_split: str = "val",
    training_protocol: Optional[Mapping[str, Any]] = None,
    training_safety: Optional[Mapping[str, Any]] = None,
    warm_start_source: Optional[DatasetSpec] = None,
    dump_config: bool = True,
) -> Config:
    """Load an immutable upstream config and apply minimal dataset overrides."""
    if mode not in {"test", "train"}:
        raise ValueError("mode must be 'test' or 'train'")
    dataset.split(test_split)
    register_framework(model.framework)
    cfg = Config.fromfile(str(resolve_upstream_config(model)))
    cfg.default_scope = model.framework
    cfg.work_dir = str(work_dir.resolve())
    cfg.load_from = None
    cfg.resume = False
    if model.bbox_only:
        _patch_bbox_only(cfg)
    _patch_class_dependent_loss_weights(cfg, dataset.num_classes)
    _walk_patch_num_classes(cfg._cfg_dict, dataset.num_classes)

    _patch_dataloader(cfg, "test_dataloader", dataset, test_split, True, 1)
    _patch_dataloader(cfg, "val_dataloader", dataset, "val", True, 1)
    _patch_evaluator(cfg, "test_evaluator", dataset, test_split)
    _patch_evaluator(cfg, "val_evaluator", dataset, "val")

    if mode == "train":
        if training_safety is None:
            raise ValueError(
                "training_safety is required when building a training config"
            )
        _patch_reppoints_point_assigner(cfg, model)
        _patch_model_gradient_clipping(cfg, model)
        _patch_training_safety(cfg, training_safety)
        _patch_semantic_head_initialization(
            cfg, model, dataset, warm_start_source, work_dir
        )
        # ``run_training`` always supplies a verified full COCO detector
        # checkpoint. Downloading an auxiliary ImageNet backbone first is both
        # redundant and an avoidable network dependency; non-pretrained
        # initializers for newly resized class heads remain untouched.
        _disable_redundant_pretrained_initializers(cfg.get("model"))
        protocol = {
            "seed": 42,
            # PyTorch 2.0's deterministic CUDA indexing path crashes in
            # MMDetection anchor target assignment. The seed is fixed while
            # cuDNN benchmarking remains disabled; exact bitwise replay is not
            # claimed for this pinned stack.
            "deterministic": False,
            "amp": False,
            "lr_scaling": "linear",
            "selection_policy": "fixed_final_epoch",
            "validation_policy": "post_training_only",
            "checkpoint_interval_epochs": 1,
            "max_keep_checkpoints": 1,
            "verify_train_images": 2,
        }
        if training_protocol is not None:
            protocol.update(dict(training_protocol))
        if protocol["lr_scaling"] != "linear":
            raise ValueError("Only linear total-batch learning-rate scaling is supported")
        if protocol["selection_policy"] != "fixed_final_epoch":
            raise ValueError("Training checkpoints must use fixed_final_epoch selection")
        if protocol["validation_policy"] != "post_training_only":
            raise ValueError("Formal validation may only run after training")
        _patch_dataloader(
            cfg,
            "train_dataloader",
            dataset,
            "train",
            False,
            model.train_batch_size,
        )
        epochs = int(model.finetune_epochs.get(dataset.id, 12))
        train_cfg = cfg.get("train_cfg")
        if isinstance(train_cfg, MutableMapping) and "max_epochs" in train_cfg:
            old_epochs = int(train_cfg["max_epochs"])
            train_cfg["max_epochs"] = epochs
            # VOC2007 test and BDD100K val are formal evaluation splits.
            # Keep them completely outside optimization/checkpoint selection.
            train_cfg["val_interval"] = epochs + 1
            _scale_epoch_scheduler(cfg, old_epochs, epochs)
            _scale_epoch_coupled_settings(cfg, old_epochs, epochs)
        default_hooks = cfg.get("default_hooks")
        if isinstance(default_hooks, MutableMapping):
            checkpoint = default_hooks.get("checkpoint")
            if isinstance(checkpoint, MutableMapping):
                checkpoint["interval"] = int(
                    protocol["checkpoint_interval_epochs"]
                )
                checkpoint["max_keep_ckpts"] = int(
                    protocol["max_keep_checkpoints"]
                )
                checkpoint["save_last"] = True
                checkpoint.pop("save_best", None)
        _patch_optimizer_batch_size(cfg, model.train_batch_size)
        cfg.auto_scale_lr = {
            "enable": True,
            "base_batch_size": model.train_base_batch_size,
        }
        optim_wrapper = cfg.get("optim_wrapper")
        if bool(protocol["amp"]) and isinstance(optim_wrapper, MutableMapping):
            wrapper_type = str(optim_wrapper.get("type", "OptimWrapper"))
            if wrapper_type == "OptimWrapper":
                optim_wrapper["type"] = "AmpOptimWrapper"
                optim_wrapper.setdefault("loss_scale", "dynamic")
            elif wrapper_type != "AmpOptimWrapper":
                raise ValueError(
                    "Cannot enable AMP for optim wrapper '{}'".format(wrapper_type)
                )
        cfg.randomness = {
            "seed": int(protocol["seed"]),
            "deterministic": bool(protocol["deterministic"]),
            "diff_rank_seed": False,
        }
        env_cfg = cfg.get("env_cfg")
        if isinstance(env_cfg, MutableMapping):
            env_cfg["cudnn_benchmark"] = False
        cfg.val_cfg = None
        cfg.val_dataloader = None
        cfg.val_evaluator = None
        cfg.test_cfg = None
        cfg.test_dataloader = None
        cfg.test_evaluator = None
        cfg.lgp_training_protocol = {
            **protocol,
            "dataset": dataset.id,
            "model": model.id,
            "epochs": epochs,
            "micro_batch_size": model.train_batch_size,
            "reference_batch_size": model.train_base_batch_size,
        }

    work_dir.mkdir(parents=True, exist_ok=True)
    if dump_config:
        cfg.dump(str(work_dir / "runtime_config.py"))
    return cfg
