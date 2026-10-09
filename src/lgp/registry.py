from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from .metrics import COCO_BBOX_METRICS
from .paths import project_root, resolve_project_path


class RegistryError(ValueError):
    pass


COMPATIBILITY_STATUSES = ("native", "external", "adaptable", "unsupported")


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RegistryError("Missing registry file: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise RegistryError("Registry root must be a mapping: {}".format(path))
    if payload.get("schema_version") != 1:
        raise RegistryError("Unsupported schema_version in {}".format(path))
    return payload


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    table_code: str
    framework: str
    family: str
    backbone: str
    source: bool
    held_out: bool
    config: str
    checkpoint: str
    coco_box_ap: float
    finetune_epochs: Mapping[str, int]
    train_batch_size: int
    train_base_batch_size: int
    bbox_only: bool = False
    train_grad_clip: Optional[Mapping[str, Any]] = None
    finetune_head_init: Mapping[str, Mapping[str, Any]] = field(
        default_factory=dict
    )

    @property
    def checkpoint_filename(self) -> str:
        return self.checkpoint.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class ModelGroupSpec:
    id: str
    display_name: str
    models: Tuple[str, ...]


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    annotation: str
    image_prefix: str
    expected_images: int
    annotations_optional: bool = False
    benchmark_required: bool = True


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    display_name: str
    root: Path
    num_classes: int
    classes: List[str]
    splits: Mapping[str, DatasetSplit]
    download: Mapping[str, Any]

    def split(self, name: str) -> DatasetSplit:
        try:
            return self.splits[name]
        except KeyError as exc:
            raise RegistryError(
                "Dataset '{}' has no '{}' split".format(self.id, name)
            ) from exc


@dataclass(frozen=True)
class AttackSpec:
    id: str
    display_name: str
    executor: str
    parameters: Mapping[str, Any]
    metadata: Mapping[str, Any]


class Registry:
    """Single source of truth for models, datasets, attacks and protocols."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or project_root()).resolve()
        model_payload = _load_yaml(self.root / "configs" / "models.yaml")
        self.framework_versions = dict(model_payload.get("framework_versions", {}))
        self.training_protocol = dict(model_payload.get("training_protocol", {}))
        self.training_safety = dict(model_payload.get("training_safety", {}))
        self.models: Dict[str, ModelSpec] = {}
        for model_id, raw in model_payload.get("models", {}).items():
            raw_grad_clip = raw.get("train_grad_clip")
            if raw_grad_clip is not None and not isinstance(
                raw_grad_clip, Mapping
            ):
                raise RegistryError(
                    "{} train_grad_clip must be a mapping".format(model_id)
                )
            raw_head_init = raw.get("finetune_head_init", {})
            if not isinstance(raw_head_init, Mapping):
                raise RegistryError(
                    "{} finetune_head_init must be a mapping".format(model_id)
                )
            for dataset_id, head_init in raw_head_init.items():
                if not isinstance(dataset_id, str) or not isinstance(
                    head_init, Mapping
                ):
                    raise RegistryError(
                        "{} finetune_head_init entries must be mappings".format(
                            model_id
                        )
                    )
            self.models[model_id] = ModelSpec(
                id=model_id,
                display_name=str(raw["display_name"]),
                table_code=str(raw["table_code"]),
                framework=str(raw["framework"]),
                family=str(raw["family"]),
                backbone=str(raw["backbone"]),
                source=bool(raw.get("source", False)),
                held_out=bool(raw.get("held_out", False)),
                bbox_only=bool(raw.get("bbox_only", False)),
                config=str(raw["config"]),
                checkpoint=str(raw["checkpoint"]),
                coco_box_ap=float(raw["coco_box_ap"]),
                finetune_epochs=dict(raw.get("finetune_epochs", {})),
                train_batch_size=int(raw.get("train_batch_size", 1)),
                train_base_batch_size=int(
                    raw.get("train_base_batch_size", raw.get("train_batch_size", 1))
                ),
                train_grad_clip=(
                    dict(raw_grad_clip) if raw_grad_clip is not None else None
                ),
                finetune_head_init={
                    str(dataset_id): dict(head_init)
                    for dataset_id, head_init in raw_head_init.items()
                },
            )
        raw_paper_order = model_payload.get("paper_order", [])
        if not isinstance(raw_paper_order, list):
            raise RegistryError("models.yaml paper_order must be a list")
        self.paper_order = [str(model_id) for model_id in raw_paper_order]
        raw_paper_groups = model_payload.get("paper_groups", [])
        if not isinstance(raw_paper_groups, list):
            raise RegistryError("models.yaml paper_groups must be a list")
        self.paper_groups: List[ModelGroupSpec] = []
        for raw_group in raw_paper_groups:
            if not isinstance(raw_group, Mapping):
                raise RegistryError("Every models.yaml paper_groups entry must be a mapping")
            raw_models = raw_group.get("models", [])
            if not isinstance(raw_models, list):
                raise RegistryError(
                    "Model group '{}' models must be a list".format(
                        raw_group.get("id", "")
                    )
                )
            self.paper_groups.append(
                ModelGroupSpec(
                    id=str(raw_group.get("id", "")),
                    display_name=str(raw_group.get("display_name", "")),
                    models=tuple(str(model_id) for model_id in raw_models),
                )
            )

        self.datasets: Dict[str, DatasetSpec] = {}
        dataset_dir = self.root / "configs" / "datasets"
        for path in sorted(dataset_dir.glob("*.yaml")):
            raw = _load_yaml(path)
            dataset_id = str(raw["id"])
            splits = {
                name: DatasetSplit(
                    name=name,
                    annotation=str(value["annotation"]),
                    image_prefix=str(value.get("image_prefix", "")),
                    expected_images=int(value["expected_images"]),
                    annotations_optional=bool(value.get("annotations_optional", False)),
                    benchmark_required=bool(value.get("benchmark_required", True)),
                )
                for name, value in raw["splits"].items()
            }
            self.datasets[dataset_id] = DatasetSpec(
                id=dataset_id,
                display_name=str(raw["display_name"]),
                root=resolve_project_path(str(raw["root"])),
                num_classes=int(raw["num_classes"]),
                classes=list(raw["classes"]),
                splits=splits,
                download=dict(raw.get("download", {})),
            )

        self.attacks: Dict[str, AttackSpec] = {}
        native = _load_yaml(self.root / "configs" / "attacks" / "svfta.yaml")
        self.attacks[str(native["id"])] = AttackSpec(
            id=str(native["id"]),
            display_name=str(native["display_name"]),
            executor=str(native["executor"]),
            parameters=dict(native.get("parameters", {})),
            metadata={key: value for key, value in native.items() if key not in {"parameters"}},
        )
        references = _load_yaml(self.root / "configs" / "attacks" / "reference.yaml")
        for attack_id, raw in references.get("attacks", {}).items():
            self.attacks[attack_id] = AttackSpec(
                id=attack_id,
                display_name=str(raw["display_name"]),
                executor=str(raw["executor"]),
                parameters=dict(raw.get("parameters", {})),
                metadata=dict(raw),
            )

        public = _load_yaml(self.root / "configs" / "attacks" / "vcsf.yaml")
        if "vcsf" in self.attacks or public.get("id") != "vcsf":
            raise RegistryError("Public VCSF must have exactly one registry definition")
        self.attacks["vcsf"] = AttackSpec(
            id="vcsf", display_name=str(public["display_name"]),
            executor=str(public["executor"]), parameters=dict(public["parameters"]),
            metadata={key: value for key, value in public.items() if key != "parameters"},
        )

        baseline_payload = _load_yaml(
            self.root / "configs" / "attacks" / "baselines.yaml"
        )
        self.baseline_profiles = dict(baseline_payload.get("methods", {}))
        self.baseline_policy = dict(
            baseline_payload.get("compute_matched_policy", {})
        )

        self.compatibility = _load_yaml(self.root / "configs" / "compatibility.yaml")
        self.budget_profiles = _load_yaml(
            self.root / "configs" / "experiments" / "budgets.yaml"
        ).get("profiles", {})
        self.protocols = _load_yaml(
            self.root / "configs" / "experiments" / "protocols.yaml"
        ).get("protocols", {})
        defense_payload = _load_yaml(
            self.root
            / "configs"
            / "experiments"
            / "preprocessing_defenses.yaml"
        )
        if defense_payload.get("schema_version") != 1:
            raise RegistryError(
                "preprocessing_defenses.yaml has an unsupported schema_version"
            )
        if (
            defense_payload.get("benchmark")
            != "coco_retained500_common2_preprocessing_defense"
        ):
            raise RegistryError(
                "preprocessing_defenses.yaml has the wrong benchmark id"
            )
        self.preprocessing_defense_policy = dict(
            defense_payload.get("implementation", {})
        )
        self.preprocessing_defense_threat_model = dict(
            defense_payload.get("threat_model", {})
        )
        self.adaptive_preprocessing_policy = dict(
            defense_payload.get("adaptive_attack", {})
        )
        self.preprocessing_defense_order = list(
            defense_payload.get("variant_order", [])
        )
        self.preprocessing_defenses = dict(
            defense_payload.get("variants", {})
        )
        self.training_state_transfer = _load_yaml(
            self.root
            / "configs"
            / "experiments"
            / "training_state_transfer.yaml"
        )
        self.training_state_fullval_confirmation = _load_yaml(
            self.root
            / "configs"
            / "experiments"
            / "training_state_fullval_confirmation.yaml"
        )
        self.multi_budget_linf = _load_yaml(
            self.root
            / "configs"
            / "experiments"
            / "multi_budget_linf.yaml"
        )
        self.ablation_studies = _load_yaml(
            self.root / "configs" / "experiments" / "ablations.yaml"
        ).get("studies", {})
        self._validate()

    def _validate(self) -> None:
        from .attacks.vcsf_public import verify_public_identity

        try:
            verify_public_identity(self.root, self.attacks["vcsf"])
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            raise RegistryError("Public A10 identity check failed: " + str(exc)) from exc
        if len(self.models) != 16:
            raise RegistryError(
                "The benchmark registry must contain exactly 16 targets; found {}".format(
                    len(self.models)
                )
            )
        if len(set(model.table_code for model in self.models.values())) != len(self.models):
            raise RegistryError("Model table_code values must be unique")
        required_training_protocol = {
            "seed",
            "deterministic",
            "amp",
            "lr_scaling",
            "selection_policy",
            "validation_policy",
            "checkpoint_interval_epochs",
            "max_keep_checkpoints",
            "verify_train_images",
        }
        missing_training_protocol = sorted(
            required_training_protocol - set(self.training_protocol)
        )
        if missing_training_protocol:
            raise RegistryError(
                "models.yaml training_protocol is missing: {}".format(
                    ", ".join(missing_training_protocol)
                )
            )
        if self.training_protocol["lr_scaling"] != "linear":
            raise RegistryError("The maintained training lr_scaling must be 'linear'")
        if self.training_protocol["selection_policy"] != "fixed_final_epoch":
            raise RegistryError(
                "The maintained training selection_policy must be 'fixed_final_epoch'"
            )
        if self.training_protocol["validation_policy"] != "post_training_only":
            raise RegistryError(
                "The maintained training validation_policy must be 'post_training_only'"
            )
        if int(self.training_protocol["checkpoint_interval_epochs"]) <= 0:
            raise RegistryError("checkpoint_interval_epochs must be positive")
        if int(self.training_protocol["max_keep_checkpoints"]) <= 0:
            raise RegistryError("max_keep_checkpoints must be positive")
        if int(self.training_protocol["verify_train_images"]) <= 0:
            raise RegistryError("verify_train_images must be positive")
        required_training_safety = {
            "invalid_loss_check_interval",
            "require_finite_checkpoint",
        }
        if set(self.training_safety) != required_training_safety:
            raise RegistryError(
                "models.yaml training_safety must contain exactly: {}".format(
                    ", ".join(sorted(required_training_safety))
                )
            )
        if int(self.training_safety["invalid_loss_check_interval"]) <= 0:
            raise RegistryError(
                "training_safety invalid_loss_check_interval must be positive"
            )
        if self.training_safety["require_finite_checkpoint"] is not True:
            raise RegistryError(
                "training_safety require_finite_checkpoint must remain true"
            )
        for model in self.models.values():
            if model.train_batch_size <= 0:
                raise RegistryError(
                    "{} train_batch_size must be positive".format(model.id)
                )
            if model.train_base_batch_size < model.train_batch_size:
                raise RegistryError(
                    "{} train_base_batch_size must be at least train_batch_size".format(
                        model.id
                    )
                )
            for dataset_id in ("voc", "bdd100k"):
                if int(model.finetune_epochs.get(dataset_id, 0)) <= 0:
                    raise RegistryError(
                        "{} must declare positive finetune_epochs for {}".format(
                            model.id, dataset_id
                        )
                    )
            grad_clip = model.train_grad_clip
            if grad_clip is not None:
                required_grad_clip = {
                    "max_norm",
                    "norm_type",
                    "basis",
                    "source_commit",
                    "source_url",
                }
                if set(grad_clip) != required_grad_clip:
                    raise RegistryError(
                        "{} train_grad_clip must contain exactly: {}".format(
                            model.id, ", ".join(sorted(required_grad_clip))
                        )
                    )
                max_norm = float(grad_clip["max_norm"])
                norm_type = float(grad_clip["norm_type"])
                if not math.isfinite(max_norm) or max_norm <= 0:
                    raise RegistryError(
                        "{} train_grad_clip max_norm must be finite and positive".format(
                            model.id
                        )
                    )
                if not math.isfinite(norm_type) or norm_type <= 0:
                    raise RegistryError(
                        "{} train_grad_clip norm_type must be finite and positive".format(
                            model.id
                        )
                    )
                if grad_clip["basis"] != "original_authors_released_recipe":
                    raise RegistryError(
                        "{} train_grad_clip basis is not an audited author recipe".format(
                            model.id
                        )
                    )
                source_commit = str(grad_clip["source_commit"])
                if (
                    len(source_commit) != 40
                    or any(
                        character not in "0123456789abcdef"
                        for character in source_commit.lower()
                    )
                ):
                    raise RegistryError(
                        "{} train_grad_clip source_commit must be a full Git SHA".format(
                            model.id
                        )
                    )
                if not str(grad_clip["source_url"]).startswith(
                    "https://github.com/"
                ):
                    raise RegistryError(
                        "{} train_grad_clip source_url must be a GitHub permalink".format(
                            model.id
                        )
                    )
            for dataset_id, head_init in model.finetune_head_init.items():
                required_head_init = {
                    "policy",
                    "source_dataset",
                    "parameter_stem",
                    "target_to_source",
                    "basis",
                    "rationale",
                }
                if set(head_init) != required_head_init:
                    raise RegistryError(
                        "{} finetune_head_init.{} must contain exactly: {}".format(
                            model.id,
                            dataset_id,
                            ", ".join(sorted(required_head_init)),
                        )
                    )
                if head_init["policy"] != "semantic_class_copy":
                    raise RegistryError(
                        "{} finetune_head_init.{} has unsupported policy".format(
                            model.id, dataset_id
                        )
                    )
                if head_init["basis"] != "canonical_semantic_class_overlap":
                    raise RegistryError(
                        "{} finetune_head_init.{} basis must be "
                        "canonical_semantic_class_overlap".format(
                            model.id, dataset_id
                        )
                    )
                if not str(head_init["rationale"]).strip():
                    raise RegistryError(
                        "{} finetune_head_init.{} rationale is empty".format(
                            model.id, dataset_id
                        )
                    )
                parameter_stem = str(head_init["parameter_stem"])
                if (
                    not parameter_stem
                    or parameter_stem.startswith(".")
                    or parameter_stem.endswith(".")
                ):
                    raise RegistryError(
                        "{} finetune_head_init.{} parameter_stem is invalid".format(
                            model.id, dataset_id
                        )
                    )
                if dataset_id not in self.datasets:
                    raise RegistryError(
                        "{} finetune_head_init names unknown target dataset {}".format(
                            model.id, dataset_id
                        )
                    )
                source_dataset_id = str(head_init["source_dataset"])
                if source_dataset_id not in self.datasets:
                    raise RegistryError(
                        "{} finetune_head_init names unknown source dataset {}".format(
                            model.id, source_dataset_id
                        )
                    )
                target_to_source = head_init["target_to_source"]
                if not isinstance(target_to_source, Mapping):
                    raise RegistryError(
                        "{} finetune_head_init.{} target_to_source must be a mapping".format(
                            model.id, dataset_id
                        )
                    )
                target_classes = self.datasets[dataset_id].classes
                source_classes = self.datasets[source_dataset_id].classes
                if set(target_to_source) != set(target_classes):
                    raise RegistryError(
                        "{} finetune_head_init.{} must map every target class "
                        "exactly once".format(model.id, dataset_id)
                    )
                mapped_sources = [str(target_to_source[name]) for name in target_classes]
                unknown_sources = sorted(set(mapped_sources).difference(source_classes))
                if unknown_sources:
                    raise RegistryError(
                        "{} finetune_head_init.{} names unknown source classes: {}".format(
                            model.id, dataset_id, ", ".join(unknown_sources)
                        )
                    )
                if len(set(mapped_sources)) != len(mapped_sources):
                    raise RegistryError(
                        "{} finetune_head_init.{} source rows must be one-to-one".format(
                            model.id, dataset_id
                        )
                    )
        if len(self.paper_order) != len(self.models):
            raise RegistryError(
                "paper_order must contain exactly {} model IDs; found {}".format(
                    len(self.models), len(self.paper_order)
                )
            )
        if len(set(self.paper_order)) != len(self.paper_order):
            raise RegistryError("paper_order contains duplicate model IDs")
        missing_from_order = sorted(set(self.models) - set(self.paper_order))
        unknown_in_order = sorted(set(self.paper_order) - set(self.models))
        if missing_from_order or unknown_in_order:
            raise RegistryError(
                "paper_order differs from the model registry: missing={}, unknown={}".format(
                    missing_from_order, unknown_in_order
                )
            )
        if not self.paper_groups:
            raise RegistryError("models.yaml must declare paper_groups")
        group_ids = [group.id for group in self.paper_groups]
        if any(not group_id for group_id in group_ids):
            raise RegistryError("Every paper group must have a non-empty id")
        if len(set(group_ids)) != len(group_ids):
            raise RegistryError("paper_groups contains duplicate group IDs")
        if any(not group.display_name for group in self.paper_groups):
            raise RegistryError("Every paper group must have a non-empty display_name")
        if any(not group.models for group in self.paper_groups):
            raise RegistryError("Every paper group must contain at least one model")
        grouped_order = [
            model_id
            for group in self.paper_groups
            for model_id in group.models
        ]
        if grouped_order != self.paper_order:
            raise RegistryError(
                "Flattened paper_groups models must exactly equal paper_order"
            )
        sources = [
            model_id for model_id in self.paper_order if self.models[model_id].source
        ]
        if len(sources) != 6:
            raise RegistryError(
                "The benchmark registry must contain exactly 6 sources; found {}".format(
                    len(sources)
                )
            )
        for dataset in self.datasets.values():
            if dataset.num_classes != len(dataset.classes):
                raise RegistryError(
                    "Dataset '{}' declares {} classes but lists {}".format(
                        dataset.id, dataset.num_classes, len(dataset.classes)
                    )
                )
        declared_source_list = list(self.compatibility.get("sources", []))
        declared_sources = set(declared_source_list)
        if declared_source_list != sources:
            raise RegistryError(
                "Compatibility source list must exactly follow the canonical paper order"
            )
        capabilities = self.compatibility.get("source_capabilities", {})
        if set(capabilities) != declared_sources:
            raise RegistryError(
                "Compatibility source_capabilities must cover every source exactly once"
            )
        method_records = self.compatibility.get("methods", {})
        if set(method_records) != set(self.attacks):
            raise RegistryError(
                "Compatibility methods differ from attack registry: missing={}, extra={}".format(
                    sorted(set(self.attacks) - set(method_records)),
                    sorted(set(method_records) - set(self.attacks)),
                )
            )
        for attack_id, record in method_records.items():
            assignments: Dict[str, List[str]] = {source: [] for source in sources}
            for status in COMPATIBILITY_STATUSES:
                selected = record.get(status, [])
                if not isinstance(selected, list):
                    raise RegistryError(
                        "Compatibility '{}.{}' must be a list".format(attack_id, status)
                    )
                unknown = sorted(set(selected) - declared_sources)
                if unknown:
                    raise RegistryError(
                        "Compatibility '{}.{}' contains unknown sources: {}".format(
                            attack_id, status, ", ".join(unknown)
                        )
                    )
                for source in selected:
                    assignments[source].append(status)
            invalid = {
                source: statuses
                for source, statuses in assignments.items()
                if len(statuses) != 1
            }
            if invalid:
                raise RegistryError(
                    "Every attack/source pair must have exactly one explicit compatibility "
                    "status; {} has {}".format(attack_id, invalid)
                )
        for study_id, study in self.ablation_studies.items():
            variants = study.get("variants", {})
            if not isinstance(variants, Mapping) or not variants:
                raise RegistryError("Ablation '{}' has no variants".format(study_id))
            method = str(study.get("method", "svfta"))
            if method not in self.attacks:
                protocol = self.protocols.get(study.get("protocol"), {})
                isolated = protocol.get("isolated_candidate", {})
                if not (
                    protocol.get("process_local_candidate_overlay") is True
                    and protocol.get("ablation_study") == study_id
                    and isolated.get("execution_alias") == method
                    and isolated.get("implementation")
                    and isolated.get("class_name")
                    and isolated.get("config_class_name")
                ):
                    raise RegistryError(
                        "Ablation '{}' references unknown method '{}'".format(
                            study_id, method
                        )
                    )
                implementation = (self.root / str(isolated["implementation"])).resolve()
                if (self.root.resolve() not in implementation.parents
                        or implementation.suffix != ".py" or not implementation.is_file()):
                    raise RegistryError("Invalid isolated ablation implementation")
        if not self.budget_profiles:
            raise RegistryError("No experiment budget profiles are configured")
        unknown_baselines = sorted(set(self.baseline_profiles) - set(self.attacks))
        if unknown_baselines:
            raise RegistryError(
                "Baseline profiles reference unknown attacks: {}".format(
                    ", ".join(unknown_baselines)
                )
            )
        for protocol_id, protocol in self.protocols.items():
            profile = protocol.get("budget_profile")
            if profile not in self.budget_profiles:
                raise RegistryError(
                    "Protocol '{}' references unknown budget profile '{}'".format(
                        protocol_id, profile
                    )
                )
            if protocol.get("report_method_order") not in (None, "declared"):
                raise RegistryError(
                    "Protocol '{}' has an invalid report method order".format(protocol_id)
                )
            method_profiles = protocol.get("method_budget_profiles", {})
            if not isinstance(method_profiles, dict) or any(
                not isinstance(attack_id, str) or not isinstance(value, str)
                or attack_id not in self.attacks or value not in self.budget_profiles
                for attack_id, value in method_profiles.items()
            ):
                raise RegistryError(
                    "Protocol '{}' has an invalid method-specific budget profile".format(protocol_id)
                )
            for field, available in (
                ("allowed_datasets", self.datasets), ("allowed_methods", self.attacks)
            ):
                if field in protocol:
                    values = protocol[field]
                    if (not isinstance(values, list) or not values
                            or any(not isinstance(value, str) for value in values)
                            or len(values) != len(set(values))
                            or any(value not in available for value in values)):
                        raise RegistryError(
                            "Protocol '{}' has an invalid {} selection".format(protocol_id, field)
                        )
                    defaults = protocol.get("datasets" if field == "allowed_datasets" else "methods", [])
                    if isinstance(defaults, list) and any(value not in values for value in defaults):
                        raise RegistryError(
                            "Protocol '{}' defaults exceed its {} selection".format(protocol_id, field)
                        )
            if profile == "compute_matched":
                selected = protocol.get("methods", [])
                method_ids = list(self.attacks) if selected == "all" else list(selected)
                missing_profiles = sorted(
                    attack_id
                    for attack_id in method_ids
                    if self.attacks[attack_id].executor != "native"
                    and attack_id not in self.baseline_profiles
                )
                if missing_profiles:
                    raise RegistryError(
                        "Compute-matched protocol '{}' lacks baseline profiles for: {}".format(
                            protocol_id, ", ".join(missing_profiles)
                        )
                    )

        background = self.protocols.get("vcsf_final_background_validation")
        if background is not None:
            try:
                request_path = self.root / background["request_file"]
                if (request_path != request_path.resolve()
                        or self.root.resolve() not in request_path.resolve().parents):
                    raise RegistryError("Final-background request must be a plain project file")
                request_raw = request_path.read_bytes()
                if hashlib.sha256(request_raw).hexdigest() != background["request_sha256"]:
                    raise RegistryError("Final-background request identity changed")
                request = json.loads(request_raw)
                scope = request["final_background_request"]
                decision = request["decision"]
                study = self.ablation_studies[background["ablation_study"]]
                budget = self.budget_profiles[background["budget_profile"]]
                controls = scope["controls"]
                order = [scope["anchor"]["variant"]] + [row["id"] for row in controls]
                expected = {
                    "dataset": scope["dataset"], "split": scope["split"],
                    "sources": [scope["source"]], "targets": "all", "seed": scope["seed"],
                    "expected_images": scope["images_per_group"],
                    "logical_groups": scope["logical_groups"],
                    "evaluation_cells": scope["logical_target_cells"],
                    "conditional_new_groups": scope["new_groups_if_anchor_reuse_qualifies"],
                    "conditional_new_evaluations": scope["new_target_evaluations_if_anchor_reuse_qualifies"],
                    "potential_reuse_variant": scope["anchor"]["variant"],
                    "seed_schedule": "full_val_numeric_image_position",
                    "generic_dispatch_forbidden": True,
                    "preparation_entrypoint": "experiments/prepare_vcsf_final_background.py",
                    "execution_entrypoint": "experiments/vcsf_final_background_experiments.py",
                    "execution_protocol": "vcsf_final_background_execution",
                    "execution_requires_independent_admission": True,
                    "device_counts": [1, 2], "preparation_only": True,
                    "execution_admitted": False, "final_method_promoted": False,
                    "independent_confirmation": False,
                    "payload_retention": {"mode": "keep_all", "cleanup_authorized": False},
                    "prediction_archive": {"format": "gzip", "retention": "keep_all"},
                }
                if any(type(background.get(k)) is not type(v) or background[k] != v
                        for k, v in expected.items()):
                    raise RegistryError("Final-background scope or preparation gate drifted")
                if (scope["targets"] != self.target_ids()
                        or scope["metrics"] != self.protocols["coco_all_methods"]["metrics"]
                        or study["row_order"] != order or list(study["variants"]) != order
                        or study["anchor"] != decision["unique_working_candidate"]
                        or study["anchor_parameters_sha256"] != decision["parameters_sha256"]
                        or budget["request_sha256"] != background["request_sha256"]
                        or len(order) != len(set(order)) or len(order) != scope["logical_groups"]):
                    raise RegistryError("Final-background registered design identity drifted")
                for key in ("epsilon", "step_size"):
                    parameter = "eps" if key == "epsilon" else key
                    if float(Fraction(budget[key])) != decision["parameters"][parameter]:
                        raise RegistryError("Final-background budget changed")
                if (budget["required_iterations"] != decision["parameters"]["iterations"]
                        or type(budget["required_iterations"]) is not int
                        or budget["max_gradient_evaluations_per_image"] != budget["required_iterations"]
                        or budget["allowed_iterations_by_initialization"] != {
                            name: [budget["required_iterations"]]
                            for name in ("detector", "none", "random_sign")}):
                    raise RegistryError("Final-background gradient schedule changed")
                declared = {row["id"]: row for row in controls}
                for variant in order:
                    overrides = study["variants"][variant]["parameters"]
                    wanted = {} if variant == study["anchor"] else declared[variant]["declared_overrides"]
                    if overrides != wanted:
                        raise RegistryError("Final-background override changed: " + variant)
                    parameters = dict(study["base_parameters"], **overrides)
                    normalized = {key: float(Fraction(value)) if isinstance(value, str)
                        and "/" in value else value for key, value in parameters.items()}
                    expected_parameters = (decision["parameters"] if variant == study["anchor"]
                        else declared[variant]["parameters"])
                    if json.dumps(normalized, sort_keys=True) != json.dumps(expected_parameters, sort_keys=True):
                        raise RegistryError("Final-background parameters changed: " + variant)
            except RegistryError:
                raise
            except (KeyError, TypeError, ValueError, OSError) as exc:
                raise RegistryError("Malformed final-background registration: " + str(exc)) from exc

        lgp_stages = [
            protocol for protocol in self.protocols.values()
            if protocol.get("lgp_corrected_stage") is True
        ]
        if lgp_stages:
            coco_stages = [stage for stage in lgp_stages if stage.get("datasets") == ["coco"]]
            voc_stages = [stage for stage in lgp_stages if stage.get("datasets") == ["voc"]]
            if len(lgp_stages) != 3 or len(coco_stages) != 2 or len(voc_stages) != 1:
                raise RegistryError("Corrected-LGP main stages must be COCO 2 plus VOC 1")
            canonical_sources = self.source_ids()
            coco_sources = [source for stage in coco_stages for source in stage.get("sources", [])]
            if sorted(coco_sources) != sorted(canonical_sources) or len(coco_sources) != len(set(coco_sources)):
                raise RegistryError("Corrected-LGP COCO stages must partition all six sources")
            if voc_stages[0].get("sources") != canonical_sources:
                raise RegistryError("Corrected-LGP VOC stage must cover all six sources")
            identities = []
            for stage in lgp_stages:
                dataset_id = stage["datasets"][0]
                source_count = len(stage.get("sources", []))
                if (
                    stage.get("methods") != ["lgp"]
                    or stage.get("split") != "val"
                    or stage.get("targets") != "all"
                    or stage.get("seed") != 42
                    or stage.get("budget_profile") != "compute_matched"
                    or stage.get("metrics") != self.protocols["coco_all_methods"]["metrics"]
                    or stage.get("expected_images") != self.datasets[dataset_id].split("val").expected_images
                    or stage.get("expected_attack_groups") != source_count
                    or stage.get("expected_attack_evaluations") != source_count * len(self.target_ids())
                    or stage.get("entrypoint") != "experiments/lgp_corrected_main.py"
                    or stage.get("prediction_archive", {}).get("format") != "gzip"
                ):
                    raise RegistryError("Corrected-LGP main stage contract drifted")
                hashes = tuple(
                    stage.get(name) for name in (
                        "qualified_lgp_implementation_sha256",
                        "qualified_lgp_factory_sha256",
                        "qualified_lgp_registry_sha256",
                        "qualified_lgp_parameters_sha256",
                    )
                )
                if any(
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                    for value in hashes
                ):
                    raise RegistryError("Corrected-LGP qualification hashes are malformed")
                identities.append(hashes)
                clean_hash = stage.get("clean_reference_acceptance_sha256")
                if (
                    not isinstance(clean_hash, str)
                    or len(clean_hash) != 64
                    or any(character not in "0123456789abcdef" for character in clean_hash)
                ):
                    raise RegistryError("Corrected-LGP clean reference hash is malformed")
                if not isinstance(stage.get("execution_authorized"), bool):
                    raise RegistryError("Corrected-LGP execution gate is malformed")
                required_free = stage.get("storage_required_free_bytes")
                additional_free = stage.get("storage_required_free_bytes_per_additional_source")
                if stage["execution_authorized"] and (
                    type(required_free) is not int or required_free < 5_000_000_000
                    or type(additional_free) is not int
                    or additional_free < 2_000_000_000
                ):
                    raise RegistryError("Corrected-LGP source-wave storage admission is incomplete")
            if len(set(identities)) != 1:
                raise RegistryError("Corrected-LGP stages disagree on implementation identity")
            common2 = [
                stage for stage in coco_stages
                if stage.get("payload_retention", {}).get("mode") == "keep_all"
            ]
            if len(common2) != 1 or common2[0].get("sources") != canonical_sources[:2]:
                raise RegistryError("Corrected-LGP Common-2 payload stage drifted")
            for stage in lgp_stages:
                if stage is common2[0]:
                    continue
                retention = stage.get("payload_retention", {})
                if (
                    retention.get("mode") != "fixed_count_after_group_validation"
                    or retention.get("retained_images") != 500
                    or retention.get("selection_algorithm")
                    != self.protocols["coco_all_methods"]["payload_retention"]["selection_algorithm"]
                    or retention.get("prune_after_complete_targets") != 16
                ):
                    raise RegistryError("Corrected-LGP fixed-500 lifecycle drifted")
                cleanup_authorized = retention.get("cleanup_authorized")
                decision_hash = retention.get("recovery_decision_sha256")
                if not isinstance(cleanup_authorized, bool) or (
                    cleanup_authorized
                    and (
                        not isinstance(decision_hash, str)
                        or len(decision_hash) != 64
                        or any(character not in "0123456789abcdef" for character in decision_hash)
                    )
                ):
                    raise RegistryError("Corrected-LGP cleanup decision is incomplete")
                if stage["execution_authorized"] and cleanup_authorized:
                    raise RegistryError("Corrected-LGP producer may not authorize cleanup")
            for stage in lgp_stages:
                audit_hash = stage.get("clean_current_audit_sha256")
                if (
                    not isinstance(audit_hash, str)
                    or len(audit_hash) != 64
                    or any(character not in "0123456789abcdef" for character in audit_hash)
                ):
                    raise RegistryError("Corrected-LGP current clean audit hash is malformed")
                binding_hash = stage.get("clean_reuse_binding_sha256")
                if binding_hash is not None and (
                    not isinstance(binding_hash, str)
                    or len(binding_hash) != 64
                    or any(character not in "0123456789abcdef" for character in binding_hash)
                ):
                    raise RegistryError("Corrected-LGP clean binding hash is malformed")
                if stage["execution_authorized"] and binding_hash is None:
                    raise RegistryError("Corrected-LGP clean reuse is not admitted")

        oblivious = self.protocols.get("lgp_corrected_coco_oblivious_preprocessing")
        if oblivious is None:
            raise RegistryError("Corrected-LGP oblivious preprocessing protocol is missing")
        if oblivious is not None:
            expected = {
                "datasets": ["coco"],
                "split": "val",
                "seed": 42,
                "budget_profile": "compute_matched",
                "source_protocol": "lgp_corrected_coco_common2_main",
                "sources": self.source_ids()[:2],
                "targets": "all",
                "methods": ["lgp"],
                "defenses": "all",
                "defense_registry": "configs/experiments/preprocessing_defenses.yaml",
                "metrics": self.protocols["coco_all_methods"]["metrics"],
                "threat_model": "oblivious_victim_input_preprocessing",
                "image_order": "numeric_coco_image_id",
                "retained_images": 500,
                "retained_selection_algorithm": "canonical_equal_bins_center_v1",
                "seed_schedule": "full_val_numeric_image_position",
                "expected_attack_evaluations": 2 * 11 * len(self.target_ids()),
                "source_payload_retention": "keep_all",
                "independent_stage_audit_required": True,
                "numerical_input_decision_required": True,
                "downstream_payload_reuse_decision_required": True,
                "storage_admission_required": True,
                "lgp_corrected_downstream_stage": "oblivious_preprocessing",
                "storage_safety_multiplier": 4,
                "storage_margin_bytes": 10000000000,
                "entrypoint": "experiments/lgp_corrected_oblivious_preprocessing.py",
                "admission_entrypoint": "tools/inspect_lgp_corrected_oblivious_preprocessing.py",
            }
            if any(oblivious.get(key) != value for key, value in expected.items()):
                raise RegistryError("Corrected-LGP oblivious preprocessing contract drifted")
            if type(oblivious.get("execution_authorized")) is not bool:
                raise RegistryError("Corrected-LGP oblivious execution gate is malformed")
            retained_hash = oblivious.get("retained_image_ids_sha256")
            defense_contract = {
                "implementation": self.preprocessing_defense_policy,
                "threat_model": self.preprocessing_defense_threat_model,
                "variant_order": self.preprocessing_defense_order,
                "variants": self.preprocessing_defenses,
            }
            defense_hash = hashlib.sha256(json.dumps(
                defense_contract, ensure_ascii=False, sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()
            if (
                not isinstance(retained_hash, str)
                or len(retained_hash) != 64
                or any(character not in "0123456789abcdef" for character in retained_hash)
                or len(self.preprocessing_defense_order) != 11
                or self.preprocessing_defense_order[0] != "identity"
                or oblivious.get("defense_contract_sha256") != defense_hash
            ):
                raise RegistryError("Corrected-LGP oblivious selection or transforms drifted")

        defended_clean = self.protocols.get("lgp_corrected_coco_defended_clean_current")
        if defended_clean is None:
            raise RegistryError("Corrected-LGP current defended-clean protocol is missing")
        clean_defenses = self.preprocessing_defense_order[1:]
        expected_defended_clean = {
            "datasets": ["coco"], "split": "val", "budget_profile": "compute_matched",
            "source": "clean", "attack": "clean",
            "paired_attack_protocol": "lgp_corrected_coco_oblivious_preprocessing",
            "source_clean_protocol": "lgp_corrected_coco_common2_main",
            "targets": "all", "defenses": clean_defenses,
            "defense_registry": "configs/experiments/preprocessing_defenses.yaml",
            "defense_contract_sha256": defense_hash,
            "metrics": self.protocols["coco_all_methods"]["metrics"],
            "image_order": "numeric_coco_image_id", "retained_images": 500,
            "retained_selection_algorithm": "canonical_equal_bins_center_v1",
            "retained_image_ids_sha256": oblivious["retained_image_ids_sha256"],
            "accepted_clean_binding_sha256": self.protocols[
                "lgp_corrected_coco_common2_main"]["clean_reuse_binding_sha256"],
            "accepted_full_clean_audit_sha256": self.protocols[
                "lgp_corrected_coco_common2_main"]["clean_current_audit_sha256"],
            "expected_transform_groups": len(clean_defenses),
            "expected_transformed_images": 500 * len(clean_defenses),
            "expected_clean_evaluations": len(clean_defenses) * len(self.target_ids()),
            "expected_target_image_uses": 500 * len(clean_defenses) * len(self.target_ids()),
            "prediction_archive": {"format": "gzip"},
            "transformed_view_retention": "keep_all_pending_independent_audit",
            "independent_stage_audit_required": True,
            "storage_admission_required": True,
            "storage_safety_multiplier": 4,
            "storage_margin_bytes": 10000000000,
            "storage_reference_basis":
                "historical_candidate_current_byte_replay_capacity_estimate_only",
            "minimum_gpu_free_mib": 16000,
            "supervisor_required": True,
            "max_supervised_elapsed_seconds": 259200,
            "max_supervisor_observation_gap_seconds": 30,
            "min_positive_gpu_observations_per_worker": 2,
            "entrypoint": "experiments/lgp_corrected_defended_clean_current.py",
            "audit_entrypoint": "tools/audit_lgp_corrected_defended_clean_current.py",
            "stage_decision_entrypoint":
                "tools/accept_lgp_corrected_defended_clean_stage.py",
            "union_candidate_entrypoint":
                "tools/audit_lgp_corrected_defended_clean_union.py",
            "union_decision_entrypoint":
                "tools/accept_lgp_corrected_defended_clean_union.py",
            "supervisor_request_entrypoint":
                "tools/prepare_lgp_corrected_defended_clean_supervisor_request.py",
            "supervisor_entrypoint":
                "tools/supervise_lgp_corrected_defended_clean_current.py",
            "supervisor_stop_diagnostic_entrypoint":
                "tools/verify_lgp_corrected_clean_supervisor_stop.py",
        }
        if any(defended_clean.get(key) != value
               for key, value in expected_defended_clean.items()):
            raise RegistryError("Corrected-LGP current defended-clean contract drifted")
        if not ((defended_clean.get("status") == "preparation_only"
                 and defended_clean.get("execution_authorized") is False)
                or (defended_clean.get("status") == "active"
                    and defended_clean.get("execution_authorized") is True)):
            raise RegistryError("Corrected-LGP current defended-clean gate drifted")
        identity_hash = defended_clean.get("identity_subset_qualification_sha256")
        if (len(clean_defenses) != 10 or self.preprocessing_defense_order[0] != "identity"
                or not isinstance(identity_hash, str) or len(identity_hash) != 64
                or any(character not in "0123456789abcdef" for character in identity_hash)):
            raise RegistryError("Corrected-LGP current defended-clean identity is malformed")
        storage_floor = defended_clean.get("storage_required_free_bytes")
        admission_hash = defended_clean.get("storage_admission_sha256")
        reference_png_bytes = defended_clean.get("storage_reference_transformed_bytes")
        reference_prediction_bytes = defended_clean.get("storage_reference_prediction_bytes")
        reference_receipt = defended_clean.get("storage_reference_receipt_sha256")
        measured_png_bytes = defended_clean.get("storage_current_measured_transformed_bytes")
        measurement_hash = defended_clean.get("storage_current_measurement_sha256")
        measurement_audit_hash = defended_clean.get("storage_current_audit_sha256")
        if (type(reference_png_bytes) is not int or reference_png_bytes <= 0
                or type(reference_prediction_bytes) is not int
                or reference_prediction_bytes <= 0
                or not isinstance(reference_receipt, str) or len(reference_receipt) != 64
                or any(character not in "0123456789abcdef"
                       for character in reference_receipt)):
            raise RegistryError("Corrected-LGP current defended-clean capacity basis is invalid")
        for name, value in (("measurement", measurement_hash),
                            ("measurement audit", measurement_audit_hash)):
            if value is not None and (
                not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RegistryError(
                    "Corrected-LGP current defended-clean {} hash is invalid".format(name))
        if measured_png_bytes is not None and (
            type(measured_png_bytes) is not int or measured_png_bytes <= 0
        ):
            raise RegistryError("Corrected-LGP current defended-clean measured bytes are invalid")
        minimum_storage = defended_clean["storage_safety_multiplier"] * (
            max(reference_png_bytes, measured_png_bytes or 0)
            + reference_prediction_bytes
        ) + defended_clean["storage_margin_bytes"]
        if storage_floor is not None and (
            type(storage_floor) is not int or storage_floor < minimum_storage
        ):
            raise RegistryError("Corrected-LGP current defended-clean storage floor is too low")
        if defended_clean["execution_authorized"] and (
            type(storage_floor) is not int
            or not isinstance(admission_hash, str) or len(admission_hash) != 64
            or any(character not in "0123456789abcdef" for character in admission_hash)
            or measurement_hash is None or measurement_audit_hash is None
            or measured_png_bytes is None
        ):
            raise RegistryError("Corrected-LGP current defended-clean storage is not admitted")

        identity_clean = self.protocols.get("lgp_corrected_coco_identity_clean_current")
        if identity_clean is None:
            raise RegistryError("Corrected-LGP current identity-clean protocol is missing")
        expected_identity_clean = {
            "datasets": ["coco"], "split": "val", "source": "clean",
            "attack": "clean", "budget_profile": "compute_matched",
            "paired_attack_protocol": "lgp_corrected_coco_oblivious_preprocessing",
            "paired_transformed_clean_protocol": "lgp_corrected_coco_defended_clean_current",
            "targets": "all", "defenses": ["identity"],
            "defense_registry": "configs/experiments/preprocessing_defenses.yaml",
            "defense_contract_sha256": defense_hash,
            "metrics": self.protocols["coco_all_methods"]["metrics"],
            "image_order": "numeric_coco_image_id", "retained_images": 500,
            "retained_selection_algorithm": "canonical_equal_bins_center_v1",
            "retained_image_ids_sha256": oblivious["retained_image_ids_sha256"],
            "accepted_clean_binding_sha256": defended_clean["accepted_clean_binding_sha256"],
            "accepted_full_clean_audit_sha256": defended_clean[
                "accepted_full_clean_audit_sha256"],
            "original_runtime_config_seal_required": True,
            "prediction_archive": {"format": "gzip"},
            "expected_clean_evaluations": len(self.target_ids()),
            "expected_target_image_uses": 500 * len(self.target_ids()),
            "independent_stage_audit_required": True,
            "supervisor_required": True,
            "storage_admission_required": True,
            "minimum_gpu_free_mib": 16000,
            "max_supervised_elapsed_seconds": 86400,
            "entrypoint": "experiments/lgp_corrected_identity_clean_current.py",
            "audit_entrypoint": "tools/audit_lgp_corrected_identity_clean_current.py",
        }
        if any(identity_clean.get(key) != value
               for key, value in expected_identity_clean.items()):
            raise RegistryError("Corrected-LGP current identity-clean contract drifted")
        if (identity_clean.get("status") != "preparation_only"
                or identity_clean.get("execution_authorized") is not False):
            raise RegistryError("Corrected-LGP current identity-clean gate drifted")

        adaptive_lgp = self.protocols.get("lgp_corrected_coco_adaptive_preprocessing")
        if adaptive_lgp is None:
            raise RegistryError("Corrected-LGP adaptive preprocessing protocol is missing")
        expected_adaptive_lgp = {
            "datasets": ["coco"], "split": "val", "seed": 42,
            "budget_profile": "compute_matched",
            "source_protocol": "lgp_corrected_coco_common2_main",
            "paired_oblivious_protocol": "lgp_corrected_coco_oblivious_preprocessing",
            "sources": self.source_ids()[:2], "targets": "all", "methods": ["lgp"],
            "defenses": "all",
            "defense_registry": "configs/experiments/preprocessing_defenses.yaml",
            "metrics": self.protocols["coco_all_methods"]["metrics"],
            "threat_model": "adaptive_source_pipeline_bpda",
            "image_order": "numeric_coco_image_id", "retained_images": 500,
            "retained_selection_algorithm": "canonical_equal_bins_center_v1",
            "seed_schedule": "full_val_numeric_image_position",
            "expected_generation_groups": 22,
            "expected_attack_evaluations": 2 * 11 * len(self.target_ids()),
            "expected_metric_records": 2 * 11 * len(self.target_ids()),
            "identity_payload_reuse_authorized": False,
            "generated_payload_retention": "keep_all_pending_independent_audit",
            "transformed_view_retention": "keep_all_pending_independent_audit",
            "clean_dependency": "separate_defended_clean_acceptance_required_for_paper_table",
            "source_stage_audit_required": True,
            "paired_oblivious_numerical_input_required": True,
            "storage_admission_required": True,
            "storage_safety_multiplier": 6,
            "storage_margin_bytes": 10000000000,
            "entrypoint": "experiments/lgp_corrected_adaptive_preprocessing.py",
        }
        if any(adaptive_lgp.get(key) != value for key, value in expected_adaptive_lgp.items()):
            raise RegistryError("Corrected-LGP adaptive preparation contract drifted")
        if not ((adaptive_lgp.get("status") == "preparation_only"
                 and adaptive_lgp.get("execution_authorized") is False)
                or (adaptive_lgp.get("status") == "active"
                    and adaptive_lgp.get("execution_authorized") is True)):
            raise RegistryError("Corrected-LGP adaptive execution gate or status drifted")
        adaptive_contract = {
            "implementation": self.preprocessing_defense_policy,
            "adaptive_attack": self.adaptive_preprocessing_policy,
            "variant_order": self.preprocessing_defense_order,
            "variants": self.preprocessing_defenses,
        }
        adaptive_hash = hashlib.sha256(json.dumps(
            adaptive_contract, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        if (adaptive_lgp.get("retained_image_ids_sha256")
                != oblivious["retained_image_ids_sha256"]
                or adaptive_lgp.get("adaptive_contract_sha256") != adaptive_hash):
            raise RegistryError("Corrected-LGP adaptive selection or BPDA contract drifted")

        training = self.protocols.get("lgp_corrected_coco_training_state")
        if training is None:
            raise RegistryError("Corrected-LGP training-state protocol is missing")
        pair_reference = self.protocols["vcsf_final_training_state_refresh"]
        expected_training = {
            "datasets": ["coco"], "split": "val", "seed": 42,
            "budget_profile": "compute_matched",
            "source_protocol": "lgp_corrected_coco_common2_main",
            "source": "mask_rcnn_swin_t", "target": "faster_rcnn_r50",
            "methods": ["lgp"],
            "victim_states": ["standard_control", "adversarial_training"],
            "metrics": self.protocols["coco_all_methods"]["metrics"],
            "expected_images": 5000, "expected_attack_evaluations": 2,
            "expected_metric_records": 2, "generation_jobs": 0,
            "source_payload_retention": "keep_all",
            "pair_registry": "configs/experiments/training_state_transfer.yaml",
            "pair_qualification": "outputs/experiments/coco_training_state_fullval_confirmation/20260816T053441Z/checkpoint_pair_qualification.json",
            "pair_qualification_sha256": pair_reference["pair_qualification_sha256"],
            "checkpoint_sha256": pair_reference["checkpoint_sha256"],
            "threat_model": "zero_query_oblivious_cross_model_transfer",
            "independent_stage_audit_required": True,
            "numerical_input_decision_required": True,
            "downstream_payload_reuse_decision_required": True,
            "prediction_archive": {"format": "gzip"},
            "minimum_gpu_free_mib": 16000,
            "storage_required_free_bytes": 10000000000,
            "entrypoint": "experiments/lgp_corrected_training_state.py",
        }
        if any(training.get(key) != value for key, value in expected_training.items()):
            raise RegistryError("Corrected-LGP training-state scope or victim pair drifted")
        if type(training.get("execution_authorized")) is not bool:
            raise RegistryError("Corrected-LGP training-state execution gate is malformed")

        radius = self.protocols.get("lgp_corrected_coco_multi_budget_linf")
        if radius is None:
            raise RegistryError("Corrected-LGP radius protocol is missing")
        source = self.protocols["lgp_corrected_coco_common2_main"]
        radii = self.budget_profiles["compute_matched_multi_epsilon"]["epsilon_order"]
        expected_radius = {
            "datasets": ["coco"], "split": "val", "seed": source["seed"],
            "budget_profile": "compute_matched_multi_epsilon",
            "source_protocol": "lgp_corrected_coco_common2_main",
            "sources": source["sources"], "targets": "all", "methods": ["lgp"],
            "metrics": self.protocols["coco_all_methods"]["metrics"],
            "image_order": "numeric_coco_image_id",
            "seed_schedule": "full_val_numeric_image_position",
            "full_images": source["expected_images"],
            "epsilon_order": radii, "reference_epsilon": "4/255",
            "stress_only": [radii[-1]],
            "parameter_policy": "only_eps_changes_from_corrected_source201",
            "reference_policy":
                "accepted_common2_payload_candidate_only_pending_exact_reuse_decision",
            "source_stage_audit_required": True,
            "numerical_input_decision_required": True,
            "downstream_payload_reuse_decision_required": True,
            "expected_reference_groups": len(source["sources"]),
            "expected_new_generation_groups": len(source["sources"]) * (len(radii) - 1),
            "expected_attack_evaluations": len(source["sources"]) * len(radii)
                * len(self.target_ids()),
            "expected_metric_records": len(source["sources"]) * len(radii)
                * len(self.target_ids()),
            "new_payload_retention": "keep_all_pending_independent_audit",
            "execution_authorized": False, "status": "preparation_only",
        }
        if any(radius.get(key) != value for key, value in expected_radius.items()):
            raise RegistryError("Corrected-LGP radius preparation contract drifted")
        if (source["payload_retention"].get("mode") != "keep_all"
                or radius["full_images"] != 5000
                or radius["epsilon_order"] != ["2/255", "4/255", "8/255", "16/255", "32/255"]
                or radius["stress_only"] != ["32/255"]):
            raise RegistryError("Corrected-LGP radius source or scientific axis drifted")

        adaptive_refresh = self.protocols.get("vcsf_final_adaptive_refresh")
        if adaptive_refresh is not None:
            expected = dict(paired_protocol="vcsf_final_oblivious_refresh",
                budget_profile="vcsf_single_source_ablation_observed_cost", methods=[],
                seed_schedule="full_val_numeric_image_position", generation_groups=20,
                generated_evaluation_cells=320, reused_identity_cells=32,
                evaluation_cells=352, execution_admitted=False)
            if any(type(adaptive_refresh.get(k)) is not type(v) or adaptive_refresh.get(k) != v
                   for k, v in expected.items()):
                raise RegistryError("Final adaptive refresh changed its fixed scope")
            if self.adaptive_preprocessing_policy.get("seed_schedule") != expected["seed_schedule"]:
                raise RegistryError("Final adaptive refresh seed policy disagrees with preprocessing")

        oblivious = self.protocols.get("vcsf_final_oblivious_refresh")
        if oblivious is not None:
            expected = dict(dataset="coco", split="val", sources=self.source_ids()[:2],
                budget_profile="vcsf_single_source_ablation_observed_cost",
                targets="all", methods=[], seed=42, full_images=5000, retained_images=500,
                generation_jobs=0, evaluation_cells=352, device_counts=[1, 2], execution_admitted=False)
            for key, value in expected.items():
                if oblivious.get(key) != value or type(oblivious.get(key)) is not type(value):
                    raise RegistryError("Final oblivious scope changed: " + key)
            reference = self.protocols["vcsf_final_training_state_refresh"]
            for key in ("parameters_sha256", "annotation_sha256", "full_image_ids_sha256", "retained_image_ids_sha256"):
                if oblivious.get(key) != reference[key]:
                    raise RegistryError("Final oblivious input identity changed: " + key)
            bindings = oblivious.get("payload_bindings", {})
            if set(bindings) != set(expected["sources"]):
                raise RegistryError("Final oblivious source bindings incomplete")
            for binding in bindings.values():
                if set(binding) != {"payload_run_sha256", "payload_manifest_sha256"} or any(
                        type(digest) is not str or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                        for digest in binding.values()):
                    raise RegistryError("Invalid final oblivious payload hash binding")

        budget_refresh = self.protocols.get("vcsf_final_budget_refresh")
        if budget_refresh is not None:
            expected = dict(dataset="coco", split="val", images=5000, seed=42,
                sources=self.source_ids()[:2], targets="all", methods=[],
                epsilon_order=list(self.multi_budget_linf["epsilon_order"]),
                stress_only=[self.multi_budget_linf["epsilon_order"][-1]],
                reference_epsilon="4/255", step_size="1/255", execution_admitted=False,
                budget_profile="vcsf_final_radius_observed_cost")
            for key, value in expected.items():
                if budget_refresh.get(key) != value or type(budget_refresh.get(key)) is not type(value):
                    raise RegistryError("Final budget refresh changed its registered scope: " + key)
            digest = budget_refresh.get("parameters_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise RegistryError("Final budget refresh requires a frozen parameter digest")
            profile = self.budget_profiles.get(budget_refresh["budget_profile"], {})
            for key, value in dict(epsilon="registered_axis", allowed_epsilons=expected["epsilon_order"],
                    step_size="1/255", required_iterations=20, max_gradient_evaluations_per_image=20,
                    threat_model="untargeted_additive_linf").items():
                if profile.get(key) != value or type(profile.get(key)) is not type(value):
                    raise RegistryError("Final radius budget profile differs: " + key)

        training_refresh = self.protocols.get("vcsf_final_training_state_refresh")
        if training_refresh is not None:
            reference = self.training_state_fullval_confirmation
            expected = dict(dataset=reference["dataset"], split=reference["evaluation_split"],
                source=reference["source"], target=reference["victim_model"],
                victim_states=reference["victim_states"], full_images=reference["formal_images"],
                seed=reference["seed"], retained_images=500, methods=[], device_counts=[1, 2],
                generation_jobs=0, inference_jobs=2, metric_records=4, reuse_payload=True,
                retained_mode="official_coco_from_same_full_predictions",
                budget_profile="vcsf_single_source_ablation_observed_cost")
            for key, value in expected.items():
                if training_refresh.get(key) != value or type(training_refresh.get(key)) is not type(value):
                    raise RegistryError("Final training-state refresh changed its scope: " + key)
            hashes = [training_refresh.get(key) for key in (
                "parameters_sha256", "payload_run_sha256", "payload_manifest_sha256",
                "annotation_sha256", "full_image_ids_sha256", "retained_image_ids_sha256",
                "pair_qualification_sha256")]
            checkpoints = training_refresh.get("checkpoint_sha256", {})
            if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(reference["victim_states"]):
                raise RegistryError("Final training-state refresh needs the existing checkpoint pair")
            hashes.extend(checkpoints.values())
            if any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value) for value in hashes):
                raise RegistryError("Final training-state refresh has an invalid SHA256 binding")
            if len(set(checkpoints.values())) != len(checkpoints):
                raise RegistryError("Final training-state checkpoints must differ")

        final_followup = self.protocols.get("vcsf_final_followup_preparation")
        if final_followup is not None:
            expected = dict(
                budget_profile="vcsf_single_source_ablation_observed_cost",
                methods=[], targets="all", main_sources="canonical_six", main_seed=42,
                datasets={"coco": {"split": "val", "images": 5000},
                          "voc": {"split": "val", "images": 4952}},
                stability_dataset="coco", stability_source=self.source_ids()[0],
                stability_seeds=[42, 43, 44, 45, 46], seed_schedule="selected_position",
                device_counts=[1, 2],
                planning_entrypoint="experiments/prepare_vcsf_final_followup.py",
                registration="docs/research/vcsf-final-followup-registration-20260917.md",
                freeze_record="docs/research/vcsf-final-method-freeze-20260917.json",
                freeze_sha256="a697b1331cf54808633bcfc57d6f07adb630ef41beb7fea709402b749e1418d0",
                reuse_policy="exact_evidence_and_scope_qualification_required",
                full_target_panel_required=True, all_twelve_metrics_required=True,
                preparation_only=True, runner_armed=False, execution_authorized=False,
                independent_confirmation=False, bootstrap_required=False)
            if (any(final_followup.get(key) != value for key, value in expected.items())
                    or list(final_followup["datasets"]) != ["coco", "voc"]
                    or any(type(final_followup.get(key)) is not bool for key in
                        ("full_target_panel_required", "all_twelve_metrics_required",
                         "preparation_only", "runner_armed", "execution_authorized",
                         "independent_confirmation", "bootstrap_required"))
                    or type(final_followup.get("main_seed")) is not int
                    or any(type(value) is not int for value in final_followup["stability_seeds"]
                           + final_followup["device_counts"])
                    or any(type(row["images"]) is not int for row in final_followup["datasets"].values())):
                raise RegistryError("Final follow-up preparation changed its frozen scope")

        operator_points = self.protocols.get("vcsf_scale_operator_points")
        if operator_points is not None:
            if not (operator_points.get("parent_protocol") == "vcsf_scale_point_stages"
                and operator_points.get("stage") == "operators_at_anchor_width"
                and operator_points.get("sources") == self.source_ids()[:1]
                and operator_points.get("targets") == "all"
                and operator_points.get("original_group_ids") == [29, 2]
                and operator_points.get("new_group_ids") == [3, 4, 5]
                and all(type(n) is int for n in operator_points["original_group_ids"] + operator_points["new_group_ids"])
                and all(type(operator_points.get(k)) is int and operator_points[k] == value for k, value in
                    (("images",5000),("seed",42),("observation_cells",80),("new_cells",48),("contrasts",6),
                     ("diagnostic_images_per_new_group",1)))
                and all(operator_points.get(k) is True for k in ("full_stage_input_required","all_twelve_metrics_required"))
                and all(operator_points.get(k) is False for k in ("bootstrap_required","new_model_calls",
                    "original_roots_modified","independent_confirmation","significance_claimed","automatic_promotion"))):
                raise RegistryError("Operator point analysis changed its complete-panel scope")

        point_stages = self.protocols.get("vcsf_scale_point_stages")
        if point_stages is not None:
            if not (point_stages.get("parent_protocol") == "vcsf_single_source_scale_execution"
                and type(point_stages.get("contract_version")) is int and point_stages["contract_version"] == 2
                and point_stages.get("point_protocol") == "vcsf_scale_point_analysis"
                and point_stages.get("owner_decision") == "docs/research/vcsf-exploration-statistics-amendment-20260909.md"
                and point_stages.get("stage_order") == ["baseline_and_identity", "operators_at_anchor_width", "remaining_widths"]
                and point_stages.get("contrast_counts") == [1, 6, 30]
                and all(type(n) is int for n in point_stages["contrast_counts"])
                and point_stages.get("contrast_assignment") == "earliest_stage_with_all_terms_available"
                and point_stages.get("metric") == "bbox_mAP"
                and point_stages.get("scalar") == "equal_weight_fifteen_source_excluded_targets"
                and point_stages.get("seed_estimand") == "conditional_on_seed42_not_resampled"
                and all(point_stages.get(k) is True for k in ("previous_stage_acceptance_required",
                    "previous_stage_decision_required", "no_early_efficacy_pruning"))
                and all(point_stages.get(k) is False for k in ("bootstrap_required", "significance_claimed",
                    "independent_confirmation", "automatic_promotion", "runner_armed"))):
                raise RegistryError("Point scale stages changed their prospective contract")

        point_analysis = self.protocols.get("vcsf_scale_point_analysis")
        if point_analysis is not None:
            if not (point_analysis.get("parent_protocol") == "vcsf_single_source_scale_execution"
                and point_analysis.get("sources") == self.source_ids()[:1]
                and point_analysis.get("dataset") == "coco" and point_analysis.get("split") == "val"
                and type(point_analysis.get("images")) is int and point_analysis["images"] == 5000
                and type(point_analysis.get("seed")) is int and point_analysis["seed"] == 42
                and point_analysis.get("targets") == "all" and point_analysis.get("methods") == []
                and point_analysis.get("metric") == "bbox_mAP"
                and point_analysis.get("scalar") == "equal_weight_fifteen_source_excluded_targets"
                and point_analysis.get("coefficient_policy") == "preserve_registered_exact_contrasts"
                and point_analysis.get("qualified_replay_reuse") == "original_paths_hashes_and_invocation_identity"
                and point_analysis.get("partial_replay_recovery") == "explicit_new_root_missing_targets_only"
                and point_analysis.get("execution_kind") == "cpu_existing_evidence_only"
                and point_analysis.get("owner_decision") == "docs/research/vcsf-exploration-statistics-amendment-20260909.md"
                and all(point_analysis.get(k) is True for k in ("full_target_panel_required",
                    "all_twelve_metrics_required", "independent_input_audit_required"))
                and all(point_analysis.get(k) is False for k in ("bootstrap_required", "confidence_intervals_produced",
                    "significance_claimed", "independent_confirmation", "automatic_promotion", "new_model_calls",
                    "original_roots_modified", "automatic_resume"))):
                raise RegistryError("Scale point analysis changed its owner-amended scope")

        final_attribution = self.protocols.get("vcsf_final_attribution_research")
        if final_attribution is not None:
            expected = dict(dataset="coco", split="val", images=5000, seed=42,
                sources=self.source_ids()[:1], targets="all", methods=[], device_counts=[1, 2, 4],
                ablation_study="vcsf_single_source_ablation",
                budget_profile="vcsf_single_source_ablation_observed_cost",
                owner_decision="docs/research/vcsf-whole-ablation-revision-20260911.md",
                attribution_request="docs/research/vcsf-final-attribution-request-20260915.json",
                attribution_request_sha256="ebc961c382c54b64530cc94abc5f5598dc557370d6209657935feda4a5b1f59e",
                configuration_cap=35, candidate_search_cycles=0,
                planning_entrypoint="experiments/prepare_vcsf_final_attribution.py",
                execution_entrypoint="experiments/vcsf_final_attribution_experiments.py",
                scheduling="independent_complete_configuration_jobs",
                execution_lifecycle="full_stage_keep_all_v1", prediction_archive="gzip",
                selection_namespace="final_background_attribution_after_fullval_selection",
                payload_retention=dict(mode="keep_all", deletion_authorized=False,
                    prune_after_seal=False, preserve_failed_and_partial=True),
                isolated_candidate=dict(execution_alias="vcsf_final_attribution_isolated",
                    implementation="src/lgp/attacks/vcsf_common_anchor_isolated.py",
                    class_name="VCSFCommonAnchorCandidate", config_class_name="VCSFCommonAnchorConfig"),
                source_matched_policy="exclude_from_BB_mean_report_separately",
                historical_reuse="exact_configuration_and_qualified_evidence_only")
            if not (all(final_attribution.get(k) == v for k, v in expected.items())
                and all(type(final_attribution.get(k)) is int for k in
                    ("images", "seed", "configuration_cap", "candidate_search_cycles"))
                and all(final_attribution.get(k) is True for k in
                    ("process_local_candidate_overlay", "execution_authorized",
                     "full_target_panel_required", "all_twelve_metrics_required"))
                and all(final_attribution.get(k) is False for k in
                    ("runner_armed", "independent_confirmation", "automatic_promotion",
                     "automatic_resume", "old_run_restart"))):
                raise RegistryError("Final attribution changed its finite registered execution scope")

        common_anchor = self.protocols.get("vcsf_single_source_common_anchor_research")
        if common_anchor is not None:
            if not (common_anchor.get("sources") == self.source_ids()[:1]
                and common_anchor.get("dataset") == "coco" and common_anchor.get("split") == "val"
                and type(common_anchor.get("images")) is int and common_anchor["images"] == 5000
                and type(common_anchor.get("seed")) is int and common_anchor["seed"] == 42
                and common_anchor.get("targets") == "all" and common_anchor.get("methods") == []
                and common_anchor.get("device_counts") == [1, 2, 4]
                and common_anchor.get("ablation_study") == "vcsf_single_source_ablation"
                and common_anchor.get("budget_profile") == "vcsf_single_source_ablation_observed_cost"
                and common_anchor.get("owner_decision") == "docs/research/vcsf-whole-ablation-revision-20260911.md"
                and common_anchor.get("planning_entrypoint") == "experiments/vcsf_common_anchor.py"
                and common_anchor.get("execution_entrypoint") == "experiments/vcsf_common_anchor_experiments.py"
                and common_anchor.get("execution_lifecycle") == "full_stage_keep_all_v1"
                and common_anchor.get("scheduling") == "independent_complete_configuration_jobs"
                and common_anchor.get("prediction_archive") == "gzip"
                and common_anchor.get("payload_retention") == dict(mode="keep_all", deletion_authorized=False,
                    prune_after_seal=False, preserve_failed_and_partial=True)
                and common_anchor.get("isolated_candidate") == dict(execution_alias="vcsf_common_anchor_isolated",
                    implementation="src/lgp/attacks/vcsf_common_anchor_isolated.py",
                    class_name="VCSFCommonAnchorCandidate", config_class_name="VCSFCommonAnchorConfig")
                and common_anchor.get("process_local_candidate_overlay") is True
                and common_anchor.get("execution_authorized") is True
                and common_anchor.get("full_target_panel_required") is True
                and all(common_anchor.get(k) is False for k in ("runner_armed", "independent_confirmation",
                    "automatic_promotion", "automatic_resume", "old_run_restart"))):
                raise RegistryError("Common-anchor protocol changed its bounded execution scope")

        operator_factorial = self.protocols.get("vcsf_single_source_operator_factorial_research")
        if operator_factorial is not None:
            if not (operator_factorial.get("sources") == self.source_ids()[:1]
                and operator_factorial.get("dataset") == "coco"
                and operator_factorial.get("split") == "val"
                and type(operator_factorial.get("images")) is int and operator_factorial["images"] == 5000
                and type(operator_factorial.get("seed")) is int and operator_factorial["seed"] == 42
                and operator_factorial.get("targets") == "all" and operator_factorial.get("methods") == []
                and operator_factorial.get("budget_profile") == "vcsf_retrospective_observed_cost"
                and operator_factorial.get("ablation_study") == "vcsf_single_source_operator_factorial"
                and operator_factorial.get("device_counts") == [1, 2, 4]
                and operator_factorial.get("requested_new_group_limit") == 4
                and type(operator_factorial.get("requested_new_group_limit")) is int
                and operator_factorial.get("execution_entrypoint") == "experiments/vcsf_operator_factorial.py"
                and operator_factorial.get("execution_lifecycle") == "full_stage_keep_all_v1"
                and operator_factorial.get("prediction_archive") == "gzip"
                and operator_factorial.get("payload_retention") == dict(mode="keep_all",
                    deletion_authorized=False, prune_after_seal=False, preserve_failed_and_partial=True)
                and all(type(operator_factorial["payload_retention"][k]) is bool for k in (
                    "deletion_authorized", "prune_after_seal", "preserve_failed_and_partial"))
                and operator_factorial.get("historical_reuse_failure") == "unresolved_not_automatic_new_run"
                and all(operator_factorial.get(k) is True for k in (
                    "process_local_candidate_overlay", "full_target_panel_required", "all_twelve_metrics_required"))
                and all(operator_factorial.get(k) is False for k in (
                    "runner_armed", "automatic_resume", "old_run_restart", "independent_confirmation",
                    "automatic_promotion", "bootstrap_required"))):
                raise RegistryError("Operator factorial changed its single-source fixed-width scope")
            study = self.ablation_studies.get(operator_factorial["ablation_study"], {})
            axes = study.get("factor_order", [])
            levels = study.get("factor_levels", {})
            variants = study.get("variants", {})
            rows = study.get("row_order", [])
            if not (axes == ["image_interpolation", "placement", "image_padding"]
                and set(levels) == set(axes)
                and all(isinstance(levels[k], list) and len(levels[k]) == 2
                    and all(isinstance(v, str) for v in levels[k]) and len(set(levels[k])) == 2 for k in axes)
                and len(rows) == len(set(rows)) == 8 and rows == list(variants)
                and study.get("half_width") == "1/3"
                and study.get("base_variant") == "levels_two"
                and study.get("base_study") == "vcsf_finalization_layered"
                and all(set(variants[row]) == {"parameters"}
                    and set(variants[row]["parameters"]) == set(axes) for row in rows)):
                raise RegistryError("Operator factorial requires eight complete three-axis cells at half-width 1/3")
            actual = [tuple(variants[row]["parameters"][axis] for axis in axes) for row in rows]
            if len(set(actual)) != 8 or set(actual) != set(product(*(levels[axis] for axis in axes))):
                raise RegistryError("Operator factorial contains a missing, duplicate or out-of-domain corner")
            previous = study.get("historical_reference_variants", {})
            requested = study.get("requested_new_variants", [])
            historical = self.ablation_studies.get("vcsf_single_source_scale", {})
            if not (len(previous) == len(requested) == 4 and len(set(requested)) == 4
                and set(previous).isdisjoint(requested) and set(previous) | set(requested) == set(rows)
                and study.get("historical_reuse_policy") == "exact_identity_and_complete_qualified_evidence_before_partition"
                and study.get("historical_failure_policy") == "unresolved_not_automatic_new_run"):
                raise RegistryError("Operator factorial must preserve four unqualified old references and four requested new cells")
            for row, old_name in previous.items():
                old_row, separator, width_index = old_name.rpartition("_h")
                if not (separator and width_index.isdigit() and old_row in historical.get("variants", {})
                    and int(width_index) < len(historical.get("half_widths", []))
                    and Fraction(str(historical["half_widths"][int(width_index)])) == Fraction(study["half_width"])
                    and variants[row]["parameters"] == historical["variants"][old_row]["parameters"]):
                    raise RegistryError("Operator factorial historical row does not preserve its original fields and width")
            analysis = study.get("analysis_policy", {})
            factorials = analysis.get("factorials", [])
            if not (isinstance(factorials, list) and len(factorials) == 1
                and isinstance(factorials[0], dict)
                and factorials[0].get("axes") == {axis: [axis] for axis in axes}
                and list(factorials[0].get("axes", {})) == axes
                and factorials[0].get("block") == "operators"
                and factorials[0].get("seeds") == [operator_factorial["seed"]]
                and factorials[0].get("low") in variants and factorials[0].get("high") in variants
                and all(variants[factorials[0]["low"]]["parameters"][axis] == levels[axis][0]
                    and variants[factorials[0]["high"]]["parameters"][axis] == levels[axis][1] for axis in axes)
                and study.get("blocks", {}).get("operators") == dict(
                    variants=rows, seeds=[operator_factorial["seed"]], allowed_fields=axes)):
                raise RegistryError("Operator factorial bit coding and contrast low/high levels disagree")
            if not (analysis.get("coefficient_convention") == "unnormalized_active_axis_finite_difference_averaged_over_inactive_axes"
                and analysis.get("expected_contrasts") == dict(averaged_factorial=7, conditional_factorial=18, anchor=7, total=32)
                and all(analysis.get("uncertainty", {}).get(k) is False for k in (
                    "bootstrap_required", "significance_claimed", "confidence_intervals_produced",
                    "independent_confirmation", "selection_adjusted_inference"))):
                raise RegistryError("Operator factorial contrast or uncertainty convention changed")

        scale_study = self.protocols.get("vcsf_single_source_scale_research")
        if scale_study is not None:
            if not (
                scale_study.get("sources") == self.source_ids()[:1]
                and scale_study.get("dataset") == "coco"
                and scale_study.get("split") == "val"
                and scale_study.get("images") == 5000
                and type(scale_study.get("seed")) is int and scale_study["seed"] == 42
                and scale_study.get("targets") == "all"
                and scale_study.get("methods") == []
                and scale_study.get("budget_profile") == "vcsf_retrospective_observed_cost"
                and scale_study.get("ablation_study") == "vcsf_single_source_scale"
                and scale_study.get("device_counts") == [1, 2]
                and scale_study.get("full_target_panel_required") is True
                and all(scale_study.get(key) is False for key in (
                    "automatic_resume", "old_run_restart", "independent_confirmation", "automatic_promotion"))
            ):
                raise RegistryError("Single-source scale study changed its authorized scope")
            bridge = self.ablation_studies["vcsf_single_source_scale"].get("bridge_diagnostic", {})
            if not (bridge.get("image_positions") == [0, 1]
                and bridge.get("image_ids") == [139, 285]
                and bridge.get("attack_seeds") == [42, 43]
                and bridge.get("trajectory_repeats") == 2
                and bridge.get("fixed_state_repeats") == 3
                and bridge.get("fixed_state_phases") == ["risk", "feature"]
                and bridge.get("complete_attack_calls") == 24
                and bridge.get("fixed_state_backward_calls") == 72
                and bridge.get("AP_evaluations") == 0
                and bridge.get("automatic_extra_probes") is False
                and bridge.get("historical_reuse_granted") is False):
                raise RegistryError("Scale bridge changed its bounded diagnostic scope")

        scale_execution = self.protocols.get("vcsf_single_source_scale_execution")
        if scale_execution is not None:
            staged = scale_execution.get("staged_analysis", {})
            if not (staged.get("contrast_assignment") == "earliest_stage_with_all_terms_available"
                and staged.get("stage_order") == self.ablation_studies["vcsf_single_source_scale"]["stage_order"]
                and staged.get("family_sizes") == [1, 6, 30]
                and all(type(n) is int for n in staged["family_sizes"])
                and staged.get("uncertainty_definition") == "first_stage_analysis"
                and staged.get("multiplicity_scope") == "separate_prospectively_registered_stage_families"
                and all(staged.get(k) is True for k in ("previous_stage_acceptance_required",
                    "previous_stage_decision_required", "no_early_efficacy_pruning"))
                and all(staged.get(k) is False for k in ("across_stage_error_control_claimed",
                    "prior_intervals_reused", "selection_adjusted_inference", "runner_armed"))):
                raise RegistryError("Scale stage analysis partition or execution gates changed")

        whole = self.protocols.get("vcsf_single_source_ablation_research")
        if whole is not None:
            if not (whole.get("sources") == self.source_ids()[:1]
                and whole.get("dataset") == "coco" and whole.get("split") == "val"
                and whole.get("images") == 5000 and type(whole.get("seed")) is int
                and whole["seed"] == 42 and whole.get("targets") == "all"
                and whole.get("methods") == [] and whole.get("device_counts") == [1, 2]
                and whole.get("budget_profile") == "vcsf_single_source_ablation_observed_cost"
                and whole.get("ablation_study") == "vcsf_single_source_ablation"
                and whole.get("full_target_panel_required") is True
                and all(whole.get(key) is False for key in (
                    "automatic_resume", "old_run_restart", "independent_confirmation", "automatic_promotion", "runner_armed"))):
                raise RegistryError("Whole single-source ablation changed its authorized scope")
            study = self.ablation_studies[whole["ablation_study"]]
            blocks, order = study.get("blocks", {}), study.get("execution_order", [])
            if not (len(order) == len(set(order)) and set(order) == set(blocks)
                and study.get("base_scale_study") == whole.get("prerequisite_study")
                and study.get("base_scale_study") in self.ablation_studies):
                raise RegistryError("Whole-ablation stage order or prerequisite is invalid")
            seen = set()
            for block in blocks.values():
                names = block.get("variants", [])
                if not (names and len(names) == len(set(names))
                    and set(names) <= set(study["variants"])
                    and block.get("seeds") == [whole["seed"]]):
                    raise RegistryError("Whole-ablation block has missing, duplicate or out-of-scope cells")
                seen.update(names)
            if seen != set(study["variants"]):
                raise RegistryError("Whole-ablation has unscheduled variants")
            stability = whole.get("final_stability", {})
            if not (stability.get("configuration_scope") == "author_frozen_final_only"
                and stability.get("seeds") == [42, 43, 44, 45, 46]
                and stability.get("source_scope") == "single_source_ablation"
                and stability.get("execution_armed") is False
                and stability.get("independent_confirmation") is False):
                raise RegistryError("Whole-ablation stability must remain final-configuration-only and unarmed")
            budget = self.budget_profiles[whole["budget_profile"]]
            if not (budget.get("epsilon") == "4/255"
                and budget.get("max_gradient_evaluations_per_image") == 20
                and budget.get("allowed_iterations_by_initialization") == {
                    "detector": [20], "none": [19, 20], "random_sign": [19, 20]}):
                raise RegistryError("Whole-ablation initializer budgets changed")
            selection = study.get("analysis_policy", {}).get("selection", {})
            if not (selection.get("complete_target_panel_required") is True
                and selection.get("changed_baseline_requires_rebinding") is True
                and type(selection.get("final_consistency_passes")) is int
                and selection["final_consistency_passes"] == 1
                and all(selection.get(key) is False for key in (
                    "automatic_promotion", "combine_winning_settings_automatically", "positive_effect_required"))):
                raise RegistryError("Whole-ablation selection or finite-closure policy changed")

        control = self.protocols.get("vcsf_single_source_ablation_preflight")
        if control is not None:
            if not (control.get("parent_protocol") == "vcsf_single_source_ablation_research"
                and whole is not None and control.get("budget_profile") == whole["budget_profile"]
                and control.get("methods") == []
                and control.get("entrypoint") == "experiments/vcsf_single_source_ablation_preflight.py"
                and control.get("source") == self.source_ids()[0]
                and control.get("dataset") == "coco" and control.get("split") == "val"
                and type(control.get("images_per_group")) is int and control["images_per_group"] == 1
                and control.get("image_selection") == "first_canonical_validation_image"
                and type(control.get("seed")) is int and control["seed"] == 42
                and control.get("device_counts") == [1, 2]
                and all(type(count) is int for count in control["device_counts"])
                and control.get("scheduling") == "round_robin_independent_complete_configuration_jobs"
                and control.get("complete_selected_stage_required") is True
                and type(control.get("target_AP_evaluations")) is int and control["target_AP_evaluations"] == 0
                and control.get("diagnostic_only") is True and control.get("payload_retention") == "keep_all"
                and all(control.get(key) is False for key in ("deletion_authorized", "automatic_resume",
                    "historical_reuse_qualified", "formal_execution_admission", "scientific_acceptance",
                    "isolated_physical_cost_measurement", "automatic_promotion"))):
                raise RegistryError("Single-source ablation preflight changed its bounded diagnostic scope")

        execution_contract = self.protocols.get("vcsf_single_source_ablation_execution_contract")
        if execution_contract is not None:
            if not (whole is not None
                and execution_contract.get("parent_protocol") == "vcsf_single_source_ablation_research"
                and execution_contract.get("budget_profile") == whole["budget_profile"]
                and execution_contract.get("methods") == []
                and execution_contract.get("entrypoint") == "experiments/prepare_vcsf_single_source_ablation_execution.py"
                and execution_contract.get("supported_stages") == ["core"]
                and execution_contract.get("device_counts") == [1, 2]
                and all(type(n) is int for n in execution_contract["device_counts"])
                and execution_contract.get("scheduling") == "round_robin_independent_complete_configuration_jobs"
                and execution_contract.get("complete_selected_stage_required") is True
                and execution_contract.get("historical_reuse") == "unassessed_do_not_subtract"
                and execution_contract.get("input_binding") == "canonical_full_split_ordered_ids_selected_position_seeds"
                and execution_contract.get("result_binding") == "all_groups_all_targets_all_twelve_metrics"
                and execution_contract.get("analysis_binding") == "exact_registered_stage_contrasts_point_estimates_only"
                and execution_contract.get("payload_retention") == dict(mode="keep_all",
                    deletion_authorized=False, prune_after_seal=False, preserve_failed_and_partial=True)
                and all(type(execution_contract["payload_retention"].get(k)) is bool
                    for k in ("deletion_authorized", "prune_after_seal", "preserve_failed_and_partial"))
                and execution_contract.get("failure_policy") == "stop_only_owned_workers_preserve_all_evidence"
                and execution_contract.get("output_policy") == "fresh_direct_namespace_child_no_merge"
                and execution_contract.get("capacity_policy") == "full_stage_keep_all_bound_plus_failure_reserve_before_admission"
                and execution_contract.get("required_admission_evidence") == [
                    "complete_preceding_scale_acceptance_and_selection",
                    "actual_source_controls_and_independent_saved_control_acceptance",
                    "source_bound_runtime_and_full_input_identity",
                    "historical_reuse_independent_adjudication",
                    "actual_single_and_dual_device_executor_qualification",
                    "full_stage_keep_all_capacity_and_failure_reserve",
                    "bound_prediction_index_and_exact_analysis_family",
                    "required_cost_evidence_and_independent_contract_acceptance",
                    "independent_formal_stage_admission"]
                and execution_contract.get("preparation_only") is True
                and all(execution_contract.get(key) is False for key in (
                    "execution_profile_installed", "runner_armed", "automatic_resume", "old_run_restart",
                    "automatic_promotion", "scientific_acceptance", "independent_confirmation"))):
                raise RegistryError("Core execution-contract preparation or keep-all gates changed")

        reuse = self.protocols.get("vcsf_single_source_ablation_reuse")
        if reuse is not None:
            if not (whole is not None and execution_contract is not None
                and reuse.get("parent_protocol") == "vcsf_single_source_ablation_research"
                and reuse.get("budget_profile") == whole["budget_profile"] and reuse.get("methods") == []
                and reuse.get("entrypoint") == "experiments/vcsf_single_source_ablation_reuse.py"
                and reuse.get("audit_entrypoint") == "experiments/audit_vcsf_single_source_ablation_reuse.py"
                and reuse.get("supported_stages") == ["core"]
                and all(type(reuse.get(key)) is int and reuse[key] == count for key, count in
                    (("logical_groups", 8), ("logical_target_slots", 128), ("registered_contrasts", 32),
                     ("original_groups", 17), ("original_target_cells", 272)))
                and reuse.get("states") == ["qualified_reuse", "requires_new", "unresolved"]
                and all(reuse.get(key) is True for key in ("complete_scale_acceptance_and_selection_required",
                    "published_runtime_input_reconstruction_required", "original_result_provenance_preserved",
                    "independent_reuse_acceptance_required"))
                and reuse.get("unknown_partition_counts", "missing") is None
                and all(reuse.get(key) is False for key in ("diagnostic_formal_qualification", "physical_cost_inheritance",
                    "trajectory_identity_claim", "formal_execution_admission", "scientific_acceptance", "automatic_promotion"))):
                raise RegistryError("Core reuse adjudication changed its complete provenance or non-admission scope")

        smoke = self.protocols.get("vcsf_single_source_ablation_executor_smoke")
        if smoke is not None:
            if not (whole is not None and execution_contract is not None
                and smoke.get("parent_protocol") == "vcsf_single_source_ablation_research"
                and smoke.get("budget_profile") == whole["budget_profile"]
                and smoke.get("methods") == []
                and smoke.get("entrypoint") == "experiments/vcsf_single_source_ablation_executor_smoke.py"
                and smoke.get("supported_stages") == ["core"]
                and type(smoke.get("images_per_group")) is int and smoke["images_per_group"] == 1
                and smoke.get("image_selection") == "first_canonical_validation_image"
                and smoke.get("device_counts") == [1, 2]
                and all(type(n) is int for n in smoke["device_counts"])
                and smoke.get("scheduling") == "round_robin_independent_complete_configuration_jobs"
                and smoke.get("complete_selected_stage_required") is True
                and smoke.get("targets") == "all" and smoke.get("prediction_archive") == "gzip"
                and smoke.get("diagnostic_only") is True
                and smoke.get("historical_reuse") == "no_historical_results_consumed_diagnostic_only"
                and smoke.get("payload_retention") == execution_contract["payload_retention"]
                and all(type(smoke["payload_retention"].get(k)) is bool
                    for k in ("deletion_authorized", "prune_after_seal", "preserve_failed_and_partial"))
                and isinstance(smoke.get("capacity"), dict)
                and set(smoke["capacity"]) == {"payload_bytes_per_group",
                    "prediction_and_metadata_bytes_per_group", "failure_reserve_bytes"}
                and all(type(n) is int and n > 0 for n in smoke["capacity"].values())
                and all(smoke.get(k) is False for k in ("formal_execution_admission",
                    "scientific_acceptance", "automatic_promotion"))):
                raise RegistryError("Core executor smoke changed its complete bounded diagnostic scope")

        amendment = self.protocols.get("vcsf_seed42_amended_analysis")
        if amendment is not None:
            parent = self.protocols.get(amendment.get("parent_protocol"), {})
            if not (
                amendment.get("parent_protocol") == "vcsf_fullval_retrospective_ablation"
                and amendment.get("budget_profile") == parent.get("budget_profile")
                and amendment.get("methods") == []
                and amendment.get("entrypoint") == "experiments/vcsf_seed42_analysis.py"
                and amendment.get("execution_kind") == "cpu_existing_evidence_only"
                and amendment.get("included_attack_seeds") == [42]
                and amendment.get("contrast_retention") == "all_terms_within_included_attack_seeds"
                and amendment.get("coefficient_policy") == "retain_original_exact_coefficients"
                and amendment.get("views") == ["pooled_source_excluded", "each_source_excluded"]
                and amendment.get("multiplicity") == "one_joint_family_centered_max_absolute_bootstrap_error"
                and amendment.get("seed_estimand") == "conditional_on_declared_attack_seeds_not_resampled"
                and amendment.get("original_resample_bytes") == "preserve_without_regeneration_or_metadata_relabeling"
                and amendment.get("expected_cpu_subset") == {
                    "tasks": 942, "cells": 896, "new_group_bindings": 46, "reused_source_groups": 10,
                    "qualified_complete_reuse_cells": 1, "new_invocations": 941}
                and all(amendment.get(key) is True for key in (
                    "exact_complete_prefix_required", "independent_input_audit_required", "amendment_after_interim_observation"))
                and all(amendment.get(key) is False for key in (
                    "selection_adjusted_inference", "finite_sample_guarantee", "independent_confirmation",
                    "original_plan_completion", "automatic_promotion"))
                and type(amendment.get("new_model_calls")) is int and amendment["new_model_calls"] == 0
            ):
                raise RegistryError("Seed-42 analysis amendment changed its retrospective evidence-only scope")

        execution = self.protocols.get("vcsf_layered_efficacy_execution")
        if execution is not None:
            parent = self.protocols.get(execution.get("parent_protocol"), {})
            retention = execution.get("payload_retention", {})
            reference = self.protocols.get("vcsf_common2_cost_calibration", {})
            if not (
                execution.get("parent_protocol") == "vcsf_fullval_retrospective_ablation"
                and execution.get("budget_profile") == parent.get("budget_profile")
                and execution.get("methods") == []
                and execution.get("device_counts") == [1, 2]
                and execution.get("scheduling") == "alternating_source_to_gpu_by_configuration_seed_pair"
                and execution.get("complete_group_atomicity") is True
                and execution.get("automatic_resume") is False
                and execution.get("accepted_group_rerun") is False
                and execution.get("independent_confirmation") is False
                and execution.get("prediction_archive") == "gzip"
                and retention.get("mode") == "fixed_count_after_group_validation"
                and retention.get("retained_images") == reference.get("retained_images")
                and retention.get("selection_algorithm") == reference.get("selection_algorithm")
                and retention.get("retained_image_ids_sha256") == reference.get("retained_image_ids_sha256")
                and retention.get("prune_after_complete_targets") == len(self.target_ids())
                and all(retention.get(field) is True for field in (
                    "preserve_all_predictions", "preserve_all_manifests",
                    "preserve_failed_groups", "preserve_historical_roots"))
                and retention.get("authorization") == "explicit_owner_new_roots_only_20260906"
                and execution.get("entrypoint") == "experiments/vcsf_layered_efficacy.py"
            ):
                raise RegistryError("Layered efficacy execution/lifecycle declaration differs from its parent")

        multi_budget = self.multi_budget_linf
        multi_budget_id = (
            "coco_fullval2017_common2_multi_budget_linf_zero_query_transfer"
        )
        if multi_budget.get("benchmark") != multi_budget_id:
            raise RegistryError("multi_budget_linf.yaml has the wrong benchmark id")
        multi_protocol = self.protocols.get(multi_budget_id)
        if not isinstance(multi_protocol, Mapping):
            raise RegistryError("Multi-budget L-infinity protocol is missing")
        for field in ("sources", "methods", "epsilon_order"):
            if list(multi_budget.get(field, [])) != list(
                multi_protocol.get(field, [])
            ):
                raise RegistryError(
                    "Multi-budget L-infinity {} differs from protocols.yaml".format(
                        field
                    )
                )
        if multi_budget.get("budget_profile") != multi_protocol.get(
            "budget_profile"
        ):
            raise RegistryError(
                "Multi-budget L-infinity budget profile differs from protocols.yaml"
            )
        if list(multi_budget.get("sources", [])) != [
            "faster_rcnn_r50",
            "mask_rcnn_swin_t",
        ]:
            raise RegistryError("Multi-budget Common-2 source order is invalid")
        if list(multi_budget.get("methods", [])) != [
            "sfim_b",
            "naa",
            "numbod",
            "hifa",
            "mlfadv",
            "tog",
            "augtrans",
            "lgp",
            "afog",
            "osfd",
            "svfta",
        ]:
            raise RegistryError("Multi-budget method order is invalid")
        expected_epsilons = ["2/255", "4/255", "8/255", "16/255", "32/255"]
        if list(multi_budget.get("epsilon_order", [])) != expected_epsilons:
            raise RegistryError("Multi-budget epsilon order is invalid")
        roles = multi_budget.get("epsilon_roles")
        if not isinstance(roles, Mapping) or list(roles) != expected_epsilons:
            raise RegistryError("Multi-budget epsilon roles are invalid")
        if roles.get("32/255") != "high_distortion_stress":
            raise RegistryError("32/255 must remain a labelled stress point")
        controlled = multi_budget.get("controlled_factor")
        if not isinstance(controlled, Mapping) or controlled.get(
            "changed_parameter"
        ) != "eps":
            raise RegistryError("Multi-budget study must change only eps")
        reference = multi_budget.get("reference")
        if not isinstance(reference, Mapping) or {
            "protocol": reference.get("protocol"),
            "epsilon": reference.get("epsilon"),
            "policy": reference.get("policy"),
        } != {
            "protocol": "coco_all_methods",
            "epsilon": "4/255",
            "policy": "reuse_accepted_records_with_hash_binding",
        }:
            raise RegistryError("Multi-budget accepted-reference policy is invalid")
        if reference.get("accepted_generation_commit") != (
            "ae9da252a8a35d82609af3f3aba8b1d203041670"
        ):
            raise RegistryError("Multi-budget accepted generation commit is invalid")
        implementation_audit = reference.get("implementation_equivalence_audit")
        if not isinstance(implementation_audit, Mapping) or {
            "unchanged_paths": list(implementation_audit.get("unchanged_paths", [])),
            "approved_additive_default_noop_paths": list(
                implementation_audit.get(
                    "approved_additive_default_noop_paths", []
                )
            ),
            "approved_patch_sha256": implementation_audit.get(
                "approved_patch_sha256"
            ),
        } != {
            "unchanged_paths": [
                "configs/attacks",
                "configs/models.yaml",
                "src/lgp/attacks",
                "src/lgp/metrics.py",
            ],
            "approved_additive_default_noop_paths": [
                "src/lgp/adapters/openmmlab.py",
                "src/lgp/modeling.py",
                "src/lgp/runners/attack.py",
                "src/lgp/runners/evaluate.py",
                "src/lgp/data/coco.py",
            ],
            "approved_patch_sha256": (
                "784b27a58119f94b3fbb67f9813d9cab8f7382be3c8d0d11958a75753150f7c9"
            ),
        }:
            raise RegistryError(
                "Multi-budget implementation-equivalence audit is invalid"
            )
        formal_counts = multi_budget.get("formal_counts")
        expected_counts = {
            "sources": 2,
            "methods": 11,
            "targets": 16,
            "epsilon_budgets": 5,
            "attack_groups": 110,
            "reference_attack_groups": 22,
            "new_attack_groups": 88,
            "conceptual_payload_images": 550000,
            "reference_payload_images": 110000,
            "new_payload_images": 440000,
            "clean_evaluations": 16,
            "attack_evaluations": 1760,
            "reference_attack_evaluations": 352,
            "new_attack_evaluations": 1408,
            "records": 1776,
            "failed_records": 0,
        }
        if not isinstance(formal_counts, Mapping) or {
            key: int(formal_counts.get(key, -1)) for key in expected_counts
        } != expected_counts:
            raise RegistryError("Multi-budget formal counts are invalid")
        eligibility = multi_budget.get("eligibility")
        if not isinstance(eligibility, Mapping) or (
            eligibility.get("full_coco_val_claim") is not True
            or eligibility.get("certified_robustness_claim") is not False
            or eligibility.get("require_zero_remaining_new_payload_pngs")
            is not True
        ):
            raise RegistryError("Multi-budget eligibility policy is invalid")
        current_whitebox = self.protocols.get("whitebox")
        if current_whitebox is not None:
            main = self.protocols["main_transfer"]
            expected_whitebox = {
                "datasets": ["coco", "voc"], "allowed_datasets": ["coco", "voc"],
                "split": "val", "budget_profile": "compute_matched",
                "sources": self.source_ids(), "targets": "source", "source_matched_only": True,
                "report_method_order": "declared", "metrics": list(COCO_BBOX_METRICS),
                "payload_retention": {"mode": "keep_all"}, "prediction_archive": {"format": "gzip"},
                "entrypoint": "experiments/whitebox.py",
            }
            expected_whitebox.update({key: main[key] for key in (
                "methods", "allowed_methods", "method_budget_profiles")})
            if (current_whitebox.get("source_matched_only") is not True
                    or any(current_whitebox.get(key) != value for key, value in expected_whitebox.items())):
                raise RegistryError("Current white-box reproduction changed its registered main-panel scope")
        current_background = self.protocols.get("current_vcsf_final_background")
        if current_background is not None:
            study = self.ablation_studies["vcsf_final_background_validation"]
            expected_background = {
                "datasets": ["coco"], "allowed_datasets": ["coco"], "split": "val",
                "budget_profile": "vcsf_final_background_observed_cost",
                "sources": ["faster_rcnn_r50"], "targets": "all",
                "methods": ["vcsf"], "allowed_methods": ["vcsf"], "seed": 42,
                "seed_schedule": "full_val_numeric_image_position",
                "full_images": self.datasets["coco"].split("val").expected_images,
                "canonical_variant_study": "vcsf_final_background_validation",
                "device_counts": [1, 2], "metrics": list(COCO_BBOX_METRICS),
                "payload_retention": {"mode": "keep_all"}, "prediction_archive": {"format": "gzip"},
                "historical_result_inheritance": False, "configuration_search": False,
                "entrypoint": "experiments/current_vcsf_final_background.py",
            }
            if (any(current_background.get(key) != value for key, value in expected_background.items())
                    or study["anchor"] != "A10" or len(study["row_order"]) != 10
                    or set(study["row_order"]) != set(study["variants"])):
                raise RegistryError("Current final-background reproduction changed its canonical scope")
        current_radius = self.protocols.get("current_vcsf_radius")
        if current_radius is not None:
            radius_budget = self.budget_profiles["vcsf_final_radius_observed_cost"]
            expected_radius = {
                "datasets": ["coco"], "allowed_datasets": ["coco"], "split": "val",
                "budget_profile": "vcsf_final_radius_observed_cost",
                "sources": ["faster_rcnn_r50", "mask_rcnn_swin_t"], "targets": "all",
                "methods": ["vcsf"], "allowed_methods": ["vcsf"], "seed": 42,
                "full_images": self.datasets["coco"].split("val").expected_images,
                "epsilon_order": radius_budget["allowed_epsilons"],
                "reference_epsilon": "4/255", "stress_epsilons": ["32/255"],
                "device_counts": [1, 2], "metrics": list(COCO_BBOX_METRICS),
                "payload_retention": {"mode": "keep_all"}, "prediction_archive": {"format": "gzip"},
                "entrypoint": "experiments/current_vcsf_radius.py",
            }
            if any(current_radius.get(key) != value for key, value in expected_radius.items()):
                raise RegistryError("Current VCSF radius reproduction changed its registered scope")
            studies = current_radius.get("ablation_studies", [])
            if len(studies) != len(current_radius["epsilon_order"]) or len(set(studies)) != len(studies):
                raise RegistryError("Current radius studies must cover the ordered five-radius axis")
            for epsilon, study_id in zip(current_radius["epsilon_order"], studies):
                study = self.ablation_studies.get(study_id, {})
                variants = study.get("variants", {})
                if (study.get("method") != "vcsf" or study.get("protocol") != "current_vcsf_radius"
                        or len(variants) != 1 or next(iter(variants.values())) != {"parameters": {"eps": epsilon}}):
                    raise RegistryError("Current radius study may change only its exact registered epsilon")
        current_cost = self.protocols.get("current_paired_cost")
        if current_cost is not None:
            historical_cost = self.protocols["vcsf_common2_cost_calibration"]
            expected_cost = {
                "datasets": ["coco"], "split": "val", "budget_profile": "compute_matched",
                "method_source_protocol": "main_transfer",
                "sources": ["faster_rcnn_r50", "mask_rcnn_swin_t"], "targets": [], "metrics": [],
                "full_images": self.datasets["coco"].split("val").expected_images,
                "execution_schedule": "serial_source_jobs_for_uncontaminated_timing",
                "device_counts": [1, 2], "concurrent_measurements": 1,
                "entrypoint": "experiments/current_paired_cost.py",
            }
            expected_cost.update({key: historical_cost[key] for key in (
                "retained_images", "warmup_images", "seed", "selection_algorithm",
                "all_image_ids_sha256", "retained_image_ids_sha256", "method_order_schedule",
                "allocator_policy", "timing_scope", "reference_method")})
            if any(current_cost.get(key) != value for key, value in expected_cost.items()):
                raise RegistryError("Current paired-cost reproduction changed its registered scope")
        current_state = self.protocols.get("current_training_state_transfer")
        if current_state is not None:
            expected_state = {
                "datasets": ["coco"], "split": "val", "budget_profile": "compute_matched",
                "method_source_protocol": "main_transfer",
                "sources": [self.training_state_transfer["source"]],
                "targets": [self.training_state_transfer["victim_model"]],
                "victim_states": list(self.training_state_transfer["victim_states"]),
                "full_images": self.datasets["coco"].split("val").expected_images,
                "retained_images": self.training_state_transfer["retained_images"],
                "retained_selection_algorithm": self.training_state_transfer["retained_selection_algorithm"],
                "training_pair_registry": "configs/experiments/training_state_transfer.yaml",
                "input_mode": "reuse_current_main_complete_payloads", "generation_jobs": 0,
                "device_counts": [1, 2], "metrics": list(COCO_BBOX_METRICS),
                "entrypoint": "experiments/training_state_transfer.py",
            }
            if any(current_state.get(key) != value for key, value in expected_state.items()):
                raise RegistryError("Current training-state reproduction changed its registered scope")
        current_preprocessing = self.protocols.get("current_oblivious_preprocessing_transfer")
        if current_preprocessing is not None:
            expected_preprocessing = {
                "datasets": ["coco"], "split": "val", "budget_profile": "compute_matched",
                "method_source_protocol": "main_transfer",
                "sources": ["faster_rcnn_r50", "mask_rcnn_swin_t"], "targets": "all",
                "full_images": self.datasets["coco"].split("val").expected_images,
                "retained_images": 500,
                "retained_selection_algorithm": "canonical_equal_bins_center_v1",
                "preprocessing_registry": "configs/experiments/preprocessing_defenses.yaml",
                "input_mode": "reuse_current_main_complete_payloads", "generation_jobs": 0,
                "device_counts": [1, 2], "metrics": list(COCO_BBOX_METRICS),
                "payload_retention": {"mode": "keep_all"}, "prediction_archive": {"format": "gzip"},
                "entrypoint": "experiments/preprocessing_transfer.py",
            }
            if any(current_preprocessing.get(key) != value for key, value in expected_preprocessing.items()):
                raise RegistryError("Current preprocessing reproduction changed its registered scope")
        current_adaptive = self.protocols.get("current_adaptive_preprocessing_transfer")
        if current_adaptive is not None:
            if current_preprocessing is None:
                raise RegistryError("Current adaptive reproduction requires its oblivious pairing")
            expected_adaptive = dict(expected_preprocessing,
                seed=42, seed_schedule=self.adaptive_preprocessing_policy["seed_schedule"],
                paired_protocol="current_oblivious_preprocessing_transfer",
                input_mode="current_main_seed_binding_with_fresh_adaptive_generation",
                identity_payload_reuse_methods=["vcsf"],
                identity_reuse_source_protocol="vcsf_final_adaptive_refresh",
                threat_model="adaptive_source_pipeline_bpda",
                entrypoint="experiments/adaptive_preprocessing_transfer.py")
            expected_adaptive.pop("generation_jobs")
            if (any(current_adaptive.get(key) != value for key, value in expected_adaptive.items())
                    or self.protocols["vcsf_final_adaptive_refresh"]["reused_identity_cells"] != 2 * len(self.paper_order)
                    or self.protocols["lgp_corrected_coco_adaptive_preprocessing"]["identity_payload_reuse_authorized"] is not False):
                raise RegistryError("Current adaptive reproduction changed its registered scope or identity reuse")
        training_state = self.training_state_transfer
        expected_benchmark = "coco_retained500_victim_training_state_transfer"
        if training_state.get("benchmark") != expected_benchmark:
            raise RegistryError(
                "training_state_transfer.yaml has the wrong benchmark id"
            )
        linked_protocol = self.protocols.get(expected_benchmark)
        if not isinstance(linked_protocol, Mapping):
            raise RegistryError("Training-state transfer protocol is missing")
        if list(training_state.get("methods", [])) != list(
            linked_protocol.get("methods", [])
        ):
            raise RegistryError(
                "Training-state method order differs from protocols.yaml"
            )
        if str(training_state.get("source")) not in self.models:
            raise RegistryError("Training-state source is not a registered model")
        if str(training_state.get("victim_model")) not in self.models:
            raise RegistryError("Training-state victim is not a registered model")
        if str(training_state.get("source")) == str(
            training_state.get("victim_model")
        ):
            raise RegistryError(
                "Training-state zero-query source must differ from the victim"
            )
        if list(training_state.get("victim_states", [])) != [
            "standard_control",
            "adversarial_training",
        ]:
            raise RegistryError("Training-state victim state order is invalid")
        if int(training_state.get("retained_images", -1)) != 500:
            raise RegistryError("Training-state protocol must retain exactly 500 images")
        if training_state.get("full_coco_val_claim") is not False:
            raise RegistryError("Training-state protocol cannot claim full COCO val")
        if training_state.get("certified_robustness_claim") is not False:
            raise RegistryError("Training-state protocol cannot claim certification")
        pair_training = training_state.get("pair_training")
        if not isinstance(pair_training, Mapping):
            raise RegistryError("Training-state pair_training must be a mapping")
        if int(pair_training.get("outer_epochs", 0)) <= 0:
            raise RegistryError("Training-state outer_epochs must be positive")
        if int(pair_training.get("replay_steps_per_minibatch", 0)) <= 0:
            raise RegistryError(
                "Training-state replay_steps_per_minibatch must be positive"
            )
        dataset_filter = pair_training.get("dataset_filter")
        if not isinstance(dataset_filter, Mapping):
            raise RegistryError(
                "Training-state dataset_filter must be a mapping"
            )
        if dataset_filter.get("filter_empty_gt") is not False:
            raise RegistryError(
                "Training-state training must retain empty-GT images"
            )
        if int(dataset_filter.get("min_size", -1)) != 0:
            raise RegistryError(
                "Training-state training must disable image-size filtering"
            )
        diagnostic_smoke = pair_training.get("diagnostic_smoke")
        if not isinstance(diagnostic_smoke, Mapping) or {
            "max_images": int(diagnostic_smoke.get("max_images", -1)),
            "required_empty_gt_dataset_index": int(
                diagnostic_smoke.get("required_empty_gt_dataset_index", -1)
            ),
            "required_empty_gt_image_id": int(
                diagnostic_smoke.get("required_empty_gt_image_id", -1)
            ),
        } != {
            "max_images": 47,
            "required_empty_gt_dataset_index": 46,
            "required_empty_gt_image_id": 262284,
        }:
            raise RegistryError(
                "Training-state diagnostic smoke must cover the frozen empty-GT sample"
            )
        adversarial_branch = pair_training.get("adversarial_branch")
        if not isinstance(adversarial_branch, Mapping) or (
            adversarial_branch.get("padding_policy")
            != "zero_outside_img_shape"
        ):
            raise RegistryError(
                "Training-state adversarial padding must remain unperturbed"
            )
        pairing_requirements = training_state.get("pairing_requirements")
        if not isinstance(pairing_requirements, Mapping) or (
            pairing_requirements.get(
                "all_registered_train_images_including_empty_gt"
            )
            is not True
        ):
            raise RegistryError(
                "Training-state pairing must require every registered train image"
            )
        if (
            training_state.get(
                "controlled_training_state_claim_requires_strict_pair"
            )
            is not True
        ):
            raise RegistryError(
                "Training-state claims must require the strict controlled pair"
            )
        formal_counts = training_state.get("formal_counts")
        if not isinstance(formal_counts, Mapping) or {
            "clean_evaluations": int(formal_counts.get("clean_evaluations", -1)),
            "attack_evaluations": int(formal_counts.get("attack_evaluations", -1)),
            "records": int(formal_counts.get("records", -1)),
            "attack_groups": int(formal_counts.get("attack_groups", -1)),
        } != {
            "clean_evaluations": 2,
            "attack_evaluations": 22,
            "records": 24,
            "attack_groups": 11,
        }:
            raise RegistryError("Training-state formal counts are invalid")
        fullval = self.training_state_fullval_confirmation
        fullval_id = (
            "coco_fullval2017_victim_training_state_zero_query_transfer_confirmation"
        )
        if fullval.get("benchmark") != fullval_id:
            raise RegistryError(
                "training_state_fullval_confirmation.yaml has the wrong benchmark id"
            )
        fullval_protocol = self.protocols.get(fullval_id)
        if not isinstance(fullval_protocol, Mapping):
            raise RegistryError("Full-val training-state protocol is missing")
        for field in ("methods", "victim_states", "metrics"):
            if list(fullval.get(field, [])) != list(fullval_protocol.get(field, [])):
                raise RegistryError(
                    "Full-val training-state {} differs from protocols.yaml".format(
                        field
                    )
                )
        if str(fullval.get("source")) != str(training_state.get("source")):
            raise RegistryError("Full-val and retained-500 sources must match")
        if str(fullval.get("victim_model")) != str(
            training_state.get("victim_model")
        ):
            raise RegistryError("Full-val and retained-500 victims must match")
        if int(fullval.get("formal_images", -1)) != 5000:
            raise RegistryError("Full-val confirmation must use exactly 5,000 images")
        if fullval.get("full_coco_val_claim") is not True:
            raise RegistryError("Full-val confirmation must declare its COCO val claim")
        if fullval.get("certified_robustness_claim") is not False:
            raise RegistryError("Full-val confirmation cannot claim certification")
        generation = fullval.get("source_generation")
        if not isinstance(generation, Mapping) or {
            "required_payload_groups": int(
                generation.get("required_payload_groups", -1)
            ),
            "required_images_per_payload": int(
                generation.get("required_images_per_payload", -1)
            ),
            "overlap_reference_images": int(
                generation.get("overlap_reference_images", -1)
            ),
            "payload_retention": generation.get("payload_retention"),
            "target_queries": int(generation.get("target_queries", -1)),
            "target_gradients": int(generation.get("target_gradients", -1)),
        } != {
            "required_payload_groups": 11,
            "required_images_per_payload": 5000,
            "overlap_reference_images": 500,
            "payload_retention": "keep_all",
            "target_queries": 0,
            "target_gradients": 0,
        }:
            raise RegistryError("Full-val source-generation policy is invalid")
        if generation.get("exact_overlap_payload_sha256_required") is not True:
            raise RegistryError("Full-val confirmation requires exact overlap hashes")
        storage = fullval.get("storage_preflight")
        if not isinstance(storage, Mapping) or {
            "estimate_from": storage.get("estimate_from"),
            "full_to_retained_multiplier": int(
                storage.get("full_to_retained_multiplier", -1)
            ),
            "additional_free_margin_gib": int(
                storage.get("additional_free_margin_gib", -1)
            ),
            "fail_before_payload_generation": storage.get(
                "fail_before_payload_generation"
            ),
        } != {
            "estimate_from": "accepted_mask_source_retained500_payload_bytes",
            "full_to_retained_multiplier": 10,
            "additional_free_margin_gib": 16,
            "fail_before_payload_generation": True,
        }:
            raise RegistryError("Full-val storage preflight policy is invalid")
        fullval_counts = fullval.get("formal_counts")
        if not isinstance(fullval_counts, Mapping) or {
            "clean_evaluations": int(
                fullval_counts.get("clean_evaluations", -1)
            ),
            "attack_evaluations": int(
                fullval_counts.get("attack_evaluations", -1)
            ),
            "records": int(fullval_counts.get("records", -1)),
            "attack_groups": int(fullval_counts.get("attack_groups", -1)),
            "payload_images": int(fullval_counts.get("payload_images", -1)),
            "retained_overlap_hashes": int(
                fullval_counts.get("retained_overlap_hashes", -1)
            ),
        } != {
            "clean_evaluations": 2,
            "attack_evaluations": 22,
            "records": 24,
            "attack_groups": 11,
            "payload_images": 55000,
            "retained_overlap_hashes": 5500,
        }:
            raise RegistryError("Full-val training-state formal counts are invalid")
        if not self.preprocessing_defense_order:
            raise RegistryError("No preprocessing-defense variants are configured")
        if len(set(self.preprocessing_defense_order)) != len(
            self.preprocessing_defense_order
        ):
            raise RegistryError(
                "preprocessing_defenses.yaml variant_order contains duplicates"
            )
        if set(self.preprocessing_defense_order) != set(
            self.preprocessing_defenses
        ):
            raise RegistryError(
                "Preprocessing-defense variant_order must cover every variant exactly once"
            )
        if self.preprocessing_defense_order[0] != "identity":
            raise RegistryError(
                "The preprocessing-defense identity control must be first"
            )
        allowed_defense_kinds = {
            "identity",
            "jpeg",
            "bit_depth",
            "gaussian",
            "median",
            "resize_roundtrip",
        }
        for defense_id in self.preprocessing_defense_order:
            defense = self.preprocessing_defenses[defense_id]
            if str(defense.get("kind")) not in allowed_defense_kinds:
                raise RegistryError(
                    "Preprocessing defense '{}' has an unsupported kind".format(
                        defense_id
                    )
                )
            if not str(defense.get("label", "")).strip():
                raise RegistryError(
                    "Preprocessing defense '{}' has no label".format(defense_id)
                )

        for dataset_id in ("coco", "voc"):
            protocol_id = "vcsf_selected_a23_{}_main".format(dataset_id)
            selected = self.protocols.get(protocol_id)
            if not isinstance(selected, dict):
                raise RegistryError("Selected A23 main protocol is missing: " + protocol_id)
            images = self.datasets[dataset_id].split("val").expected_images
            sources = self.source_ids()
            targets = self.target_ids()
            required = {
                "budget_profile": "compute_matched",
                "datasets": [dataset_id],
                "split": "val",
                "seed": 42,
                "selection_scope": "author_authorized_full_coco_val2017_retrospective",
                "selection_overlap_with_evaluation": dataset_id == "coco",
                "images": images,
                "sources": sources,
                "targets": "all",
                "methods": [],
                "isolated_method": "vcsf",
                "execution_mode": "selected_a23",
                "payload_retention": "keep_all_pending_independent_audit",
            }
            if any(selected.get(key) != value for key, value in required.items()):
                raise RegistryError("Selected A23 main preparation drifted: " + protocol_id)
            closed = (
                selected.get("status") == "preparation_only"
                and selected.get("execution_authorized") is False
                and selected.get("clean_reuse_admitted") is False
                and not any(key in selected for key in (
                    "accepted_clean_binding_sha256",
                    "accepted_clean_audit_sha256",
                    "formal_wave_admissions",
                ))
            )
            promoted = (
                selected.get("status") == "active"
                and selected.get("execution_authorized") is True
                and selected.get("clean_reuse_admitted") is True
                and selected.get("accepted_clean_binding_sha256")
                    == selected.get("candidate_clean_binding_sha256")
                and selected.get("accepted_clean_audit_sha256")
                    == selected.get("candidate_clean_audit_sha256")
                and isinstance(selected.get("formal_wave_admissions"), dict)
                and bool(selected["formal_wave_admissions"])
            )
            if not (closed or promoted):
                raise RegistryError("Selected A23 main preparation drifted: " + protocol_id)
            if promoted:
                for wave_key, entry in selected["formal_wave_admissions"].items():
                    if (
                        not isinstance(wave_key, str) or len(wave_key) != 64
                        or any(char not in "0123456789abcdef" for char in wave_key)
                        or not isinstance(entry, dict)
                        or set(entry) != {"path", "sha256"}
                        or not isinstance(entry["path"], str)
                        or not Path(entry["path"]).is_absolute()
                        or Path(entry["path"]).name != "receipt.json"
                        or not isinstance(entry["sha256"], str)
                        or len(entry["sha256"]) != 64
                        or any(char not in "0123456789abcdef" for char in entry["sha256"])
                    ):
                        raise RegistryError("Selected A23 wave admission registry is invalid")
            expected_execution = {
                "candidate_reused_clean_cells": len(targets),
                "attack_groups": len(sources),
                "attack_evaluations": len(sources) * len(targets),
                "logical_report_cells_if_clean_admitted": len(targets) * (len(sources) + 1),
                "blackbox_decision_cells": len(sources) * (len(targets) - 1),
                "source_matched_whitebox_cells": len(sources),
                "generated_images": len(sources) * images,
                "gradient_evaluations_per_image": self.budget_profiles[
                    "compute_matched"]["max_gradient_evaluations_per_image"],
                "failed_records": 0,
            }
            if selected.get("expected_execution") != expected_execution:
                raise RegistryError("Selected A23 main counts drifted: " + protocol_id)
            for field_name in ("parameters_sha256", "freeze_manifest_sha256",
                               "candidate_clean_binding_sha256",
                               "candidate_clean_audit_sha256"):
                value = selected.get(field_name)
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise RegistryError("Selected A23 main hash is invalid: " + field_name)
            freeze_path = (self.root / str(selected.get("freeze_manifest", ""))).resolve()
            if self.root not in freeze_path.parents or not freeze_path.is_file():
                raise RegistryError("Selected A23 freeze path is invalid")
            if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != selected[
                    "freeze_manifest_sha256"]:
                raise RegistryError("Selected A23 freeze bytes drifted")
            freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
            if (freeze.get("execution_mode") != "selected_a23"
                    or freeze.get("formal_execution_authorized") is not False
                    or freeze.get("parameters_sha256") != selected["parameters_sha256"]):
                raise RegistryError("Selected A23 freeze is not preparation-only")

        from .runners.vcsf_a10_main_contract import validate_protocol
        for dataset_id in ("coco", "voc"):
            if "vcsf_selected_a10_" + dataset_id + "_main" in self.protocols:
                try:
                    validate_protocol(self, dataset_id)
                except RuntimeError as error:
                    raise RegistryError(str(error)) from error

    def model(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as exc:
            raise RegistryError(
                "Unknown model '{}'. Choose one of: {}".format(
                    model_id, ", ".join(self.paper_order)
                )
            ) from exc

    def dataset(self, dataset_id: str) -> DatasetSpec:
        try:
            return self.datasets[dataset_id]
        except KeyError as exc:
            raise RegistryError(
                "Unknown dataset '{}'. Choose one of: {}".format(
                    dataset_id, ", ".join(self.datasets)
                )
            ) from exc

    def attack(self, attack_id: str) -> AttackSpec:
        try:
            return self.attacks[attack_id]
        except KeyError as exc:
            raise RegistryError(
                "Unknown attack '{}'. Choose one of: {}".format(
                    attack_id, ", ".join(self.attacks)
                )
            ) from exc

    def source_ids(self) -> List[str]:
        return [
            model_id
            for model_id in self.paper_order
            if self.models[model_id].source
        ]

    def target_ids(self, include_source: bool = True) -> List[str]:
        return [
            model_id
            for model_id in self.paper_order
            if include_source or not self.models[model_id].source
        ]

    def model_group(self, model_id: str) -> ModelGroupSpec:
        self.model(model_id)
        for group in self.paper_groups:
            if model_id in group.models:
                return group
        raise RegistryError("Model '{}' is not assigned to a paper group".format(model_id))

    def model_position(self, model_id: str) -> int:
        self.model(model_id)
        return self.paper_order.index(model_id) + 1

    def compatibility_status(self, attack_id: str, source_id: str) -> Dict[str, Any]:
        self.attack(attack_id)
        self.model(source_id)
        record = self.compatibility.get("methods", {}).get(attack_id, {})
        for status in COMPATIBILITY_STATUSES:
            if source_id in record.get(status, []):
                source_reasons = record.get("source_reasons", {})
                reasons = record.get("reasons", {})
                reason = source_reasons.get(
                    source_id,
                    reasons.get(status, record.get("reason", "")),
                )
                return {
                    "status": status,
                    "reason": str(reason),
                    "requires": list(record.get("requires", [])),
                }
        raise RegistryError(
            "Attack/source compatibility is not explicitly classified: {}/{}".format(
                attack_id, source_id
            )
        )

    def expand_ids(self, values: Iterable[str], universe: Iterable[str]) -> List[str]:
        values = list(values)
        universe = list(universe)
        if values == ["all"] or "all" in values:
            return universe
        unknown = sorted(set(values) - set(universe))
        if unknown:
            raise RegistryError("Unknown selection(s): {}".format(", ".join(unknown)))
        return values
