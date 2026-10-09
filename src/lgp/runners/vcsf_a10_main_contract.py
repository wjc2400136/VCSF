"""Declarative native-A10 main waves and fail-closed SHA-bound evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

from ..io import file_digest
from ..metrics import COCO_BBOX_METRICS

SCHEMA = "vcsf_a10_native_main_wave_v1"
AUDIT_CONTRACT = "docs/research/vcsf-a10-main-audit-contract.json"
ENTRY = "experiments/vcsf_a10_main.py"
AUDIT_ENTRY = "experiments/prepare_vcsf_a10_main_audit.py"


def require(condition, message):
    if not condition:
        raise RuntimeError("Native A10 main: " + message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path, expected=None):
    path = Path(path).absolute()
    require(path.resolve() == path and path.is_file(), "evidence must be a plain file")
    raw = path.read_bytes()
    if expected is not None:
        require(sha(expected) and hashlib.sha256(raw).hexdigest() == expected,
                "evidence SHA-256 mismatch")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
        parse_constant=lambda value: require(False, "non-finite JSON constant"))
    require(file_digest(path) == hashlib.sha256(raw).hexdigest(), "evidence changed during read")
    return value


def reference(value):
    require(type(value) is dict and set(value) == {"file", "sha256"}
            and type(value["file"]) is str and Path(value["file"]).is_absolute()
            and sha(value["sha256"]), "incomplete evidence reference")
    return read_json(value["file"], value["sha256"])


def no_acceptance():
    return {key: False for key in ("formal_result_eligible", "scientific_acceptance",
        "implementation_promoted", "result_reuse_accepted", "project_wide_device_release_accepted",
        "publication_accepted", "deletion_authorized")}


def validate_protocol(registry, dataset):
    name = "vcsf_selected_a10_" + dataset + "_main"
    spec = registry.protocols.get(name)
    require(type(spec) is dict and spec.get("execution_mode") == "native_a10_main"
        and spec.get("datasets") == [dataset] and spec.get("split") == "val"
        and spec.get("sources") == "all_whitebox" and spec.get("targets") == "all"
        and spec.get("methods") == ["vcsf"] and spec.get("parameters_source") == "configs/attacks/vcsf.yaml"
        and spec.get("parameters_sha256") == registry.attack("vcsf").metadata["parameters_sha256"]
        and spec.get("images") == registry.dataset(dataset).split("val").expected_images
        and spec.get("metrics") == list(COCO_BBOX_METRICS)
        and spec.get("device_counts") == [1, 2] and type(spec.get("seed")) is int
        and spec.get("seed_schedule") == "full_val_numeric_image_position"
        and spec.get("payload_retention") == "keep_all_pending_independent_audit"
        and spec.get("prediction_archive") == "gzip"
        and spec.get("generic_dispatch_forbidden") is True
        and spec.get("preparation_entrypoint") == ENTRY
        and spec.get("status") == "preparation_only"
        and spec.get("execution_authorized") is False
        and spec.get("result_reuse_accepted") is False
        and spec.get("audit_contract") == AUDIT_CONTRACT,
        "registered native protocol drift")
    sources, targets = registry.source_ids(), registry.target_ids()
    reuse = spec.get("candidate_reuse_sources")
    require(type(reuse) is list and reuse == [s for s in sources if s in reuse]
        and len(reuse) == len(set(reuse)) and (dataset == "coco" or not reuse),
        "reuse candidates are not a canonical source subset")
    require(spec.get("scientific_attack_cells") == len(sources) * len(targets)
        and spec.get("candidate_reuse_cells") == len(reuse) * len(targets)
        and spec.get("new_attack_cells") == (len(sources) - len(reuse)) * len(targets),
        "scientific partition omits or duplicates jobs")
    require(spec["budget_profile"] in registry.budget_profiles, "unknown cost regime")
    require(type(spec.get("configuration_choice_record")) is str
        and sha(spec.get("configuration_choice_sha256"))
        and file_digest(safe_child(registry.root, spec["configuration_choice_record"], True))
            == spec["configuration_choice_sha256"],
        "registered configuration choice provenance changed")
    return name, spec


def compile_plan(registry, dataset, devices, sources=None, max_images=None,
                 wave_id="planned-wave", inputs_reference=None):
    from ..attacks.vcsf_public import verify_public_identity
    name, spec = validate_protocol(registry, dataset)
    require(type(devices) is list and len(devices) in spec["device_counts"]
        and len(set(devices)) == len(devices)
        and all(type(d) is str and re.fullmatch(r"cuda:[0-9]+", d) for d in devices),
        "one or two distinct explicitly named devices are required")
    require(type(wave_id) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", wave_id),
        "wave ID must be a simple leaf name")
    available = [s for s in registry.source_ids() if s not in spec["candidate_reuse_sources"]]
    selected = list(available if sources is None else sources)
    require(selected and selected == [s for s in available if s in selected]
        and len(selected) == len(set(selected)), "wave is not a canonical genuinely-new source subset")
    require(len(selected) >= len(devices), "every selected device requires a complete nonempty lane")
    total_images = spec["images"]
    require(max_images is None or (type(max_images) is int and 1 <= max_images < total_images),
        "diagnostic image limit must be a strict positive subset, never full-split disguised")
    identity = verify_public_identity(registry.root, registry.attack("vcsf"))
    targets = registry.target_ids()
    jobs = [dict(job_id=i + 1, source=s, dataset=dataset, split=spec["split"],
        attack="vcsf", seed=spec["seed"], parameters_sha256=identity["parameters_sha256"],
        targets=list(targets), worker_slot=i % len(devices))
        for i, s in enumerate(selected)]
    require(inputs_reference is None or (type(inputs_reference) is dict
        and set(inputs_reference) == {"file", "sha256"}
        and sha(inputs_reference["sha256"]) and Path(inputs_reference["file"]).is_absolute()),
        "input package must be an explicit immutable reference")
    return dict(schema=SCHEMA, status="prepared_not_admitted", protocol=name,
        wave_id=wave_id, dataset=dataset, split=spec["split"], seed=spec["seed"],
        max_images=max_images, images_per_job=max_images or total_images,
        full_population_images=total_images, diagnostic_only=max_images is not None,
        scientific_sources=registry.source_ids(), targets=list(targets),
        scientific_attack_cells=spec["scientific_attack_cells"],
        candidate_reuse_sources=list(spec["candidate_reuse_sources"]),
        candidate_reuse_cells=spec["candidate_reuse_cells"],
        genuinely_new_sources=available, genuinely_new_cells=spec["new_attack_cells"],
        selected_sources=selected, jobs=jobs, wave_attack_cells=len(jobs) * len(targets),
        devices=list(devices), lanes=[[j["job_id"] for j in jobs if j["worker_slot"] == slot]
            for slot in range(len(devices))],
        budget_profile=spec["budget_profile"],
        budget_profile_sha256=canonical(registry.budget_profiles[spec["budget_profile"]]),
        parameters_sha256=identity["parameters_sha256"],
        numeric_source_sha256=identity["numeric_source_sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        source_freeze_sha256=identity["source_freeze_sha256"],
        protocols_sha256=file_digest(registry.root / "configs/experiments/protocols.yaml"),
        configuration_choice_sha256=spec["configuration_choice_sha256"],
        audit_contract_sha256=file_digest(registry.root / AUDIT_CONTRACT),
        input_package_reference=inputs_reference, execution_admitted=False,
        payload_retention="keep_all", prediction_archive="gzip",
        execution_model="independent_complete_source_jobs_no_DDP_no_source_ensemble",
        cost_scope="sampled_instrumented_execution_not_paired_performance",
        current_input_identity_status="NR" if inputs_reference is None else "bound_pending_admission",
        calibrated_whole_detector_BE=None, FLOPs=None, **no_acceptance())


def verify_plan(registry, plan):
    require(type(plan) is dict and plan.get("schema") == SCHEMA, "foreign plan")
    expected = compile_plan(registry, plan["dataset"], plan["devices"],
        plan["selected_sources"], plan["max_images"], plan["wave_id"], plan["input_package_reference"])
    require(canonical(plan) == canonical(expected), "plan does not match current registered source/jobs")
    return plan


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def verify_runtime():
    require(sys.platform == "linux" and sys.version_info[:3] == (3, 8, 20),
            "pinned server Python is required")
    require(Path(sys.executable).parent.name == "bin" and Path(sys.executable).parent.parent.name == "oda",
            "the fixed ODA interpreter is required")
    pins = {"torch": "2.0.0+cu118", "torchvision": "0.15.1+cu118", "mmcv": "2.0.1",
        "mmengine": "0.7.4", "mmdet": "3.0.0", "mmyolo": "0.6.0"}
    actual = {name: importlib.metadata.version(name) for name in pins}
    require(actual == pins, "pinned ODA packages changed")
    return dict(python="3.8.20", packages=actual)


def verify_input_shape(registry, plan, inputs):
    targets = registry.target_ids()
    require(type(inputs) is dict and inputs.get("schema") == "a10_main_current_inputs_v1"
        and inputs.get("dataset") == plan["dataset"] and inputs.get("split") == plan["split"],
        "foreign current input package")
    summary = inputs["current_clean_inputs"]
    require(summary["dataset"] == plan["dataset"] and summary["split"] == plan["split"]
        and summary["expected_images"] == plan["full_population_images"]
        and set(summary["target_checkpoint_sha256"]) == set(targets)
        and all(sha(x) for x in summary["target_checkpoint_sha256"].values()),
        "current clean population or checkpoint panel differs")
    rows = inputs["images"]
    require(type(rows) is list and len(rows) == plan["full_population_images"]
        and all(type(r) is dict and set(r) == {"image_id", "file_name", "sha256"}
            and type(r["image_id"]) is int and type(r["file_name"]) is str
            and not Path(r["file_name"]).is_absolute() and ".." not in Path(r["file_name"]).parts
            and sha(r["sha256"]) for r in rows),
        "current ordered image byte bindings are incomplete")
    ids = [row["image_id"] for row in rows]
    require(ids == sorted(set(ids)) and canonical(ids) == summary["ordered_image_ids_sha256"]
        and canonical([[r["image_id"], r["file_name"], r["sha256"]] for r in rows])
            == summary["ordered_image_files_sha256"], "input bytes or canonical image order differs")
    require(set(inputs["upstream_configs"]) == set(targets)
        and all(sha(row["sha256"]) and sha(row["normalized_effective_sha256"])
            for row in inputs["upstream_configs"].values()),
        "complete upstream and normalized effective target configs are required")
    require(type(inputs["installed_config_manifest"]) is dict and inputs["installed_config_manifest"]
        and all(sha(v) for v in inputs["installed_config_manifest"].values()),
        "transitive installed config identity is missing")
    require(set(inputs["checkpoint_readiness"]) == set(targets), "checkpoint readiness panel incomplete")
    for target, item in inputs["checkpoint_readiness"].items():
        require(item.get("dataset") == plan["dataset"] and item.get("model") == target
            and item.get("checkpoint_sha256") == summary["target_checkpoint_sha256"][target]
            and item.get("ready") is True and item.get("strict_loading_verified") is True,
            "checkpoint is not independently ready")
        if plan["dataset"] != "coco":
            require(item.get("training_manifest_verified") is True
                and item.get("fixed_final_epoch_verified") is True
                and item.get("training_image_inference_verified") is True
                and item.get("formal_split_selection_used") is False,
                "fine-tuned checkpoint provenance is incomplete")
    return inputs


def verify_current_inputs(registry, plan, inputs):
    from ..data.coco import CocoIndex
    from ..runtime_config import resolve_upstream_config, package_root
    from .lgp_corrected_main import _current_clean_inputs
    verify_input_shape(registry, plan, inputs)
    require(_current_clean_inputs(registry, plan["dataset"]) == inputs["current_clean_inputs"],
            "current complete image/annotation/16-weight bytes differ from accepted clean input")
    index = CocoIndex(registry.dataset(plan["dataset"]), plan["split"])
    actual = [(int(r["id"]), str(r["file_name"])) for r in index.images]
    require(actual == [(r["image_id"], r["file_name"]) for r in inputs["images"]],
            "current index differs from bound image records")
    frameworks = set()
    for target in registry.target_ids():
        model = registry.model(target)
        frameworks.add(model.framework)
        require(file_digest(resolve_upstream_config(model)) == inputs["upstream_configs"][target]["sha256"],
                "installed detector entry config changed")
    installed = {}
    for framework in sorted(frameworks):
        base = package_root(framework)
        for path in sorted((base / ".mim/configs").rglob("*.py")):
            require(path.is_file() and not path.is_symlink(), "linked installed config")
            installed[framework + "/" + path.relative_to(base).as_posix()] = file_digest(path)
    require(installed == inputs["installed_config_manifest"], "transitive installed configs changed")
    return inputs


def verify_promotion_scope(promotion, plan, plan_ref):
    formal = plan["max_images"] is None
    schema = ("native_a10_exact_implementation_formal_wave_promotion_v1" if formal
        else "native_a10_exact_implementation_bounded_promotion_v1")
    require(plan.get("diagnostic_only") is not formal
        and promotion.get("schema") == schema
        and promotion.get("formal_execution_authorized") is formal
        and promotion.get("formal_native_executor_qualified") is formal,
        "implementation promotion is not qualified for this formal/diagnostic regime")
    permitted = promotion.get("permitted_wave_plan_references")
    require(type(permitted) is list and permitted
        and all(type(item) is dict and set(item) == {"file", "sha256"}
            and type(item["file"]) is str and Path(item["file"]).is_absolute()
            and sha(item["sha256"]) for item in permitted)
        and len({canonical(item) for item in permitted}) == len(permitted)
        and plan_ref in permitted, "implementation promotion excludes this exact wave plan")
    scope = promotion.get("execution_scope", {})
    require(type(scope) is dict and "max_images" in scope
        and type(scope["max_images"]) is type(plan["max_images"])
        and scope["max_images"] == plan["max_images"]
        and scope.get("keep_all") is True and scope.get("automatic_resume") is False
        and type(scope.get("canonical_targets_per_source")) is int
        and scope["canonical_targets_per_source"] == len(plan["targets"]),
        "implementation promotion image/target/retention scope differs")
    datasets, sources, modes = (scope.get(key) for key in ("datasets", "sources", "modes"))
    require(type(datasets) is list and datasets
        and all(type(item) is str and item in ("coco", "voc") for item in datasets)
        and len(set(datasets)) == len(datasets)
        and plan["dataset"] in datasets
        and type(sources) is list and sources
        and sources == [item for item in plan["scientific_sources"] if item in sources]
        and len(set(sources)) == len(sources)
        and all(item in sources for item in plan["selected_sources"])
        and type(modes) is list and modes
        and all(type(item) is int and item in (1, 2) for item in modes)
        and len(set(modes)) == len(modes)
        and len(plan["devices"]) in modes,
        "implementation promotion dataset/source/device scope differs")


def verify_admission(registry, plan, plan_ref, admission_ref, current=False):
    verify_plan(registry, plan)
    require(plan["input_package_reference"] is not None, "input binding is still NR")
    require(reference(plan_ref) == plan, "plan reference differs")
    admission = reference(admission_ref)
    expected = dict(schema="vcsf_a10_native_main_wave_admission_v1",
        status="independently_admitted_native_a10_main_wave",
        wave_plan_reference=plan_ref, execution_admitted=True, wave_id=plan["wave_id"],
        dataset=plan["dataset"], max_images=plan["max_images"],
        selected_sources=plan["selected_sources"], jobs=plan["jobs"], lanes=plan["lanes"],
        devices=plan["devices"], source_manifest_sha256=plan["source_manifest_sha256"],
        source_freeze_sha256=plan["source_freeze_sha256"], parameters_sha256=plan["parameters_sha256"],
        input_package_reference=plan["input_package_reference"], audit_contract_sha256=plan["audit_contract_sha256"],
        payload_retention="keep_all", prediction_archive="gzip", automatic_resume=False,
        formal_execution_authorized=not plan["diagnostic_only"],
        **no_acceptance())
    require(all(type(admission.get(k)) is type(v) and canonical(admission[k]) == canonical(v)
        for k, v in expected.items()),
            "exact per-wave admission is absent or mismatched")
    require(type(admission.get("physical_gpu_uuids")) is list
        and len(admission["physical_gpu_uuids"]) == len(plan["devices"])
        and len(set(admission["physical_gpu_uuids"])) == len(plan["devices"])
        and all(type(u) is str and u.startswith("GPU-") for u in admission["physical_gpu_uuids"]),
        "physical device binding is incomplete")
    now = datetime.now(timezone.utc)
    start = datetime.fromisoformat(admission["not_before_utc"])
    end = datetime.fromisoformat(admission["expires_utc"])
    require(start.tzinfo is not None and end.tzinfo is not None and start <= now < end,
            "wave admission is outside its validity interval")
    limits = admission["limits"]
    for field in ("max_elapsed_seconds", "startup_timeout_seconds", "poll_interval_seconds",
            "maximum_sample_gap_seconds", "teardown_timeout_seconds"):
        require(positive(limits.get(field)), "missing positive admitted resource/time bound")
    for field in ("max_output_bytes", "minimum_free_disk_bytes", "minimum_free_vram_bytes",
            "reserve_disk_bytes"):
        require(type(limits.get(field)) is int and limits[field] > 0,
                "missing positive integer admitted byte bound")
    require(limits["minimum_free_disk_bytes"] >= limits["max_output_bytes"] + limits["reserve_disk_bytes"],
            "keep-all output bound plus emergency reserve exceeds admitted initial floor")
    require(limits["poll_interval_seconds"] <= 1
        and limits["maximum_sample_gap_seconds"] >= 2 * limits["poll_interval_seconds"],
        "ownership observation cadence cannot support its own bound")
    references = admission["evidence"]
    require(set(references) == {"engineering_bridge", "bounded_diagnostic", "selection",
        "method_promotion", "clean", "environment", "storage", "saved_result_auditor"},
        "per-wave admission evidence family is incomplete")
    evidence = {key: reference(value) for key, value in references.items()}
    bridge = evidence["engineering_bridge"]
    require(bridge.get("accepted_engineering_bridge") is True
        and bridge.get("no_material_findings") is True
        and bridge.get("new_source_manifest_sha256") == plan["source_manifest_sha256"]
        and bridge.get("numeric_source_sha256") == plan["numeric_source_sha256"]
        and bridge.get("parameters_sha256") == plan["parameters_sha256"]
        and bridge.get("historical_full_tree_acceptance_inherited") is False,
        "current engineering source tree lacks a reviewed exact numerical bridge")
    diagnostic = evidence["bounded_diagnostic"]
    require(diagnostic.get("bounded_six_source_public_factory_diagnostic_accepted") is True
        and diagnostic.get("parameters_sha256") == plan["parameters_sha256"]
        and diagnostic.get("scope", {}).get("source_order") == plan["scientific_sources"]
        and diagnostic.get("formal_result_eligible") is False
        and bridge.get("bounded_diagnostic_decision_sha256") == references["bounded_diagnostic"]["sha256"],
        "bounded diagnostic identity is not bridged to current engineering sources")
    selection = evidence["selection"]
    verify_configuration_choice(selection, plan, references["selection"])
    promotion = evidence["method_promotion"]
    require(promotion.get("implementation_promoted") is True
        and promotion.get("parameters_sha256") == plan["parameters_sha256"]
        and promotion.get("source_manifest_sha256") == plan["source_manifest_sha256"],
        "exact native implementation promotion is still pending")
    verify_promotion_scope(promotion, plan, plan_ref)
    inputs = verify_input_shape(registry, plan, reference(plan["input_package_reference"]))
    clean = evidence["clean"]
    require(clean.get("method_independent") is True
        and clean.get("clean_input_reuse_accepted") is True and clean.get("dataset") == plan["dataset"]
        and clean.get("accepted_current_clean_inputs") == inputs["current_clean_inputs"]
        and clean.get("targets") == plan["targets"] and clean.get("clean_cells") == len(plan["targets"])
        and clean.get("all_twelve_metrics_and_predictions_verified") is True,
        "method-independent current full-panel clean acceptance is pending")
    environment = evidence["environment"]
    require(environment.get("pinned_oda_verified") is True
        and environment.get("deep_data_verified") is True
        and environment.get("source_manifest_sha256") == plan["source_manifest_sha256"]
        and environment.get("current_clean_inputs") == inputs["current_clean_inputs"],
        "current ODA/deep-data qualification is missing")
    storage = evidence["storage"]
    require(storage.get("wave_plan_reference") == plan_ref
        and type(storage.get("registered_complete_source_jobs")) is int
        and storage["registered_complete_source_jobs"] == len(plan["jobs"])
        and storage.get("keep_all") is True
        and storage.get("estimated_max_output_bytes") == limits["max_output_bytes"]
        and storage.get("minimum_free_disk_bytes") == limits["minimum_free_disk_bytes"]
        and storage.get("reserve_disk_bytes") == limits["reserve_disk_bytes"]
        and storage.get("retention_or_deletion_assumed") is False,
        "storage does not cover this exact wave without cleanup")
    auditor = evidence["saved_result_auditor"]
    require(auditor.get("saved_result_auditor_qualified") is True
        and auditor.get("audit_contract_sha256") == plan["audit_contract_sha256"]
        and auditor.get("parameters_sha256") == plan["parameters_sha256"],
        "independent saved-result auditor qualification is pending")
    if current:
        require(verify_runtime() == environment["runtime"], "admitted ODA runtime changed")
        verify_current_inputs(registry, plan, inputs)
    return admission, inputs


def verify_configuration_choice(selection, plan, selection_reference):
    require(selection.get("parameters_sha256") == plan["parameters_sha256"]
        and selection.get("selected_configuration") == "A10"
        and all(selection.get(field) is True for field in ("configuration_choice_recorded",
            "selected_configuration_independently_reviewed",
            "selection_recommendation_accepted_for_declared_scope")),
        "configuration choice is not the independently reviewed recorded A10 decision")
    numeric = selection.get("selected_numerical_implementation_identity", {})
    require(numeric.get("numeric_source_sha256") == plan["numeric_source_sha256"]
        and numeric.get("numeric_source_map_sha256") == canonical(plan["numeric_source_sha256"])
        and canonical(selection.get("chosen_parameters_for_qualification")) == plan["parameters_sha256"],
        "configuration choice complete parameters or numerical source map differ")
    require(selection_reference["sha256"] == plan["configuration_choice_sha256"],
            "configuration choice bytes differ from registered original accepted provenance")


def safe_child(root, relative, must_exist=False):
    root = Path(root).absolute()
    require(type(relative) is str and relative and not Path(relative).is_absolute()
        and not any(part in (".", "..") for part in Path(relative).parts), "unsafe relative artifact path")
    path = root / relative
    require(path.resolve() == path and root in path.parents, "linked or escaped output path")
    require(not must_exist or path.is_file(), "required output artifact is missing")
    return path


def fresh_destination(root, plan):
    category = "diagnostics" if plan["diagnostic_only"] else "experiments"
    path = Path(root) / "outputs" / category / plan["protocol"] / plan["wave_id"]
    require(path.resolve() == path and not path.exists(), "immutable wave leaf already exists or is linked")
    return path


def contract_plan(registry, dataset):
    name, spec = validate_protocol(registry, dataset)
    value = read_json(registry.root / AUDIT_CONTRACT)
    return dict(protocol=name, scientific_sources=registry.source_ids(), targets=registry.target_ids(),
        metrics=list(COCO_BBOX_METRICS), expected_images=spec["images"], contract=value,
        audit_contract_sha256=file_digest(registry.root / AUDIT_CONTRACT),
        current_outputs="NR", actual_audit_executed=False, model_calls=0, GPU_calls=0,
        AP_replays=0, **no_acceptance())
