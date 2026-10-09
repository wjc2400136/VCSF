from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ..io import atomic_json, file_digest
from ..paths import project_root
from ..registry import Registry
from ..reporting.all_methods import write_artifact_manifest
from ..reporting.scope import SCOPE_FIELDS, escape_scope, scope_csv, scope_from_records, scope_label as scope_description
from .evaluate import run_evaluation
from .formal_parallel import validate_available_cuda_devices
from .preprocessing_defense import (
    _attack_input,
    _clean_input,
    verify_source_evidence,
)
from .training_state_pair import qualify_checkpoint_pair


PROTOCOL_ID = "coco_retained500_victim_training_state_transfer"
STATE_ORDER = ["standard_control", "adversarial_training"]
METRICS = (
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_small",
    "bbox_mAP_medium",
    "bbox_mAP_large",
    "bbox_AR_1",
    "bbox_AR_10",
    "bbox_AR_100",
    "bbox_AR_small",
    "bbox_AR_medium",
    "bbox_AR_large",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve_inside(root: Path, value: Any) -> Path:
    path = Path(str(value))
    path = path if path.is_absolute() else root / path
    resolved = path.resolve()
    if resolved != root.resolve() and root.resolve() not in resolved.parents:
        raise RuntimeError("Path escapes repository: {}".format(value))
    return resolved


def _git_provenance(root: Path) -> Dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "git_head": head,
        "worktree_dirty": status is None or bool(status),
        "captured_at_utc": _utc_now(),
    }


def build_training_state_plan(
    registry: Registry,
    method_override: Optional[Sequence[str]] = None,
    state_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    protocol = registry.protocols[PROTOCOL_ID]
    if str(protocol.get("status", "active")) != "active":
        raise RuntimeError(
            "The retained-500 training-state attack protocol is historical "
            "after the SFIM-B/NAA fidelity revision and cannot be regenerated"
        )
    spec = registry.training_state_transfer
    methods = list(method_override or spec["methods"])
    states = list(state_override or spec["victim_states"])
    unknown_methods = [method for method in methods if method not in spec["methods"]]
    if unknown_methods:
        raise ValueError("Unknown training-state methods: {}".format(unknown_methods))
    if len(set(methods)) != len(methods):
        raise ValueError("Duplicate training-state methods")
    if any(state not in STATE_ORDER for state in states) or len(set(states)) != len(states):
        raise ValueError("Invalid or duplicate victim states")
    selected_images = 500 if max_images is None else int(max_images)
    if selected_images <= 0 or selected_images > 500:
        raise ValueError("max_images must lie in [1, 500]")
    jobs = []
    for state in states:
        jobs.append(
            {
                "record_type": "clean",
                "source": "clean",
                "attack": "clean",
                "training_state": state,
                "target": str(spec["victim_model"]),
                "status": "ready",
            }
        )
        for method in methods:
            jobs.append(
                {
                    "record_type": "attacked",
                    "source": str(spec["source"]),
                    "attack": method,
                    "training_state": state,
                    "target": str(spec["victim_model"]),
                    "status": "ready",
                }
            )
    full = methods == list(spec["methods"]) and states == STATE_ORDER
    return {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "dataset": "coco",
        "split": "val",
        "source": str(spec["source"]),
        "victim_model": str(spec["victim_model"]),
        "methods": methods,
        "states": states,
        "metrics": list(METRICS),
        "retained_images": selected_images,
        "max_images": max_images,
        "full_registered_selection": full,
        "formal_requested": max_images is None and full,
        "threat_model": str(spec["threat_model"]),
        "full_coco_val_claim": False,
        "certified_robustness_claim": False,
        "counts": {
            "clean_evaluations": len(states),
            "attack_evaluations": len(states) * len(methods),
            "attack_groups": len(methods),
            "records": len(jobs),
        },
        "jobs": jobs,
    }


def _write_plan_csv(path: Path, jobs: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "record_type",
        "source",
        "attack",
        "training_state",
        "target",
        "status",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({field: job.get(field) for field in fields})


def _reference_view(
    view: Path,
    annotation: Mapping[str, Any],
    image_root: Path,
    source: str,
    attack: str,
) -> Path:
    view.mkdir(parents=True, exist_ok=False)
    atomic_json(view / "annotations.json", dict(annotation))
    image_root_relative = os.path.relpath(str(image_root.resolve()), str(view.resolve()))
    atomic_json(
        view / "run.json",
        {
            "schema_version": 1,
            "status": "complete",
            "dataset": "coco",
            "split": "val",
            "source": source,
            "attack": attack,
            "annotation": "annotations.json",
            "image_root": image_root_relative.replace("\\", "/"),
            "full_payload_available": True,
            "reference_only": True,
            "threat_model": "zero_query_oblivious_cross_model_transfer",
        },
    )
    return view


def _worker(
    root_value: str,
    run_dir_value: str,
    state: str,
    device: str,
    checkpoint_value: str,
    source_run_value: str,
    source: str,
    methods: Sequence[str],
    image_ids: Sequence[int],
) -> None:
    root = Path(root_value)
    run_dir = Path(run_dir_value)
    state_root = run_dir / "workers" / state
    try:
        if not device.startswith("cuda:"):
            raise ValueError("Evaluation worker requires explicit CUDA device")
        import torch

        torch.cuda.set_device(int(device.split(":", 1)[1]))
        registry = Registry(root)
        checkpoint = Path(checkpoint_value)
        source_run = Path(source_run_value)
        jobs = [("clean", "clean")] + [("attacked", method) for method in methods]
        for index, (record_type, method) in enumerate(jobs, start=1):
            atomic_json(
                state_root / "worker_state.json",
                {
                    "schema_version": 1,
                    "status": "running",
                    "training_state": state,
                    "record_type": record_type,
                    "source": "clean" if record_type == "clean" else source,
                    "attack": method,
                    "job_index": index,
                    "job_total": len(jobs),
                    "images": len(image_ids),
                    "updated_at_utc": _utc_now(),
                },
            )
            view = state_root / "views" / record_type / method
            if record_type == "clean":
                annotation, image_root = _clean_input(registry, image_ids)
                view = _reference_view(view, annotation, image_root, "clean", "clean")
            else:
                annotation, image_root = _attack_input(
                    registry, source_run, source, method, image_ids
                )
                view = _reference_view(view, annotation, image_root, source, method)
            completed = run_evaluation(
                registry,
                "coco",
                str(registry.training_state_transfer["victim_model"]),
                split="val",
                adversarial_run=view,
                output_dir=state_root / "evaluations" / record_type / method,
                max_images=None,
                device=device,
                download_weights=False,
                keep_going=False,
                prediction_archive="gzip",
                target_checkpoint=checkpoint,
            )
            record = json.loads((completed / "metrics.json").read_text(encoding="utf-8"))
            if record.get("status") != "complete":
                raise RuntimeError("Victim-state evaluation did not complete")
            if int(record.get("images", -1)) != len(image_ids):
                raise RuntimeError("Victim-state evaluation image count mismatch")
            if record.get("evaluated_image_ids_match_expected") is not True:
                raise RuntimeError("Victim-state evaluation IDs mismatch")
            record.update(
                record_type=record_type,
                training_state=state,
                victim_model=str(registry.training_state_transfer["victim_model"]),
                threat_model="zero_query_oblivious_cross_model_transfer",
                target_queries_during_attack_generation=0,
                target_gradients_during_attack_generation=0,
            )
            atomic_json(
                state_root / "completed" / "{:02d}_{}.json".format(index, method),
                record,
            )
        atomic_json(
            state_root / "worker_state.json",
            {
                "schema_version": 1,
                "status": "complete",
                "training_state": state,
                "jobs_completed": len(jobs),
                "updated_at_utc": _utc_now(),
            },
        )
    except BaseException as exc:
        atomic_json(
            state_root / "worker_state.json",
            {
                "schema_version": 1,
                "status": "failed",
                "training_state": state,
                "reason": "{}: {}".format(type(exc).__name__, exc),
                "updated_at_utc": _utc_now(),
            },
        )
        raise


def _collect_records(run_dir: Path, plan: Mapping[str, Any]) -> List[Dict[str, Any]]:
    by_key = {}
    for state in plan["states"]:
        completed = run_dir / "workers" / state / "completed"
        if not completed.is_dir():
            continue
        for path in completed.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            key = (
                str(record["record_type"]),
                str(record["training_state"]),
                str(record["attack"]),
            )
            if key in by_key:
                raise RuntimeError("Duplicate victim-state record key")
            by_key[key] = record
    ordered = []
    for job in plan["jobs"]:
        key = (
            str(job["record_type"]),
            str(job["training_state"]),
            str(job["attack"]),
        )
        if key in by_key:
            ordered.append(by_key[key])
    return ordered


def _metric(record: Mapping[str, Any], name: str) -> Optional[float]:
    value = record.get("metrics", {}).get(name)
    return (float(value) if type(value) in (int, float)
            and math.isfinite(value) and 0 <= value <= 1 else None)


def training_state_rows(
    records: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    metric: str,
    include_failure_markers: bool = False,
) -> List[Dict[str, Any]]:
    lookup = {
        (str(record["record_type"]), str(record["training_state"]), str(record["attack"])): record
        for record in records
        if record.get("status") == "complete"
        or (include_failure_markers and record.get("status") == "failed")
    }
    def value(record):
        return "ERR" if record.get("status") == "failed" else _metric(record, metric)

    def difference(adversarial, standard):
        if adversarial == "ERR" or standard == "ERR":
            return "ERR"
        return adversarial - standard if adversarial is not None and standard is not None else None

    def retention(attacked, benign):
        if attacked == "ERR" or benign == "ERR":
            return "ERR"
        return attacked / benign if attacked is not None and benign else None

    clean = {
        state: value(lookup.get(("clean", state, "clean"), {}))
        for state in STATE_ORDER
    }
    rows = []
    for method in methods:
        standard = value(lookup.get(("attacked", "standard_control", method), {}))
        adversarial = value(lookup.get(("attacked", "adversarial_training", method), {}))
        rows.append(
            {
                "attack": method,
                "standard_clean": clean["standard_control"],
                "adversarial_clean": clean["adversarial_training"],
                "standard_attacked": standard,
                "adversarial_attacked": adversarial,
                "attacked_delta_at_minus_standard": difference(adversarial, standard),
                "standard_retention": retention(standard, clean["standard_control"]),
                "adversarial_retention": retention(adversarial, clean["adversarial_training"]),
            }
        )
    return rows


def _format(value: Any) -> str:
    if value in ("ERR", "NR"):
        return value
    return "NR" if value is None else "{:.4f}".format(float(value))


def write_training_state_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output: Path,
    methods: Sequence[str],
    *,
    scope_label: str = "retained-500",
    full_coco_val_claim: bool = False,
    include_failure_markers: bool = False,
    report_scope: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, Path] = {}
    context = report_scope or {}
    scope_records = []
    for record in records:
        row = dict(record)
        projection = row.get("projection_source_evaluation") or {}
        if (row.get("scope") == "retained500" and row.get("status") == "complete"
                and row.get("evaluated_image_ids_match_expected") is True
                and row.get("evaluated_image_ids_sha256") == row.get("expected_image_ids_sha256")
                and isinstance(projection, dict) and projection.get("scope") == "fullval"
                and row.get("projection_model_calls") == 0):
            row["evaluated_images"] = row.get("images")
        scope_records.append(row)
    presentation = scope_from_records(
        scope_records, dataset=context.get("dataset") or "coco", split=context.get("split"),
        max_images=context.get("max_images"),
        diagnostic_only=context.get("diagnostic_only", False if full_coco_val_claim else None))
    sample_text = presentation["sample_n"] if presentation["sample_n"] is not None else "unknown"
    population = scope_description(presentation) or "actual dataset={}; split={}; sample N={}".format(
        presentation["dataset"] or "unknown", presentation["split"] or "unknown", sample_text)
    if presentation["diagnostic_only"] is True or presentation["sample_n"] != 5000:
        full_coco_val_claim = False
    recorded_scopes = {record.get("scope") for record in records}
    label_scope = (next(iter(recorded_scopes)) if recorded_scopes in (
        {"fullval"}, {"retained500"}) else
        ("fullval" if full_coco_val_claim else "retained500"))
    for metric in METRICS:
        rows = training_state_rows(records, methods, metric, include_failure_markers)
        csv_path = output / "training_state_{}.csv".format(metric)
        fields = list(rows[0]) if rows else [
            "attack",
            "standard_clean",
            "adversarial_clean",
            "standard_attacked",
            "adversarial_attacked",
            "attacked_delta_at_minus_standard",
            "standard_retention",
            "adversarial_retention",
        ]
        fields += list(SCOPE_FIELDS) + ["selection_scope"]
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(
                {key: value if key == "attack" else _format(value) for key, value in row.items()},
                **scope_csv(presentation), selection_scope=scope_label) for row in rows)
        tex_path = output / "training_state_{}.tex".format(metric)
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{COCO "
            + scope_label.replace("_", r"\_")
            + r" zero-query transfer under paired victim training states ("
            + metric.replace("_", r"\_")
            + r"). " + escape_scope(population) + r".}",
            r"\label{tab:training_state_"
            + label_scope + "_"
            + metric
            + "}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            r"Attack & STD clean & AT clean & STD adv. & AT adv. & $\Delta$(AT$-$STD) \\",
            r"\midrule",
        ]
        for row in rows:
            label = registry.attack(str(row["attack"])).display_name
            lines.append(
                "{} & {} & {} & {} & {} & {} \\\\".format(
                    label,
                    _format(row["standard_clean"]),
                    _format(row["adversarial_clean"]),
                    _format(row["standard_attacked"]),
                    _format(row["adversarial_attacked"]),
                    _format(row["attacked_delta_at_minus_standard"]),
                )
            )
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        artifacts[csv_path.name] = csv_path
        artifacts[tex_path.name] = tex_path
    readme_en = output / "README.md"
    readme_zh = output / "README.zh-CN.md"
    readme_en.write_text(
        "# Victim training-state transfer results\n\n"
        "[English](README.md) | [简体中文](README.zh-CN.md)\n\n"
        "Values are generated from `records.json`. The source is held-out "
        "Mask R-CNN Swin-T; the paired victim is Faster R-CNN. Lower attacked "
        "AP means a stronger transfer attack. Positive AT-STD delta means the "
        "adversarially trained victim retained more AP. This report covers "
        "COCO {} and {} a full COCO val2017 claim. It is empirical rather "
        "than certified robustness.\n".format(
            scope_label,
            "is" if full_coco_val_claim else "is not",
        ),
        encoding="utf-8",
    )
    readme_zh.write_text(
        "# 受害模型训练状态迁移结果\n\n"
        "[English](README.md) | [简体中文](README.zh-CN.md)\n\n"
        "所有数值均由 `records.json` 自动生成。固定源为与受害架构不同的 "
        "Mask R-CNN Swin-T，严格配对受害模型为 Faster R-CNN。攻击后 AP 越低表示"
        "迁移攻击越强；AT-STD 为正表示对抗训练受害模型保留了更多 AP。该报告覆盖 "
        "COCO {}，{}完整 COCO val2017 结论，但仍是经验鲁棒性而非认证鲁棒性。\n".format(
            scope_label,
            "属于" if full_coco_val_claim else "不属于",
        ),
        encoding="utf-8",
    )
    artifacts[readme_en.name] = readme_en
    artifacts[readme_zh.name] = readme_zh
    return artifacts


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (
        project_root()
        / "outputs"
        / "experiments"
        / "coco_training_state_transfer"
        / stamp
    )


def run_training_state_transfer(
    registry: Registry,
    execute: bool = True,
    output_dir: Optional[Path] = None,
    source_acceptance: Optional[Path] = None,
    pair_manifest: Optional[Path] = None,
    method_override: Optional[Sequence[str]] = None,
    state_override: Optional[Sequence[str]] = None,
    max_images: Optional[int] = None,
    devices: Optional[Sequence[str]] = None,
) -> Path:
    plan = build_training_state_plan(
        registry,
        method_override=method_override,
        state_override=state_override,
        max_images=max_images,
    )
    run_dir = (output_dir or _default_output()).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError("Refusing to overwrite non-empty experiment")
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "plan.json", plan)
    _write_plan_csv(run_dir / "plan.csv", plan["jobs"])
    atomic_json(run_dir / "records.json", [])
    qualification = qualify_checkpoint_pair(registry, pair_manifest)
    atomic_json(run_dir / "checkpoint_pair_qualification.json", qualification)
    if not execute:
        write_training_state_reports(
            [], registry, run_dir / "reports" / "coco", plan["methods"]
        )
        atomic_json(
            run_dir / "summary.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL_ID,
                "status": "planned",
                "jobs": plan["counts"]["records"],
                "records": 0,
                "checkpoint_pair_qualified": bool(
                    qualification["strict_training_controlled_pair_eligible"]
                ),
                "retained500_training_state_benchmark_eligible": False,
                "full_coco_val_claim": False,
            },
        )
        return run_dir
    formal = bool(plan["formal_requested"])
    required_key = (
        "strict_training_controlled_pair_eligible"
        if formal
        else "diagnostic_pair_eligible"
    )
    if qualification.get(required_key) is not True:
        raise RuntimeError("Checkpoint pair failed the required qualification tier")
    provenance = _git_provenance(registry.root)
    atomic_json(run_dir / "provenance.json", provenance)
    if formal and (provenance["worktree_dirty"] or provenance["git_head"] is None):
        raise RuntimeError("Formal training-state evaluation requires clean Git provenance")
    selected_devices = list(devices or ["cuda:0", "cuda:1"])
    if len(selected_devices) != len(plan["states"]):
        raise ValueError("Provide one device per selected victim state")
    validate_available_cuda_devices(selected_devices)
    source_evidence = verify_source_evidence(
        registry,
        acceptance_path=source_acceptance,
        deep=True,
        max_images=max_images,
    )
    atomic_json(run_dir / "source_evidence.json", source_evidence)
    selected_count = 500 if max_images is None else int(max_images)
    image_ids = list(source_evidence["retained_image_ids"][:selected_count])
    if _canonical_digest(image_ids) != (
        source_evidence["retained_image_ids_sha256"]
        if max_images is None
        else _canonical_digest(image_ids)
    ):
        raise RuntimeError("Retained image-ID digest mismatch")
    pair_path = (
        pair_manifest.resolve()
        if pair_manifest is not None
        else _resolve_inside(
            registry.root,
            registry.training_state_transfer["artifacts"]["pair_manifest"],
        )
    )
    pair = json.loads(pair_path.read_text(encoding="utf-8"))
    manifests = {
        state: json.loads(
            _resolve_inside(registry.root, pair["manifests"][state]).read_text(
                encoding="utf-8"
            )
        )
        for state in plan["states"]
    }
    checkpoints = {
        state: _resolve_inside(registry.root, manifests[state]["checkpoint"])
        for state in plan["states"]
    }
    source_run = registry.root / source_evidence["source_run"]
    context = mp.get_context("spawn")
    processes = {}
    for state, device in zip(plan["states"], selected_devices):
        worker_root = run_dir / "workers" / state
        worker_root.mkdir(parents=True, exist_ok=False)
        process = context.Process(
            target=_worker,
            args=(
                str(registry.root),
                str(run_dir),
                state,
                device,
                str(checkpoints[state]),
                str(source_run),
                str(plan["source"]),
                list(plan["methods"]),
                image_ids,
            ),
            name="training-state-eval-{}".format(state),
        )
        process.start()
        processes[state] = process
    records: List[Dict[str, Any]] = []
    while any(process.is_alive() for process in processes.values()):
        records = _collect_records(run_dir, plan)
        atomic_json(run_dir / "records.json", records)
        workers = []
        for state, process in processes.items():
            state_path = run_dir / "workers" / state / "worker_state.json"
            current = (
                json.loads(state_path.read_text(encoding="utf-8"))
                if state_path.is_file()
                else {"status": "starting", "training_state": state}
            )
            current["pid"] = process.pid
            current["process_alive"] = process.is_alive()
            workers.append(current)
        attacked = [r for r in records if r.get("record_type") == "attacked"]
        clean = [r for r in records if r.get("record_type") == "clean"]
        completed_groups = len(
            {
                r["attack"]
                for r in attacked
                if sum(
                    1
                    for item in attacked
                    if item["attack"] == r["attack"]
                )
                == len(plan["states"])
            }
        )
        atomic_json(
            run_dir / "execution_state.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL_ID,
                "status": "running",
                "phase": "evaluating_paired_victim_states",
                "clean_evaluations_completed": len(clean),
                "clean_evaluations_total": plan["counts"]["clean_evaluations"],
                "attack_groups_completed": completed_groups,
                "attack_groups_total": plan["counts"]["attack_groups"],
                "attack_evaluations_completed": len(attacked),
                "attack_evaluations_total": plan["counts"]["attack_evaluations"],
                "failed_records": 0,
                "workers": workers,
                "updated_at_utc": _utc_now(),
            },
        )
        time.sleep(15)
    for process in processes.values():
        process.join()
    failures = {
        state: process.exitcode
        for state, process in processes.items()
        if process.exitcode != 0
    }
    records = _collect_records(run_dir, plan)
    atomic_json(run_dir / "records.json", records)
    if failures or len(records) != plan["counts"]["records"]:
        reason = "worker failures={}, records={}/{}".format(
            failures, len(records), plan["counts"]["records"]
        )
        atomic_json(
            run_dir / "execution_state.json",
            {
                "schema_version": 1,
                "protocol": PROTOCOL_ID,
                "status": "failed",
                "phase": "failed",
                "reason": reason,
                "failed_records": len(failures),
                "updated_at_utc": _utc_now(),
            },
        )
        raise RuntimeError(reason)
    reports = write_training_state_reports(
        records, registry, run_dir / "reports" / "coco", plan["methods"]
    )
    exact = (
        formal
        and len(records) == 24
        and qualification["strict_training_controlled_pair_eligible"] is True
        and source_evidence["deep_payload_hash_verification"] is True
        and int(source_evidence["verified_images_per_payload"]) == 500
        and provenance["worktree_dirty"] is False
        and provenance["git_head"] is not None
    )
    acceptance = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "status": "accepted" if exact else "diagnostic_complete",
        "accepted_at_utc": _utc_now(),
        "retained500_training_state_benchmark_eligible": bool(exact),
        "strict_training_controlled_pair": bool(
            qualification["strict_training_controlled_pair_eligible"]
        ),
        "zero_query_transfer_claim": True,
        "full_coco_val_claim": False,
        "certified_robustness_claim": False,
        "counts": {
            **plan["counts"],
            "completed_records": len(records),
            "failed_records": 0,
            "retained_images": selected_count,
        },
        "checkpoint_sha256": qualification.get("checkpoint_sha256"),
        "retained_image_ids_sha256": source_evidence[
            "retained_image_ids_sha256"
        ],
        "evidence_sha256": {
            "plan.json": file_digest(run_dir / "plan.json"),
            "records.json": file_digest(run_dir / "records.json"),
            "source_evidence.json": file_digest(run_dir / "source_evidence.json"),
            "checkpoint_pair_qualification.json": file_digest(
                run_dir / "checkpoint_pair_qualification.json"
            ),
            "provenance.json": file_digest(run_dir / "provenance.json"),
            **{
                _relative(path, run_dir): file_digest(path)
                for path in reports.values()
            },
        },
        "limitations": [
            "The scope is the pre-frozen paired retained-500 subset, not full COCO val2017.",
            "Only one held-out source and one Faster R-CNN victim architecture pair are evaluated.",
            "This is empirical zero-query transfer, not certified robustness or a target-adaptive attack.",
        ],
    }
    atomic_json(run_dir / "formal_training_state_transfer_acceptance.json", acceptance)
    summary = {
        "schema_version": 1,
        "protocol": PROTOCOL_ID,
        "status": "complete",
        "records": len(records),
        "failed_records": 0,
        "retained500_training_state_benchmark_eligible": bool(exact),
        "full_coco_val_claim": False,
        "certified_robustness_claim": False,
        "counts": acceptance["counts"],
        "completed_at_utc": _utc_now(),
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "execution_state.json",
        {
            "schema_version": 1,
            "protocol": PROTOCOL_ID,
            "status": "complete",
            "phase": "complete",
            "clean_evaluations_completed": plan["counts"]["clean_evaluations"],
            "clean_evaluations_total": plan["counts"]["clean_evaluations"],
            "attack_groups_completed": plan["counts"]["attack_groups"],
            "attack_groups_total": plan["counts"]["attack_groups"],
            "attack_evaluations_completed": plan["counts"]["attack_evaluations"],
            "attack_evaluations_total": plan["counts"]["attack_evaluations"],
            "failed_records": 0,
            "workers": [],
            "updated_at_utc": _utc_now(),
        },
    )
    write_artifact_manifest(run_dir)
    return run_dir
