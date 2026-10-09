"""Structural plan for selected A23 six-source main experiments; no execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional, Sequence

from ..attacks.isolated import (
    A23_CONFIG,
    A23_IMPLEMENTATION,
    validate_isolated_vcsf_descriptor,
)
from ..io import file_digest
from ..registry import Registry
from .formal_parallel import build_execution_assignment, normalize_execution_devices


PROTOCOLS = {
    "coco": "vcsf_selected_a23_coco_main",
    "voc": "vcsf_selected_a23_voc_main",
}


def _protocol_contract_sha256(protocol: Dict[str, Any]) -> str:
    contract = {
        key: value for key, value in protocol.items()
        if key != "formal_wave_admissions"
    }
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_selected_a23_main_plan(
    registry: Registry,
    dataset_id: str,
    *,
    device: str = "cuda:0",
    devices: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if dataset_id not in PROTOCOLS:
        raise ValueError("Selected A23 main supports only COCO and VOC")
    protocol_id = PROTOCOLS[dataset_id]
    protocol = registry.protocols[protocol_id]
    closed = (
        protocol["status"] == "preparation_only"
        and protocol["execution_authorized"] is False
        and protocol["clean_reuse_admitted"] is False
    )
    promoted = (
        protocol["status"] == "active"
        and protocol["execution_authorized"] is True
        and protocol["clean_reuse_admitted"] is True
    )
    if not (closed or promoted):
        raise RuntimeError("Selected A23 main registry state is inconsistent")

    freeze_path = registry.root / protocol["freeze_manifest"]
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    selection_path = (registry.root / freeze["selection_record"]).resolve()
    if (registry.root not in selection_path.parents or not selection_path.is_file()
            or file_digest(selection_path) != freeze["selection_record_sha256"]):
        raise RuntimeError("Selected A23 configuration adjudication drifted")

    descriptor = {
        "candidate_id": freeze["candidate_id"],
        "execution_alias": protocol["isolated_method"],
        "execution_mode": protocol["execution_mode"],
        "implementation": A23_IMPLEMENTATION,
        "config": A23_CONFIG,
        "freeze_manifest": protocol["freeze_manifest"],
        "native_sources": list(protocol["sources"]),
        "file_sha256": dict(freeze["file_sha256"]),
    }
    attack_spec, sources = validate_isolated_vcsf_descriptor(registry, descriptor)
    targets = registry.target_ids()
    if (list(sources) != registry.source_ids()
            or attack_spec.id != "vcsf"
            or protocol["parameters_sha256"] != freeze["parameters_sha256"]):
        raise RuntimeError("Selected A23 source or parameter identity drifted")

    execution_devices = normalize_execution_devices(device, devices)
    if (len(execution_devices) not in (1, 2)
            or any(value not in ("cuda:0", "cuda:1") for value in execution_devices)):
        raise ValueError("Plan one or two distinct explicit CUDA devices")
    budget = dict(registry.budget_profiles[protocol["budget_profile"]])
    jobs = []
    group_items = []
    for group_index, source in enumerate(sources, start=1):
        group_jobs = []
        for target in targets:
            job = {
                "dataset": dataset_id,
                "split": "val",
                "source": source,
                "attack": attack_spec.id,
                "study": "",
                "variant": "default",
                "parameters": {},
                "seed": protocol["seed"],
                "target": target,
                "budget_profile": protocol["budget_profile"],
                "budget": dict(budget),
                "status": "planned_not_authorized" if closed else "planned_pending_wave_verification",
                "compatibility": "native",
                "group_index": group_index,
            }
            jobs.append(job)
            group_jobs.append(job)
        group_items.append(
            ((dataset_id, "val", source, attack_spec.id, "", "default"), group_jobs)
        )
    assignment = build_execution_assignment([], group_items, execution_devices)
    expected = protocol["expected_execution"]
    if (len(group_items) != expected["attack_groups"]
            or len(jobs) != expected["attack_evaluations"]
            or len(sources) * protocol["images"] != expected["generated_images"]
            or len({(job["source"], job["target"]) for job in jobs}) != len(jobs)):
        raise RuntimeError("Selected A23 one-method matrix is incomplete")
    if [group["source"] for group in assignment["attack_groups"]] != list(sources):
        raise RuntimeError("Selected A23 worker assignment changed source order")
    return {
        "schema_version": 1,
        "status": "preparation_only" if closed else "admitted_pending_wave_verification",
        "formal_eligible": False,
        "execution_authorized": protocol["execution_authorized"],
        "model_calls": 0,
        "gpu_calls": 0,
        "protocol": protocol_id,
        "protocol_contract_sha256": _protocol_contract_sha256(protocol),
        "dataset": dataset_id,
        "split": "val",
        "images": protocol["images"],
        "seed": protocol["seed"],
        "selection_scope": protocol["selection_scope"],
        "selection_overlap_with_evaluation": protocol["selection_overlap_with_evaluation"],
        "selection_record": {
            "path": freeze["selection_record"],
            "sha256": freeze["selection_record_sha256"],
        },
        "sources": list(sources),
        "targets": targets,
        "methods": [attack_spec.id],
        "isolated_candidate": descriptor,
        "parameters_sha256": protocol["parameters_sha256"],
        "clean_reuse": {
            "status": "candidate_not_admitted" if closed else "registry_promoted_pending_wave_verification",
            "candidate_reference_cells": expected["candidate_reused_clean_cells"],
            "binding_sha256": protocol["candidate_clean_binding_sha256"],
            "audit_sha256": protocol["candidate_clean_audit_sha256"],
        },
        "payload_retention": protocol["payload_retention"],
        "expected_execution": dict(expected),
        "assignment": assignment,
        "jobs": jobs,
    }
