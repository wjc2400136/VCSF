"""Exact physical-device resolver, independent of historical selection evidence."""
from __future__ import annotations

import csv
import os
import re
import subprocess
from typing import Sequence

_GPU_UUID = re.compile(r"^GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")

def _formal_require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError("Selected A23 formal admission: " + message)

def selected_physical_gpu_uuids(devices: Sequence[str]) -> Sequence[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visibility = visible.split(",") if visible else []
    _formal_require(
        visibility and all(
            token == token.strip() and _GPU_UUID.fullmatch(token)
            for token in visibility
        )
        and len(set(visibility)) == len(visibility),
        "CUDA_VISIBLE_DEVICES must list distinct full physical GPU UUIDs",
    )
    rows = list(csv.reader(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
        text=True, timeout=15,
    ).splitlines()))
    available = [row[0].strip() for row in rows if len(row) == 1]
    _formal_require(
        len(available) == len(rows)
        and all(_GPU_UUID.fullmatch(value) for value in available)
        and len(set(available)) == len(available),
        "physical GPU UUID inventory is ambiguous",
    )
    _formal_require(
        all(token in available for token in visibility),
        "visible physical GPU UUID is absent from nvidia-smi",
    )
    selected = []
    for device in devices:
        _formal_require(
            isinstance(device, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device),
            "selected device is not an explicit CUDA ordinal",
        )
        ordinal = int(device.split(":", 1)[1])
        _formal_require(ordinal < len(visibility), "selected CUDA ordinal is not visible")
        selected.append(visibility[ordinal])
    _formal_require(
        1 <= len(selected) <= 2 and len(set(selected)) == len(selected),
        "selected physical GPU UUIDs are not one or two distinct devices",
    )
    return selected
