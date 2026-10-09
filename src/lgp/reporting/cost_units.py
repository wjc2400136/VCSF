"""Read existing counters without equating logical updates and physical BE."""
from __future__ import annotations

import math


COST_FIELDS = ("declared_budget", "declared_budget_profile", "declared_count_unit",
               "observed_actual_count", "observed_count_unit", "calibrated_whole_detector_BE")


def actual_count(record):
    value = record.get("attack_actual_gradient_evaluations_mean",
                       record.get("actual_gradient_evaluations_mean"))
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def count_unit(record):
    definition = record.get("budget_profile_definition") or record.get("budget") or {}
    unit = definition.get("accounting_unit")
    if not isinstance(unit, str) or not unit.strip():
        return "unknown"
    if "logical" in unit.lower():
        return "logical updates; not whole-detector BE"
    if "complete" in unit.lower() and "detector" in unit.lower():
        return "complete source-detector backward image-views"
    return unit


def cost_csv(record):
    value = actual_count(record)
    definition = record.get("budget_profile_definition") or record.get("budget") or {}
    calibrated = record.get("calibrated_whole_detector_BE")
    return dict(declared_budget=record.get("gradient_evaluations_per_image"),
                declared_budget_profile=record.get("budget_profile") or "unknown",
                declared_count_unit=definition.get("accounting_unit") or "unknown",
                observed_actual_count="{:.4f}".format(value) if value is not None else "NR",
                observed_count_unit=count_unit(record),
                calibrated_whole_detector_BE=(
                    "{:.4f}".format(calibrated) if calibrated is not None else "NR"))


COST_NOTE = (
    "Declared budget/profile and observed actual mean counts are separate. "
    "Counts use the registered method/profile unit, not a shared calibrated physical cost. "
    "VCSF logical updates can include detector initialization and partial backbone/neck "
    "backward paths; paired clean-reference image rows and adversarial differentiable "
    "views are separate work, not free or complete-detector BE. Explicit auxiliary zero "
    "does not mean those paths/references are absent. Missing observed counts and "
    "calibrated whole-detector BE remain NR; neither is inferred from the declared budget."
    " Preprocessing generation counts describe the bound generation population N: "
    "reused original generation is not newly executed adaptation or its smaller evaluation subset."
)
