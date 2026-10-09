from __future__ import annotations

import math
from typing import Any


METRIC_DECIMAL_PLACES = 4


def format_decimal(
    value: Any,
    *,
    places: int = METRIC_DECIMAL_PLACES,
    missing: str = "",
) -> str:
    """Format a finite numeric value without changing its stored raw value."""
    if value is None:
        return missing
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(numeric):
        return missing
    return ("{:.%df}" % places).format(numeric)


def format_metric(value: Any, *, missing: str = "") -> str:
    """Use the project-wide presentation precision for AP-style metrics."""
    return format_decimal(
        value,
        places=METRIC_DECIMAL_PLACES,
        missing=missing,
    )
