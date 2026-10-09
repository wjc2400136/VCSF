"""Presentation-only scope metadata; missing evidence is never a population."""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


SCOPE_FIELDS = ("report_scope", "report_dataset", "report_split", "sample_n", "max_images")


def _count(value):
    return value if type(value) is int and value >= 0 else None


def evaluated_samples(record):
    scope = record.get("report_scope") or {}
    if _count(scope.get("sample_n")) is not None:
        return scope["sample_n"]
    if _count(record.get("evaluated_images")) is not None:
        return record["evaluated_images"]
    images = _count(record.get("images"))
    failures = record.get("failures")
    if images is not None and isinstance(failures, list):
        return images - len(failures) if len(failures) <= images else None
    # Without a failure list, "images" alone need not mean successful evaluation.
    return None


def scope_from_records(records=(), *, dataset=None, split=None, max_images=None,
                       diagnostic_only=None, sample_n=None):
    records = list(records)
    values = sorted({n for n in (evaluated_samples(row) for row in records) if n is not None})
    unknown = any(evaluated_samples(row) is None for row in records)
    datasets = {row.get("dataset") for row in records if row.get("dataset")}
    splits = {row.get("split") for row in records if row.get("split")}
    limited = max_images is not None or any(
        (row.get("report_scope") or {}).get("diagnostic_only") is True
        or row.get("diagnostic_only") is True or row.get("max_images") is not None
        or bool(row.get("failures")) for row in records)
    if diagnostic_only is not True and limited:
        diagnostic_only = True
    if sample_n is None and len(values) == 1 and not unknown:
        sample_n = values[0]
    return dict(dataset=dataset or (next(iter(datasets)) if len(datasets) == 1 else None),
                split=split or (next(iter(splits)) if len(splits) == 1 else None),
                sample_n=_count(sample_n), sample_n_values=values,
                sample_n_unknown=unknown or (sample_n is None and not values),
                max_images=_count(max_images), diagnostic_only=diagnostic_only)


def inferred_scope(records):
    records = list(records)
    scopes = [row.get("report_scope") for row in records if row.get("report_scope")]
    caps = {row.get("max_images") for row in records if row.get("max_images") is not None}
    caps.update(scope.get("max_images") for scope in scopes if scope.get("max_images") is not None)
    if not scopes and not caps and not any(
            row.get("diagnostic_only") is True or bool(row.get("failures")) for row in records):
        return None
    return scope_from_records(records, max_images=next(iter(caps)) if len(caps) == 1 else None,
                              diagnostic_only=True if any(
                                  scope.get("diagnostic_only") is True for scope in scopes) else None)


def scope_for_record(record, parent):
    if parent is None:
        return inferred_scope([record])
    return scope_from_records([record], dataset=record.get("dataset") or parent.get("dataset"),
                              split=record.get("split") or parent.get("split"),
                              max_images=parent.get("max_images"),
                              diagnostic_only=parent.get("diagnostic_only"))


def scope_label(scope):
    if scope is None or scope.get("diagnostic_only") is False:
        return ""
    status = "PARTIAL DIAGNOSTIC" if scope.get("diagnostic_only") is True else "SCOPE UNKNOWN"
    n = scope.get("sample_n")
    if n is None:
        values = scope.get("sample_n_values") or []
        n = "per-cell " + "/".join(map(str, values)) if values else "unknown"
        if values and scope.get("sample_n_unknown"):
            n += "/unknown"
    return "{}; actual dataset={}; split={}; sample N={}; max_images={}".format(
        status, scope.get("dataset") or "unknown", scope.get("split") or "unknown",
        n, scope.get("max_images") if scope.get("max_images") is not None else "unknown")


def scope_csv(scope):
    return dict(report_scope=("PARTIAL DIAGNOSTIC" if scope.get("diagnostic_only") is True
                              else "NON-DIAGNOSTIC" if scope.get("diagnostic_only") is False
                              else "SCOPE UNKNOWN"),
                report_dataset=scope.get("dataset") or "unknown",
                report_split=scope.get("split") or "unknown",
                sample_n=scope.get("sample_n") if scope.get("sample_n") is not None else "unknown",
                max_images=scope.get("max_images") if scope.get("max_images") is not None else "unknown")


def escape_scope(text):
    return "".join({c: "\\" + c for c in "&%$#_{}"}.get(c, c) for c in text)


def scoped_tex(text, scope):
    label = scope_label(scope)
    if not label:
        return text
    label = escape_scope(label)
    if label in text:
        return text
    if r"\caption{" in text:
        return text.replace(r"\caption{", r"\caption{" + label + ". ")
    return r"\noindent\textbf{" + label + r"}\par" + "\n" + text