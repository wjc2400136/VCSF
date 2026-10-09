from __future__ import annotations

from .scope import SCOPE_FIELDS, inferred_scope, scope_csv, scope_for_record, scope_label, scoped_tex
from .cost_units import COST_FIELDS, COST_NOTE, actual_count, cost_csv, count_unit
from .preprocessing_cost import GENERATION_FIELDS, generation_cost_csv

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..metrics import COCO_BBOX_METRICS
from ..registry import Registry
from .formatting import format_decimal, format_metric


TRANSFER_TABLE_METHOD_ORDER: Tuple[str, ...] = (
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
)
TRANSFER_TABLE_DISPLAY_SCALE = 100.0
TRANSFER_TABLE_DECIMAL_PLACES = 2
TRANSFER_TABLE_RANK_TOLERANCE = 5.0e-7


def _transfer_metric_value(value: Any) -> Optional[float]:
    return (float(value) if type(value) in (int, float)
            and math.isfinite(value) and 0 <= value <= 1 else None)


def metric_report_view(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Keep undefined official metrics out of derived displays, not raw records."""
    return [dict(record, metrics={
        key: _transfer_metric_value(value) if key in COCO_BBOX_METRICS else value
        for key, value in (record.get("metrics") or {}).items()}) for record in records]


def _escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _label(value: str) -> str:
    """Keep common LaTeX label characters without text-mode escaping."""
    return "".join(
        character if character.isalnum() or character in ":._-" else "_"
        for character in value
    )


def _number(value: Optional[float], bold: bool = False) -> str:
    rendered = format_metric(value, missing=r"\textit{NR}")
    if rendered == r"\textit{NR}":
        return r"\textit{NR}"
    return r"\textbf{" + rendered + "}" if bold else rendered


def _ordered_transfer_method_ids(method_ids: Sequence[str]) -> List[str]:
    """Apply the publication-table order without changing protocol order."""
    selected = list(dict.fromkeys(str(method_id) for method_id in method_ids))
    selected_set = set(selected)
    ordered = [
        method_id
        for method_id in TRANSFER_TABLE_METHOD_ORDER
        if method_id in selected_set
    ]
    ordered.extend(method_id for method_id in selected if method_id not in ordered)
    return ordered


def _minimum_rank_values(
    values: Sequence[float],
) -> Tuple[Optional[float], Optional[float]]:
    """Return the lowest and second-lowest distinct finite values."""
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None, None
    best = ordered[0]
    second = next(
        (
            value
            for value in ordered[1:]
            if abs(value - best) >= TRANSFER_TABLE_RANK_TOLERANCE
        ),
        None,
    )
    return best, second


def _minimum_rank(
    value: float,
    ranked_values: Tuple[Optional[float], Optional[float]],
) -> Optional[str]:
    best, second = ranked_values
    if best is not None and abs(float(value) - best) < TRANSFER_TABLE_RANK_TOLERANCE:
        return "best"
    if second is not None and abs(float(value) - second) < TRANSFER_TABLE_RANK_TOLERANCE:
        return "second"
    return None


def _transfer_number(
    value: Optional[float],
    *,
    rank: Optional[str] = None,
    whitebox: bool = False,
    decimal_places: int = TRANSFER_TABLE_DECIMAL_PLACES,
) -> str:
    """Render a raw [0, 1] metric on the publication table's 0--100 scale."""
    if value is None:
        return r"\textit{NR}"
    rendered = format_decimal(
        float(value) * TRANSFER_TABLE_DISPLAY_SCALE,
        places=decimal_places,
    )
    if whitebox:
        rendered += r"$^\ast$"
    if rank == "best":
        return r"\textbf{" + rendered + "}"
    if rank == "second":
        return r"\underline{" + rendered + "}"
    return rendered


def _transfer_dataset_scope(dataset: str) -> str:
    return {
        "coco": "COCO val2017",
        "voc": "Pascal VOC 2007 test",
        "bdd100k": "BDD100K val",
    }.get(dataset, dataset)


def _missing_result_code(
    record_status: Optional[str], compatibility_status: str
) -> str:
    """Keep missing execution state distinct from structural incompatibility."""
    if record_status == "skipped" or compatibility_status == "unsupported":
        return "--"
    if record_status in {"failed", "complete_with_failures"}:
        return r"\textit{ERR}"
    if record_status == "artifact_required" or compatibility_status == "external":
        return r"\textit{A}"
    if record_status == "adapter_required" or compatibility_status == "adaptable":
        return r"\textit{P}"
    return r"\textit{NR}"


def _attack_display_name(
    registry: Registry, attack: str, budget_profile: Optional[str] = None
) -> str:
    name = registry.attack(attack).display_name if attack in registry.attacks else attack
    if budget_profile == "compute_matched" and attack in registry.baseline_profiles:
        return name + "-CM"
    return name


def _canonical_model_selection(
    registry: Registry,
    selected: Sequence[str],
    *,
    sources_only: bool = False,
) -> List[str]:
    canonical = registry.source_ids() if sources_only else registry.target_ids()
    unknown = sorted(set(selected) - set(canonical))
    if unknown:
        raise ValueError(
            "Unknown {} model(s): {}".format(
                "source" if sources_only else "target",
                ", ".join(unknown),
            )
        )
    selected_set = set(selected)
    return [model_id for model_id in canonical if model_id in selected_set]


def _chunks(values: Sequence[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        raise ValueError("Table part size must be positive")
    return [
        list(values[index : index + size])
        for index in range(0, len(values), size)
    ]


def _canonical_attack_ids(
    registry: Registry,
    dataset: str,
) -> List[str]:
    ordered: List[str] = []
    for protocol in registry.protocols.values():
        if (
            protocol.get("formal_all_methods") is True
            and list(protocol.get("datasets") or []) == [dataset]
        ):
            for method in protocol.get("methods") or []:
                method_id = str(method)
                if method_id not in ordered:
                    ordered.append(method_id)
    for method_id in registry.attacks:
        if method_id not in ordered:
            ordered.append(method_id)
    return ordered


def _canonical_record_order_key(
    record: Mapping[str, Any],
    registry: Registry,
) -> Tuple[Any, ...]:
    source = str(record.get("source", ""))
    attack = str(record.get("attack", ""))
    target = str(record.get("target", ""))
    source_rank = {
        model_id: index for index, model_id in enumerate(registry.source_ids())
    }
    attack_rank = {
        attack_id: index
        for index, attack_id in enumerate(
            _canonical_attack_ids(
                registry, str(record.get("dataset", ""))
            )
        )
    }
    target_rank = {
        model_id: index for index, model_id in enumerate(registry.target_ids())
    }
    if attack == "clean":
        return (
            0,
            target_rank.get(target, len(target_rank)),
            target,
        )
    return (
        1,
        source_rank.get(source, len(source_rank)),
        source,
        attack_rank.get(attack, len(attack_rank)),
        attack,
        target_rank.get(target, len(target_rank)),
        target,
        str(record.get("variant", "") or ""),
    )


def _part_caption(caption: str, part: int, total: int) -> str:
    if total <= 1:
        return caption
    return "{} (part {} of {})".format(caption.rstrip("."), part, total)


def _part_label(label: str, part: int) -> str:
    if part <= 1:
        return label
    return "{}_part_{}".format(label, part)


def _latex_model_code(registry: Registry, model_id: str) -> str:
    model = registry.model(model_id)
    markers: List[str] = []
    if model.source:
        markers.append(r"\dagger")
    if model.held_out:
        markers.append(r"\ddagger")
    suffix = r"$^{" + "".join(markers) + "}$" if markers else ""
    return _escape(model.table_code) + suffix


def _target_group_segments(
    registry: Registry, target_ids: Sequence[str]
) -> List[Tuple[str, str, int]]:
    segments: List[Tuple[str, str, int]] = []
    for target in target_ids:
        group = registry.model_group(target)
        if segments and segments[-1][0] == group.id:
            previous = segments[-1]
            segments[-1] = (previous[0], previous[1], previous[2] + 1)
        else:
            segments.append((group.id, group.display_name, 1))
    return segments


def _latex_group_header(
    registry: Registry,
    target_ids: Sequence[str],
    *,
    prefix_columns: int,
    trailing_columns: int,
) -> List[str]:
    segments = _target_group_segments(registry, target_ids)
    cells = [r"\multicolumn{" + str(prefix_columns) + r"}{c}{}"]
    rules: List[str] = []
    column = prefix_columns + 1
    for _, display_name, count in segments:
        cells.append(
            r"\multicolumn{"
            + str(count)
            + r"}{c}{"
            + _escape(display_name)
            + "}"
        )
        rules.append(
            r"\cmidrule(lr){"
            + str(column)
            + "-"
            + str(column + count - 1)
            + "}"
        )
        column += count
    if trailing_columns:
        cells.append(r"\multicolumn{" + str(trailing_columns) + r"}{c}{}")
    return [" & ".join(cells) + r" \\", " ".join(rules)]


def _latex_role_note(
    registry: Registry,
    target_ids: Sequence[str],
    *,
    total_columns: int,
    include_blackbox_mean: bool,
    include_whitebox_asterisk: bool = False,
) -> Optional[str]:
    notes: List[str] = []
    if include_whitebox_asterisk:
        notes.append(r"$^\ast$ source-identical white-box cell")
    if any(registry.model(target).source for target in target_ids):
        notes.append(r"$^\dagger$ white-box source and evaluation target")
    if any(registry.model(target).held_out for target in target_ids):
        notes.append(r"$^\ddagger$ held-out evaluation target")
    if include_blackbox_mean:
        notes.append("BB Mean excludes the source-matched target")
    if not notes:
        return None
    return (
        r"\multicolumn{"
        + str(total_columns)
        + r"}{l}{\footnotesize "
        + "; ".join(notes)
        + r".} \\"
    )


def latex_transfer_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    metric: str = "bbox_mAP_50",
    caption: Optional[str] = None,
    label: Optional[str] = None,
    source_ids: Optional[Sequence[str]] = None,
    target_ids: Optional[Sequence[str]] = None,
    method_ids: Optional[Sequence[str]] = None,
    preserve_method_order: bool = False,
    require_complete_panel: bool = False,
    decimal_places: int = TRANSFER_TABLE_DECIMAL_PLACES,
    report_scope=None,
) -> str:
    if type(decimal_places) is not int or not 2 <= decimal_places <= 8:
        raise ValueError("Transfer table precision must be an integer from two to eight")
    records = metric_report_view(records)
    metric_name = {
        "bbox_mAP": "AP",
        "bbox_mAP_50": "AP@.50",
        "bbox_mAP_75": "AP@.75",
    }.get(metric, metric)
    label = label or "tab:{}_detector_{}".format(dataset, metric)
    present_sources: List[str] = []
    present_targets: List[str] = []
    present_methods: List[str] = []
    for record in records:
        source = str(record.get("source", ""))
        target = str(record.get("target", ""))
        method = str(record.get("attack", ""))
        if (
            method != "clean"
            and source in registry.models
            and source not in present_sources
        ):
            present_sources.append(source)
        if target in registry.models and target not in present_targets:
            present_targets.append(target)
        if (
            method in registry.attacks
            and method not in present_methods
        ):
            present_methods.append(method)
    source_ids = _canonical_model_selection(
        registry,
        list(source_ids or present_sources or registry.source_ids()),
        sources_only=True,
    )
    target_ids = _canonical_model_selection(
        registry,
        list(target_ids or present_targets or registry.target_ids()),
    )
    method_ids = list(
        method_ids
        or present_methods
        or ["tog", "lgp", "osfd", "afog", "numbod", "svfta"]
    )
    method_ids = (list(dict.fromkeys(method_ids)) if preserve_method_order
                  else _ordered_transfer_method_ids(method_ids))
    resolved_scope = report_scope or inferred_scope(records)
    dataset_caption = _transfer_dataset_scope(dataset)
    if scope_label(resolved_scope):
        dataset_caption = "{} (split={})".format(
            dataset.upper(), resolved_scope.get("split") or "unknown")
    caption = caption or (
        "{} of {} target detectors under adversarial examples generated on {} "
        "white-box source detectors on {}; source-identical entries are marked "
        "with an asterisk and excluded from BB Mean. Values are 100 times the "
        "raw [0,1] JSON metrics."
    ).format(
        metric_name,
        len(target_ids),
        len(source_ids),
        dataset_caption,
    )
    values: Dict[Tuple[str, str, str], Optional[float]] = {}
    record_statuses: Dict[Tuple[str, str, str], str] = {}
    method_profiles: Dict[str, set] = defaultdict(set)
    benign: Dict[str, Optional[float]] = {}
    benign_statuses: Dict[str, str] = {}
    for record in records:
        if record.get("dataset") != dataset:
            continue
        target = str(record.get("target", ""))
        metrics = record.get("metrics") or {}
        value = metrics.get(metric)
        if record.get("attack") == "clean":
            benign[target] = float(value) if value is not None else None
            benign_statuses[target] = str(record.get("status", ""))
            continue
        attack_id = str(record.get("attack", ""))
        if record.get("budget_profile"):
            method_profiles[attack_id].add(str(record["budget_profile"]))
        key = (
            str(record.get("source", "")),
            str(record.get("attack", "")),
            target,
        )
        if value is not None:
            values[key] = float(value)
            record_statuses[key] = str(record.get("status", "complete"))
        elif key not in values:
            values[key] = None
            record_statuses[key] = str(record.get("status", ""))

    cell_ranks: Dict[
        Tuple[str, str], Tuple[Optional[float], Optional[float]]
    ] = {}
    blackbox_means: Dict[Tuple[str, str], Optional[float]] = {}
    mean_ranks: Dict[
        str, Tuple[Optional[float], Optional[float]]
    ] = {}
    for source in source_ids:
        comparison_complete = not require_complete_panel or all(
            values.get((source, method, target)) is not None
            for method in method_ids for target in target_ids
        )
        for target in target_ids:
            candidates = [
                values.get((source, method, target))
                for method in method_ids
                if values.get((source, method, target)) is not None
            ]
            if candidates and comparison_complete:
                cell_ranks[(source, target)] = _minimum_rank_values(
                    [float(value) for value in candidates if value is not None]
                )
        for method in method_ids:
            numeric = [
                float(values[(source, method, target)])
                for target in target_ids
                if target != source
                and values.get((source, method, target)) is not None
            ]
            blackbox_means[(source, method)] = (
                sum(numeric) / len(numeric) if numeric and (
                    not require_complete_panel or len(numeric) == sum(target != source for target in target_ids)
                ) else None
            )
        mean_candidates = [
            float(value)
            for method in method_ids
            for value in [blackbox_means[(source, method)]]
            if value is not None
        ]
        mean_ranks[source] = _minimum_rank_values(mean_candidates) if comparison_complete else (None, None)

    columns = "c l " + " ".join("c" for _ in target_ids) + " c"
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{" + _escape(caption) + "}",
        r"\label{" + _label(label) + "}",
        r"\begingroup",
        r"\setlength{\tabcolsep}{1pt}",
        r"\renewcommand{\arraystretch}{0.65}",
        r"\setbox0=\hbox{%",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
    ]
    lines.extend(
        _latex_group_header(
            registry,
            target_ids,
            prefix_columns=2,
            trailing_columns=1,
        )
    )
    benign_cells = [
        _transfer_number(benign.get(target), decimal_places=decimal_places)
        if benign.get(target) is not None
        else (_missing_result_code(benign_statuses.get(target), "native")
              if require_complete_panel else r"\textit{NR}")
        for target in target_ids
    ]
    lines.append(
        r"\multicolumn{2}{c}{Benign " + _escape(metric_name) + "} & "
        + " & ".join(benign_cells)
        + r" & \textit{N/A} \\"
    )
    lines.append(r"\midrule")
    backbones = [
        _escape(registry.model(target).backbone) for target in target_ids
    ]
    lines.append(
        r"\multicolumn{2}{c}{Backbone} & "
        + " & ".join(backbones)
        + r" & \multirow{2}{*}{BB Mean} \\"
    )
    lines.append(
        "Model & Attack & "
        + " & ".join(
            _latex_model_code(registry, target) for target in target_ids
        )
        + r" & \\"
    )
    lines.append(r"\midrule")

    for source_index, source in enumerate(source_ids):
        for method_index, method in enumerate(method_ids):
            cells = []
            compatibility = registry.compatibility_status(method, source)[
                "status"
            ]
            for target in target_ids:
                key = (source, method, target)
                value = values.get(key)
                if value is None:
                    cells.append(
                        _missing_result_code(
                            record_statuses.get(key), str(compatibility)
                        )
                    )
                else:
                    cells.append(
                        _transfer_number(
                            value,
                            rank=_minimum_rank(
                                float(value),
                                cell_ranks.get((source, target), (None, None)),
                            ),
                            whitebox=target == source,
                            decimal_places=decimal_places,
                        )
                    )
            profiles = method_profiles.get(method, set())
            if len(profiles) > 1:
                raise ValueError(
                    "Do not mix budget profiles for '{}' in one transfer table: {}".format(
                        method, ", ".join(sorted(profiles))
                    )
                )
            method_name = _attack_display_name(
                registry,
                method,
                next(iter(profiles)) if profiles else None,
            )
            prefix = ""
            if method_index == 0:
                source_code = _escape(registry.model(source).table_code)
                prefix = (source_code if len(method_ids) == 1 else
                          r"\multirow{" + str(len(method_ids)) + r"}{*}{" + source_code + "}")
            mean_value = blackbox_means[(source, method)]
            lines.append(
                prefix
                + " & "
                + _escape(method_name)
                + " & "
                + " & ".join(cells)
                + " & "
                + (
                    _transfer_number(
                        mean_value,
                        decimal_places=decimal_places,
                        rank=_minimum_rank(
                            float(mean_value), mean_ranks[source]
                        ),
                    )
                    if mean_value is not None
                    else _missing_result_code(None, str(compatibility))
                )
                + r" \\"
            )
        if source_index != len(source_ids) - 1:
            lines.append(r"\midrule")
    role_note = _latex_role_note(
        registry,
        target_ids,
        total_columns=2 + len(target_ids) + 1,
        include_blackbox_mean=True,
        include_whitebox_asterisk=True,
    )
    if role_note:
        lines.extend([r"\midrule", role_note])
    lines.extend(
        [r"\bottomrule", r"\end{tabular}}",
         r"\ifdim\wd0>\textwidth",
         r"\setbox0=\hbox{\resizebox{\textwidth}{!}{\copy0}}",
         r"\fi",
         r"\leavevmode",
         r"\ifdim\dimexpr\ht0+\dp0\relax>0.88\textheight",
         r"\resizebox*{!}{0.88\textheight}{\copy0}",
         r"\else",
         r"\copy0",
         r"\fi",
         r"\endgroup",
         r"\end{table*}", ""]
    )
    return scoped_tex("\n".join(lines), resolved_scope)


def write_transfer_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    dataset: str,
    metric: str = "bbox_mAP_50",
    include_failure_markers: bool = False,
    report_scope=None,
    **table_options: Any
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / ("transfer_{}.tex".format(metric))
    tex_path.write_text(
        latex_transfer_table(
            records,
            registry,
            dataset=dataset,
            metric=metric,
            report_scope=report_scope,
            **table_options
        ),
        encoding="utf-8",
    )
    csv_path = output_dir / "transfer_records.csv"
    fields = [
        "dataset",
        "split",
        "source",
        "attack",
        "budget_profile",
        "gradient_evaluations_per_image",
        "parameters_sha256",
        "code_commit",
        "target",
        "status",
        *COCO_BBOX_METRICS,
        "reason",
    ]
    report_scope = report_scope or inferred_scope(
        [record for record in records if record.get("dataset") == dataset])
    scoped = bool(scope_label(report_scope))
    if scoped:
        fields.extend(SCOPE_FIELDS)
    has_cost = any(record.get("attack") != "clean" and (
        "attack_actual_gradient_evaluations_mean" in record
        or bool(record.get("budget_profile_definition"))) for record in records)
    if has_cost:
        fields.extend(COST_FIELDS)
    has_generation = any(row.get("generation_cost_scope") for row in records)
    if has_generation:
        fields.extend(GENERATION_FIELDS)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        source_rank = {
            model_id: index for index, model_id in enumerate(registry.source_ids())
        }
        selected_methods = list(
            table_options.get("method_ids") or registry.attacks
        )
        attack_rank = {
            attack_id: index
            for index, attack_id in enumerate(selected_methods)
        }
        target_rank = {
            model_id: index for index, model_id in enumerate(registry.target_ids())
        }
        ordered_records = sorted(
            (record for record in records if record.get("dataset") == dataset),
            key=lambda record: (
                str(record.get("dataset", "")),
                0 if str(record.get("attack", "")) == "clean" else 1,
                source_rank.get(str(record.get("source", "")), len(source_rank)),
                attack_rank.get(str(record.get("attack", "")), len(attack_rank)),
                target_rank.get(str(record.get("target", "")), len(target_rank)),
            ),
        )
        for record in ordered_records:
            metrics = record.get("metrics") or {}
            def metric_text(name):
                raw_value = metrics.get(name)
                value = _transfer_metric_value(raw_value)
                if not include_failure_markers:
                    if raw_value is not None and value is None:
                        return "NR"
                    return format_metric(value, missing="")
                if record.get("status") == "skipped":
                    return "--"
                if record.get("status") in {"failed", "complete_with_failures"}:
                    return "ERR"
                if value is None:
                    return "NR"
                return format_metric(value, missing="NR")

            writer.writerow(
                {
                    "dataset": record.get("dataset"),
                    "split": record.get("split"),
                    "source": record.get("source"),
                    "attack": record.get("attack"),
                    "budget_profile": record.get("budget_profile"),
                    "gradient_evaluations_per_image": record.get(
                        "gradient_evaluations_per_image"
                    ),
                    "parameters_sha256": record.get("parameters_sha256"),
                    "code_commit": record.get("code_commit"),
                    "target": record.get("target"),
                    "status": record.get("status", "ok"),
                    **{
                        metric_name: metric_text(metric_name)
                        for metric_name in COCO_BBOX_METRICS
                    },
                    "reason": record.get("reason", ""),
                    **(scope_csv(scope_for_record(record, report_scope)) if scoped else {}),
                    **(cost_csv(record) if has_cost else {}),
                    **(generation_cost_csv(record) if has_generation else {}),
                }
            )
    return {"tex": tex_path, "csv": csv_path}


def latex_analysis_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    caption: str = "Detection, perceptual-quality and efficiency breakdown.",
    label: Optional[str] = None,
    rows_per_table: int = 24,
) -> str:
    """Render canonical, page-sized source/attack/target audit tables."""
    rows = [
        record
        for record in records
        if record.get("dataset") == dataset
        and bool(record.get("metrics"))
        and str(record.get("status", "complete")) == "complete"
    ]
    rows.sort(key=lambda record: _canonical_record_order_key(record, registry))
    row_parts = _chunks(rows, rows_per_table) or [[]]
    base_label = label or "tab:" + dataset + "_attack_analysis"
    caption += " Observed count uses the method/profile unit, not calibrated whole-detector BE; see runtime accounting."
    output: List[str] = []
    for part_index, part_rows in enumerate(row_parts, start=1):
        lines = [
            r"\begin{table*}[!htb]",
            r"\centering",
            r"\caption{"
            + _escape(_part_caption(caption, part_index, len(row_parts)))
            + "}",
            r"\label{"
            + _label(_part_label(base_label, part_index))
            + "}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l l l *{11}{c}}",
            r"\toprule",
            r"Source & Attack & Target & AP & AP$_S$ & AP$_M$ & AP$_L$ & PSNR & SSIM & Observed count & Aux bwd. & Attack s/img & Eval s/img & Peak MiB \\",
            r"\midrule",
        ]
        for record in part_rows:
            source = str(record.get("source", ""))
            attack = str(record.get("attack", ""))
            target = str(record.get("target", ""))
            metrics = record.get("metrics") or {}
            quality = record.get("attack_quality_mean") or {}
            source_name = (
                registry.model(source).table_code
                if source in registry.models
                else source
            )
            attack_name = _attack_display_name(
                registry,
                attack,
                str(record.get("budget_profile") or "") or None,
            )
            target_name = (
                registry.model(target).table_code
                if target in registry.models
                else target
            )
            values = [
                metrics.get("bbox_mAP"),
                metrics.get("bbox_mAP_small"),
                metrics.get("bbox_mAP_medium"),
                metrics.get("bbox_mAP_large"),
                quality.get("psnr"),
                quality.get("ssim"),
                actual_count(record),
                (record.get("auxiliary_passes") or {}).get("backward_mean"),
                record.get("attack_runtime_seconds_mean"),
                record.get("inference_seconds_mean"),
                record.get("attack_peak_cuda_memory_mb"),
            ]
            lines.append(
                " & ".join(
                    [
                        _escape(source_name),
                        _escape(attack_name),
                        _escape(target_name),
                    ]
                    + [_number(value) for value in values]
                )
                + r" \\"
            )
        lines.extend(
            [r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""]
        )
        output.extend(lines)
        if part_index != len(row_parts):
            output.extend([r"\clearpage", ""])
    return "\n".join(output)


def latex_category_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    caption: str = "Per-category AP breakdown.",
    label: str = "tab:category_ap",
    rows_per_table: int = 24,
    categories_per_table: int = 20,
) -> str:
    """Render canonical per-category AP in bounded row and column parts."""
    categories = list(registry.dataset(dataset).classes)
    rows = [
        record for record in records if record.get("dataset") == dataset
    ]
    rows.sort(key=lambda record: _canonical_record_order_key(record, registry))
    row_parts = _chunks(rows, rows_per_table) or [[]]
    category_parts = _chunks(categories, categories_per_table) or [[]]
    total_parts = len(row_parts) * len(category_parts)
    output: List[str] = []
    part_index = 0
    for part_categories in category_parts:
        for part_rows in row_parts:
            part_index += 1
            columns = "l l l " + " ".join(
                "c" for _ in part_categories
            )
            lines = [
                r"\begin{table*}[!htb]",
                r"\centering",
                r"\caption{"
                + _escape(_part_caption(caption, part_index, total_parts))
                + "}",
                r"\label{"
                + _label(_part_label(label, part_index))
                + "}",
                r"\resizebox{\textwidth}{!}{",
                r"\begin{tabular}{" + columns + "}",
                r"\toprule",
                "Source & Attack & Target & "
                + " & ".join(
                    _escape(name) for name in part_categories
                )
                + r" \\",
                r"\midrule",
            ]
            for record in part_rows:
                source = str(record.get("source", ""))
                attack = str(record.get("attack", ""))
                target = str(record.get("target", ""))
                category_ap = record.get("per_category_ap") or {}
                compatibility = (
                    registry.compatibility_status(attack, source)["status"]
                    if attack in registry.attacks
                    and source in registry.source_ids()
                    else "native"
                )
                missing_cell = _missing_result_code(
                    str(record.get("status", "")), str(compatibility)
                )
                lines.append(
                    " & ".join(
                        [
                            _escape(
                                registry.model(source).table_code
                                if source in registry.models
                                else source
                            ),
                            _escape(
                                _attack_display_name(
                                    registry,
                                    attack,
                                    str(
                                        record.get("budget_profile") or ""
                                    )
                                    or None,
                                )
                            ),
                            _escape(
                                registry.model(target).table_code
                                if target in registry.models
                                else target
                            ),
                        ]
                        + [
                            _number(category_ap.get(name))
                            if category_ap.get(name) is not None
                            else missing_cell
                            for name in part_categories
                        ]
                    )
                    + r" \\"
                )
            lines.extend(
                [
                    r"\bottomrule",
                    r"\end{tabular}",
                    "}",
                    r"\end{table*}",
                    "",
                ]
            )
            output.extend(lines)
            if part_index != total_parts:
                output.extend([r"\clearpage", ""])
    return "\n".join(output)


def latex_family_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    metric: str = "bbox_mAP_50",
    rows_per_table: int = 24,
) -> str:
    families: List[str] = []
    for target in registry.target_ids():
        family = registry.model(target).family
        if family not in families:
            families.append(family)
    grouped: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    row_keys = set()
    row_profiles: Dict[Tuple[str, str], set] = defaultdict(set)
    for record in records:
        if record.get("dataset") != dataset:
            continue
        target = str(record.get("target", ""))
        if target not in registry.models:
            continue
        source = str(record.get("source", ""))
        attack = str(record.get("attack", ""))
        row = (source, attack)
        row_keys.add(row)
        if record.get("budget_profile"):
            row_profiles[row].add(str(record["budget_profile"]))
        family = registry.model(target).family
        value = (record.get("metrics") or {}).get(metric)
        if value is not None and math.isfinite(float(value)):
            grouped[(source, attack, family)].append(float(value))
    source_rank = {
        model_id: index for index, model_id in enumerate(registry.source_ids())
    }
    attack_rank = {
        attack_id: index
        for index, attack_id in enumerate(
            _canonical_attack_ids(registry, dataset)
        )
    }
    row_order = sorted(
        row_keys,
        key=lambda row: (
            0 if row[1] == "clean" else 1,
            source_rank.get(row[0], len(source_rank)),
            row[0],
            attack_rank.get(row[1], len(attack_rank)),
            row[1],
        ),
    )
    rendered_rows: List[str] = []
    for source, attack in row_order:
        cells = []
        for family in families:
            values = grouped.get((source, attack, family), [])
            cells.append(
                _number(sum(values) / len(values) if values else None)
            )
        source_name = (
            registry.model(source).table_code
            if source in registry.models
            else source
        )
        profiles = row_profiles.get((source, attack), set())
        if len(profiles) > 1:
            raise ValueError(
                "Do not mix budget profiles for '{}/{}' in one family table".format(
                    source, attack
                )
            )
        matching_profile = next(iter(profiles)) if profiles else None
        attack_name = _attack_display_name(
            registry, attack, matching_profile
        )
        rendered_rows.append(
            " & ".join(
                [_escape(source_name), _escape(attack_name)] + cells
            )
            + r" \\"
        )
    row_parts = _chunks(rendered_rows, rows_per_table) or [[]]
    caption = "Mean AP@.50 grouped by target architecture family."
    label = "tab:target_family"
    output: List[str] = []
    for part_index, part_rows in enumerate(row_parts, start=1):
        lines = [
            r"\begin{table*}[!htb]",
            r"\centering",
            r"\caption{"
            + _escape(_part_caption(caption, part_index, len(row_parts)))
            + "}",
            r"\label{" + _label(_part_label(label, part_index)) + "}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l l "
            + " ".join("c" for _ in families)
            + "}",
            r"\toprule",
            "Source & Attack & "
            + " & ".join(_escape(value) for value in families)
            + r" \\",
            r"\midrule",
            *part_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            r"\end{table*}",
            "",
        ]
        output.extend(lines)
        if part_index != len(row_parts):
            output.extend([r"\clearpage", ""])
    return "\n".join(output)


def latex_qualitative_figure(
    records: Iterable[Mapping[str, Any]],
    limit: int = 3,
    allow_payload_pairs: bool = True,
) -> str:
    records = list(records)
    examples: List[Tuple[str, str]] = []
    clean_overlays: Dict[Tuple[str, str, str, int], str] = {}
    adversarial_overlays: List[Tuple[str, str, str, int, str]] = []
    for record in records:
        visualization = record.get("prediction_visualizations") or {}
        manifest_value = visualization.get("manifest")
        if not manifest_value:
            continue
        manifest = Path(str(manifest_value))
        if not manifest.is_file():
            continue
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if any(
            record.get(field) and payload.get(field)
            and record[field] != payload[field]
            for field in ("dataset", "split")
        ):
            continue
        # Missing legacy identity is an exact bucket, never a wildcard.
        dataset = str(record.get("dataset") or payload.get("dataset") or "")
        split = str(record.get("split") or payload.get("split") or "")
        target = str(record.get("target", ""))
        is_clean = record.get("attack") == "clean"
        for item in payload.get("images", []):
            overlay = Path(str(item.get("overlay_path", "")))
            if not overlay.is_file():
                continue
            image_id = int(item["image_id"])
            normalized = str(overlay.resolve()).replace("\\", "/")
            if is_clean:
                clean_overlays[(dataset, split, target, image_id)] = normalized
            else:
                adversarial_overlays.append(
                    (dataset, split, target, image_id, normalized)
                )

    seen_overlay_pairs = set()
    for dataset, split, target, image_id, adversarial in adversarial_overlays:
        clean = clean_overlays.get((dataset, split, target, image_id))
        key = (dataset, split, target, image_id, adversarial)
        if clean is None or key in seen_overlay_pairs:
            continue
        seen_overlay_pairs.add(key)
        examples.append((clean, adversarial))
        if len(examples) >= limit:
            break

    uses_prediction_overlays = bool(examples)
    seen = set()
    if not examples and allow_payload_pairs:
        for record in records:
            run_value = record.get("adversarial_run")
            if not run_value or str(run_value) in seen:
                continue
            seen.add(str(run_value))
            run_dir = Path(str(run_value))
            manifest = run_dir / "manifest.jsonl"
            if not manifest.is_file():
                continue
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if item.get("status") != "ok":
                    continue
                clean = Path(str(item["source_file"]))
                adversarial = run_dir / str(item["output_file"])
                if clean.is_file() and adversarial.is_file():
                    examples.append(
                        (
                            str(clean.resolve()).replace("\\", "/"),
                            str(adversarial.resolve()).replace("\\", "/"),
                        )
                    )
                    break
            if len(examples) >= limit:
                break
    if not examples:
        return "% No qualitative artifact pair was available in these records.\n"
    width = 0.95 / (2 * len(examples))
    lines = [r"\begin{figure*}[!htb]", r"\centering"]
    for index, (clean, adversarial) in enumerate(examples, start=1):
        lines.append(
            r"\includegraphics[width={:.3f}\textwidth]{{\detokenize{{{}}}}}".format(width, clean)
        )
        lines.append(
            r"\includegraphics[width={:.3f}\textwidth]{{\detokenize{{{}}}}}".format(width, adversarial)
        )
    lines.extend(
        [
            (
                r"\caption{Clean/adversarial detector predictions with "
                r"post-NMS boxes, class labels and confidence scores "
                r"(left/right for each example).}"
                if uses_prediction_overlays
                else r"\caption{Qualitative clean/adversarial pairs "
                r"(left/right for each example).}"
            ),
            r"\label{fig:qualitative_transfer}",
            r"\end{figure*}",
            "",
        ]
    )
    return "\n".join(lines)


def _unique_attack_runs(
    records: Iterable[Mapping[str, Any]], dataset: str
) -> List[Mapping[str, Any]]:
    unique: List[Mapping[str, Any]] = []
    seen = set()
    for record in records:
        if (
            record.get("dataset") != dataset
            or record.get("attack") == "clean"
            or not record.get("adversarial_run")
            or str(record.get("status", "complete")) != "complete"
        ):
            continue
        key = (
            record.get("source"),
            record.get("attack"),
            record.get("study"),
            record.get("variant"),
            record.get("adversarial_run") or record.get("parameters_sha256"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def latex_quality_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    rows_per_table: int = 24,
) -> str:
    """Render page-sized perceptual metrics per adversarial run."""
    unique = _unique_attack_runs(records, dataset)
    unique.sort(
        key=lambda record: _canonical_record_order_key(record, registry)
    )
    rendered_rows: List[str] = []
    for record in unique:
        source = str(record.get("source", ""))
        attack = str(record.get("attack", ""))
        quality = record.get("attack_quality_mean") or {}
        variant = str(record.get("variant", "") or "default")
        rendered_rows.append(
            " & ".join(
                [
                    _escape(
                        registry.model(source).table_code
                        if source in registry.models
                        else source
                    ),
                    _escape(
                        _attack_display_name(
                            registry,
                            attack,
                            str(record.get("budget_profile") or "") or None,
                        )
                    ),
                    _escape(variant),
                    _number(quality.get("mse_pixel")),
                    _number(quality.get("psnr")),
                    _number(quality.get("ssim")),
                    _number(quality.get("linf_normalized")),
                    _number(quality.get("l2_rms_normalized")),
                    _number(quality.get("l0_fraction_gt_half_pixel")),
                    _number(quality.get("mean_abs_normalized")),
                    _number(quality.get("nmse")),
                    _number(quality.get("total_variation_normalized")),
                ]
            )
            + r" \\"
        )
    row_parts = _chunks(rendered_rows, rows_per_table) or [[]]
    caption = "Perceptual quality of generated adversarial examples."
    base_label = "tab:" + dataset + "_perceptual_quality"
    output: List[str] = []
    for part_index, part_rows in enumerate(row_parts, start=1):
        lines = [
            r"\begin{table*}[!htb]",
            r"\centering",
            r"\caption{"
            + _escape(_part_caption(caption, part_index, len(row_parts)))
            + "}",
            r"\label{"
            + _label(_part_label(base_label, part_index))
            + "}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l l l *{9}{c}}",
            r"\toprule",
            r"Source & Attack & Variant & MSE & PSNR & SSIM & $L_\infty$ & RMS-$L_2$ & $L_0$ frac. & Mean $|\Delta|$ & NMSE & TV \\",
            r"\midrule",
            *part_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            r"\end{table*}",
            "",
        ]
        output.extend(lines)
        if part_index != len(row_parts):
            output.extend([r"\clearpage", ""])
    return "\n".join(output)


def latex_attack_success_transfer_rate_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    metric: str = "bbox_mAP_50",
    rows_per_table: int = 24,
) -> str:
    """Derive page-sized ASR/TR tables from matched clean performance."""
    records = list(records)
    clean: Dict[str, float] = {}
    for record in records:
        if (
            record.get("dataset") != dataset
            or record.get("attack") != "clean"
        ):
            continue
        value = (record.get("metrics") or {}).get(metric)
        if value is not None and math.isfinite(float(value)):
            clean[str(record.get("target", ""))] = float(value)

    grouped: Dict[Tuple[str, str, str], Dict[str, float]] = defaultdict(dict)
    profiles: Dict[Tuple[str, str, str], set] = defaultdict(set)
    for record in records:
        if (
            record.get("dataset") != dataset
            or record.get("attack") in {None, "", "clean"}
        ):
            continue
        value = (record.get("metrics") or {}).get(metric)
        target = str(record.get("target", ""))
        clean_value = clean.get(target)
        if (
            value is None
            or clean_value is None
            or clean_value <= 0.0
            or not math.isfinite(float(value))
        ):
            continue
        key = (
            str(record.get("source", "")),
            str(record.get("attack", "")),
            str(record.get("variant", "") or "default"),
        )
        grouped[key][target] = max(
            0.0, (clean_value - float(value)) / clean_value
        )
        if record.get("budget_profile"):
            profiles[key].add(str(record["budget_profile"]))

    source_rank = {
        model_id: index for index, model_id in enumerate(registry.source_ids())
    }
    attack_rank = {
        attack_id: index
        for index, attack_id in enumerate(
            _canonical_attack_ids(registry, dataset)
        )
    }
    ordered_keys = sorted(
        grouped,
        key=lambda key: (
            source_rank.get(key[0], len(source_rank)),
            key[0],
            attack_rank.get(key[1], len(attack_rank)),
            key[1],
            key[2],
        ),
    )
    rendered_rows: List[str] = []
    for key in ordered_keys:
        source, attack, variant = key
        by_target = grouped[key]
        white_asr = by_target.get(source)
        black_values = [
            value for target, value in by_target.items() if target != source
        ]
        black_asr = (
            sum(black_values) / len(black_values) if black_values else None
        )
        transfer_rate = (
            black_asr / white_asr
            if black_asr is not None
            and white_asr is not None
            and white_asr > 0.0
            else None
        )
        profile_values = profiles.get(key, set())
        if len(profile_values) > 1:
            raise ValueError(
                "Do not mix budget profiles in one ASR/TR row: {}".format(key)
            )
        profile = next(iter(profile_values)) if profile_values else None
        rendered_rows.append(
            " & ".join(
                [
                    _escape(
                        registry.model(source).table_code
                        if source in registry.models
                        else source
                    ),
                    _escape(
                        _attack_display_name(registry, attack, profile)
                    ),
                    _escape(variant),
                    _number(white_asr),
                    _number(black_asr),
                    _number(transfer_rate),
                ]
            )
            + r" \\"
        )

    row_parts = _chunks(rendered_rows, rows_per_table) or [[]]
    caption = (
        "Attack success rate (ASR) and black-box transfer rate (TR), "
        "derived from matched clean and attacked detection performance."
    )
    base_label = "tab:" + dataset + "_asr_transfer_rate"
    output: List[str] = []
    for part_index, part_rows in enumerate(row_parts, start=1):
        lines = [
            r"\begin{table*}[!htb]",
            r"\centering",
            r"\caption{"
            + _escape(_part_caption(caption, part_index, len(row_parts)))
            + "}",
            r"\label{"
            + _label(_part_label(base_label, part_index))
            + "}",
            r"\begin{tabular}{l l l c c c}",
            r"\toprule",
            r"Source & Attack & Variant & White-box ASR & Mean black-box ASR & TR \\",
            r"\midrule",
            *part_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
        output.extend(lines)
        if part_index != len(row_parts):
            output.extend([r"\clearpage", ""])
    return "\n".join(output)


def latex_efficiency_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    rows_per_table: int = 24,
    report_scope=None,
) -> str:
    """Render page-sized compute accounting per adversarial run."""
    unique = _unique_attack_runs(records, dataset)
    unique.sort(
        key=lambda record: _canonical_record_order_key(record, registry)
    )
    rendered_rows: List[str] = []
    for record in unique:
        source = str(record.get("source", ""))
        attack = str(record.get("attack", ""))
        auxiliary = record.get("auxiliary_passes") or {}
        variant = str(record.get("variant", "") or "default")
        rendered_rows.append(
            " & ".join(
                [
                    _escape(
                        registry.model(source).table_code
                        if source in registry.models
                        else source
                    ),
                    _escape(
                        _attack_display_name(
                            registry,
                            attack,
                            str(record.get("budget_profile") or "") or None,
                        )
                    ),
                    _escape(variant),
                    _escape(str(record.get("budget_profile") or "unknown")),
                    _number(record.get("gradient_evaluations_per_image")),
                    _number(actual_count(record)),
                    _escape(count_unit(record)),
                    _number(record.get("calibrated_whole_detector_BE")),
                    _number(auxiliary.get("forward_mean")),
                    _number(auxiliary.get("backward_mean")),
                    _number(record.get("attack_runtime_seconds_mean")),
                    _number(record.get("attack_peak_cuda_memory_mb")),
                ]
            )
            + r" \\"
        )
    row_parts = _chunks(rendered_rows, rows_per_table) or [[]]
    caption = "Attack compute and runtime accounting. " + COST_NOTE
    base_label = "tab:" + dataset + "_attack_efficiency"
    output: List[str] = []
    for part_index, part_rows in enumerate(row_parts, start=1):
        lines = [
            r"\begin{table*}[!htb]",
            r"\centering",
            r"\caption{"
            + _escape(_part_caption(caption, part_index, len(row_parts)))
            + "}",
            r"\label{"
            + _label(_part_label(base_label, part_index))
            + "}",
            r"\resizebox{\textwidth}{!}{",
            r"\begin{tabular}{l l l l c c l c c c c c}",
            r"\toprule",
            r"Source & Attack & Variant & Declared profile & Declared budget & Observed count & Count unit & Calibrated BE & Aux fwd. & Aux bwd. & Seconds/image & Peak MiB \\",
            r"\midrule",
            *part_rows,
            r"\bottomrule",
            r"\end{tabular}",
            "}",
            r"\end{table*}",
            "",
        ]
        output.extend(lines)
        if part_index != len(row_parts):
            output.extend([r"\clearpage", ""])
    return scoped_tex("\n".join(output), report_scope or inferred_scope(unique))


def write_analysis_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    dataset: str,
    include_qualitative_payload_pairs: bool = True,
    *,
    split: Optional[str] = None,
    report_scope=None,
    include_saved_visualizations: bool = True,
) -> Dict[str, Path]:
    records = [
        record for record in records
        if record.get("dataset") == dataset
        and (split is None or record.get("split") == split)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = output_dir / "analysis.tex"
    categories = output_dir / "per_category_ap.tex"
    families = output_dir / "target_family.tex"
    qualitative = output_dir / "qualitative.tex"
    quality = output_dir / "perceptual_quality.tex"
    efficiency = output_dir / "runtime_efficiency.tex"
    success = output_dir / "attack_success_transfer_rate.tex"
    analysis.write_text(latex_analysis_table(records, registry, dataset), encoding="utf-8")
    categories.write_text(latex_category_table(records, registry, dataset), encoding="utf-8")
    families.write_text(latex_family_table(records, registry, dataset), encoding="utf-8")
    qualitative.write_text(
        (latex_qualitative_figure(
            records, allow_payload_pairs=include_qualitative_payload_pairs,
        ) if include_saved_visualizations else "% Saved image/prediction payloads not read.\n"),
        encoding="utf-8",
    )
    quality.write_text(latex_quality_table(records, registry, dataset), encoding="utf-8")
    efficiency.write_text(latex_efficiency_table(records, registry, dataset), encoding="utf-8")
    success.write_text(
        latex_attack_success_transfer_rate_table(records, registry, dataset),
        encoding="utf-8",
    )
    report_scope = report_scope or inferred_scope(records)
    if scope_label(report_scope):
        for path in (analysis, categories, families, qualitative, quality, efficiency, success):
            path.write_text(scoped_tex(path.read_text(encoding="utf-8"), report_scope),
                            encoding="utf-8")
    return {
        "analysis_tex": analysis,
        "category_tex": categories,
        "family_tex": families,
        "qualitative_tex": qualitative,
        "quality_tex": quality,
        "efficiency_tex": efficiency,
        "success_tex": success,
    }


def latex_ablation_table(
    records: Iterable[Mapping[str, Any]],
    registry: Registry,
    dataset: str,
    study: str,
    metric: str = "bbox_mAP_50",
    caption: Optional[str] = None,
    target_ids: Optional[Sequence[str]] = None,
) -> str:
    """Render one study with variants as rows and target detectors as columns."""
    records = list(records)
    present_targets = {
        str(record.get("target", ""))
        for record in records
        if record.get("dataset") == dataset and record.get("study") == study
    }
    present_sources = {
        str(record.get("source", ""))
        for record in records
        if record.get("dataset") == dataset
        and record.get("study") == study
        and str(record.get("source", "")) in registry.models
    }
    source_id = next(iter(present_sources)) if len(present_sources) == 1 else None
    target_ids = _canonical_model_selection(
        registry,
        list(
            target_ids
            or [target for target in registry.target_ids() if target in present_targets]
            or registry.target_ids()
        ),
    )
    variant_order: List[str] = []
    values: Dict[Tuple[str, str], Optional[float]] = {}
    for record in records:
        if record.get("dataset") != dataset or record.get("study") != study:
            continue
        variant = str(record.get("variant", ""))
        target = str(record.get("target", ""))
        if variant and variant not in variant_order:
            variant_order.append(variant)
        value = (record.get("metrics") or {}).get(metric)
        values[(variant, target)] = float(value) if value is not None else None

    best: Dict[str, float] = {}
    for target in target_ids:
        candidates = [
            values[(variant, target)]
            for variant in variant_order
            if values.get((variant, target)) is not None
        ]
        if candidates:
            best[target] = min(float(value) for value in candidates)

    study_spec = registry.ablation_studies.get(study, {})
    method_id = str(study_spec.get("method", ""))
    method_name = (
        registry.attack(method_id).display_name
        if method_id in registry.attacks
        else "Attack"
    )
    title = caption or "{} ablation: {} (lower is better).".format(
        method_name, study
    )
    columns = "l " + " ".join("c" for _ in target_ids) + " c"
    lines = [
        r"\begin{table*}[!htb]",
        r"\centering",
        r"\caption{" + _escape(title) + "}",
        r"\label{" + _label("tab:" + method_id + "_" + study) + "}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        *_latex_group_header(
            registry,
            target_ids,
            prefix_columns=1,
            trailing_columns=1,
        ),
        "Variant & "
        + " & ".join(_latex_model_code(registry, target) for target in target_ids)
        + (r" & BB Mean \\" if source_id is not None else r" & Mean \\"),
        r"\midrule",
    ]
    for variant in variant_order:
        cells: List[str] = []
        numeric: List[float] = []
        for target in target_ids:
            value = values.get((variant, target))
            if value is not None and target != source_id:
                numeric.append(float(value))
            is_best = (
                value is not None
                and target in best
                and abs(float(value) - best[target]) < 5.0e-7
            )
            cells.append(_number(value, bold=is_best))
        mean_value = sum(numeric) / len(numeric) if numeric else None
        lines.append(
            _escape(variant)
            + " & "
            + " & ".join(cells)
            + " & "
            + _number(mean_value)
            + r" \\"
        )
    role_note = _latex_role_note(
        registry,
        target_ids,
        total_columns=1 + len(target_ids) + 1,
        include_blackbox_mean=source_id is not None,
    )
    if role_note:
        lines.extend([r"\midrule", role_note])
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_ablation_reports(
    records: Sequence[Mapping[str, Any]],
    registry: Registry,
    output_dir: Path,
    dataset: str,
    metric: str = "bbox_mAP_50",
    report_scope=None,
) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    studies: List[str] = []
    for record in records:
        study = str(record.get("study", ""))
        if record.get("dataset") == dataset and study and study not in studies:
            studies.append(study)
    for study in studies:
        path = output_dir / "ablation_{}_{}.tex".format(study, metric)
        path.write_text(
            scoped_tex(latex_ablation_table(records, registry, dataset, study, metric=metric),
                       report_scope or inferred_scope(records)),
            encoding="utf-8",
        )
        paths["tex_{}".format(study)] = path

    csv_path = output_dir / "ablation_records.csv"
    fields = [
        "dataset",
        "source",
        "study",
        "variant",
        "target",
        "status",
        "parameters",
        "bbox_mAP",
        "bbox_mAP_50",
        "bbox_mAP_75",
        "reason",
    ]
    report_scope = report_scope or inferred_scope(records)
    scoped = bool(scope_label(report_scope))
    if scoped:
        fields.extend(SCOPE_FIELDS)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            if record.get("dataset") != dataset or not record.get("study"):
                continue
            metrics = record.get("metrics") or {}
            writer.writerow(
                {
                    "dataset": record.get("dataset"),
                    "source": record.get("source"),
                    "study": record.get("study"),
                    "variant": record.get("variant"),
                    "target": record.get("target"),
                    "status": record.get("status", "ok"),
                    "parameters": json.dumps(
                        record.get("parameters") or {}, sort_keys=True
                    ),
                    "bbox_mAP": format_metric(metrics.get("bbox_mAP")),
                    "bbox_mAP_50": format_metric(
                        metrics.get("bbox_mAP_50")
                    ),
                    "bbox_mAP_75": format_metric(
                        metrics.get("bbox_mAP_75")
                    ),
                    "reason": record.get("reason", ""),
                    **(scope_csv(scope_for_record(record, report_scope)) if scoped else {}),
                }
            )
    paths["csv"] = csv_path
    return paths
