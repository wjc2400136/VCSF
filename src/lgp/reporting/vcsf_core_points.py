"""Core factorial arithmetic; given numbers are not accepted prediction evidence."""
from copy import deepcopy
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import re

from ..io import atomic_json, file_digest
from ..metrics import COCO_BBOX_METRICS
from ..runners.vcsf_research_plan import canonical_hash
from .vcsf_paired_statistics import paired_panel_scalars, require


CONTEXT_FIELDS = (
    "science_sha256", "current_stage_identity_sha256", "working_baseline",
    "nonzero_comparator", "working_is_identity", "working_stage_parameters_sha256",
    "selected_working_variant", "factorial_high_variant", "comparator_role",
    "identity_context_projection",
)
STATUS = "core_point_arithmetic_only_pending_input_and_independent_acceptance"
SCOPE = "Arithmetic Only / Input Provenance Pending"
SCHEDULE_FIELDS = ("logical_gradients", "risk_updates", "feature_updates", "feature_terms")


def _same(left, right):
    return canonical_hash(left) == canonical_hash(right)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _stage_view(stage, target_order, max_images):
    require(max_images is None or type(max_images) is int and max_images == 1,
        "Core arithmetic accepts full scope or an explicitly prepared one-image diagnostic")
    images = 5000 if max_images is None else 1
    require(isinstance(stage, dict) and stage.get("record_type") == "vcsf_ablation_stage_preparation"
        and stage.get("protocol") == "vcsf_single_source_ablation_research"
        and stage.get("selected_stage") == "core"
        and stage.get("dataset") == "coco" and stage.get("split") == "val"
        and _same(stage.get("seeds"), [42])
        and _same(stage.get("max_images"), max_images)
        and stage.get("diagnostic_only") is (max_images is not None)
        and type(stage.get("synthetic")) is bool,
        "Not the original prepared core scope")
    content = {k: v for k, v in stage.items() if k not in
        ("science_sha256", "requested_devices", "device_availability_checked", "lane_plan")}
    require(_sha(stage.get("science_sha256")) and canonical_hash(content) == stage["science_sha256"],
        "Original stage content differs from its scientific identity")
    require(isinstance(target_order, list) and len(target_order) == len(set(target_order)) == 16
        and stage.get("targets") == target_order and stage.get("sources") == target_order[:1],
        "Core arithmetic requires the registry's canonical sixteen targets and first source")
    groups = stage["groups"]
    require(isinstance(groups, list) and _same([g.get("group_id") for g in groups], list(range(1, 9)))
        and len({g["variant"] for g in groups}) == len({g["parameters_sha256"] for g in groups}) == 8
        and all(type(g.get("images")) is int and g["images"] == images
            and type(g.get("seed")) is int and g["seed"] == 42
            and g.get("source") == target_order[0] and g.get("targets") == target_order
            and _sha(g.get("parameters_sha256"))
            and canonical_hash(g.get("parameters")) == g["parameters_sha256"] for g in groups),
        "Core group identity, full configuration or image scope differs")
    require(stage.get("selected_working_variant") in {g["variant"] for g in groups}
        and stage.get("factorial_high_variant") in {g["variant"] for g in groups}
        and all(k in stage for k in CONTEXT_FIELDS), "Missing original working/comparator roles")
    for group in groups:
        parameters = group["parameters"]
        require(type(parameters.get("iterations")) is int and parameters["iterations"] > 0
            and type(parameters.get("levels_per_stage")) is int and parameters["levels_per_stage"] > 0,
            "Original schedule must declare integer iterations and feature levels")
        risk = int(parameters["initialization"] == "detector")
        schedule = dict(logical_gradients=parameters["iterations"], risk_updates=risk,
            feature_updates=parameters["iterations"] - risk,
            feature_terms=parameters["levels_per_stage"] * (2 if parameters["surface"] == "cross_stage" else 1))
        require(_same({name: group.get(name) for name in SCHEDULE_FIELDS}, schedule),
            "Original planned gradient schedule or feature-term count differs")
    analysis = stage["analysis"]
    require(analysis.get("mode") == "point_estimate_only" and analysis.get("metric") == "bbox_mAP"
        and type(analysis.get("family_size")) is int and analysis["family_size"] == 32
        and len(analysis.get("contrasts", [])) == 32
        and len({c["id"] for c in analysis["contrasts"]}) == 32
        and analysis.get("uncertainty", {}).get("bootstrap_required") is False
        and analysis.get("uncertainty", {}).get("significance_claim") is False
        and analysis.get("independent_confirmation") is False,
        "Original exact descriptive core family is missing or changed")
    return dict(sources=stage["sources"], targets=target_order, groups=groups,
        blackbox_cells_per_configuration_seed=len(target_order) - 1), images


def assemble_core_point_matrix(stage_plan, normalized_cells, *, target_order, max_images=None):
    """Compute the original family without compiling a new stage or qualifying inputs."""
    stage = stage_plan
    view, images = _stage_view(stage, target_order, max_images)
    require(isinstance(normalized_cells, list), "Normalized cells must be the complete original list")
    panels = paired_panel_scalars(view, normalized_cells)
    image_hashes = {c.get("image_ids_sha256") for c in normalized_cells}
    require(len(image_hashes) == 1 and all(_sha(h) for h in image_hashes),
        "Core contrasts require one common bound ordered image identity")
    for cell in normalized_cells:
        require(_sha(cell.get("input_evidence_sha256")) and isinstance(cell.get("metrics"), dict)
            and set(cell["metrics"]) == set(COCO_BBOX_METRICS), "Missing full normalized input metrics")
        require(all(type(v) in (int, float) and math.isfinite(v) and (v == -1 or 0 <= v <= 1)
            for v in cell["metrics"].values())
            and type(cell.get("bbox_mAP")) in (int, float)
            and cell["bbox_mAP"] == cell["metrics"]["bbox_mAP"] >= 0,
            "Core metrics must preserve raw values and all twelve summaries")
        require(all(type(cell[name]) is bool for name in ("diagnostic_only", "formal_metrics_eligible") if name in cell),
            "Optional cell eligibility markers must be explicit booleans")
        if max_images is None:
            require(cell.get("diagnostic_only") is not True
                and cell.get("formal_metrics_eligible") is not False
                and cell.get("observation_origin") != "core_executor_smoke_diagnostic",
                "Diagnostic evidence cannot be relabelled as full-size core arithmetic input")
    lookup = {(c["group_id"], c["target"]): c for c in normalized_cells}
    groups = {(g["variant"], g["seed"]): g for g in stage["groups"]}
    configurations = [dict(group_id=g["group_id"], variant=g["variant"], source=g["source"],
        seed=g["seed"], parameters_sha256=g["parameters_sha256"],
        BB_mean=panels[g["variant"], g["seed"]]["mean"],
        whitebox_AP=lookup[g["group_id"], g["source"]]["bbox_mAP"],
        **{"planned_" + name: g[name] for name in SCHEDULE_FIELDS}) for g in stage["groups"]]
    contrasts = []
    for declaration in stage["analysis"]["contrasts"]:
        terms = declaration["terms"]
        keys = [(t["variant"], t["seed"]) for t in terms]
        require(keys and len(keys) == len(set(keys)) and set(keys) <= set(groups)
            and all(type(t["seed"]) is int and isinstance(t["coefficient"], str) for t in terms),
            "Core contrast has missing, duplicate or invalid terms")
        coefficients = [Fraction(t["coefficient"]) for t in terms]
        require(all(coefficients) and sum(coefficients) == 0,
            "Core contrast must be a nonzero signed difference")
        require(not {"estimate", "per_target", "inference"}.intersection(declaration),
            "A contrast declaration contains computed result fields")
        per_target = []
        for target in target_order:
            metrics = {}
            for metric in COCO_BBOX_METRICS:
                values = [lookup[groups[key]["group_id"], target]["metrics"][metric] for key in keys]
                metrics[metric] = None if any(v == -1 for v in values) else math.fsum(
                    float(coefficient) * value for coefficient, value in zip(coefficients, values))
            per_target.append(dict(target=target, source_matched=target == stage["sources"][0],
                estimate=metrics["bbox_mAP"], metrics=metrics))
        bb = [r["estimate"] for r in per_target if not r["source_matched"]]
        estimate = math.fsum(bb) / len(bb)
        scalar = math.fsum(float(c) * panels[k]["mean"] for c, k in zip(coefficients, keys))
        require(abs(estimate - scalar) <= 1e-12, "Core paired target and BB15 arithmetic disagree")
        contrasts.append(dict(deepcopy(declaration), estimate=estimate, per_target=per_target,
            inference="descriptive_only_no_significance_claim"))
    ordered = [deepcopy(lookup[g["group_id"], target]) for g in stage["groups"] for target in target_order]
    return dict(schema_version=1, record_type="vcsf_core_point_arithmetic", protocol=stage["protocol"],
        stage="core", status=STATUS, stage_plan_content_sha256=canonical_hash(stage),
        input_cells_content_sha256=canonical_hash(normalized_cells),
        contrast_family_sha256=canonical_hash(stage["analysis"]["contrasts"]),
        stage_context={k: deepcopy(stage[k]) for k in CONTEXT_FIELDS},
        synthetic=stage["synthetic"], max_images=max_images, diagnostic_only=max_images is not None,
        images=images, image_ids_sha256=next(iter(image_hashes)), groups=8, cells=128,
        sources=deepcopy(stage["sources"]), targets=list(target_order), normalized_cells=ordered,
        planned_groups=deepcopy(stage["groups"]), planned_schedule_scope="declarations_not_measured_cost",
        physical_cost_status="not_assessed",
        configuration_seed=configurations, contrasts=contrasts, input_provenance_acceptance="pending",
        independent_result_acceptance=False, formal_metrics_eligible=False, scientific_acceptance=False,
        independent_confirmation=False, automatic_promotion=False, bootstrap_replicates_computed=0,
        confidence_intervals_produced=False, significance_claimed=False, source_images_verified=False,
        prediction_archives_replayed=False, original_cells_modified=False,
        limitations=["Arithmetic only: original images, predictions, metrics and provenance are not verified.",
            "A complete numeric grid is not full-COCO execution, independent result acceptance or formal admission.",
            "Single-source reused-validation contrasts are descriptive, not significance or confirmation."])


def _display(value):
    return "NA" if value is None else format(value, ".4f")


def _tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def write_core_point_reports(directory, result, *, claimed_empty_directory=False):
    """Write a fresh arithmetic-only report; publication does not promote its inputs."""
    require(result.get("status") == STATUS and result.get("input_provenance_acceptance") == "pending"
        and all(result.get(k) is False for k in ("formal_metrics_eligible", "independent_result_acceptance",
            "scientific_acceptance", "independent_confirmation", "automatic_promotion")),
        "Only a pending-input arithmetic result can be exported")
    directory = Path(directory)
    if claimed_empty_directory:
        require(directory.is_dir() and not directory.is_symlink() and not any(directory.iterdir()),
            "The coordinator-claimed report directory must still be empty")
    else:
        directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "analysis.json", result)
    declarations = [{k: deepcopy(v) for k, v in c.items() if k not in ("estimate", "per_target", "inference")}
        for c in result["contrasts"]]
    atomic_json(directory / "contrast_definitions.json", declarations)
    metric_rows = [dict(group_id=c["group_id"], variant=c["variant"], seed=c["seed"], source=c["source"],
        target=c["target"], metric=metric, value=_display(c["metrics"][metric]))
        for c in result["normalized_cells"] for metric in COCO_BBOX_METRICS]
    configuration_rows = [dict(row, BB_mean=_display(row["BB_mean"]), whitebox_AP=_display(row["whitebox_AP"]))
        for row in result["configuration_seed"]]
    contrast_rows = [dict(id=c["id"], block=c["block"], kind=c["kind"], estimate=_display(c["estimate"]),
        declaration_json=json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False))
        for c, declaration in zip(result["contrasts"], declarations)]
    target_rows = [dict(contrast=c["id"], target=t["target"], source_matched=t["source_matched"],
        metric=metric, value=_display(t["metrics"][metric]))
        for c in result["contrasts"] for t in c["per_target"] for metric in COCO_BBOX_METRICS]
    for name, fields, rows in (
            ("metrics.csv", ["group_id", "variant", "seed", "source", "target", "metric", "value"], metric_rows),
            ("configuration_seed.csv", ["group_id", "variant", "source", "seed", "parameters_sha256",
                *["planned_" + name for name in SCHEDULE_FIELDS], "BB_mean", "whitebox_AP"], configuration_rows),
            ("contrasts.csv", ["id", "block", "kind", "estimate", "declaration_json"], contrast_rows),
            ("target_contrasts.csv", ["contrast", "target", "source_matched", "metric", "value"], target_rows)):
        with (directory / name).open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    for name, header, rows in (
            ("contrasts.tex", "Contrast & BB Mean Difference", [(c["id"], _display(c["estimate"])) for c in result["contrasts"]]),
            ("metrics.tex", "Group / Target / Metric & Value", [
                (str(r["group_id"]) + " / " + r["target"] + " / " + r["metric"], r["value"]) for r in metric_rows])):
        lines = ["% " + SCOPE + "; descriptive values, no scientific acceptance.",
            r"\begin{longtable}{lr}", header + r" \\", r"\hline"]
        lines.extend(_tex(label) + " & " + value + r" \\" for label, value in rows)
        lines.append(r"\end{longtable}")
        (directory / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "report_scope.txt").write_text(SCOPE + "\n"
        + "Synthetic stage: " + str(result["synthetic"]) + "\n"
        + "Diagnostic only: " + str(result["diagnostic_only"]) + "\n"
        + "The report does not verify images, predictions, official metric replay or original provenance.\n"
        + "No formal result slot, execution admission, scientific acceptance or method promotion is issued.\n",
        encoding="utf-8")
    manifest = {p.name: file_digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    atomic_json(directory / "manifest.json", manifest)
    return manifest
