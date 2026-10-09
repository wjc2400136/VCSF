"""Independent core point arithmetic; never accepts input provenance or results."""
from copy import deepcopy
import csv
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
from itertools import combinations, product
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lgp.metrics import COCO_BBOX_METRICS
from lgp.reporting.vcsf_analysis_inputs import canonical_hash


PROTOCOL = "vcsf_single_source_ablation_research"
STATUS = "core_point_arithmetic_only_pending_input_and_independent_acceptance"
INFERENCE = "descriptive_only_no_significance_claim"
TOLERANCE = Decimal("1e-12")
CONTEXT_FIELDS = (
    "science_sha256", "current_stage_identity_sha256", "working_baseline",
    "nonzero_comparator", "working_is_identity", "working_stage_parameters_sha256",
    "selected_working_variant", "factorial_high_variant", "comparator_role",
    "identity_context_projection",
)
FALSE_FLAGS = (
    "independent_result_acceptance", "formal_metrics_eligible", "scientific_acceptance",
    "independent_confirmation", "automatic_promotion", "confidence_intervals_produced",
    "significance_claimed", "source_images_verified", "prediction_archives_replayed",
    "original_cells_modified",
)
PLANNED_FIELDS = (
    "planned_logical_gradients", "planned_risk_updates", "planned_feature_updates", "planned_feature_terms",
)
MAX_EXPORT_FILE_BYTES = 128 << 20


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, label):
    _check(canonical_hash(actual) == canonical_hash(expected), label + " differs")


def _sha(value):
    return (type(value) is str and len(value) == 64
        and all(c in "0123456789abcdef" for c in value))


def _number(value, label, metric=False):
    _check(type(value) in (int, float) and math.isfinite(value), "Invalid number: " + label)
    if metric:
        _check(value == -1 or 0 <= value <= 1, "Not a raw COCO metric: " + label)
    return Decimal.from_float(value) if type(value) is float else Decimal(value)


def _near(actual, expected, label):
    difference = abs(_number(actual, label) - expected)
    _check(difference <= TOLERANCE, "Independent arithmetic differs: " + label)
    return difference


def _weighted(values, terms):
    total = Decimal(0)
    for value, term in zip(values, terms):
        weight = Fraction(term["coefficient"])
        total += _number(value, "contrast input") * Decimal(weight.numerator) / Decimal(weight.denominator)
    return total


def _planned_schedule(group):
    parameters = group["parameters"]
    iterations, levels = parameters["iterations"], parameters["levels_per_stage"]
    _check(type(iterations) is int and iterations > 0 and type(levels) is int and levels > 0,
        "Planned iteration and level counts must be positive integers")
    risk = int(parameters["initialization"] == "detector")
    values = (iterations, risk, iterations - risk, levels * (2 if parameters["surface"] == "cross_stage" else 1))
    for field, value in zip(PLANNED_FIELDS, values):
        _same(group[field[len("planned_"):]], value, "Original planned schedule " + field)
    return dict(zip(PLANNED_FIELDS, values))


def _registered_design(stage, groups):
    """Resolve only the original fragment, never the current registry/compiler."""
    fragment = stage["selected_fragment"]
    _check(type(fragment) is dict and _sha(stage["selected_fragment_sha256"]), "Missing original selected fragment")
    _same(canonical_hash(fragment), stage["selected_fragment_sha256"], "Original selected fragment hash")
    _check(set(fragment["blocks"]) == {"core"} and fragment["execution_order"] == ["core"],
        "Selected fragment contains another stage")
    block, policy = fragment["blocks"]["core"], fragment["analysis_policy"]
    variants = block["variants"]
    _check(type(variants) is list and len(variants) == 8
        and all(type(name) is str and name for name in variants) and len(set(variants)) == 8,
        "Original core fragment needs eight unique ordered variants")
    _same([g["variant"] for g in groups], variants, "Original core variant order")
    _check(set(fragment["variants"]) == set(variants), "Core fragment has missing or extra dependencies")
    _same(block["seeds"], [42], "Original block seeds")
    _same(policy["primary_seed"], 42, "Original anchor seed")
    _check(policy["metric"] == "bbox_mAP" and policy["scalar"] == "equal_weight_source_excluded_cells"
        and policy["anchor_comparisons"] == "every_nonanchor_variant_at_primary_seed"
        and policy["additional_pairs"] == [], "Original core comparison policy differs")
    factorials = policy["factorials"]
    _check(type(factorials) is list and len(factorials) == 1 and factorials[0]["block"] == "core",
        "Core requires exactly one original factorial")
    spec = factorials[0]
    _same(spec["seeds"], [42], "Original factorial seeds")
    fields_by_axis = spec["axes"]
    _check(type(fields_by_axis) is dict and len(fields_by_axis) == 3
        and all(type(name) is str and name and type(fields) is list and fields
            and all(type(field) is str and field for field in fields) for name, fields in fields_by_axis.items()),
        "Original core factorial requires three declared axes")
    fields = [field for axis_fields in fields_by_axis.values() for field in axis_fields]
    allowed = block["allowed_fields"]
    _check(type(allowed) is list and len(allowed) == len(set(allowed))
        and len(fields) == len(set(fields)) and set(fields) == set(allowed),
        "Factorial field sets overlap or differ from the original allowed fields")
    # The declared field list survives JSON object-key sorting and fixes axis order.
    axes = sorted(fields_by_axis, key=lambda axis: min(allowed.index(field) for field in fields_by_axis[axis]))
    by_name = {g["variant"]: g for g in groups}
    low_name, high_name, anchor = spec["low"], spec["high"], policy["anchor"]
    _check(low_name in by_name and high_name in by_name and low_name != high_name and anchor in by_name,
        "Original factorial endpoints or anchor are missing")
    _same(anchor, stage["selected_working_variant"], "Original working anchor")
    _same(high_name, stage["factorial_high_variant"], "Original factorial high role")
    low, high = by_name[low_name]["parameters"], by_name[high_name]["parameters"]
    _check(set(low) == set(high) and set(fields) <= set(low), "Original endpoint parameter fields differ")
    _same({k: v for k, v in low.items() if k not in fields},
        {k: v for k, v in high.items() if k not in fields}, "Non-factor endpoint parameters")
    _check(all(any(low[field] != high[field] for field in fields_by_axis[axis]) for axis in axes),
        "An original factorial axis has collapsed")
    grid = {}
    for bits in product((0, 1), repeat=3):
        parameters = deepcopy(low)
        for axis, bit in zip(axes, bits):
            for field in fields_by_axis[axis]:
                parameters[field] = (high if bit else low)[field]
        matches = [name for name in variants if canonical_hash(by_name[name]["parameters"]) == canonical_hash(parameters)]
        _check(len(matches) == 1 and matches[0] not in grid, "Original factorial cell is missing or ambiguous")
        grid[matches[0]] = bits
    _check(set(grid) == set(variants), "Original factorial does not cover all eight groups")
    return dict(axes=axes, grid=grid, variants=variants, low=low_name, high=high_name, anchor=anchor)


def _declarations(stage, design):
    """Reconstruct the registered questions without importing their compiler."""
    expected = []
    axes, grid = design["axes"], design["grid"]
    for order in (1, 2, 3):
        for active in combinations(range(3), order):
            inactive = tuple(i for i in range(3) if i not in active)
            conditions = [None] + (list(product((0, 1), repeat=len(inactive))) if inactive else [])
            for condition in conditions:
                conditioning = {} if condition is None else dict(zip((axes[i] for i in inactive), condition))
                suffix = "averaged" if condition is None else "_".join(
                    "{}{}".format(axis, bit) for axis, bit in conditioning.items())
                terms = []
                for variant, bits in sorted(grid.items()):
                    if condition is not None and tuple(bits[i] for i in inactive) != condition:
                        continue
                    numerator = -1 if sum(bits[i] == 0 for i in active) % 2 else 1
                    denominator = 2 ** len(inactive) if condition is None else 1
                    terms.append(dict(variant=variant, seed=42,
                        coefficient=str(Fraction(numerator, denominator))))
                expected.append(dict(id="core:{}:{}".format("*".join(axes[i] for i in active), suffix),
                    block="core", kind="factorial_difference", factors=[axes[i] for i in active],
                    conditioning=conditioning, averaged_over=[axes[i] for i in inactive] if condition is None else [],
                    averaged_seeds=[42], low_variant=design["low"], high_variant=design["high"], terms=terms))
    anchor = design["anchor"]
    for variant in design["variants"]:
        if variant != anchor:
            expected.append(dict(id="anchor:" + variant, block="anchor", kind="candidate_minus_reference",
                candidate=variant, reference=anchor, averaged_seeds=[42],
                terms=[dict(variant=name, seed=42, coefficient="1" if name == variant else "-1")
                    for name in sorted((variant, anchor))]))
    analysis = stage["analysis"]
    _same(analysis["family_size"], 32, "Core family size")
    _check(analysis["metric"] == "bbox_mAP" and analysis["mode"] == "point_estimate_only"
        and analysis["all_twelve_target_metrics_required"] is True
        and analysis["independent_confirmation"] is False
        and analysis["uncertainty"] == dict(status="not_requested_not_computed",
            bootstrap_required=False, significance_claim=False), "Core analysis scope differs")
    declared = analysis["contrasts"]
    _check(type(declared) is list and len(declared) == 32, "Incomplete core declaration family")
    for actual, row in zip(declared, expected):
        _check(type(actual) is dict and set(row) <= set(actual), "Missing contrast metadata")
        _same({key: actual[key] for key in row}, row, "Registered contrast " + row["id"])
        _check(not {"estimate", "per_target", "inference"} & set(actual), "Declaration contains result fields")
    return declared


def _stage(stage, target_order, max_images):
    _check(max_images is None or type(max_images) is int and max_images == 1,
        "Only full 5000-image scope or explicit one-image arithmetic diagnostics are supported")
    _check(type(stage) is dict and stage.get("protocol") == PROTOCOL
        and stage.get("selected_stage") == "core", "Not the registered core stage")
    _same(stage["schema_version"], 1, "Stage schema")
    _check(stage["record_type"] == "vcsf_ablation_stage_preparation", "Wrong stage record type")
    excluded = {"science_sha256", "requested_devices", "device_availability_checked", "lane_plan"}
    _check(_sha(stage["science_sha256"]), "Missing stage science hash")
    _same(stage["science_sha256"], canonical_hash({k: v for k, v in stage.items() if k not in excluded}),
        "Complete stage scientific content hash")
    _check(type(target_order) in (list, tuple) and len(target_order) == 16
        and all(type(t) is str and t for t in target_order) and len(set(target_order)) == 16,
        "Canonical target_order must contain sixteen unique targets")
    targets = list(target_order)
    _same(stage["targets"], targets, "Canonical target panel")
    _same(stage["sources"], targets[:1], "Single source must be the first canonical target")
    _check(stage["sources"] == ["faster_rcnn_r50"] and stage["dataset"] == "coco"
        and stage["split"] == "val", "Wrong source or dataset scope")
    _same(stage["seeds"], [42], "Stage seeds")
    _same(stage["max_images"], max_images, "Stage image mode")
    _same(stage["diagnostic_only"], max_images is not None, "Stage diagnostic mode")
    _check(type(stage["synthetic"]) is bool and type(stage["working_is_identity"]) is bool,
        "Ambiguous stage origin or working role")
    for key in ("scientific_acceptance", "independent_confirmation", "automatic_promotion", "runner_armed"):
        _check(stage[key] is False, "Stage preparation promoted: " + key)
    for key in CONTEXT_FIELDS:
        _check(key in stage, "Missing stage context: " + key)
    groups = stage["groups"]
    _check(type(groups) is list and len(groups) == 8, "Core needs all eight groups")
    design = _registered_design(stage, groups)
    images = 5000 if max_images is None else 1
    for index, (group, variant) in enumerate(zip(groups, design["variants"]), 1):
        for field, value in dict(group_id=index, variant=variant, source=targets[0], seed=42,
                images=images, targets=targets, source_matched_target=targets[0], blackbox_targets=targets[1:]).items():
            _same(group[field], value, "Core group " + field)
        _same(group["parameters_sha256"], canonical_hash(group["parameters"]), "Group parameter hash")
        _planned_schedule(group)
    _check(len({g["parameters_sha256"] for g in groups}) == 8, "Collapsed core factorial")
    by_name = {g["variant"]: g for g in groups}
    anchor, high = design["anchor"], by_name[design["high"]]["parameters"]
    identity = stage["working_is_identity"]
    working = stage["working_stage_parameters"]
    _same(identity, working["scale_min"] == working["scale_max"] == 1, "Working identity-scale role")
    _check(high["scale_min"] < 1 < high["scale_max"], "Core high endpoint is not the nonzero comparator")
    _check(stage["comparator_role"] == ("retained_nonzero_rejection_deletion_comparator"
        if identity else "same_selected_identity"), "Working comparator role differs")
    expected_working = dict(high, scale_min=1.0, scale_max=1.0) if identity else high
    _same(working, expected_working, "Working/high parameter relationship")
    _check((anchor != design["high"]) is identity, "Working/high group identity differs")
    _same(stage["working_stage_parameters_sha256"], by_name[anchor]["parameters_sha256"], "Working endpoint hash")
    _same(stage["working_stage_parameters"], by_name[anchor]["parameters"], "Working endpoint parameters")
    _same(stage["nonzero_comparator"]["parameters"], high, "Nonzero comparator parameters")
    return groups, targets, images, _declarations(stage, design)


def _scope(result):
    _check(type(result) is dict, "Point result must be an object")
    _check(result.get("status") == STATUS and result.get("record_type") == "vcsf_core_point_arithmetic",
        "Wrong arithmetic-only result identity")
    _same(result["schema_version"], 1, "Report schema")
    _check(result["protocol"] == PROTOCOL and result["stage"] == "core", "Wrong report protocol")
    _check(result["input_provenance_acceptance"] == "pending", "Input provenance was promoted")
    _check(result.get("planned_schedule_scope") == "declarations_not_measured_cost"
        and result.get("physical_cost_status") == "not_assessed", "Planned schedules were promoted to measured cost")
    for key in FALSE_FLAGS:
        _check(result.get(key) is False, "Report promoted or changed scope: " + key)
    _same(result["bootstrap_replicates_computed"], 0, "Bootstrap count")
    required = {"schema_version", "record_type", "protocol", "stage", "status", "stage_plan_content_sha256",
        "input_cells_content_sha256", "contrast_family_sha256", "stage_context", "synthetic", "max_images",
        "diagnostic_only", "images", "groups", "cells", "sources", "targets", "normalized_cells",
        "configuration_seed", "contrasts", "image_ids_sha256", "input_provenance_acceptance",
        "bootstrap_replicates_computed", "planned_groups", "planned_schedule_scope", "physical_cost_status"} | set(FALSE_FLAGS)
    metadata = {"limitations", "input_files", "source_sha256", "environment"}
    _check(required <= set(result) <= required | metadata, "Missing or unregistered top-level report fields")
    if "limitations" in result:
        _check(type(result["limitations"]) is list and all(type(s) is str for s in result["limitations"]),
            "Report limitations must be text")
    for key in ("input_files", "source_sha256", "environment"):
        if key in result:
            _check(type(result[key]) is dict, "Report metadata must be an object: " + key)


def verify_core_point_matrix(stage_plan, cells, result, *, target_order, max_images=None):
    """Recompute data-only arithmetic; caller-supplied bytes are not provenance proof."""
    groups, targets, images, declarations = _stage(stage_plan, target_order, max_images)
    _scope(result)
    _same(result["stage_plan_content_sha256"], canonical_hash(stage_plan), "Consumed stage content")
    _same(result["input_cells_content_sha256"], canonical_hash(cells), "Original-order input cell content")
    _same(result["contrast_family_sha256"], canonical_hash(declarations), "Full declaration family")
    _same(result["stage_context"], {k: deepcopy(stage_plan[k]) for k in CONTEXT_FIELDS}, "Complete stage roles/context")
    _same(result["planned_groups"], groups, "Complete original planned groups")
    for key, value in dict(synthetic=stage_plan["synthetic"], max_images=max_images,
            diagnostic_only=max_images is not None, images=images, groups=8, cells=128,
            sources=stage_plan["sources"], targets=targets).items():
        _same(result[key], value, "Report " + key)
    _check(type(cells) is list and len(cells) == 128, "Exactly 128 normalized input cells are required")
    expected_keys = [(g["group_id"], t) for g in groups for t in targets]
    planned = {g["group_id"]: g for g in groups}
    lookup = {}
    image_hashes = set()
    for cell in cells:
        _check(type(cell) is dict and type(cell["group_id"]) is int, "Invalid cell identity")
        key = cell["group_id"], cell["target"]
        _check(key in expected_keys and key not in lookup, "Missing, duplicate or foreign grid cell")
        group = planned[key[0]]
        for field in ("source", "variant", "seed", "images", "parameters_sha256"):
            _same(cell[field], group[field], "Cell " + field)
        _check(cell["status"] == "complete" and cell["failures"] == [], "Failed/NR/incomplete cell")
        for marker in ("diagnostic_only", "formal_metrics_eligible"):
            if marker in cell:
                _check(type(cell[marker]) is bool, "Cell scope marker must be a strict bool: " + marker)
        if max_images is None:
            _check(cell.get("diagnostic_only") is not True and cell.get("formal_metrics_eligible") is not False
                and cell.get("observation_origin") != "core_executor_smoke_diagnostic",
                "Diagnostic observations cannot enter the full-image arithmetic scope")
        _check(_sha(cell["image_ids_sha256"]) and _sha(cell["input_evidence_sha256"]), "Missing input identity hash")
        image_hashes.add(cell["image_ids_sha256"])
        _check(type(cell["metrics"]) is dict and set(cell["metrics"]) == set(COCO_BBOX_METRICS),
            "All twelve target metrics are required")
        for metric, value in cell["metrics"].items():
            _number(value, metric, metric=True)
        _number(cell["bbox_mAP"], "primary AP", metric=True)
        _check(cell["bbox_mAP"] >= 0 and cell["bbox_mAP"] == cell["metrics"]["bbox_mAP"],
            "Undefined or inconsistent primary AP")
        lookup[key] = cell
    _check(len(image_hashes) == 1 and set(lookup) == set(expected_keys), "Unpaired or incomplete image/target panel")
    _same(result["image_ids_sha256"], next(iter(image_hashes)), "Report common image hash")
    ordered = [lookup[key] for key in expected_keys]
    _same(result["normalized_cells"], ordered, "Complete original cell records in canonical order")
    configuration = result["configuration_seed"]
    _check(type(configuration) is list and len(configuration) == 8, "Incomplete configuration summary")
    maximum = Decimal(0)
    with localcontext() as context:
        context.prec = 80
        means = {}
        for group, row in zip(groups, configuration):
            identity_fields = ("group_id", "variant", "source", "seed", "parameters_sha256")
            _check(set(row) == set(identity_fields) | set(PLANNED_FIELDS) | {"BB_mean", "whitebox_AP"},
                "Configuration summary schema differs")
            _same({k: row[k] for k in identity_fields}, {k: group[k] for k in identity_fields}, "Configuration identity")
            _same({k: row[k] for k in PLANNED_FIELDS}, _planned_schedule(group), "Reported planned schedule integers")
            values = [lookup[group["group_id"], target]["bbox_mAP"] for target in targets[1:]]
            mean = sum((_number(v, "BB input") for v in values), Decimal(0)) / Decimal(15)
            means[group["variant"], group["seed"]] = mean
            maximum = max(maximum, _near(row["BB_mean"], mean, "configuration BB15"),
                _near(row["whitebox_AP"], _number(lookup[group["group_id"], targets[0]]["bbox_mAP"], "WB input"), "whitebox AP"))
        by_pair = {(g["variant"], g["seed"]): g["group_id"] for g in groups}
        _check(type(result["contrasts"]) is list and len(result["contrasts"]) == 32, "Incomplete report family")
        for declaration, actual in zip(declarations, result["contrasts"]):
            _check(set(actual) == set(declaration) | {"estimate", "per_target", "inference"}, "Contrast metadata schema differs")
            _same({k: actual[k] for k in declaration}, declaration, "Complete contrast metadata")
            _check(actual["inference"] == INFERENCE, "Contrast claims statistical inference")
            rows, terms = actual["per_target"], declaration["terms"]
            _check(type(rows) is list and len(rows) == 16, "Missing contrast targets")
            bb = []
            for target, row in zip(targets, rows):
                _check(set(row) == {"target", "source_matched", "estimate", "metrics"}, "Target contrast schema differs")
                _same(row["target"], target, "Contrast target order")
                _same(row["source_matched"], target == targets[0], "Whitebox exclusion")
                _check(set(row["metrics"]) == set(COCO_BBOX_METRICS), "Missing contrast metrics")
                primary = None
                for metric in COCO_BBOX_METRICS:
                    values = [lookup[by_pair[t["variant"], t["seed"]], target]["metrics"][metric] for t in terms]
                    if any(v == -1 for v in values):
                        _check(row["metrics"][metric] is None, "Undefined metric must remain null")
                    else:
                        value = _weighted(values, terms)
                        maximum = max(maximum, _near(row["metrics"][metric], value, "target " + metric))
                        if metric == "bbox_mAP":
                            primary = value
                _check(primary is not None, "Undefined primary contrast")
                maximum = max(maximum, _near(row["estimate"], primary, "target AP estimate"))
                if target != targets[0]:
                    bb.append(primary)
            expected = sum(bb, Decimal(0)) / Decimal(15)
            independent_scalar = sum((means[t["variant"], t["seed"]] * Decimal(Fraction(t["coefficient"]).numerator)
                / Decimal(Fraction(t["coefficient"]).denominator) for t in terms), Decimal(0))
            _check(abs(expected - independent_scalar) <= TOLERANCE, "Independent scalar/target reconstruction differs")
            maximum = max(maximum, _near(actual["estimate"], expected, "contrast BB15"))
    return dict(schema_version=1, record_type="vcsf_core_point_arithmetic_check",
        status="independently_verified_core_point_arithmetic_only_pending_input_acceptance",
        stage_plan_content_sha256=canonical_hash(stage_plan), input_cells_content_sha256=canonical_hash(cells),
        report_content_sha256=canonical_hash(result), contrast_family_sha256=canonical_hash(declarations),
        groups=8, cells=128, contrasts=32, images=images, max_images=max_images,
        diagnostic_only=max_images is not None, synthetic=stage_plan["synthetic"],
        maximum_absolute_arithmetic_error=float(maximum), arithmetic_tolerance=1e-12,
        planned_schedule_scope="declarations_not_measured_cost", physical_cost_status="not_assessed",
        input_provenance_acceptance="pending", independent_result_acceptance=False,
        formal_metrics_eligible=False, scientific_acceptance=False, independent_confirmation=False,
        automatic_promotion=False, source_images_verified=False, prediction_archives_replayed=False,
        original_cells_modified=False, significance_claimed=False, confidence_intervals_produced=False,
        bootstrap_replicates_computed=0)


def _json_bytes(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _check(key not in result, "Duplicate JSON key: " + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError("Non-finite JSON constant: " + value)

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def _csv_rows(raw, fields):
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8"), newline=""))
    _check(reader.fieldnames == fields, "CSV columns or order differ")
    rows = list(reader)
    _check(all(set(row) == set(fields) and all(v is not None for v in row.values()) for row in rows),
        "Malformed CSV row")
    return rows


def _display(value):
    return "NA" if value is None else format(value, ".4f")


def _tex(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


def _bounded_read(path):
    before = path.stat()
    _check(0 <= before.st_size <= MAX_EXPORT_FILE_BYTES, "Export file exceeds the 128 MiB byte limit")
    with path.open("rb") as handle:
        raw = handle.read(MAX_EXPORT_FILE_BYTES + 1)
    _check(len(raw) <= MAX_EXPORT_FILE_BYTES, "Export file grew beyond its byte limit")
    after = path.stat()
    _check(len(raw) == before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns,
        "Export file changed during bounded reading")
    return raw


def verify_core_point_exports(directory, result):
    """Check export bytes against the supplied report, not its provenance or math."""
    _scope(result)
    directory = Path(directory)
    _check(directory.is_dir() and not any(p.is_symlink() for p in (directory,) + tuple(directory.parents)),
        "Export directory must be a plain directory")
    names = {"analysis.json", "contrast_definitions.json", "metrics.csv", "configuration_seed.csv",
        "contrasts.csv", "target_contrasts.csv", "metrics.tex", "contrasts.tex", "report_scope.txt"}
    entries = list(directory.iterdir())
    _check({p.name for p in entries} == names | {"manifest.json"}
        and all(p.is_file() and not p.is_symlink() for p in entries), "Export file inventory differs")
    contents = {p.name: _bounded_read(p) for p in entries}
    manifest = _json_bytes(contents["manifest.json"])
    _check(type(manifest) is dict and set(manifest) == names, "Export manifest inventory differs")
    for name in names:
        _check(_sha(manifest[name]) and hashlib.sha256(contents[name]).hexdigest() == manifest[name],
            "Export SHA-256 differs: " + name)
    _same(_json_bytes(contents["analysis.json"]), result, "Exported complete analysis")
    declarations = [{k: deepcopy(v) for k, v in c.items() if k not in ("estimate", "per_target", "inference")}
        for c in result["contrasts"]]
    _same(_json_bytes(contents["contrast_definitions.json"]), declarations, "Exported complete declarations")
    _same(canonical_hash(declarations), result["contrast_family_sha256"], "Exported declaration hash")
    _check(type(result["planned_groups"]) is list and len(result["planned_groups"]) == 8
        and type(result["configuration_seed"]) is list and len(result["configuration_seed"]) == 8,
        "Exported planned group and configuration coverage differs")
    for group, row in zip(result["planned_groups"], result["configuration_seed"]):
        fields = ("group_id", "variant", "source", "seed", "parameters_sha256")
        _same({k: row[k] for k in fields}, {k: group[k] for k in fields}, "Exported planned group identity")
        _same({k: row[k] for k in PLANNED_FIELDS}, _planned_schedule(group), "Exported planned schedule integers")
    metric_fields = ["group_id", "variant", "seed", "source", "target", "metric", "value"]
    metric_rows = [dict(group_id=str(c["group_id"]), variant=c["variant"], seed=str(c["seed"]), source=c["source"],
        target=c["target"], metric=m, value=_display(c["metrics"][m]))
        for c in result["normalized_cells"] for m in COCO_BBOX_METRICS]
    config_fields = ["group_id", "variant", "source", "seed", "parameters_sha256"] + list(PLANNED_FIELDS) + ["BB_mean", "whitebox_AP"]
    config_rows = [{k: _display(r[k]) if k in ("BB_mean", "whitebox_AP") else str(r[k]) for k in config_fields}
        for r in result["configuration_seed"]]
    contrast_fields = ["id", "block", "kind", "estimate", "declaration_json"]
    contrast_rows = [dict(id=c["id"], block=c["block"], kind=c["kind"], estimate=_display(c["estimate"]),
        declaration_json=json.dumps(declaration, sort_keys=True, separators=(",", ":"), allow_nan=False))
        for c, declaration in zip(result["contrasts"], declarations)]
    target_fields = ["contrast", "target", "source_matched", "metric", "value"]
    target_rows = [dict(contrast=c["id"], target=t["target"], source_matched=str(t["source_matched"]),
        metric=m, value=_display(t["metrics"][m]))
        for c in result["contrasts"] for t in c["per_target"] for m in COCO_BBOX_METRICS]
    for name, fields, expected in (("metrics.csv", metric_fields, metric_rows),
            ("configuration_seed.csv", config_fields, config_rows), ("contrasts.csv", contrast_fields, contrast_rows),
            ("target_contrasts.csv", target_fields, target_rows)):
        _same(_csv_rows(contents[name], fields), expected, "Export " + name)
    scope_title = "Arithmetic Only / Input Provenance Pending"
    for name, header, rows in (("metrics.tex", "Group / Target / Metric & Value",
            [(r["group_id"] + " / " + r["target"] + " / " + r["metric"], r["value"]) for r in metric_rows]),
            ("contrasts.tex", "Contrast & BB Mean Difference", [(r["id"], r["estimate"]) for r in contrast_rows])):
        lines = contents[name].decode("utf-8").splitlines()
        _check(len(lines) == len(rows) + 5 and lines[1] == r"\begin{longtable}{lr}"
            and lines[3] == r"\hline" and lines[-1] == r"\end{longtable}", "Incomplete TeX export")
        _same(lines[4:-1], [_tex(label) + " & " + value + r" \\" for label, value in rows], "TeX rows")
        _same(lines[0], "% " + scope_title + "; descriptive values, no scientific acceptance.", "TeX scope")
        _same(lines[2], header + r" \\", "TeX header")
    scope = contents["report_scope.txt"].decode("utf-8")
    expected_scope = (scope_title + "\nSynthetic stage: " + str(result["synthetic"])
        + "\nDiagnostic only: " + str(result["diagnostic_only"])
        + "\nThe report does not verify images, predictions, official metric replay or original provenance.\n"
        + "No formal result slot, execution admission, scientific acceptance or method promotion is issued.\n")
    _same(scope.replace("\r\n", "\n"), expected_scope, "Exact report scope")
    _check({p.name for p in directory.iterdir()} == set(contents)
        and all(not (directory / name).is_symlink() and _bounded_read(directory / name) == raw
            for name, raw in contents.items()), "Exports changed during audit")
    return dict(schema_version=1, record_type="vcsf_core_point_export_check",
        status="independently_verified_core_point_exports_only_pending_input_acceptance",
        planned_schedule_scope="declarations_not_measured_cost", physical_cost_status="not_assessed",
        report_content_sha256=canonical_hash(result), checked_file_sha256={name: hashlib.sha256(raw).hexdigest()
            for name, raw in contents.items()}, input_provenance_acceptance="pending",
        independent_result_acceptance=False, formal_metrics_eligible=False, scientific_acceptance=False,
        independent_confirmation=False, automatic_promotion=False, source_images_verified=False,
        prediction_archives_replayed=False, original_cells_modified=False)
