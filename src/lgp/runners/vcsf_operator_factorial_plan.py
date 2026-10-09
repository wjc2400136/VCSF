"""Bind the fixed-width eight-operator design without accepting reuse or execution."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import inspect
import json
from pathlib import Path
import re
import sys

from ..attacks.factory import ATTACK_TYPES
from ..attacks.vcsf_operator_factorial_isolated import (
    VCSFOperatorFactorialCandidate, VCSFOperatorFactorialConfig,
)
from ..io import atomic_json, file_digest
from ..registry import Registry
from .vcsf_layered_contrasts import compile_contrasts
from .vcsf_research_plan import canonical_hash, require
from .vcsf_scale_plan import compile_scale_plan


PROTOCOL = "vcsf_single_source_operator_factorial_research"
ENTRYPOINT = "experiments/prepare_vcsf_operator_factorial.py"


def compile_operator_factorial_plan(registry, max_images=None, devices=("cuda:0", "cuda:1")):
    registry._validate()
    require(registry.__dict__ == Registry(registry.root).__dict__,
        "In-memory registry differs from the current declared source")
    require(max_images is None or (type(max_images) is int and 0 < max_images < 5000),
        "Diagnostic image limit must be an integer from 1 through 4999")
    protocol = registry.protocols[PROTOCOL]
    require(isinstance(devices, (tuple, list)) and len(devices) in protocol["device_counts"]
        and len(set(devices)) == len(devices)
        and all(isinstance(d, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", d) for d in devices),
        "Select one, two or four distinct explicit CUDA ordinals")
    study = registry.ablation_studies[protocol["ablation_study"]]
    implementation = protocol["isolated_candidate"]
    require(implementation == dict(execution_alias=study["method"],
        implementation=VCSFOperatorFactorialCandidate.implementation_path,
        class_name=VCSFOperatorFactorialCandidate.__name__,
        config_class_name=VCSFOperatorFactorialConfig.__name__), "Operator implementation declaration differs")
    require(Path(inspect.getfile(VCSFOperatorFactorialCandidate)).resolve()
        == (registry.root / implementation["implementation"]).resolve(),
        "Operator implementation is not imported from the registry source root")
    require(study["method"] not in registry.attacks and study["method"] not in ATTACK_TYPES,
        "Operator research must not alter the public method registry or factory")
    require(protocol["planning_entrypoint"] == ENTRYPOINT
        and protocol["owner_decision"] == "docs/research/vcsf-whole-ablation-revision-20260911.md",
        "Operator preparation authority or entry point differs")
    base = registry.ablation_studies[study["base_study"]]
    baseline = dict(base["defaults"], **base["variants"][study["base_variant"]]["parameters"])
    width = Fraction(study["half_width"])
    baseline.update(scale_min=float(1 - width), scale_max=float(1 + width))
    sources, targets = protocol["sources"], registry.target_ids()
    images = protocol["images"] if max_images is None else max_images
    resolved, groups = {}, []
    for index, row in enumerate(study["row_order"], 1):
        config = VCSFOperatorFactorialConfig.from_mapping(dict(baseline, **study["variants"][row]["parameters"]))
        config.validate()
        parameters = asdict(config)
        resolved[row] = dict(parameters=parameters, parameters_sha256=canonical_hash(parameters))
        groups.append(dict(group_id=index, variant=row, source=sources[0], source_matched_target=sources[0],
            seed=protocol["seed"], images=images, targets=list(targets),
            blackbox_targets=[target for target in targets if target != sources[0]],
            factor_bits={axis: study["factor_levels"][axis].index(parameters[axis]) for axis in study["factor_order"]},
            logical_gradients=20, detector_initialization_updates=1, feature_updates=19,
            detached_reference_forwards=19, complete_detector_backward_equivalence_claimed=False,
            physical_cost_status="unmeasured_no_inheritance", reuse_status="unassessed",
            historical_variant=study["historical_reference_variants"].get(row),
            requested_new=row in study["requested_new_variants"], execution_status="not_armed", **resolved[row]))
    require(len({group["parameters_sha256"] for group in groups}) == len(groups),
        "Operator cells are not distinct complete configurations")
    analysis = compile_contrasts(study, resolved, [(row, protocol["seed"]) for row in study["row_order"]])
    actual_counts = dict(averaged_factorial=0, conditional_factorial=0, anchor=0, total=analysis["family_size"])
    for contrast in analysis["contrasts"]:
        if contrast["block"] == "anchor":
            actual_counts["anchor"] += 1
        elif contrast["conditioning"]:
            actual_counts["conditional_factorial"] += 1
        else:
            actual_counts["averaged_factorial"] += 1
    require(actual_counts == study["analysis_policy"]["expected_contrasts"],
        "Operator factorial contrast family differs from its registration")
    require(analysis["metric"] == "bbox_mAP" and analysis["independent_confirmation"] is False,
        "Operator contrast interpretation differs")
    analysis.update(contrast_counts=actual_counts,
        coefficient_convention=study["analysis_policy"]["coefficient_convention"],
        target_order=list(targets), source_excluded_target_count=len(targets) - 1,
        all_twelve_metrics_required=True, outcome_values=None)

    # Match declarations only. Original saved outputs are deliberately not read here.
    historical = compile_scale_plan(registry)
    old_groups = {group["variant"]: group for group in historical["groups"]}
    references = []
    for group in groups:
        old_name = group["historical_variant"]
        if old_name is None:
            continue
        original = old_groups[old_name]
        require(original["parameters_sha256"] == group["parameters_sha256"],
            "Requested historical configuration differs from its original full parameter mapping")
        references.append(dict(variant=group["variant"], historical_variant=old_name,
            historical_protocol=historical["protocol"], parameters_sha256=group["parameters_sha256"],
            declaration_matches=True, original_evidence_read=False, reuse_qualified=False))
    totals = dict(source_groups=len(groups), generated_image_instances=len(groups) * protocol["images"],
        target_evaluations=len(groups) * len(targets))
    require(totals == study["planning_totals_before_reuse"], "Operator group totals differ from registry")
    requested = [group for group in groups if group["requested_new"]]
    conditional = dict(source_groups=len(requested), generated_image_instances=len(requested) * protocol["images"],
        target_evaluations=len(requested) * len(targets))
    require(conditional == study["conditional_new_work_after_four_qualified_reuses"]
        and len(requested) == protocol["requested_new_group_limit"], "Requested new-operator scope changed")
    files = set(historical["source_sha256"])
    files.update([ENTRYPOINT, __file__, implementation["implementation"],
        "src/lgp/runners/vcsf_layered_contrasts.py", protocol["owner_decision"]])
    source_hashes = {}
    for name in files:
        path = Path(name)
        path = (path if path.is_absolute() else registry.root / path).resolve()
        require(registry.root in path.parents and path.is_file(), "Missing or foreign operator plan source")
        source_hashes[path.relative_to(registry.root).as_posix()] = file_digest(path)
    science = dict(protocol=PROTOCOL, protocol_sha256=canonical_hash(protocol), study_sha256=canonical_hash(study),
        base_study_sha256=canonical_hash(base), source_sha256=dict(sorted(source_hashes.items())),
        sources=list(sources), targets=list(targets), half_width=str(width), seed=protocol["seed"],
        max_images=max_images, diagnostic_only=max_images is not None, groups=groups, analysis=analysis,
        historical_declaration_references=references, totals_before_qualified_reuse=totals,
        conditional_new_work_after_four_qualified_reuses=conditional)
    return dict(schema_version=1, status="prepared_operator_factorial_not_executable",
        scientific_contract=science, science_sha256=canonical_hash(science),
        requested_devices=list(devices), device_availability_checked=False,
        conditional_new_lane_group_ids=[[g["group_id"] for g in requested[i::len(devices)]] for i in range(len(devices))],
        scheduling_status="conditional_only_not_admission", new_group_ids=None, qualified_reused_group_ids=None,
        new_generation_image_instances=None, new_target_evaluations=None,
        runner_armed=False, formal_execution_admission=False, independent_reuse_acceptance=False,
        scientific_acceptance=False, independent_confirmation=False, automatic_promotion=False,
        model_calls=0, AP_evaluations=0, original_run_restart=False,
        remaining_gates=["ODA_compile_registry_and_exact_contrast_tests", "joint_operator_forward_backward_rng_tests",
            "exact_original_four_cell_reuse_qualification", "full_inputs_checkpoint_environment_binding",
            "independent_fresh_execution_and_lifecycle_admission",
            "before_four_device_start_unique_jobs_model_input_loading_real_forward_backward_and_sample_writes",
            "one_two_and_four_device_release_validation_may_overlap_ablation_execution",
            "complete_5000_image_sixteen_target_group_acceptance", "point_cost_and_scientific_working_selection"])


def write_operator_factorial_plan(output, plan):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "plan.json", plan)
    science = plan["scientific_contract"]
    with (output / "cells.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["group_id", "variant", "image_interpolation", "placement", "image_padding", "images",
            "parameters_sha256", "historical_variant", "requested_new"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group in science["groups"]:
            writer.writerow({name: group["parameters"][name] if name in group["factor_bits"] else group[name] for name in fields})
    with (output / "contrasts.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["contrast_id", "variant", "seed", "exact_coefficient"])
        for contrast in science["analysis"]["contrasts"]:
            for term in contrast["terms"]:
                writer.writerow([contrast["id"], term["variant"], term["seed"], term["coefficient"]])
    manifest = {name: dict(sha256=file_digest(output / name), bytes=(output / name).stat().st_size)
        for name in ("plan.json", "cells.csv", "contrasts.csv")}
    atomic_json(output / "artifact_manifest.json", dict(status="prepared_no_result_metrics", artifacts=manifest,
        science_sha256=plan["science_sha256"], formal_execution_admission=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-only", action="store_true", help="Inspect without writing or reading experiment results.")
    parser.add_argument("--max-images", type=int, help="Diagnostic limit from 1 to 4999; omit for the full-size design.")
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"], help="One, two or four distinct explicit CUDA planning lanes; no GPU is reserved.")
    parser.add_argument("--output", type=Path, help="Fresh directory for the bound plan and generated design CSVs.")
    args = parser.parse_args(argv)
    if args.plan_only and args.output is not None:
        parser.error("--plan-only does not accept --output")
    if sys.platform != "linux" or Path(sys.prefix).name != "oda" or sys.executable != str(Path(sys.prefix) / "bin" / "python"):
        parser.error("Operator preparation requires the pinned server ODA interpreter")
    try:
        registry = Registry()
        plan = compile_operator_factorial_plan(registry, args.max_images, args.devices)
        output = None
        if not args.plan_only:
            output = args.output or registry.root / "outputs/plans" / PROTOCOL / datetime.now(
                timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            write_operator_factorial_plan(output, plan)
        science = plan["scientific_contract"]
        print(json.dumps(dict(status=plan["status"], output=None if output is None else str(output),
            science_sha256=plan["science_sha256"], design_groups=len(science["groups"]),
            contrasts=science["analysis"]["contrast_counts"],
            conditional_new_lane_group_ids=plan["conditional_new_lane_group_ids"],
            new_group_ids=None, reuse_qualified=False, runner_armed=False, model_calls=0), indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
