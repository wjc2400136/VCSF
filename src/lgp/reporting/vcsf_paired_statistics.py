"""Paired scalar/interval algebra; input acceptance is a separate mandatory gate."""
from fractions import Fraction
import math

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _ap(value):
    require(not isinstance(value, (bool, np.bool_))
        and isinstance(value, (int, float, np.integer, np.floating)), "AP must be an unrounded number")
    value = float(value)
    require(math.isfinite(value) and 0 <= value <= 1, "AP must be finite in raw [0,1] units")
    return value


def paired_panel_scalars(plan, cells):
    """Validate normalized cell identities and exclude only each source's own target."""
    sources, targets = plan["sources"], plan["targets"]
    require(sources and targets and len(set(sources)) == len(sources)
        and len(set(targets)) == len(targets) and set(sources) <= set(targets),
        "Invalid canonical source/target panel")
    require(plan["blackbox_cells_per_configuration_seed"] == len(sources) * (len(targets) - 1),
        "Incorrect black-box panel size")
    require(all(type(g["group_id"]) is int and type(g["seed"]) is int for g in plan["groups"]),
        "Invalid planned group or seed identity")
    groups = {group["group_id"]: group for group in plan["groups"]}
    require(len(groups) == len(plan["groups"]) and groups, "Invalid analysis group plan")
    require(all(type(g["images"]) is int and g["images"] > 0 for g in groups.values())
        and len({g["images"] for g in groups.values()}) == 1,
        "All paired groups must declare the same positive integer image count")
    scheduled = {}
    for group in groups.values():
        require(group["targets"] == targets and group["source"] in sources,
            "Group disagrees with the canonical panel")
        pair = group["variant"], group["seed"]
        scheduled.setdefault(pair, []).append(group["source"])
    require(all(len(value) == len(sources) and set(value) == set(sources) for value in scheduled.values()),
        "Missing or duplicate source within a configuration-seed pair")
    expected = {(group_id, target) for group_id, group in groups.items() for target in group["targets"]}
    require(len(expected) == sum(len(group["targets"]) for group in groups.values()),
        "Duplicate planned target")
    lookup = {}
    for cell in cells:
        key = cell["group_id"], cell["target"]
        require(type(cell["group_id"]) is int and key in expected and key not in lookup,
            "Unexpected or duplicate analysis cell")
        group = groups[key[0]]
        for field in ("source", "variant", "seed", "parameters_sha256", "images"):
            require(type(cell[field]) is type(group[field]) and cell[field] == group[field],
                "Analysis cell identity mismatch: " + field)
        require(cell["status"] == "complete" and cell["failures"] == [],
            "Failed or incomplete analysis cell cannot be dropped")
        lookup[key] = _ap(cell["bbox_mAP"])
    require(set(lookup) == expected, "Incomplete analysis matrix")
    pairs = {}
    for group_id, group in groups.items():
        require(group["source_matched_target"] == group["source"] in group["targets"]
            and group["blackbox_targets"] == [t for t in group["targets"] if t != group["source"]],
            "Invalid source-exclusion definition")
        pair = (group["variant"], group["seed"])
        records = pairs.setdefault(pair, [])
        for target in group["blackbox_targets"]:
            records.append({"source": group["source"], "target": target, "value": lookup[group_id, target]})
    summary = {}
    source_order, target_order = {s: i for i, s in enumerate(sources)}, {t: i for i, t in enumerate(targets)}
    for pair, records in pairs.items():
        require(len(records) == plan["blackbox_cells_per_configuration_seed"],
            "Incomplete source-excluded panel")
        records.sort(key=lambda row: (source_order[row["source"]], target_order[row["target"]]))
        summary[pair] = {"mean": math.fsum(record["value"] for record in records) / len(records),
            "cells": records,
            "source_means": {source: math.fsum(r["value"] for r in records if r["source"] == source)
                / sum(r["source"] == source for r in records) for source in sources},
            "target_means": {target: math.fsum(r["value"] for r in records if r["target"] == target)
                / sum(r["target"] == target for r in records) for target in targets
                if any(r["target"] == target for r in records)}}
    return summary


def image_resamples(image_count, definition):
    """Yield sorted positions; each cell restarts this same outcome-free stream."""
    require(type(image_count) is int and image_count > 0, "Invalid image count")
    require(definition["rng"] == "numpy_randomstate_mt19937"
        and definition["resampling"] == "paired_nonparametric_image_bootstrap"
        and definition["image_occurrence_order"] == "original_id_then_duplicate_occurrence",
        "Unregistered bootstrap generator")
    replicates, seed = definition["replicates"], definition["seed"]
    require(type(replicates) is int and replicates >= 2 and type(seed) is int
        and 0 <= seed < 2**32, "Invalid bootstrap count or seed")
    rng = np.random.RandomState(seed)
    for _ in range(replicates):
        yield np.sort(rng.randint(0, image_count, size=image_count).astype("<u4"))


def simultaneous_contrast_intervals(analysis, observed, bootstrap):
    """Centered max-error bands in raw AP units, conditional on the fixed seeds."""
    rows, definition = analysis["contrasts"], analysis["uncertainty"]
    require(analysis["metric"] == "bbox_mAP" and analysis["family_size"] == len(rows)
        and rows and len({row["id"] for row in rows}) == len(rows), "Incomplete contrast family")
    estimand = definition["seed_estimand"]
    require(definition["multiplicity"] == "one_family_centered_max_absolute_bootstrap_error"
        and estimand in ("conditional_on_the_five_registered_seeds_not_resampled",
                         "conditional_on_declared_attack_seeds_not_resampled"),
        "Unregistered interval interpretation")
    alpha, replicates = definition["alpha"], definition["replicates"]
    require(type(alpha) is float and 0 < alpha < 1 and type(replicates) is int and replicates >= 2,
        "Invalid interval settings")
    pairs = sorted({(term["variant"], term["seed"]) for row in rows for term in row["terms"]})
    if estimand == "conditional_on_declared_attack_seeds_not_resampled":
        seeds = analysis.get("declared_attack_seeds")
        require(type(seeds) is list and seeds and all(type(seed) is int for seed in seeds)
            and seeds == sorted(set(seeds)) == sorted({seed for _, seed in pairs}),
            "The interval family does not match its declared fixed attack seeds")
    require(set(observed) == set(bootstrap) == set(pairs), "Missing or unexpected paired estimates")
    values = np.array([_ap(observed[pair]) for pair in pairs], dtype=np.float64)
    sampled = []
    for pair in pairs:
        array = np.asarray(bootstrap[pair])
        require(array.shape == (replicates,) and array.dtype.kind in "fi"
            and np.isfinite(array).all() and ((array >= 0) & (array <= 1)).all(),
            "Incomplete, non-finite or wrongly scaled bootstrap estimates")
        sampled.append(array.astype(np.float64, copy=False))
    weights = np.zeros((len(rows), len(pairs)), dtype=np.float64)
    position = {pair: i for i, pair in enumerate(pairs)}
    for index, row in enumerate(rows):
        terms = {(term["variant"], term["seed"]): Fraction(term["coefficient"]) for term in row["terms"]}
        require(len(terms) == len(row["terms"]) and terms and sum(terms.values()) == 0
            and any(terms.values()), "Invalid or duplicate signed contrast terms")
        for pair, weight in terms.items():
            weights[index, position[pair]] = float(weight)
    estimates = weights @ values
    replicated = np.stack(sampled, axis=1) @ weights.T
    radius_samples = np.max(np.abs(replicated - estimates[None, :]), axis=1)
    radius = float(np.sort(radius_samples)[math.ceil((1 - alpha) * replicates) - 1])
    pointwise = np.quantile(replicated, [alpha / 2, 1 - alpha / 2], axis=0, method="linear")
    result = []
    for index, row in enumerate(rows):
        estimate = float(estimates[index])
        lower, upper = estimate - radius, estimate + radius
        result.append({"id": row["id"], "estimate": estimate,
            "simultaneous_lower": lower, "simultaneous_upper": upper,
            "pointwise_percentile_lower": float(pointwise[0, index]),
            "pointwise_percentile_upper": float(pointwise[1, index]),
            "simultaneous_band_excludes_zero": lower > 0 or upper < 0})
    return {"status": "computed_pending_input_and_pipeline_acceptance", "metric": "bbox_mAP",
        "family_size": len(rows), "bootstrap_replicates": replicates, "alpha": alpha,
        "maximum_error_radius": radius, "contrasts": result,
        "finite_sample_guarantee": False, "independent_confirmation": False,
        "scientific_acceptance": False}
