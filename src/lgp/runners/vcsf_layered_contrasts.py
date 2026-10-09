"""Exact, outcome-independent contrast coefficients for the registered study."""
from fractions import Fraction
from itertools import combinations, product


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _fraction(value):
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else str(value)


def compile_contrasts(study, resolved, scheduled_pairs):
    policy = study["analysis_policy"]
    require(policy["metric"] == "bbox_mAP"
        and policy["scalar"] == "equal_weight_source_excluded_cells",
        "Layered contrasts require the registered raw source-excluded AP scalar")
    scheduled = set(scheduled_pairs)
    rows = []

    def emit(identifier, block, kind, coefficients, **details):
        coefficients = {pair: Fraction(weight) for pair, weight in coefficients.items() if weight}
        require(coefficients and set(coefficients) <= scheduled,
            "A contrast refers to missing configuration-seed evidence: " + identifier)
        require(sum(coefficients.values()) == 0, "Contrast weights do not sum to zero")
        rows.append({"id": identifier, "block": block, "kind": kind, **details,
            "terms": [{"variant": variant, "seed": seed, "coefficient": _fraction(weight)}
                for (variant, seed), weight in sorted(coefficients.items())]})

    for spec in policy["factorials"]:
        axes = list(spec["axes"])
        require(len(axes) >= 2, "A factorial needs at least two registered axes")
        fields = [field for axis in axes for field in spec["axes"][axis]]
        require(len(fields) == len(set(fields)), "Factorial axes overlap in implementation fields")
        low = resolved[spec["low"]]["parameters"]
        high = resolved[spec["high"]]["parameters"]
        require(set(low) == set(high)
            and all(low[k] == high[k] for k in low if k not in fields),
            "Factorial endpoints differ outside the declared axes")
        require(all(any(low[k] != high[k] for k in spec["axes"][axis]) for axis in axes),
            "A factorial axis has identical low and high settings")
        grid = {}
        for bits in product((0, 1), repeat=len(axes)):
            parameters = dict(low)
            for axis, bit in zip(axes, bits):
                for field in spec["axes"][axis]:
                    parameters[field] = (high if bit else low)[field]
            matches = [variant for variant in study["blocks"][spec["block"]]["variants"]
                if resolved[variant]["parameters"] == parameters]
            require(len(matches) == 1, "Factorial cell is missing or ambiguous: " + str(bits))
            grid[bits] = matches[0]
        seeds = spec["seeds"]
        require(seeds and len(seeds) == len(set(seeds))
            and set(seeds) <= set(study["blocks"][spec["block"]]["seeds"]),
            "Factorial seeds differ from their registered block")
        # Every nonempty effect, both averaged over and conditional on the other axes.
        for size in range(1, len(axes) + 1):
            for active in combinations(range(len(axes)), size):
                other = [i for i in range(len(axes)) if i not in active]
                conditions = [None] + (list(product((0, 1), repeat=len(other))) if other else [])
                for condition in conditions:
                    weights = {}
                    for bits, variant in grid.items():
                        if condition is not None and tuple(bits[i] for i in other) != condition:
                            continue
                        sign = (-1) ** sum(1 - bits[i] for i in active)
                        divisor = len(seeds) * (2 ** len(other) if condition is None else 1)
                        for seed in seeds:
                            weights[(variant, seed)] = Fraction(sign, divisor)
                    active_names = [axes[i] for i in active]
                    conditioning = {} if condition is None else {
                        axes[i]: bit for i, bit in zip(other, condition)}
                    suffix = "averaged" if condition is None else "_".join(
                        "{}{}".format(axis, bit) for axis, bit in conditioning.items())
                    emit("{}:{}:{}".format(spec["block"], "*".join(active_names), suffix),
                        spec["block"], "factorial_difference", weights,
                        factors=active_names, conditioning=conditioning,
                        averaged_over=[axes[i] for i in other] if condition is None else [],
                        averaged_seeds=list(seeds),
                        low_variant=spec["low"], high_variant=spec["high"])

    anchor, seed = policy["anchor"], policy["primary_seed"]
    require(policy["anchor_comparisons"] == "every_nonanchor_variant_at_primary_seed",
        "Anchor-comparison scope changed")
    for variant in study["variants"]:
        if variant != anchor:
            emit("anchor:" + variant, "anchor", "candidate_minus_reference",
                {(variant, seed): Fraction(1), (anchor, seed): Fraction(-1)},
                candidate=variant, reference=anchor, averaged_seeds=[seed])
    for pair in policy["additional_pairs"]:
        candidate, reference, seeds = pair["candidate"], pair["reference"], pair["seeds"]
        require(candidate != reference and seeds and len(seeds) == len(set(seeds)),
            "Invalid extra paired comparison")
        block = study["blocks"][pair["block"]]
        require({candidate, reference} <= set(block["variants"])
            and set(seeds) <= set(block["seeds"]), "Extra contrast is outside its block")
        weights = {(variant, s): Fraction(sign, len(seeds))
            for variant, sign in ((candidate, 1), (reference, -1)) for s in seeds}
        emit("{}:{}-{}".format(pair["block"], candidate, reference), pair["block"],
            "candidate_minus_reference", weights, candidate=candidate, reference=reference,
            averaged_seeds=list(seeds))
    require(len(rows) == len({row["id"] for row in rows}), "Duplicate contrast identifier")
    return {"status": "prepared_not_numerically_validated", "metric": policy["metric"],
        "family_size": len(rows), "contrasts": rows,
        "interpretation": "negative_on_minus_off_or_candidate_minus_reference_lowers_AP",
        "uncertainty": policy["uncertainty"], "selection": policy["selection"],
        "independent_confirmation": False}
