"""Generation accounting provenance for prepared preprocessing views."""
import hashlib
import json
import math
from pathlib import Path


NATIVE_COUNTERS = ("actual_logical_updates", "detector_input_backwards",
                   "feature_partial_input_backwards", "clean_reference_image_rows",
                   "adversarial_differentiable_views", "calibrated_whole_detector_BE")
GENERATION_FIELDS = ("generation_cost_scope", "generation_sample_n",
                     "generation_run_file", "generation_run_sha256",
                     "generation_native_sample_n",
                     *("generation_mean_" + key for key in NATIVE_COUNTERS))


def generation_cost_csv(record):
    means = record.get("generation_native_counts_mean") or {}
    values = dict(record, **{"generation_mean_" + key: means.get(key) for key in NATIVE_COUNTERS})
    return {key: ("NR" if values.get(key) is None else
                  "{:.4f}".format(values[key]) if type(values[key]) is float else values[key])
            for key in GENERATION_FIELDS}


def generation_cost_metadata(group, original, read_bytes=None):
    if group.get("record_type") == "clean" or group.get("attack") == "clean":
        return {}
    read_bytes = read_bytes or (lambda path: path.read_bytes())
    path = Path(original["root"]).resolve() / "run.json"
    raw = read_bytes(path)
    run = json.loads(raw)
    expected = dict(status="complete", source=group["source"], attack=group["attack"],
                    parameters_sha256=group["parameters_sha256"],
                    budget_profile=group["budget_profile"])
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Generation accounting metadata differs from the assigned payload")
    for binding in original.get("headers", []):
        if Path(binding["file"]).resolve() == path and binding["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("Original generation accounting binding changed")
    values = {key: run.get(key) for key in (
        "gradient_evaluations_per_image", "budget_profile_definition",
        "baseline_profile_definition", "auxiliary_passes",
        "actual_gradient_evaluations_total", "actual_gradient_evaluations_mean",
        "calibrated_whole_detector_BE")}
    values.update(generation_cost_scope=("current_adaptive_generation" if group.get("adaptive_generation")
                                        else "reused_original_generation"),
                  generation_sample_n=run.get("successful_images"),
                  generation_run_file=str(path), generation_run_sha256=hashlib.sha256(raw).hexdigest(),
                  generation_native_sample_n=None, generation_native_counts_mean=None)
    native = path.parent / "native_cost.jsonl"
    if native.is_file():
        raw = read_bytes(native)
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        means = {}
        for key in NATIVE_COUNTERS:
            observed = [row.get(key) for row in rows]
            means[key] = (math.fsum(observed) / len(observed) if observed and all(
                type(value) in (int, float) and math.isfinite(value) and value >= 0
                for value in observed) else None)
        values.update(generation_native_sample_n=len(rows), generation_native_counts_mean=means,
                      generation_native_cost_reference=dict(
                          file=str(native), sha256=hashlib.sha256(raw).hexdigest()))
    return values


def record_cost_metadata(metadata):
    values = dict(metadata)
    for suffix in ("total", "mean"):
        key = "actual_gradient_evaluations_" + suffix
        values["attack_" + key] = values.pop(key, None)
    return values
