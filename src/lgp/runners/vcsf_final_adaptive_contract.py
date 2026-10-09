"""Frozen adaptive refresh planning only; no execution admission or model calls."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from .vcsf_final_oblivious_contract import build_plan as oblivious_plan, verify_source
from .vcsf_research_plan import canonical_hash

PROTOCOL = "vcsf_final_adaptive_refresh"


def build_plan(registry, devices):
    spec = registry.protocols[PROTOCOL]
    paired = oblivious_plan(registry, devices)
    policy = deepcopy(registry.adaptive_preprocessing_policy)
    if (spec["paired_protocol"] != paired["protocol"]
            or spec["seed_schedule"] != policy["seed_schedule"]
            or policy["seed_schedule"] != "full_val_numeric_image_position"):
        raise ValueError("Adaptive pairing or seed policy differs")
    if paired["defense_order"].count("identity") != 1:
        raise ValueError("Exactly one identity reference is required")
    generated = [deepcopy(job) for job in paired["jobs"] if job["defense"] != "identity"]
    reused = [deepcopy(job) for job in paired["jobs"] if job["defense"] == "identity"]
    groups = []
    seen = set()
    for job in generated:
        key = (job["source"], job["defense"])
        if key not in seen:
            seen.add(key)
            groups.append({key: job[key] for key in ("source", "defense", "device", "images")})
    if (len(groups) != spec["generation_groups"]
            or len(generated) != spec["generated_evaluation_cells"]
            or len(reused) != spec["reused_identity_cells"]
            or len(generated) + len(reused) != spec["evaluation_cells"]):
        raise ValueError("Adaptive refresh scope differs from the registry")
    return dict(protocol=PROTOCOL, paired_protocol=paired["protocol"],
        parameters_sha256=paired["parameters_sha256"], budget_profile=spec["budget_profile"],
        devices=list(devices), generation_groups=groups, generated_evaluations=generated,
        identity_reuse=reused, canonical_jobs=deepcopy(paired["jobs"]),
        defense_order=paired["defense_order"], defense_parameters=paired["defense_parameters"],
        adaptive_policy=policy, seed_schedule=spec["seed_schedule"],
        clean_jobs=0, baseline_jobs=0, formal_execution_admitted=False)


def seed_offsets_from_verified_inputs(inputs, image_ids, base_seed):
    """Retain full-split positions; a diagnostic subset must not renumber seeds."""
    full = inputs["image_ids"]
    retained = inputs["retained_image_ids"]
    rows = inputs["retained_seed_mapping"]
    if (type(base_seed) is not int or len(full) != len(set(full))
            or full != sorted(full) or len(retained) != len(set(retained))
            or retained != sorted(retained)
            or [row["image_id"] for row in rows] != retained):
        raise ValueError("Malformed authenticated seed population")
    positions = {image_id: position for position, image_id in enumerate(full)}
    offsets = {}
    for row in rows:
        image_id = row["image_id"]
        if (image_id not in positions or type(row["full_position"]) is not int
                or type(row["attack_seed"]) is not int
                or row["full_position"] != positions[image_id]
                or row["attack_seed"] != base_seed + positions[image_id]):
            raise ValueError("Seed mapping differs from the full split")
        offsets[image_id] = positions[image_id]
    if (not image_ids or len(image_ids) != len(set(image_ids))
            or list(image_ids) != [value for value in retained if value in set(image_ids)]):
        raise ValueError("Requested images must be an ordered retained subset")
    return {image_id: offsets[image_id] for image_id in image_ids}


def prepare_generation_source(registry, source, payload):
    """Authenticate once per source, not once per transform; grants no execution."""
    from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
    from .vcsf_oblivious_admission import bound_json

    paired = registry.protocols[registry.protocols[PROTOCOL]["paired_protocol"]]
    inputs = verify_source(registry, source, payload)
    run = bound_json(Path(payload) / "run.json",
        paired["payload_bindings"][source]["payload_run_sha256"])
    parameters = run["parameters"]
    if canonical_hash(parameters) != paired["parameters_sha256"]:
        raise ValueError("Adaptive generation parameters differ from frozen payload")
    config = VCSFCommonAnchorConfig.from_mapping(deepcopy(parameters))
    config.validate()
    if canonical_hash(asdict(config)) != paired["parameters_sha256"]:
        raise ValueError("Adaptive configuration normalization changes frozen parameters")
    return dict(source=source, inputs=inputs, parameters=deepcopy(parameters),
        parameters_sha256=paired["parameters_sha256"],
        payload_run_sha256=paired["payload_bindings"][source]["payload_run_sha256"],
        seed=paired["seed"], execution_admitted=False)


def generation_binding(registry, plan, prepared, group_index, max_images=None):
    """Bind one generation job; caller must separately authenticate/admit the run."""
    if plan != build_plan(registry, plan["devices"]):
        raise ValueError("Adaptive plan differs from the canonical matrix")
    groups = plan["generation_groups"]
    if type(group_index) is not int or not 0 <= group_index < len(groups):
        raise ValueError("Unknown adaptive generation group")
    group = groups[group_index]
    paired = registry.protocols[plan["paired_protocol"]]
    if (prepared["source"] != group["source"]
            or prepared["inputs"]["source"] != group["source"]
            or prepared["seed"] != paired["seed"]
            or prepared["parameters_sha256"] != plan["parameters_sha256"]
            or canonical_hash(prepared["parameters"]) != plan["parameters_sha256"]
            or prepared["payload_run_sha256"] !=
                paired["payload_bindings"][group["source"]]["payload_run_sha256"]):
        raise ValueError("Prepared source differs from the registered adaptive group")
    ids = list(prepared["inputs"]["retained_image_ids"])
    if len(ids) != group["images"]:
        raise ValueError("Adaptive source has an incomplete retained population")
    if max_images is not None:
        if type(max_images) is not int or not 0 < max_images <= len(ids):
            raise ValueError("Diagnostic image limit outside retained population")
        ids = ids[:max_images]
    offsets = seed_offsets_from_verified_inputs(prepared["inputs"], ids, prepared["seed"])
    return dict(protocol=PROTOCOL, group_index=group_index, source=group["source"],
        defense=group["defense"], defense_parameters=deepcopy(plan["defense_parameters"][group["defense"]]),
        device=group["device"], image_ids=ids,
        seed_mapping=[dict(image_id=value, offset=offsets[value],
            attack_seed=prepared["seed"] + offsets[value]) for value in ids],
        seed=prepared["seed"], parameters=deepcopy(prepared["parameters"]),
        parameters_sha256=plan["parameters_sha256"], budget_profile=plan["budget_profile"],
        max_images=max_images, formal_scope=max_images is None, execution_admitted=False)
