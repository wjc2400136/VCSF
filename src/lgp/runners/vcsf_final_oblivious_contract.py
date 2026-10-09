"""Evaluation-only Common-2 preprocessing plan and immutable payload binding."""
from copy import deepcopy
from .vcsf_final_training_state_contract import verify_payload_spec

PROTOCOL = "vcsf_final_oblivious_refresh"


def build_plan(registry, devices):
    spec = registry.protocols[PROTOCOL]
    if (not isinstance(devices, (list, tuple)) or len(devices) not in spec["device_counts"]
            or len(set(devices)) != len(devices)
            or any(type(device) is not str or not device.startswith("cuda:")
                   or not device[5:].isdigit() or str(int(device[5:])) != device[5:] for device in devices)):
        raise ValueError("Select one or two unique explicit CUDA devices")
    defenses = registry.preprocessing_defenses
    jobs = [dict(source=source, defense=defense, target=target,
                 device=devices[source_index % len(devices)], images=spec["retained_images"])
            for source_index, source in enumerate(spec["sources"])
            for defense in registry.preprocessing_defense_order for target in registry.paper_order]
    if len(jobs) != spec["evaluation_cells"]:
        raise ValueError("Incomplete registered preprocessing matrix")
    return dict(protocol=PROTOCOL, parameters_sha256=spec["parameters_sha256"],
        devices=list(devices), generation_jobs=0, jobs=jobs,
        defense_order=list(registry.preprocessing_defense_order), defense_parameters=deepcopy(defenses),
        transform_implementation=deepcopy(registry.preprocessing_defense_policy),
        threat_model=deepcopy(registry.preprocessing_defense_threat_model),
        clean_jobs=0, formal_execution_admitted=False)


def verify_source(registry, source, root):
    spec = registry.protocols[PROTOCOL]
    if source not in spec["sources"]:
        raise ValueError("Unregistered oblivious source")
    bound = dict(spec, source=source, **spec["payload_bindings"][source])
    result = verify_payload_spec(bound, root)
    positions = {image_id: position for position, image_id in enumerate(result["image_ids"])}
    result["retained_seed_mapping"] = [dict(image_id=image_id, full_position=positions[image_id],
        attack_seed=spec["seed"]+positions[image_id]) for image_id in result["retained_image_ids"]]
    result["source"] = source
    return result
