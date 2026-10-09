"""Validate the frozen adaptive invocation without relaxing legacy resolvers."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from ..defenses.preprocessing import AdaptivePreprocessor
from ..registry import AttackSpec
from ..runners.vcsf_research_plan import canonical_hash
from .factory import ATTACK_TYPES
from .vcsf_common_anchor_execution_isolation import build_common_anchor_execution
from .vcsf_common_anchor_isolated import VCSFCommonAnchorConfig


MODE = "final_adaptive_execution"
ALIAS = "vcsf_final_adaptive_isolated"


def validate_invocation(registry, binding, *, dataset_id, split, source_id,
                        attack_id, seed, max_images, parameter_overrides,
                        budget_profile, image_ids, seed_offsets, input_transform,
                        output_dir, expected_output, device):
    """The caller must authenticate binding, assets and execution permission first."""
    if ALIAS in registry.attacks or ALIAS in ATTACK_TYPES:
        raise ValueError("Adaptive execution must remain a process-local overlay")
    if (dataset_id != "coco" or split != "val" or attack_id != ALIAS
            or source_id != binding["source"] or type(seed) is not int
            or seed != binding["seed"] or device != "cuda:0"):
        raise ValueError("Adaptive invocation identity differs from bound group")
    if (type(max_images) is not type(binding["max_images"])
            or max_images != binding["max_images"]
            or budget_profile != binding["budget_profile"]):
        raise ValueError("Adaptive scope or cost accounting changed")
    ids = binding["image_ids"]
    if (not isinstance(image_ids, (list, tuple))
            or any(type(value) is not int for value in image_ids)
            or list(image_ids) != ids):
        raise ValueError("Adaptive ordered image selection differs")
    offsets = {row["image_id"]: row["offset"] for row in binding["seed_mapping"]}
    if (not isinstance(seed_offsets, dict)
            or any(type(key) is not int or type(value) is not int for key, value in seed_offsets.items())
            or seed_offsets != offsets):
        raise ValueError("Adaptive full-position seed offsets differ")
    if (type(input_transform) is not AdaptivePreprocessor
            or canonical_hash(input_transform.spec) != canonical_hash(binding["defense_parameters"])):
        raise ValueError("Adaptive transform implementation or parameters differ")
    if (not isinstance(parameter_overrides, dict)
            or canonical_hash(parameter_overrides) != binding["parameters_sha256"]
            or canonical_hash(binding["parameters"]) != binding["parameters_sha256"]):
        raise ValueError("Adaptive parameter override changes the frozen method")
    config = VCSFCommonAnchorConfig.from_mapping(deepcopy(parameter_overrides))
    config.validate()
    if canonical_hash(asdict(config)) != binding["parameters_sha256"]:
        raise ValueError("Adaptive configuration normalization changes parameters")
    if output_dir is None or Path(output_dir).resolve() != Path(expected_output).resolve():
        raise ValueError("Adaptive output escaped its assigned group leaf")


def invocation_spec(binding):
    """Build metadata only after validated invocation and separate run admission."""
    return AttackSpec(id=ALIAS, display_name="VCSF", executor="native",
        parameters=deepcopy(binding["parameters"]), metadata=dict(
            protocol=binding["protocol"], group_index=binding["group_index"],
            defense=binding["defense"], parameters_sha256=binding["parameters_sha256"],
            fidelity_status="frozen_parameters_pending_independent_result_acceptance",
            fidelity_basis="bound_common_anchor_with_registered_source_pipeline_bpda",
            semantic_contract="retained500_source_pipeline_adaptive_not_independent_confirmation",
            global_registry_mutated=False, global_factory_mutated=False,
            formal_AP_eligible=False))


build_final_adaptive = build_common_anchor_execution


def resolve_final_adaptive(registry, descriptor, *, run_metadata, output_dir,
                           device, gpu_owner, **arguments):
    """Authenticate a worker request; formal admission remains a separate gate."""
    import os
    from ..io import file_digest
    from ..runners.training_pair_processes import process_identity
    from ..runners.vcsf_final_adaptive_contract import build_plan, generation_binding
    from ..runners.vcsf_final_training_state import source_identity
    from ..runners.vcsf_oblivious_admission import bound_json, isolate_output
    from ..runners.vcsf_structure_workers import verify_reservation
    from ..runners.vcsf_gpu_context_owner import RegisteredGpuOwner

    fields = {"mode", "execution_alias", "worker_request", "worker_request_sha256", "group_index"}
    if (not isinstance(descriptor, dict) or set(descriptor) != fields
            or descriptor["mode"] != MODE or descriptor["execution_alias"] != ALIAS):
        raise ValueError("Unsupported final adaptive descriptor")
    request = bound_json(descriptor["worker_request"], descriptor["worker_request_sha256"])
    permit_ref = request["permit"]
    permit = bound_json(permit_ref["file"], permit_ref["sha256"])
    if permit["protocol"] != "vcsf_final_adaptive_refresh":
        raise ValueError("Adaptive execution permit namespace differs")
    limit = permit["max_images"]
    if permit["status"] == "approved_adaptive_diagnostic":
        if type(limit) is not int or not 0 < limit <= 500:
            raise ValueError("Adaptive diagnostic image limit is invalid")
    else:
        from ..runners.vcsf_adaptive_admission import verify_formal_permit
        verify_formal_permit(registry, permit)
    plan = build_plan(registry, permit["devices"])
    if permit["plan"] != plan or source_identity(registry.root) != permit["source_identity"]:
        raise ValueError("Adaptive registered plan or execution source changed")
    group_index = descriptor["group_index"]
    if (type(group_index) is not int or group_index not in permit["group_indices"]
            or group_index not in request["group_indices"]):
        raise ValueError("Adaptive group is not assigned to this worker")
    terminal_ref = permit["prepared"]
    terminal_path = Path(terminal_ref["file"]).resolve()
    terminal = bound_json(terminal_path, terminal_ref["sha256"])
    if terminal["status"] != "inputs_prepared_pending_execution_qualification":
        raise ValueError("Adaptive input preparation did not complete")
    prepared_files = {}
    for name, digest in terminal["files"].items():
        path = (terminal_path.parent / name).resolve()
        if path.parent != terminal_path.parent or file_digest(path) != digest:
            raise ValueError("Adaptive input preparation file changed or escaped")
        prepared_files[name] = path
    source = plan["generation_groups"][group_index]["source"]
    prepared = bound_json(prepared_files[source + ".json"], terminal["files"][source + ".json"])
    binding = generation_binding(registry, plan, prepared, group_index, max_images=limit)
    if request["device"] != binding["device"] or permit["gpu_uuids"][request["device"]] != request["gpu_uuid"]:
        raise ValueError("Adaptive worker device assignment changed")
    own = process_identity(os.getpid())
    parent = process_identity(os.getppid())
    coordinator = request["coordinator"]
    if (parent["pid"] != coordinator["pid"] or parent["start_ticks"] != coordinator["start_ticks"]
            or own["process_group"] != own["pid"] or own["session_id"] != own["pid"]
            or os.environ.get("CUDA_VISIBLE_DEVICES") != request["gpu_uuid"]):
        raise ValueError("Adaptive worker ownership or GPU visibility differs")
    verify_reservation(request)
    if (type(gpu_owner) is not RegisteredGpuOwner
            or canonical_hash(gpu_owner._request) != canonical_hash(request)):
        raise ValueError("Adaptive GPU context owner is missing or bound to another request")
    gpu_owner.verify()
    output = Path(permit["output"]).resolve()
    request_path = Path(descriptor["worker_request"]).resolve()
    if output not in request_path.parents:
        raise ValueError("Adaptive worker request escaped its execution root")
    assets_ref = permit["assets"]
    assets = bound_json(assets_ref["file"], assets_ref["sha256"])
    payload_roots = []
    for source_name in registry.protocols[plan["paired_protocol"]]["sources"]:
        item = bound_json(prepared_files[source_name + ".json"], terminal["files"][source_name + ".json"])
        payload_roots.append(Path(item["inputs"]["payload"]))
    isolate_output(output, payload_roots + [terminal_path.parent, Path(assets_ref["file"]).parent,
                           registry.root / "data", registry.root / "checkpoints"])
    expected_metadata = dict(protocol=plan["protocol"], group_index=group_index,
        defense=binding["defense"], permit_sha256=permit_ref["sha256"],
        assets_sha256=assets_ref["sha256"], diagnostic_only=limit is not None)
    if run_metadata != expected_metadata:
        raise ValueError("Adaptive run metadata differs from assigned invocation")
    validate_invocation(registry, binding, output_dir=output_dir,
        expected_output=output / "groups" / "{:06d}".format(group_index) / "attack",
        device=device, **arguments)
    from ..data.coco import CocoIndex, select_coco_images
    from ..modeling import checkpoint_path
    index = CocoIndex(registry.dataset("coco"), "val")
    if (str(index.annotation_path) != assets["annotation"]["file"]
            or file_digest(index.annotation_path) != assets["annotation"]["sha256"]):
        raise ValueError("Adaptive clean annotation changed")
    clean = {row["image_id"]: row["input"] for row in assets["clean"]}
    for image in select_coco_images(index.images, image_ids=binding["image_ids"]):
        path = index.image_path(image)
        expected = clean[image["id"]]
        if (str(path) != expected["file"] or path.stat().st_size != expected["bytes"]
                or file_digest(path) != expected["sha256"]):
            raise ValueError("Adaptive actual clean image changed")
    checkpoint = checkpoint_path(registry.model(source), registry.dataset("coco")).resolve()
    expected = assets["checkpoints"][source]
    if str(checkpoint) != expected["file"] or file_digest(checkpoint) != expected["sha256"]:
        raise ValueError("Adaptive actual source checkpoint changed")
    return invocation_spec(binding)
