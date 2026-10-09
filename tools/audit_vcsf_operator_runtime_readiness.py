"""Read real max1 operator roots before admitting full5000 execution.

This audit performs mechanical raw-record and saved-pixel readback. It does not
rerun engineering tests, construct detectors, evaluate AP, or accept science.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from lgp.io import file_digest
from lgp.runners import vcsf_operator_execution_contract as contract
from lgp.runners import vcsf_efficacy_runner as runner
from lgp.runners.retention import _validate_prediction_archive
from lgp.runners.vcsf_research_plan import canonical_hash, require


AUDITOR = "tools/audit_vcsf_operator_runtime_readiness.py"
STATUS = contract.FORMAL_READINESS_STATUS
SCOPE = "real_max1_one_and_two_device_mechanical_readback_not_result_or_scientific_acceptance"
FOUR_DEVICE_STATUS = "independently_verified_four_device_research_launch_pending_acceptance"
FOUR_DEVICE_SCOPE = "real_max1_four_device_launch_not_full_release_or_result_acceptance"


def _verify_context_registration(state, identity):
    receipt = state.get("gpu_context_registration")
    require(isinstance(receipt, dict), "Missing GPU context registration evidence")
    require(receipt.get("schema") == "vcsf_gpu_context_registration_v1"
        and all(type(receipt.get(key)) is type(identity[key]) and receipt[key] == identity[key]
            for key in ("worker_pid", "worker_pid_start_ticks", "gpu_uuid"))
        and type(receipt.get("nvml_pid")) is int and receipt["nvml_pid"] > 0
        and receipt.get("before_context_pids") == []
        and canonical_hash(receipt.get("registered_context_pids")) == canonical_hash([receipt["nvml_pid"]])
        and receipt.get("assumption") == "NVML_enumerates_all_active_compute_context_processes_on_selected_device",
        "GPU context registration differs from the bound worker or exclusive transition")
    try:
        stamp = datetime.fromisoformat(receipt["registered_at"])
        require(stamp.utcoffset() is not None, "GPU context registration timestamp lacks timezone")
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Invalid GPU context registration timestamp") from exc


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return dict(file=str(path), sha256=file_digest(path))


def _ref(path):
    return dict(file=str(path), sha256=file_digest(path))


def _plain_root(registry, path):
    path = Path(path).absolute()
    require(path.parent == registry.root / "outputs/diagnostics" / contract.EXECUTION_PROTOCOL,
        "Runtime readiness requires a direct operator diagnostic root")
    return contract.child(registry.root, path.relative_to(registry.root))


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Runtime evidence contains a symlink")
        if path.is_file():
            require(path.name not in ("failure.json", "abort.json", "traceback.txt"),
                "Failed or aborted runtime evidence cannot admit full execution")
            files[path.relative_to(root).as_posix()] = file_digest(path)
    require(files, "Runtime diagnostic evidence is empty")
    return files


def _read(root, name, inventory):
    require(name in inventory, "Missing completed runtime evidence: " + name)
    return contract.read_bound(contract.child(root, name), inventory[name])


def _json_value(root, name, inventory):
    """Lists occur in runner records; retain pin checking before JSON decoding."""
    path = contract.child(root, name)
    require(name in inventory and file_digest(path) == inventory[name], "Runtime record bytes changed")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate runtime JSON record key")
            result[key] = value
        return result

    def nonfinite(value):
        raise RuntimeError("Nonfinite runtime JSON record: " + value)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=nonfinite)


def _same_science(formal, diagnostic):
    require(formal.get("max_images") is None and formal.get("diagnostic_only") is False
        and type(diagnostic.get("max_images")) is int and diagnostic["max_images"] == 1
        and diagnostic.get("diagnostic_only") is True,
        "Formal readiness requires one full plan and its exact max1 diagnostic plan")
    execution_only = {"max_images", "diagnostic_only", "new_images", "capacity_contract", "keep_all_policy_sha256"}
    require(canonical_hash({k: v for k, v in formal.items() if k not in execution_only})
        == canonical_hash({k: v for k, v in diagnostic.items() if k not in execution_only}),
        "Formal and max1 plans differ in science/source/input/reuse or execution method")


def _workers(registry, root, inventory, plan, plan_ref, admission_ref, binding, devices):
    groups = contract.execution_groups(plan, 1)
    stage = contract.scheduled_stages(plan, groups)[0]
    assignment = _read(root, "stages/operator_factorial/assignment.json", inventory)
    require(canonical_hash(assignment.get("stage")) == canonical_hash(stage), "Runtime stage changed")
    workers = assignment.get("workers")
    require(isinstance(workers, list) and len(workers) == len(devices), "Runtime worker count differs")
    lanes = contract.balanced_lanes(groups, plan["sources"], devices)
    pids, uuids, completed = set(), set(), []
    root_relative = root.relative_to(registry.root).as_posix()
    coordinator = _read(root, "execution_state.json", inventory)
    for slot, (worker, lane) in enumerate(zip(workers, lanes)):
        ids = [g["group_id"] for g in lane]
        prefix = "workers/operator_factorial/" + str(slot)
        state = _read(root, prefix + "/execution_state.json", inventory)
        request = _read(root, prefix + "/request.json", inventory)
        rows = _json_value(root, prefix + "/completed_groups.json", inventory)
        pid, uuid = worker.get("pid"), worker.get("gpu_uuid")
        require(type(pid) is int and pid > 0 and pid not in pids
            and isinstance(uuid, str) and uuid and uuid not in uuids,
            "Runtime workers must have distinct process and physical-device identities")
        pids.add(pid)
        uuids.add(uuid)
        require(type(worker.get("slot")) is int and worker["slot"] == slot
            and worker.get("directory") == root_relative + "/" + prefix
            and canonical_hash(worker.get("group_ids")) == canonical_hash(ids)
            and worker.get("physical_device") == devices[slot]
            and worker.get("request_sha256") == inventory[prefix + "/request.json"]
            and state.get("status") == "complete" and state.get("failed_records") == 0
            and state.get("completed_groups") == len(ids)
            and state.get("completed_evaluations") == len(ids) * len(plan["targets"])
            and state.get("expected_groups") == len(ids)
            and state.get("current") is None,
            "Runtime worker status, owned jobs or complete target counts differ")
        descriptor = request.get("descriptor", {})
        require(canonical_hash(request.get("devices")) == canonical_hash(devices)
            and canonical_hash(request.get("group_ids")) == canonical_hash(ids)
            and type(request.get("worker_slot")) is int and request["worker_slot"] == slot
            and request.get("stage") == stage["name"] and request.get("physical_device") == devices[slot]
            and request.get("gpu_uuid") == uuid
            and request.get("coordinator_pid") == coordinator.get("coordinator_pid")
            and request.get("coordinator_pid_start_ticks") == coordinator.get("coordinator_pid_start_ticks")
            and request.get("payload_policy_sha256") == binding["payload_policy_sha256"]
            and "retained_selection_sha256" not in request,
            "Runtime worker request lost its exact operator lane or keep-all policy")
        expected_descriptor = dict(candidate_id=contract.CANDIDATE_ID, execution_alias=contract.ALIAS,
            mode=contract.MODE, execution_plan=Path(plan_ref["file"]).relative_to(registry.root).as_posix(),
            execution_plan_sha256=plan_ref["sha256"],
            admission=Path(admission_ref["file"]).relative_to(registry.root).as_posix(),
            admission_sha256=admission_ref["sha256"], execution_root=root_relative,
            run_binding_sha256=inventory["plan_binding.json"], max_images=1)
        require(canonical_hash(descriptor) == canonical_hash(expected_descriptor),
            "Actual runtime worker descriptor differs from the qualified max1 invocation")
        identity = dict(worker_pid=pid, worker_pid_start_ticks=worker["pid_start_ticks"],
            coordinator_pid=request["coordinator_pid"], coordinator_pid_start_ticks=request["coordinator_pid_start_ticks"],
            worker_slot=slot, physical_device=devices[slot], gpu_uuid=uuid, stage=stage["name"],
            request_sha256=inventory[prefix + "/request.json"], execution_plan_sha256=plan_ref["sha256"])
        require(all(type(identity[k]) is int and identity[k] > 0 for k in (
            "worker_pid", "worker_pid_start_ticks", "coordinator_pid", "coordinator_pid_start_ticks"))
            and canonical_hash({k: state.get(k) for k in identity}) == canonical_hash(identity)
            and isinstance(rows, list) and canonical_hash([r.get("group_id") for r in rows]) == canonical_hash(ids)
            and all(canonical_hash({k: r.get(k) for k in identity}) == canonical_hash(identity) for r in rows),
            "Runtime process ownership or completed worker records disagree")
        _verify_context_registration(state, identity)
        completed.extend(rows)
    completed.sort(key=lambda row: row["group_id"])
    stage_completion = _read(root, "stages/operator_factorial/completion.json", inventory)
    require(stage_completion.get("status") == "complete_stage_mechanical" and stage_completion.get("errors") == []
        and stage_completion.get("assignment_sha256") == inventory["stages/operator_factorial/assignment.json"]
        and canonical_hash(stage_completion.get("stage")) == canonical_hash(stage)
        and canonical_hash(stage_completion.get("groups")) == canonical_hash(completed)
        and stage_completion.get("scientific_acceptance") is False,
        "Runtime stage has no matching terminal mechanical completion")
    return completed


def read_runtime_root(registry, diagnostic, diagnostic_plan_ref, engineering_ref, completion_ref, device_count):
    completion = contract.read_reference(registry, completion_ref)
    root = _plain_root(registry, Path(completion_ref["file"]).parent)
    require(Path(completion_ref["file"]).absolute() == root / "completion.json",
        "Runtime reference must pin the root completion.json")
    inventory = _inventory(root)
    require(inventory["completion.json"] == completion_ref["sha256"], "Runtime completion changed")
    binding = _read(root, "plan_binding.json", inventory)
    state = _read(root, "execution_state.json", inventory)
    require(all(type(state.get(key)) is int for key in (
        "max_images", "expected_groups", "completed_groups", "expected_evaluations",
        "completed_evaluations", "failed_records", "coordinator_pid", "coordinator_pid_start_ticks"))
        and all(type(completion.get(key)) is int for key in (
            "max_images", "groups", "evaluations", "generated_images")),
        "Runtime terminal counts and ownership must be exact integers")
    groups = contract.execution_groups(diagnostic, 1)
    ids = [g["group_id"] for g in groups]
    require(len(ids) == 4 and len(diagnostic["targets"]) == 16, "Runtime audit requires exact four-new/64-target scope")
    devices = binding.get("devices")
    require(isinstance(devices, list) and len(devices) == device_count, "Runtime root has another device count")
    contract.balanced_lanes(groups, diagnostic["sources"], devices)
    require(binding.get("execution_root") == root.relative_to(registry.root).as_posix()
        and binding.get("execution_plan_sha256") == diagnostic_plan_ref["sha256"]
        and binding.get("admission_sha256") == engineering_ref["sha256"]
        and type(binding.get("max_images")) is int and binding["max_images"] == 1
        and canonical_hash(binding.get("group_ids")) == canonical_hash(ids)
        and canonical_hash(binding.get("stages")) == canonical_hash(contract.scheduled_stages(diagnostic, groups))
        and binding.get("execution_lifecycle") == "full_stage_keep_all_v1"
        and canonical_hash(binding.get("payload_authorization")) == canonical_hash(diagnostic["payload_authorization"]),
        "Runtime binding changed plan/admission/scientific jobs or lifecycle")
    require(state.get("status") == "complete_pending_independent_acceptance"
        and state.get("protocol") == contract.EXECUTION_PROTOCOL
        and state.get("execution_plan_sha256") == diagnostic_plan_ref["sha256"]
        and type(state.get("max_images")) is int and state["max_images"] == 1
        and state.get("expected_groups") == state.get("completed_groups") == 4
        and state.get("expected_evaluations") == state.get("completed_evaluations") == 64
        and state.get("failed_records") == 0 and state.get("workers") == []
        and state.get("scientific_acceptance") is False,
        "Runtime root has not completed its complete max1 operator matrix")
    require(completion.get("status") == "complete_pending_independent_acceptance"
        and completion.get("errors") == [] and completion.get("groups") == 4
        and completion.get("evaluations") == 64 and completion.get("generated_images") == 4
        and type(completion.get("max_images")) is int and completion["max_images"] == 1
        and completion.get("execution_plan_sha256") == diagnostic_plan_ref["sha256"]
        and all(completion.get(k) is False for k in (
            "independent_confirmation", "scientific_acceptance", "formal_metrics_eligible")),
        "Runtime completion does not prove the exact max1 four-new/64-target diagnostic")
    completed = _workers(registry, root, inventory, diagnostic, diagnostic_plan_ref, engineering_ref, binding, devices)
    require(canonical_hash(completion.get("completed_groups")) == canonical_hash(completed),
        "Root completion differs from actual completed worker jobs")
    runner.verify_keep_all_groups(diagnostic, diagnostic_plan_ref["sha256"], root, completed, ids,
        binding["payload_policy_sha256"], 1, profile=contract)
    raw_records = []
    for group, result in zip(groups, completed):
        prefix = "groups/{:06d}".format(group["group_id"])
        attack = contract.child(root, prefix + "/attack")
        generation = runner.validate_generation(registry, diagnostic, diagnostic_plan_ref["sha256"],
            group, attack, 1, decode_pixels=True, profile=contract)
        require(generation["images"] == generation["decoded_png_count"] == 1
            and canonical_hash(result.get("generation")) == canonical_hash(generation),
            "Saved runtime generation receipt differs from raw PNG and generation record readback")
        group_records = []
        for target in diagnostic["targets"]:
            directory = contract.child(root, prefix + "/evaluations/" + target)
            record = _read(root, prefix + "/evaluations/" + target + "/metrics.json", inventory)
            runner.validate_target(diagnostic, group, target, record, generation, profile=contract)
            require(Path(record["adversarial_run"]).resolve() == attack.resolve(),
                "Runtime target record points to another generated payload")
            _validate_prediction_archive(directory, record)
            group_records.append(dict(record, variant=group["variant"], seed=group["seed"],
                group_id=group["group_id"], execution_protocol=contract.EXECUTION_PROTOCOL,
                execution_plan_sha256=diagnostic_plan_ref["sha256"],
                runtime_snapshot_sha256=diagnostic["runtime_snapshot_sha256"],
                independent_confirmation=False, formal_metrics_eligible=False))
        require(canonical_hash(_json_value(root, prefix + "/records.json", inventory)) == canonical_hash(group_records),
            "Runtime group records differ from the raw target panel")
        raw_records.extend(group_records)
    require(completion.get("records_sha256") == inventory["records.json"]
        and canonical_hash(_json_value(root, "records.json", inventory)) == canonical_hash(raw_records),
        "Runtime root records differ from the complete raw 64-target readback")
    manifest = _read(root, "artifact_manifest.json", inventory)
    require(completion.get("artifact_manifest_sha256") == inventory["artifact_manifest.json"]
        and manifest.get("artifacts") == {name: pin for name, pin in inventory.items()
            if not name.endswith((".png", ".log"))
            and name not in ("execution_state.json", "artifact_manifest.json", "completion.json")},
        "Runtime root artifact inventory differs from its sealed completion")
    require(_inventory(root) == inventory, "Runtime evidence changed during independent readback")
    return dict(completion=deepcopy(completion_ref), run_binding=_ref(root / "plan_binding.json"),
        device_count=device_count, devices=devices, group_ids=ids, generated_images=4,
        target_evaluations=64, raw_generation_groups_checked=4, raw_target_records_checked=64,
        decoded_pngs_checked=4, keep_all_verified=True, worker_ownership_verified=True,
        checked_files_sha256=inventory, inventory_sha256=canonical_hash(inventory),
        result_acceptance=False, formal_metrics_eligible=False, scientific_acceptance=False)


def compile_runtime_admission(registry, plan_ref, diagnostic_plan_ref, engineering_ref,
        one_device_completion_ref, two_device_completion_ref, *, four_device_completion_ref=None):
    formal = contract.read_reference(registry, plan_ref)
    diagnostic = contract.read_reference(registry, diagnostic_plan_ref)
    contract.verify_execution_plan(registry, formal)
    contract.verify_execution_plan(registry, diagnostic)
    contract.verify_environment(formal)
    _same_science(formal, diagnostic)
    engineering = contract.verify_engineering_admission(registry, diagnostic, diagnostic_plan_ref["sha256"],
        engineering_ref["file"], engineering_ref["sha256"], 1)
    four_device = four_device_completion_ref is not None
    if four_device:
        require(one_device_completion_ref is None and two_device_completion_ref is None,
            "Four-device research launch must not mix unrelated runtime modes")
        contract.verify_device_admission(engineering, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
        completions = [(four_device_completion_ref, 4)]
    else:
        require(Path(one_device_completion_ref["file"]).absolute().parent
            != Path(two_device_completion_ref["file"]).absolute().parent,
            "One- and two-device readiness must come from two distinct immutable execution roots")
        contract.verify_device_admission(engineering, ["cuda:0"])
        contract.verify_device_admission(engineering, ["cuda:0", "cuda:1"])
        completions = [(one_device_completion_ref, 1), (two_device_completion_ref, 2)]
    snapshot = formal["runtime_sha256"]
    contract.verify_process_source(registry, snapshot)
    require(Path(__file__).resolve() == registry.root / AUDITOR
        and snapshot[AUDITOR] == file_digest(registry.root / AUDITOR), "Runtime readiness auditor source changed")
    readbacks = [read_runtime_root(registry, diagnostic, diagnostic_plan_ref, engineering_ref, ref, count)
        for ref, count in completions]
    for readback in readbacks:
        require(_inventory(Path(readback["completion"]["file"]).parent) == readback["checked_files_sha256"],
            "One of the runtime roots changed while the other root was being audited")
    contract.verify_process_source(registry, snapshot)
    for ref in [plan_ref, diagnostic_plan_ref, engineering_ref] + [ref for ref, _ in completions]:
        contract.read_reference(registry, ref)
    receipt = dict(schema_version=1, status=FOUR_DEVICE_STATUS if four_device else STATUS,
        execution_plan=deepcopy(plan_ref),
        diagnostic_plan=deepcopy(diagnostic_plan_ref), engineering_admission=deepcopy(engineering_ref),
        one_device_completion=deepcopy(one_device_completion_ref), two_device_completion=deepcopy(two_device_completion_ref),
        auditor_source=_ref(registry.root / AUDITOR), runtime_snapshot_sha256=formal["runtime_snapshot_sha256"],
        science_sha256=formal["science_sha256"], input_binding_sha256=formal["input_binding_sha256"],
        reuse_audit=deepcopy(formal["reuse_audit"]), max_images=None,
        device_counts=[4] if four_device else [1, 2],
        scope=FOUR_DEVICE_SCOPE if four_device else SCOPE, runtime_readbacks=readbacks, execution_admitted=True,
        engineering_checks_rerun=False, new_detector_calls=0, new_AP_evaluations=0,
        result_acceptance=False, formal_metrics_eligible=False, scientific_acceptance=False,
        independent_confirmation=False, automatic_promotion=False, errors=[])
    if four_device:
        del receipt["one_device_completion"], receipt["two_device_completion"]
        receipt.update(four_device_completion=deepcopy(four_device_completion_ref),
            full_release_acceptance=False, research_results_pending_acceptance=True)
    return receipt


def verify_runtime_admission(registry, plan, plan_hash, path, digest):
    path = Path(path).absolute()
    _plain_root(registry, path.parent)
    require(path.name in ("formal_admission.json", "research_launch_admission.json")
        and not (path.parent / "failure.json").exists()
        and not (path.parent / "failure.json").is_symlink(), "Failed or foreign formal readiness publication")
    receipt = contract.read_bound(path, digest)
    four_device = receipt.get("status") == FOUR_DEVICE_STATUS
    require(receipt.get("status") in (STATUS, FOUR_DEVICE_STATUS) and receipt.get("max_images") is None
        and path.name == ("research_launch_admission.json" if four_device else "formal_admission.json"),
        "Full5000 requires independently read back real max1 worker roots, not engineering tests")
    require(receipt["execution_plan"]["sha256"] == plan_hash
        and canonical_hash(contract.read_reference(registry, receipt["execution_plan"])) == canonical_hash(plan),
        "Formal readiness belongs to another execution plan")
    rebuilt = compile_runtime_admission(registry, receipt["execution_plan"], receipt["diagnostic_plan"],
        receipt["engineering_admission"], receipt.get("one_device_completion"), receipt.get("two_device_completion"),
        four_device_completion_ref=receipt.get("four_device_completion") if four_device else None)
    require(canonical_hash(receipt) == canonical_hash(rebuilt), "Formal readiness or real runtime evidence changed")
    require(not (path.parent / "failure.json").exists()
        and not (path.parent / "failure.json").is_symlink()
        and canonical_hash(contract.read_bound(path, digest)) == canonical_hash(receipt),
        "Formal runtime-readiness publication changed or failed during verification")
    return receipt


def audit_runtime_readiness(registry, plan_path, plan_sha256, diagnostic_plan_ref,
        engineering_admission_ref, one_device_completion_ref, two_device_completion_ref, output,
        *, four_device_completion_ref=None):
    """Publish a complete derived formal admission, never start or rerun a job."""
    plan_ref = dict(file=str(Path(plan_path).absolute()), sha256=plan_sha256)
    formal = contract.read_reference(registry, plan_ref)
    contract.verify_execution_plan(registry, formal)
    contract.verify_environment(formal)
    output = _plain_root(registry, output)
    output.mkdir(parents=True, exist_ok=False)
    four_device = four_device_completion_ref is not None
    _write(output / "request.json", dict(execution_plan=plan_ref, diagnostic_plan=diagnostic_plan_ref,
        engineering_admission=engineering_admission_ref, one_device_completion=one_device_completion_ref,
        two_device_completion=two_device_completion_ref, four_device_completion=four_device_completion_ref,
        scope=FOUR_DEVICE_SCOPE if four_device else SCOPE))
    try:
        receipt = compile_runtime_admission(registry, plan_ref, diagnostic_plan_ref, engineering_admission_ref,
            one_device_completion_ref, two_device_completion_ref,
            four_device_completion_ref=four_device_completion_ref)
        reference = _write(output / ("research_launch_admission.json" if four_device else "formal_admission.json"), receipt)
        verify_runtime_admission(registry, formal, plan_sha256, reference["file"], reference["sha256"])
        _write(output / "terminal.json", dict(status=receipt["status"], admission=reference,
            completed_at=datetime.now(timezone.utc).isoformat(), scope=receipt["scope"], errors=[],
            result_acceptance=False, scientific_acceptance=False))
        return reference
    except BaseException as exc:
        _write(output / "failure.json", dict(status="failed_preserved_no_formal_admission",
            error_type=type(exc).__name__, error=str(exc), execution_plan=plan_ref,
            result_acceptance=False, scientific_acceptance=False))
        raise
