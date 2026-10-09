"""Per-group CPU acceptance and explicitly authorized, non-recursive retirement."""
import json
import hashlib
from contextlib import ExitStack
import os
from pathlib import Path, PureWindowsPath
import stat
import subprocess
import sys
import uuid

from ..io import atomic_json, file_digest
from .vcsf_research_plan import canonical_hash


DELETE_POLICY = "delete_new_images_after_group_acceptance_no_backup"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _no_links(path):
    path = Path(path).absolute()
    for item in [path] + list(path.parents):
        info = item.lstat()
        _require(not stat.S_ISLNK(info.st_mode)
                 and not getattr(info, "st_file_attributes", 0) & 0x400,
                 "Symlink/reparse point is not owned payload: " + str(item))
    return path


def owned_image(payload, name):
    """Validate lexical ownership before resolving, including Windows paths on POSIX."""
    _require(isinstance(name, str) and bool(name) and "\\" not in name, "Invalid image path")
    relative = Path(name)
    _require(not relative.is_absolute() and not PureWindowsPath(name).drive
             and all(part not in ("", ".", "..") for part in name.split("/"))
             and len(relative.parts) >= 2 and relative.parts[0] == "images"
             and relative.suffix.lower() == ".png", "Image is not under owned attack/images")
    payload = _no_links(payload).resolve()
    path = _no_links(payload / relative)
    _require(path.resolve() == path and payload / "images" in path.parents
             and path.is_file(), "Image escaped owned payload")
    _require(path.stat().st_nlink == 1, "Hardlinked payload cannot be retired")
    return path


def _read(path, digest=None):
    path = Path(path)
    before = file_digest(path)
    _require(digest is None or digest == before, "Bound file changed: " + str(path))
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(file_digest(path) == before, "Bound file changed while reading")
    return value


def _qualified(receipt, permit, binding, permit_ref, registry):
    expected = dict(status="independently_verified_radius_group", protocol=permit["protocol"],
        group_index=binding["group_index"], source=binding["source"], epsilon=binding["epsilon"],
        image_count=len(binding["image_ids"]), cells=16, metric_values=192, model_calls=0,
        formal_result_eligible=True, scientific_acceptance=False, whole_program_acceptance=False,
        diagnostic_only=False, full_scope=True, permit=permit_ref, permit_sha256=permit_ref["sha256"],
        tool_sha256=permit["acceptance_tool"]["sha256"])
    _require(all(type(receipt.get(k)) is type(v) and receipt.get(k) == v for k, v in expected.items()),
             "Group acceptance receipt is not qualified for retirement")
    _require(isinstance(receipt.get("generation_binding"), dict)
             and canonical_hash(receipt["generation_binding"]) == canonical_hash(binding),
             "Receipt generation binding differs")
    results = receipt["results"]
    _require(len(results) == 16 and [row["target"] for row in results] == list(registry.paper_order)
             and all(row["group_index"] == binding["group_index"] and row["source"] == binding["source"]
                     and row["epsilon"] == binding["epsilon"] for row in results), "Receipt target panel differs")


def _require_descriptor_platform():
    _require(os.name == "posix" and hasattr(os, "O_DIRECTORY") and hasattr(os, "O_NOFOLLOW")
             and os.open in os.supports_dir_fd and os.unlink in os.supports_dir_fd
             and os.rename in os.supports_dir_fd and os.mkdir in os.supports_dir_fd
             and os.stat in os.supports_dir_fd and os.stat in os.supports_follow_symlinks,
             "Retirement requires POSIX no-follow descriptor-relative operations; no fallback")


def _open_directory(path, stack):
    """Walk from the filesystem root without following any path component link."""
    path = Path(path).absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    fd = os.open(path.anchor, flags)
    stack.callback(os.close, fd)
    for part in path.parts[1:]:
        fd = os.open(part, flags, dir_fd=fd)
        stack.callback(os.close, fd)
    return fd


def _file_identity(directory_fd, name):
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "Non-regular/linked retirement file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 8 << 20)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
        _require(all(getattr(before, k) == getattr(after, k) for k in fields), "Image changed while hashing descriptor")
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _require(all(getattr(after, k) == getattr(named, k) for k in fields), "Image name no longer binds opened inode")
        return dict(device=after.st_dev, inode=after.st_ino, bytes=after.st_size, sha256=digest.hexdigest())
    finally:
        os.close(fd)


def _retirement_counts(rows):
    confirmed = sum(row.get("deletion_outcome") == "confirmed_deleted" for row in rows)
    unknown = sum(row.get("deletion_outcome") == "unknown" for row in rows)
    return dict(deleted_images=None if unknown else confirmed,
                confirmed_deleted_images=confirmed, unknown_deletion_images=unknown,
                staged_retained_images=sum(row.get("stage_returned") is True
                    and row.get("deletion_outcome") == "not_deleted" for row in rows),
                unknown_stage_images=sum(row.get("stage_attempted") is True
                    and row.get("stage_returned") is None for row in rows))


def _retire(payload, receipt, destination):
    """Atomically quarantine each name, verify the moved inode, then unlink privately."""
    _require_descriptor_platform()
    payload = _no_links(payload).resolve()
    run_path = payload / "run.json"
    evidence = receipt["evidence"]
    run = _read(run_path, evidence[str(run_path)])
    manifest = payload / "manifest.jsonl"
    _require(file_digest(manifest) == run["manifest_sha256"] == evidence[str(manifest)], "Manifest changed")
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = receipt["image_evidence"]
    _require([row["image_id"] for row in rows] == [row["image_id"] for row in images], "Retirement population differs")
    plan, outcomes = [], []
    try:
        with ExitStack() as stack:
            expected_anchor = payload.stat()
            anchor = _open_directory(payload, stack)
            actual_anchor = os.fstat(anchor)
            _require((expected_anchor.st_dev, expected_anchor.st_ino) == (actual_anchor.st_dev, actual_anchor.st_ino),
                     "Owned payload anchor changed")
            directories = {(): anchor}
            for row, accepted in zip(rows, images):
                path = owned_image(payload, row["output_file"])
                parts = Path(row["output_file"]).parts
                for length in range(1, len(parts)):
                    key = parts[:length]
                    if key not in directories:
                        fd = os.open(parts[length-1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                     dir_fd=directories[key[:-1]])
                        stack.callback(os.close, fd)
                        directories[key] = fd
                identity = _file_identity(directories[parts[:-1]], parts[-1])
                _require(str(path) == accepted["file"]
                         and identity["bytes"] == row["output_bytes"] == accepted["bytes"]
                         and identity["sha256"] == row["output_sha256"] == accepted["sha256"] == evidence[str(path)],
                         "Retirement file differs from independent acceptance")
                plan.append(dict(image_id=row["image_id"], file=str(path), relative=row["output_file"],
                    state="planned", unlink_returned=False, deletion_outcome="not_deleted", **identity))
            _require(len(plan) == receipt["image_count"] and len({row["file"] for row in plan}) == len(plan),
                     "Retirement paths are missing or duplicated")
            staging_name = ".retirement-staging-" + uuid.uuid4().hex
            os.mkdir(staging_name, mode=0o700, dir_fd=anchor)
            staging_fd = os.open(staging_name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=anchor)
            stack.callback(os.close, staging_fd)
            staging_stat = os.fstat(staging_fd)
            _require(stat.S_IMODE(staging_stat.st_mode) == 0o700 and staging_stat.st_uid == os.geteuid()
                     and staging_stat.st_dev == actual_anchor.st_dev, "Staging directory is not private and owned")
            os.fsync(anchor)
            for index, row in enumerate(plan):
                row.update(staging_directory=str(payload / staging_name), staging_name=str(index) + ".png",
                    staging_directory_identity=dict(device=staging_stat.st_dev, inode=staging_stat.st_ino),
                    stage_attempted=False, stage_returned=False)
            atomic_json(destination / "disposition_plan.json", dict(policy=DELETE_POLICY, files=plan,
                limitation="Hashes identify prior bytes but do not restore them. No backup is created or verified by this operation."))
            outcomes = [dict(row) for row in plan]
            for index, row in enumerate(plan):
                state_path = destination / ("file-{:06d}.json".format(index))
                parts = Path(row["relative"]).parts
                parent_fd, name = directories[parts[:-1]], parts[-1]
                current = outcomes[index]
                try:
                    expected = {key: row[key] for key in ("device", "inode", "bytes", "sha256")}
                    _require(_file_identity(parent_fd, name) == expected, "Image changed before staging intent")
                    current.update(state="stage_intent", stage_attempted=True, stage_returned=None)
                    atomic_json(state_path, dict(current, interruption_semantics=
                        "A stage intent may have moved the file; inspect the private staging directory, never retry automatically."))
                    # A final-name swap can move the wrong inode, but cannot cause its deletion:
                    # verification happens only after atomic rename into our private directory.
                    os.rename(name, row["staging_name"], src_dir_fd=parent_fd, dst_dir_fd=staging_fd)
                    current.update(state="staged_pending_verification", stage_returned=True)
                    atomic_json(state_path, current)
                    os.fsync(parent_fd)
                    os.fsync(staging_fd)
                    moved = os.stat(row["staging_name"], dir_fd=staging_fd, follow_symlinks=False)
                    current["staged_observed_inode"] = dict(device=moved.st_dev, inode=moved.st_ino,
                        bytes=moved.st_size, mode=moved.st_mode)
                    actual = _file_identity(staging_fd, row["staging_name"])
                    current["staged_observed_identity"] = actual
                    _require(actual == expected, "Staged image identity mismatch; retained without deletion")
                    current.update(state="unlink_intent", unlink_returned=None, deletion_outcome="unknown")
                    atomic_json(state_path, dict(current, interruption_semantics=
                        "Without a final state the unlink outcome is unknown; never automatically retry."))
                    _require(_file_identity(staging_fd, row["staging_name"]) == expected,
                             "Private staged image changed before unlink")
                    os.unlink(row["staging_name"], dir_fd=staging_fd)
                    current.update(state="deleted", unlink_returned=True, deletion_outcome="confirmed_deleted",
                                   directory_fsync_returned=False)
                    os.fsync(staging_fd)
                    current["directory_fsync_returned"] = True
                    atomic_json(state_path, current)
                except BaseException as exc:
                    current.update(state="interrupted_or_failed", error=repr(exc), automatic_retry_allowed=False)
                    try:
                        atomic_json(state_path, current)
                    except BaseException as log_error:
                        current["disposition_write_error"] = repr(log_error)
                    raise
            _require(file_digest(run_path) == evidence[str(run_path)], "Original run metadata changed")
            return len(plan)
    except BaseException as exc:
        # In-memory evidence survives a failed disposition write and feeds the lifecycle receipt.
        exc.retirement_summary = dict(_retirement_counts(outcomes), file_outcomes=outcomes)
        raise


def accept_and_retire_group(registry, permit, binding):
    """FORMAL workers wait for a pinned CPU audit; diagnostics never invoke it."""
    if permit.get("status") != "approved_budget_formal":
        _require(permit.get("status") == "approved_budget_diagnostic", "Unknown permit status")
        return dict(status="diagnostic_no_worker_audit_no_deletion", deleted_images=0)
    _require(permit.get("max_images", "missing") is None and binding.get("max_images", "missing") is None,
             "Retirement requires explicit formal scope")
    root = _no_links(permit["output"]).resolve()
    index = binding["group_index"]
    _require(type(index) is int and index >= 0, "Invalid group index")
    groups = [row for row in permit["plan"]["generation_groups"] if row["index"] == index]
    _require(len(groups) == 1 and groups[0]["action"] == "generate_and_evaluate"
             and groups[0]["source"] == binding["source"] and groups[0]["epsilon"] == binding["epsilon"]
             and binding["epsilon"] != permit["plan"]["reference_epsilon"], "Cannot retire reference groups")
    permit_ref = _read(root / "permit.json")
    _require(_read(permit_ref["file"], permit_ref["sha256"]) == permit, "Startup permit differs")
    tool = Path(permit["acceptance_tool"]["file"]).resolve()
    _require(tool == (registry.root / "tools" / "audit_vcsf_budget_group.py").resolve()
             and file_digest(tool) == permit["acceptance_tool"]["sha256"], "Unbound acceptance executable")
    group = _no_links(root / "groups" / "{:06d}".format(index)).resolve()
    terminal = _read(group / "terminal.json")
    _require(terminal.get("status") == "complete_pending_independent_acceptance"
             and terminal.get("formal_result_eligible") is False
             and terminal.get("scientific_acceptance") is False, "Failed/incomplete groups cannot be retired")
    lifecycle = group / "lifecycle"
    lifecycle.mkdir(exist_ok=False)
    output = root.parent / (root.name + ".group-audit-{:06d}-".format(index) + uuid.uuid4().hex)
    command = [sys.executable, str(tool), "--source", str(registry.root.resolve()),
        "--permit", permit_ref["file"], "--permit-sha256", permit_ref["sha256"],
        "--group-index", str(index), "--output", str(output)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=str(registry.root / "src"))
    state = dict(status="auditing", group_index=index, permit=permit_ref, audit_output=str(output),
        command=command, cuda_visible_devices="", scientific_acceptance=False,
        deleted_images=0, payload_state="available_before_retirement")
    atomic_json(lifecycle / "receipt.json", state)
    child = None
    try:
        with (lifecycle / "audit.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=str(registry.root), env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            atomic_json(lifecycle / "audit_process.json", dict(pid=child.pid, command=command))
            code = child.wait()
        atomic_json(lifecycle / "audit_exit.json", dict(exit_code=code, pid=child.pid))
        _require(code == 0, "Independent CPU audit failed; payload retained")
        acceptance_path = output / "receipt.json"
        accepted = _read(acceptance_path)
        _qualified(accepted, permit, binding, permit_ref, registry)
        _require(accepted["payload"] == str(group / "attack") and accepted["run"] == str(root)
                 and accepted["results"] == terminal["results"] == _read(group / "results.json"),
                 "Acceptance belongs to another payload/results")
        for path, digest in accepted["evidence"].items():
            _require(file_digest(Path(path)) == digest, "Acceptance evidence changed before retirement")
        state.update(status="accepted_payload_retained", acceptance=dict(file=str(acceptance_path),
            sha256=file_digest(acceptance_path)), payload_state="available", formal_result_eligible=True)
        atomic_json(lifecycle / "receipt.json", state)
        if permit.get("lifecycle") == DELETE_POLICY:
            state.update(status="retiring", payload_state="retirement_in_progress")
            atomic_json(lifecycle / "receipt.json", state)
            count = _retire(group / "attack", accepted, lifecycle)
            state.update(status="accepted_images_retired", deleted_images=count,
                payload_state="manifest_listed_attack_images_unavailable", full_payload_available=False,
                limitation="No backup was created or verified. Hashes are prior-byte identity records, not recoverable payloads.")
            atomic_json(lifecycle / "receipt.json", state)
        return state
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            atomic_json(lifecycle / "audit_exit.json", dict(exit_code=child.returncode,
                pid=child.pid, interrupted=True))
        state.update(status="failed_or_interrupted", error=repr(exc), automatic_retry_allowed=False)
        summary = getattr(exc, "retirement_summary", None)
        if summary is None:
            dispositions = [_read(path) for path in sorted(lifecycle.glob("file-*.json"))]
            summary = _retirement_counts(dispositions)
        state.update(summary)
        if state.get("payload_state") == "retirement_in_progress":
            state["payload_state"] = "retirement_incomplete_inspect_per_file_dispositions"
        atomic_json(lifecycle / "receipt.json", state)
        raise
