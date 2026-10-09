"""Independent CPU replay of one completed radius group while its worker is live."""
import argparse
import ast
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def normalized_model_config(value):
    """Same declarative rules as bind_vcsf_radius_evaluators; no model builder."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("Non-string configuration key")
        return {key: normalized_model_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized_model_config(item) for item in value]
    if value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("Non-declarative configuration value")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Non-finite configuration value")
    if isinstance(value, str) and value.startswith("/"):
        return str(Path(value).resolve())
    return value


def evaluator_reference(permit, override):
    bound = permit.get("evaluator_receipt")
    if override is not None and bound is not None:
        if (Path(override["file"]).resolve() != Path(bound["file"]).resolve()
                or override["sha256"] != bound["sha256"]):
            raise ValueError("Evaluator override differs from permit")
    if permit["status"] == "approved_budget_formal" and bound is None:
        raise ValueError("Formal permit lacks evaluator receipt")
    ref = bound if bound is not None else override
    if ref is None:
        raise ValueError("An approved evaluator receipt is required for saved-config acceptance")
    return dict(file=str(Path(ref["file"]).resolve()), sha256=ref["sha256"])


def parse_declarative_config(text):
    """Parse MMEngine's resolved pretty-text data without executing Python."""
    if len(text) > 16 * 1024 * 1024:
        raise ValueError("Saved config is too large")
    tree = ast.parse(text, mode="exec")
    if sum(1 for _ in ast.walk(tree)) > 100000:
        raise ValueError("Saved config has too many syntax nodes")

    def value(node, depth=0):
        if depth > 128:
            raise ValueError("Saved config nesting is too deep")
        if isinstance(node, ast.Constant) and (node.value is None or type(node.value) in (str, bool, int, float)):
            if type(node.value) is float and not math.isfinite(node.value):
                raise ValueError("Non-finite config literal")
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            items = [value(item, depth + 1) for item in node.elts]
            return tuple(items) if isinstance(node, ast.Tuple) else items
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            item = value(node.operand, depth + 1)
            if type(item) not in (int, float):
                raise ValueError("Signed config literal is not numeric")
            return -item if isinstance(node.op, ast.USub) else item
        if isinstance(node, ast.Dict):
            pairs = [(value(key, depth + 1), value(item, depth + 1)) for key, item in zip(node.keys, node.values)]
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "dict" and not node.args):
            # Interpret this syntax ourselves; never call a name from the config.
            pairs = [(keyword.arg, value(keyword.value, depth + 1)) for keyword in node.keywords]
        else:
            raise ValueError("Non-declarative config syntax: " + type(node).__name__)
        result = {}
        for key, item in pairs:
            if type(key) is not str or key in result:
                raise ValueError("Config dictionary key is non-string, expanded or duplicated")
            result[key] = item
        return result

    result = {}
    for statement in tree.body:
        if (not isinstance(statement, ast.Assign) or len(statement.targets) != 1
                or not isinstance(statement.targets[0], ast.Name)):
            raise ValueError("Only declarative top-level config assignments are allowed")
        name = statement.targets[0].id
        if name in result or name in ("dict", "_base_") or name.startswith("__"):
            raise ValueError("Duplicate, shadowing or executable config directive")
        result[name] = value(statement.value)
    return result


def verify_actual_config(path, approved_hash, bind):
    from lgp.runners.vcsf_research_plan import canonical_hash
    path = bind(path)
    value = parse_declarative_config(path.read_text(encoding="utf-8"))
    value.pop("work_dir", None)
    digest = canonical_hash(normalized_model_config(value))
    if digest != approved_hash:
        raise ValueError("Saved actual runtime config differs from approved evaluator: " + str(path))
    bind(path)
    return digest


def audit(registry, permit_path, permit_sha256, group_index, output, evaluator_ref=None):
    import numpy as np
    from PIL import Image
    from lgp.io import atomic_json, file_digest
    from lgp.runners.vcsf_final_budget import read_permit
    from lgp.runners.vcsf_budget_lifecycle import owned_image
    from lgp.runners.vcsf_final_training_state import verify_evaluation
    from lgp.runners.vcsf_final_training_state_contract import ids_digest
    from lgp.runners.vcsf_research_plan import canonical_hash
    from lgp.runners.preprocessing_defense import _subset_annotation
    from lgp.reporting.coco_subset import evaluate_bbox_subset
    from lgp.reporting.vcsf_training_state_audit import (
        require, compare_metrics, verify_owner, verify_archive_metadata,
    )
    from lgp.reporting.vcsf_oblivious_audit import (
        require_isolated, bind_source_snapshot, verify_evidence_unchanged,
    )

    output = Path(output).resolve()
    permit_path = Path(permit_path).resolve()
    require(not output.exists(), "Audit output already exists")
    # Establish isolation before writing even failure evidence.
    untrusted = json.loads(permit_path.read_text(encoding="utf-8"))
    root = Path(untrusted["output"]).resolve()
    for ref in (untrusted.get("evaluator_receipt"), evaluator_ref):
        if ref is not None:
            require_isolated(output, Path(ref["file"]).parent)
    for path in (root, registry.root, permit_path.parent,
                 Path(untrusted["prepared"]["file"]).parent,
                 Path(untrusted["assets"]["file"]).parent,
                 Path(untrusted["checkpoint_bindings"]["file"]).parent):
        require_isolated(output, path)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {}
    def bind(path, expected=None):
        path = Path(path).resolve()
        require_isolated(output, path)
        digest = file_digest(path)
        require(expected is None or digest == expected, "Pinned evidence changed: " + str(path))
        require(str(path) not in evidence or evidence[str(path)] == digest,
                "Previously read evidence changed")
        evidence[str(path)] = digest
        return path
    def read(path, expected=None):
        path = bind(path, expected)
        value = json.loads(path.read_text(encoding="utf-8"))
        require(file_digest(path) == evidence[str(path)], "Evidence changed while reading")
        return value
    try:
        atomic_json(output / "state.json", dict(status="auditing", group_index=group_index,
            permit=dict(file=str(permit_path), sha256=permit_sha256), scientific_acceptance=False))
        read(permit_path, permit_sha256)
        permit, bindings = read_permit(registry, permit_path, permit_sha256)
        selected = [row for row in bindings if row["group_index"] == group_index]
        require(type(group_index) is int and len(selected) == 1, "Group not uniquely admitted")
        binding = selected[0]
        ids = binding["image_ids"]
        formal = permit["status"] == "approved_budget_formal" and permit["max_images"] is None
        evaluator_ref = evaluator_reference(permit, evaluator_ref)
        evaluator = read(evaluator_ref["file"], evaluator_ref["sha256"])
        require(evaluator["status"] == "radius_target_configs_match_historical_pending_reuse"
                and evaluator["normalization"] == "exclude_work_dir_resolve_absolute_paths_json_sequences"
                and [row["model"] for row in evaluator["comparisons"]] == list(registry.paper_order),
                "Approved evaluator scope/normalization differs")
        bind(registry.root / "tools" / "bind_vcsf_radius_evaluators.py", evaluator["tool_sha256"])
        for path, digest in evaluator["evidence"].items():
            bind(path, digest)
        approved_configs = {row["model"]: row for row in evaluator["comparisons"]}
        tool = Path(__file__).resolve()
        if formal:
            require(Path(permit["acceptance_tool"]["file"]).resolve() == tool
                    and file_digest(tool) == permit["acceptance_tool"]["sha256"],
                    "This is not the prospectively bound acceptance tool")
            from lgp.runners.vcsf_budget_admission import verify_formal_permit
            # Rebind the returned evidence below without imposing next-group disk admission.
            ready = verify_formal_permit(registry, permit, recheck_evidence=False)
            if isinstance(ready, dict):
                for prerequisite in ready.values():
                    if isinstance(prerequisite, dict):
                        for path, digest in prerequisite.get("evidence", {}).items():
                            bind(path, digest)
            for key in ("target_receipt", "evaluator_receipt", "reference_reuse", "diagnostic_receipt"):
                read(permit[key]["file"], permit[key]["sha256"])
        bind(tool)
        bind_source_snapshot(evidence, registry.root, permit["source_identity"])
        prepared = read(permit["prepared"]["file"], permit["prepared"]["sha256"])
        for name, digest in prepared["files"].items():
            bind(Path(permit["prepared"]["file"]).parent / name, digest)
        source_inputs = read(Path(permit["prepared"]["file"]).parent / (binding["source"] + ".json"))
        require_isolated(output, source_inputs["inputs"]["payload"])
        assets = read(permit["assets"]["file"], permit["assets"]["sha256"])
        read(permit["checkpoint_bindings"]["file"], permit["checkpoint_bindings"]["sha256"])
        for path, digest in assets["evidence"].items():
            bind(path, digest)
        clean_annotation = read(assets["annotation"]["file"], assets["annotation"]["sha256"])
        clean = {row["image_id"]: row["input"] for row in assets["clean"]}
        require(len(clean) == len(assets["clean"]) and set(ids) <= set(clean), "Clean population differs")
        require(ids == sorted(set(ids)) and binding["max_images"] == permit["max_images"],
                "Invalid group image scope")
        if formal:
            spec = registry.protocols[permit["protocol"]]
            require(len(ids) == spec["images"] and ids_digest(ids) == permit["plan"]["full_image_ids_sha256"],
                    "Formal group is not the complete population")
        for row in permit["checkpoints"].values():
            bind(row["path"], row["sha256"])
        permit_ref = dict(file=str(permit_path), sha256=permit_sha256)
        require(read(root / "permit.json") == permit_ref, "Startup permit differs")
        coordinator = read(root / "coordinator.json")
        device = binding["device"]
        slot = permit["devices"].index(device)
        worker = root / "workers" / str(slot)
        request = read(worker / "request.json")
        process = read(worker / "process.json")
        gpu = read(worker / "gpu_context_registration.json")
        require(request["group_indices"] == [row["group_index"] for row in bindings if row["device"] == device]
                and request["device"] == device and request["coordinator"] == coordinator
                and request["permit"] == permit_ref
                and request["source_root"] == str(registry.root.resolve())
                and request["gpu_uuid"] == permit["gpu_uuids"][device] == gpu["gpu_uuid"],
                "Saved live worker assignment differs")
        owner = dict(worker_pid=process["pid"], worker_start_ticks=process["start_ticks"],
            process_group=process["pid"], coordinator_pid=coordinator["pid"],
            coordinator_start_ticks=coordinator["start_ticks"])
        verify_owner(coordinator, process, owner, gpu)
        group = root / "groups" / "{:06d}".format(group_index)
        payload = group / "attack"
        run = read(payload / "run.json")
        metadata = dict(protocol=binding["protocol"], group_index=group_index,
            epsilon=binding["epsilon"], permit_sha256=permit_sha256,
            assets_sha256=permit["assets"]["sha256"], diagnostic_only=not formal)
        expected = dict(status="complete", dataset="coco", split="val", source=binding["source"],
            attack="vcsf_final_budget_isolated", seed=binding["seed"],
            parameters_sha256=binding["parameters_sha256"], budget_profile=binding["budget_profile"],
            run_metadata=metadata, successful_images=len(ids), failed_images=0,
            full_payload_available=True, image_root=".", seed_schedule="explicit_offsets")
        require(all(run.get(k) == v for k, v in expected.items()), "Generated run binding differs")
        research = run["research_execution"]
        require(all(research.get(k) == v for k, v in dict(protocol=binding["protocol"],
            group_index=group_index, epsilon=binding["epsilon"], stress_only=binding["stress_only"],
            parameters_sha256=binding["parameters_sha256"], formal_AP_eligible=False,
            global_registry_mutated=False, global_factory_mutated=False).items())
            and run["formal_AP_eligible"] is False, "Generation execution provenance differs")
        require(canonical_hash(run["parameters"]) == canonical_hash(binding["parameters"])
                == binding["parameters_sha256"]
                and run["parameters"]["eps"] == float(Fraction(binding["epsilon"])),
                "Radius parameters differ")
        mapping = binding["seed_mapping"]
        require([row["image_id"] for row in mapping] == ids
                and all(row["offset"] == n and row["attack_seed"] == binding["seed"] + n
                        for n, row in enumerate(mapping)), "Seed mapping differs")
        require(run["image_ids_sha256"] == run["requested_ordered_image_ids_sha256"] == ids_digest(ids)
                and run["seed_offsets_sha256"] == canonical_hash([[row["image_id"], row["offset"]] for row in mapping]),
                "Saved image/seed hashes differ")
        manifest_path = bind(payload / "manifest.jsonl", run["manifest_sha256"])
        rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        annotation = read(payload / "annotations.json", run["annotation_sha256"])
        require([row["image_id"] for row in rows] == ids
                and [row["id"] for row in annotation["images"]] == ids, "Manifest population differs")
        names = {row["id"]: row["file_name"] for row in annotation["images"]}
        require(annotation == _subset_annotation(clean_annotation, ids, names), "Ground truth changed")
        source_weight = assets["checkpoints"][binding["source"]]
        source_path = bind(source_weight["file"], source_weight["sha256"])
        require(Path(run["checkpoint"]).resolve() == source_path
                and run["checkpoint_sha256"] == source_weight["sha256"]
                and source_path.stat().st_size == source_weight["bytes"], "Source checkpoint differs")
        limit = Fraction(binding["epsilon"]) * 255
        image_evidence = []
        for row, image, seed in zip(rows, annotation["images"], mapping):
            path = owned_image(payload, row["output_file"])
            require(row["output_file"] == image["file_name"] and row["status"] == "ok"
                    and row["attack_seed"] == seed["attack_seed"]
                    and type(row["linf_pixel"]) in (int, float) and math.isfinite(row["linf_pixel"])
                    and 0 <= row["linf_pixel"] <= limit
                    and path.stat().st_size == row["output_bytes"], "Manifest image/seed/radius differs")
            bind(path, row["output_sha256"])
            clean_ref = clean[row["image_id"]]
            clean_path = bind(clean_ref["file"], clean_ref["sha256"])
            require(Path(row["source_file"]).resolve() == clean_path
                    and clean_path.stat().st_size == clean_ref["bytes"], "Clean source bytes differ")
            with Image.open(path) as img:
                require(img.format == "PNG" and img.mode == "RGB"
                        and img.size == (image["width"], image["height"]), "Saved PNG dimensions/mode differ")
                pixels = np.asarray(img, dtype=np.int16)
            with Image.open(clean_path) as img:
                clean_pixels = np.asarray(img.convert("RGB"), dtype=np.int16)
            require(pixels.shape == clean_pixels.shape, "Clean/saved image dimensions differ")
            actual = int(np.max(np.abs(pixels - clean_pixels)))
            require(actual <= limit, "Actual saved pixels exceed the exact rational radius")
            image_evidence.append(dict(image_id=row["image_id"], file=str(path),
                sha256=row["output_sha256"], bytes=row["output_bytes"], linf_pixel=actual,
                clean=clean_ref, attack_seed=row["attack_seed"]))
        require(len({row["file"] for row in image_evidence}) == len(ids), "Payload paths alias")
        results = read(group / "results.json")
        require(len(results) == 16 and [row["target"] for row in results] == list(registry.paper_order),
                "Group does not have all sixteen canonical targets")
        terminal = read(group / "terminal.json")
        require(terminal["status"] == "complete_pending_independent_acceptance"
                and terminal["results"] == results and terminal["formal_result_eligible"] is False
                and terminal["scientific_acceptance"] is False, "Group terminal differs")
        actual_configs = []
        for result in results:
            target = result["target"]
            evaluation = group / "evaluations" / target
            require(result["group_index"] == group_index and result["source"] == binding["source"]
                    and result["epsilon"] == binding["epsilon"]
                    and result["evaluation"] == str(evaluation), "Result scope differs")
            record = read(evaluation / "metrics.json", result["metrics_sha256"])
            checkpoint = permit["checkpoints"][target]
            approved = approved_configs[target]
            require(approved["checkpoint_sha256"] == checkpoint["sha256"],
                    "Approved evaluator checkpoint differs")
            config_path = evaluation / "model" / "runtime_config.py"
            config_hash = verify_actual_config(config_path, approved["normalized_sha256"], bind)
            actual_configs.append(dict(target=target, file=str(config_path),
                sha256=evidence[str(config_path.resolve())], normalized_sha256=config_hash))
            verify_evaluation(record, dict(binding, dataset="coco", split="val", target=target),
                              ids, checkpoint["path"], checkpoint["sha256"])
            require(record["adversarial_run"] == str(payload)
                    and Path(record["annotation"]).resolve() == (payload / "annotations.json").resolve()
                    and record["attack"] == expected["attack"], "Evaluation used another payload")
            archive = read(evaluation / "predictions_artifact.json")
            archive_path = bind(evaluation / "predictions.json.gz", archive["archive_sha256"])
            with gzip.open(str(archive_path), "rb") as stream:
                raw = stream.read()
            require(hashlib.sha256(raw).hexdigest() == archive["uncompressed_sha256"] == record["predictions_sha256"],
                    "Prediction content differs")
            predictions = json.loads(raw)
            verify_archive_metadata(archive, archive_path.stat().st_size, raw, predictions, record)
            replay = evaluate_bbox_subset(annotation, predictions, ids, ids)
            compare_metrics(replay["metrics"], record["metrics"])
            atomic_json(output / (target + ".replay.json"), replay)
        verify_evidence_unchanged(evidence)
        receipt = dict(status="independently_verified_radius_group", protocol=permit["protocol"],
            group_index=group_index, source=binding["source"], epsilon=binding["epsilon"],
            image_count=len(ids), cells=16, metric_values=192, model_calls=0,
            formal_result_eligible=formal, scientific_acceptance=False, whole_program_acceptance=False,
            diagnostic_only=not formal, full_scope=formal, permit=permit_ref, permit_sha256=permit_sha256,
            generation_binding=binding, run=str(root), payload=str(payload), results=results,
            evaluator_receipt=evaluator_ref, actual_evaluator_configs=actual_configs,
            image_evidence=image_evidence, evidence=evidence, tool_sha256=file_digest(tool),
            interpretation="One saved radius group only; not whole-program scientific acceptance.")
        atomic_json(output / "receipt.json", receipt)
        atomic_json(output / "state.json", dict(status="complete", receipt_sha256=file_digest(output / "receipt.json")))
        return receipt
    except BaseException as exc:
        atomic_json(output / "state.json", dict(status="failed", error=repr(exc), evidence=evidence,
            group_index=group_index, permit_sha256=permit_sha256,
            formal_result_eligible=False, scientific_acceptance=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--permit", required=True, type=Path)
    parser.add_argument("--permit-sha256", required=True)
    parser.add_argument("--group-index", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluator-receipt", type=Path)
    parser.add_argument("--evaluator-sha256")
    args = parser.parse_args()
    if (args.evaluator_receipt is None) != (args.evaluator_sha256 is None):
        parser.error("--evaluator-receipt and --evaluator-sha256 must be supplied together")
    sys.path.insert(0, str(args.source.resolve() / "src"))
    from lgp.registry import Registry
    receipt = audit(Registry(args.source.resolve()), args.permit, args.permit_sha256,
                    args.group_index, args.output, evaluator_ref=None if args.evaluator_receipt is None else
                    dict(file=str(args.evaluator_receipt.resolve()), sha256=args.evaluator_sha256))
    print(json.dumps({key: receipt[key] for key in ("status", "group_index", "source", "epsilon",
        "image_count", "cells", "metric_values", "formal_result_eligible", "scientific_acceptance")}))


if __name__ == "__main__":
    main()
