"""Build new-user qualitative displays from one complete current-main run.

This module performs no detector loading, inference, AP replay or acceptance.
The loader validates complete archives and selected pixels; this builder checks
the current-main layout, declared scope, budgets and saved method provenance.
"""
import ast
import hashlib
import json
from itertools import islice
from pathlib import Path
import tempfile
from types import SimpleNamespace

from ..attacks.vcsf_public import PARAMETERS_SHA256
from ..data.coco import CocoIndex
from ..registry import _load_yaml
from .qualitative_inputs import (
    MAX_INPUT_BYTES, _canonical, _hash, _json, _path, _sha,
    load_panel_request, verify_panel_inputs,
)


PROTOCOL = "current_saved_main_qualitative"
MAX_SUFFIXES = 64


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bound(path):
    path = _path(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(64 << 10), b""):
            size += len(block)
            _require(size <= MAX_INPUT_BYTES, "Builder input exceeds resource cap")
            digest.update(block)
    return dict(file=str(path), sha256=digest.hexdigest(), bytes=size)


def _document(path):
    path = _path(path)
    _require(path.stat().st_size <= MAX_INPUT_BYTES, "Builder JSON exceeds resource cap")
    with path.open("rb") as handle:
        raw = handle.read(MAX_INPUT_BYTES + 1)
    _require(len(raw) <= MAX_INPUT_BYTES, "Builder JSON exceeds resource cap")
    value = _json(raw)
    _require(type(value) is dict, "Builder requires JSON objects")
    return value, dict(file=str(path), sha256=_hash(raw), bytes=len(raw))


def _ref(binding):
    return {key: binding[key] for key in ("file", "sha256")}


def display_contract(registry, max_images=None):
    """Return the registered display contract without reading main payloads."""
    protocol = registry.protocols[PROTOCOL]
    _require(protocol.get("method_source_protocol") == "main_transfer"
             and protocol.get("budget_source_protocol") == "main_transfer",
             "Qualitative protocol must reference current main methods and budgets")
    main = registry.protocols["main_transfer"]
    methods = main.get("methods")
    _require(type(methods) is list and len(methods) == 10
             and all(type(method) is str and method in registry.attacks for method in methods)
             and len(set(methods)) == len(methods)
             and main.get("report_method_order") == "declared",
             "Current main must declare ten unique methods in manuscript order")
    _require(protocol.get("datasets") == ["coco"] and protocol.get("split") == "val"
             and protocol.get("sources") == ["faster_rcnn_r50"]
             and protocol.get("targets") == ["cascade_rcnn_r50"]
             and protocol.get("image_ids") == [785, 1503, 2157]
             and protocol.get("score_threshold") == 0.5
             and protocol.get("max_detections") == 100
             and type(protocol.get("full_images")) is int and protocol["full_images"] == 5000
             and protocol.get("originals") is False,
             "Registered fixed COCO display contract changed")
    _require("coco" in main["datasets"] and main["split"] == protocol["split"]
             and protocol["sources"][0] in main["sources"],
             "Display scope is outside current main")
    _require(max_images is None or type(max_images) is int and 1 <= max_images <= 3,
             "max_images must be in 1..3 and limits display only")
    profiles = main.get("method_budget_profiles", {})
    budgets = {method: profiles.get(method, main["budget_profile"]) for method in methods}
    _require(all(budget in registry.budget_profiles for budget in budgets.values()),
             "Unregistered current-main method budget")
    ids = list(protocol["image_ids"])
    return dict(
        protocol=PROTOCOL, source_protocol="main_transfer", dataset="coco", split="val",
        source=protocol["sources"][0], target=protocol["targets"][0],
        requested_image_ids=ids, image_ids=ids[:max_images] if max_images else ids,
        method_order=["clean"] + list(methods),
        method_budget_profiles=dict(clean=None, **budgets), budget_profile=None,
        score_threshold=protocol["score_threshold"], max_detections=protocol["max_detections"],
        originals=False, evaluation_images=protocol["full_images"],
        display_smoke=max_images is not None, max_images=max_images,
        spatial_transform="none", layout="uncropped_aspect_preserving_contain_same_scale_per_image",
        selection_note=protocol["selection_note"],
        model_calls=0, GPU_calls=0, AP_replays=0,
        scientific_acceptance=False, independent_acceptance=False,
        formal_metrics_eligible=False, author_figure_certified=False,
    )


def _metrics_path(root, contract, method):
    base = root / "evaluations" / contract["dataset"]
    target = contract["target"]
    if method == "clean":
        branch = _path(base / "clean" / target, root, directory=True)
        entries = list(islice(branch.iterdir(), MAX_SUFFIXES + 1))
        _require(len(entries) <= MAX_SUFFIXES, "Too many clean evaluation entries")
        _require(not any(path.is_symlink() for path in entries), "Symlink clean entry is forbidden")
        extra = [path / "metrics.json" for path in entries if path.is_dir()
                 and (path / "metrics.json").exists()]
        _require(not extra, "Duplicate clean metrics are forbidden")
        return _path(branch / "metrics.json", root)
    branch = _path(base / contract["source"] / method, root, directory=True)
    entries = list(islice(branch.iterdir(), MAX_SUFFIXES + 1))
    _require(len(entries) <= MAX_SUFFIXES, "Too many evaluation suffixes for bounded discovery")
    candidates = []
    for suffix in entries:
        _require(not suffix.is_symlink(), "Symlink evaluation suffix is forbidden")
        if suffix.is_dir():
            candidate = suffix / target / "metrics.json"
            if candidate.exists() or candidate.is_symlink():
                candidates.append(_path(candidate, root))
    _require(len(candidates) == 1, "Missing or duplicate metrics for " + method)
    expected = branch / "default" / target / "metrics.json"
    _require(candidates[0] == expected, "Current main requires the default evaluation suffix")
    return candidates[0]


def _complete_metrics(metrics, contract, method, ids_hash):
    expected = dict(dataset=contract["dataset"], split=contract["split"],
                    source="clean" if method == "clean" else contract["source"],
                    target=contract["target"], attack=method)
    _require(all(metrics.get(key) == value for key, value in expected.items()),
             "Metrics dataset/split/source/target/attack mismatch: " + method)
    _require(metrics.get("status") == "complete" and metrics.get("failures") == []
             and type(metrics.get("images")) is int
             and metrics["images"] == contract["evaluation_images"]
             and metrics.get("evaluated_image_ids_sha256") == ids_hash
             and metrics.get("expected_image_ids_sha256") == ids_hash
             and metrics.get("evaluated_image_ids_match_expected") is True,
             "Missing, failed or incomplete 5000-image metrics: " + method)
    _require(_sha(metrics.get("checkpoint_sha256")), "Missing target checkpoint identity")
    _require(metrics.get("budget_profile") == contract["method_budget_profiles"][method],
             "Metrics budget mismatch: " + method)


def _saved_numerical_sources(main_run):
    """Validate a saved manifest without comparing unrelated historical files."""
    provenance, provenance_ref = _document(main_run / "provenance.json")
    manifest = provenance.get("source_manifest")
    _require(type(provenance.get("schema_version")) is int and provenance["schema_version"] == 1
             and provenance.get("protocol") == "main_transfer" and type(manifest) is dict,
             "Missing current-main source provenance")
    files = manifest.get("files")
    _require(manifest.get("algorithm") == "sha256" and type(files) is list
             and 0 < len(files) <= 4096 and type(manifest.get("file_count")) is int
             and manifest["file_count"] == len(files)
             and manifest.get("tree_sha256") == _hash(_canonical(files)),
             "Invalid current-main source manifest count/hash")
    recorded = {}
    for item in files:
        _require(type(item) is dict and set(item) == {"path", "sha256"}
                 and type(item["path"]) is str and _sha(item["sha256"]),
                 "Invalid source manifest row")
        relative = Path(item["path"])
        _require(relative.parts and not relative.is_absolute() and ".." not in relative.parts
                 and "\\" not in item["path"] and relative.as_posix() == item["path"]
                 and item["path"] not in recorded,
                 "Unsafe or duplicate source manifest path")
        recorded[item["path"]] = item["sha256"]
    return recorded, provenance_ref


def _lgp_source_proof(registry, main_run):
    """Match only canonical LGP numerical sources, not historical report code."""
    recorded, provenance_ref = _saved_numerical_sources(main_run)
    implementation = registry.attack("lgp").metadata.get("implementation")
    _require(implementation == "src/lgp/attacks/lgp.py",
             "LGP must use the registered canonical implementation")
    freeze_path = registry.attack("vcsf").metadata.get("source_freeze")
    _require(type(freeze_path) is str and not Path(freeze_path).is_absolute(),
             "Missing registered source-freeze reference")
    freeze, freeze_ref = _document(_path(freeze_path, registry.root))
    frozen = freeze.get("source_file_sha256")
    _require(type(frozen) is dict and freeze.get("source_manifest_sha256") == _hash(_canonical(frozen)),
             "Invalid registered source-freeze map")
    source = _bound(registry.root / implementation)
    _require(source["bytes"] <= (1 << 20), "Canonical LGP source exceeds parsing cap")
    with Path(source["file"]).open("rb") as handle:
        raw = handle.read((1 << 20) + 1)
    _require(_hash(raw) == source["sha256"], "Canonical LGP source changed while reading")
    tree = ast.parse(raw, filename=implementation)
    directory = Path(implementation).parent
    prefix = str(directory / "lgp_")
    helpers = {path for path in frozen if path.startswith(prefix) and path.endswith(".py")}
    current_helpers = list(islice((registry.root / directory).glob("lgp_*.py"), 33))
    _require(0 < len(current_helpers) <= 32
             and helpers == {path.relative_to(registry.root).as_posix() for path in current_helpers},
             "Canonical LGP helper family differs from its registered freeze")
    # common supplies the imported optimizer/projection/config numerical helpers.
    numerical_common = {str(directory / (node.module + ".py")) for node in ast.walk(tree)
                        if isinstance(node, ast.ImportFrom) and node.level == 1
                        and node.module == "common"}
    required = helpers | numerical_common | {implementation}
    bindings, expected = [], {}
    for relative in sorted(required):
        binding = _bound(_path(relative, registry.root))
        _require(frozen.get(relative) == binding["sha256"],
                 "Current canonical LGP source differs from registered freeze: " + relative)
        _require(recorded.get(relative) == binding["sha256"],
                 "Saved main canonical LGP numerical source mismatch or missing: " + relative)
        bindings.append(binding)
        expected[relative] = binding["sha256"]
    registry_refs = [_bound(registry.root / "configs/attacks/reference.yaml"),
                     _bound(registry.root / "configs/attacks/vcsf.yaml")]
    proof = dict(method="lgp", implementation=implementation, provenance=_ref(provenance_ref),
                 source_freeze=_ref(freeze_ref), registry=[_ref(ref) for ref in registry_refs],
                 source_sha256=expected, source_map_sha256=_hash(_canonical(expected)),
                 scope="canonical_lgp_executor_helper_family_and_direct_numerical_common",
                 historical_reporting_sources_compared=False, scientific_acceptance=False,
                 independent_acceptance=False)
    return proof, [provenance_ref, freeze_ref] + registry_refs + bindings


def _vcsf_source_proof(registry, main_run):
    """Bind the saved A10 numerical subset to its registered selection identity."""
    recorded, provenance_ref = _saved_numerical_sources(main_run)
    metadata = registry.attack("vcsf").metadata
    selection_path = metadata.get("selection_protocol")
    _require(type(selection_path) is str and not Path(selection_path).is_absolute(),
             "Missing registered VCSF selection reference")
    selection, selection_ref = _document(_path(selection_path, registry.root))
    decision, execution = selection.get("decision"), selection.get("execution_identity")
    _require(type(decision) is dict and type(execution) is dict
             and decision.get("unique_working_candidate") == "A10"
             and decision.get("parameters_sha256") == PARAMETERS_SHA256
             and type(decision.get("parameters")) is dict
             and _hash(_canonical(decision["parameters"])) == PARAMETERS_SHA256,
             "VCSF selection parameter identity mismatch")
    numeric = execution.get("numeric_source_sha256")
    _require(type(numeric) is dict and 0 < len(numeric) <= 64
             and execution.get("numeric_source_map_sha256") == _hash(_canonical(numeric)),
             "Invalid registered VCSF numerical source map")
    bindings = []
    for relative, expected in sorted(numeric.items()):
        _require(type(relative) is str and _sha(expected),
                 "Invalid VCSF numerical source row")
        binding = _bound(_path(relative, registry.root))
        _require(binding["sha256"] == expected,
                 "Current VCSF numerical source differs from registered selection: " + relative)
        _require(recorded.get(relative) == expected,
                 "Saved main final VCSF numerical source mismatch or missing: " + relative)
        bindings.append(binding)
    implementation = metadata.get("implementation")
    _require(type(implementation) is str and implementation in numeric,
             "VCSF implementation is absent from its registered numerical map")
    declaration = _bound(registry.root / "configs/attacks/vcsf.yaml")
    proof = dict(method="vcsf", implementation=implementation,
                 parameters_sha256=PARAMETERS_SHA256, provenance=_ref(provenance_ref),
                 selection_protocol=_ref(selection_ref), registry=_ref(declaration),
                 source_sha256=numeric, source_map_sha256=execution["numeric_source_map_sha256"],
                 scope="registered_final_A10_numerical_subset",
                 historical_reporting_sources_compared=False, scientific_acceptance=False,
                 independent_acceptance=False)
    return proof, [provenance_ref, selection_ref, declaration] + bindings


def build_current_request(registry, main_run):
    """Bind a schema-2 request to originals within one unmerged main run."""
    contract = display_contract(registry)
    root = _path(Path(main_run).absolute(), directory=True)
    plan, plan_ref = _document(root / "plan.json")
    _require(plan.get("protocol") == "main_transfer"
             and plan.get("split") == contract["split"]
             and contract["dataset"] in plan.get("datasets", [])
             and contract["source"] in plan.get("sources", [])
             and contract["target"] in plan.get("targets", [])
             and plan.get("methods") == contract["method_order"][1:]
             and plan.get("budget_profile") == registry.protocols["main_transfer"]["budget_profile"]
             and plan.get("method_budget_profiles", {}) == registry.protocols["main_transfer"].get("method_budget_profiles", {})
             and not plan.get("image_selection") and plan.get("execution_retired") is not True,
             "Input is not the registered current-main plan/scope/order/budget")
    # The runner stores the execution limit in summary.json, not plan.json.
    execution, execution_ref = _document(root / "summary.json")
    _require(execution.get("protocol") == "main_transfer"
             and execution.get("status") == "complete"
             and "max_images" in execution and execution["max_images"] is None
             and type(execution.get("failed_records")) is int and execution["failed_records"] == 0
             and type(execution.get("image_failures")) is int and execution["image_failures"] == 0,
             "Execution metadata must explicitly prove complete, unlimited current main")
    if "max_images" in plan:
        _require(plan["max_images"] is None, "Limited main plan is forbidden")
    lgp_source_proof, lgp_source_inputs = _lgp_source_proof(registry, root)
    vcsf_source_proof, vcsf_source_inputs = _vcsf_source_proof(registry, root)
    dataset = registry.dataset(contract["dataset"])
    split = dataset.split(contract["split"])
    _require(split.expected_images == contract["evaluation_images"], "Registry full population changed")
    annotation_path = _path(dataset.root / split.annotation)
    annotation, annotation_ref = _document(annotation_path)
    index = CocoIndex(dataset, contract["split"])
    _require(_hash(_canonical(index.payload)) == _hash(_canonical(annotation)),
             "Canonical annotation changed during index construction")
    full_ids = [image["id"] for image in index.images]
    _require(len(full_ids) == contract["evaluation_images"]
             and all(type(iid) is int for iid in full_ids)
             and full_ids == sorted(set(full_ids))
             and set(contract["requested_image_ids"]) <= set(full_ids),
             "Canonical COCO val must contain 5000 unique image IDs")
    ids_hash = _hash(_canonical(full_ids))
    inputs = [plan_ref, execution_ref, annotation_ref] + lgp_source_inputs + vcsf_source_inputs
    clean_refs, by_id = [], {image["id"]: image for image in index.images}
    image_root = _path(dataset.root / split.image_prefix, directory=True)
    for iid in contract["requested_image_ids"]:
        filename = Path(by_id[iid]["file_name"])
        _require(not filename.is_absolute() and ".." not in filename.parts
                 and filename.parts, "Unsafe canonical clean filename")
        path = _path(image_root / filename, image_root)
        _require(index.image_path(by_id[iid]) == path, "CocoIndex image-root mismatch")
        binding = _bound(path)
        clean_refs.append(dict(binding, image_id=iid))
        inputs.append(binding)
    rows, identities = [], []
    target_checkpoint = source_checkpoint = None
    jobs = plan.get("jobs")
    _require(type(jobs) is list and all(type(job) is dict for job in jobs), "Missing current-main jobs")
    for method in contract["method_order"]:
        metrics, metrics_ref = _document(_metrics_path(root, contract, method))
        _complete_metrics(metrics, contract, method, ids_hash)
        if target_checkpoint is None:
            target_checkpoint = metrics["checkpoint_sha256"]
        _require(target_checkpoint == metrics["checkpoint_sha256"], "Target checkpoint mismatch")
        row = dict(label="Clean" if method == "clean" else registry.attack(method).display_name,
                   attack=method, metrics=_ref(metrics_ref))
        inputs.append(metrics_ref)
        identity = dict(attack=method, budget_profile=contract["method_budget_profiles"][method],
                        metrics=_ref(metrics_ref), target_checkpoint_sha256=target_checkpoint)
        if method == "clean":
            _require(_path(metrics.get("annotation")) == annotation_path,
                     "Clean metrics do not reference canonical annotation")
            _require(metrics.get("adversarial_run") is None and metrics.get("parameters_sha256") is None,
                     "Clean metrics contain attack provenance")
            row.update(annotation=_ref(annotation_ref), clean_images=clean_refs)
        else:
            budget = contract["method_budget_profiles"][method]
            matching = [job for job in jobs if all(job.get(key) == value for key, value in
                        dict(dataset=contract["dataset"], split=contract["split"],
                             source=contract["source"], target=contract["target"], attack=method).items())]
            _require(len(matching) == 1 and matching[0].get("status") == "ready"
                     and matching[0].get("budget_profile") == budget
                     and matching[0].get("study") == "" and matching[0].get("variant") == "default",
                     "Missing, duplicate or incompatible current-main job: " + method)
            generation_root = _path(root / "attacks" / contract["dataset"] / contract["source"]
                                    / method / "default", root, directory=True)
            _require(_path(metrics.get("adversarial_run"), directory=True) == generation_root,
                     "Metrics cannot borrow generation from another/merged run")
            run, run_ref = _document(generation_root / "run.json")
            _require(all(run.get(key) == value for key, value in
                         dict(dataset=contract["dataset"], split=contract["split"],
                              source=contract["source"], attack=method, budget_profile=budget,
                              status="complete").items())
                     and all(type(run.get(key)) is int and run[key] == value for key, value in
                             dict(requested_images=5000, successful_images=5000, failed_images=0).items())
                     and run.get("full_payload_available") is not False,
                     "Incomplete or incompatible current-main generation: " + method)
            parameters = run.get("parameters")
            _require(type(parameters) is dict and _sha(run.get("parameters_sha256"))
                     and _hash(_canonical(parameters)) == run["parameters_sha256"]
                     and metrics.get("parameters_sha256") == run["parameters_sha256"],
                     "Generation/metrics parameter identity mismatch: " + method)
            if method == "vcsf":
                _require(run["parameters_sha256"] == PARAMETERS_SHA256,
                         "Old or unregistered VCSF parameter identity")
                _require(run.get("implementation") == vcsf_source_proof["implementation"],
                         "VCSF generation must declare the registered final implementation")
                identity["numerical_source_proof"] = vcsf_source_proof
            if method == "lgp":
                _require(run.get("implementation") == lgp_source_proof["implementation"],
                         "LGP generation must declare the canonical implementation")
                identity["numerical_source_proof"] = lgp_source_proof
            _require(_sha(run.get("checkpoint_sha256")), "Missing source checkpoint identity")
            if source_checkpoint is None:
                source_checkpoint = run["checkpoint_sha256"]
            _require(source_checkpoint == run["checkpoint_sha256"], "Source checkpoint mismatch")
            row.update(generation=_ref(run_ref), budget_profile=budget)
            inputs.append(run_ref)
            identity.update(generation=_ref(run_ref), parameters_sha256=run["parameters_sha256"],
                            source_checkpoint_sha256=source_checkpoint)
        rows.append(row)
        identities.append(identity)
    for relative in ("configs/experiments/protocols.yaml", "configs/experiments/budgets.yaml",
                     "configs/models.yaml", "configs/datasets/coco.yaml"):
        inputs.append(_bound(registry.root / relative))
    request = dict(schema_version=2, dataset=contract["dataset"], split=contract["split"],
                   source=contract["source"], target=contract["target"],
                   image_ids=contract["requested_image_ids"], score_threshold=contract["score_threshold"],
                   max_detections=contract["max_detections"], originals=contract["originals"],
                   selection_note=contract["selection_note"], rows=rows)
    return dict(request=request, source_inputs=inputs,
                identity=dict(protocol=PROTOCOL, main_run=str(root), rows=identities,
                              canonical_lgp_source_proof=lgp_source_proof,
                              final_vcsf_source_proof=vcsf_source_proof,
                              evaluation_images=5000, evaluation_image_ids_sha256=ids_hash,
                              execution_metadata=_ref(execution_ref), scientific_acceptance=False,
                              independent_acceptance=False, author_figure_certified=False))


def _verify_source_inputs(bundle):
    for binding in bundle["source_inputs"]:
        _require(_bound(binding["file"]) == binding, "Current-main builder input changed")


def _load_bound_request(bundle, path, digest, max_images):
    plan = load_panel_request(path, digest, max_images=max_images)
    _verify_source_inputs(bundle)
    _require(plan["evaluation_images"] == 5000 and plan["budget_profile"] is None,
             "Loader did not preserve the schema-2 full population/mixed-budget contract")
    _require([(row["attack"], row.get("budget_profile")) for row in plan["rows"]]
             == [(row["attack"], row.get("budget_profile")) for row in bundle["request"]["rows"]],
             "Loader row budget/order contract mismatch")
    plan["current_main_builder"] = bundle["identity"]
    existing = {ref["file"]: ref for ref in plan["evidence_inputs"]}
    for ref in bundle["source_inputs"]:
        _require(ref["file"] not in existing or existing[ref["file"]] == ref,
                 "Conflicting builder/loader input binding")
        existing[ref["file"]] = ref
    plan["evidence_inputs"] = list(existing.values())
    plan["evidence_sha256"] = _hash(_canonical({k: v for k, v in plan.items() if k != "evidence_sha256"}))
    verify_panel_inputs(plan)
    return plan


def _new_output(output, main_run):
    output = Path(output).absolute()
    _require(".." not in output.parts, "Output traversal is forbidden")
    _require(not any(path.is_symlink() for path in (output, *output.parents)),
             "Symlink output is forbidden")
    _require(not output.exists(), "Output must be a new directory; existing directories are forbidden")
    root = _path(Path(main_run).absolute(), directory=True)
    _require(output != root and root not in output.parents and output not in root.parents,
             "Output must be outside the original main run")
    _require(output.parent.is_dir(), "Output parent must already exist")
    _require(output.resolve() == output, "Output path must not alias another location")
    return output


def plan_current_display(root, max_images=None):
    """Read declarative display metadata only, without granting execution admission."""
    root = Path(root).absolute()
    protocols = _load_yaml(root / "configs/experiments/protocols.yaml")["protocols"]
    budgets = _load_yaml(root / "configs/experiments/budgets.yaml")["profiles"]
    attacks = _load_yaml(root / "configs/attacks/reference.yaml")["attacks"]
    public = _load_yaml(root / "configs/attacks/vcsf.yaml")
    attacks[public["id"]] = public
    snapshot = SimpleNamespace(protocols=protocols, budget_profiles=budgets, attacks=attacks)
    return run_current_qualitative(snapshot, plan_only=True, max_images=max_images)


def run_current_qualitative(registry, main_run=None, *, plan_only=False, output=None, max_images=None):
    """Validate privately for plan-only, or bind request then render PNG only."""
    contract = display_contract(registry, max_images)
    if main_run is None:
        _require(plan_only, "Execution requires --main-run; no historical/author figures are substituted")
        return dict(contract, status="display_contract_only_no_inputs_validated", output_written=False)
    if output is not None:
        _new_output(output, main_run)
    bundle = build_current_request(registry, main_run)
    raw = json.dumps(bundle["request"], indent=2, allow_nan=False).encode("utf-8") + b"\n"
    digest = _hash(raw)
    if plan_only:
        # This request is private, ephemeral validation evidence, never a render plan.
        with tempfile.TemporaryDirectory(prefix="lgp-current-qualitative-") as directory:
            request = Path(directory) / "request.json"
            request.write_bytes(raw)
            plan = _load_bound_request(bundle, request, digest, max_images)
            result = dict(contract, status="current_main_inputs_validated_no_output_written",
                          request_sha256=digest, evidence_sha256=plan["evidence_sha256"],
                          evidence_files=len(plan["evidence_inputs"]), output_written=False,
                          render_plan_retained=False, ephemeral_request=True,
                          source_inputs=bundle["source_inputs"], current_main=bundle["identity"])
        return result
    _require(output is not None, "Execution requires --output pointing to a new directory")
    output = _new_output(output, main_run)
    _verify_source_inputs(bundle)
    output.mkdir(exist_ok=False)
    request = output / "request.json"
    request.write_bytes(raw)
    (output / "request.sha256").write_text(digest + "  request.json\n", encoding="ascii")
    provenance = dict(bundle["identity"], source_inputs=bundle["source_inputs"],
                      request=dict(file=str(request), sha256=digest))
    (output / "source_inputs.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n",
                                             encoding="utf-8")
    try:
        plan = _load_bound_request(bundle, request, digest, max_images)
        from .qualitative_panel import render_panel

        rendered = render_panel(plan, output / "panel", pdf=False)
        manifest = dict(contract, status=rendered["status"], current_main=bundle["identity"],
                        request=_bound(request), source_inputs=_bound(output / "source_inputs.json"),
                        panel_manifest=_bound(output / "panel/manifest.json"),
                        evidence_sha256=plan["evidence_sha256"])
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n",
                                            encoding="utf-8")
        return dict(contract, status=rendered["status"], output=str(output),
                    request_sha256=digest, manifest=str(output / "manifest.json"),
                    panel_manifest=str(output / "panel/manifest.json"), output_written=True)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps(
            dict(status="failed_display_attempt_preserved", error=repr(error),
                 request_sha256=digest, scientific_acceptance=False,
                 independent_acceptance=False, author_figure_certified=False),
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        raise
