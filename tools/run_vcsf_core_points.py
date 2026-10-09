"""Direct data-only core arithmetic and separate independent arithmetic checking."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import audit_vcsf_cpu_analysis as evidence
from lgp.io import atomic_json, file_digest
from lgp.metrics import COCO_BBOX_METRICS
from lgp.registry import Registry
from lgp.reporting.vcsf_core_points import assemble_core_point_matrix, write_core_point_reports


ROOT = Path(__file__).resolve().parents[1]
ENTRIES = ("experiments/vcsf_core_points.py", "experiments/audit_vcsf_core_points.py")
BOOTSTRAP = "experiments/vcsf_single_source_ablation_executor_smoke.py"


def source_binding(entry, entry_sha256, bootstrap_sha256):
    evidence.check(file_digest(ROOT / entry) == entry_sha256, "Executed entry source differs from current bytes")
    evidence.check(file_digest(ROOT / BOOTSTRAP) == bootstrap_sha256,
        "Executed source-only bootstrap differs from its original bytes")
    result = {entry: entry_sha256, BOOTSTRAP: bootstrap_sha256}
    for name, module in list(sys.modules.items()):
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            evidence.check(name != "lgp" and not name.startswith("lgp."), "Unbound project import")
            continue
        path = Path(raw)
        if ROOT not in path.parents:
            evidence.check(name != "lgp" and not name.startswith("lgp."), "Foreign project import")
            continue
        if module is sys.modules.get("__main__") and path == ROOT / entry:
            continue
        loader = getattr(module, "__loader__", None)
        digest = file_digest(evidence.plain(path))
        evidence.check(getattr(loader, "source_only", None) is True
            and getattr(loader, "source_sha256", None) == digest, "Executed project module is not source-bound: " + name)
        result[path.relative_to(ROOT).as_posix()] = digest
    registry_files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*")
        if p.is_file() and p.suffix in (".yaml", ".yml")}
    for name in sorted(registry_files | {"requirements/locked-cu118.txt", "environment.yml"}):
        result[name] = file_digest(ROOT / name)
    return dict(sorted(result.items()))


def bound_registry(read):
    names = {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*")
        if p.is_file() and p.suffix in (".yaml", ".yml")}
    evidence.check(names, "Missing registry input files")
    for name in sorted(names):
        read.bytes(ROOT / name, file_digest(ROOT / name))
    registry = Registry(ROOT)
    evidence.check(names == {p.relative_to(ROOT).as_posix() for p in (ROOT / "configs").rglob("*")
        if p.is_file() and p.suffix in (".yaml", ".yml")}, "Registry file membership changed while loading")
    read.unchanged()
    targets = registry.target_ids()
    evidence.check(len(targets) == 16
        and registry.protocols["vcsf_single_source_ablation_research"]["sources"] == targets[:1],
        "Registry no longer defines the canonical single-source core panel")
    return registry


def output_path(workspace, path, audit=False):
    path = evidence.plain(Path(path).absolute())
    namespace = workspace / "outputs" / ("audits" if audit else "analyses") / "vcsf_core_point_arithmetic"
    evidence.check(path.parent == namespace and not path.exists(),
        "Use a fresh direct output under outputs/{}/vcsf_core_point_arithmetic".format("audits" if audit else "analyses"))
    return path


def main(argv=None, *, audit=False, entry_sha256=None, bootstrap_sha256=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-plan", type=Path, help="Original core stage plan, never an executor smoke index.")
    parser.add_argument("--stage-plan-sha256")
    parser.add_argument("--normalized-cells", type=Path, help="JSON list of all 128 normalized cells; provenance remains pending.")
    parser.add_argument("--cells-sha256")
    parser.add_argument("--max-images", type=int, help="Only 1 is supported and must match a separately prepared diagnostic stage.")
    parser.add_argument("--plan-only", action="store_true", help="Inspect scope without writing or computing contrasts.")
    parser.add_argument("--output", type=Path)
    if audit:
        parser.add_argument("--report", type=Path, help="Original arithmetic report directory.")
        parser.add_argument("--report-manifest-sha256")
    args = parser.parse_args(argv)
    output = None
    owns_output = False
    try:
        evidence.check(sys.platform == "linux" and Path(sys.prefix).name == "oda",
            "Use the authorized server ODA environment")
        evidence.check(args.max_images is None or type(args.max_images) is int and args.max_images == 1,
            "Use full scope or an explicitly prepared --max-images 1 diagnostic")
        read = evidence.Evidence()
        if args.plan_only:
            evidence.check(args.output is None and not any((args.stage_plan, args.stage_plan_sha256,
                args.normalized_cells, args.cells_sha256, getattr(args, "report", None),
                getattr(args, "report_manifest_sha256", None))), "Plan-only scope inspection takes no evidence or output paths")
            registry = bound_registry(read)
            targets = registry.target_ids()
            source = source_binding(ENTRIES[int(audit)], entry_sha256, bootstrap_sha256)
            for name, digest in source.items():
                read.bytes(ROOT / name, digest)
            read.unchanged()
            print(json.dumps(dict(status="core_point_arithmetic_scope_only", groups=8,
                target_cells=128, contrasts=32, targets=targets, metrics=list(COCO_BBOX_METRICS),
                max_images=args.max_images, input_provenance_acceptance="pending", formal_metrics_eligible=False,
                independent_result_acceptance=False, model_calls=0, AP_evaluations=0), sort_keys=True))
            return 0
        evidence.check(all((args.stage_plan, args.stage_plan_sha256, args.normalized_cells, args.cells_sha256, args.output)),
            "Supply the original stage/cells, both file hashes and a fresh output")
        if audit:
            evidence.check(args.report is not None and evidence.is_sha(args.report_manifest_sha256),
                "Independent arithmetic checking requires the original report and manifest hash")
        output = output_path(ROOT, args.output, audit)
        stage_path = evidence.plain(args.stage_plan.absolute())
        cells_path = evidence.plain(args.normalized_cells.absolute())
        evidence.check(stage_path != cells_path and all(output != p and output not in p.parents
            and p not in output.parents for p in (stage_path, cells_path)),
            "Original stage and cells must be distinct files outside the new output")
        if audit:
            report = evidence.plain(args.report.absolute())
            evidence.check(report != output and output not in report.parents and report not in output.parents,
                "Independent output must be separate from the original report")
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        stage = read.json(stage_path, args.stage_plan_sha256)
        cells = read.json(cells_path, args.cells_sha256)
        registry = bound_registry(read)
        targets = registry.target_ids()
        evidence.check(len(targets) == 16 and registry.protocols["vcsf_single_source_ablation_research"]["sources"] == targets[:1],
            "Registry no longer defines the canonical single-source core panel")
        entry = ENTRIES[int(audit)]
        if audit:
            from audit_vcsf_core_points import verify_core_point_matrix, verify_core_point_exports
        source = source_binding(entry, entry_sha256, bootstrap_sha256)
        for name, digest in source.items():
            read.bytes(ROOT / name, digest)
        environment = dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"))
        bindings = dict(stage_plan=dict(file=str(stage_path), sha256=args.stage_plan_sha256),
            normalized_cells=dict(file=str(cells_path), sha256=args.cells_sha256))
        if audit:
            manifest = read.json(report / "manifest.json", args.report_manifest_sha256)
            evidence.check(isinstance(manifest, dict) and len(manifest) == 9, "Incomplete arithmetic report manifest")
            for name, digest in manifest.items():
                read.bytes(evidence.child(report, name), digest)
            result = read.json(report / "analysis.json", manifest["analysis.json"])
            evidence.check(result.get("input_files") == bindings, "Report references different original stage or cell files")
            proof = verify_core_point_matrix(stage, cells, result, target_order=targets, max_images=args.max_images)
            exports = verify_core_point_exports(report, result)
            read.unchanged()
            atomic_json(output / "audit.json", dict(proof, exports=exports, input_files=bindings,
                report_manifest_sha256=args.report_manifest_sha256, source_sha256=source, environment=environment,
                checked_at=datetime.now(timezone.utc).isoformat()))
            summary = dict(status=proof["status"], groups=8, target_cells=128, contrasts=32,
                input_provenance_acceptance="pending", independent_result_acceptance=False,
                output=str(output / "audit.json"))
        else:
            result = assemble_core_point_matrix(stage, cells, target_order=targets, max_images=args.max_images)
            result.update(input_files=bindings, source_sha256=source, environment=environment)
            read.unchanged()
            manifest = write_core_point_reports(output, result, claimed_empty_directory=True)
            read.unchanged()
            summary = dict(status=result["status"], groups=8, target_cells=128, contrasts=32,
                input_provenance_acceptance="pending", independent_result_acceptance=False,
                manifest_sha256=file_digest(output / "manifest.json"), output=str(output))
        print(json.dumps(summary, sort_keys=True))
        return 0
    except Exception as error:
        if owns_output and output is not None and output.is_dir():
            failure = output / "failure.json"
            if not failure.exists():
                atomic_json(failure, dict(status="failed_preserved", error_type=type(error).__name__, error=str(error),
                    input_provenance_acceptance="pending", independent_result_acceptance=False))
        print(str(error), file=sys.stderr)
        return 1
