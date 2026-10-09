from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rich.console import Console
from rich.table import Table
from rich import box

from .attacks.common import parse_fraction
from .data.manager import (
    download_dataset,
    extract_dataset,
    minimize_dataset_storage,
    prepare_dataset,
    validate_dataset,
)
from .doctor import run_doctor
from .io import atomic_json
from .registry import Registry, RegistryError
from .reporting import (
    write_ablation_reports,
    write_analysis_reports,
    write_experiment_plots,
    write_paper_catalog,
    write_transfer_reports,
)
from .runtime_config import build_runtime_config
from .runners.artifacts import import_external_artifact
from .runners.attack import run_attack
from .runners.evaluate import run_evaluation
from .runners.experiment import run_experiment
from .runners.train import run_training
from .runners.weights import download_weights


console = Console()


def _csv(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _path(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser().resolve() if value else None


def _fraction_float(value: str) -> float:
    parsed = parse_fraction(value)
    if not isinstance(parsed, (int, float)):
        raise argparse.ArgumentTypeError(
            "expected a number or fraction such as 4/255"
        )
    return float(parsed)


def _add_common_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    parser.add_argument(
        "--download-weights",
        action="store_true",
        help="Download a missing official COCO checkpoint automatically",
    )
    parser.add_argument("--keep-going", action="store_true", help="Record failures and continue")


def _add_prediction_visualization(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--visualize-predictions",
        action="store_true",
        help="Save post-NMS boxes, class labels and confidence scores.",
    )
    parser.add_argument(
        "--visualization-score-threshold",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--visualization-max-images",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--visualization-max-detections",
        type=int,
        default=100,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lgp",
        description="Reproducible transfer attacks for COCO and Pascal VOC",
    )
    parser.add_argument("--debug", action="store_true", help="Show a full traceback on failure")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Verify environment, configs, data and weights")
    doctor.add_argument(
        "--datasets",
        help="Comma-separated registered dataset IDs (default: all registered datasets)",
    )
    doctor.add_argument("--deep-data", action="store_true", help="Verify every referenced image")
    doctor.add_argument("--json", dest="json_path", help="Also write the report to JSON")

    models = subparsers.add_parser("models", help="List the fixed 16-detector panel")
    models.add_argument("action", choices=["list"])

    compatibility = subparsers.add_parser(
        "compatibility", help="Show paper-verified attack/source compatibility"
    )
    compatibility.add_argument("--method", help="Limit to one attack")

    data = subparsers.add_parser("data", help="Download, extract, prepare or validate datasets")
    data_sub = data.add_subparsers(dest="data_action", required=True)
    for action in ("download", "extract", "prepare", "validate"):
        child = data_sub.add_parser(action)
        child.add_argument("--dataset", default="all", choices=["all", "coco", "voc", "bdd100k"])
        if action == "download":
            child.add_argument(
                "--full",
                action="store_true",
                help="Also download optional train/test payloads not used by the benchmark",
            )
        if action == "validate":
            child.add_argument("--deep", action="store_true")
            child.add_argument(
                "--full",
                action="store_true",
                help="Also require and validate optional dataset splits",
            )
            child.add_argument("--json", dest="json_path")
    minimize = data_sub.add_parser(
        "minimize",
        help="Remove extracted/downloaded payloads that no registered experiment reads",
    )
    minimize.add_argument(
        "--apply",
        action="store_true",
        help="Perform the deletion; without this flag only print the verified plan",
    )
    minimize.add_argument("--json", dest="json_path")

    weights = subparsers.add_parser("weights", help="Manage official detector checkpoints")
    weights_sub = weights.add_subparsers(dest="weights_action", required=True)
    weights_download = weights_sub.add_parser("download")
    weights_download.add_argument(
        "--models", default="all", help="Comma-separated model IDs or all"
    )

    render = subparsers.add_parser("config", help="Render a resolved upstream runtime config")
    render.add_argument("action", choices=["render"])
    render.add_argument("--model", required=True)
    render.add_argument("--dataset", required=True, choices=["coco", "voc", "bdd100k"])
    render.add_argument("--mode", default="test", choices=["test", "train"])
    render.add_argument("--split", default="val", choices=["train", "dev", "val"])
    render.add_argument("--output", required=True)

    train = subparsers.add_parser("train", help="Fine-tune one detector (VOC maintained; BDD100K legacy)")
    train.add_argument("--dataset", required=True, choices=["voc", "bdd100k"])
    train.add_argument("--model", required=True, help="Comma-separated model IDs or all")
    train.add_argument("--work-dir")
    train.add_argument("--overwrite-checkpoint", action="store_true")
    train.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        help="Resume one model from work-dir/last_checkpoint, or from an explicit checkpoint",
    )
    _add_common_runtime(train)

    attack = subparsers.add_parser("attack", help="Generate adversarial images and optionally evaluate")
    attack.add_argument("--dataset", required=True, choices=["coco", "voc", "bdd100k"])
    attack.add_argument("--source", required=True)
    attack.add_argument("--method", required=True)
    attack.add_argument("--split", default="val", choices=["train", "dev", "val"])
    attack.add_argument("--targets", default="all", help="Comma-separated target IDs or all")
    attack.add_argument("--evaluate", action="store_true", help="Evaluate selected targets after generation")
    attack.add_argument("--max-images", type=int)
    attack.add_argument("--seed", type=int, default=42)
    attack.add_argument("--output")
    attack.add_argument(
        "--strict",
        action="store_true",
        help="Fail rather than record any non-native compatibility status",
    )
    _add_common_runtime(attack)
    _add_prediction_visualization(attack)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate clean or adversarial images")
    evaluate.add_argument("--dataset", required=True, choices=["coco", "voc", "bdd100k"])
    evaluate.add_argument("--target", required=True)
    evaluate.add_argument("--split", default="val", choices=["train", "dev", "val"])
    evaluate.add_argument("--adversarial-run")
    evaluate.add_argument("--max-images", type=int)
    evaluate.add_argument("--output")
    evaluate.add_argument(
        "--prediction-archive",
        choices=["none", "gzip"],
        default="none",
        help="Losslessly archive prediction JSON after evaluation.",
    )
    _add_common_runtime(evaluate)
    _add_prediction_visualization(evaluate)

    artifact = subparsers.add_parser("artifact", help="Register verified external baseline images")
    artifact_sub = artifact.add_subparsers(dest="artifact_action", required=True)
    artifact_import = artifact_sub.add_parser("import")
    artifact_import.add_argument("--dataset", required=True, choices=["coco", "voc", "bdd100k"])
    artifact_import.add_argument("--source", required=True)
    artifact_import.add_argument("--method", required=True)
    artifact_import.add_argument("--annotation", required=True)
    artifact_import.add_argument("--image-root", required=True)
    artifact_import.add_argument(
        "--clean-root",
        help="Matching clean-image root used for pixel-level budget verification",
    )
    artifact_import.add_argument(
        "--expected-eps",
        type=_fraction_float,
        help="Normalized perturbation budget, for example 4/255",
    )
    artifact_import.add_argument(
        "--budget-profile",
        choices=["compute_matched", "paper_default"],
        help="Declared comparison regime from configs/experiments/budgets.yaml",
    )
    artifact_import.add_argument(
        "--gradient-evals-per-image",
        type=int,
        help=(
            "Source-detector backward-pass image equivalents; a batch of k "
            "transformed views counts as k"
        ),
    )
    artifact_import.add_argument("--attack-seconds-per-image", type=float)
    artifact_import.add_argument("--peak-memory-mb", type=float)
    artifact_import.add_argument("--parameters-json")
    artifact_import.add_argument("--code-commit")
    artifact_import.add_argument("--output")

    experiment = subparsers.add_parser("experiment", help="Plan or execute a complete protocol")
    experiment.add_argument("--protocol", required=True)
    experiment.add_argument("--execute", action="store_true", help="Execute native jobs; default is safe plan-only")
    experiment.add_argument("--datasets", help="Comma-separated override")
    experiment.add_argument("--sources", help="Comma-separated override")
    experiment.add_argument("--targets", help="Comma-separated override")
    experiment.add_argument("--methods", help="Comma-separated override")
    experiment.add_argument(
        "--studies",
        help="Comma-separated ablation-study override (svfta_ablation protocol)",
    )
    experiment.add_argument("--max-images", type=int)
    experiment.add_argument("--seed", type=int, default=42)
    experiment.add_argument("--output")
    experiment.add_argument(
        "--payload-retention",
        choices=["keep_all", "fixed_count_after_group_validation"],
    )
    experiment.add_argument("--retained-image-count", type=int)
    experiment.add_argument(
        "--prediction-archive", choices=["none", "gzip"]
    )
    _add_common_runtime(experiment)
    _add_prediction_visualization(experiment)

    report = subparsers.add_parser("report", help="Regenerate CSV and TeX from result JSON files")
    report.add_argument("--input", required=True, help="records.json or a directory containing metrics.json")
    report.add_argument("--dataset", required=True, choices=["coco", "voc", "bdd100k"])
    report.add_argument(
        "--metric",
        default="all",
        choices=["all", "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75"],
    )
    report.add_argument("--output", required=True)

    return parser


def _show_models(registry: Registry) -> None:
    if not console.is_terminal:
        print(
            "ORDER\tGROUP\tID\tMODEL\tFRAMEWORK\tFAMILY\tBACKBONE\tSOURCE\tHELD_OUT\tCOCO_AP"
        )
        for model_id in registry.target_ids():
            model = registry.model(model_id)
            print(
                "\t".join(
                    [
                        str(registry.model_position(model.id)),
                        registry.model_group(model.id).display_name,
                        model.id,
                        model.display_name,
                        model.framework,
                        model.family,
                        model.backbone,
                        "yes" if model.source else "",
                        "yes" if model.held_out else "",
                        "{:.1f}".format(model.coco_box_ap),
                    ]
                )
            )
        return
    table = Table(
        title="Fixed detector panel (16 targets / 6 sources)",
        box=box.ASCII,
        safe_box=True,
    )
    for column in (
        "#",
        "Paper group",
        "ID",
        "Model",
        "Framework",
        "Family",
        "Backbone",
        "Source",
        "Held-out",
        "COCO AP",
    ):
        table.add_column(column, overflow="fold")
    for model_id in registry.target_ids():
        model = registry.model(model_id)
        table.add_row(
            str(registry.model_position(model.id)),
            registry.model_group(model.id).display_name,
            model.id,
            model.display_name,
            model.framework,
            model.family,
            model.backbone,
            "yes" if model.source else "",
            "yes" if model.held_out else "",
            "{:.1f}".format(model.coco_box_ap),
        )
    console.print(table)


def _show_compatibility(registry: Registry, method: Optional[str]) -> None:
    methods = [method] if method else list(registry.compatibility["methods"])
    table = Table(
        title=(
            "Attack/source compatibility "
            "(N=native, --=structural skip; A/P are legacy artifact states only)"
        ),
        box=box.ASCII,
        safe_box=True,
    )
    table.add_column("Method")
    for source in registry.source_ids():
        table.add_column(registry.model(source).table_code)
    for attack_id in methods:
        cells = []
        for source in registry.source_ids():
            status = registry.compatibility_status(attack_id, source)["status"]
            cells.append(
                {
                    "native": "N",
                    "external": "A",
                    "adaptable": "P",
                    "unsupported": "--",
                }[status]
            )
        table.add_row(registry.attack(attack_id).display_name, *cells)
    console.print(table)


def _load_records(path: Path) -> List[Dict[str, Any]]:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload["records"]
        raise ValueError("Input JSON must be a record list or contain a records list")
    if not path.is_dir():
        raise FileNotFoundError(path)
    records = []
    for item in sorted(path.rglob("metrics.json")):
        records.append(json.loads(item.read_text(encoding="utf-8")))
    for item in sorted(path.rglob("run.json")):
        payload = json.loads(item.read_text(encoding="utf-8"))
        if payload.get("status") in {
            "skipped",
            "artifact_required",
            "adapter_required",
        }:
            records.append({**payload, "target": "", "metrics": {}})
    return records


def dispatch(args: argparse.Namespace, registry: Registry) -> int:
    if args.command == "doctor":
        report = run_doctor(registry, deep_data=args.deep_data, datasets=args.datasets)
        table = Table(title="LGP environment doctor", box=box.ASCII, safe_box=True)
        table.add_column("Check")
        table.add_column("Status")
        # Fold long Windows paths instead of emitting a Unicode ellipsis;
        # older `conda run` uses the GBK parent console and cannot re-encode it.
        table.add_column("Detail", overflow="fold")
        for check in report["checks"]:
            table.add_row(check["name"], check["status"], check["detail"])
        console.print(table)
        if args.json_path:
            atomic_json(_path(args.json_path), report)
        return 0 if report["status"] == "ok" else 1

    if args.command == "models":
        _show_models(registry)
        return 0
    if args.command == "compatibility":
        _show_compatibility(registry, args.method)
        return 0

    if args.command == "data":
        if args.data_action == "minimize":
            report = minimize_dataset_storage(registry, apply=args.apply)
            console.print_json(data=report)
            if args.json_path:
                atomic_json(_path(args.json_path), report)
            return 0
        ids = list(registry.datasets) if args.dataset == "all" else [args.dataset]
        if args.data_action == "validate":
            reports = {
                dataset_id: validate_dataset(
                    registry.dataset(dataset_id), deep=args.deep, full=args.full
                )
                for dataset_id in ids
            }
            console.print_json(data=reports)
            if args.json_path:
                atomic_json(_path(args.json_path), reports)
            return 0 if all(value["status"].startswith("valid") for value in reports.values()) else 1
        for dataset_id in ids:
            dataset = registry.dataset(dataset_id)
            if args.data_action == "download":
                paths = download_dataset(dataset, full=args.full)
            elif args.data_action == "extract":
                paths = extract_dataset(dataset)
            else:
                paths = prepare_dataset(dataset)
            console.print("{}: {}".format(dataset_id, ", ".join(str(path) for path in paths)))
        return 0

    if args.command == "weights":
        ids = registry.target_ids() if args.models == "all" else registry.expand_ids(_csv(args.models), registry.models)
        console.print_json(data=download_weights(registry, ids))
        return 0

    if args.command == "config":
        model = registry.model(args.model)
        dataset = registry.dataset(args.dataset)
        output = _path(args.output)
        cfg = build_runtime_config(
            model,
            dataset,
            output.parent / (output.stem + "_work"),
            mode=args.mode,
            test_split=args.split,
            training_protocol=(
                registry.training_protocol if args.mode == "train" else None
            ),
            training_safety=(
                registry.training_safety if args.mode == "train" else None
            ),
            warm_start_source=(
                registry.dataset("coco") if args.mode == "train" else None
            ),
        )
        cfg.dump(str(output))
        console.print(str(output))
        return 0

    if args.command == "train":
        ids = (
            registry.target_ids()
            if args.model == "all"
            else registry.expand_ids(_csv(args.model), registry.models)
        )
        base_work_dir = _path(args.work_dir)
        if args.resume is not None and len(ids) != 1:
            raise ValueError("--resume requires exactly one selected model")
        checkpoints = {}
        for model_id in ids:
            model_work_dir = (
                base_work_dir / model_id
                if base_work_dir is not None and len(ids) > 1
                else base_work_dir
            )
            try:
                checkpoint = run_training(
                    registry,
                    args.dataset,
                    model_id,
                    work_dir=model_work_dir,
                    device=args.device,
                    download_weights=args.download_weights,
                    overwrite_checkpoint=args.overwrite_checkpoint,
                    resume=args.resume,
                )
                checkpoints[model_id] = {"status": "complete", "path": str(checkpoint)}
            except Exception as exc:
                checkpoints[model_id] = {
                    "status": "failed",
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                }
                if not args.keep_going:
                    raise
        console.print_json(data=checkpoints)
        return 1 if any(
            record["status"] == "failed" for record in checkpoints.values()
        ) else 0

    if args.command == "attack":
        run_dir = run_attack(
            registry,
            args.dataset,
            args.source,
            args.method,
            split=args.split,
            output_dir=_path(args.output),
            max_images=args.max_images,
            seed=args.seed,
            device=args.device,
            download_weights=args.download_weights,
            keep_going=args.keep_going,
            strict=args.strict,
        )
        console.print("Attack run: {}".format(run_dir))
        if args.evaluate:
            targets = registry.target_ids() if args.targets == "all" else registry.expand_ids(_csv(args.targets), registry.models)
            records = []
            for target in targets:
                result_dir = run_evaluation(
                    registry,
                    args.dataset,
                    target,
                    split=args.split,
                    adversarial_run=run_dir,
                    output_dir=run_dir / "evaluations" / target,
                    max_images=args.max_images,
                    device=args.device,
                    download_weights=args.download_weights,
                    keep_going=args.keep_going,
                    save_visualizations=args.visualize_predictions,
                    visualization_score_threshold=(
                        args.visualization_score_threshold
                    ),
                    visualization_max_images=(
                        args.visualization_max_images
                    ),
                    visualization_max_detections=(
                        args.visualization_max_detections
                    ),
                )
                records.append(json.loads((result_dir / "metrics.json").read_text(encoding="utf-8")))
            write_transfer_reports(records, registry, run_dir / "reports", dataset=args.dataset)
        return 0

    if args.command == "evaluate":
        result = run_evaluation(
            registry,
            args.dataset,
            args.target,
            split=args.split,
            adversarial_run=_path(args.adversarial_run),
            output_dir=_path(args.output),
            max_images=args.max_images,
            device=args.device,
            download_weights=args.download_weights,
            keep_going=args.keep_going,
            save_visualizations=args.visualize_predictions,
            visualization_score_threshold=(
                args.visualization_score_threshold
            ),
            visualization_max_images=args.visualization_max_images,
            visualization_max_detections=(
                args.visualization_max_detections
            ),
            prediction_archive=(
                None
                if args.prediction_archive == "none"
                else args.prediction_archive
            ),
        )
        console.print(str(result))
        return 0

    if args.command == "artifact":
        result = import_external_artifact(
            registry,
            args.dataset,
            args.source,
            args.method,
            annotation=_path(args.annotation),
            image_root=_path(args.image_root),
            clean_root=_path(args.clean_root),
            output_dir=_path(args.output),
            expected_eps=args.expected_eps,
            budget_profile=args.budget_profile,
            gradient_evaluations_per_image=args.gradient_evals_per_image,
            attack_seconds_per_image=args.attack_seconds_per_image,
            peak_memory_mb=args.peak_memory_mb,
            parameters_json=_path(args.parameters_json),
            code_commit=args.code_commit,
        )
        console.print(str(result))
        return 0

    if args.command == "experiment":
        result = run_experiment(
            registry,
            args.protocol,
            execute=args.execute,
            output_dir=_path(args.output),
            dataset_override=_csv(args.datasets),
            source_override=_csv(args.sources),
            target_override=_csv(args.targets),
            method_override=_csv(args.methods),
            study_override=_csv(args.studies),
            max_images=args.max_images,
            seed=args.seed,
            device=args.device,
            download_weights=args.download_weights,
            keep_going=args.keep_going,
            save_visualizations=args.visualize_predictions,
            visualization_score_threshold=(
                args.visualization_score_threshold
            ),
            visualization_max_images=args.visualization_max_images,
            visualization_max_detections=(
                args.visualization_max_detections
            ),
        )
        console.print(str(result))
        return 0

    if args.command == "report":
        records = _load_records(_path(args.input))
        metrics = (
            ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75"]
            if args.metric == "all"
            else [args.metric]
        )
        paths = {}
        output_dir = _path(args.output)
        is_ablation = any(record.get("study") for record in records)
        for metric in metrics:
            if is_ablation:
                generated = write_ablation_reports(
                    records,
                    registry,
                    output_dir,
                    dataset=args.dataset,
                    metric=metric,
                )
                paths.update(
                    {
                        "{}_{}".format(key, metric): value
                        for key, value in generated.items()
                    }
                )
            else:
                generated = write_transfer_reports(
                    records,
                    registry,
                    output_dir,
                    dataset=args.dataset,
                    metric=metric,
                )
                paths["tex_{}".format(metric)] = generated["tex"]
                paths["csv"] = generated["csv"]
        paths.update(
            write_analysis_reports(records, registry, output_dir, dataset=args.dataset)
        )
        paths.update(
            {
                "plot_{}".format(key): value
                for key, value in write_experiment_plots(
                    records,
                    registry,
                    output_dir / "figures",
                    dataset=args.dataset,
                ).items()
            }
        )
        paths.update(
            {
                "catalog_{}".format(key): value
                for key, value in write_paper_catalog(
                    registry, output_dir / "paper_catalog"
                ).items()
            }
        )
        console.print_json(data={key: str(value) for key, value in paths.items()})
        return 0
    raise RuntimeError("Unhandled command: {}".format(args.command))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return dispatch(args, Registry())
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted; completed files were left intact.[/yellow]")
        return 130
    except Exception as exc:
        console.print("[red]{}: {}[/red]".format(type(exc).__name__, exc))
        if args.debug:
            traceback.print_exc()
        return 2
