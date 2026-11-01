"""Public interface boundaries; no detector, data or CUDA execution."""
from pathlib import Path
import json

import pytest

from lgp.cli import build_parser, dispatch
from lgp.public_package import ENTRY_SUPPORT_FILES, MANDATORY_ENTRIES
from lgp.registry import Registry
from lgp.reporting.reexport import reexport_saved_reports


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("argv", [
    ["data", "download", "--dataset", "all"],
    ["data", "validate", "--dataset", "bdd100k"],
    ["data", "minimize"],
    ["train", "--dataset", "bdd100k", "--model", "faster_rcnn_r50"],
    ["experiment", "--protocol", "current_vcsf_radius"],
    ["experiment", "--protocol", "svfta_ablation"],
    ["artifact", "import"],
])
def test_out_of_scope_cli_is_rejected(argv):
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


def test_preparation_defaults_do_not_select_legacy_data():
    assert build_parser().parse_args(["doctor"]).datasets == "coco,voc"
    assert build_parser().parse_args(["data", "validate"]).dataset == "coco"


@pytest.mark.parametrize("argv, message", [
    (["doctor", "--datasets", "bdd100k"], "only COCO"),
    (["experiment", "--protocol", "main_transfer", "--studies", "anything"], "Ablation"),
    (["attack", "--dataset", "coco", "--source", "faster_rcnn_r50", "--method", "svfta"], "outside"),
])
def test_execution_rejects_scope_before_runtime(argv, message):
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError, match=message):
        dispatch(args, Registry(ROOT))


def test_only_main_and_preparation_direct_entries_are_shipped():
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").glob("*.py")}
    assert files == set(MANDATORY_ENTRIES) | set(ENTRY_SUPPORT_FILES)
    assert len(MANDATORY_ENTRIES) == 6


def test_main_plan_surface_remains_available():
    args = build_parser().parse_args([
        "experiment", "--protocol", "main_transfer", "--datasets", "coco,voc",
        "--sources", "faster_rcnn_r50", "--methods", "vcsf", "--seed", "43",
    ])
    assert args.protocol == "main_transfer" and not args.execute and args.seed == 43


@pytest.mark.parametrize("protocol", [None, "current_paired_cost", "current_vcsf_radius",
                                     "current_oblivious_preprocessing_transfer"])
def test_public_report_rejects_supplementary_runs_before_output(tmp_path, protocol):
    source = tmp_path / "saved"
    source.mkdir()
    (source / "plan.json").write_text(json.dumps({"protocol": protocol}), encoding="utf-8")
    output = tmp_path / "derived"
    with pytest.raises(ValueError, match="only saved COCO/VOC main"):
        reexport_saved_reports(Registry(ROOT), source, output,
                               allowed_protocols=("main_transfer",))
    assert not output.exists()
