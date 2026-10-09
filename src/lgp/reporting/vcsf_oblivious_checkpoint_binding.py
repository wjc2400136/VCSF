"""Derive target identities from a hash-pinned accepted main-result export."""
import json
from pathlib import Path

from ..io import atomic_json, file_digest
from ..runners.vcsf_final_oblivious_contract import PROTOCOL


def derive(registry, report_path, report_sha256):
    evidence = {}
    def read(path, digest):
        path = Path(path).resolve()
        if file_digest(path) != digest:
            raise ValueError("Accepted evidence hash mismatch: " + path.name)
        data = json.loads(path.read_text(encoding="utf-8"))
        if file_digest(path) != digest:
            raise ValueError("Evidence changed while read")
        evidence[str(path)] = digest
        return data
    report = read(report_path, report_sha256)
    spec = registry.protocols[PROTOCOL]
    if (report.get("independent_result_acceptance") is not True
            or report.get("parameters_sha256") != spec["parameters_sha256"]
            or report.get("targets") != registry.paper_order):
        raise ValueError("Export is not the accepted canonical frozen-method panel")
    groups = [row for row in report["groups"] if row.get("dataset") == "coco"
        and row.get("source") == "mask_rcnn_swin_t" and row.get("seed") == 42
        and "coco_main_vcsf" in row.get("report_roles", [])]
    if len(groups) != 1:
        raise ValueError("Expected exactly one accepted Mask-Swin COCO seed42 panel")
    cells = [row for row in report["cells"] if row["group_id"] == groups[0]["group_id"]]
    if [row["target"] for row in cells] != registry.paper_order:
        raise ValueError("Missing, duplicate or reordered target cells")
    bindings = {}
    for cell in cells:
        expected = dict(dataset="coco", split="val", source="mask_rcnn_swin_t", seed=42,
            parameters_sha256=spec["parameters_sha256"], images=spec["full_images"],
            image_ids_sha256=spec["full_image_ids_sha256"])
        if any(cell.get(key) != value for key, value in expected.items()):
            raise ValueError("Accepted cell scope mismatch")
        ref = cell["record_reference"]
        replay = read(ref["file"], ref["sha256"])
        archive = replay["archive_binding"]
        record = archive["metrics"]
        paths = [(path, digest) for path, digest in
            replay["input_binding"]["checked_metadata_sha256"].items()
            if Path(path).name == "metrics.json"]
        if len(paths) != 1 or paths[0][1] != archive["metrics_sha256"]:
            raise ValueError("Raw metric binding is ambiguous")
        if read(*paths[0]) != record:
            raise ValueError("Replay record differs from original metrics")
        target = cell["target"]
        if (record.get("status") != "complete" or record.get("failures") != []
                or record.get("dataset") != "coco" or record.get("split") != "val"
                or record.get("source") != "mask_rcnn_swin_t" or record.get("target") != target
                or record.get("parameters_sha256") != spec["parameters_sha256"]
                or record.get("images") != spec["full_images"]
                or record.get("evaluated_image_ids_sha256") != spec["full_image_ids_sha256"]
                or record.get("metrics") != cell["metrics"]):
            raise ValueError("Raw record does not support the accepted target cell")
        checkpoint = Path(record["checkpoint"]).resolve()
        digest = record["checkpoint_sha256"]
        if file_digest(checkpoint) != digest:
            raise ValueError("Current checkpoint bytes differ from accepted execution")
        evidence[str(checkpoint)] = digest
        bindings[target] = dict(path=str(checkpoint), sha256=digest)
    return bindings, evidence


def export(registry, report_path, report_sha256, output):
    bindings, evidence = derive(registry, report_path, report_sha256)
    output = Path(output).resolve()
    for name in evidence:
        parent = Path(name).resolve().parent
        if output == parent or parent in output.parents or output in parent.parents:
            raise ValueError("Binding output overlaps accepted evidence")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "checkpoints.json", bindings)
    receipt = dict(status="accepted_main_checkpoints_bound", targets=registry.paper_order,
        checkpoints_sha256=file_digest(output / "checkpoints.json"), evidence=evidence,
        model_calls=0, formal_execution_admitted=False)
    atomic_json(output / "receipt.json", receipt)
    return receipt
