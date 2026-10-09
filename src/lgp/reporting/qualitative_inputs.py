"""Read-only, caller-hash-bound inputs for offline qualitative panels.

No inference, AP replay, registry loading, or scientific acceptance occurs here.
Category labels follow the clean annotation's category array (not numeric IDs).
All request paths are absolute; run-owned relative paths stay inside their run.
Metadata, annotations, manifests and images are capped at 512 MiB per input.
Prediction archives and their decompressed JSON are each capped at 8 GiB.
Archives are read in 64 KiB chunks; the parser holds at most one 1 MiB UTF-8
record plus a chunk (Python string/object overhead is additional). Metadata and
selected predictions still occupy memory; unselected prediction records do not
accumulate. The entire array is validated, including gzip EOF/CRC and both hashes.
Clean JPEG/PNG inputs convert to RGB without changing spatial coordinates.
Both clean and adversarial inputs require absent/identity EXIF: the archive does
not bind the evaluator decoder's orientation policy. No boxes are transformed.
"""
import codecs
import gzip
import hashlib
import io
import json
import math
import stat
import warnings
import zlib
from pathlib import Path

from PIL import Image, ImageOps


MAX_INPUT_BYTES = 512 << 20
MAX_ARCHIVE_BYTES = 8 << 30
MAX_DECOMPRESSED_BYTES = 8 << 30
STREAM_CHUNK_BYTES = 64 << 10
COMPRESSED_CHUNK_BYTES = 64 << 10
MAX_RECORD_BYTES = 1 << 20


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _sha(value):
    return (type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite JSON value: " + value)


def _float(value):
    result = float(value)
    _require(math.isfinite(result), "Non-finite JSON number")
    return result


def _json(raw):
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant,
                      parse_float=_float)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _path(value, root=None, directory=False):
    _require(isinstance(value, (str, Path)) and str(value), "Missing input path")
    path = Path(value)
    _require(".." not in path.parts, "Path traversal is forbidden")
    if root is not None and not path.is_absolute():
        _require(not path.drive and not path.root, "Ambiguous relative path")
        path = root / path
    _require(path.is_absolute(), "Input path must be absolute")
    try:
        for part in (path, *path.parents):
            info = part.lstat()
            _require(not stat.S_ISLNK(info.st_mode)
                     and not (getattr(info, "st_file_attributes", 0) & 0x400),
                     "Symlink/reparse input is forbidden")
        resolved = path.resolve(strict=True)
        _require(resolved.is_dir() if directory else resolved.is_file(),
                 "Input has the wrong file type")
    except OSError as exc:
        raise ValueError("Missing or unreadable input: " + str(path)) from exc
    if root is not None:
        _require(root in resolved.parents, "Run input escapes its root")
    return resolved


class _Inputs:
    def __init__(self):
        self.files = {}

    def read(self, path, sha256, size=None):
        _require(_sha(sha256), "Invalid SHA-256")
        path = _path(path)
        if size is not None:
            _require(type(size) is int and 0 < size <= MAX_INPUT_BYTES,
                     "Invalid input byte count")
        _require(path.stat().st_size <= MAX_INPUT_BYTES, "Input exceeds resource cap")
        with path.open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
        _require(len(raw) <= MAX_INPUT_BYTES, "Input exceeds resource cap")
        _require(_hash(raw) == sha256, "Input SHA-256 mismatch: " + str(path))
        _require(size is None or len(raw) == size, "Input byte-count mismatch")
        self.bind(path, sha256, len(raw))
        return raw

    def bind(self, path, sha256, size):
        bound = dict(file=str(path), sha256=sha256, bytes=size)
        _require(str(path) not in self.files or self.files[str(path)] == bound,
                 "Conflicting input binding")
        self.files[str(path)] = bound

    def document(self, ref):
        _require(type(ref) is dict and {"file", "sha256"} <= set(ref),
                 "Missing document binding")
        value = _json(self.read(ref["file"], ref["sha256"]))
        _require(type(value) is dict, "Expected JSON object")
        return value


class _ArchiveReader:
    """Hash compressed bytes as gzip consumes them, without unbounded reads."""

    def __init__(self, handle, expected_size, expected_sha256):
        self.handle = handle
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        size = COMPRESSED_CHUNK_BYTES if size < 0 else min(size, COMPRESSED_CHUNK_BYTES)
        raw = self.handle.read(size)
        self.size += len(raw)
        _require(self.size <= self.expected_size and self.size <= MAX_ARCHIVE_BYTES,
                 "Compressed prediction byte-count/resource cap exceeded")
        self.digest.update(raw)
        return raw

    def verify(self):
        _require(self.size == self.expected_size, "Compressed prediction byte-count mismatch")
        _require(self.digest.hexdigest() == self.expected_sha256,
                 "Prediction archive SHA-256 mismatch")


def _prediction_records(stream, expected_size, expected_sha256):
    """Yield array objects using JSONDecoder; only array delimiters are framed here.

    Objects have an unambiguous closing delimiter, unlike a numeric token split
    across reads. Require an object before raw_decode, then let the strict decoder
    validate all keys, numbers, strings and nesting. Exhaustion verifies EOF/hash.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_constant,
                               parse_float=_float)
    utf8 = codecs.getincrementaldecoder("utf-8")("strict")
    digest, total = hashlib.sha256(), 0
    buffer, position, eof = "", 0, False

    def fill():
        nonlocal buffer, position, eof, total
        raw = stream.read(STREAM_CHUNK_BYTES)
        total += len(raw)
        _require(total <= expected_size and total <= MAX_DECOMPRESSED_BYTES,
                 "Decompressed byte-count/resource cap exceeded")
        digest.update(raw)
        eof = not raw
        buffer = buffer[position:] + utf8.decode(raw, final=eof)
        position = 0
        if eof:
            _require(total == expected_size, "Decompressed byte-count mismatch")
            _require(digest.hexdigest() == expected_sha256, "Decompressed SHA-256 mismatch")

    def token():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position] in " \t\r\n":
                position += 1
            if position < len(buffer):
                return buffer[position]
            if eof:
                return ""
            fill()

    _require(token() == "[", "Prediction JSON must be an array")
    position += 1
    allow_end = True
    while True:
        char = token()
        if char == "]":
            _require(allow_end, "Trailing comma in prediction array")
            position += 1
            _require(token() == "", "Trailing data after prediction array")
            return
        _require(char == "{", "Expected prediction object or truncated array")
        while True:
            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError as exc:
                _require(not eof, "Invalid or truncated prediction JSON: " + str(exc))
                _require(len(buffer[position:].encode("utf-8")) <= MAX_RECORD_BYTES,
                         "Prediction record exceeds resource cap")
                fill()
                continue
            except RecursionError as exc:
                raise ValueError("Prediction JSON nesting exceeds parser capacity") from exc
            _require(len(buffer[position:end].encode("utf-8")) <= MAX_RECORD_BYTES,
                     "Prediction record exceeds resource cap")
            position = end
            yield record
            break
        char = token()
        _require(char in (",", "]"), "Invalid or truncated prediction array delimiter")
        if char == ",":
            position += 1
            allow_end = False
        else:
            allow_end = True


def _index(items, key, name):
    _require(type(items) is list, "Invalid " + name)
    result = {}
    for item in items:
        _require(type(item) is dict and type(item.get(key)) is int
                 and item[key] >= 0 and item[key] not in result,
                 "Invalid or duplicate " + name + " ID")
        result[item[key]] = item
    return result


def _annotation(value):
    images = _index(value.get("images"), "id", "annotation image")
    categories = _index(value.get("categories"), "id", "category")
    _require(images and categories, "Empty annotation images/categories")
    for image in images.values():
        _require(all(type(image.get(k)) is int and image[k] > 0
                     for k in ("width", "height")), "Invalid annotation dimensions")
        _require(type(image.get("file_name")) is str and image["file_name"],
                 "Missing annotation filename")
    names = [c.get("name") for c in categories.values()]
    _require(all(type(n) is str and n.strip() for n in names)
             and len(set(names)) == len(names), "Invalid category names")
    annotations = _index(value.get("annotations"), "id", "GT annotation")
    for item in annotations.values():
        _require(type(item.get("image_id")) is int and item["image_id"] in images
                 and type(item.get("category_id")) is int
                 and item["category_id"] in categories, "Invalid GT identity")
        _box(item.get("bbox"))
    return images, categories, annotations


def _box(box):
    _require(type(box) is list and len(box) == 4 and all(_number(x) for x in box)
             and box[2] >= 0 and box[3] >= 0, "Invalid bbox")
    x, y, w, h = box
    _require(all(_number(v) for v in (x + w, y + h, w * h)), "BBox overflow")
    return [x, y, x + w, y + h]


def _image(inputs, ref, shape, *, clean=False):
    raw = inputs.read(ref["file"], ref["sha256"], ref["bytes"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as image:
                _require(getattr(image, "n_frames", 1) == 1, "Multi-frame input is forbidden")
                if clean:
                    _require(image.format in ("JPEG", "PNG"), "Clean input must be JPEG or PNG")
                    # Evaluation passes filenames to inference_detector, not the
                    # attack loader. Its archived decode flags are not bound here.
                    _require(image.getexif().get(274, 1) == 1,
                             "Clean non-identity EXIF has an unproven prediction coordinate frame")
                    with ImageOps.exif_transpose(image) as oriented:
                        with oriented.convert("RGB") as rgb:
                            rgb.load()
                            _require(rgb.size == (shape["width"], shape["height"]),
                                     "Decoded clean image/annotation shape mismatch")
                else:
                    _require(image.format == "PNG", "Adversarial input must be PNG")
                    _require(image.mode == "RGB", "Adversarial image must decode as RGB")
                    _require(image.getexif().get(274, 1) == 1,
                             "Non-identity EXIF orientation is forbidden")
                    _require(image.size == (shape["width"], shape["height"]),
                             "Image/annotation shape mismatch")
                    image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("Invalid image input") from exc
    return dict(inputs.files[str(_path(ref["file"]))])


def _predictions(inputs, metrics, directory, images, categories, selected):
    artifact = metrics.get("predictions_artifact")
    _require(type(artifact) is dict, "Missing prediction archive binding")
    _require(type(artifact.get("schema_version")) is int and artifact["schema_version"] == 1
             and artifact.get("status") == "verified_lossless_archive"
             and artifact.get("format") == "gzip"
             and artifact.get("archive_file") == "predictions.json.gz"
             and artifact.get("uncompressed_file") == "predictions.json"
             and artifact.get("fields") == ["image_id", "category_id", "bbox", "score"],
             "Unsupported prediction archive schema")
    for field in ("archive_bytes", "uncompressed_bytes", "prediction_records",
                  "image_ids_with_detections"):
        _require(type(artifact.get(field)) is int and artifact[field] >= 0,
                 "Invalid archive count: " + field)
    _require(0 < artifact["archive_bytes"] <= MAX_ARCHIVE_BYTES,
             "Compressed prediction resource cap exceeded")
    _require(2 <= artifact["uncompressed_bytes"] <= MAX_DECOMPRESSED_BYTES,
             "Decompressed prediction resource cap exceeded")
    _require(_sha(artifact.get("archive_sha256"))
             and _sha(artifact.get("uncompressed_sha256"))
             and metrics.get("predictions_sha256") == artifact["uncompressed_sha256"]
             and type(metrics.get("detections")) is int
             and metrics["detections"] == artifact["prediction_records"],
             "Prediction provenance mismatch")
    sidecar = directory / "predictions_artifact.json"
    absent = None
    if sidecar.exists() or sidecar.is_symlink():
        sidecar = _path(sidecar)
        _require(sidecar.stat().st_size <= MAX_INPUT_BYTES, "Sidecar exceeds resource cap")
        with sidecar.open("rb") as handle:
            raw = handle.read(MAX_INPUT_BYTES + 1)
        _require(len(raw) <= MAX_INPUT_BYTES, "Sidecar exceeds resource cap")
        _require(_canonical(_json(raw)) == _canonical(artifact), "Sidecar differs from metrics")
        inputs.read(sidecar, _hash(raw), len(raw))
    else:
        absent = str(sidecar)
    path = _path(artifact["archive_file"], directory)
    _require(path.stat().st_size == artifact["archive_bytes"], "Input byte-count mismatch")
    keep = {i: [] for i in selected}
    detected = set()
    labels = {category: label for label, category in enumerate(categories)}
    count = 0
    try:
        with path.open("rb") as handle:
            archive = _ArchiveReader(handle, artifact["archive_bytes"], artifact["archive_sha256"])
            with gzip.GzipFile(fileobj=archive, mode="rb") as stream:
                for record in _prediction_records(stream, artifact["uncompressed_bytes"],
                                                  artifact["uncompressed_sha256"]):
                    count += 1
                    _require(count <= artifact["prediction_records"], "Prediction record-count mismatch")
                    _require(type(record) is dict
                             and set(record) == {"image_id", "category_id", "bbox", "score"},
                             "Unexpected prediction fields")
                    iid, cid, score = record["image_id"], record["category_id"], record["score"]
                    _require(type(iid) is int and iid in images
                             and type(cid) is int and cid in categories,
                             "Prediction image/category outside evaluation scope")
                    box = _box(record["bbox"])
                    _require(_number(score) and 0 <= score <= 1, "Invalid prediction score")
                    detected.add(iid)
                    if iid in keep:
                        keep[iid].append((box, score, labels[cid]))
            archive.verify()
    except (OSError, EOFError, zlib.error) as exc:
        raise ValueError("Invalid gzip or CRC") from exc
    _require(count == artifact["prediction_records"], "Prediction record-count mismatch")
    _require(len(detected) == artifact["image_ids_with_detections"], "Detected-image count mismatch")
    inputs.bind(path, artifact["archive_sha256"], archive.size)
    return keep, absent


def load_panel_request(request_path, expected_sha256, max_images=None):
    """Return a serializable display plan, preserving declared row/column order.

    ``max_images`` truncates display only, never the complete evaluation scope.
    Cells expose ``detections_available`` before filtering and
    ``detections_selected`` after thresholding and the display count limit.
    All hashes are caller-declared consistency evidence, not independent approval.
    Invalid/missing evidence raises ValueError (malformed JSON is a subclass).
    """
    try:
        return _load(request_path, expected_sha256, max_images)
    except (KeyError, TypeError, OSError, OverflowError) as exc:
        raise ValueError("Malformed or unreadable panel input: " + str(exc)) from exc


def _load(request_path, expected_sha256, max_images):
    inputs = _Inputs()
    request = inputs.document(dict(file=request_path, sha256=expected_sha256))
    version = request.get("schema_version")
    _require(type(version) is int and version in (1, 2), "Unsupported request schema")
    keys = {"schema_version", "dataset", "split", "source", "target",
            "image_ids", "score_threshold", "max_detections", "originals", "selection_note", "rows"}
    if version == 1:
        keys.add("budget_profile")
    _require(set(request) == keys, "Unexpected or missing request fields")
    identity_fields = ("dataset", "split", "source", "target", "selection_note")
    if version == 1:
        identity_fields += ("budget_profile",)
    for key in identity_fields:
        _require(type(request[key]) is str and request[key].strip(), "Empty request " + key)
    _require(request["dataset"] in ("coco", "voc", "bdd100k"), "Unsupported dataset")
    ids = request["image_ids"]
    _require(type(ids) is list and 1 <= len(ids) <= 6
             and all(type(i) is int and i >= 0 for i in ids)
             and len(set(ids)) == len(ids), "Explicit unique image_ids (1..6) required")
    _require(max_images is None or type(max_images) is int and 1 <= max_images <= 6,
             "max_images must be an integer in 1..6")
    selected = ids[:max_images] if max_images is not None else list(ids)
    threshold = request["score_threshold"]
    _require(_number(threshold) and 0 <= threshold <= 1, "Invalid score threshold")
    _require(type(request["max_detections"]) is int and request["max_detections"] > 0,
             "Invalid max_detections")
    _require(type(request["originals"]) is bool, "originals must be boolean")
    rows = request["rows"]
    _require(type(rows) is list and 2 <= len(rows) <= 12, "Expected 2..12 rows")
    _require(all(type(r) is dict for r in rows), "Invalid row")
    _require(rows[0].get("attack") == "clean"
             and sum(r.get("attack") == "clean" for r in rows) == 1, "Exactly one clean first row required")
    clean = rows[0]
    baseline = inputs.document(clean["annotation"])
    images, categories, gt = _annotation(baseline)
    _require(set(ids) <= set(images), "Selected image missing from annotation")
    clean_refs = _index(clean["clean_images"], "image_id", "clean image")
    _require(set(clean_refs) == set(ids), "clean_images must cover exactly declared image_ids")
    for iid in ids:
        filename = Path(images[iid]["file_name"])
        _require(".." not in filename.parts and filename.parts, "Invalid clean annotation filename")
        actual = _path(clean_refs[iid]["file"])
        if filename.is_absolute():
            _require(_path(filename) == actual, "Clean annotation filename mismatch")
        else:
            _require(not filename.drive and not filename.root
                     and actual.parts[-len(filename.parts):] == filename.parts,
                     "Clean annotation filename mismatch")
    clean_bound = {i: _image(inputs, clean_refs[i], images[i], clean=True) for i in ids}
    ids_hash = _hash(_canonical(sorted(images)))
    plan_rows, absent_sidecars = [], []
    checkpoint = None
    for row in rows:
        is_clean = row["attack"] == "clean"
        required = {"label", "attack", "metrics"} | ({"annotation", "clean_images"} if is_clean else {"generation"})
        if version == 2 and not is_clean:
            required.add("budget_profile")
        _require(set(row) == required, "Unexpected or missing row fields")
        for key in ("label", "attack"):
            _require(type(row[key]) is str and row[key].strip(), "Empty row " + key)
        budget_profile = None if is_clean else (
            request["budget_profile"] if version == 1 else row["budget_profile"])
        if not is_clean:
            _require(type(budget_profile) is str and budget_profile.strip(),
                     "Empty row budget_profile")
        metrics = inputs.document(row["metrics"])
        for key in ("dataset", "split", "target"):
            _require(metrics.get(key) == request[key], "Metrics identity mismatch: " + key)
        _require(metrics.get("attack") == row["attack"], "Metrics attack mismatch")
        _require(_sha(metrics.get("checkpoint_sha256")), "Missing target checkpoint hash")
        if checkpoint is None:
            checkpoint = metrics["checkpoint_sha256"]
        _require(metrics["checkpoint_sha256"] == checkpoint, "Target checkpoint mismatch")
        _require(metrics.get("status") == "complete" and metrics.get("failures") == []
                 and type(metrics.get("images")) is int and metrics["images"] == len(images)
                 and metrics.get("evaluated_image_ids_sha256") == ids_hash
                 and metrics.get("expected_image_ids_sha256") == ids_hash
                 and metrics.get("evaluated_image_ids_match_expected") is True,
                 "Incomplete annotation evaluation scope (partial clean evaluations unsupported)")
        if is_clean:
            _require(metrics.get("adversarial_run") is None
                     and metrics.get("parameters_sha256") is None, "Clean row contains attack provenance")
            if version == 2:
                _require(metrics.get("budget_profile") is None,
                         "Clean row contains attack budget provenance")
            annotation_path = _path(clean["annotation"]["file"])
            bound = clean_bound
        else:
            run_path = _path(row["generation"]["file"])
            run = inputs.document(row["generation"])
            root = run_path.parent
            _require(_path(metrics.get("adversarial_run"), directory=True) == root,
                     "Metrics/generation root mismatch")
            for key in ("dataset", "split", "source"):
                _require(run.get(key) == request[key], "Generation identity mismatch: " + key)
            _require(metrics.get("source") == request["source"], "Metrics identity mismatch: source")
            _require(run.get("budget_profile") == budget_profile,
                     "Generation identity mismatch: budget_profile")
            _require(metrics.get("budget_profile") == budget_profile,
                     "Metrics identity mismatch: budget_profile")
            _require(run.get("status") == "complete" and run.get("attack") == row["attack"]
                     and type(run.get("parameters")) is dict
                     and _sha(run.get("parameters_sha256"))
                     and _hash(_canonical(run["parameters"])) == run["parameters_sha256"]
                     and metrics.get("parameters_sha256") == run["parameters_sha256"],
                     "Generation method/parameters mismatch")
            _require(all(type(run.get(k)) is int and run[k] == v for k, v in
                         (("requested_images", len(images)), ("successful_images", len(images)), ("failed_images", 0))),
                     "Incomplete generation")
            annotation_path = _path(run["annotation"], root)
            adv = inputs.document(dict(file=annotation_path, sha256=run["annotation_sha256"]))
            adv_images, adv_categories, adv_gt = _annotation(adv)
            _require(_canonical(list(adv_categories.values())) == _canonical(list(categories.values()))
                     and _canonical(adv_gt) == _canonical(gt) and set(adv_images) == set(images),
                     "Adversarial categories/GT/image IDs differ")
            for iid in images:
                _require(_canonical({k: v for k, v in images[iid].items() if k != "file_name"})
                         == _canonical({k: v for k, v in adv_images[iid].items() if k != "file_name"}),
                         "Adversarial image metadata differs beyond filename")
            raw_manifest = inputs.read(_path(run["manifest"], root), run["manifest_sha256"])
            manifest = _index([_json(line) for line in raw_manifest.splitlines() if line.strip()],
                              "image_id", "manifest image")
            _require(set(manifest) == set(images)
                     and all(m.get("status") == "ok" for m in manifest.values()), "Incomplete generation manifest")
            bound = {}
            for iid in ids:
                item = manifest[iid]
                _require(_path(item["source_file"]) == _path(clean_refs[iid]["file"]),
                         "Manifest source differs from clean image")
                output = _path(item["output_file"], root)
                _require(output == _path(adv_images[iid]["file_name"], root),
                         "Manifest/annotation output mismatch")
                bound[iid] = _image(inputs, dict(file=str(output), sha256=item["output_sha256"],
                                                bytes=item["output_bytes"]), images[iid])
        _require(_path(metrics.get("annotation")) == annotation_path, "Metrics annotation path mismatch")
        if "annotation_sha256" in metrics:
            _require(metrics["annotation_sha256"] == inputs.files[str(annotation_path)]["sha256"],
                     "Metrics annotation hash mismatch")
        predictions, absent = _predictions(inputs, metrics, _path(row["metrics"]["file"]).parent,
                                          images, categories, selected)
        if absent:
            absent_sidecars.append(absent)
        cells = []
        for iid in selected:
            detections = sorted((p for p in predictions[iid] if p[1] >= threshold),
                                key=lambda p: -p[1])[:request["max_detections"]]
            cells.append(dict(image_id=iid, image=bound[iid], width=images[iid]["width"],
                              height=images[iid]["height"], boxes=[p[0] for p in detections],
                              scores=[p[1] for p in detections], labels=[p[2] for p in detections],
                              detections_available=len(predictions[iid]),
                              detections_selected=len(detections)))
        plan_rows.append(dict(label=row["label"], attack=row["attack"],
                              budget_profile=budget_profile, cells=cells))
    plan = {k: request[k] for k in ("dataset", "split", "source", "target",
                                    "score_threshold", "max_detections", "originals", "selection_note")}
    plan.update(schema_version=version, budget_profile=request.get("budget_profile"),
                budget_binding="per_attack_row" if version == 2 else "shared_request",
                image_ids=selected, requested_image_ids=list(ids),
                max_images=max_images, display_smoke=max_images is not None,
                evaluation_images=len(images), evaluation_image_ids_sha256=ids_hash,
                checkpoint_sha256=checkpoint, rows=plan_rows,
                class_names=[c["name"] for c in categories.values()],
                image_coordinate_contract={
                    "schema_version": 1,
                    "frame": "annotation_sized_stored_image_pixels",
                    "clean_decode": "single_frame_jpeg_or_png_to_rgb",
                    "adversarial_decode": "single_frame_rgb_png",
                    "exif_orientation": "absent_or_1_only_all_rows",
                    "nonidentity_exif": "rejected_evaluator_decode_policy_not_bound",
                    "evaluator_decode_policy_verified": False,
                    "image_spatial_transform": "none",
                    "boxes_input": "archived_coco_xywh",
                    "boxes_output": "xyxy_same_pixel_frame",
                    "box_spatial_transform": "none",
                    "shape_check": "decoded_width_height_equal_annotation",
                },
                model_calls=0, AP_replays=0, scientific_acceptance=False,
                independent_acceptance=False, formal_metrics_eligible=False,
                evidence_inputs=list(inputs.files.values()), absent_sidecars=absent_sidecars,
                request_sha256=expected_sha256)
    plan["evidence_sha256"] = _hash(_canonical(plan))
    verify_panel_inputs(plan)
    return plan


def verify_panel_inputs(plan):
    """Rehash every consumed input and check the plan seal; return None.

    Call immediately before rendering. This is not an atomic filesystem snapshot
    and cannot prevent a later external write. No evidence files are written.
    """
    _require(type(plan) is dict and _sha(plan.get("evidence_sha256")), "Missing plan evidence seal")
    _require(_hash(_canonical({k: v for k, v in plan.items() if k != "evidence_sha256"}))
             == plan["evidence_sha256"], "Plan evidence seal mismatch")
    _require(type(plan.get("evidence_inputs")) is list and plan["evidence_inputs"], "Missing evidence inputs")
    for ref in plan["evidence_inputs"]:
        path = _path(ref["file"])
        cap = MAX_ARCHIVE_BYTES if path.name == "predictions.json.gz" else MAX_INPUT_BYTES
        _require(type(ref.get("bytes")) is int and 0 <= ref["bytes"] <= cap
                 and _sha(ref.get("sha256")), "Invalid evidence input binding/resource cap")
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(STREAM_CHUNK_BYTES), b""):
                size += len(block)
                _require(size <= cap and size <= ref["bytes"], "Input exceeds resource cap/byte count")
                digest.update(block)
        _require(size == ref["bytes"] and digest.hexdigest() == ref["sha256"],
                 "Panel input changed: " + str(path))
    for value in plan["absent_sidecars"]:
        path = Path(value)
        _require(not path.exists() and not path.is_symlink(), "Prediction sidecar appeared after loading")
