# Online Prediction Visualization

[English](prediction-visualization-device-interface.md) | [Simplified Chinese](prediction-visualization-device-interface.zh-CN.md)

Run commands from the repository root on the Linux execution server in the
pinned `oda` environment. Owned-process execution uses the existing Linux
containment helpers; these checks do not qualify Windows execution. This
online tool calls the existing evaluator and optionally reads an existing
adversarial manifest. It never generates attacks. Offline figure-rendering tools
remain CPU tools and do not acquire GPU flags.

```bash
conda activate oda
python -B experiments/visualize_predictions.py --help
python -B experiments/visualize_predictions.py --plan-only --dataset coco --devices cuda:0
python -B experiments/visualize_predictions.py --plan-only --dataset voc --targets vfnet_r50 faster_rcnn_r50 --devices cuda:0 cuda:1
```

`--plan-only` reads registry declarations and the existing device-helper source,
prints JSON, and creates no output. It does not validate dataset payloads,
checkpoints, adversarial manifests or GPU availability, import the detector stack,
load models, run inference or calculate AP. A successful plan is not execution
approval or a scientific result.

After the integration owner has qualified the source freeze and authorized GPU
execution, a small clean panel can be requested with:

```bash
python -B experiments/visualize_predictions.py --dataset coco --target faster_rcnn_r50 --device cuda:0 --max-images 20
```

For planning a shared existing adversarial input:

```bash
python -B experiments/visualize_predictions.py --plan-only --dataset voc --split val --adversarial-run outputs/attacks/voc/completed_run --targets vfnet_r50 faster_rcnn_r50 --devices cuda:0 cuda:1
```

Replace `outputs/attacks/voc/completed_run` with the completed attack-run directory
for the selected dataset and split. Remove `--plan-only` only after execution is
authorized. Runtime retains the evaluator's manifest, dataset and split checks.
The maintained interface accepts COCO and VOC only. BDD100K is outside the
current project scope; these checks do not establish project-wide release readiness.

The default remains one complete Faster R-CNN target job, 20 images, overlay
threshold 0.30 and at most 100 drawn detections per image. Two selected GPUs do
not add a second job. `--targets` selects unique registered targets, always in
canonical registry order. One GPU serializes the same jobs; two GPUs assign
independent complete target jobs round-robin. No image sharding, DDP, source
ensemble or attack-budget change is introduced. Each selected GPU must fit its
own detector. Missing requested GPUs fail before output creation.

Legacy `--target` and `--device` remain available, including the legacy single
`cpu` device. `--targets` cannot be combined with `--target`; `--devices` cannot
be combined with `--device`. `--devices` accepts one or two distinct explicit
ordinals such as `cuda:0 cuda:1`, not `cuda`, `cpu` or comma-separated strings.

Single-target metrics and overlays retain the original output layout. Multiple
targets use `targets/TARGET` inside the run directory. The adjacent run directory
with suffix `.dispatch` holds the immutable plan, hashed worker requests, logs,
ownership and terminal cleanup records. Both directories must be new, including
when `--output` is supplied. Default outputs use a fresh UTC timestamp under
`outputs/visualizations`. Open the folder printed for the completed target to
inspect post-NMS boxes, class names and confidence scores.

Overlay threshold and count affect drawing only, not evaluator predictions.
Limited-image AP remains diagnostic. Success requires every selected job to
complete and all owned workers to be reaped. Failures produce a nonzero exit;
`ERR` and unstarted `NR` jobs are not zero metrics. By default independent jobs
continue after a target error. `--stop-on-error` stops on image errors and aborts
pending jobs after a target failure. Interruption terminates owned process groups
and waits for cleanup. Preserve partial output and `.dispatch` evidence; inspect
the logs and use a new output directory for an explicitly authorized rerun.

## Caption Layout

The renderer finishes all box outlines before captions and reserves the title
rectangle before placing detection labels. Labels keep the same class colors,
four-decimal confidence text and source-image scale. Placement tries nearby
positions deterministically, then at most 64 x 64 image-grid origins. Thin
leaders connect materially moved labels to their box anchors and are drawn
before any text. Detection clipping, sorting, threshold and maximum selection
are unchanged; no detection is removed to declutter the image.

The original bearing-aware font fit and scalable-font 8 px fitting floor remain
in place. A caption that cannot fit or find a collision-free position in the
bounded search is not painted, but its box remains and the specific limitation
is recorded. An empty new `caption_layout_limitations` means neither of these
conditions occurred. Historically this field recorded fit failures only; an
empty old list did not claim collision-free layout or invalidate native execution.

Python callers can request `include_layout_rectangles=True` to add optional
`layout_rectangles` to the returned record without removing existing fields.
Each drawn label and title reports a half-open `[left, top, right, bottom]`
rectangle in original image pixels, text and effective font size; detection
labels also identify their sorted detection index, box and any leader line.
These rectangles support direct bounds and non-overlap checks, including title
occupancy. This drawing repair does not change predictions or metrics, and a
new diagnostic rendering does not replace accepted manuscript image identities.
