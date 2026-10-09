# Optional COCO and VOC White-Box Diagnostics

[English](current-whitebox-reproduction.md) | [简体中文](current-whitebox-reproduction.zh-CN.md)

Run every command from the repository root. This guide covers the optional
`experiments/whitebox.py` entry for a fresh user's source-matched diagonal
diagnostics. The [main reproduction](current-public-reproduction.md) already
includes each source as an evaluation target. Reuse its accepted white-box rows
when the inputs, method configuration and checkpoint identities match; do not
rerun an accepted matrix or generate extra scientific evidence for this guide.
The separate white-box entry does not inherit historical acceptance.

Actual single/dual-GPU diagnostics have completed for COCO/VOC Common-2,
LGP/VCSF self-target cells and one image per group. Main qualified the original
pixel, prediction and numerical content. Original dual registration remains
`NR`; a separate later dual diagnostic has accepted journal/registration evidence
for this limited scope, without backfilling the old records. Report-only repairs
have direct source review and saved compile/render receipts. This does not
qualify the six-source, full-population panel or certify off-diagonal transfer.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Preconditions

Use the pinned `oda` environment from [environment.yml](../environment.yml) and
the [locked requirements](../requirements/locked-cu118.txt).
Keep Python 3.8.20, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, MMCV 2.0.1,
MMEngine 0.7.4, MMDetection 3.0.0 and MMYOLO 0.6.0 together; do not upgrade one
component to make this optional entry work.

On a fresh machine without an `oda` environment, create it from the repository
root; do not recreate or migrate an existing pinned environment:

```powershell
conda env create -f environment.yml
conda activate oda
```

The direct Python entry adds the repository source directory to its import path.
Prepare and validate the registered dataset manifests and detector checkpoints.
COCO full-population diagnostics use all 5,000 val2017 images. VOC uses all
4,952 VOC2007 test images registered as `val`, with dataset-specific fine-tuned
checkpoints, not an 80-class COCO head. BDD100K is not supported by this entry.
For each diagonal cell, generation and evaluation must use the same current
detector identity, dataset label space, resolved configuration and checkpoint
SHA-256. A filename alone is not checkpoint qualification.

Actual execution needs one or two available GPUs, enough memory on each selected
GPU for its complete detector/attack job, and storage for all generated PNGs and
lossless gzip predictions. Two GPUs do not pool memory. The checks below must
succeed before starting a full-population diagnostic:

```powershell
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

For a COCO-only run, use the COCO data-validation command and doctor
`--datasets coco`; validate VOC as well when selecting VOC. Do not use the
unscoped validation defaults, which also select BDD100K. Doctor limits the data
and checkpoint panels to the requested datasets. Deep VOC checkpoint validation
still verifies the original COCO source weights and manifest as part of training
provenance, not a COCO dataset evaluation. Inspect the selected datasets' warnings
and qualification: doctor exit code zero alone does not prove that their data
and checkpoints are ready.

## Inspect the Plan

These commands do not load images or models or execute GPU jobs:

```powershell
python experiments/whitebox.py --help
python experiments/whitebox.py --plan-only --devices cuda:0
python experiments/whitebox.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

The entry defaults to COCO. Its registered scope is COCO and VOC only. Each
dataset has 60 source-method cells: 52 ready jobs and eight audited structural
skips. The ten default methods match the current main experiment in order:
`numbod`, `hifa`, `mlfadv`, `tog`, `augtrans`, `lgp`, `afog`, `osfd`, `sfim_b`,
`vcsf`. The six canonical sources are `faster_rcnn_r50`, `mask_rcnn_swin_t`,
`yolov3_d53`, `vfnet_r50`, `sparse_rcnn_r50` and `deformable_detr_r50`.

Every target is its own source. Do not pass `--targets`, even to restate a source;
any supplied target override is rejected. Select dataset, method and source
subsets with `--datasets`, `--methods` and `--sources`; the planner preserves
canonical source/method order, explicit dataset input order and compatibility
skips. Optional `naa` is defined only on
Faster R-CNN, and `corrupting_attention` only on Deformable DETR. Select them
explicitly. The retired `svfta` cannot be selected.
Repeated method IDs are rejected instead of creating duplicate jobs.

Baselines retain their main `compute_matched` schedules. VCSF retains the same
`vcsf_final_background_observed_cost` profile and selected implementation as
main. Its reference forwards and partial feature backwards are observed costs,
not free work or calibrated complete-detector backward equivalents. Changing
the number of devices does not change these budgets.

## Bounded Diagnostics

Without `--plan-only`, this file executes by default and stops on the first
error. The following commands use one image per independent source job. Run one
mode at a time after checking device availability; these are not formal mAP:

```powershell
python experiments/whitebox.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/whitebox.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

Both modes run the same complete jobs, images, seeds, parameters and diagonal
evaluations. One GPU serializes them; two GPUs schedule independent complete
groups with one owner per group and a coordinator-owned result writer. This is
not DDP, source ensembling, a reduced matrix or a batch-size/learning-rate change.
Explicitly unavailable devices fail before execution starts. Bitwise-identical
CUDA outputs and a twofold speedup are not promised.

Only a fresh user who intends an optional full-population diagnostic should omit
`--max-images` after the preconditions succeed:

```powershell
python experiments/whitebox.py --datasets coco --devices cuda:0,cuda:1
python experiments/whitebox.py --datasets voc --devices cuda:0,cuda:1
```

Replace the device list with `cuda:0` for the same workload on one GPU. Do not
launch duplicate commands to simulate dual-GPU scheduling. These commands are
not an instruction to repeat the author's already accepted main white-box rows.

## Outputs and Recovery

Each invocation creates a new UTC leaf under `outputs/experiments/whitebox/`.
Inspect `plan.json`, `provenance.json`, `execution_assignments.json`,
`execution_state.json`, `records.json` and `summary.json`. The lifecycle is
`keep_all`: retain all generated PNGs and lossless gzip predictions. Optional
post-NMS prediction overlays do not change evaluator predictions. VCSF attack
directories also retain per-image `native_cost.jsonl` observations.

Raw JSON retains unrounded values for all twelve standard bbox metrics: AP,
AP50, AP75, AP-small/medium/large, AR1, AR10, AR100 and AR-small/medium/large.
`reports/` contains record-generated CSV, metric-specific TeX and plots. Plan
reports have `NR` for unrun metrics and `--` for audited structural skips;
execution failures are `ERR`, never zero. Source targets carry the source
`\dagger` marker. The standard transfer-table layout may show off-diagonal
columns, but those cells are not run by this entry: there is no black-box panel
and no available black-box mean or ranking. A missing `BB Mean` must not be
filled with zero or used to infer black-box strength.

Completion requires intended image and prediction coverage, all twelve metrics
for every ready diagonal cell, zero failed records and saved-output checks.
A successful exit, full image count or unchanged source freeze alone is not
formal numerical acceptance. Do not insert `--max-images` results or optional
diagnostics into efficacy tables as additional accepted scientific evidence.

On failure, preserve the original directory, partial records and logs. Confirm
the actual process identity before concluding it stopped; losing SSH visibility
does not authorize a second launch. Fix the identified cause, inspect a plan for
the explicitly missing dataset/source/method subset, and use a fresh output leaf.
There is no automatic resume, overwrite or merge of immutable run directories.
Do not prune images, trim predictions, change weights or widen method scope as a
failure-recovery shortcut.
