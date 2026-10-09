# COCO and VOC Detector Checkpoints

[English](README.md) | [简体中文](README.zh-CN.md) | [Project](../README.md)

Run commands from the repository root after following the
[environment](../README.md) and [data](../data/README.md) guides.
[models.yaml](../configs/models.yaml) is the only sixteen-detector registry,
canonical order and training-recipe source. COCO uses released 80-class weights;
VOC needs sixteen separately fine-tuned 20-class detectors, never a substituted COCO head.

## COCO Released Weights

For a fresh checkpoint directory with network access and enough space:

```bash
conda activate oda
python -m lgp weights download --help
python -m lgp weights download --models all
```

`--models all` means the sixteen registered detectors, not all datasets.
Downloads go to `checkpoints/coco/` using registered filenames; the ordered
`checkpoints/coco/manifest.json` records sizes, URLs and SHA-256 values.
Download completion is not strict loading or scientific acceptance.
Preserve existing qualified weights and manifests; the downloader may replace
invalid existing files. Do not use it to repair those files without approved,
verified backup/recovery scope.

## VOC Fine-Tuning Candidate

First prepare the complete VOC07+12 trainval set and the corresponding COCO
source checkpoints. Inspect the full jobs without training:

```bash
python experiments/finetune_voc.py --help
python experiments/finetune_voc.py --plan-only --models all --devices cuda:0
python experiments/finetune_voc.py --plan-only --models all --devices cuda:0,cuda:1
```

Expect a planned status, sixteen canonical model jobs and unchanged registered
epochs, micro-batches and reference batches. Plans go to new UTC leaves under
`outputs/training/voc_finetuning/`. A plan does not qualify weights or devices.
The bounded VOC single/dual-GPU diagnostic has been accepted: two independent
models (Faster R-CNN and Mask R-CNN Swin-T), two training IDs, four jobs,
twelve registered epochs per job and 72 optimizer steps in total, with no
formal validation attached. This does not qualify full VOC training or the
sixteen-checkpoint panel. The device-dispatch reconciliation separately
accepts all 37 declared paths at their roles, with no minimum actual paths
remaining: 10 independent GPU paths, 16 shared-scheduler thin wrappers,
six CPU-only paths and five support paths. It does not establish native GPU
execution of all default studies or every CLI option, scientific acceptance,
redistribution rights or full project release.

Only after the actual device/input checks and an intended new training decision,
choose **one** mode below. Existing qualified checkpoints should be preserved,
not routinely retrained; `--frozen-existing` names already qualified model IDs
(or `all`) to keep, and is not a substitute for their evidence.

```bash
python experiments/finetune_voc.py --models all --devices cuda:0
python experiments/finetune_voc.py --models all --devices cuda:0,cuda:1
```

These are alternatives, not commands to run concurrently. Each GPU runs complete
independent model jobs; there is no DDP, pooled memory, changed batch size or
learning rate. Each device needs enough memory, and disk space must satisfy
`--min-free-gib` (default 12 GiB free before each model; not total storage planning).
`--max-images` is diagnostic training and cannot publish canonical weights.
The default storage check is not an outstanding blocker for the completed
bounded diagnostic; its scope-specific admission did not change this
training default or admit a new run.

Training resolves installed upstream configs and patches them at runtime.
Keep the declared final epoch and linearly scaled learning rate from the registered
micro/reference batch. VOC2007 test must not be attached to the training runner.
Do not choose epochs/checkpoints, stop early or retune from formal AP.

## Readiness and Recovery

Canonical VOC files are `checkpoints/voc/MODEL_ID.pth`, with
`checkpoints/voc/manifest.json` and the associated training/verification records.
A `.pth` file alone is **not ready**. Require the canonical dataset manifest,
source/final SHA-256, annotation and resolved-config hashes, Git provenance,
the fixed final epoch, strict state-dict loading and actual registered train-image
inference verification. Preserve the COCO source provenance as well.

After those artifacts exist, inspect the dataset-scoped preflight on available GPUs:

```bash
python -m lgp doctor --datasets coco,voc --deep-data
```

Require the selected data/checkpoint checks to be ready and resolve warnings;
a zero exit code alone is insufficient. COCO-only work uses `--datasets coco`.
Deep VOC checkpoint verification may also inspect its original COCO source
manifest/weights for provenance; it is not a COCO efficacy evaluation.

Mask R-CNN Swin-T is bbox-only: only released `roi_head.mask_head.*` keys may
be unused. Missing backbone/bbox keys or other unexpected keys are errors.
Any checkpoint change invalidates affected clean/adversarial comparisons.

Preserve failed training leaves, logs and partial checkpoints. Check process
ownership before retrying after disconnect; select the explicitly missing models
in a fresh output directory. Do not overwrite qualified artifacts, merge runs,
assume automatic resume or delete VOC training inputs before the full panel is verified.
Continue with [main reproduction](../docs/current-public-reproduction.md) and
[output retention](../outputs/README.md).
