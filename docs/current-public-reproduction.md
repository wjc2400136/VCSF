# COCO and VOC Main Comparison

[English](current-public-reproduction.md) | [简体中文](current-public-reproduction.zh-CN.md)

Run from this public source root after [environment setup](environment.md),
[data validation](../data/README.md) and
[checkpoint preparation](../checkpoints/README.md). This package covers the COCO/VOC main comparison.

COCO uses all 5,000 val2017 images and released COCO weights. VOC uses all
4,952 VOC2007 test images and sixteen dataset-specific fine-tuned twenty-class
detectors. Never substitute a COCO head. Targets/sources and report order come
from the canonical registries.

## Inspect Before Execution

```bash
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

The default dataset is COCO, with ten default manuscript methods, six independent
sources and sixteen targets. Optional NAA and Corrupting Attention remain main-row
selectors; their structural skips remain `--`. Retired SVFTA cannot be selected.
Ablations are not exposed. Read the job set, population, seed and assignments.

Baseline schedules keep their frozen compute-matched settings. VCSF uses one
detector update plus nineteen feature updates, its complete parameters and actual
reference/partial-backward cost accounting. Neither reference work nor partial
feature gradients are free/calibrated whole-detector backward equivalents.

## Bounded Diagnostic

On the GPU host, choose one mode, not both simultaneously:

```bash
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
```

Use `--devices cuda:0,cuda:1` instead for a deliberate paired device check.
Each mode preserves complete jobs, IDs, seeds, configuration and evaluation.
Two devices schedule independent jobs, not DDP/ensembling/pooled memory.
Explicitly unavailable devices fail; each GPU must fit its own workload.
No bitwise CUDA equality or fixed speedup is promised. Diagnostic mAP is not formal.

## Full Reproduction

Before intentionally running the full workloads:

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

Review the reports, not only exit codes. For COCO-only work use `--datasets coco`.
VOC provenance verification also checks its original COCO source weights; this is
not a separate COCO efficacy run. With qualified inputs and intended compute,
select one device mode and execute each desired dataset:

```bash
python experiments/main_transfer.py --datasets coco --devices cuda:0
python experiments/main_transfer.py --datasets voc --devices cuda:0
```

For two GPUs replace the device argument with `--devices cuda:0,cuda:1`.
Do not launch duplicate commands to implement dual mode. Formal commands omit
`--max-images`.

## Outputs and Recovery

A new UTC leaf under `outputs/experiments/main_transfer/` contains
`plan.json`, `execution_assignments.json`, `execution_state.json`,
`records.json`, `summary.json`, attack PNGs and lossless prediction archives.
Records generate CSV/TeX/plots in `reports/`. Optional main-run overlays draw
post-NMS boxes/classes/scores without changing evaluator predictions.
VCSF `native_cost.jsonl` is a production observation, not a separate paired profiler.

Require intended coverage, complete target results, no failed/missing records and
saved-output checks. Preserve all twelve AP/AR values and unrounded JSON;
`BB Mean` excludes the source.
Use [saved scalar export](saved-report-export.md) for reports without models/AP.

Keep partial leaves and logs. A disconnected terminal does not prove a worker
stopped. Verify ownership, repair the exact cause, then select an explicit missing
subset into a new leaf. Do not overwrite, merge, auto-resume or encode failure as zero.
See [outputs](../outputs/README.md) for the directory layout and record formats.
