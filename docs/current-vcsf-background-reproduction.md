# Current VCSF Final-Background Reproduction

[English](current-vcsf-background-reproduction.md) | [简体中文](current-vcsf-background-reproduction.zh-CN.md)

This fresh-input entry reproduces selected A10 and its nine registered controls.
It generates a fresh A10 anchor and does not require the author's historical
results. Use the [main guide](current-public-reproduction.md) for shared setup.
The entry has completed bounded CPU qualification in pinned ODA: 100 affected
contract/regression cases passed, including the real serial coordinator with
mocked producer/evaluator boundaries. CLI help, both full plan-only modes,
compilation, model registry, compatibility and the current main plan passed.
The two supported independent-review findings were fixed and rechecked.

A subsequent actual single/dual-GPU diagnostic has been accepted for all ten
registered rows (A10 and F01-F09), the Faster R-CNN source, three selected COCO
targets and an image cap of one. Its report-only repairs preserve the original
raw records and failed roots. This does not qualify the full 5,000-image,
sixteen-target workload; commands below remain usage examples, not accepted
efficacy results.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Scope

The sole source is `faster_rcnn_r50`. Every row uses all 5,000 COCO val2017 images,
seed 42 plus the canonical numeric image-position offset, and all sixteen targets.
The canonical row order comes directly from
[`vcsf_final_background_validation`](../configs/experiments/ablations.yaml):

| Row | Change from A10 |
| --- | --- |
| A10 | Complete selected method; newly generated reference. |
| F01 | Identity geometric view. |
| F02 | Neck-only feature surface. |
| F03 | Backbone-only feature surface. |
| F04 | Independent offsets for the clean/adversarial views. |
| F05 | Independent geometry for the clean/adversarial views. |
| F06 | No detector initialization; 20 feature-gradient updates. |
| F07 | Random-sign initialization; 20 feature-gradient updates. |
| F08 | Uniform spatial weighting. |
| F09 | Mean feature aggregation. |

All rows retain `eps=4/255`, step size `1/255`, and the registered 20 logical
gradient updates. Detector-initialized rows use one detector gradient and 19
feature gradients. F06/F07 instead use 20 feature gradients; this is the original
budget-internal optimization comparison, not an equal-feature-update causal
contrast. F02/F03 change the representation and number of terms together.
Reference forwards, transformed views and partial feature backwards are retained
in observations, not counted as free work or calibrated whole-detector BE.

The full fresh workload contains ten generation groups, 160 adversarial target
cells and sixteen shared clean cells. One GPU serializes this exact workload;
two GPUs schedule independent complete groups without DDP, ensembling or changes
to batch size, image IDs, seed, checkpoints or scientific settings. Each device
must independently fit its detector. Explicit unavailable devices fail execution.

This is not a new configuration search or independent confirmation of selection.
COCO val2017 and DINO were observed during offline selection; DINO is absent from
the source set, not an untouched selection holdout. A10 is not automatically
replaced by a control. For the author, the existing historical exact-reuse route
remains separate; this public entry does not authorize rerunning accepted panels
or relabelling A01/A23 outcomes as A10.

## Prepare Inputs

Run commands from the repository root. Follow the [root setup](../README.md),
[COCO data](../data/README.md) and [checkpoint](../checkpoints/README.md) guides.
Reuse the pinned `oda` environment without upgrades. This entry needs the complete
COCO validation manifest/images and sixteen registered COCO checkpoints; it needs
neither VOC nor BDD100K. It does not download data or weights. Allocate storage
for all ten PNG sets and complete prediction archives; automatic pruning or resume
is not provided.

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

Proceed only with no failed readiness checks. Readiness is not experiment acceptance.

## Inspect and Run

```bash
python experiments/current_vcsf_final_background.py --help
python experiments/current_vcsf_final_background.py --plan-only --devices cuda:0
python experiments/current_vcsf_final_background.py --plan-only --devices cuda:0,cuda:1
```

Both plans must contain the same ten rows and 160 scientific cells. Plan-only
runs no GPU inference and does not prove device availability.

A bounded diagnostic still covers the ten registered rows, but limits images and
targets. It cannot supply formal mAP:

```bash
python experiments/current_vcsf_final_background.py --devices cuda:0,cuda:1 --max-images 1 --targets faster_rcnn_r50,cascade_rcnn_r50
```

For full-population reproduction, omit both limiting options:

```bash
python experiments/current_vcsf_final_background.py --devices cuda:0,cuda:1
```

Use `--devices cuda:0` for the same work in serial. Do not launch two copies of
the same command. `--device cuda:0` is a mutually exclusive single-device alias.
`--output` selects a new empty run leaf; `--visualize-predictions` optionally draws
post-NMS boxes, classes and scores at the registered 0.50 threshold, maximum 100
boxes and at most three images per target. Visualizations do not change predictions.

## Outputs and Recovery

The terminal reports `[OUTPUT]` followed by the timestamped run directory beneath
`outputs/experiments/current_vcsf_final_background/`. Check the original plan,
execution assignments, `execution_state.json`, `records.json`, group summaries, PNG
manifests, lossless saved predictions and observed cost records. Successful
production requires complete coverage and zero failures, not merely a surviving
PID or one completed target.

`background_summary.json` retains all unrounded twelve metrics, source-excluded
BB Mean and all nine control-minus-A10 contrasts, including per-target deltas.
`reports/background/A10/` and `F01/` through `F09/` provide all twelve per-target
CSV/TeX tables in canonical detector order. `reports/background/summary.csv` and
`summary.tex` display BB Mean and contrasts to four decimal AP/AR percentage
points. Raw JSON remains on the official 0-to-1 scale. Smaller attacked AP means
a stronger attack; a positive control-minus-A10 AP contrast favours A10.
Failed aggregates show `ERR`, unrun/insufficient aggregates show `NR`, and raw
missing values remain null, never zero. The A10 self-contrast is labelled reference.

Independent pixel/prediction coverage, official saved-prediction replay, provenance,
and scientific interpretation remain separate from this producer's completion.
Do not turn limited-image output, a CPU test, or a plan into formal efficacy.
Keep failed directories and complete prefixes. After correcting the actual cause,
use a new output leaf for explicitly scoped recovery; do not merge immutable runs,
restart after a connection timeout, delete retained payloads, or inherit author
acceptance. There is no automatic numerical-result promotion.
