# Current Training-State Reproduction

[English](current-training-state-reproduction.md) | [简体中文](current-training-state-reproduction.zh-CN.md)

The entry point is `experiments/training_state_transfer.py`. It evaluates current
COCO Swin-T source PNGs against the existing registered Faster R-CNN victim pair.
It generates no attack images and performs no training.

Actual single/dual-GPU diagnostics have completed for clean plus LGP/VCSF
against both registered victim states, using a six-image COCO prefix and its
one-image retained projection. Main qualified the original numerical, input and
sampled-ownership content. Original stale `running` snapshots remain unchanged;
native terminals are authoritative. Later source/report repairs address scope
labels and future terminal-state publication, with direct source review and
saved export checks. The repaired bounded single/dual terminal and cleanup
contract has separately been accepted without relabelling those original
snapshots. This does not qualify the default ten-method full panel or
grant strict controlled-training or scientific acceptance.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Prerequisites

Run all commands from the repository root with the existing `oda` environment
activated (`conda activate oda`). Follow the [current reproduction guide](current-public-reproduction.md)
for environment and data preparation; do not independently upgrade its pinned
packages. GPU execution requires Linux process ownership, one or two available
explicit CUDA devices, and sufficient memory on each selected device. Two GPUs
do not pool memory. Plan-only inspection does not require GPUs or input artifacts.

For the final COCO/VOC reproduction, validate each dataset separately and limit
the doctor data/checkpoint checks explicitly:

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

Require valid dataset reports, an `ok` doctor status and `ok` COCO/VOC checkpoint
checks before GPU execution. BDD100K is not a prerequisite for this reproduction.
Omitting `--datasets` retains all registered datasets; use `--datasets coco` for
this COCO-only entry when checking it independently. Environment and NMS checks
are unchanged, so run the real doctor only when CUDA resources are available.

Use the already qualified `standard_control` / `adversarial_training` pair.
Its recipe and relative artifact locations remain registered in
`configs/experiments/training_state_transfer.yaml`: paired continuation from an
identical initial detector checkpoint, one fixed final epoch, four replay steps
per minibatch, batch size two, the registered optimizer/schedule and train-image
coverage, and the registered adversarial branch with epsilon and step size `4/255`.
This entry does not retrain or select a checkpoint. Do not retrain an existing
qualified pair merely to use this interface. A `.pth` file alone is insufficient:
the pair qualification must authenticate the training traces, invariants and
checkpoint verification. Full-split execution requires
`strict_training_controlled_pair_eligible`; a `--max-images` diagnostic requires
`diagnostic_pair_eligible` and is not a strict controlled-training result.

## Inputs and Scope

The ten methods are inherited in order from the current `main_transfer` protocol:
`numbod`, `hifa`, `mlfadv`, `tog`, `augtrans`, `lgp`, `afog`, `osfd`, `sfim_b`, `vcsf`.
This training-state experiment is COCO-only, with `mask_rcnn_swin_t` as the source
and `faster_rcnn_r50` as the victim. Historical training-state method lists do not
replace the current main list. Baselines keep their registered compute-matched
budget; VCSF keeps its registered observed-path budget and parameter identity.

`--source-main` must identify a current main run containing `plan.json`,
`provenance.json`, and preserved Swin-T artifacts under
`attacks/coco/mask_rcnn_swin_t/METHOD/default/`: `run.json`, `annotations.json`,
`manifest.jsonl`, and the referenced PNG files. The runner checks numerical source
identity, complete payload records, resolved parameter hashes, budgets, canonical
ground truth and image IDs, seeds, source checkpoint consistency and selected
payload bytes. Do not substitute historical results or missing payloads.

For an existing qualified main run, reuse it directly. Only when current payloads
are absent and a new run is authorized, prepare them via the main entry below.
The first command is inspection; the second executes generation and evaluation
for that source/victim selection. Choose different new output names if these
directories already exist. These are not commands to rerun an accepted matrix.

```bash
python experiments/main_transfer.py --plan-only --datasets coco --sources mask_rcnn_swin_t --targets faster_rcnn_r50 --output outputs/experiments/main_transfer/example-source-plan
python experiments/main_transfer.py --datasets coco --sources mask_rcnn_swin_t --targets faster_rcnn_r50 --devices cuda:0 --payload-retention keep_all --prediction-archive gzip --output outputs/experiments/main_transfer/example-source-main
```

## Inspect and Run

Inspect all 22 jobs without reading images, payloads or weights:

```bash
python experiments/training_state_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/training_state_transfer/example-plan
```

Expect `Status: planned; inference jobs=22; generation jobs=0`. Both report scopes
contain `NR` placeholders. This is a plan, not execution or numerical evidence.

After GPU qualification and input readiness are confirmed, use either command
below. They request the same 22 jobs (two states, each with clean plus ten methods).
Single-GPU mode serializes both states; dual-GPU mode assigns one complete state
to each owned worker. This is not DDP, changed training batches or source ensembling.

```bash
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-one
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-two
```

The default pair manifest is the registered relative location in this repository.
An optional anchor example is `--pair-repository artifacts/qualified-pair-repository`;
that directory must contain the manifest at its registered relative location and
all pair-referenced artifacts. It is an artifact-root anchor, not a different
training recipe. `--pair-manifest` can select a manifest inside that anchor.
Without `--pair-repository`, pair paths are anchored to this repository.

For a bounded diagnostic, select a current-main method and add `--max-images`:

```bash
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --methods numbod --max-images 10 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-diagnostic
```

`--max-images` must be an integer from 1 to 5000. Omit it for the full 5,000-image
scope; even `--max-images 5000` remains diagnostic. Method subsets are incomplete
panels. Duplicate/unknown methods and non-explicit, duplicate or more than two
devices are rejected. Default execution requires `--source-main`; unavailable
devices are rejected before creating the output directory.

## Outputs and Recovery

Each job saves all twelve raw COCO bbox AP/AR metrics and a hash-bound gzip
prediction archive. Full 5,000-image inference and the deterministic retained-500
projection use those same saved predictions: the projection makes zero extra
model calls and does not copy full-inference timing or memory fields. Diagnostics
project only the intersection with retained-500; an empty intersection is `NR`,
not numerical zero. Full scope and retained-500 are different claims.

- `plan.json`, `input_binding.json`, `records.json`: assigned jobs, verified inputs and unrounded records.
- `evaluations/STATE/METHOD/`: `metrics.json`, `predictions.json.gz`, `predictions_artifact.json`, and retained projection records/summaries.
- `reports/fullval/`, `reports/retained500/`: twelve CSV/TeX tables per scope; CSV uses four decimals. `ERR` means failure, `NR` means unrun/unavailable, and raw undefined metric `-1` displays as `NR` without changing JSON.
- `workers/SLOT/`: requests, process/GPU ownership receipts, logs, progress, completed-job files and terminal receipts.
- `summary.json`, `terminal.json`, `cleanup.json`, `execution_state.json`: completion, failure and owned-process containment evidence.

Execution succeeds only when all assigned full-scope records complete, nonempty
retained projections complete, worker terminals bind the exact request/owner/job
hashes, and owned-worker cleanup is verified. The resulting status is
`complete_pending_independent_acceptance`, not scientific acceptance. A full
default panel has 44 metric records (22 inference jobs, each with two scopes).

On failure, inspect coordinator/worker terminals and `worker.log`; preserve the
entire directory and completed prefix. `ERR` must not become zero. Fix missing or
changed inputs without overwriting qualified artifacts. Use a new output directory
for an explicitly chosen rerun; there is no automatic resume or merging of runs.
An existing output directory is deliberately refused. Optional prediction overlays
can be enabled with `--visualize-predictions`; they do not change evaluator predictions.

For separately creating a new controlled pair, see the [training-pair guide](current-training-pair-reproduction.md). This evaluation entry does not train the pair.
