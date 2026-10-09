# Current Preprocessing Reproduction

The separate [adaptive guide](current-adaptive-preprocessing-reproduction.md) covers defense-aware source-pipeline BPDA; it does not change this oblivious threat model.

[English](current-preprocessing-reproduction.md) | [简体中文](current-preprocessing-reproduction.zh-CN.md)

`experiments/preprocessing_transfer.py` evaluates preserved current-main COCO PNGs
under oblivious victim-side preprocessing. It generates no adversarial payloads,
does no training and requires no historical author-run acceptance files.

Actual single/dual-GPU diagnostics have completed for COCO Common-2, LGP/VCSF,
`identity` and `bit_depth_4`, three selected targets and one retained image.
Main qualified the original bounded pixel, prediction, twelve-metric and
task-ownership content; no attack payload was regenerated for this oblivious
route. Later report-only repairs have direct source review and saved export
checks, without overwriting the original outputs. This does not qualify the
complete retained-500 panel or independently accept its reports.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Prerequisites

Run commands from the repository root after `conda activate oda`. Follow the
[current reproduction guide](current-public-reproduction.md) for the pinned
environment, canonical COCO validation data and registered COCO checkpoints.
Do not independently upgrade OpenMMLab packages. Actual execution requires Linux
process-ownership support and one or two available explicit CUDA devices, with
sufficient memory on each device. Two devices do not pool memory. No GPU, image
or checkpoint is required for `--plan-only`.

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

## Scope and Inputs

The defaults use two independent sources in order, `faster_rcnn_r50` and
`mask_rcnn_swin_t`, all sixteen registered targets, and the ten current main
methods in their declared order: `numbod`, `hifa`, `mlfadv`, `tog`, `augtrans`,
`lgp`, `afog`, `osfd`, `sfim_b`, `vcsf`. Baselines retain their registered
compute-matched budgets; VCSF retains its observed-path budget and exact current
parameter identity. This experiment is COCO-only, not a VOC or BDD100K panel.

The eleven registered preprocessing variants are identity, JPEG qualities 90,
75 and 50, six- and four-bit quantization, Gaussian radii `1/2` and `1`, median
filter size 3, and bilinear resize round trips at `3/4` and `5/4`. They run before
the victim detector preprocessor. The attacker does not know the selected
victim-side preprocessing and does not query or differentiate through target
detectors during attack generation. This is not adaptive/BPDA evaluation.

The population is the fixed retained-500 selection from the complete,
numeric-ID-ordered 5,000-image COCO val2017 split. Its equal-bin-center positions
are `((2*i+1)*5000)//1000` for `i=0..499`; original full-split attack positions
and seeds are retained. `--max-images` takes the prefix of this fixed selection,
not the first images of val2017. Even `--max-images 500` is diagnostic. Omit the
flag for the full registered retained-500 scope; that scope is not full-5,000
evaluation or an independent dataset after configuration selection.

`--source-main` must point to a current `main_transfer` output with `plan.json`,
`provenance.json`, and preserved files under
`attacks/coco/SOURCE/METHOD/default/`: `run.json`, `annotations.json`,
`manifest.jsonl` and the referenced PNGs. The input check binds numerical source
identity, parameter hashes, comparison budgets, canonical ground truth, exact
IDs, original seeds, manifest bytes and the currently registered COCO source
checkpoint hash. Compared methods using the same foreign source checkpoint are
rejected. Target checkpoint paths and hashes are also bound. No historical AP
rows are inherited.

Reuse a qualified current-main output whenever available. Only if these payloads
are absent and new execution is authorized, prepare them through the main entry.
The first command inspects the selection; the second runs it. These commands are
not instructions to rerun an accepted matrix. Keep every generated PNG:

```bash
python experiments/main_transfer.py --plan-only --datasets coco --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50 --output outputs/experiments/main_transfer/preprocessing-source-plan
python experiments/main_transfer.py --datasets coco --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50 --devices cuda:0,cuda:1 --payload-retention keep_all --prediction-archive gzip --output outputs/experiments/main_transfer/preprocessing-source-main
```

A one-image main smoke does not cover even the first center-selected image.
For a diagnostic preprocessing selection, its source main prefix must contain
every requested retained ID. The full retained-500 mode requires all 5,000 main
payload images. Do not rename older runs or fabricate missing manifests.

## Inspect and Run

Inspect the full matrix without accessing input artifacts:

```bash
python experiments/preprocessing_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_oblivious_preprocessing_transfer/example-plan
```

Expect `Status: planned; inference jobs=3696; generation jobs=0`: 176 clean and
3520 attacked evaluations, organized into 231 complete preprocessing groups.
Reports contain `NR`, not measurements.

After device qualification and input readiness, choose one of these commands:

```bash
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-one
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-two
```

Both modes request the same scientific jobs and seed mapping. Single-GPU mode
serializes all groups; dual-GPU mode assigns independent whole groups in round
robin, keeping a complete target panel with its worker. It does not use DDP,
change batch sizes, ensemble sources or reduce the matrix. Use fresh output
names instead of executing both examples into an existing directory.

For a bounded diagnostic, select registered subsets and at most 500 retained IDs:

```bash
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --methods numbod --sources faster_rcnn_r50 --targets faster_rcnn_r50,dino_r50 --defenses identity,bit_depth_4 --max-images 1 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-diagnostic
```

Unknown or duplicate selections and unavailable explicit devices fail clearly.
Method/target subsets are not the complete comparison panel. Optional overlays
use `--visualize-predictions` and the visualization threshold/count flags; they
do not change evaluator predictions.

## Outputs and Recovery

- `plan.json`, `input_binding.json`, `records.json`: exact jobs, authenticated inputs and unrounded results.
- `scratch/GROUP/`, `view_manifests/GROUP.json`: preserved view annotations and manifests; nonidentity groups retain every transformed PNG. Identity groups reference the original images without copying them: canonical JPEGs for clean groups and preserved adversarial PNGs for attacked groups.
- `evaluations/GROUP/TARGET/`: all twelve raw COCO bbox AP/AR metrics, complete gzip predictions and their integrity artifacts. Official replay uses saved predictions, with no new model inference.
- `workers/SLOT/`: request, process/GPU ownership, log, progress, one full `view_GROUP.json` per group, group receipts, compact completed-job records and terminal evidence.
- `reports/DEFENSE/`: twelve TeX tables and `transfer_records.csv` with all twelve metrics at four decimal places. Reports use `ERR` for failures and `NR` for unavailable results; CSV also retains explicit status. Raw undefined `-1` displays as `NR` without modifying JSON.
- `summary.json`, `terminal.json`, `cleanup.json`, `execution_state.json`: overall completion/failure and owned-worker containment.

For each source/method row, BB Mean excludes the source-matched white-box target
and is computed only when every selected black-box target has a numeric value;
an incomplete black-box row receives no partial mean. Best/second-best rankings
require every metric cell in that source's entire selected method-by-target
panel. A complete individual row can therefore retain an unranked BB Mean while
another method is incomplete.

Completed records contain compact view metadata and a file/hash reference to the
group's full pixel bindings, not repeated image-binding arrays. Headers are
fully rehashed at worker start/end; prepared pixels are fully checked at group
start/end. Per-target checks use bound metadata/stat information and saved
prediction replay. Keep the full view files and every referenced PNG/prediction.

Success requires every selected evaluation, an exact owner/request/job-hash-bound
worker terminal, and verified process containment. The resulting status is
`complete_pending_independent_acceptance`, never automatic scientific acceptance.
Final result acceptance also needs its independent saved-result review.

On failure, preserve the original directory, worker logs, transformed images and
completed prefix. Inspect `terminal.json` and `workers/SLOT/worker.log`, resolve
the concrete cause, then use a new directory for an explicitly selected rerun.
There is no automatic resume, deletion or merge. Missing results are never zero;
an existing output directory is refused rather than overwritten.
