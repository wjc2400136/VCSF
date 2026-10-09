# Current Adaptive Preprocessing Reproduction

[English](current-adaptive-preprocessing-reproduction.md) | [简体中文](current-adaptive-preprocessing-reproduction.zh-CN.md)

`experiments/adaptive_preprocessing_transfer.py` runs the current COCO Common-2
retained-500 defense-aware source-pipeline BPDA comparison. It uses current user
inputs, not author-only acceptance files or numerical results.

Actual single/dual-GPU diagnostics have completed for COCO Common-2, LGP/VCSF,
`identity` and `bit_depth_4`, three selected targets and one retained image,
including six source-pipeline generation jobs per mode under the registered
identity-reuse policy. Main qualified the original bounded pixel, prediction,
twelve-metric and task-ownership content. Later report-only repairs have direct
source review and saved export checks, without overwriting the original outputs.
This does not qualify the complete retained-500 panel or independently accept
its reports.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Prerequisites and Identity

Run from the repository root in the pinned `oda` environment. Follow the
[current main guide](current-public-reproduction.md) for COCO data, released
checkpoints, and preserved current-main source PNGs. This entry is COCO-only;
VOC and BDD100K are not part of this preprocessing experiment.

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
python experiments/adaptive_preprocessing_transfer.py --help
```

Require valid data and an `ok` doctor/checkpoint panel before GPU execution.
Do not upgrade the unified environment. Execution requires Linux process
ownership support and one or two explicit available CUDA devices. Each device
must fit its complete assigned source/target jobs; memory is not pooled.

`--source-main` names your own current `main_transfer` output. It binds exact
method implementations, parameters, budgets, source checkpoints, canonical
annotations, retained IDs and full-validation seed positions. Its base seed must
be the registered 42. It needs the selected current methods' preserved source
PNG, manifest and run metadata, plus `plan.json` and `provenance.json`. Full mode
requires full 5,000-image source coverage. A diagnostic source prefix must cover
every selected retained ID, not merely the same number of initial main images.
See the [oblivious guide](current-preprocessing-reproduction.md) for source-main
preparation and the fixed equal-bin-center population. Reuse valid existing
payloads; these instructions do not authorize repeating an accepted author run.

## Fixed Scope and BPDA

Defaults retain all ten current main methods, both independent sources
(`faster_rcnn_r50`, `mask_rcnn_swin_t`), eleven registered defenses and the
canonical sixteen targets. Method and target order are unchanged. The fixed 500
IDs are center positions in the numeric-ID-ordered full COCO validation split;
`--max-images` selects their prefix, not the beginning of val2017. Any explicit
limit, including 500, is diagnostic. Omit it for the registered retained-500
scope, which is not a full-5,000 or independent-selection evaluation.

Attack generation starts from the original clean image and differentiates
through the known preprocessing and the source detector only. Target models are
not queried or differentiated during generation. Epsilon remains relative to
the original clean image, not the transformed view. The deployed forward and
fixed derivative rules come from `preprocessing_defenses.yaml`: identity uses
exact autograd; JPEG, bit depth and median use identity BPDA; Gaussian uses the
registered normalized Gaussian derivative; resize uses bilinear differentiation.
The forward path remains the exact registered deployment transform. No extra
EOT, result-dependent surrogate or new method objective is introduced.

Each attack seed remains 42 plus its original full-validation position. Baselines
retain `compute_matched`; VCSF retains selected A10 and
`vcsf_final_background_observed_cost`. Its reference rows and partial backward
paths are observed separately, not treated as free work or calibrated whole-
detector equivalents. The registered A10 identity branch reuses current-main
pixels; corrected-LGP and the other baseline identity branches regenerate as
already declared. No old payload is renamed or altered to match this policy.

The full fresh-user matrix has 218 generated source/method/defense payloads,
two reused A10 identity payloads, 176 clean and 3,520 attacked target evaluations.
That is 3,696 evaluation jobs in 231 complete target-panel groups. These are
reproduction workload counts, not a list of additional author experiments.

## Inspect and Execute

Inspect without reading source payloads, data or checkpoints, and without GPUs:

```bash
python experiments/adaptive_preprocessing_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_adaptive_preprocessing_transfer/example-plan
```

Expect `Status: planned; inference jobs=3696; generation jobs=218`. Planned reports
show `NR`. When device qualification and input readiness are satisfied, choose
one execution mode with a new output name:

```bash
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-one
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-two
```

Single-GPU mode serializes the same groups. Dual-GPU mode schedules independent
complete generation-and-target-panel groups without source ensembling, DDP,
changed budgets or omitted jobs. Do not launch two coordinators to implement
dual-GPU mode. Bitwise CUDA identity and an exact twofold speedup are not promised.

For a bounded interface diagnostic with already sufficient source-main coverage:

```bash
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --methods numbod,vcsf --sources faster_rcnn_r50 --targets faster_rcnn_r50,dino_r50 --defenses identity,bit_depth_4 --max-images 1 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-diagnostic
```

## Preserved Outputs and Recovery

Inspect `plan.json`, `input_binding.json`, `records.json`, `summary.json`,
`terminal.json`, `execution_state.json`, and `cleanup.json`. Worker request hashes,
process births, GPU registrations, full group views and terminal receipts remain
under `workers/SLOT/`. Progress distinguishes adaptive generation and target
evaluation. Generated raw PNGs, annotations, manifests and native A10 cost
observations stay under `adaptive_attacks/GROUP/`; nothing is pruned.

`scratch/GROUP/` and `view_manifests/GROUP.json` bind the deployed target views.
Identity views reference the original clean, generated or reused pixels without
copying them. All nonidentity view PNGs remain. `evaluations/GROUP/TARGET/`
preserves all twelve raw metrics and complete lossless gzip predictions. Saved
prediction replay uses the official evaluator without new model inference.

`reports/DEFENSE/` contains twelve TeX tables and four-decimal CSV records. BB Mean
excludes the source-matched target and requires the whole selected black-box row;
rankings require the complete selected comparison panel. Failures are `ERR`,
missing/unrun values are `NR`, and raw undefined `-1` displays as `NR`, never zero.
The reports are adaptive/BPDA results, not oblivious or certified robustness.

A successful producer ends `complete_pending_independent_acceptance`; it is not
automatic result acceptance. Keep the full pixels, predictions and receipts for
independent saved-result review. On failure, retain logs and the complete prefix,
check the actual process identity, resolve the concrete cause, and use a new
output for an explicitly selected missing subset. There is no automatic resume,
overwrite, deletion or merge. Lost connectivity does not prove a worker stopped.
