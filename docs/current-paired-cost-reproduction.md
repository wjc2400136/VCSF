# Current Paired Cost Reproduction

[English](current-paired-cost-reproduction.md) | [简体中文](current-paired-cost-reproduction.zh-CN.md)

This fresh-input entry measures attack API time and memory on the registered
COCO retained-500 population. It computes no AP, saves no adversarial image
payload, and neither reads nor inherits the author's private historical result
directories. It is not a GPU efficacy matrix or a request to rerun accepted AP
comparisons. See the [main reproduction guide](current-public-reproduction.md)
for the separate efficacy workflow.

Actual single/dual-GPU diagnostics have completed for Common-2 and LGP/VCSF,
with two retained images per source-method pair: eight measured calls and eight
warmup calls per mode. Both modes kept source measurements globally serial;
the dual mode observed both devices without concurrent measurement. Main
qualified the original numerical, input and sampled-ownership content. Later
report-only repairs have direct source review and saved export checks. This is
not qualification of the ten-method retained-500 panel or independent publishable
cost acceptance; declared caps remain distinct from calibrated physical work.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Prerequisites

Run every command from the repository root in the pinned Conda `oda`
environment. Prepare the canonical COCO validation manifest, all 5,000
val2017 inputs, and the registered COCO source checkpoints. Use the existing
dataset/checkpoint preparation workflow; do not supply historical AP records.

```powershell
conda activate oda
python experiments/current_paired_cost.py --help
python -m lgp doctor --help
```

Before actual measurement, verify the current environment, selected COCO
dataset and checkpoint panel:

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

Doctor must report no failed checks. Its COCO selection validates the registered
COCO panel, not merely these two cost sources. It is a preflight, not a GPU
execution or scientific acceptance receipt.

Choose one or two available explicit devices. Every chosen GPU must individually
fit its complete detector and attack; memory is not pooled. The measurement gate
requires **no foreign GPU compute process globally**, including on unselected
GPUs. Reservation of selected devices does not replace that global exclusivity
check. Do not launch a second measurement command or run AP jobs concurrently.

## Inspect the Plan

These commands do not load image or checkpoint payloads, instantiate attacks or
detectors, query CUDA availability, or invoke `nvidia-smi`. They inspect current
source/configuration identity and write a new plan directory with `NR` cost rows.
Plan-only output therefore does not prove that requested devices exist.

```powershell
python experiments/current_paired_cost.py --plan-only --devices cuda:0 --output outputs/experiments/current_paired_cost/plan-single-01
python experiments/current_paired_cost.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/plan-dual-01
```

The default order is `numbod,hifa,mlfadv,tog,augtrans,lgp,afog,osfd,sfim_b,vcsf`.
Common-2 sources are Faster R-CNN R50, then Mask R-CNN Swin-T (bbox-only).
An explicitly selected subset must contain an even number of unique current
methods and include `vcsf`; request order is normalized to the canonical order.
Unknown, duplicate or malformed methods/devices and invalid image limits fail.

Both device modes preserve the same parameters, budget profiles, source
checkpoints, selected image IDs and full-position seed mapping. Source jobs run
**serially in both modes**, with `concurrent_measurements=1`: dual mode assigns
the first source to the first device and the second source to the second device,
but does not overlap their measurements. This isolates timing, rather than
maximizing GPU throughput. It introduces no source ensembling or DDP.

## Tiny Diagnostic and Full Measurement

Run one mode at a time, after preflight and global exclusivity checks. The tiny
commands execute two retained images per source-method pair and are not
publishable cost or efficacy results:

```powershell
python experiments/current_paired_cost.py --methods lgp,vcsf --max-images 2 --devices cuda:0 --output outputs/experiments/current_paired_cost/tiny-single-01
python experiments/current_paired_cost.py --methods lgp,vcsf --max-images 2 --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/tiny-dual-01
```

Default actual execution omits `--plan-only`, `--max-images` and `--methods`;
it measures the full registered 500-image, ten-method, two-source cost panel:

```powershell
python experiments/current_paired_cost.py --devices cuda:0 --output outputs/experiments/current_paired_cost/full-single-01
python experiments/current_paired_cost.py --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/full-dual-01
```

These are alternative execution modes, not instructions to run both. The default
panel has 10,000 measured calls and 100 separate warmup calls: five warmup images
per source-method pair. Warmup uses the first five selected images and is not
included in measured summaries. With a smaller image limit, warmup uses
`min(5, selected_images)`. Any method subset or image limit, including
`--max-images 500`, is diagnostic-only.

Selection uses the registered equal-bin centers in the sorted 5,000-image
population. Each call uses `seed = 42 + original_full_population_position`,
not the retained-subset ordinal. The parameter hashes, VCSF selected A10
identity, original baseline schedules and method-specific budget profiles stay
bound to the current declarations. The retained-500 cost measurement is **not a
5,000-image efficacy result** and cannot populate an AP table.

## Records and Cost Boundaries

Each output leaf must be new. Omitting `--output` creates a new UTC leaf under
`outputs/experiments/current_paired_cost/`. Input/output overlap with resolved
code, configuration, dataset and checkpoint paths is rejected, including
checkpoint symlink targets and protected ancestors.

Inspect `plan.json`, `input_binding.json`, `environment.json`,
`coordinator.json`, `execution_state.json`, source-local `warmup.jsonl` and
`measurements.jsonl`, and aggregate `measurements.json`, `summary.json`,
`terminal.json`, `artifact_manifest.json`, `cost_by_source_method.csv` and
`cost_by_source_method.tex`. Input bindings preserve annotation, image and
source checkpoint hashes; current source and public identity are rechecked.

| Field | Meaning and boundary |
| --- | --- |
| `wall_seconds` | Synchronized attack API wall time, including reference work; excludes decode, input preload and external validation. |
| `peak_allocated_mib` | Peak allocated memory during the call, including resident model/input allocations; not incremental attack-only memory. |
| `base_allocated_mib` | Allocation immediately before the timed call. |
| `peak_reserved_mib_shared_allocator` | Shared warm-cache allocator peak; not an independent per-method allocation requirement. |
| `declared_logical_gradients` | Registered logical gradient cap, not physical kernels or calibrated full-detector backward equivalents. |
| `actual_logical_gradients_from_method_diagnostics` | Method-reported count when available; identity-output count or declared-cap fallback otherwise. Check `logical_gradient_count_source`. |
| `actual_source_row_forwards_from_method_diagnostics` / `actual_auxiliary_forward_from_method_diagnostics` | Method-reported counts when supported; unavailable counters remain null. |
| `declared_auxiliary_forward` / `declared_auxiliary_backward` | Declarations, not a substitute for observed physical work. |
| `mean_paired_ratio_to_vcsf` | Mean same-source, same-image time ratio to VCSF; not a ratio of rounded summary means. |

`logical_gradient_count_source` distinguishes `method_diagnostics`,
`identity_output` and `declared_cap`; `counter_scope` explicitly excludes
physical kernel accounting. Clean references, partial backwards and transformed
views are not free work or automatically whole-detector equivalents. Recorded
logical caps cannot establish equal physical computation. The allocator is not
emptied between measured calls; Williams ordering balances method position and
within-image carryover. Warmup/load/validation time is not attack API time.

Raw JSON retains unrounded values. Missing statistics are null in JSON and
`NR` in CSV/TeX, never zero. Aggregate CSV float values use at least four decimal
places; TeX displays time and ratios to three, memory to two. Rankings and paired
ratios must use raw observations, not display-rounded values.

A complete default panel is `complete_pending_independent_cost_audit`; a tiny
or subset panel is `passed_cost_diagnostic_not_publishable`. Neither is
independent acceptance. `formal_AP_eligible`, `cost_report_eligible`,
`scientific_acceptance`, `project_wide_device_release_accepted` and
`historical_result_inheritance` remain false where recorded.
`evaluation_calls=0` and `payload_files=0` are actual scope counters, not
placeholders for missing measurements.

## Failure Recovery

Preserve the failed directory, `failure.json`, traceback and any source-local
measurement prefix. There is no automatic resume, overwrite or directory merge.
Fix the reported cause and choose a new output leaf; change the example suffix
for every new attempt. Verify actual process ownership before retrying: a lost
connection is not proof that measurement stopped. Missing observations or a
partial source prefix cannot be accepted as a completed cost panel.
