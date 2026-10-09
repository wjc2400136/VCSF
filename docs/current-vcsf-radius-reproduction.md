# Current A10 VCSF Five-Radius Reproduction

[English](current-vcsf-radius-reproduction.md) | [简体中文](current-vcsf-radius-reproduction.zh-CN.md)

This guide is for a new GitHub user generating the current selected A10 VCSF
results from their own current COCO inputs and registered checkpoints. The
default workload covers all five registered radii; it does not inherit the
author's numerical results or their acceptance. Use the
[main reproduction guide](current-public-reproduction.md) for shared preparation.

The [entry](../experiments/current_vcsf_radius.py) and
[runner](../src/lgp/runners/current_radius.py) have completed bounded CPU
qualification in the pinned `oda` environment: CLI help, single/dual-device
`--plan-only`, and execution-guard/report contracts have passed their applicable
checks.

Actual single/dual-GPU diagnostics cover COCO Common-2 at `4/255`, the two
source-matched targets and one image per group. Main qualified their original
pixel, prediction and numerical content. Original dual registration remains
`NR`; a separate later dual diagnostic has accepted journal/registration evidence
for this limited scope, without backfilling the old records. Report-only repairs
have direct source review and saved compile/render receipts, not full
project release or scientific acceptance. These diagnostics do not qualify the
full-population, all-five-radius workload or certify the commands below as
accepted efficacy results.

Main has separately accepted the reviewed source-package increment and saved CPU
plans. One/two-device dispatch is accepted at the declared maintained-path roles;
no minimum actual device paths remain open in that reconciliation. This is not
native GPU qualification of every default study or every CLI option. Redistribution
clearance and full project release remain pending Main's decision; no publication
or external upload is authorized. CPU checks and limited GPU diagnostics are not
formal mAP or scientific acceptance.

## Scientific Scope

Only the perturbation radius changes. The selected A10 implementation,
checkpoints, image population, source-specific configuration, seed policy and
evaluator remain fixed. This is a radius sensitivity reproduction, not a method
search, target-query optimization or independent confirmation of method selection.

| Setting | Fixed scope |
| --- | --- |
| Dataset | COCO, all 5,000 val2017 images per source-radius group. |
| Common-2 sources | `faster_rcnn_r50`, then `mask_rcnn_swin_t`; two independent attacks, not an ensemble. Swin-T is bbox-only. |
| Radii, in registered order | `2/255`, `4/255`, `8/255`, `16/255`, `32/255`. |
| Step size | `1/255` at every radius; it is not scaled with epsilon. |
| Iterations | 20: one detector-initialization gradient step plus 19 feature-gradient steps. |
| Seed | Fixed base seed 42, with the existing per-image position offset; no seed-search option. |
| Targets | The canonical sixteen detectors, including the source-matched white-box target and held-out DINO. |

The default plan has ten independent source-radius groups and 160
adversarial source-radius-target evaluations, plus sixteen clean target
evaluations shared across the radii. Each group generates attacks on the same
5,000-image population and evaluates all sixteen targets. `32/255` is a
high-distortion stress test, not a routine perturbation budget. No radius may
silently alter the objective, optimizer, detector initialization, feature steps
or model preprocessing. Logical update counts are not calibrated complete-detector
backward equivalents; reference forwards and partial backwards are not free work.

This public entry is separate from the historical eleven-method
`experiments/coco_multi_budget_linf.py`; do not use that entry as a substitute.
For the author, continue exact-identity reuse at `4/255` and separately authorized
missing shards. This guide neither authorizes duplication nor requests a rerun
of any accepted author matrix.

## Prepare the Repository

Open a terminal in the repository root, containing `environment.yml`,
`pyproject.toml`, `experiments/` and `src/`. All commands below assume that root.
Reuse the pinned `oda` setup in the [root guide](../README.md), the
[COCO data guide](../data/README.md), the
[checkpoint guide](../checkpoints/README.md), and the
[main guide](current-public-reproduction.md). Do not recreate an existing correct
`oda` environment or upgrade its pinned Python, PyTorch or OpenMMLab components.
This workflow needs the canonical COCO validation manifest, all val2017 images
and all sixteen registered COCO detector checkpoints; no VOC or BDD100K inputs
are needed. Allow disk space for all generated PNGs and complete prediction
archives, without assuming pruning.

The entry does not download missing data or weights. A missing or invalid input
fails execution; complete the linked preparation workflow before retrying.

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

Proceed only after deep data validation and doctor report no failed checks.
They establish input/environment readiness, not GPU execution or independent
scientific acceptance. Each selected GPU must individually fit its assigned
detector and attack; two cards do not pool memory.

## Inspect the Work

Inspect the actual installed interface and choose the plan for your device mode.
The help and single/dual-device plan-only interfaces below have completed bounded
CPU qualification; they do not demonstrate real GPU execution:

```bash
python experiments/current_vcsf_radius.py --help
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0,cuda:1
```

Plan-only is a no-GPU workload inspection, not an execution or availability
check. Verify both mode plans retain the same sources, radii, image IDs, fixed
seed policy, checkpoints, attack settings and target set. Inspect the saved plan
and assignments; do not infer GPU readiness from them.

| Option | Meaning |
| --- | --- |
| `--help` | Show the entry's actual installed interface. |
| `--plan-only` | Inspect the work without running GPU experiments. |
| `--max-images` | Integer from 1 to 5,000, always diagnostic; omit for full-split results. |
| `--sources` | Comma-separated subset of the two Common-2 source IDs only. |
| `--targets` | Comma-separated subset of the sixteen canonical target IDs only. |
| `--epsilons` | Comma-separated subset of the five registered fractions only. |
| `--devices` | One or two explicit devices; default `cuda:0`. |
| `--device` | Strict single-device legacy compatibility, such as `cuda:0`; comma-separated device strings are rejected. Use it instead of `--devices`. |
| `--output` | A new run leaf; never an existing or accepted result directory. |
| `--visualize-predictions` | Optional post-NMS box, class-label and confidence overlays; evaluator predictions must stay unchanged. |

The defaults are both sources, all sixteen targets, all five radii and base seed
42. Selected lists must be nonempty, unique registered subsets and are normalized
to registry order, not command-line order. Use exact epsilon labels such as
`2/255,4/255`; decimals and alternative spellings are not registered selectors.
Each epsilon maps to its single registered `current_vcsf_radius_2_255` study or
the corresponding other-radius study, not to the historical multi-method entry.
`--devices` and `--device` are mutually exclusive. For two devices, use
`--devices cuda:0,cuda:1`; `--device` accepts only one device. There is no method
selector, seed override, target-query search, source ensembling or DDP.

## Diagnostic and Full Reproduction

After prerequisites, a tiny diagnostic can select one radius and two targets:

```bash
python experiments/current_vcsf_radius.py --epsilons 4/255 --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
```

This keeps both independent sources but does not run the complete radius/target
panel. Any `--max-images` result is diagnostic, not formal AP or mAP, regardless
of exit status. A subset of radii, sources or targets is not the complete default
reproduction and is explicitly marked `diagnostic_only`, even without an image
limit. This also applies to `--max-images 5000`.

When the complete intended workload is ready, choose exactly one device mode:

```bash
python experiments/current_vcsf_radius.py --devices cuda:0
```

Alternatively, on two individually capable and available GPUs:

```bash
python experiments/current_vcsf_radius.py --devices cuda:0,cuda:1
```

These are alternatives, not instructions to run both. With no options, the entry
executes all five radii on `cuda:0`; execution is the default.
Single-card mode serializes the same complete groups. Dual-card mode schedules
independent complete source-radius groups, each with its full target selection;
it must not change images, seeds, budget, checkpoint identity or evaluation.
Do not start duplicate commands to simulate dual-card scheduling. Neither bitwise
CUDA equality nor a twofold speedup is promised. An explicitly unavailable device
must fail clearly, not trigger an undisclosed fallback. The limited diagnostics
above do not establish full-population, all-five-radius qualification in either
mode.

## Outputs and Success Conditions

Without `--output`, each invocation creates a new UTC leaf under
`outputs/experiments/current_vcsf_radius/`. For an explicit destination, use a new
relative leaf, for example `--output outputs/experiments/current_vcsf_radius/my-run-01`,
and change the suffix for every new attempt. See the
[output guide](../outputs/README.md) for retention rules.

Inspect `plan.json`, `radius_plan.json`, `records.json`, `summary.json`,
`execution_state.json`, `execution_assignments.json` and `radius_summary.json`.
The wrapper's `radius_plan.json` preserves radius selection and resolved parameter
hashes; `plan.json` and the assignment/state files describe shared execution.
`radius_summary.json` retains each epsilon's cells, source-excluded means and
stress designation. Its `historical_result_inheritance`, `scientific_acceptance`
and `project_wide_device_release_accepted` flags remain false.

Reports are under `reports/radius/2_255/`, `4_255/`, `8_255/`, `16_255/` and
`32_255/`, for the selected radii. Each directory contains one
`transfer_records.csv` with all twelve metric columns and twelve metric-specific
TeX files, for example `transfer_bbox_mAP.tex` and `transfer_bbox_AR_100.tex`.
Plan-only output carries unrun `NR` values; it is not numerical evidence.

Preserve every generated attack PNG and each complete lossless gzip prediction
archive. Optional overlays do not replace these artifacts or alter evaluator
predictions. Their current settings are confidence threshold 0.50, at most three
images per evaluation and at most 100 displayed detections per image.
The CPU checks cover interface, execution-guard and report contracts; the separate
GPU diagnostics cover only the limited scope stated above, not full-run completion.

The wrapper rejects output destinations equal to, inside or ancestral to the
resolved `src/`, `configs/`, `checkpoints/` or COCO data root. The shared runner
rejects nonempty output leaves. Always choose a new leaf under `outputs/`.

Reports must be generated from unrounded JSON, not manually entered numbers.
Each epsilon must retain all twelve standard COCO summaries in CSV and TeX:
AP, AP50, AP75, AP-small, AP-medium, AP-large, AR-at-1, AR-at-10, AR-at-100,
AR-small, AR-medium and AR-large. Keep the following canonical target order and
architecture headers even when input records arrive out of order:

| Architecture group | Targets in order |
| --- | --- |
| Two-stage | `faster_rcnn_r50`, `cascade_rcnn_r50`, `mask_rcnn_swin_t` |
| YOLO family | `yolov3_d53`, `yolov5_s`, `yolox_s`, `yolov8_s` |
| Dense/point-based | `retinanet_free_anchor_r50`, `reppoints_r50`, `vfnet_r50`, `tood_r50`, `rtmdet_s` |
| Query/set-based | `sparse_rcnn_r50`, `detr_r50`, `deformable_detr_r50`, `dino_r50` |

DINO's `held-out` header role is not an independent selection holdout or a claim
of exclusion from offline configuration selection: DINO did participate in the
current offline selection, while each attack generation still uses only the
designated source and makes no queries to transfer targets.

The TeX header uses `\dagger` for registered white-box source roles and
`\ddagger` for held-out DINO. An asterisk marks the current source-matched
white-box result cell; it is distinct from the header's source-role marker.
For the full panel, `BB Mean` excludes the source-matched white-box target and
uses the remaining fifteen targets. A target subset must disclose its actual
denominator, not present itself as the full-panel `BB Mean`. Raw JSON retains
unrounded values on the official 0-to-1 metric scale. CSV uses that same scale
with four decimal places; the radius supplemental reproduction TeX displays
percentage points with four decimal places. Author-approved manuscript display
precision is applied separately in the manuscript projection, not to these
supplemental reproduction outputs. Display rounding must not change raw JSON,
ranking or means. Unrun, failed and audited structurally
undefined results are `NR`, `ERR` and `--`, respectively, never artificial zeros.

Complete default reproduction requires all ten groups to cover all 5,000 image
IDs, all 160 adversarial and sixteen clean target evaluations to be present,
no failed/missing records, retained
payloads and complete radius reports consistent with the saved plan and current
input/configuration identities. Plan-only or limited-image output does not meet
these conditions. Exit code zero is not independent saved-result audit,
scientific acceptance or project-wide device release acceptance. Reusing the
COCO split for sensitivity analysis does not restore dataset independence.

## Failure Recovery

Preserve the failed leaf, logs, records and partial PNG/prediction evidence.
There is no automatic resume, overwrite or merging of immutable run directories.
A lost terminal/SSH connection does not prove the worker stopped; ask the run
owner to confirm its state before retrying. Resolve the actual environment,
input, device or reported execution error without changing the method or radius
contract.

The wrapper writes the radius-specific plan, summary and reports after the
shared runner returns. A failure can therefore leave shared execution evidence
without final radius-specific artifacts; preserve that evidence rather than
assuming nothing ran. If radius report projection itself fails, preserve
`radius_failure.json` and its reported stage/reason as well. That failure record
does not confer scientific or device-release acceptance.

After the owner confirms recovery is appropriate, select only the explicitly
missing source/radius/target subset into a new leaf using supported selectors.
An arbitrary missing image shard cannot be expressed by inventing a flag or
using `--max-images` as resume; coordinate that case with the main run owner.
Keep original provenance separate and never promote a partial record to a
complete result. For the author, retain exact `4/255` reuse and the authorized
missing-shard scope rather than rerunning the accepted panel.
