# COCO and VOC Main Reproduction

For current defense-aware source-pipeline BPDA with the same retained population, see the [adaptive preprocessing guide](current-adaptive-preprocessing-reproduction.md).

For offline fixed-three panels from your own complete current-main prediction
archives, see the [qualitative guide](current-qualitative-reproduction.md).

For fresh A10 and its nine unchanged final-background controls, including a fresh
anchor rather than author-only history, see the
[final-background guide](current-vcsf-background-reproduction.md).

[English](current-public-reproduction.md) | [简体中文](current-public-reproduction.zh-CN.md)

Run all commands from the repository root after activating the pinned `oda`
environment. Prepare and validate the dataset manifests and detector checkpoints
first. COCO uses 5,000 val2017 images and released COCO checkpoints. VOC uses
4,952 VOC2007 test images and the dataset-specific fine-tuned checkpoints; a COCO
head is not a substitute. BDD100K is outside the final maintained programme.

The current main interface has accepted bounded one/dual-GPU diagnostics:
four COCO/VOC cases (COCO image cap 6; VOC image cap 1), two sources,
LGP and VCSF, and three targets. The final-background interface has two accepted
COCO cases (image cap 1, ten controls, three targets), also in both modes.
This accepts only those diagnostic scopes, not project-wide release or formal
efficacy results. Plan-only and CPU checks do not establish GPU execution.

For the current COCO controlled training-state evaluation, reuse the preserved
Swin-T source payloads with the [training-state guide](current-training-state-reproduction.md).
That entry evaluates the registered victim pair without retraining or generating
new attack images.

For fixed oblivious COCO preprocessing, use the
[preprocessing guide](current-preprocessing-reproduction.md) with preserved
current-main Common-2 payloads. It evaluates the canonical retained-500 images
without regenerating attacks or inheriting historical numerical results.

For current A10 COCO radius sensitivity, use the
[five-radius guide](current-vcsf-radius-reproduction.md). The default fresh-user
workload covers both Common-2 sources and all five registered radii; the author
continues exact-identity reuse rather than rerunning accepted panels.

Main already includes the source-matched white-box rows. Reuse accepted rows
with matching current identities; no additional white-box run is required.
The [optional white-box diagnostic guide](current-whitebox-reproduction.md)
describes the separate fresh-user, diagonal-only entry, not extra scientific
evidence or an instruction to repeat accepted matrices.

## Inspect Before Execution

For current-input paired attack time and memory, use the
[cost guide](current-paired-cost-reproduction.md). That entry measures the current
A10 and corrected-LGP implementations on the fixed Common-2 retained population.
It computes no AP and does not repeat or inherit accepted author cost results.
Source jobs remain serial even with two devices to avoid concurrent timing contamination.

```powershell
conda activate oda
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

The default dataset is COCO. The default main comparison uses the ten methods
in the current manuscript order, six independent source detectors and all sixteen
targets. NAA and Corrupting Attention are explicitly selectable using `--methods`;
their audited structural skips remain `--`. The retired SVFTA method cannot be
selected through this current main entry.

The output plan contains the complete job set and worker assignments. Baselines
retain their original `compute_matched` schedules. VCSF uses its unchanged
selected A10 implementation, complete parameters and logical-update observed-cost
profile. Clean-reference rows and partial feature backwards are not free work or
calibrated whole-detector backward equivalents.

## Bounded Interface Checks

Run one mode at a time, only when the requested devices are available. These
commands execute; they are diagnostics, not full-dataset efficacy results.

```powershell
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

Both modes run the same complete source jobs, image population, seeds, parameters
and target evaluations. Dual-GPU mode assigns independent complete jobs; it does
not combine source gradients, use DDP or pool the devices' memory. Each GPU must
individually fit its assigned detector and attack. Explicitly unavailable devices
fail before output execution begins. Bitwise CUDA equality and a twofold speedup
are not promised.

## Full Reproduction

Do not omit `--max-images` until dataset and checkpoint validation succeeds and
the complete planned workload is intended.

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
python experiments/main_transfer.py --datasets coco --devices cuda:0,cuda:1
python experiments/main_transfer.py --datasets voc --devices cuda:0,cuda:1
```

Doctor selects only the named data and checkpoint panels. For COCO-only work,
use `--datasets coco`; deep VOC checkpoint validation still verifies its original
COCO source weights and manifest as part of the training provenance chain, not a
COCO dataset evaluation. BDD100K is not selected by the command above.

Use `--devices cuda:0` to serialize the same work on one GPU. Do not launch two
copies of a command to implement dual-GPU mode. New user runs bind their current
inputs and checkpoints and do not inherit the author's historical results.

## Outputs and Recovery

Each run is written to a new UTC leaf below `outputs/experiments/main_transfer/`.
Inspect `plan.json`, `execution_assignments.json`, `execution_state.json`,
`records.json` and `summary.json`. Generated PNGs are kept, predictions are
losslessly archived, and records generate the CSV, TeX and figures in `reports/`.
The [dispatch evidence guide](current-dispatch-evidence.md) explains durable
worker ownership, device context registration and terminal cleanup records.
VCSF attack directories additionally contain per-image `native_cost.jsonl`
observations; these are not a paired performance experiment or FLOP calibration.

For CPU-only reexport of saved scalar reports to a new output directory, use
the [saved-report export guide](saved-report-export.md) and its
[direct entry](../experiments/reexport_saved_reports.py). Immutable originals
are preserved; no models are loaded, predictions read or AP recomputed, and
the export does not constitute scientific acceptance.

Success requires the intended image coverage, complete target results, zero
failed records and the resulting saved-output checks. A successful process exit
alone is not independent scientific acceptance. A `--max-images` result must not
be inserted into a formal mAP table. Preserve all twelve AP/AR values in raw JSON.

When a command fails, preserve its directory and logs. Check the actual process
identity before deciding it stopped; a lost SSH connection does not imply failure.
Fix the identified cause, then use a new output leaf for an explicitly selected
missing subset. There is no automatic resume, overwrite or merge of immutable
run directories. Never replace missing or failed values with zero.
