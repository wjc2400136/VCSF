# Current Dispatch Evidence

[English](current-dispatch-evidence.md) | [简体中文](current-dispatch-evidence.zh-CN.md)

Use the pinned `oda` environment from the repository root. The complete commands,
input prerequisites and detector memory requirements remain in the
[radius guide](current-vcsf-radius-reproduction.md) and
[training-state guide](current-training-state-reproduction.md).
Device selection does not change the scientific job set. Two GPUs do not pool
their memory, and explicitly requested unavailable devices are errors.

To inspect the same bounded radius job set without data, checkpoints or CUDA:

```bash
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0 --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --epsilons 4/255 --max-images 1 --output outputs/experiments/current_vcsf_radius/evidence-plan-one
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0,cuda:1 --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --epsilons 4/255 --max-images 1 --output outputs/experiments/current_vcsf_radius/evidence-plan-two
```

Each new output must contain four planned attack cells and twelve source-aware
TeX reports under `reports/radius/4_255/`, plus `transfer_records.csv`.
Every report carries the actual dataset/split, observed sample N and requested
image cap. A cap is not evidence of evaluated N: unavailable N remains unknown,
and unrun metrics remain `NR`. The dedicated radius caller does not also create
the generic source-collapsed ablation, analysis or heatmap projections. Other
callers retain their existing default projections.

During generic dual-GPU execution on Linux, `dispatch_journal/STAGE/` holds an
append-only, flushed coordinator journal and one worker journal per slot.
These bind PID/start ticks/parent, assigned device, complete assigned tasks and
the worker's existing CUDA context to its logical-device UUID and observed NVML
PID. They do not allocate an extra context, change RNG, reserve extra resources,
or treat CUDA initialization alone as context evidence. Early missing NVML
observations remain pending/null; the first observed own PID registers once.
A nonempty worker may not complete without registration. These journals do not
claim exclusive GPU use. The existing selected-owner protocol is separate.
The final coordinator journal records worker exit and cleanup; clearing the live
`workers` list does not erase this evidence. These journals are execution evidence,
not efficacy evidence or independent acceptance. Original missing registrations
are not reconstructed or relabelled as passed.

For training-state execution, the final `execution_state.json` follows the status
of `terminal.json` and binds its SHA-256 and, when available, `cleanup.json`.
Success remains `complete_pending_independent_acceptance`. Failure preserves the
first exception and separately reports any cleanup or report-export error.
The terminal receipt is authoritative; the state file is a snapshot, not a resume
instruction. Plan-only output has no actual worker or context registration.

On failure, preserve the entire output, original terminal, journals and completed
prefix. Do not fill missing identities with inferred values or restart from a
stale running snapshot. Have the run owner reconcile terminal and cleanup evidence;
any authorized missing work must use a new output leaf. CPU mock tests or plan-only
inspection do not establish actual GPU qualification or project-wide release.

## Native Visualization Scope

The independently accepted native online-visualization diagnostic comprises six CLI cases, twelve complete target outputs/PNGs and 144 unrounded metric values. Paired one/dual predictions and PNG bytes matched exactly; simultaneous dual-panel NVML observations showed distinct workers on distinct GPUs. One VOC single-case NVML window was unsampled; continuous ownership is not claimed. This accepts native execution only, not publication layout quality, the fixed-three paper figures or global release.

Separately, the accepted caption-layout component was checked on six offline redraws from those saved predictions: all selected boxes and predictions were retained, title and caption rectangles stayed within bounds and did not overlap, and all six images were directly reviewed. Its exact four source/test/guide files are integrated here without repeating GPU execution. This component acceptance does not replace the three fixed manuscript figures or certify every possible crowded image.
