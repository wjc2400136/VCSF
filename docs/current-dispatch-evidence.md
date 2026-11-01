# Main Dispatch Evidence

[English](current-dispatch-evidence.md) | [简体中文](current-dispatch-evidence.zh-CN.md)

Use pinned `oda` from the public source root. Current commands and input checks
are in the [main guide](current-public-reproduction.md). The public surface is
COCO/VOC main and necessary preparation only; earlier broader path inventories
are historical, not additional shipped experiment entries.

## Inspect the Same Jobs

```bash
python experiments/main_transfer.py --plan-only --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/main_transfer.py --plan-only --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

These are no-model plans, not GPU diagnostics. New output leaves contain planned
jobs, assignments and reports; metrics stay NR and observed N remains unknown.
Device selection preserves jobs, inputs, seeds, settings and report order.
Two GPUs do not pool memory. Explicit unavailable devices fail actual admission.

## Actual Execution Records

On Linux dual-GPU execution, `dispatch_journal/STAGE/` contains flushed coordinator
and worker journals. They bind PID/start ticks/parent, assigned device/tasks and
the existing CUDA context to logical device UUID and observed NVML PID.
They do not create an extra context, change RNG or prove exclusive GPU use.
Missing early observations remain pending/null; a nonempty worker cannot finish
without its required registration. Final coordinator records retain exit/cleanup
evidence after the live worker list is cleared.

These records distinguish planned jobs, actual owned execution, completed outputs
and cleanup. They do not establish efficacy or independent scientific acceptance.
Do not reconstruct missing registrations or infer observed identity from a plan.

Optional integrated main prediction overlays retain evaluator predictions.
Their bounded historical component/layout checks are not a certificate for every
crowded image or a separate public qualitative experiment.

## Failure and Evidence Limits

Preserve the full leaf, original terminal/state, journals and completed prefix.
A stale running snapshot or lost connection does not authorise a duplicate job.
The run owner must reconcile actual terminal/cleanup evidence before an explicit
missing subset enters a fresh leaf. Do not overwrite, merge or auto-resume.

Source review, CPU mocks and plans qualify only those checks. Historical bounded
main evidence remains bounded; this source-scope cleanup does not claim a new
full-GPU matrix, every CUDA branch or project-wide scientific acceptance.
