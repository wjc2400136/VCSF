# Controlled COCO Checkpoint-Pair Training

[English](current-training-pair-reproduction.md) | [简体中文](current-training-pair-reproduction.zh-CN.md)

Run commands from the repository root in the pinned `oda` environment. Training
executes on Linux with one or two explicitly selected CUDA devices. Each device
must fit the complete Faster R-CNN training branch, including optimizer state;
two devices do not pool memory. Prepare COCO train2017, its canonical annotation
and dataset manifest, and the registered released Faster R-CNN R50 checkpoint.
Keep sufficient storage for both candidate weights, temporary training
checkpoints, logs and qualification records.

This entry **trains** two victim states, `standard_control` and
`adversarial_training`. Both use the same initial checkpoint and registered
train-only recipe in `configs/experiments/training_state_transfer.yaml`.
One device serializes these complete branches; two devices schedule one branch
on each device. Device count does not change the batch, learning rate, image
order, seed, augmentation policy or number of optimizer updates. This is not
DDP or source ensembling. Bitwise equality and a twofold speedup are not promised.

For **evaluation** against an already qualified victim pair, use the
[training-state evaluation guide](current-training-state-reproduction.md).
Existing qualified checkpoints should be reused for that purpose; neither
evaluation nor a portability checklist requires retraining the pair.

## Inspect The Plan

```powershell
conda activate oda
python experiments/coco_training_state_pair.py --help
python experiments/coco_training_state_pair.py --plan-only --devices cuda:0 --max-images 47
python experiments/coco_training_state_pair.py --plan-only --devices cuda:0,cuda:1 --max-images 47
```

`--devices` takes a comma-separated list and defaults to `cuda:0,cuda:1`.
Specify `cuda:0` when only one GPU is intended. `--plan-only` writes a fresh plan,
assignment manifest and summary without reading runtime image payloads or using
a GPU. The logical training work is the same in both plans; device placement
differs. Inspect the printed output directory before executing.

## Bounded Diagnostic

These two commands execute the registered 47-image diagnostic. Run one mode at
a time when its devices are available.

```powershell
python experiments/coco_training_state_pair.py --devices cuda:0 --max-images 47
python experiments/coco_training_state_pair.py --devices cuda:0,cuda:1 --max-images 47
```

The registered order includes the empty-ground-truth image at dataset index 46.
Each branch consumes 47 images, 24 outer minibatches and 96 optimizer updates.
Both modes have completed saved-output validation at this bounded scope. This
does not establish full-dataset training or project-wide release readiness.
The diagnostic exports candidates only: `diagnostic_pair_eligible` must be true,
`strict_training_controlled_pair_eligible` and `publication_committed` must be
false. Do not use its weights as a formal victim pair or insert its metrics
into a formal efficacy table.

## Full Pair Training

Omit `--max-images` only when the full 118,287-image COCO train2017 workload is
intended and data, checkpoint and resource checks have passed.
Formal training additionally requires a clean Git worktree with an identifiable
commit. Keep the initial checkpoint and publication paths inside the project;
the entry rejects paths that escape the project through symlinks.

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
python experiments/coco_training_state_pair.py --devices cuda:0,cuda:1
```

Use `--devices cuda:0` for the same full job set in serial mode. Training does
not attach COCO val2017, choose a checkpoint from validation AP or stop early
from formal-split feedback. Only the predeclared final epoch is eligible.

Successful formal training requires two complete branches, zero failed records,
the full registered image and update coverage, matching paired trace and recipe
identities, strict finite state-dict loading, and registered training-image
inference verification. A `.pth` file or a successful process exit is not enough.
Publication must commit the qualified pair manifest without replacing an
existing canonical pair.

## Outputs And Recovery

The default run directory is a new UTC leaf under
`outputs/training/coco_training_state_pair/`. Inspect `plan.json`,
`execution_assignments.json`, `execution_state.json`, `provenance.json` and
`summary.json`. Each `branches/` child retains its launch and execution records,
log, `branch_manifest.json`, resolved runtime config and
`candidate_model_only.pth`. The run also records `candidate_pair_manifest.json`
and `candidate_pair_qualification.json`.

The canonical formal pair lives under
`checkpoints/coco/training_state_transfer/`. If that publication directory
already exists, the entry refuses replacement. Do not delete a qualified pair
to make a new run publish; preserve the existing evidence and resolve the
intended checkpoint lineage first.

Keep failed runs and logs. Confirm the actual coordinator and worker identities
before interpreting a lost connection as failure. There is no automatic resume,
overwrite, retry or merge of immutable run directories. After correcting a
confirmed cause, use a fresh output directory for a separately intended run.
Changing a formal victim checkpoint requires the affected clean and attacked
evaluations to be refreshed.
