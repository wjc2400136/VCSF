# COCO and VOC Data

[English](README.md) | [简体中文](README.zh-CN.md) | [Project](../README.md)

Run commands from the repository root in the pinned `oda` environment.
The dataset registries are [coco.yaml](../configs/datasets/coco.yaml) and
[voc.yaml](../configs/datasets/voc.yaml), loaded by
[manager.py](../src/lgp/data/manager.py). There is no combined `configs/datasets.yaml`.

| Dataset/Split | Images | Role and Location |
| --- | --- | --- |
| COCO `val` | 5,000 | Formal val2017; `data/coco/val2017/` |
| COCO `dev` | 1,000 | Frozen train2017 selection, development only; `data/coco/dev2017/` |
| VOC `val` | 4,952 | Formal VOC2007 test; `data/VOCdevkit/VOC2007/JPEGImages/` |
| VOC `train` | 16,551 | VOC07+12 trainval, detector fine-tuning; `data/VOCdevkit/` |

Do not select checkpoints on formal splits. Any authorized retrospective COCO
full-val selection must be labelled as selection, not independent confirmation.

## Download, Extract and Prepare

These commands download/write data; use network access and sufficient disk space.
For existing qualified data, reuse its verified manifest rather than overwrite it.
Always name the dataset: omitted/default `all` also selects legacy datasets.

```bash
conda activate oda
python -m lgp data download --dataset coco
python -m lgp data extract --dataset coco
python -m lgp data prepare --dataset coco
python -m lgp data download --dataset voc
python -m lgp data extract --dataset voc
python -m lgp data prepare --dataset voc
```

COCO default download includes val2017 and train/val annotations, not the full
train2017 image archive. Preparation creates the frozen dev annotation, selection
manifest and selected train-origin images; it can download missing dev images.
VOC downloads all three registered archives: 2007 trainval, 2007 test and 2012
trainval. Preparation writes `data/VOCdevkit/annotations/voc0712_trainval.json`
and `voc2007_test.json`. Experimental readers use prepared annotations/images,
never `data/raw/`; raw archives are temporary preparation inputs.

COCO `download --full` additionally downloads optional train payloads, and
`validate --full` requires optional splits. They are not needed for the default
main reproduction. Preserve any train payload required by a separately registered
training workflow; do not download or train a victim pair just to inspect an interface.

## Validate Before Actual Work

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
```

Check the reports, not only the exit code: expected annotation counts, class
counts, referenced-image existence, duplicate/escaped paths and train/val overlap
must pass; inspect inventory warnings. A `valid_with_warnings` result needs review,
not an automatic claim of complete readiness. Preparation alone is not deep validation.
Before formal execution, also complete the dataset-scoped doctor and checkpoint
checks described in the [checkpoint guide](../checkpoints/README.md).

On failure, retain the report and repair the exact missing/corrupt archive,
annotation or referenced image using its registered source. Do not redirect
experiments to raw archives or duplicate extracted trees. Confirm hashes and
rerun only the affected data checks; never replace missing evidence with zero.

## Storage and Deletion Boundary

Keep one referenced runtime image tree per dataset; do not manually create
duplicate archive trees or unreferenced image copies. Retain VOC training inputs
until all sixteen required VOC detector checkpoints are trained and verified.

Optional storage-plan inspection is **not part of the COCO/VOC setup**:

```bash
python -m lgp data minimize
```

Without `--apply` it does not delete, but the current implementation calls COCO
dev preparation and scans every registered dataset, including legacy ones.
It has no dataset selector and is not strictly read-only; use it only when all
registered dependencies are available and those preparation effects are acceptable.
Do not use it to bypass a COCO/VOC-only data boundary. Any deletion, including
adding `--apply`, requires explicit approval, verified targets, a backup/recovery
decision and protected training/output scope. Do not remove data based on this guide alone.

Continue with [checkpoints](../checkpoints/README.md),
[main reproduction](../docs/current-public-reproduction.md) and
[outputs](../outputs/README.md).
