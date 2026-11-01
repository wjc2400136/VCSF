# COCO and VOC Data

[English](README.md) | [简体中文](README.zh-CN.md) | [Project](../README.md)

Run from the public source root in pinned `oda` after
[environment setup](../docs/environment.md). Dataset declarations remain in
[coco.yaml](../configs/datasets/coco.yaml) and [voc.yaml](../configs/datasets/voc.yaml).

| Split | Images | Role / Runtime Path |
| --- | --- | --- |
| COCO `val` | 5,000 | Formal main, `data/coco/val2017/` |
| COCO `dev` | 1,000 | Frozen development selection, `data/coco/dev2017/`; not formal main |
| VOC `val` | 4,952 | VOC2007 test, `data/VOCdevkit/VOC2007/JPEGImages/` |
| VOC `train` | 16,551 | VOC07+12 trainval, VOC detector preparation |

## Download, Extract, Prepare

These commands download/write files. For existing qualified inputs, preserve
their manifests and repair only explicitly identified missing/corrupt items;
do not overwrite a qualified tree. On a fresh host with network/storage:

```bash
python -m lgp data download --dataset coco
python -m lgp data extract --dataset coco
python -m lgp data prepare --dataset coco
python -m lgp data download --dataset voc
python -m lgp data extract --dataset voc
python -m lgp data prepare --dataset voc
```

COCO defaults include val2017 and train/val annotations, not all train2017 images.
Preparation writes the frozen dev selection/annotation and can download missing
selected development images. VOC uses the registered 2007 trainval, 2007 test and
2012 trainval archives, and prepares `voc0712_trainval.json`/`voc2007_test.json`
under `data/VOCdevkit/annotations/`. Experiments never read `data/raw/`.

Do not download optional `--full` payloads for the main comparison. BDD100K and
the old `all` dataset selector are not exposed by this public CLI.
In the local institutional project, use its explicit resource plan to reuse the
outer data without duplication; do not change the public registry's root.

## Deep Validation

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
```

Inspect expected counts/classes, referenced-image existence, duplicate/escaped
paths and train/val overlap. Review warnings even if status is
`valid_with_warnings`; preparation or a zero exit alone is not readiness.
Then complete the selected [checkpoint/doctor checks](../checkpoints/README.md).

On failure retain the report and repair the exact registered input. Do not point
experiments at archives or duplicate extracted trees. Formal splits cannot select
checkpoints or hyperparameters. Retrospective full-val method selection, where
historically declared, is not independent confirmation.

## Retention

Keep one referenced runtime image tree. Retain VOC training data until all sixteen
required detectors are trained and qualified. There is no public `data minimize`
command; do not use a legacy minimizer that scans retired datasets.
No data deletion is authorised by this guide. Further deletion needs explicit
scope, verified targets and a backup/recovery decision.

Continue with [checkpoints](../checkpoints/README.md),
[main](../docs/current-public-reproduction.md) and [outputs](../outputs/README.md).
