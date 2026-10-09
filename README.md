# VCSF COCO and Pascal VOC Reproduction

[English](README.md) | [简体中文](README.zh-CN.md)

This lightweight source distribution maintains COCO and Pascal VOC. It preserves
the selected A10 numerical implementation and complete registered parameters,
corrected-LGP source201, sixteen canonical targets and six independent sources.
BDD100K execution is cancelled; its declaration and retired SVFTA dependencies
remain internal compatibility inputs, not maintained experiment programmes.

Project-owned source and its accompanying project-owned source guides use **GPL-3.0-only**;
see [LICENSE](LICENSE), [licence scope](docs/licensing/license-scope.md), and
[Third-Party Notices](THIRD_PARTY_NOTICES.md). Retained upstream terms remain in force.
This source distribution does not create numerical acceptance or complete the research goal.
The original six selection/provenance documents are unchanged.
The [package identity guide](docs/public-package-identity.md) explains the new
closure identity; it does not claim equivalence to the old whole-author-tree freeze.

## Prerequisites and Source Check

Get the project repository from its supplied GitHub page using **Code > Download ZIP**,
then extract it, or use an existing checkout. Open a terminal in the repository root,
containing `environment.yml`, `pyproject.toml`, `experiments/` and `src/`.
A fresh machine needs Conda, network access for dependencies/data/weights, sufficient
disk space and a compatible NVIDIA driver for the CUDA 11.8 stack.

Use Linux and the pinned Conda environment named `oda`: Python 3.8.20,
PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, MMCV 2.0.1, MMEngine 0.7.4,
MMDetection 3.0.0 and MMYOLO 0.6.0. The exact additional dependencies are in
[the lock file](requirements/locked-cu118.txt). Create the environment only if `oda`
does not already exist:

```bash
conda env create -f environment.yml
```

For an existing pinned `oda`, activate it without recreating or independently
upgrading packages. This direct-entry setup uses `PYTHONPATH` and does not require
an editable package installation. Do not install `mmcv-full==1.7.2`, `mmcv-lite`
or MMDetection 2.x, or independently upgrade OpenMMLab components. Do not add
top-level source copies that shadow the installed packages. On setup failure,
retain the error log and resolve the pinned wheel/driver requirement.
Run the following from the repository root. The source check needs no datasets or weights:

```bash
conda activate oda
export PYTHONPATH="$PWD/src"
python -m lgp --help
CUDA_VISIBLE_DEVICES="" python experiments/verify_public_package.py --plan-only
CUDA_VISIBLE_DEVICES="" python experiments/qualify_vcsf_a10_public.py --plan-only
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

Success means the source checks exit zero, the complete source manifest matches
and the plan lists the intended full jobs. CPU checks and plans are not GPU or
formal-mAP acceptance. Use `--devices cuda:0` for the same work on one GPU;
dual mode schedules independent complete jobs without DDP or pooled memory.
Every selected device must be available and individually fit its workload.
Accepted one/dual evidence covers only the documented scopes, not global release.

## Reproduction Guides

- [Data preparation and validation](data/README.md), [checkpoints](checkpoints/README.md), [outputs](outputs/README.md).
- [COCO/VOC main](docs/current-public-reproduction.md): ten default methods; NAA and Corrupting Attention remain selectable.
- [Fixed preprocessing](docs/current-preprocessing-reproduction.md) and [adaptive/BPDA preprocessing](docs/current-adaptive-preprocessing-reproduction.md).
- [Training-state evaluation](docs/current-training-state-reproduction.md) and distinct [controlled-pair training](docs/current-training-pair-reproduction.md).
- [Final-background controls](docs/current-vcsf-background-reproduction.md), [A10 radii](docs/current-vcsf-radius-reproduction.md), [paired cost](docs/current-paired-cost-reproduction.md).
- [Offline qualitative panels](docs/current-qualitative-reproduction.md), [online prediction visualization](docs/prediction-visualization-device-interface.md).
- [Optional white-box diagnostic](docs/current-whitebox-reproduction.md), [saved report export](docs/saved-report-export.md), [dispatch evidence](docs/current-dispatch-evidence.md).

Retained baseline wrappers, `all_method_ablations.py`, `coco_all_methods.py`,
`extended_transfer.py`, `naa_experiments.py`, `visualize_predictions.py`
and `voc_transfer.py` are shipped. Inspect each entry with `--help` and
`--plan-only` before intentionally executing it. Shipping an entry does not
reactivate retired SVFTA protocols or require rerunning accepted matrices.
The full study declarations remain unchanged; author-history support modules
are not new GPU tasks. Formal runs omit `--max-images`; bounded results are diagnostics.

## Verification and Recovery

Prepare data manifests and strict detector checkpoint provenance before execution.
VOC needs its dataset-specific checkpoints, not a COCO classification head.
A new run writes an immutable output leaf; follow the corresponding guide for
its plan, state, records and automatic CSV/TeX/plot outputs. Keep raw JSON precision
and all twelve metrics. Preserve failures and partial originals; do not overwrite,
merge immutable runs or treat loss of visibility as permission to relaunch.

If the source check fails, stop dependent execution and compare the exact missing,
tampered or extra path with `public-package.json`. Restore a clean extracted
source distribution or create an explicitly reviewed new identity; do not silently refresh
hashes. Data, weights, accepted author results and private launch/audit history
are not distributed here and have not been deleted. No automatic result inheritance,
checkpoint publication or external upload is performed.

## Evidence and Publication Status

[Current evidence status](current-evidence-status.json) describes the accepted
manuscript, corrected-LGP radius integration and device scopes. The manuscript
remains a revised working draft, and its private Overleaf synchronization is
accepted; no paper/photo copy or raw result payload is shipped here. Earlier
scientific identity and historical inventory JSON files retain their original
snapshot boundaries. Their old documentation-only distribution fields are not
the current licence decision; use [the licence scope](docs/licensing/license-scope.md)
and the current source manifest for this distribution.
