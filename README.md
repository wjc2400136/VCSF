# VCSF: COCO and Pascal VOC Main Reproduction

[English](README.md) | [简体中文](README.zh-CN.md)

This repository contains the VCSF implementation and the COCO/Pascal VOC
object-detection transfer benchmark. The experiment panel uses six independent
source detectors, sixteen target detectors and the registered comparison methods,
including LGP. The workflow covers environment and input preparation, main
experiments, evaluation and record-derived reports.

The public experiment scope is the COCO/VOC main comparison.

Project-owned public source/guides use **GPL-3.0-only**, with retained third-party
terms: [LICENSE](LICENSE), [licence scope](docs/licensing/license-scope.md),
[Third-Party Notices](THIRD_PARTY_NOTICES.md). This does not license datasets,
weights, paper assets or private records.

## Start From Zero

Follow [environment setup](docs/environment.md) for Git/ZIP acquisition, Conda,
Linux/PowerShell syntax, pinned dependencies and failure recovery. Linux with a
compatible NVIDIA driver is the maintained GPU target. From this source root,
only if `oda` is absent:

```bash
conda env create -f environment.yml
```

Then activate without recreating/upgrading an existing qualified environment:

```bash
conda activate oda
export PYTHONPATH="$PWD/src"
CUDA_VISIBLE_DEVICES="" python experiments/verify_public_package.py --plan-only
CUDA_VISIBLE_DEVICES="" python experiments/qualify_vcsf_a10_public.py --plan-only
python -m lgp models list
python -m lgp compatibility
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
```

The pinned core is Python 3.8.20, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118,
MMCV 2.0.1, MMEngine 0.7.4, MMDetection 3.0.0 and MMYOLO 0.6.0. Use the complete
[lock file](requirements/locked-cu118.txt); do not independently upgrade the stack.
No editable installation or destructive environment recreation is required.

Source checks and plans do not load models. Inspect the planned job set before execution.
For two available GPUs use `--devices cuda:0,cuda:1`; one GPU serializes the same
jobs. There is no DDP, source ensembling, pooled memory or promised twofold speedup.

## Prepare, Run and Recover

1. Prepare the named [COCO/VOC data](data/README.md) and deep-validate it.
2. Obtain [COCO checkpoints and qualified VOC detectors](checkpoints/README.md).
3. Follow the [main reproduction guide](docs/current-public-reproduction.md):
   inspect a plan, perform a tiny diagnostic when needed, then deliberately run
   the full population without `--max-images`.
4. Check [immutable outputs](outputs/README.md) and generated CSV/TeX/figures.
   [Saved main-report export](docs/saved-report-export.md) does not rerun models/AP.

COCO main uses all 5,000 val2017 images; VOC uses all 4,952 VOC2007 test images
and its own twenty-class detector checkpoints. The default comparison contains
ten methods in manuscript order; optional registered main rows remain selectable.
Targets retain canonical architecture-group order; DINO is held out and
`BB Mean` excludes the matched white-box target.

## Public Entry Points

| Entry in `experiments/` | Role |
| --- | --- |
| `main_transfer.py` | COCO/VOC main generation, clean/transfer evaluation and reports |
| `clean_map.py` | Separate clean baseline preparation/check |
| `finetune_voc.py` | Registered fixed-final VOC detector preparation |
| `verify_public_package.py` | Source-file integrity check, no models |
| `qualify_vcsf_a10_public.py` | CPU-only method configuration check |
| `reexport_saved_reports.py` | New scalar report export from saved records |

`_common.py` and `visualization_settings.py` are entry support files, not extra
studies. Optional main-run post-NMS overlays remain part of the main interface.
Generic preparation CLI commands are restricted to COCO/VOC; the experiment CLI
admits only `main_transfer`. No public `data minimize` or legacy `all` dataset
selector is provided.

## Outputs and Recovery

Preserve failures/partial outputs and verify process ownership after a disconnect.
Use a fresh immutable leaf for an explicit missing subset; never overwrite, merge
or assume automatic resume. `NR` is not run, `ERR` is failure and `--` is an
audited structural skip; none is zero.
