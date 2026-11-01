# Environment From Zero

[English](environment.md) | [简体中文](environment.zh-CN.md) | [Project](../README.md)

This prepares the public COCO/VOC runtime, not a different method, dataset or
training job. Linux with a compatible NVIDIA driver is the maintained GPU target.
Windows can inspect source/plans; PowerShell syntax is not proof of full Windows
GPU qualification. Each GPU must individually fit its detector; memory is not pooled.

## Obtain and Locate the Source

Install Git and Miniconda/Anaconda, then open a fresh terminal. For a new checkout:

```bash
git clone https://github.com/wjc2400136/VCSF.git
cd VCSF
git status --short
conda --version
```

Alternatively choose **Code > Download ZIP** on the repository page and extract it.
The source root contains `environment.yml`, `pyproject.toml`, `src/`, `configs/`
and `experiments/`. In a local institutional ODA project, enter its existing
`reproduction/` instead of cloning or overwriting the outer tree.

## Create Once

```bash
conda env list
```

Only if `oda` is absent, run from this source root:

```bash
conda env create -f environment.yml
```

This installs [the Conda declaration](../environment.yml) and
[locked dependencies](../requirements/locked-cu118.txt), including the CUDA and
OpenMMLab wheel indexes; network access is required. The fixed core is Python
3.8.20, PyTorch 2.0.0+cu118, torchvision 0.15.1+cu118, MMCV 2.0.1, MMEngine 0.7.4,
MMDetection 3.0.0 and MMYOLO 0.6.0. The NVIDIA driver must support CUDA 11.8.
Data, weights, attack images and prediction archives need additional disk space.

For an existing qualified `oda`, activate it without recreation, pruning or
automatic updates. The name alone does not qualify an incompatible environment.
Do not install `mmcv-full==1.7.2`, `mmcv-lite`, MMDetection 2.x, independently
upgrade OpenMMLab components, or add top-level package copies that shadow them.
Do not copy destructive `-Recreate` bootstrap commands from historical guides:
those remove named environments and have extra installation/download side effects.
This direct-entry workflow does not require an editable installation.

## Source-Only Checks

Linux, from the source root:

```bash
conda activate oda
export PYTHONPATH="$PWD/src"
python -m pip check
python -c "import sys, importlib.metadata as m; print(sys.version); print({n: m.version(n) for n in ('torch','torchvision','mmcv','mmengine','mmdet','mmyolo')})"
CUDA_VISIBLE_DEVICES="" python experiments/verify_public_package.py --plan-only
CUDA_VISIBLE_DEVICES="" python experiments/qualify_vcsf_a10_public.py --plan-only
python -m lgp models list
python -m lgp compatibility
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
```

PowerShell source-selection/check equivalents:

```powershell
conda activate oda
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = ''
python experiments/verify_public_package.py --plan-only
python experiments/qualify_vcsf_a10_public.py --plan-only
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
```

The empty device visibility is intentional for inspection. Open a fresh correctly
configured GPU shell before actual execution; never change an active job's mapping.
Success means zero exit, matching source bytes/versions and the intended plan,
not dataset/checkpoint readiness or numerical acceptance. `.gitattributes` keeps
LF text in new checkouts; unexplained line-ending/hash failures must not be fixed
by silently repinning the manifest.

## Inputs, GPU Checks and Recovery

Follow [data](../data/README.md), [checkpoints](../checkpoints/README.md), then
[main reproduction](current-public-reproduction.md). Select `coco`/`voc` explicitly;
the public CLI rejects legacy `all` and BDD100K selections. On the authorized GPU host:

```bash
nvidia-smi
python -c "import torch; from mmcv.ops import nms; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m lgp doctor --datasets coco,voc --deep-data
```

Read the doctor report, resolve critical errors and review warnings. Reuse
qualified weights/results rather than routinely retraining or rerunning them.
Plan-only and bounded diagnostics are not formal mAP; formal runs omit `--max-images`.

Keep failed commands/logs and source identity. Resolve the exact Python, wheel
or driver mismatch without upgrading the scientific stack. Inspect a partially
created environment; never automatically delete a pre-existing `oda`. If activation
is unavailable, use `conda init bash` or `conda init powershell` for the actual
shell and reopen it. For source identity failures, use a clean checkout/extraction
or a reviewed new identity, not a bypass. Data/weight repairs have separate
write/provenance boundaries. Delivering this guide does not claim installation.
