# 从零创建环境

[English](environment.md) | [简体中文](environment.zh-CN.md) | [项目首页](../README.zh-CN.md)

本页准备公开 COCO/VOC 环境，不另造方法、不下载数据、不启动训练。GPU 以
兼容 NVIDIA 驱动的 Linux 为维护目标；Windows 可以检查源码/计划，提供
PowerShell 写法不等于完整 Windows GPU 验证。每张卡独立容纳检测器，不合并显存。

## 获取并定位源码

先安装 Git、Miniconda/Anaconda，打开新终端。新工作副本的命令为：

```bash
git clone https://github.com/wjc2400136/VCSF.git
cd VCSF
git status --short
conda --version
```

也可在仓库页选择 **Code > Download ZIP** 后解压。源码根应有
`environment.yml`、`pyproject.toml`、`src/`、`configs/`、`experiments/`。
本地学院版进入已有 `reproduction/`，不要重新克隆或覆盖外层研究树。

## 只创建一次

```bash
conda env list
```

仅当 `oda` 不存在时，在源码根执行：

```bash
conda env create -f environment.yml
```

它安装 [Conda 声明](../environment.yml)和[固定依赖](../requirements/locked-cu118.txt)，
包括 CUDA 与 OpenMMLab wheel 索引，需要网络。核心固定为 Python 3.8.20、
PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118、MMCV 2.0.1、MMEngine 0.7.4、
MMDetection 3.0.0、MMYOLO 0.6.0。NVIDIA 驱动须支持 CUDA 11.8；数据、权重、
攻击图像和预测档案另需规划磁盘空间。

已有合格 `oda` 只激活，不重建、不裁剪、不自动更新；仅同名不代表版本合格。
不要装 `mmcv-full==1.7.2`、`mmcv-lite`、MMDetection 2.x，不单独升级 OpenMMLab，
不加遮蔽已安装包的顶层同名源码。不要照抄历史指南中带 `-Recreate` 的破坏性
bootstrap，它会删除指定环境并附带安装/下载。当前直接入口不需要可编辑安装。

## 无模型源码检查

Linux，在源码根执行：

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

PowerShell 的源码选择和检查写法：

```powershell
conda activate oda
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:CUDA_VISIBLE_DEVICES = ''
python experiments/verify_public_package.py --plan-only
python experiments/qualify_vcsf_a10_public.py --plan-only
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
```

以上仅进行 CPU 检查。运行实验前另开终端并配置正确的 GPU 可见性，不要修改
活动任务的设备映射。

## 输入、GPU 检查与恢复

依次阅读[数据](../data/README.zh-CN.md)、[检查点](../checkpoints/README.zh-CN.md)
和[主实验](current-public-reproduction.zh-CN.md)。显式选择 `coco`/`voc`，
公开 CLI 拒绝旧 `all` 和 BDD100K 选择。在 GPU 主机执行：

```bash
nvidia-smi
python -c "import torch; from mmcv.ops import nms; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m lgp doctor --datasets coco,voc --deep-data
```

阅读 doctor 具体报告，处理严重错误并审阅警告。计划和限图诊断不是正式 mAP，
正式运行不得带 `--max-images`。

保留失败命令和日志，修复具体 Python、wheel 或驱动不匹配，不升级
科学依赖栈。创建中断后先检查环境，不自动删除已有 `oda`。激活不可用时按
实际 shell 使用 `conda init bash` 或 `conda init powershell`，再重开终端。
源文件检查失败时，重新获取干净工作副本或 ZIP，不要跳过检查。
修复数据或权重后，重新执行相应验证命令，再启动实验。
