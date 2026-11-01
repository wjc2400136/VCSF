# VCSF：COCO 与 Pascal VOC 主实验复现

[English](README.md) | [简体中文](README.zh-CN.md)

本项目包含 VCSF 实现及 COCO/Pascal VOC 目标检测对抗迁移基准，采用六个独立
源检测器、十六个目标检测器，以及包含 LGP 在内的注册比较方法。复现流程覆盖
环境与输入准备、主实验、评估和记录生成的报告。

公开实验范围为 COCO/VOC 主比较；补充研究代码和论文材料保留在 ODA 学院存档工程中。

项目自有公开源码/指南采用 **GPL-3.0-only**，保留第三方条款，见
[LICENSE](LICENSE)、[授权范围](docs/licensing/license-scope.zh-CN.md)、
[第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)。它不授权数据、权重、论文素材或
私有记录，也不产生科学验收。

## 从零开始

[环境指南](docs/environment.zh-CN.md)包含 Git/ZIP 获取、Conda、Linux/PowerShell、
固定依赖与恢复。GPU 以兼容 NVIDIA 驱动的 Linux 为维护目标。在源码根目录，
仅当 `oda` 不存在时执行：

```bash
conda env create -f environment.yml
```

然后激活，不重建或升级已有合格环境：

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

核心固定为 Python 3.8.20、PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118、
MMCV 2.0.1、MMEngine 0.7.4、MMDetection 3.0.0、MMYOLO 0.6.0，按
[完整锁定依赖](requirements/locked-cu118.txt)安装，不单独升级组件。
不需要可编辑安装或破坏性重建环境。

源码检查和计划不使用数据/模型/AP，不证明 GPU 运行合格；计划可能写新输出
叶目录，执行前要看完整任务集。双卡可用时选 `--devices cuda:0,cuda:1`，
单卡串行同一工作集，不引入 DDP、源集成、显存合并，不承诺固定两倍加速。

## 准备、执行与恢复

1. 按[COCO/VOC 数据指南](data/README.zh-CN.md)准备指定数据并深度验证。
2. 准备[COCO 权重及合格 VOC 检测器](checkpoints/README.zh-CN.md)。
3. 按[主实验指南](docs/current-public-reproduction.zh-CN.md)检查计划，按需要做
   极小诊断，再有意启动不带 `--max-images` 的完整人群。
4. 核对[不可变输出](outputs/README.zh-CN.md)及自动 CSV/TeX/图。
   [保存主实验报告导出](docs/saved-report-export.zh-CN.md)不重跑模型或 AP。

COCO 主实验使用全部 5,000 张 val2017；VOC 使用全部 4,952 张 VOC2007 test
及其二十类专用权重。默认按论文顺序比较十方法，已注册的可选主比较行仍可
选择。目标保持规范架构分组顺序，DINO 为留出目标，`BB Mean` 排除源匹配白盒目标。

## 公开入口

| `experiments/` 文件 | 职责 |
| --- | --- |
| `main_transfer.py` | COCO/VOC 主生成、干净/迁移评估和报告 |
| `clean_map.py` | 独立干净基线准备/检查 |
| `finetune_voc.py` | 注册的固定最终轮 VOC 检测器准备 |
| `verify_public_package.py` | 无模型完整源闭包检查 |
| `qualify_vcsf_a10_public.py` | CPU 上方法参数与工厂身份检查 |
| `reexport_saved_reports.py` | 从已保存记录导出新标量报告 |

`_common.py`、`visualization_settings.py` 是入口支撑文件，不是额外实验。
主实验可选的 NMS 后预测叠框保留。准备 CLI 限于 COCO/VOC，实验 CLI 仅接受
`main_transfer`；不再提供公开 `data minimize` 或旧数据集 `all` 选择器。

## 身份与证据

[完整源码清单](public-package.json)绑定当前文件集合。[身份指南](docs/public-package-identity.zh-CN.md)
解释冻结配置/来源和必要祖先辅助代码为何仍作为内部依赖保留，而不是额外
公开实验计划。九个数值核心文件、完整 VCSF 参数、规范模型/兼容注册表及六份
来源文档不变。原始字节哈希要求 LF；不能静默刷新清单来绕过失败。

[当前证据状态](current-evidence-status.json)将历史验收范围与本次收窄的软件
界面分开。未变化的有界主实验/设备证据不证明全部默认完整任务或论文数值
独立复现；发布源码不等于整个科学研究完成。

失败/中断后保留原产物并核对进程归属，明确缺失子集后写新叶目录，不覆盖、
不合并、不自动续跑。`NR` 是未运行，`ERR` 是失败，`--` 是审计后的结构
不适用，均不是零。不为验证源码/文档改动而重跑合格基线。
