# VCSF COCO 与 Pascal VOC 复现源码

[English](README.md) | [简体中文](README.zh-CN.md)

这个轻量源码分发维护 COCO 与 Pascal VOC，保留最终 A10 的九个数值源码、
完整参数与实际 MRO、corrected-LGP source201、规范十六目标及六个独立源。
BDD100K 执行已经取消；其声明和退休 SVFTA 的必要依赖仅供内部兼容性读取，
不是维护实验项目。

项目自有源码及配套自有源码指南采用 **GPL-3.0-only**，见
[LICENSE](LICENSE)、[许可证范围](docs/licensing/license-scope.zh-CN.md)和
[第三方声明](THIRD_PARTY_NOTICES.zh-CN.md)。保留的上游条款仍然适用。
源码分发不产生新的数值接受，也不表示整个研究目标完成；原六份选型/provenance 文档保持原样。
[包身份指南](docs/public-package-identity.zh-CN.md)说明新的源码闭包身份，
不宣称与旧作者全树 freeze 等价。

## 前提与源码检查

从项目给出的 GitHub 页面选择 **Code > Download ZIP** 并解压，或使用已有 checkout。
在含有 `environment.yml`、`pyproject.toml`、`experiments/` 和 `src/` 的仓库根目录
打开终端。新机器需要 Conda、下载依赖/数据/权重的网络连接、足够磁盘空间和
兼容 CUDA 11.8 栈的 NVIDIA 驱动。

执行使用 Linux 与固定 Conda 环境 `oda`：Python 3.8.20、PyTorch 2.0.0+cu118、
torchvision 0.15.1+cu118、MMCV 2.0.1、MMEngine 0.7.4、MMDetection 3.0.0、
MMYOLO 0.6.0。其余依赖见[锁定清单](requirements/locked-cu118.txt)。
仅在 `oda` 尚不存在时创建环境：

```bash
conda env create -f environment.yml
```

已有固定 `oda` 时只需激活，不重建或独立升级组件。本直接入口流程使用
`PYTHONPATH`，不要求 editable 安装。禁止安装 `mmcv-full==1.7.2`、`mmcv-lite`
或 MMDetection 2.x，也不能独立升级 OpenMMLab 组件、添加会遮蔽已安装包的
顶层源码副本。安装失败时保留错误日志，核对固定 wheel 和驱动要求。
以下命令在仓库根目录运行，源码检查无需数据或权重：

```bash
conda activate oda
export PYTHONPATH="$PWD/src"
python -m lgp --help
CUDA_VISIBLE_DEVICES="" python experiments/verify_public_package.py --plan-only
CUDA_VISIBLE_DEVICES="" python experiments/qualify_vcsf_a10_public.py --plan-only
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

成功条件是源码检查退出码为零、完整 manifest 匹配、计划列出预期全工作集。
CPU 检查和计划不等于 GPU 执行或正式 mAP 接受。
`--devices cuda:0` 串行同一工作集；双卡调度独立完整任务，不使用 DDP 或显存池化。
所选设备须真实可用，每张卡分别容纳其任务。已有一/双卡接受仅覆盖指南说明的范围，
不构成全项目发布。

## 复现指南

- [数据准备与校验](data/README.zh-CN.md)、[检查点](checkpoints/README.zh-CN.md)、[输出](outputs/README.zh-CN.md)。
- [COCO/VOC 主实验](docs/current-public-reproduction.zh-CN.md)：默认十方法，NAA 和 Corrupting Attention 仍可选择。
- [固定预处理](docs/current-preprocessing-reproduction.zh-CN.md)、[自适应/BPDA](docs/current-adaptive-preprocessing-reproduction.zh-CN.md)。
- [训练状态评估](docs/current-training-state-reproduction.zh-CN.md)、另行[受控训练对训练](docs/current-training-pair-reproduction.zh-CN.md)。
- [最终背景控制](docs/current-vcsf-background-reproduction.zh-CN.md)、[A10 半径](docs/current-vcsf-radius-reproduction.zh-CN.md)、[配对成本](docs/current-paired-cost-reproduction.zh-CN.md)。
- [离线定性图](docs/current-qualitative-reproduction.zh-CN.md)、[在线预测可视化](docs/prediction-visualization-device-interface.zh-CN.md)。
- [可选白盒诊断](docs/current-whitebox-reproduction.zh-CN.md)、[已保存报告导出](docs/saved-report-export.zh-CN.md)、[调度证据](docs/current-dispatch-evidence.zh-CN.md)。

保留基线包装入口，以及 `all_method_ablations.py`、`coco_all_methods.py`、
`extended_transfer.py`、`naa_experiments.py`、`visualize_predictions.py`、
`voc_transfer.py`。有意执行前逐项检查 `--help` 与 `--plan-only`。
随包提供不恢复退休 SVFTA 协议，也不要求重跑已接受矩阵。
完整研究声明没有缩小；作者历史辅助模块不是新增 GPU 任务。
正式运行不使用 `--max-images`，限图结果只作诊断。

## 核验与恢复

执行前准备数据 manifest 及严格的检测器检查点来源证据。
VOC 必须使用本数据集检查点，不能借用 COCO 分类头。
每次运行写入新的不可变输出叶目录；按对应指南检查计划、状态、记录及自动
CSV/TeX/绘图输出。保留原始 JSON 精度和十二项指标。
保留失败及部分原件，不覆盖、不合并不可变运行，也不因失去连接重复启动。

源码守卫失败时停止依赖执行，将遗漏、篡改或多出的路径与 `public-package.json`
逐项比较。恢复干净解压包，或另建经审查的新身份，不可静默刷新哈希。
数据、权重、作者已接受结果和私有启动/审计历史不随包分发，也没有删除。
本候选不自动继承结果、不发布检查点、不向外部上传。

## 证据与发布状态

[当前证据状态](current-evidence-status.json)说明已接受的论文、corrected-LGP
半径整合及设备验证范围。论文仍为修订工作稿，私有 Overleaf 同步已接受；
本源码包不含论文照片副本或原始结果载荷。既有科学身份及历史清单 JSON 保留
原始快照边界，其中旧的仅文档分发字段不是当前许可证决定；当前分发应查阅
[许可证范围](docs/licensing/license-scope.zh-CN.md)与当前源码 manifest。
