# COCO 与 VOC 主比较

[English](current-public-reproduction.md) | [简体中文](current-public-reproduction.zh-CN.md)

先完成[环境](environment.zh-CN.md)、[数据验证](../data/README.zh-CN.md)及
[检查点准备](../checkpoints/README.zh-CN.md)，所有命令在本公开源码根执行。
本包只提供主比较，不提供作者本地补充研究计划。

COCO 使用全部 5,000 张 val2017 和官方 COCO 权重；VOC 使用全部 4,952 张
VOC2007 test 及十六个数据集专用二十类检测器，不能用 COCO head 代替。
目标/源及报告顺序来自规范注册表。

## 先检查计划

```bash
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

默认数据集为 COCO，十个论文默认方法、六个独立源、十六目标。NAA 和
Corrupting Attention 仍是可选主比较行，结构不适用记 `--`。不能选退役 SVFTA，
不开放消融。核对任务、人群、种子和设备分配。

基线保留冻结 compute-matched 设置。A10 保留一次 detector 加十九次 feature
更新、完整参数及实际参考/部分反向成本口径；参考工作不免费，部分特征梯度
不自动换算成完整检测器反向等价量。

## 极小诊断

在已授权 GPU 主机选择一种模式，不同时执行两种：

```bash
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
```

有意核对双卡时改为 `--devices cuda:0,cuda:1`。两种模式保留完整任务、ID、
种子、配置与评估；双卡是独立任务，不引入 DDP、集成或显存合并。显式设备
不可用应失败，每卡须容纳自己的工作量；不承诺 CUDA 字节相同或固定加速。
限图 mAP 不是正式结果。

历史已接受有界主实验包括四个 COCO/VOC 情形、两个源、LGP/VCSF 和三个目标
（COCO 上限六图，VOC 一图），不是完整人群资格。本次收窄仅做源码层核验，
不重跑该矩阵。

## 完整复现

有意启动全量前先执行：

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

看报告而不只看退出码。只做 COCO 时用 `--datasets coco`。VOC 溯源还会核对
原 COCO 来源权重，不等于另跑 COCO 效果实验。输入合格且确实要投入完整计算
后，选择一种设备模式执行所需数据集：

```bash
python experiments/main_transfer.py --datasets coco --devices cuda:0
python experiments/main_transfer.py --datasets voc --devices cuda:0
```

双卡将设备参数替换为 `--devices cuda:0,cuda:1`，不要重复启动两份命令充当
双卡。正式命令不带 `--max-images`。作者旧结果仅按原精确身份复用；新用户
输入不继承作者数值。

## 输出与恢复

新 UTC 叶目录位于 `outputs/experiments/main_transfer/`，含 `plan.json`、
`execution_assignments.json`、`execution_state.json`、`records.json`、
`summary.json`、攻击 PNG 和无损预测档案。记录自动生成 `reports/` 下的
CSV/TeX/图。可选主实验叠框使用 NMS 后框/类别/分数，不改变评估预测。
VCSF 的 `native_cost.jsonl` 是生成观测，不是单独配对计时实验。

要求预期覆盖、完整目标、无失败/缺失和保存产物检查，保留全部十二项 AP/AR
及未舍入 JSON，`BB Mean` 排除源。零退出不等于独立科学验收。
[标量导出](saved-report-export.zh-CN.md)不加载模型或重算 AP。

保留部分叶目录和日志。终端断连不证明 worker 停止；先查归属、修具体原因，
把明确缺失子集放新目录，不覆盖、不合并、不自动续跑、不把失败写零。
另见[输出](../outputs/README.zh-CN.md)和[调度记录](current-dispatch-evidence.zh-CN.md)。
