# COCO 与 VOC 检测器检查点

[English](README.md) | [简体中文](README.zh-CN.md) | [项目首页](../README.zh-CN.md)

先按[环境](../README.zh-CN.md)与[数据](../data/README.zh-CN.md)指南准备，
所有命令从仓库根目录执行。[models.yaml](../configs/models.yaml)是唯一的
十六检测器注册表、规范顺序及训练配方来源。COCO 使用官方 80 类权重；
VOC 必须使用十六个各自微调的 20 类检测器，不能替换为 COCO head。

## COCO 官方权重

对于新建的检查点目录，在网络及磁盘空间满足要求后：

```bash
conda activate oda
python -m lgp weights download --help
python -m lgp weights download --models all
```

`--models all` 指十六个注册检测器，不是全部数据集。
权重按注册文件名保存至 `checkpoints/coco/`；
有序的 `checkpoints/coco/manifest.json` 保存大小、URL 与 SHA-256。
下载完成不等于严格加载或科学接受。
保留已有核验权重和 manifest；下载器可能替换无效的已有文件，
未经批准并验证备份／恢复范围，不得用它修复这些文件。

## VOC 微调候选接口

先准备完整 VOC07+12 trainval 与对应 COCO 源检查点。
以下命令只查看完整任务，不训练：

```bash
python experiments/finetune_voc.py --help
python experiments/finetune_voc.py --plan-only --models all --devices cuda:0
python experiments/finetune_voc.py --plan-only --models all --devices cuda:0,cuda:1
```

预期为 planned 状态、十六个规范模型任务，注册 epoch、micro-batch、
reference batch 不变。计划写入 `outputs/training/voc_finetuning/` 下的新 UTC 目录。
计划不证明权重就绪或设备可用。有界 VOC 单卡、双卡诊断已获接受：
两个独立模型（Faster R-CNN 与 Mask R-CNN Swin-T）、两个训练 ID、
合计四个 job、每个 job 完成十二个注册 epoch，共 72 个 optimizer step，
未附加正式验证集。这不构成完整 VOC 训练或十六检查点面板的资格。
设备调度核对另行按角色接受全部 37 条声明路径，最小实际路径缺口已为空：
10 条独立 GPU 路径、16 条共享调度器薄封装、六条 CPU-only 与五条支持路径。
这不证明全部默认研究或每种 CLI 选项的原生 GPU 执行，也不构成科学接受、
分发权利清除或全项目发布。

仅在实际设备／输入检查完成且确实决定新增训练后，选择下面**一种**模式。
已有核验检查点应保留，不默认重训；`--frozen-existing` 接收要保留的
已核验模型 ID（或 `all`），但不能替代其核验证据。

```bash
python experiments/finetune_voc.py --models all --devices cuda:0
python experiments/finetune_voc.py --models all --devices cuda:0,cuda:1
```

两条命令互为替代，不能并发重复启动。每张 GPU 执行独立完整模型任务，
不使用 DDP、显存池化，不改变 batch size 或学习率。
每张卡必须有足够显存；磁盘须满足 `--min-free-gib`
（默认每个模型启动前至少剩余 12 GiB，这不是总存储预算）。
`--max-images` 仅为诊断训练，不能产出规范正式权重。
该默认存储检查不是已完成有界诊断的待办阻塞；该诊断的限定准入
不改变此训练默认值，也不准入新的运行。

训练读取安装库的 upstream config，再运行时修补。
保留预声明最终 epoch，按注册 micro／reference batch 线性缩放学习率。
VOC2007 test 不得附加到训练 runner；不能按正式 AP 选择 epoch／检查点、
early stop 或重新调参。

## 就绪条件与恢复

VOC 规范文件为 `checkpoints/voc/MODEL_ID.pth`，
配套 `checkpoints/voc/manifest.json` 及训练／验证记录。
只有 `.pth` 文件**不算就绪**。必须具备规范数据 manifest、
源／最终 SHA-256、标注与解析配置哈希、Git 来源、固定最终 epoch、
严格 state-dict 加载及注册训练图像上的实际推理验证。
COCO 源权重溯源也须保留。

上述产物齐备后，在设备可用时执行限定数据集的预检：

```bash
python -m lgp doctor --datasets coco,voc --deep-data
```

需确认所选数据／检查点检查就绪，并处理警告；仅退出码为零不够。
仅运行 COCO 时使用 `--datasets coco`。
VOC 深度检查点验证也可能读取原 COCO 源 manifest／权重以核对训练溯源，
这不属于 COCO 效果评估。

Mask R-CNN Swin-T 在此只评估 bbox：只允许官方
`roi_head.mask_head.*` 键未使用；缺失 backbone／bbox 键或其他多余键均为错误。
检查点改变会使受影响的 clean／adversarial 对比需要重评。

失败时保留训练目录、日志与部分检查点。掉线后先确认进程所有权，
再对明确缺失模型使用新输出目录。不得覆盖核验产物、合并运行、
假定自动续跑，也不得在完整面板验证前删除 VOC 训练输入。
下一步见[主复现](../docs/current-public-reproduction.zh-CN.md)与
[输出保留](../outputs/README.zh-CN.md)。
