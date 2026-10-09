# COCO 受控检查点对训练

[English](current-training-pair-reproduction.md) | [简体中文](current-training-pair-reproduction.zh-CN.md)

在仓库根目录、固定的 `oda` 环境中执行命令。训练在 Linux 上使用明确指定的一张
或两张 CUDA 设备；每张卡都必须独立容纳完整的 Faster R-CNN 训练分支和优化器状态，
两张卡不会合并显存。先准备 COCO train2017、规范标注与数据集 manifest，以及登记的
Faster R-CNN R50 官方检查点。磁盘还需容纳两个候选权重、临时训练检查点、日志和资格记录。

此入口用于**训练** `standard_control` 和 `adversarial_training` 两个受害模型状态。
两者从同一初始检查点出发，使用 `configs/experiments/training_state_transfer.yaml`
登记的仅训练集配方。一卡依次运行两个完整分支；双卡各运行一个分支。设备数量不改变
batch、学习率、图像顺序、种子、数据增强策略或优化器更新次数，也不引入 DDP 或源集成。
不承诺 CUDA 位级相同或两倍加速。

如需对已经合格的检查点对做**评估**，请使用
[训练状态评估指南](current-training-state-reproduction.zh-CN.md)。该用途应复用现有合格
检查点；评估和接口支持检查均不要求重新训练检查点对。

## 先查看计划

```powershell
conda activate oda
python experiments/coco_training_state_pair.py --help
python experiments/coco_training_state_pair.py --plan-only --devices cuda:0 --max-images 47
python experiments/coco_training_state_pair.py --plan-only --devices cuda:0,cuda:1 --max-images 47
```

`--devices` 接收逗号分隔的设备列表，默认值为 `cuda:0,cuda:1`；只使用一卡时请明确指定
`cuda:0`。`--plan-only` 会写入新的计划、任务分配和摘要，不读取运行时图像载荷，也不使用
GPU。两个计划包含相同的逻辑训练任务，区别仅在设备分配。执行前检查终端输出的目录。

## 有界诊断

下面两条命令会实际执行登记的 47 张图像诊断。确认所需设备可用后，一次运行一种模式。

```powershell
python experiments/coco_training_state_pair.py --devices cuda:0 --max-images 47
python experiments/coco_training_state_pair.py --devices cuda:0,cuda:1 --max-images 47
```

登记顺序包含数据集索引 46 处的空 GT 图像。每个分支处理 47 张图像、24 个外层 minibatch
和 96 次优化器更新。一卡和双卡已在这一有界范围完成保存输出核验；这不等于全量训练或
全项目发布验收。诊断只导出候选：`diagnostic_pair_eligible` 应为真，
`strict_training_controlled_pair_eligible` 和 `publication_committed` 应为假。
不得将这些权重用作正式受害模型检查点对，也不得将其指标放入正式效能表。

## 全量检查点对训练

只有确实计划执行全部 118,287 张 COCO train2017 图像，并完成数据、检查点和资源检查后，
才省略 `--max-images`。
正式训练还要求具有可识别 commit 的干净 Git 工作树。初始检查点和发布路径须位于
项目内；入口会拒绝通过符号链接逃出项目的路径。

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
python experiments/coco_training_state_pair.py --devices cuda:0,cuda:1
```

改为 `--devices cuda:0` 即以串行方式执行相同的全量任务。训练不接入 COCO val2017，
不根据验证 AP 选择检查点，也不根据正式划分反馈提前停止；只有预先登记的最终 epoch
具备资格。

正式训练成功需要两个分支完整结束、失败记录为零、登记图像与更新次数完整覆盖，配对
trace 和配方身份一致，state dict 严格加载且有限，并通过登记训练图像的推理检查。
仅有 `.pth` 文件或进程正常退出并不足够。发布必须以不替换既有规范检查点对的方式，
提交合格的配对 manifest。

## 输出与恢复

默认输出位于 `outputs/training/coco_training_state_pair/` 下新的 UTC 目录。
检查 `plan.json`、`execution_assignments.json`、`execution_state.json`、
`provenance.json` 和 `summary.json`。`branches/` 内各分支保留启动与执行记录、日志、
`branch_manifest.json`、解析后的运行配置和 `candidate_model_only.pth`。
运行目录还保存 `candidate_pair_manifest.json` 和 `candidate_pair_qualification.json`。

规范正式检查点对位于 `checkpoints/coco/training_state_transfer/`。该发布目录已经存在时，
入口会拒绝替换。不要为让新运行发布而删除现有合格检查点对；先保留原证据，明确预期的
检查点谱系。

失败时保留目录和日志，先核对真实协调器与 worker 身份；连接中断不等于运行失败。
不可变运行目录不支持自动续跑、覆盖、重试或合并。确认并修复原因后，为另行计划的运行
使用新输出目录。正式受害模型检查点发生变化时，必须刷新受影响的干净与攻击评估。
