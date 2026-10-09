# 当前训练状态复现

[English](current-training-state-reproduction.md) | [简体中文](current-training-state-reproduction.zh-CN.md)

入口为 `experiments/training_state_transfer.py`。它复用当前 COCO Swin-T 源攻击 PNG，
评估已有注册 Faster R-CNN 受害模型训练对，不生成攻击图像，也不训练模型。

两个登记受害模型状态下 clean 与 LGP/VCSF 的真实单卡、双卡诊断已完成，使用
六张 COCO 前缀图像及其中一张 retained 投影。Main 已限定认可原始数值、输入
和采样所有权内容。原有过期 `running` 快照保持不变，以原生 terminal 为权威。
后继源码与报告修复处理范围标签及未来终态发布，已有直接源码审阅与保存的
导出核验。修复后的有界单／双卡终态及清理合同已另行接受，不改标原快照。
这不构成默认十方法完整面板的资格，也不授予严格受控训练或科学接受。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 前置条件

所有命令均从仓库根目录执行，先激活已有环境：`conda activate oda`。
环境与数据准备请参阅[当前复现指南](current-public-reproduction.zh-CN.md)，不要独立升级锁定的软件包。
GPU 执行要求 Linux 进程所有权支持、一个或两个可用的显式 CUDA 设备，以及每卡足够的显存。
双卡不能合并显存。仅查看计划不需要 GPU、图像或输入产物。

最终 COCO/VOC 复现应分别验证数据集，并明确限定 doctor 的数据和检查点范围：

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

GPU 执行前，要求数据报告有效、doctor 状态为 `ok`，且 COCO/VOC 检查点项均为 `ok`。
BDD100K 不是本次复现的前置依赖。省略 `--datasets` 仍检查全部登记数据集；
单独检查这个仅 COCO 的入口时可用 `--datasets coco`。环境与 NMS 检查没有改变，
应在 CUDA 资源可用时才运行真实 doctor。

直接复用已经合格的 `standard_control` / `adversarial_training` 训练对。
训练配方和相对产物位置仍由 `configs/experiments/training_state_transfer.yaml` 注册：
从同一初始检测器检查点配对继续训练、固定最终一个 epoch、每 minibatch 四次 replay、
batch size 为二、注册优化器与调度和训练图像覆盖，以及 epsilon 和步长均为 `4/255` 的注册对抗分支。
本入口不重训、不选择检查点；不要为了使用新接口而重训已有合格训练对。
仅有 `.pth` 文件不代表合格，资格记录还须认证训练轨迹、配对不变量和检查点验证。
完整划分执行要求 `strict_training_controlled_pair_eligible`；带 `--max-images` 的诊断要求
`diagnostic_pair_eligible`，不能据此宣称严格受控训练结果。

## 输入与范围

十个方法按当前 `main_transfer` 协议顺序继承：
`numbod`、`hifa`、`mlfadv`、`tog`、`augtrans`、`lgp`、`afog`、`osfd`、`sfim_b`、`vcsf`。
这个训练状态实验仅支持 COCO，固定源为 `mask_rcnn_swin_t`，受害模型为 `faster_rcnn_r50`。
历史训练状态方法列表不能替代当前主实验列表。基线保留注册的 compute-matched 预算，
VCSF 保留注册的 observed-path 预算与参数身份。

`--source-main` 应指向当前主实验目录，其中包含 `plan.json`、`provenance.json`，
以及 `attacks/coco/mask_rcnn_swin_t/METHOD/default/` 下的 `run.json`、
`annotations.json`、`manifest.jsonl` 和引用的 PNG 文件。
入口检查数值源码身份、完整载荷记录、实际解析参数哈希、预算、规范真值与图像 ID、seed、
源检查点一致性和选中载荷的字节。不得用历史结果或缺失载荷替代。

已有合格的当前主实验目录时直接复用。只有当前载荷缺失且已授权新运行时，才通过以下主入口准备。
第一条命令只检查计划；第二条会执行这一源／受害模型选择的攻击生成与评估。
示例输出目录若已存在，请改用新的名称。这些命令不是要求重跑已验收矩阵。

```bash
python experiments/main_transfer.py --plan-only --datasets coco --sources mask_rcnn_swin_t --targets faster_rcnn_r50 --output outputs/experiments/main_transfer/example-source-plan
python experiments/main_transfer.py --datasets coco --sources mask_rcnn_swin_t --targets faster_rcnn_r50 --devices cuda:0 --payload-retention keep_all --prediction-archive gzip --output outputs/experiments/main_transfer/example-source-main
```

## 查看与执行

不读取图像、载荷或权重，查看全部 22 个作业：

```bash
python experiments/training_state_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/training_state_transfer/example-plan
```

应显示 `Status: planned; inference jobs=22; generation jobs=0`。
两个报告范围均为 `NR` 占位。这是计划，不是执行或数值证据。

确认 GPU 路径资格与输入就绪后，可以选择下面任一执行命令。
两种模式请求同样 22 个作业：两种训练状态各包含 clean 与十种方法。
单卡串行执行两种状态，双卡把每种状态的完整作业交给一个受所有权约束的 worker。
这不是 DDP，不改变训练 batch，也不是源模型集成。

```bash
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-one
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-two
```

默认使用本仓库中注册相对位置的 pair manifest。
可选根目录示例为 `--pair-repository artifacts/qualified-pair-repository`，该目录必须在注册相对位置
包含 manifest 及训练对引用的全部产物。它只是产物根目录锚点，不代表另一套训练配方。
`--pair-manifest` 可选择该锚点内的 manifest；不指定 `--pair-repository` 时，以本仓库为锚点。

限定诊断时选择当前主实验方法并加上 `--max-images`：

```bash
python experiments/training_state_transfer.py --source-main outputs/experiments/main_transfer/example-source-main --methods numbod --max-images 10 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/training_state_transfer/example-diagnostic
```

`--max-images` 必须是 1 到 5000 的整数；完整 5000 张范围必须省略此参数，
即使指定 `--max-images 5000` 也仍是诊断。方法子集不是完整比较面板。
重复或未知方法、非显式或重复设备、超过两张设备均被拒绝。默认执行必须提供 `--source-main`；
不可用设备在创建输出目录前被拒绝。

## 输出与恢复

每个作业保存十二项原始 COCO bbox AP/AR 指标和哈希绑定的 gzip 预测归档。
完整 5000 张推理与确定性 retained-500 投影使用同一份保存预测，投影增加零次模型调用，
也不复制完整推理的时间或显存字段。诊断只投影选中图像与 retained-500 的交集；
空交集显示 `NR`，不写成数值零。完整范围与 retained-500 对应不同的结论范围。

- `plan.json`、`input_binding.json`、`records.json`：分配作业、验证输入和未舍入记录。
- `evaluations/STATE/METHOD/`：`metrics.json`、`predictions.json.gz`、`predictions_artifact.json` 和保留范围投影记录／汇总。
- `reports/fullval/`、`reports/retained500/`：每个范围各十二份 CSV／TeX 表；CSV 保留四位小数。`ERR` 表示失败，`NR` 表示未运行或不可用；原始未定义指标 `-1` 显示为 `NR`，不修改 JSON。
- `workers/SLOT/`：请求、进程／GPU 所有权记录、日志、进度、已完成作业文件和 terminal 回执。
- `summary.json`、`terminal.json`、`cleanup.json`、`execution_state.json`：完成、失败和自有进程收尾证据。

执行成功要求全部指定完整范围记录完成、非空保留范围投影完成、worker terminal 绑定准确的请求、
所有者和作业文件哈希，以及自有 worker 收尾通过验证。
最终状态为 `complete_pending_independent_acceptance`，不是科学验收。
默认完整面板包含 44 条指标记录，即 22 个推理作业各对应两个范围。

失败时检查协调进程和 worker terminal 以及 `worker.log`，保留整个目录与已完成前缀。
`ERR` 不得转换为零。修复缺失或变动输入时不要覆盖合格产物。
明确选择重跑后使用新输出目录；没有自动续跑，也不合并不同运行。
已有输出目录会被主动拒绝。可用 `--visualize-predictions` 开启可选预测叠图，不改变评估器预测。

如需另行创建新的受控训练对，参见[训练对指南](current-training-pair-reproduction.zh-CN.md)。本评估入口不训练该模型对。
