# 当前预处理复现

另见[自适应指南](current-adaptive-preprocessing-reproduction.zh-CN.md)中的防御感知源管线 BPDA；这不改变本入口的非自适应威胁模型。

[English](current-preprocessing-reproduction.md) | [简体中文](current-preprocessing-reproduction.zh-CN.md)

`experiments/preprocessing_transfer.py` 在非自适应的受害检测器输入预处理下，评估当前主实验保留的 COCO PNG。该入口不重新生成对抗载荷、不训练检测器，也不要求历史作者运行的验收文件。

COCO Common-2、LGP/VCSF、`identity` 与 `bit_depth_4`、三个所选目标及一张 retained
图像的真实单卡、双卡诊断已完成。Main 已限定认可原始像素、预测、十二指标和
任务所有权内容；此非自适应路径未重新生成攻击载荷。后继仅报告修复已有直接
源码审阅与保存的导出核验，不覆盖原输出。这不构成完整 retained-500 面板的
资格，也不独立接受其报告。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 前提

在仓库根目录运行命令，先执行 `conda activate oda`。按[当前复现指南](current-public-reproduction.zh-CN.md)准备固定环境、规范 COCO 验证集和登记的 COCO 检查点，不要单独升级 OpenMMLab 组件。实际执行需要 Linux 进程所有权支持、明确指定的一张或两张可用 CUDA 卡，以及每张卡各自充足的显存。两卡显存不合并。`--plan-only` 不读取图像、检查点，也不要求 GPU。

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

## 范围与输入

默认使用两个彼此独立的源，顺序为 `faster_rcnn_r50`、`mask_rcnn_swin_t`，完整十六目标，以及当前主实验登记的十方法：`numbod`、`hifa`、`mlfadv`、`tog`、`augtrans`、`lgp`、`afog`、`osfd`、`sfim_b`、`vcsf`。基线保留登记的计算匹配预算，VCSF 保留实际路径成本预算和准确的当前参数身份。本实验仅使用 COCO，不是 VOC 或 BDD100K 面板。

十一种登记预处理为 identity、JPEG 质量 90/75/50、六位/四位量化、半径 `1/2` 和 `1` 的高斯滤波、大小 3 的中值滤波，以及比例 `3/4`、`5/4` 的双线性缩放往返。预处理位于受害检测器自身预处理之前。攻击者不知道所选受害端变换，攻击生成期间不查询目标检测器，也不获取目标梯度。这不是 adaptive/BPDA 评估。

图像人口是完整 COCO val2017 的 5,000 张图像按数值 ID 排序后，固定等分区间中心选择的 500 张。位置为 `((2*i+1)*5000)//1000`，其中 `i=0..499`；攻击使用原完整分割位置对应的种子。`--max-images` 截取这个固定选择的前缀，而非 val2017 的最前几张。即使 `--max-images 500` 仍属诊断；完整登记 retained-500 模式应省略该参数。这个范围不是 full-5,000 评估，也不会恢复用于配置选择的数据集独立性。

`--source-main` 必须指向当前 `main_transfer` 输出，含 `plan.json`、`provenance.json`，以及 `attacks/coco/SOURCE/METHOD/default/` 下的 `run.json`、`annotations.json`、`manifest.jsonl` 和它们引用的 PNG。输入核验绑定数值实现、参数哈希、比较预算、规范标注、准确图像 ID、原种子、清单字节和当前登记的 COCO 源检查点哈希。即使所有方法使用相同的其他权重，也会被拒绝。目标检查点路径和哈希同样绑定。不继承任何历史 AP 行。

已有合格当前主实验输出时直接复用。只有缺少载荷且新执行已获授权时，才通过主实验入口准备。第一条命令检查计划，第二条执行；它们不是要求重跑已接受矩阵。保留全部生成 PNG：

```bash
python experiments/main_transfer.py --plan-only --datasets coco --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50 --output outputs/experiments/main_transfer/preprocessing-source-plan
python experiments/main_transfer.py --datasets coco --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50 --devices cuda:0,cuda:1 --payload-retention keep_all --prediction-archive gzip --output outputs/experiments/main_transfer/preprocessing-source-main
```

主实验的一张图 smoke 连第一个区间中心选择 ID 都不覆盖。预处理诊断所需的每个 retained ID 都必须出现在主实验载荷前缀内；完整 retained-500 模式要求全部 5,000 张主实验载荷。不要给旧运行改名，也不要补造缺失清单。

## 检查与执行

不读取输入工件即可检查完整矩阵：

```bash
python experiments/preprocessing_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_oblivious_preprocessing_transfer/example-plan
```

预期显示 `Status: planned; inference jobs=3696; generation jobs=0`：176 个 clean 评估和 3520 个 attacked 评估，组织为 231 个完整预处理组。报告中的 `NR` 是占位，不是测量结果。

设备实机验证和输入准备完成后，选择以下其中一种模式：

```bash
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-one
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-two
```

两种模式的科学任务和种子映射相同。一卡串行执行全部组，双卡轮转分配彼此独立的完整组，同组目标面板留在同一工作进程。它不引入 DDP，不改变 batch，不集成源，也不缩小矩阵。使用新的输出目录，不要向已有目录重复执行示例。

有界诊断可选择登记子集，并限制至多 500 个 retained ID：

```bash
python experiments/preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --methods numbod --sources faster_rcnn_r50 --targets faster_rcnn_r50,dino_r50 --defenses identity,bit_depth_4 --max-images 1 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_oblivious_preprocessing_transfer/example-diagnostic
```

未知或重复选择、明确指定但不可用的设备会报错。方法或目标子集不是完整比较面板。可用 `--visualize-predictions` 和可视化阈值、数量参数启用可选叠框；这不改变评估器预测。

## 输出与恢复

- `plan.json`、`input_binding.json`、`records.json`：准确任务、认证输入和未舍入结果。
- `scratch/GROUP/`、`view_manifests/GROUP.json`：保留视图标注与清单；非 identity 组保留全部变换 PNG。identity 组直接引用原图像，不复制图像：clean 组引用规范 JPEG，attacked 组引用保留的对抗 PNG。
- `evaluations/GROUP/TARGET/`：完整十二项原始 COCO bbox AP/AR、完整 gzip 预测和完整性记录。官方重放读取保存预测，不重新模型推理。
- `workers/SLOT/`：请求、进程/GPU 所有权、日志、进度，每组一个完整 `view_GROUP.json`、组收据、紧凑完成记录和终态证据。
- `reports/DEFENSE/`：十二个 TeX 表和含全部十二指标、四位小数的 `transfer_records.csv`。报告中失败为 `ERR`、缺失为 `NR`；CSV 同时保留明确状态。原始未定义值 `-1` 仅在展示时显示 `NR`，不改 JSON。
- `summary.json`、`terminal.json`、`cleanup.json`、`execution_state.json`：总体完成、失败和所属工作进程收束。

每个源/方法行的 BB Mean 排除与源相同的白盒目标，只有该行全部所选黑盒目标都有数值时才计算；黑盒目标行不完整时，不计算局部均值。最优/次优排名则要求该源整个所选“方法 × 目标”面板的对应指标都有数值。因此，某一完整行仍可保留未标排名的 BB Mean，即使另一个方法尚不完整。

完成记录只保存紧凑视图元数据及完整像素绑定文件的路径/哈希引用，不反复复制图像绑定数组。工作进程开始和结束时完整重算输入头文件哈希，组开始和结束时完整检查视图像素；逐目标检查使用绑定的元数据/文件状态和保存预测重放。完整视图文件及其引用的每个 PNG、预测都必须保留。

执行成功要求全部所选评估完成、工作进程终态准确绑定所有权/请求/任务文件哈希，并验证进程收束。此时状态为 `complete_pending_independent_acceptance`，不自动成为科学接受。最终结果接受还需要独立的保存结果复核。

失败时保留原目录、工作日志、变换图像和完成前缀。查看 `terminal.json` 与 `workers/SLOT/worker.log`，修复具体原因后，将明确选择的重跑放入新目录。没有自动恢复、删除或合并。缺失结果不记零；已有输出目录会被拒绝，不被覆盖。
