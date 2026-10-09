# 可选 COCO 与 VOC 白盒诊断

[English](current-whitebox-reproduction.md) | [简体中文](current-whitebox-reproduction.zh-CN.md)

所有命令均在仓库根目录运行。`experiments/whitebox.py` 是供新用户选择的源与目标相同的对角诊断入口。[主实验复现](current-public-reproduction.zh-CN.md)已经将每个源检测器纳入评估目标；输入、方法配置和检查点身份一致时，直接复用其已接受白盒行，不为本指南重跑已接受矩阵，也不将独立白盒诊断视为额外科学证据。单独入口不继承历史接受状态。

COCO/VOC Common-2、LGP/VCSF 源与目标相同单元、每组一张图的真实单卡和双卡
诊断已完成。Main 已限定认可原始像素、预测与数值内容。原双卡登记仍为 `NR`；
另一次后继双卡诊断已接受此限定范围的 journal/registration 证据，不回填旧记录。
仅报告修复已有直接源码审阅及保存的编译、渲染收据。这不构成六源、全图面板
的资格，也不证明非对角迁移。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 前置条件

按 [environment.yml](../environment.yml) 和[固定依赖](../requirements/locked-cu118.txt)准备 `oda` 环境。保持 Python 3.8.20、PyTorch 2.0.0+cu118、torchvision 0.15.1+cu118、MMCV 2.0.1、MMEngine 0.7.4、MMDetection 3.0.0 和 MMYOLO 0.6.0 的整体版本，不为这个可选入口单独升级组件。

仅在尚无 `oda` 环境的新机器上，从仓库根目录创建环境；不要重建或迁移既有固定环境：

```powershell
conda env create -f environment.yml
conda activate oda
```

直接 Python 入口会将仓库源码目录加入导入路径。

准备并核验登记的数据清单和检测器检查点。COCO 全量诊断使用全部 5,000 张 val2017 图像；VOC 使用全部 4,952 张 VOC2007 test 图像，登记为 `val`，并使用各检测器单独微调的 VOC 检查点，不能代用 80 类 COCO 分类头。该入口不支持 BDD100K。每个对角单元的生成与评估须使用相同的当前检测器身份、数据标签空间、解析后配置与检查点 SHA-256；文件名或文件存在不代表检查点已经合格。

实际执行需要一张或两张可用 GPU，每张卡均须独立容纳其完整检测器与攻击作业，并为全部生成 PNG 和无损 gzip 预测准备存储空间。双卡显存不池化。启动全量诊断前，下列检查必须成功：

```powershell
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
```

仅运行 COCO 时，执行 COCO 数据校验命令并使用 doctor `--datasets coco`；选择 VOC 时还须核验 VOC。不要使用未指定数据集的校验默认行为，它会同时选择 BDD100K。doctor 将数据与检查点面板限定在所选数据集，但 VOC 深度检查仍核验原始 COCO 来源权重和 manifest，以保留训练证据链，而不是执行 COCO 数据集评估。请检查所选数据集的警告和资格；doctor 退出码零不单独证明这些数据与检查点已经就绪。

## 先检查计划

以下命令不读取图像或模型，也不执行 GPU 作业：

```powershell
python experiments/whitebox.py --help
python experiments/whitebox.py --plan-only --devices cuda:0
python experiments/whitebox.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

入口默认 COCO，登记范围仅为 COCO 和 VOC。每个数据集有 60 个源与方法单元，其中 52 个可执行作业、8 个已审定的结构性跳过。默认十方法及顺序与当前主实验一致：`numbod`、`hifa`、`mlfadv`、`tog`、`augtrans`、`lgp`、`afog`、`osfd`、`sfim_b`、`vcsf`。六个规范源依次为 `faster_rcnn_r50`、`mask_rcnn_swin_t`、`yolov3_d53`、`vfnet_r50`、`sparse_rcnn_r50`、`deformable_detr_r50`。

每个目标都固定为该作业的源。不要传入 `--targets`，即使只是再次指定该源；任何显式目标覆盖都会被拒绝。可使用 `--datasets`、`--methods` 和 `--sources` 选择子集，计划保留 source／method 的规范顺序、显式 dataset 输入顺序和兼容性跳过。可选 `naa` 仅在 Faster R-CNN 上有定义，`corrupting_attention` 仅在 Deformable DETR 上有定义，需要显式选择。退休的 `svfta` 不能选择。

重复的方法 ID 会被拒绝，不会生成重复逻辑作业。

基线沿用主实验的 `compute_matched` 优化安排；VCSF 沿用相同的 `vcsf_final_background_observed_cost` 预算口径与已裁决实现。参考前向与特征部分反向属于实际路径成本，不是免费操作，也不是已校准的完整检测器反向当量。设备数不会改变这些预算。

## 有界诊断

不带 `--plan-only` 时，入口默认实际执行，并在首个错误处停止。下列命令对每个独立源作业使用一张图像；先确认所选设备可用，每次只运行一种模式。这些结果不是正式 mAP：

```powershell
python experiments/whitebox.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/whitebox.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

两种模式执行相同的完整作业、图像、种子、参数和对角评估。单卡串行执行；双卡调度独立完整组，每组有唯一工作进程所有者，结果只由协调者写入。这不是 DDP、源集成、缩小矩阵或改变训练 batch／学习率。显式请求但不可用的设备会在执行开始前失败。不保证 CUDA 位级一致或两倍加速。

仅当新用户确实需要可选全量诊断，且前置检查成功后，才省略 `--max-images`：

```powershell
python experiments/whitebox.py --datasets coco --devices cuda:0,cuda:1
python experiments/whitebox.py --datasets voc --devices cuda:0,cuda:1
```

将设备列表改为 `cuda:0` 即可在单卡串行执行相同工作。不要启动重复命令模拟双卡调度。这些示例不要求重做作者已经接受的主实验白盒行。

## 输出与恢复

每次调用在 `outputs/experiments/whitebox/` 下创建新的 UTC 子目录。核对 `plan.json`、`provenance.json`、`execution_assignments.json`、`execution_state.json`、`records.json` 和 `summary.json`。载荷生命周期为 `keep_all`：完整保留生成 PNG 和无损 gzip 预测。可选的 post-NMS 预测框可视化不改变评估器预测。VCSF 攻击目录还保存逐图 `native_cost.jsonl` 观察记录。

原始 JSON 保留未舍入的十二项标准 bbox 指标：AP、AP50、AP75、AP-small／medium／large、AR1、AR10、AR100 和 AR-small／medium／large。记录自动生成 `reports/` 中的 CSV、逐指标 TeX 和图。计划报告中，未运行指标显示 `NR`，已审定结构性跳过显示 `--`；执行失败为 `ERR`，不能替换成零。源目标保留 `\dagger` 标记。标准 transfer 表格布局可能仍展示非对角列，但该入口不运行这些单元，没有黑盒面板，也没有可用的黑盒均值或排名。缺失的 `BB Mean` 不能填零或用于推断黑盒攻击强度。

完成要求预期图像与预测覆盖完整、每个可执行对角单元有十二指标、失败记录为零，并通过保存结果核验。成功退出、全量图像计数或未变的源码冻结都不单独构成正式数值接受。限图结果与可选诊断不得作为额外接受科学证据写入效能表。

失败时保留原目录、部分记录和日志。先确认真实进程身份，再判断是否停止；SSH 不可见不授权重复启动。修复已查明原因后，为明确缺失的数据集／源／方法子集检查计划，并使用新输出目录。没有自动续跑、覆盖或合并不可变运行目录的行为。不得通过裁剪图像或预测、替换权重、扩大方法范围来恢复失败。
