# COCO 与 VOC 主实验复现

同一保留人口的当前防御感知源管线 BPDA 复现见[自适应预处理指南](current-adaptive-preprocessing-reproduction.zh-CN.md)。

使用本次完整主实验的保存预测离线绘制固定三图，见[定性图指南](current-qualitative-reproduction.zh-CN.md)。

新用户复现 A10 及九个不变的最终背景控制、重新生成锚点而不依赖作者历史结果，
请参见[最终背景指南](current-vcsf-background-reproduction.zh-CN.md)。

[English](current-public-reproduction.md) | [简体中文](current-public-reproduction.zh-CN.md)

所有命令均在仓库根目录执行，并先激活固定的 `oda` 环境。请先准备并核验数据集清单和检测器检查点。COCO 使用全部 5,000 张 val2017 图像及发布的 COCO 检查点；VOC 使用全部 4,952 张 VOC2007 test 图像及各检测器单独微调的 VOC 检查点，不能替换为 COCO 分类头。BDD100K 不属于最终维护的研究范围。

当前主入口已通过有界单卡／双卡诊断：COCO／VOC 共四例（COCO 限图 6、VOC 限图 1），
覆盖两个源、LGP／VCSF 和三个目标。最终背景入口也已通过两例 COCO 单卡／双卡诊断
（限图 1、十个控制、三个目标）。这只接受上述诊断范围，不代表全项目发布或正式效能结果接受。
`--plan-only` 和 CPU 接口检查本身不能证明 GPU 执行成功。

当前 COCO 配对训练状态评估请参见[训练状态指南](current-training-state-reproduction.zh-CN.md)，复用主实验保留的 Swin-T 源载荷。该入口只评估登记的受害者检查点对，不重新训练或生成攻击图像。

固定非自适应 COCO 预处理评估请参见[预处理指南](current-preprocessing-reproduction.zh-CN.md)，复用当前主实验保留的 Common-2 载荷，在规范的 retained-500 图像上评估；不重新生成攻击，也不继承历史数值结果。

当前 A10 COCO 半径敏感性请参见[五半径指南](current-vcsf-radius-reproduction.zh-CN.md)。新用户的默认工作集覆盖两个 Common-2 源和全部五个登记半径；作者自身继续精确身份复用，不重跑已接受面板。

主实验已包含源与目标相同的白盒行；当前身份一致时复用已接受行，无需额外运行白盒。[可选白盒诊断指南](current-whitebox-reproduction.zh-CN.md)介绍独立的新用户对角入口，不构成额外科学证据，也不要求重跑已接受矩阵。

当前输入的配对攻击时间与显存测量见
[成本复现指南](current-paired-cost-reproduction.zh-CN.md)。该入口在固定 Common-2
保留图像集上测量当前 A10 与 corrected-LGP 实现，不计算 AP，不重做或继承作者
已接受的成本结果。选择两张卡时仍按源串行测量，以避免并发污染计时。

## 执行前检查

```powershell
conda activate oda
python experiments/main_transfer.py --help
python experiments/main_transfer.py --plan-only --datasets coco,voc --devices cuda:0,cuda:1
```

默认数据集是 COCO。默认主比较沿用当前论文的十个方法及其顺序，覆盖六个独立源检测器和完整十六目标。NAA 与 Corrupting Attention 可通过 `--methods` 显式选择，已审定的结构性不适用仍显示为 `--`。退休的 SVFTA 不能通过这个当前主入口选择。

输出计划包含完整作业集和工作进程分配。各基线保留原 `compute_matched` 优化安排；VCSF 保留已裁决的 A10 唯一实现、完整参数和逻辑更新／实际路径成本口径。clean reference 行与特征部分反向并非免费操作，也不能直接改称已校准的完整检测器反向当量。

## 有界接口检查

每次只运行一种模式，且确认所选设备可用。以下命令会实际执行，但仅为诊断，不是完整数据集效能结果。

```powershell
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/main_transfer.py --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

两种模式执行相同的完整源作业、图像人口、种子、参数与目标评估。双卡只调度独立完整作业，不合并源梯度、不启用 DDP，也不池化显存。每张卡均须独立容纳其检测器和攻击；显式请求但不可用的设备会在开始输出执行前失败。不保证 CUDA 位级相同或两倍加速。

## 完整复现

仅在数据、检查点核验成功且确实准备执行完整计划时，才省略 `--max-images`。

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
python -m lgp doctor --datasets coco,voc --deep-data
python experiments/main_transfer.py --datasets coco --devices cuda:0,cuda:1
python experiments/main_transfer.py --datasets voc --devices cuda:0,cuda:1
```

doctor 只校验选定的数据与检查点面板。仅做 COCO 时可使用 `--datasets coco`；VOC 深度检查仍核验其原始 COCO 来源权重和 manifest，以保留训练证据链，而不是执行 COCO 数据集评估。上述命令不选择 BDD100K。

使用 `--devices cuda:0` 可在单卡上串行同一工作集。不要启动两份相同命令来模拟双卡。新用户运行绑定其当前输入和检查点，不继承作者历史结果。

## 输出与恢复

每次运行在 `outputs/experiments/main_transfer/` 下生成新的 UTC 目录。核对 `plan.json`、`execution_assignments.json`、`execution_state.json`、`records.json` 与 `summary.json`。生成 PNG 保留，预测无损归档，原始记录自动生成 `reports/` 中的 CSV、TeX 和图。VCSF 攻击目录额外保存逐图 `native_cost.jsonl` 观察记录；这不是配对性能实验或 FLOP 校准。

[调度证据指南](current-dispatch-evidence.zh-CN.md)说明持久化的工作进程归属、设备上下文登记与终态清理记录。

仅用 CPU 将已保存标量报告重导出到新的输出目录，见
[已保存报告重导出指南](saved-report-export.zh-CN.md)及其
[直接入口](../experiments/reexport_saved_reports.py)。不可变原件保持不动；
不加载模型、不读取预测、不重算 AP，也不构成科学结果接受。

成功要求图像覆盖符合计划、目标结果完整、失败记录为零，并通过对应的保存结果核验。单独的成功退出不等于独立科学接受。`--max-images` 结果不得进入正式 mAP 表；原始 JSON 保留完整十二项 AP／AR 值。

命令失败时保留原目录和日志。先核对真实进程身份，再判断其是否已停止；SSH 断连不代表执行失败。修复已查明的原因后，在新目录执行明确选择的缺失子集；没有自动续跑、覆盖或合并不可变运行目录的行为。缺失与失败值不得替换为零。
