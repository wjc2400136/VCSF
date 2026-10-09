# 当前 VCSF 最终背景归因复现

[English](current-vcsf-background-reproduction.md) | [简体中文](current-vcsf-background-reproduction.zh-CN.md)

本入口使用新用户的当前输入，复现已选 A10 及九个登记控制。它重新生成本次
A10 锚点，不要求用户拥有作者历史结果。共用准备步骤见
[主复现指南](current-public-reproduction.zh-CN.md)。本入口已在固定 ODA 完成
有界 CPU 资格：100 个受影响合同与回归用例通过，包括替换 producer/evaluator
边界后的真实单卡协调控制流。CLI help、两种完整 plan-only、编译、模型登记、
兼容矩阵和当前主实验计划均通过；独立审阅提出的两项实际缺陷已修复复查。

后续真实单卡、双卡限图诊断已获限定接受：覆盖十个登记行（A10 与 F01-F09）、
Faster R-CNN 源、三个所选 COCO 目标及一张图像上限。其仅报告修复保留原始数值
记录与失败目录。这不构成完整 5,000 图、十六目标工作集的资格；下列命令仍是
用法示例，不是已接受的效能结果。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 科学范围

唯一源为 `faster_rcnn_r50`。每行使用完整 5,000 张 COCO val2017、基础 seed 42
加规范数字图像位置偏移，并评估全部十六目标。十行顺序直接来自
[登记的最终背景研究](../configs/experiments/ablations.yaml)：

| 行 | 相对 A10 的变化 |
| --- | --- |
| A10 | 完整已选方法；本次重新生成的参考行。 |
| F01 | 恒等几何视图。 |
| F02 | 仅颈部特征表征。 |
| F03 | 仅骨干特征表征。 |
| F04 | clean/adversarial 视图采用独立偏移。 |
| F05 | clean/adversarial 视图采用独立几何。 |
| F06 | 无检测初始化，20 次特征梯度更新。 |
| F07 | 随机符号初始化，20 次特征梯度更新。 |
| F08 | 均匀空间权重。 |
| F09 | 均值特征聚合。 |

所有行保持 `eps=4/255`、步长 `1/255` 和登记的 20 次逻辑梯度更新。
检测初始化行是 1 次检测梯度加 19 次特征梯度；F06/F07 为 20 次特征梯度。
这是原登记的预算内优化安排比较，不是隔离特征更新次数的因果对照。
F02/F03 同时改变表征与特征项数。参考前向、变换视图和部分特征反向保留在
观测记录中，不能当作免费工作或已校准的完整检测器 BE。

完整新用户工作集有十组生成、160 个攻击目标单元及十六个共用 clean 单元。
单卡串行相同工作集；双卡分配彼此独立的完整组，不引入 DDP、集成，不改变
batch、图像、seed、检查点或科学配置。每张卡必须独立容纳相应模型；显式
请求但不可用的设备会报错。

这不是重新搜索配置或独立选型确认。COCO val2017 和 DINO 已在离线选型中被
观察；DINO 不属于源集合，但不是未触碰的选型留出。控制行不会自动替换 A10。
作者自身原有的历史精确复用路径保持独立，本入口不授权重跑已接受面板，也不
把 A01/A23 的结果改名为 A10。

## 准备输入

所有命令在仓库根目录执行。按[根指南](../README.zh-CN.md)、
[COCO 数据指南](../data/README.zh-CN.md)和
[检查点指南](../checkpoints/README.zh-CN.md)准备，复用固定 `oda` 环境，不升级
组件。本入口需要完整 COCO 验证 manifest、图像和十六个登记 COCO 检查点，
不需要 VOC 或 BDD100K，不自动下载数据或权重。为十组 PNG 和完整预测归档
准备存储，不假设自动裁剪或自动续跑。

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

输入与环境检查无失败后再执行；准备通过不等于实验验收。

## 查看与执行

```bash
python experiments/current_vcsf_final_background.py --help
python experiments/current_vcsf_final_background.py --plan-only --devices cuda:0
python experiments/current_vcsf_final_background.py --plan-only --devices cuda:0,cuda:1
```

两种模式的计划都必须保留十行及 160 个科学单元。plan-only 不执行 GPU 推理，
不能证明设备可用。

有界诊断仍覆盖十行控制，只限制图像和目标，不能提供正式 mAP：

```bash
python experiments/current_vcsf_final_background.py --devices cuda:0,cuda:1 --max-images 1 --targets faster_rcnn_r50,cascade_rcnn_r50
```

完整复现须省略这两个限制选项：

```bash
python experiments/current_vcsf_final_background.py --devices cuda:0,cuda:1
```

单卡改为 `--devices cuda:0`，工作集不变。不要启动两个相同命令模拟双卡。
`--device cuda:0` 是互斥的单设备兼容写法。`--output` 指向新的空运行叶目录；
`--visualize-predictions` 可选绘制 NMS 后的框、类别和置信度，阈值 0.50、最多
100 框，每个目标至多三张图，不能改变评估预测。

## 输出与失败恢复

终端 `[OUTPUT]` 后为本次运行目录，默认位于
`outputs/experiments/current_vcsf_final_background/` 下的时间戳目录。
检查原始计划、分配、`execution_state.json`、`records.json`、组 summary、PNG manifest、
无损保存预测和实际成本观测。生产终态必须覆盖完整且零失败；PID 存活或单个
目标完成都不是终态验收。

`background_summary.json` 保留未舍入十二指标、排除源同名目标的 BB Mean，
以及九个 control-minus-A10 对照，包括逐目标差值。
`reports/background/A10/` 和 `F01/` 至 `F09/` 自动输出十二指标逐目标 CSV/TeX，
保持规范检测器顺序。`reports/background/summary.csv` 与 `summary.tex` 显示
BB Mean 和对照差值，单位为 AP/AR 百分点，至少四位小数；原 JSON 为官方
0 到 1 的未舍入数值。攻击后 AP 越低越强；control-minus-A10 的 AP 正值有利于
A10。失败汇总显示 `ERR`，未运行或不足以汇总显示 `NR`，原始缺失为 null，
不能记零；A10 自身的对照显示 reference。

独立像素与预测覆盖、官方保存预测重放、身份核验和科学解释仍是生产后的独立
步骤。限图结果、CPU 测试和计划都不能充当正式效能。保留失败目录与完整前缀；
修正真实原因后，只在新叶目录执行明确范围的恢复，不合并不可变目录，不因
连接超时重启，不删除保留载荷，不继承作者接受决定，不自动提升数值结果。
