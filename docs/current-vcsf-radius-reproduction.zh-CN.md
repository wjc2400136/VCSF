# 当前 A10 VCSF 五半径复现

[English](current-vcsf-radius-reproduction.md) | [简体中文](current-vcsf-radius-reproduction.zh-CN.md)

本指南面向首次使用 GitHub 项目的用户，从自己的当前 COCO 输入和登记检查点生成当前已选定 A10 VCSF 的结果。默认工作集覆盖全部五个登记半径，不继承作者的数值结果或其验收状态。共享准备步骤请复用[主实验复现指南](current-public-reproduction.zh-CN.md)。

本[入口](../experiments/current_vcsf_radius.py)及 [runner](../src/lgp/runners/current_radius.py) 已在固定 `oda` 环境完成有界 CPU 资格：入口 help、单卡／双卡 `--plan-only`、执行 guard 和报告契约均通过相应检查。

真实单卡、双卡诊断覆盖 COCO Common-2、`4/255`、两个源匹配目标及每组一张图。
Main 已限定认可原始像素、预测与数值内容。原双卡登记仍为 `NR`；另一次后继双卡
诊断已接受此限定范围的 journal/registration 证据，不回填旧记录。仅报告修复已有
直接源码审阅及保存的编译、渲染收据，不等于全项目发布或科学接受。这些诊断
不证明全图、全五半径工作集的资格，也不将下列命令认定为已接受的效能结果。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 科学范围

仅扰动半径发生变化。已选定 A10 的实现、检查点、图像集合、各源配置、种子策略和评估器保持不变。这是半径敏感性复现，不是方法搜索、目标查询优化，也不是对方法选择的独立确认。

| 设置 | 固定范围 |
| --- | --- |
| 数据集 | COCO；每个源与半径组合使用全部 5,000 张 val2017 图像。 |
| Common-2 源 | 按顺序为 `faster_rcnn_r50`、`mask_rcnn_swin_t`；两个独立攻击，不做源集成。Swin-T 仅评估 bbox。 |
| 半径及登记顺序 | `2/255`、`4/255`、`8/255`、`16/255`、`32/255`。 |
| 步长 | 所有半径均为 `1/255`，不随 epsilon 缩放。 |
| 迭代 | 20 次：检测初始化梯度 1 步，加特征梯度 19 步。 |
| 种子 | 基础种子固定 42，沿用逐图位置偏移；不提供种子搜索选项。 |
| 目标 | 规范十六检测器，含源匹配白盒目标和 held-out DINO。 |

默认计划共有十个独立的源×半径组、160 个源×半径×目标攻击评估，另外执行十六个跨半径共享的 clean 目标评估。每组在相同的全部 5,000 张图像上生成攻击，并评估十六目标。`32/255` 是高失真压力测试，不应表述为常规扰动预算。任何半径都不得暗中改变目标函数、优化器、检测初始化、特征步或模型预处理。逻辑更新次数并非已校准的完整检测器反向当量；参考前向与部分反向也不是免费工作。

本公开入口独立于历史十一方法的 `experiments/coco_multi_budget_linf.py`，不要用历史入口替代。作者自身继续 `4/255` 的精确身份复用与另行授权的缺失分片工作。本指南不授权重复执行，也不要求重跑任何作者已接受矩阵。

## 准备仓库

在包含 `environment.yml`、`pyproject.toml`、`experiments/` 和 `src/` 的仓库根目录打开终端。下列所有命令都假设当前目录是该根目录。固定 `oda` 环境请复用[根指南](../README.zh-CN.md)，数据和检查点请复用 [COCO 数据指南](../data/README.zh-CN.md)、[检查点指南](../checkpoints/README.zh-CN.md)及[主实验指南](current-public-reproduction.zh-CN.md)。已有正确的 `oda` 时只需激活，不重新创建，也不升级其固定 Python、PyTorch 或 OpenMMLab 组件。本流程需要规范 COCO 验证清单、完整 val2017 图像和全部十六个登记 COCO 检测器检查点；不需要 VOC 或 BDD100K 输入。为全部生成 PNG 和完整预测归档预留磁盘空间，不以裁剪输出为前提。

入口不会下载缺失的数据或权重；缺失或无效输入会导致执行失败。重试前先按链接完成准备流程。

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

只有深度数据校验和 doctor 均没有失败检查后，才继续。这些检查证明输入及环境就绪，不证明 GPU 执行或独立科学接受。每张所选 GPU 都必须独立容纳分配的检测器和攻击；两张卡不池化显存。

## 检查工作集

查看实际安装版本的接口，并选择对应设备模式的计划。下列 help 与单卡／双卡 plan-only 接口已完成有界 CPU 资格，不据此证明真实 GPU 执行：

```bash
python experiments/current_vcsf_radius.py --help
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0,cuda:1
```

`--plan-only` 用于无 GPU 工作集检查，不是执行或设备可用性检查。核对两种模式保留相同的源、半径、图像 ID、固定种子策略、检查点、攻击设置与目标集合。查看保存的计划和调度分配，不从计划推断 GPU 已就绪。

| 参数 | 含义 |
| --- | --- |
| `--help` | 显示实际安装版本的入口接口。 |
| `--plan-only` | 检查工作集，不运行 GPU 实验。 |
| `--max-images` | 1 至 5,000 的整数，始终仅用于诊断；完整数据集结果必须省略。 |
| `--sources` | 仅限两个 Common-2 源 ID 的逗号分隔子集。 |
| `--targets` | 仅限规范十六目标 ID 的逗号分隔子集。 |
| `--epsilons` | 仅限五个登记分数的逗号分隔子集。 |
| `--devices` | 显式选择一张或两张卡；默认 `cuda:0`。 |
| `--device` | 严格单设备的旧兼容写法，如 `cuda:0`；拒绝逗号分隔设备字符串，替代 `--devices` 使用。 |
| `--output` | 新运行目录，不得使用已有或已接受结果目录。 |
| `--visualize-predictions` | 可选的 NMS 后框、类别和置信度叠加图；不得改变评估器预测。 |

默认值为两个源、全部十六目标、全部五半径和基础种子 42。所选列表必须是非空、无重复的登记子集，自动恢复登记顺序，不按命令中给出的顺序报告。epsilon 须使用 `2/255,4/255` 这样的确切标签；小数及其他等价写法不是登记选择项。每个 epsilon 映射到单独登记的 `current_vcsf_radius_2_255` 或其他相应半径 study，不转入历史多方法入口。`--devices` 与 `--device` 互斥；双卡使用 `--devices cuda:0,cuda:1`，`--device` 只接受一个设备。没有方法选择、种子覆盖、目标查询搜索、源集成或 DDP。

## 诊断与完整复现

准备成功后，可选择单个半径、两个目标做极小诊断：

```bash
python experiments/current_vcsf_radius.py --epsilons 4/255 --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
```

该命令保留两个彼此独立的源，但不覆盖完整半径与目标面板。任何带 `--max-images` 的结果都只是诊断，无论退出状态如何，都不是正式 AP 或 mAP。选择半径、源或目标子集，即使不限制图像，也会明确标为 `diagnostic_only`，不构成默认完整复现。`--max-images 5000` 同样仅为诊断。

确实准备好执行完整工作集后，只选择一种设备模式：

```bash
python experiments/current_vcsf_radius.py --devices cuda:0
```

或者在两张均有足够显存且可用的 GPU 上执行：

```bash
python experiments/current_vcsf_radius.py --devices cuda:0,cuda:1
```

两条命令是备选模式，不要求都运行。不带选项时，入口在 `cuda:0` 执行全部五半径，默认会实际执行。单卡串行同一完整工作集；双卡调度独立完整的源×半径组，每组覆盖其完整目标选择，不改变图像、种子、预算、检查点身份或评估。不要启动重复命令模拟双卡调度。不保证 CUDA 位级相同或两倍速度。显式请求的设备不可用时必须明确失败，不得静默切换。上述限图诊断不构成任一模式全图、全五半径工作集的资格。

## 输出与成功条件

不指定 `--output` 时，每次调用在 `outputs/experiments/current_vcsf_radius/` 下新建 UTC 目录。显式输出请使用新的相对目录，例如 `--output outputs/experiments/current_vcsf_radius/my-run-01`，每次新尝试更换后缀。保存规则见[输出指南](../outputs/README.zh-CN.md)。

核对 `plan.json`、`radius_plan.json`、`records.json`、`summary.json`、`execution_state.json`、`execution_assignments.json` 和 `radius_summary.json`。wrapper 的 `radius_plan.json` 保存半径选择及解析后参数哈希；`plan.json` 和调度／状态文件描述共享执行。`radius_summary.json` 保留各 epsilon 的结果单元格、排除源匹配目标的均值及压力测试标识，其 `historical_result_inheritance`、`scientific_acceptance`、`project_wide_device_release_accepted` 均保持 false。

报告按所选半径位于 `reports/radius/2_255/`、`4_255/`、`8_255/`、`16_255/`、`32_255/`。每个目录包含一个具有全部十二项 metric 列的 `transfer_records.csv`，以及十二个分别对应 metric 的 TeX 文件，例如 `transfer_bbox_mAP.tex`、`transfer_bbox_AR_100.tex`。仅计划输出中的未运行数值为 `NR`，不是数值证据。

保留所有生成的攻击 PNG 及每个完整、无损 gzip 预测归档。可选叠加图不能替代这些产物，也不改变评估器预测。当前叠加设置为置信度阈值 0.50，每次评估最多三张图，每张图最多展示 100 个检测框。CPU 检查覆盖接口、执行 guard 和报告契约；独立 GPU 诊断仅覆盖上述限定范围，不证明完整运行已完成。

wrapper 拒绝与解析后的 `src/`、`configs/`、`checkpoints/` 或 COCO 数据根目录相同、位于其中或包含这些目录的输出位置；共享 runner 拒绝非空输出目录。始终选择 `outputs/` 下的新目录。

报告必须从未舍入 JSON 自动生成，不手填数值。每个 epsilon 的 CSV 与 TeX 都须保留全部十二项标准 COCO 汇总：AP、AP50、AP75、AP-small、AP-medium、AP-large、AR-at-1、AR-at-10、AR-at-100、AR-small、AR-medium、AR-large。即使输入记录乱序，也须保留以下目标顺序和架构组标题：

| 架构组 | 组内目标顺序 |
| --- | --- |
| Two-stage | `faster_rcnn_r50`、`cascade_rcnn_r50`、`mask_rcnn_swin_t` |
| YOLO family | `yolov3_d53`、`yolov5_s`、`yolox_s`、`yolov8_s` |
| Dense/point-based | `retinanet_free_anchor_r50`、`reppoints_r50`、`vfnet_r50`、`tood_r50`、`rtmdet_s` |
| Query/set-based | `sparse_rcnn_r50`、`detr_r50`、`deformable_detr_r50`、`dino_r50` |

DINO 的 held-out 表头角色不表示未参与离线配置选择，也不是独立 selection holdout：当前 DINO 确曾参与选型，但每次攻击生成仍只使用指定源，不查询迁移目标。

TeX 表头的 `\dagger` 表示登记白盒源角色，`\ddagger` 表示 held-out DINO；当前源匹配白盒结果单元格另用星号标记，不与表头源角色混淆。完整面板的 `BB Mean` 排除源匹配白盒目标，使用其余十五目标。目标子集必须披露实际分母，不得冒充完整面板的 `BB Mean`。原始 JSON 保留官方 0 至 1 数值尺度上的未舍入值；CSV 使用同一官方尺度、展示四位小数，半径补充复现 TeX 使用百分制、展示四位小数。论文作者允许的显示精度在稿件投影中另行处理，不应用于这些补充复现输出。展示舍入不得改变原始 JSON、排序或均值。未运行、失败和已审定结构性不适用分别用 `NR`、`ERR`、`--`，不能填成零。

默认完整复现要求十个组均覆盖全部 5,000 个图像 ID，160 个攻击目标评估与十六个 clean 目标评估完整，无失败或缺失记录，载荷完整保留，半径报告与保存计划及当前输入、配置身份一致。仅计划输出或有限图像输出不满足这些条件。退出码 0 不等于独立保存结果审计、科学接受或全项目设备模式发布验收。重复使用 COCO 划分进行敏感性分析不会恢复数据集独立性。

## 失败恢复

保留失败目录、日志、记录及部分 PNG／预测证据。不可变运行目录没有自动续跑、覆盖或合并。终端／SSH 断连不证明 worker 已停止；重试前请运行负责人确认其状态。针对实际环境、输入、设备或已报告执行错误修复，不改变方法或半径约定。

wrapper 在共享 runner 返回后才写半径专属计划、汇总和报告。因此失败时可能已有共享执行证据，却没有最终半径专属产物；请保留已有证据，不据此认定没有执行过。若半径报告投影本身失败，也保留 `radius_failure.json` 及其报告的阶段和原因；失败记录不授予科学或设备模式发布接受。

负责人确认可恢复后，只用已支持的选择参数，将明确缺失的源／半径／目标子集写入新目录。任意缺失图像分片不能靠虚构参数或将 `--max-images` 当续跑实现，须与主运行负责人协调。原始来源记录保持独立，不把部分记录升级为完整结果。作者仍限于 `4/255` 精确复用及已授权缺失分片，不重跑已接受面板。
