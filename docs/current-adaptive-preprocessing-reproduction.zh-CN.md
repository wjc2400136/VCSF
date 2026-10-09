# 当前自适应预处理复现

[English](current-adaptive-preprocessing-reproduction.md) | [简体中文](current-adaptive-preprocessing-reproduction.zh-CN.md)

`experiments/adaptive_preprocessing_transfer.py` 运行当前 COCO Common-2 固定 500 张的防御感知源管线 BPDA 比较，使用新用户自己的输入，不要求作者历史接受文件，也不继承作者数值结果。

COCO Common-2、LGP/VCSF、`identity` 与 `bit_depth_4`、三个所选目标及一张 retained
图像的真实单卡、双卡诊断已完成，包括在登记的 identity 复用政策下每种模式
六个源管线生成作业。Main 已限定认可原始像素、预测、十二指标和任务所有权
内容。后继仅报告修复已有直接源码审阅与保存的导出核验，不覆盖原输出。
这不构成完整 retained-500 面板的资格，也不独立接受其报告。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 准备与身份

在仓库根目录、固定 `oda` 环境中执行。按照[当前主实验指南](current-public-reproduction.zh-CN.md)准备 COCO 数据、发布检查点及当前主实验保存的源攻击 PNG。本入口仅用于 COCO，VOC 和 BDD100K 不属于这个预处理实验。

```bash
conda activate oda
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
python experiments/adaptive_preprocessing_transfer.py --help
```

GPU 执行前需要数据检查通过，以及 doctor 和检查点面板为 `ok`。不要独立升级统一环境。执行要求 Linux 进程归属支持和一张或两张明确指定、可用的 CUDA 卡；每张卡都必须独立容纳分配给它的完整源和目标作业，显存不合并。

`--source-main` 指向你自己的当前 `main_transfer` 输出，绑定准确的方法实现、参数、预算、源检查点、规范标注、保留图像及全验证集 seed 位置。其基础 seed 必须为登记的 42。输入需包含所选当前方法的源 PNG、manifest、run 元数据，以及 `plan.json` 和 `provenance.json`。完整模式需要全部 5,000 张源图像覆盖。诊断输入的主实验前缀也必须覆盖每个请求的保留 ID，不能只提供相同数量的最前几张图。[非自适应指南](current-preprocessing-reproduction.zh-CN.md)说明源主实验准备和固定等分中心人口。已有合格载荷应直接复用，这些命令不授权重复作者已接受的实验。

## 固定范围与 BPDA

默认保留当前主实验十个方法、两个独立源 `faster_rcnn_r50` 和 `mask_rcnn_swin_t`、十一种登记预处理、规范十六目标，方法和目标顺序不变。固定 500 个 ID 来自数值 ID 排序的完整 COCO 验证集等分中心；`--max-images` 取它们的前缀，不取 val2017 的最前几张。任何显式限制，包括 500，都属于诊断；完整登记的固定 500 范围要省略该参数。这不是完整 5,000 张评估，也不能恢复选型后的数据独立性。

生成从原始 clean 图像开始，只对已知预处理和源检测器的组合求导。生成时不查询目标模型，也不使用目标梯度。epsilon 相对于原始 clean 图像，而非预处理视图。部署前向及固定导数规则来自 `preprocessing_defenses.yaml`：identity 用准确 autograd；JPEG、位深和中值用 identity BPDA；高斯用登记的归一化高斯导数；缩放用双线性求导。前向保持准确登记的部署变换，不增加额外 EOT、按结果选择的替代梯度或新方法目标。

每图 seed 仍为 42 加原始全验证集位置。基线保留 `compute_matched`，VCSF 保留最终 A10 和 `vcsf_final_background_observed_cost`；其参考行和局部反向路径另行观察，不能算免费操作或已标定的完整检测器反向等效。登记的 A10 identity 分支复用当前主实验像素；corrected-LGP 和其他基线的 identity 分支按既有声明重新生成。不通过给旧载荷换名或改参数匹配这项政策。

完整新用户矩阵包含 218 个新生成的源/方法/预处理载荷、两个复用的 A10 identity 载荷、176 个 clean 与 3,520 个攻击目标评估，共 3,696 个评估作业、231 个完整目标面板组。这是复现工作量，不是新增作者实验清单。

## 检查与执行

以下计划不读取源载荷、数据或检查点，也不使用 GPU：

```bash
python experiments/adaptive_preprocessing_transfer.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_adaptive_preprocessing_transfer/example-plan
```

预期显示 `Status: planned; inference jobs=3696; generation jobs=218`，计划报告为 `NR`。设备资格和输入就绪后，选择一种模式，并使用新的输出名称：

```bash
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-one
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --devices cuda:0,cuda:1 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-two
```

单卡串行同一工作集；双卡调度独立的完整生成及目标面板组，不引入源集成、DDP、预算变化或删减作业。不要启动两个协调器实现双卡。不承诺 CUDA 位级同一或正好两倍速度。

已有足够源主实验覆盖时，可以执行有界接口诊断：

```bash
python experiments/adaptive_preprocessing_transfer.py --source-main outputs/experiments/main_transfer/preprocessing-source-main --methods numbod,vcsf --sources faster_rcnn_r50 --targets faster_rcnn_r50,dino_r50 --defenses identity,bit_depth_4 --max-images 1 --devices cuda:0 --no-visualize-predictions --output outputs/experiments/current_adaptive_preprocessing_transfer/example-diagnostic
```

## 保存输出与恢复

检查 `plan.json`、`input_binding.json`、`records.json`、`summary.json`、`terminal.json`、`execution_state.json` 和 `cleanup.json`。`workers/SLOT/` 保留请求哈希、进程出生身份、GPU 注册、完整组视图和终态收据。进度区分自适应生成和目标评估。`adaptive_attacks/GROUP/` 保留原始攻击 PNG、标注、manifest 及原生 A10 成本观察，不裁剪载荷。

`scratch/GROUP/` 和 `view_manifests/GROUP.json` 绑定部署到目标的视图。identity 视图直接引用原 clean、新生成或复用像素，不重复复制；所有非 identity 视图 PNG 都保留。`evaluations/GROUP/TARGET/` 保存十二个未舍入指标和完整无损 gzip 预测。官方保存预测重放不重新运行模型推理。

`reports/DEFENSE/` 包含十二张 TeX 表及至少四位小数 CSV 记录。BB Mean 排除源匹配目标，并要求所选黑盒整行完整；排名要求所选比较面板完整。失败为 `ERR`，缺失和未运行为 `NR`，原始未定义 `-1` 显示为 `NR`，不填零。报告属于 adaptive/BPDA，不是 oblivious 或认证鲁棒性。

成功生产者的终态是 `complete_pending_independent_acceptance`，不自动接受结果。保留完整像素、预测与收据供独立保存结果核验。失败后保留原目录、日志和完整前缀，检查真实进程身份、修复具体原因，再用新输出执行明确缺失子集。不自动续跑、覆盖、删除或合并；连接中断不证明 worker 已停止。
