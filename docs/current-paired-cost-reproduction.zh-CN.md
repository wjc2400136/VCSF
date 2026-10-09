# 当前配对成本复现

[English](current-paired-cost-reproduction.md) | [简体中文](current-paired-cost-reproduction.zh-CN.md)

此新输入入口在登记的 COCO retained-500 图像人口上测量攻击 API 时间与显存，不计算 AP、不保存对抗图像载荷，也不读取或继承作者私有历史结果目录。这不是 GPU 效能矩阵，不要求重跑已接受的 AP 比较；独立的效能流程见[主实验复现指南](current-public-reproduction.zh-CN.md)。

Common-2 与 LGP/VCSF 的真实单卡、双卡成本诊断已完成，每个源方法对使用两张
retained 图像，每种模式八次测量及八次 warmup。两种模式均保持源测量全局串行；
双卡模式实际观察到两设备，但没有并发测量。Main 已限定认可原始数值、输入
和采样所有权内容。后继仅报告修复已有直接源码审阅与保存的导出核验。这不
构成十方法 retained-500 面板的资格或可发表成本的独立接受；声明上限与已校准
物理工作仍须区分。

Main 已另行限定接受已审阅的源包增量与保存的 CPU 计划。
单／双设备调度已按所声明维护路径的角色范围接受，该核对不再有最小实际设备路径缺口。
这不代表每个默认研究或每种 CLI 选项均获原生 GPU 执行资格。
版权、分发许可及全项目发布仍待 Main 裁决，不授权公开发布或外部上传。
CPU 检查与限图 GPU 诊断均不是正式 mAP 或科学接受。

## 前提条件

所有命令均从仓库根目录运行，并使用固定的 Conda `oda` 环境。先准备规范 COCO 验证集清单、完整 5,000 张 val2017 输入和登记的 COCO 源检查点。沿用现有数据／检查点准备流程，不提供历史 AP 记录。

```powershell
conda activate oda
python experiments/current_paired_cost.py --help
python -m lgp doctor --help
```

实际测量前，核验当前环境、所选 COCO 数据与检查点面板：

```powershell
python -m lgp data validate --dataset coco --deep
python -m lgp doctor --datasets coco --deep-data
```

doctor 应无失败项。选择 COCO 时会校验登记的 COCO 检查点面板，不仅是两个成本源。该检查只是执行前核验，不是 GPU 执行或科学接受凭据。

选择一张或两张可用的显式设备。每张所选 GPU 必须独立容纳完整检测器与攻击，不能池化显存。测量门禁要求**全局不存在 foreign GPU 计算进程**，包括未选 GPU 上的其他计算；所选设备的 reservation 不能替代此全局检查。不要同时启动第二份测量命令或 AP 作业。

## 查看计划

以下命令不加载图像／检查点载荷、不实例化攻击／检测器、不查询 CUDA 可用性，也不调用 `nvidia-smi`。它们检查当前源码／配置身份，并在新目录写出计划与 `NR` 成本行。因此，计划成功不证明所请求设备实际存在。

```powershell
python experiments/current_paired_cost.py --plan-only --devices cuda:0 --output outputs/experiments/current_paired_cost/plan-single-01
python experiments/current_paired_cost.py --plan-only --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/plan-dual-01
```

默认顺序为 `numbod,hifa,mlfadv,tog,augtrans,lgp,afog,osfd,sfim_b,vcsf`。Common-2 两源依次为 Faster R-CNN R50 和 Mask R-CNN Swin-T（仅 bbox）。显式方法子集须为偶数个互不重复的当前方法，且必须包含 `vcsf`；用户传入顺序会规范化。未知、重复或格式错误的方法／设备以及非法图像上限会失败。

两种设备模式保持相同参数、预算身份、源检查点、所选图像 ID 与完整位置种子映射。**单卡和双卡均串行源作业**，`concurrent_measurements=1`：双卡将第一源分配给第一设备、第二源分配给第二设备，但不重叠测量。目的是隔离计时，不是提高 GPU 吞吐量；不引入源集成或 DDP。

## 小规模诊断与完整测量

执行前完成数据检查及全局独占核验，每次只运行一种模式。以下小规模命令实际执行，每个源方法对测量两张 retained 图像，不是可发表成本或效能结果：

```powershell
python experiments/current_paired_cost.py --methods lgp,vcsf --max-images 2 --devices cuda:0 --output outputs/experiments/current_paired_cost/tiny-single-01
python experiments/current_paired_cost.py --methods lgp,vcsf --max-images 2 --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/tiny-dual-01
```

默认真实执行省略 `--plan-only`、`--max-images` 和 `--methods`，测量完整登记的 500 图像、十方法、两源成本面板：

```powershell
python experiments/current_paired_cost.py --devices cuda:0 --output outputs/experiments/current_paired_cost/full-single-01
python experiments/current_paired_cost.py --devices cuda:0,cuda:1 --output outputs/experiments/current_paired_cost/full-dual-01
```

这两条是可选模式，不要求两种都运行。默认共有 10,000 次测量调用及另外 100 次 warmup 调用：每个源方法对使用五张 warmup 图像。warmup 取前五张所选图像，不进入测量汇总；图像限制较小时取 `min(5, selected_images)`。任何方法子集或图像限制，包括 `--max-images 500`，都仅是诊断。

所选图像来自排序后 5,000 图像人口的登记等分区间中心。每次调用使用 `seed = 42 + original_full_population_position`，不是 retained 子集内序号。参数哈希、VCSF 已选 A10 身份、原基线优化安排及逐方法预算身份均绑定当前声明。retained-500 成本测量**不是 5,000 张效能结果**，不能填入 AP 表。

## 记录与成本边界

每个输出叶目录必须全新；省略 `--output` 会在 `outputs/experiments/current_paired_cost/` 下生成新 UTC 目录。输出不得与解析后的源码、配置、数据或检查点路径重叠，包括检查点软链接目标及受保护祖先目录。

检查 `plan.json`、`input_binding.json`、`environment.json`、`coordinator.json`、`execution_state.json`、各源的 `warmup.jsonl` 与 `measurements.jsonl`，以及汇总 `measurements.json`、`summary.json`、`terminal.json`、`artifact_manifest.json`、`cost_by_source_method.csv` 和 `cost_by_source_method.tex`。输入绑定保留注释、图像及源检查点哈希，执行前后重新检查当前源码与公开身份。

| 字段 | 含义与边界 |
| --- | --- |
| `wall_seconds` | 同步攻击 API 墙钟时间，包含 reference 工作，不含解码、输入预载与外部校验。 |
| `peak_allocated_mib` | 调用期间分配显存峰值，包含常驻模型／输入，不是攻击独占增量显存。 |
| `base_allocated_mib` | 计时调用前的显存分配量。 |
| `peak_reserved_mib_shared_allocator` | 共享 warm-cache 分配器峰值，不是独立逐方法显存需求。 |
| `declared_logical_gradients` | 登记的逻辑梯度上限，不是物理 kernel 或已校准完整检测器反向当量。 |
| `actual_logical_gradients_from_method_diagnostics` | 有方法诊断时使用其报告值，否则采用 identity-output 计数或 declared-cap 回退；须检查 `logical_gradient_count_source`。 |
| `actual_source_row_forwards_from_method_diagnostics`／`actual_auxiliary_forward_from_method_diagnostics` | 支持时由方法报告；不可用计数保持 null。 |
| `declared_auxiliary_forward`／`declared_auxiliary_backward` | 声明值，不能代替实际物理工作。 |
| `mean_paired_ratio_to_vcsf` | 同源、同图像相对 VCSF 时间比的均值，不是显示精度下两个均值的比。 |

`logical_gradient_count_source` 区分 `method_diagnostics`、`identity_output` 与 `declared_cap`；`counter_scope` 明确排除物理 kernel 计账。clean reference、部分反向与变换视图不是免费工作，也不能自动折算为完整检测器当量。逻辑计数／上限不能证明物理计算相等。测量调用之间不清空分配器；Williams 顺序平衡方法位置与图内 carryover。warmup、加载及校验时间不属于攻击 API 时间。

原始 JSON 保留未舍入值。缺失统计在 JSON 中为 null，在 CSV／TeX 中为 `NR`，不能记成零。汇总 CSV 浮点值至少四位小数；TeX 时间与比值三位、显存两位。排序与配对比值必须由原始观察计算，不能使用已显示舍入值。

完整默认面板状态为 `complete_pending_independent_cost_audit`；小规模或子集面板为 `passed_cost_diagnostic_not_publishable`。两者都不是独立接受。记录中的 `formal_AP_eligible`、`cost_report_eligible`、`scientific_acceptance`、`project_wide_device_release_accepted` 和 `historical_result_inheritance` 保持 false。`evaluation_calls=0` 与 `payload_files=0` 是真实范围计数，不是缺失测量的占位值。

## 失败恢复

保留失败目录、`failure.json`、traceback 和各源已有的测量前缀。没有自动续跑、覆盖或目录合并。修复报告的原因后选择新输出叶目录，每次更换示例后缀。重试前核实实际进程归属；连接丢失不证明测量已停止。缺失观察或不完整源前缀不能接受为完整成本面板。
