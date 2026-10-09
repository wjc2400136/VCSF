# 已保存报告重导出

[English](saved-report-export.md)

此接口只用 CPU 从既有 records 修复报告范围和计算单位标签，不评估预测、不读取图像或加载 detector，
不改变攻击、不测量成本，也不作科学结果接受。

## 前提

使用固定 `oda` 环境（Python 3.8.20），在仓库根目录执行。输入是包含 `records.json` 的不可变 saved run。
保留同目录的 `summary.json`、`plan.json`；final-background 还使用 `background_plan.json`。
这些文件提供已知的限图值、split 和诊断身份。缺失范围证据保持 unknown。

current preprocessing 还需保留 `input_binding.json`、绑定的 prepared-view 小型元数据和原生成元数据。
报告按 defense 分别输出到 `reports/defense/` 和 `evaluations/defense/`，不会合并不同 defense 的同一目标。
上述接口同时支持 oblivious 和 adaptive saved run；输入 `records.json` 原样字节复制，
仅为新派生报告行补充已有元数据。

preprocessing transfer 表直接复用维护中的 runtime reporter，保留 defense/protocol 专属
label 和 100-times-raw 比例说明。未定义 COCO `-1` 仅在非修改性的表格、分析、图表展示视图
中显示为 NR，且不参与排名，不变成零或负 AP；有效零仍为数值。原 `records.json` 字节和
`presentation_records.json` 原始指标值（包括 `-1`）保持不变。
标准逐目标 COCO 原始摘要保留既有十二行约定。

通用 transfer 表也会在排名和计算 BB Mean 前排除未定义的官方指标。此展示视图不改变
已保存原始 JSON、有效零值或源匹配 white-box 单元；black-box 子集为空时保留 NR。

current-radius run 必须保留原始 `radius_plan.json`。导出器复制并绑定此小 JSON 和
`records.json`，然后仅调用维护中的 `current_radius.write_reports` 源-半径专用 reporter。
此路径禁止 generic ablation 表、标量图和 `--native-counts`。缺少 radius selection 时直接
报错，不回退到折叠源维度的投影。训练状态表的 label 按记录中的 `fullval`、
`retained500` 角色区分，诊断也不例外；label 本身不授权完整验证集结论。

公共 transfer 表仅在需要时缩小：先限制宽度不超过 `textwidth`，再保留既有
`0.88 textheight` 总高度上限。小面板保持自然尺寸。此变化仅涉及 TeX 排版容器，
不改变数值 tabular 正文或原始结果。

隔离报告修复时可用 `--registry-root` 指定未变的冻结项目作为只读注册表来源，保留原 identity guard。
这不使修复目录获得 detector 执行资格，也不更新 source freeze。
整合及可能需要的非数值来源重新绑定属于后续独立决策。

## 执行

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 conda run -n oda python -B experiments/reexport_saved_reports.py --run qualification/saved-diagnostic --output qualification/derived-scope01 --plots --native-counts
```

将 `qualification/saved-diagnostic` 替换为实际已保存运行目录，输出必须选择尚不存在的新目录。
隔离目录可追加 `--registry-root ../frozen-project`，替换为实际冻结注册表根目录。

`--plots` 仅从标量 records 在 CPU 上渲染 PNG/PDF，不读取攻击轨迹或预测、图像 payload。
`--native-counts` 只读输入运行所属攻击组的既有 `native_cost.jsonl`，逐图复制原计数，不重新测量或校准物理成本。
最小 text/CSV/TeX 重导出可省略这两个选项。
preprocessing 仅通过绑定的 input/view 来源引用评估运行外的生成计数，不创建新测量。

## 产出与含义

- `reports/dataset/`：transfer/ablation 表、标量分析和 runtime 计数说明。
- `reports/background/`：final-background aggregate 和逐 control 报告。
- `evaluations/`：逐目标 text/CSV/TeX 摘要，保留全部十二个原始指标。
- `reports/dataset/native_counts/`：可选的逐图原计数 JSON/CSV/TeX。
- `records.json`：与输入原件字节一致的 records 副本。
- `report_export.json`：输入文件哈希和 label-only 范围说明。

限图或明确诊断输出显示 `PARTIAL DIAGNOSTIC`，并注明 actual dataset、split、sample N 和 max_images。
N 是成功评估图像数，不是请求上限。不同 cell 的 N 不一致或缺失时，不替换成共同全量或零。
未知范围标为 `SCOPE UNKNOWN`；明确非诊断的全 split 范围维持既有范围展示。

evaluator 可接收只用于展示的 selection context。current preprocessing 传入已声明的
protocol/selection：完整 registered retained-500 比较人群不是 smoke；max-images 1/6
仍是诊断。未声明身份的 image-ID 子集保持 unknown，不因小于全数据集就判为诊断。
这些参数不改变选择的 IDs 或指标。

runtime 表将 declared budget/profile 与 observed actual mean count 分开，单位来自注册的 accounting definition。
LGP early-stop 实际计数不补成声明的 20。VCSF logical update、backbone/neck partial backward、
clean-reference row 和 adversarial differentiable view 不能互换成 complete-detector BE。
auxiliary zero 不代表 reference/partial 路径免费；缺失实际计数和 calibrated BE 保留 NR。

preprocessing CSV 分开保存 `generation_cost_scope`、原生成 N、run 哈希和成功评估 N。
`reused_original_generation` 不表示新执行了自适应生成；
`current_adaptive_generation` 表示 saved adaptive run 当时实际生成。
原生成人群的均值不会改称较小评估子集的计数。native logical updates、detector/partial backwards、
reference rows 和 differentiable views 分列保存；缺失均值或 calibrated BE 仍为 NR。

paired-cost CSV/TeX 显示实际测量 N、绑定 plan 的 cap/诊断身份和逐方法 recorded-count 单位。
training-state CSV/TeX 接收已声明的展示 context，分别注明 fullval 与 retained500 的实际 N。
fullval 六图诊断可能只交集到 retained 一图，二者均不得标成 500/5000 图结果。
这些展示修改不改变校准、训练、调度或指标计算。

`current_paired_cost` 的原输入类型是 `plan.json` 与 `summary.json`，
以及存在的 `input_binding.json`/`measurements.json`，不是 `records.json`。
使用同一命令并省略 `--plots`、`--native-counts`；接口字节复制原输入，不编造 records 文件。
saved counter 来源为 `declared_cap` 或 `unavailable` 时，仍展示其原元数据，
但 observed actual mean 保持 NR。`current_training_state_transfer` 则读取原 records，
分 fullval/retained500 生成表，不重放预测、不访问 detector。

原 native `diagnostic_only=null` 在新 native JSON 中仍为 null，CSV 中为 unknown，
与报告层已知的 global max-image 诊断身份分开保存。
不改变原 JSON 数值、十二指标顺序、canonical 模型顺序、white-box/held-out 标记或既有数值展示精度。

成功条件为退出码零、`[OUTPUT]` 指向新目录，以及新旧 `records.json` 的 SHA-256 一致。
使用前核对新标签和输入绑定。已有输出不会覆盖；失败产出也保留，解决输入问题后换一个新目录重试。
输出 records 使用第一次哈希绑定的输入字节；封存 `report_export.json` 前重新核验全部小型绑定原件。
检测到输入变动时导出失败、不产生封存 receipt，并保留该 derived 目录。
不要合并不同运行或 split，也不要将诊断解释为正式全 split mAP。
