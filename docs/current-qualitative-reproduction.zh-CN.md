# Current Main 离线定性图复现

[English](current-qualitative-reproduction.md)

在仓库根目录、固定的 `oda` 环境中运行。本入口仅在 CPU 上读取一个已保存的
`main_transfer` run，不加载检测器、不生成攻击、不重算 AP、不下载或搬运 payload，
也不修改原 run。

## 固定显示合同

维护范围仅为 COCO `val`，Faster R-CNN R50 source 到 Cascade R-CNN R50 target；
图片 ID 固定为 `785, 1503, 2157`，显示阈值 `0.50`，每张最多显示 `100` 个预测框。
第一行为 clean，之后十个攻击方法严格使用 `main_transfer` 的声明顺序，
包含正确顺序的 OSFD、SFIM-B、VCSF。预算从该协议的
`method_budget_profiles` 逐方法解析，未单独声明时使用主协议默认值。
当前 VCSF 使用 observed-cost，其他方法使用 compute-matched；
schema 2 request、图片与 manifest 均保留并披露混合预算。

不裁图、不改变预测坐标；显示时保留宽高比，同一图片在各方法行使用相同尺度。
预测框、类别和置信度来自已保存的 post-NMS 预测。这三张图是复用作者预选的
示例，不构成独立选择或独立确认，不为 VOC 编造固定示例。

## 无输入检查

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --help
conda run -n oda python -B experiments/current_saved_qualitative.py --plan-only
```

未提供 `--main-run` 时，plan-only 仅输出固定显示合同和
`display_contract_only_no_inputs_validated`。它只读声明式注册信息，
不验证 payload，不授予执行许可或 source-freeze 准入。

## 验证保存原件

保留原 clean 图片、COCO 标准 annotation、攻击像素、生成记录及完整无损
`predictions.json.gz`。原 run 的 `plan.json` 必须声明协议 `main_transfer`；
`summary.json` 必须明确记录 `max_images: null`、完成状态、零失败记录和零图片失败。
本次读取的每个评估和生成记录均必须覆盖完整的 5,000 张验证图片。

把以下 `saved_run` 替换为你的实际 main run 目录：

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --main-run outputs/experiments/main_transfer/saved_run --plan-only
```

有输入的 plan-only 在私有临时目录写临时 request，并调用完整 loader 验证预测
压缩包的所有字节与记录、哈希、覆盖范围、GT 和所选像素；不写目标输出目录。
验证后清理临时 request，返回的摘要不能作为正式渲染计划。
加载实际 main run 时仍遵守当前 registry 的源码身份校验。

仅使用下列原始布局：

```text
evaluations/coco/clean/cascade_rcnn_r50/metrics.json
evaluations/coco/faster_rcnn_r50/METHOD/default/cascade_rcnn_r50/metrics.json
attacks/coco/faster_rcnn_r50/METHOD/default/run.json
```

不读取聚合 records，不混合多个 run。缺失、重复、失败、部分完成、错误 dataset、
split、source、预算以及相互冲突的检查点或参数证据均会拒绝。
VCSF 必须匹配维护的 public 参数身份；方法同名或 AP 数值不能替代原件绑定。

## 输出 PNG

输出父目录必须存在；目标必须是原 main run 之外的新目录，已有空目录也会拒绝。

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --main-run outputs/experiments/main_transfer/saved_run --output outputs/current_qualitative_01
```

默认执行必须同时提供 `--main-run` 与 `--output`。输出包括绑定本次原件的
`request.json`、`request.sha256`、`source_inputs.json`，以及
`panel/panel.png`、各 cell PNG、`panel/manifest.json`。
不生成 PDF，不替换论文或手册 PDF，也不覆盖作者 canonical 图件。
输出根目录的 `manifest.json` 绑定 renderer manifest，明确保留新用户范围和
非作者认证边界。执行失败会保留 `failure.json` 与 request，不覆盖或删除失败记录。

可选 `--max-images 1` 仅缩短显示列，是 display diagnostic：
仍验证声明的三张输入及完整 5,000 张评估范围，不把部分指标包装成正式 AP。

## 证据边界与恢复

输出是新用户的保存输入复现，不是作者认证图，不授予科学接受或独立确认。
哈希只证明与本次消费原件一致，不证明独立审核。本入口记录原检查点身份，
但不加载检查点或重新授予其科学资格。

缺少压缩包或指标不完整时，保留原件，提供一个完整且正确注册的 run；
输出路径错误时使用新目录。不要合并时间戳 run、补造输入、把旧 VCSF 候选
冒充当前方法，或覆盖失败输出。保留失败尝试，后续重试另建新目录。

## Canonical LGP 数值源码身份

原 main run 还必须保留 `provenance.json`。Builder 检查其中 `source_manifest`
的条目数、完整性哈希和唯一相对路径，然后仅核对注册的 canonical LGP 执行器、
freeze 中的 `lgp_*.py` helper 家族及直接导入的数值公共 helper 是否与当前项目一致。
预期哈希来自 registry 引用的 source freeze，不在 Python 中复制哈希常量。
这是有限数值源码子集核对；历史报告源码允许不同，不要求整个历史工程完全相同。

原 provenance、registry、freeze 引用与匹配的数值源码 map 保留在
`source_inputs.json`、根 manifest 和 renderer 的输入证据中。
即使方法同名、参数哈希相同，缺失或旧 LGP 源码哈希仍会拒绝，不能仅换标签把
old raw LGP 图当成当前 canonical 实现。本核对不授予科学接受，也不声明整条
执行链等价。

VCSF 同样要求原 run provenance 与 registry 引用的最终 A10 选型协议中的
数值源码子集匹配。Builder 联合检查 map 完整性哈希、当前源码哈希、完整参数身份
和 generation 声明的实现；原选型协议引用及匹配的源码 map 与 LGP 证据一起保留。
同名、同参数但数值源码不同或缺失的 run 会拒绝；无关的历史报告源码仍允许不同。
