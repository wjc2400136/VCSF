# Finalization scope amendment

Owner decision, 2026-09-08. This record supersedes the earlier instruction to
execute the five-configuration, five-seed component block as part of the
current layered ablation programme. It does not erase the original registered
plan, its source identity, completed observations or failed evidence.

## Effective objective

在服务器 ODA 环境中自动推进 VCSF 最终收束。保留当前冻结 VCSF、所有已验收结果、原始计划及运行证据作为不可变参照；完成已登记的 COCO-5000、seed 42 单 seed 简化、消融和敏感性对照并保留全部结果，撤回尚未开始的旧五配置 40 组多 seed 组件对照，不再把它作为本轮消融的完成条件。结合两源及完整目标面板、真实计算代价、实现复杂度和文献证据，发挥 Astra 的数学推导、反例分析、机制分析与自主判断能力，主动提出、检验和比较隔离候选，不把工作限制为机械执行旧计划、微调参数或必须保留当前构造；允许提出更换最终候选。两源分歧必须单独解释，不能用平均值掩盖退化，也不能由单源胜出直接推出统一设置。最终选型须有完整、可复核的证据和明确的收益与代价；正式替换或冻结最终候选仍须作者确认。仅在最终方法的代码与配置确定并冻结后，单独预注册并执行该方法的五 seed 稳定性分析；只有完全同一配置且证据合格的 seed 42 结果可以复用，新组合不得继承旧结果，同一数据集上的新 seed 不构成独立数据确认。保留大致同量级而非严格相等的物理计算量原则，必要时刷新新候选受影响的结果，不重跑未受影响的已验收基线。更新研究、统计、选型与确认边界，不将原 96 组、1622 项任务或 69 项对照误报为完整完成。继续完成论文、图表、引用、科技写作手册、双语开源指南和不含大文件的服务器/本地同步。普通已授权步骤自动推进，长运行具名监控，在原运行根外实施可审计的执行边界控制；仅在验收失败、实质证据冲突、不可逆删除、正式候选替换或新增研究范围时暂停相关动作并报告。持续保留跨窗口记录与历史追溯，不伪造实验、数学证明、最优性或验收结果。

## Execution partition

The unchanged original GPU plan contains 96 source groups: ten qualified reused
groups and 86 newly scheduled groups. The amendment permits all 56 seed-42
groups, including the ten reused groups and 46 new groups. It withdraws groups
57 through 96, the forty additional-seed component groups. The last permitted
GPU stage is `initialization_specificity`, with groups 53 through 56. These
counts must be checked against the bound live plan before any control action.

The original CPU grid remains immutable. Its permitted prefix is 896 target
cells plus 46 new-group bindings: 942 tasks, including one qualified completed
cell reuse. The original full-grid and 69-contrast acceptance conditions are
not rewritten into success for this prefix. A revised analysis family and
selection record must disclose the timing of this amendment after descriptive
interim observations. No significance or equivalence claim follows merely from
removing the seed-component block.

The external analysis adapter is now implemented and server-ODA validated;
see [implementation and evidence](vcsf-seed42-analysis-adapter-20260909.md).
It retains 61 original seed-42 contrasts and reports three source-labelled
views in one 183-label family. Its tests are not completion of the real
942-task input prefix, final analysis or scientific selection.

An external boundary guard was armed and independently reobserved after SSH
logout on 2026-09-08; see the separate
[deployment and verification record](vcsf-seed42-boundary-guard-20260908.md).
The production boundary hold itself is not yet observed. The guard must not
modify active source, plans or result JSON,
stop an allowed incomplete group, delete evidence, restart a root or silently
resume a held coordinator. Its implementation and observed deployment need a
separate receipt. A held process does not imply released execution reservations.

## Candidate judgment

The objective is an effective, defensible final method, not a requirement that
the current `full` configuration survive. Retain the distinction between an
unchanged scientific reference and a replaceable research candidate. Mathematical
work should establish exact definitions, gradient structure, invariances,
counterexamples and testable predictions; elegant notation alone is not method
evidence or novelty acceptance.

Prefer a single documented rule across sources. First compare each source's
paired change against its own frozen reference, report target-level variation
and the existing declared aggregate, and inspect actual cost. A shared-target
descriptive check may diagnose differences between source-excluded target sets;
it must not silently replace the registered headline metric. Do not impose a
new minimax score, source-specific switch, concession threshold or significance
criterion after observing intermediate rankings without an explicit recorded
decision. The existing maximum concession is an upper limit, not an automatic
selection target or proof of equivalence.

If a proposed setting helps one source and harms the other, distinguish a real
mechanism tradeoff from uncertainty, implementation mismatch and scale-dependent
gradient effects. Use already authorized evidence and bounded diagnostics first.
Do not average the disagreement away, and do not automatically combine settings
that won separate one-factor comparisons. A new combination has a new identity
and needs its own applicable validation and affected-result refresh.

After a final candidate is selected and the owner confirms its freeze, register
five-seed stability for that exact method. Under the current two-source scope,
this means eight new groups if its exact seed-42 pair is qualified for reuse,
otherwise ten new groups. These are conditional counts, not authorization to
launch a not-yet-defined candidate or expand the source/data scope.
