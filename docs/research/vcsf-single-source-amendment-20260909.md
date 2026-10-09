# VCSF single-source working-baseline amendment, 2026-09-09

## Authority and supersession

The owner explicitly accepted the single-source recommendation and requested:
"确定按你推荐来，请直接停止现在的实验，修正现在‘进行中的目标’".
This supersedes the requirement to finish the remaining old two-source seed-42
queue in the [September 8 amendment](vcsf-finalization-scope-amendment-20260908.md).
It does not promote a final method or invalidate completed historical evidence.

The fixed source for all subsequent ablation work is `faster_rcnn_r50`.
`levels_two`, retaining two backbone and two neck levels, is the next isolated
working baseline. Swin is not a mandatory second ablation source or mandatory
key-check source. Preserve its existing observations and the backbone-only
direction conflict. The final six-source main transfer panel is unchanged.

## Effective objective

在服务器 ODA 环境中自动推进 VCSF 最终收束。执行并保持作者于2026-09-09确认的旧实验停止决定：停止当前双源 GPU 实验、配套 CPU 分析和旧边界守卫，不恢复或重启旧队列，不再把旧56/96组、942/1622项任务或69项对照的补齐作为完成条件。保留当前冻结 VCSF、所有已验收结果、原始计划、已完成与未完成运行证据作为不可变历史参照，不删除旧结果，不把主动停止伪称完整完成或实验技术失败。

后续整个消融研究固定 Faster R-CNN R50 单源，保持 COCO val2017 全5000张、初始探索 seed42 和完整十六目标面板及规范 BB Mean；不强制 Swin 补齐或双源关键复核。以 levels_two（backbone 与 neck 各两层、共四个特征项）作为下一阶段隔离工作基线，而非正式最终候选冻结。保留已观察到的 Swin 结果和两源分歧，单独解释，不平均掩盖、不选择性丢弃；单源消融结论限定在该设置，不推出跨源机制普适性。最终六源主实验范围不因消融单源化而缩减。

在新命名空间中预先登记每阶段设置、对照与选择规则，采用分阶段、有判别力的研究，而非每次替换后重跑整套消融。优先研究尺度处理方式与尺度范围的二维交互；已有机制交互的组件采用适当因子设计，其余参数先一维，不把所有因素展开为全笛卡尔积。发挥 Astra 的数学推导、反例分析、机制分析与文献研究能力，主动提出、检验和比较隔离候选，不局限于机械执行旧计划、微调参数或必须保留当前构造。允许根据证据更新工作基线或提出替代构造；新增研究范围仍须明确登记并在超出已授权范围时报告。完整保留每阶段结论和选择时序，明确全验证集选择不构成独立数据确认。

只有精确同一配置、协议和实现且证据合格的旧结果可以复用。旧六层变体不得改名为四层变体，不混用不同基线形成伪控制变量消融表；不重跑未受影响的已验收基线、clean 或其他方法。最终配置稳定后，仅补齐受影响的核心消融、主实验和所需稳定性证据。结合完整目标面板、真实计算代价、实现复杂度和文献证据给出可复核的收益与代价；正式替换或冻结最终候选仍须作者确认。仅在最终代码与配置确定并冻结后，单独预注册并执行该方法的五 seed 稳定性分析，不沿用已撤回的多配置稳定性矩阵；只有完全同一配置且证据合格的 seed42 结果可复用，新组合不得继承旧结果，新 seed 不恢复同一数据集的独立性。

保持大致同量级而非严格相等的物理计算量原则，单列梯度、参考前向、部分反向、变换视图、真实时间和内存。模型、数值及研究测试只在授权服务器 ODA 环境执行。继续完成论文、图表、引用、科技写作手册、双语开源指南、全项目单卡/双卡接口验收，以及不含大文件的服务器/本地同步。普通已授权步骤自动推进，长运行具名监控；验收失败、实质证据冲突、不可逆删除、正式候选替换或新增研究范围仅暂停相关动作并报告。持续保留跨窗口记录与历史追溯，不伪造实验、数学证明、最优性或验收结果。

## Execution partition

- Retired original GPU execution: `vcsf_layered_efficacy_execution/20260906T071152Z`.
- Retired original CPU analysis: `vcsf_cpu_analysis/20260906T095441Z`.
- Original GPU plan SHA-256 remains
  `d29a624cba32f8e07e4391ddc617194054b77a0d8844e29f1de317a3a96f0611`.
- External stop evidence: `outputs/source_recovery/owner_stop_20260909T023917Z/`
  in the separate device-validation tree; local mirror uses `outputs/staging/`.
- Stop receipt SHA-256:
  `4e9c85fee67f2c6e16a9e7a053048e505cd04814717f15ad0cbeabc2fd848219`.
- Ten owned processes were verified by identity and stopped: two GPU workers,
  three active CPU workers, two coordinators, two supervisors and the guard.
- A separate observation at 02:40:22 UTC found no live root-bound process;
  `nvidia-smi` returned no compute process, with exit code zero.
- The retained last CPU snapshot has 589 completed tasks, 563 completed cells,
  26 bindings, 28 closed source groups, and zero recorded failures. These are
  partial execution counters, not completed statistical-family acceptance.
- The retained GPU snapshot has two interrupted `uncentered_gradient`
  generations at 4275 and 4800 of 5000 images, with no completed target panel
  for those interrupted groups. They are not formal efficacy rows.

Root state files intentionally retain their last `running` snapshots. Do not
edit them to manufacture a clean terminal state. The external stop receipt and
actual process observations establish retirement. Keep locks and partial
payloads; no automatic resume, merging, pruning or new invocation in old roots.

## Next action and remaining gates

Prepare the single-source staged registry and exact `levels_two` descriptor in
an isolated namespace, validate its source binding and direct entry in ODA,
then execute only the registered, authorized discriminative stage. No new
experiment is launched merely by this amendment. Historical mathematical and
statistics checks remain evidence only for their own exact implementation and
scope. Final scientific selection, author confirmation, final-config stability,
affected-result refresh, release qualification and final writing remain open.

The interrupted source-gate repair remains a separate unfinished implementation
task: its two local files have not yet been ODA-tested or deployed. It does not
justify changing or reviving the retired experiment.
