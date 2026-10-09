# Whole-ablation owner amendment, 2026-09-09

## Authority

At 2026-09-09 04:31:21 UTC the owner explicitly accepted the recommended key-gap
completion, requested a reasonable and finitely closing whole-ablation design,
and requested correction of the actual active goal. This amendment supplements
the [single-source amendment](vcsf-single-source-amendment-20260909.md), replacing
its underspecified ablation-completion condition, not its stop decision or other
deliverables. The [accepted stage contract](vcsf-whole-ablation-closure-20260909.md)
specifies the bounded design. Research execution remains gated and server-only.

## Effective objective

在服务器 ODA 环境中自动推进 VCSF 最终收束。执行并保持作者于2026-09-09确认的旧实验停止决定：停止当前双源 GPU 实验、配套 CPU 分析和旧边界守卫，不恢复或重启旧队列，不再把旧56/96组、942/1622项任务或69项对照的补齐作为完成条件。保留当前冻结 VCSF、所有已验收结果、原始计划、已完成与未完成运行证据作为不可变历史参照，不删除旧结果，不把主动停止伪称完整完成或实验技术失败。

后续整个消融研究固定 Faster R-CNN R50 单源，保持 COCO val2017 全5000张、初始探索 seed42 和完整十六目标面板及规范 BB Mean；不强制 Swin 补齐或双源关键复核。以 levels_two（backbone 与 neck 各两层、共四个特征项）作为下一阶段隔离工作基线，而非正式最终候选冻结。保留已观察到的 Swin 结果和两源分歧，单独解释，不平均掩盖、不选择性丢弃；单源消融结论限定在该设置，不推出跨源机制普适性。最终六源主实验范围不因消融单源化而缩减。

执行作者于2026-09-09进一步接受的完整消融收束方案，完整性针对整个消融体系而非只有尺度。按 docs/research/vcsf-whole-ablation-closure-20260909.md 的有界阶段推进：尺度处理方式与范围二维研究；尺度开关、neck/cross-stage、初始化有无的2×2×2核心因子；原始嵌套1/2/3层与阶段覆盖，并补齐neck两项、backbone两项、cross-stage各一项的等项数对照；detector/none/random初始化配合19/20个特征更新，明确区分等后续步数和等逻辑梯度上限；共享/独立位置/独立几何对应；energy/uniform与log-mean-exp/mean的2×2因子及单独空间支持对照；梯度去均值开关与零/基线动量的2×2交互及其余动量一维敏感性；最终配置归因、真实成本和冻结后五seed稳定性。保留所有结果，不把所有因素展开为全笛卡尔积，不要求每个组件都产生正收益。

在新命名空间中于每阶段执行前登记确切设置、基线与实现哈希、对照、选择规则、完整工作项和验收合同。使用完整十六目标、未舍入JSON、配对变化、真实成本和实现复杂度做阶段keep/change/reject判断；不依据不完整面板或四舍五入排名选择，不自动拼接各自赢家。单源结果与反复使用全验证集均须如实限定，不能称为独立确认。对同配置同seed的轨迹差异先明确复现性与历史复用边界；不能将固定状态数值相容等同于整条轨迹等价，不能静默改变确定性设置或扩展诊断矩阵。

发挥 Astra 的数学推导、反例分析、机制分析与文献研究能力，主动提出、检验和比较隔离候选，不局限于机械执行旧计划、微调参数或必须保留当前构造。允许按证据更新工作基线或提出替代构造；超出已批准有界单元的新研究需另行明确前瞻范围。每阶段保留选择时序和负结果。配置稳定后仅进行一次针对受影响必要对照的最终一致性补齐；若不支持预期贡献则缩小主张或否决候选，不无限调参重开矩阵。正式替换或冻结最终候选仍须作者确认。

只有精确同一配置、协议和实现且证据合格的旧结果可以复用。旧六层变体不得改名为四层变体，不混用不同基线形成伪控制变量消融表；不重跑未受影响的已验收基线、clean 或其他方法。完整消融的验收要求每项保留主张有同基线反事实、每项交互主张有完整因子对照、初始化预算与阶段项数混杂得到控制或明确披露、所有正式组完成5000张与十六目标面板、哈希/记录/独立验收与自动图表齐全。探索计划、静态检查、单组运行和数值相容均不能替代整套完成。

最终代码与配置确定并经作者冻结后，单独预注册并执行该方法的五seed稳定性（42、43、44、45、46），不沿用已撤回的多配置稳定性矩阵；只有完全同一配置且证据合格的seed42结果可复用。新组合不得继承旧结果，新seed不恢复同一数据集的独立性。仅更新受影响的VCSF六源主实验结果；最终收益、成本与限制须可复核，不能伪称机制证明、全局最优或跨源普适性。

保持大致同量级而非严格相等的物理计算量原则，单列梯度、参考前向、部分反向、变换视图、真实时间和内存。模型、数值及研究测试只在授权服务器 ODA 环境执行。继续完成论文、图表、引用、科技写作手册、双语开源指南、全项目单卡/双卡接口验收，以及不含大文件的服务器/本地同步。普通已授权步骤自动推进，长运行具名监控；验收失败、实质证据冲突、不可逆删除、正式候选替换或新增研究范围仅暂停相关动作并报告。持续保留跨窗口记录与历史追溯，不伪造实验、数学证明、最优性或验收结果。

## Execution partition

This owner decision authorizes preparation and completion of the stated key-gap
controls through the established admission gates; it does not promote a final
candidate, reopen retired queues, authorize deletion, or assert current numerical
completion. The active goal must retain its original accounting and active status.
Its update is verified by a separate live goal read, not this file alone.
