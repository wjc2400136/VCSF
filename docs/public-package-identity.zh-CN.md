# 公共源码分发身份

[English](public-package-identity.md) | [简体中文](public-package-identity.zh-CN.md)

`public-package.json` 绑定全部随包源码、配置、直接入口、辅助代码、公共指南和
轻量 provenance。单独版本化的 `vcsf-a10-public-package-freeze-v1.json`
绑定精确 manifest 及源码映射。这两份控制文件明确列出，不作递归自哈希。
它们不构成许可证或科学接受。

守卫检查普通文件、POSIX 相对路径、哈希、随包文件完整枚举、本地 Python 导入、
运行时/AST 辅助文件依赖、原九个数值哈希、全部原配置字节及六份原 provenance。
遗漏依赖、文件变更、符号链接或额外未绑定可执行文件均拒绝。
运行数据和结果不属于源码身份，仍由各自实验合同核验。

保留的 `vcsf-a10-public-source-freeze.json` 描述历史作者全树。
裁剪包不满足旧源码映射。新守卫明确返回不同的源码闭包身份并保留旧 freeze 摘要，
不改写历史哈希，也不宣称旧结果接受了整套新编排。

非数值的加载配置核验、只读观测和物理设备解析辅助代码按原 AST 原样迁移。
Linux ODA 解释器检查解析当前环境，不嵌入私有服务器路径。
Registry、factory、攻击数学、九个数值文件、A10 参数、MRO 及原选型证据均未改动。

有界 CPU 导入、CLI 帮助和无 GPU 计划只证明对应接口，
不证明全部 CUDA 分支、运行载荷接受、分发权利或科学结果。
未公开作者证据留在包外，历史辅助代码不代表恢复退休协议。
维护实验范围见[主指南](current-public-reproduction.zh-CN.md)。

原树中已缺少一份由历史 final-budget/training-state refresh 协议引用的 amendment。manifest 明确列出这两处历史元数据引用，不视为当前运行输入。本源码分发没有伪造、修复或接受该证据，也不认证这两条历史 refresh 路径可执行。

当前分发采用 `lgp_public_source_distribution_v2`。守卫仍接受历史的仅候选 v1
schema，不提升其权利或科学状态。freeze 文件名沿用，但在本新副本中重新绑定
schema、源码映射和许可范围。v2 的 `release` 只指当前软件源码分发；
`science` 与 `formal` 仍为 false。许可范围、原上游许可证字节及双语第三方声明
均被明确绑定。原 A10 科学身份保留原状态和数值哈希。
