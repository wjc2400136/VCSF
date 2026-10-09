# 第三方声明

[English](THIRD_PARTY_NOTICES.md)

本分发保留完整的当前维护源码包，项目自有源码按[许可证范围](docs/licensing/license-scope.zh-CN.md)采用 GPL-3.0-only。第三方条款适用于相应材料，收录许可原文不等于为每个第三方部分重新授权。科学实现与配置保持不变；本声明不是独立作者身份或无条件法律许可的保证。

## NumbOD

`src/lgp/attacks/numbod.py` 包含已记录的 NumbOD 本地适配，包括本地表达的 Haar 切片与检测器集成，而非逐字节复制上游文件。所保留的适配材料对应[原始 MIT 声明](docs/third_party/licenses/NumbOD-MIT.txt)：Copyright (c) 2024 CGCL-codes。

来源：[NumbOD，提交 0b22cbfb020ea20730fafd23e8e4ebe33b3adda7](https://github.com/CGCL-codes/NumbOD/tree/0b22cbfb020ea20730fafd23e8e4ebe33b3adda7)，主要对应 `attack.py` 与 `DWT.py`。

## LGP 与源自 OpenMMLab 的部分

已记录的署名范围是以下本地文件中的派生部分：`src/lgp/attacks/lgp.py`，以及同目录的 `lgp_assignment.py`、`lgp_fidelity.py`、`lgp_heatmap.py`、`lgp_query.py`、`lgp_roi.py`、`lgp_source.py`、`lgp_vfnet.py` 和 `lgp_yolo.py`。这些文件将登记的 LGP 公式、源模型接口适配与本地基准执行安排结合；这不表示每一行都属于第三方代码。

登记采用的狭义来源范围是 [LGP，提交 fce86da91f2dc4a69cc69751806f0caae80e51a3 的 `mmdet/` 子树](https://github.com/liguopeng0923/LGP/tree/fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet)，包括 `adv/attacks/LGP.py`、`adv/utils/FBS.py` 和 `adv/models/losses/logit_loss_adv.py`。[原始 Apache-2.0 文本](docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt) 保留 OpenMMLab 2018-2023 署名。该范围不意味着上游仓库根目录或整个本地项目采用 Apache 许可证；相应本地文件附有显著修改通知。

## OSFD

`src/lgp/attacks/osfd.py` 保留了已记录的 OSFD 随机旋转、Gaussian blur 组件及本地检测器集成适配。来源是 [OSFD，提交 3744caf69e60b46012a6895c095f18c33db491a9](https://github.com/wakuwu/OSFD/tree/3744caf69e60b46012a6895c095f18c33db491a9) 的 `attack/base/RRB.py` 和 `attack/ours/OSFD.py`。

本副本收录[未经修改的上游 GPL 第 3 版文本](docs/third_party/licenses/GPL-3.0-upstream.txt)，不从其示例附录或许可证文档本身的版权推断软件允许使用后续版本或推断软件权利人。本地文件附有带日期的修改通知。集成源码包按 GPL 第 3 版分发，项目自有代码的授权为 GPL-3.0-only；已记录的上游版本范围保持不变。

## 本地重新表达与共享辅助代码

对 `src/lgp/attacks/tog.py` 与 `afog.py` 的有界来源检查发现，它们是参考发布实现的本地数学重新表达，并非逐字节复制整文件。仅凭 port 标签不能证明复制了受保护的表达；这一观察也不证明独立作者身份或法律许可已闭合。论文及固定版本实现出处仍保留在 `configs/attacks/reference.yaml`；本次没有排除或改写任何方法。

`src/lgp/attacks/common.py`、`src/lgp/adapters/openmmlab.py` 与 `src/lgp/attacks/lgp_semantic.py` 中所有已检查部分的表达来源尚未完全建立。本声明不会为这些共享文件虚构整文件上游许可证或版权。限定范围及源码哈希保留在[来源记录](docs/third_party/origin-scope.json)中。

## 外部安装的框架依赖

本包引用固定框架版本，但不捆绑框架源码、配置树、wheel、模型权重或数据集。MMDetection 3.0.0 对应[上游 Apache-2.0 许可证](https://github.com/open-mmlab/mmdetection/blob/v3.0.0/LICENSE)；MMYOLO 0.6.0 提供[上游 GPL 第 3 版文本](https://github.com/open-mmlab/mmyolo/blob/v0.6.0/LICENSE)，与本包收录的 GPL 文本逐字节相同。仅将依赖分开安装，不能解决集成组合程序的许可问题。本次声明准备不改变冻结环境。

## 修改日期与分发范围

文件级声明的添加日期为 2026-10-09。此前的基准适配早于声明日期；不虚构未记录的历史适配日期。本次添加改变注释与文件哈希，不改变 Python 语法树或数值行为。原科学运行身份和历史逐字节固定协议仍对应原始文件。

第三方许可原文未经修改。项目自有代码的授权单独列于[许可证范围](docs/licensing/license-scope.zh-CN.md)，不从代码许可证推断数据集或照片许可。源码包不包含论文或照片载荷。
