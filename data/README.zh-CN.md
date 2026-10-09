# COCO 与 VOC 数据

[English](README.md) | [简体中文](README.zh-CN.md) | [项目首页](../README.zh-CN.md)

在仓库根目录、锁定的 `oda` 环境中执行命令。
真实数据注册文件是 [coco.yaml](../configs/datasets/coco.yaml) 与
[voc.yaml](../configs/datasets/voc.yaml)，由 [manager.py](../src/lgp/data/manager.py)
使用；不存在合并的 `configs/datasets.yaml`。

| 数据集／Split | 图像数 | 用途与位置 |
| --- | --- | --- |
| COCO `val` | 5,000 | 正式 val2017；`data/coco/val2017/` |
| COCO `dev` | 1,000 | 固定 train2017 子集，仅开发；`data/coco/dev2017/` |
| VOC `val` | 4,952 | 正式 VOC2007 test；`data/VOCdevkit/VOC2007/JPEGImages/` |
| VOC `train` | 16,551 | VOC07+12 trainval，检测器微调；`data/VOCdevkit/` |

不得在正式 split 上选择检查点。经授权的 COCO full-val 回顾性配置选择必须
标为选择分析，不能称独立确认。

## 下载、解压与准备

以下命令会下载／写入数据，需要网络和足够磁盘空间。
已有核验数据应复用其已验证 manifest，不默认覆盖。
每条命令都必须指定数据集；省略参数／默认 `all` 还会选择历史支线。

```bash
conda activate oda
python -m lgp data download --dataset coco
python -m lgp data extract --dataset coco
python -m lgp data prepare --dataset coco
python -m lgp data download --dataset voc
python -m lgp data extract --dataset voc
python -m lgp data prepare --dataset voc
```

COCO 默认下载 val2017 和 train／val 标注，不下载完整 train2017 图像归档。
prepare 生成固定 dev 标注、选择 manifest 和对应的 train 来源图像，
必要时会下载缺失 dev 图像。VOC 下载三个注册归档：2007 trainval、
2007 test、2012 trainval。prepare 写入
`data/VOCdevkit/annotations/voc0712_trainval.json` 和 `voc2007_test.json`。
实验只读取准备后的标注／图像，不读取 `data/raw/`；raw 归档只是临时准备输入。

COCO 的 `download --full` 会额外下载可选训练载荷，
`validate --full` 会要求可选 split；默认主复现不需要它们。
另有注册训练工作流依赖的 train 载荷必须保留，不能为了查看接口默认下载
或重训受害模型对。

## 实际运行前校验

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
```

检查报告而不只看退出码：标注图像数、类别数、引用图像存在性、重复／越界路径、
train／val 重叠等均须通过，并审查库存警告。`valid_with_warnings` 需要审查，
不自动表示完整就绪。prepare 成功不等于 deep validation 通过。
正式执行前还需完成[检查点指南](../checkpoints/README.zh-CN.md)中的限定数据集
doctor 与权重核验。

失败时保留报告，根据注册来源修复具体缺失／损坏归档、标注或引用图像。
不能改让实验读取 raw，也不能再复制重复解压树。
确认哈希后仅重查受影响的数据项，不将缺失证据记为零。

## 存储与删除边界

每个数据集保留一份被引用的运行图像树，不手动复制重复归档树或未引用图片。
十六个所需 VOC 检测器检查点全部完成训练与验证前，不得删除 VOC 训练输入。

以下可选存储计划检查**不属于 COCO／VOC 准备流程**：

```bash
python -m lgp data minimize
```

不带 `--apply` 不会删除，但当前实现仍调用 COCO dev prepare，
并扫描所有注册数据集，包括历史支线。它没有数据集筛选参数，也不是严格只读。
仅在全部注册依赖可用、且允许这些准备副作用时使用；不能用它绕过
仅限 COCO／VOC 的数据访问边界。
任何删除（包括添加 `--apply`）都须明确批准、验证目标、
确定备份／恢复方案并保护训练与输出范围，不能仅凭本指南删除数据。

下一步见[检查点](../checkpoints/README.zh-CN.md)、
[主复现](../docs/current-public-reproduction.zh-CN.md)及[输出](../outputs/README.zh-CN.md)。
