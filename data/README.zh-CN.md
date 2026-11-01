# COCO 与 VOC 数据

[English](README.md) | [简体中文](README.zh-CN.md) | [项目首页](../README.zh-CN.md)

按[环境指南](../docs/environment.zh-CN.md)准备固定 `oda`，所有命令在公开源码根
运行。数据声明保留在 [coco.yaml](../configs/datasets/coco.yaml)、
[voc.yaml](../configs/datasets/voc.yaml)。

| 划分 | 图像数 | 用途 / 运行路径 |
| --- | --- | --- |
| COCO `val` | 5,000 | 正式主实验，`data/coco/val2017/` |
| COCO `dev` | 1,000 | 冻结开发选择，`data/coco/dev2017/`，不是正式主实验 |
| VOC `val` | 4,952 | VOC2007 test，`data/VOCdevkit/VOC2007/JPEGImages/` |
| VOC `train` | 16,551 | VOC07+12 trainval，用于 VOC 检测器准备 |

## 下载、解压、准备

以下命令下载/写入文件。已有合格输入应保留清单，仅修复明确缺失/损坏项，
不覆盖合格树。新主机网络与空间满足后执行：

```bash
python -m lgp data download --dataset coco
python -m lgp data extract --dataset coco
python -m lgp data prepare --dataset coco
python -m lgp data download --dataset voc
python -m lgp data extract --dataset voc
python -m lgp data prepare --dataset voc
```

COCO 默认下载 val2017 和 train/val 标注，不下载全部 train2017 图像。准备会
写冻结 dev 选择/标注，并可能下载缺失的开发图像。VOC 使用注册的 2007 trainval、
2007 test、2012 trainval 归档，生成 `data/VOCdevkit/annotations/` 下的
`voc0712_trainval.json`、`voc2007_test.json`。实验不读 `data/raw/`。

主比较不需要可选 `--full` 载荷。公开 CLI 不开放 BDD100K 或旧 `all` 数据集
选择。本地学院工程用显式资源计划复用外层数据，不复制、不改公开注册表根。

## 深度验证

```bash
python -m lgp data validate --dataset coco --deep
python -m lgp data validate --dataset voc --deep
```

核对计数/类别、引用图像、重复/越界路径及训练验证交叠。`valid_with_warnings`
也要审阅警告，准备完成或零退出不自动合格。再做选定[权重/doctor 检查](../checkpoints/README.zh-CN.md)。

失败后保留报告、修复具体注册输入，不改读归档、不复制解压树。不能在正式
划分上选择检查点或超参；历史单独声明的全验证集回顾选择不是独立确认。

## 保留

只留一套被引用的运行图像树。VOC 训练数据保留至十六个必要检测器全部合格。
公开版不提供 `data minimize`，不要调用会扫描退役数据集的旧最小化器。
本指南不授权删除数据；进一步删除需明确范围、核实目标及备份/恢复决定。

继续[检查点](../checkpoints/README.zh-CN.md)、
[主实验](../docs/current-public-reproduction.zh-CN.md)和[输出](../outputs/README.zh-CN.md)。
