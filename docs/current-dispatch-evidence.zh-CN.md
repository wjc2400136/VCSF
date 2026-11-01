# 主实验调度证据

[English](current-dispatch-evidence.md) | [简体中文](current-dispatch-evidence.zh-CN.md)

在公开源码根使用固定 `oda`，当前命令和输入检查见
[主指南](current-public-reproduction.zh-CN.md)。公开面仅为 COCO/VOC 主比较与
必要准备，早期更宽路径清单是历史记录，不是额外公开入口。

## 检查同一任务集

```bash
python experiments/main_transfer.py --plan-only --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0
python experiments/main_transfer.py --plan-only --datasets coco --methods vcsf --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --max-images 1 --devices cuda:0,cuda:1
```

这是无模型计划，不是 GPU 诊断。新叶目录包含计划任务、分配与报告，指标保持
NR，实际 N 未知。设备选择不改任务、输入、种子、配置与顺序；双卡不合并显存，
实际准入应拒绝显式不可用设备。

## 实际执行记录

Linux 双卡运行的 `dispatch_journal/STAGE/` 保留及时刷盘的协调器/worker
日志，绑定 PID/启动 ticks/父进程、设备/任务以及已有 CUDA context 对应的
逻辑设备 UUID 和实际 NVML PID。不额外创建 context、不改 RNG、不证明独占。
早期未观测身份保持 pending/null；非空 worker 必须完成所需登记才可结束。
清空活动 worker 列表后，最终协调器仍保留退出和收束证据。

这些记录区分计划、实际归属、产物完成与清理，不证明效果或独立科学接受。
不补造缺失登记，不从计划反推已观测身份。

主实验可选叠框保留评估器预测；其历史有界组件/布局检查不证明所有拥挤图像，
也不是另一个公开定性实验。

## 失败与证据限制

保留完整叶目录、原终态/状态、日志和完成前缀。旧 running 快照或断连不授权
重复任务。运行 owner 先核对实际终态/收束，再把明确缺失子集放新叶目录，
不覆盖、不合并、不自动续跑。

源码审阅、CPU mock 和计划只证明相应检查。历史有界主实验证据仍有边界；
本次源码范围整理不声称新跑完整 GPU 矩阵、覆盖每个 CUDA 分支或科学验收。
