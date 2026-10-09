# 当前调度证据

[English](current-dispatch-evidence.md) | [简体中文](current-dispatch-evidence.zh-CN.md)

在仓库根目录使用固定的 `oda` 环境。完整命令、输入前提及检测器显存要求仍见
[半径复现指南](current-vcsf-radius-reproduction.zh-CN.md)与
[训练状态指南](current-training-state-reproduction.zh-CN.md)。设备选择不改变科学作业集合。
两卡不合并显存，明确请求但不可用的设备须报错。

以下命令在不访问数据、权重或 CUDA 的情况下检查同一组有界半径作业：

```bash
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0 --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --epsilons 4/255 --max-images 1 --output outputs/experiments/current_vcsf_radius/evidence-plan-one
python experiments/current_vcsf_radius.py --plan-only --devices cuda:0,cuda:1 --sources faster_rcnn_r50,mask_rcnn_swin_t --targets faster_rcnn_r50,mask_rcnn_swin_t --epsilons 4/255 --max-images 1 --output outputs/experiments/current_vcsf_radius/evidence-plan-two
```

每个新目录应有四个计划攻击单元格、`reports/radius/4_255/` 下十二份保留 source 的 TeX 报告
及 `transfer_records.csv`。各报告附带实际数据集、划分、观测样本 N 和请求的图像上限。
上限不是已评估 N 的证据：N 不可得时保持 unknown，未运行指标保持 `NR`。
专用半径 caller 不再同时生成会折叠 source 的通用消融、analysis 或热力图投影；
其他 caller 保持原有默认投影。

Linux 上的通用双卡执行在 `dispatch_journal/STAGE/` 保存追加式、已 flush 的协调进程登记
及各 slot 的 worker 登记，绑定 PID、启动 ticks、父进程、分配设备、完整分配作业集合，
并将 worker 已有 CUDA 上下文绑定到逻辑设备 UUID 和观测 NVML PID。
此路径不额外创建上下文，不改变 RNG，不追加资源预留，也不将 CUDA 初始化本身当作上下文证据。
早期未观测到 NVML PID 时保持 pending/null，首次观测到自身 PID 后只登记一次；
非空任务 worker 不能在缺失登记时通过终态。登记也不声称 GPU 独占；
原 selected-owner 协议保持独立。协调登记的末尾记录退出及清理状态，清空实时 `workers`
列表不会擦除这些证据。登记是执行证据，不是效能结论或独立验收；
原先缺失的登记不会被推测补回或改标为通过。

训练状态执行的最终 `execution_state.json` 与 `terminal.json` 状态一致，
绑定 terminal SHA-256，并在存在时绑定 `cleanup.json`。
成功仍为 `complete_pending_independent_acceptance`；失败保留最先发生的异常，
将清理或报告导出错误分别记录。terminal 是权威收据，状态文件只是快照，不是续跑指令。
仅计划输出没有实际 worker 或上下文登记。

失败时保留整个目录、原 terminal、登记及已完成前缀。不能靠推测补身份，也不能据过期
running 快照重启。由运行负责人核对终态及清理证据；明确授权的缺失作业使用新目录。
CPU mock 测试或仅计划检查不证明实际 GPU 资格或全项目发布已通过。

## 原生可视化范围

独立接受的在线可视化原生诊断包括六个 CLI 情形、十二份完整目标输出/PNG 和 144 个未舍入指标。一/双卡成对预测及 PNG 字节精确一致；双卡面板的同时 NVML 观察显示不同 GPU 上的独立 worker。一个 VOC 单卡情形的 NVML 窗口未采样，不宣称连续所有权。此接受仅涉及原生执行，不接受出版布局质量、固定三图论文素材或全项目发布。

另行接受的标签布局组件以这些保存预测离线重绘六张图片：所选检测框和预测均保留，标题与标签矩形在边界内且互不重叠，六张图片均经过直接查看。这里按原字节集成其四个源码、测试和指南文件，不重复 GPU 执行。该组件接受不替换三张固定论文图，也不证明所有可能的拥挤图片均能完成标签排布。
