# 在线预测可视化

[English](prediction-visualization-device-interface.md) | [简体中文](prediction-visualization-device-interface.zh-CN.md)

请在 Linux 执行服务器的仓库根目录、固定的 `oda` 环境中执行命令。
进程隔离复用现有 Linux helper；这些检查不证明 Windows 实际执行支持。
这个在线工具调用原评估器，
可读取已有对抗样本 manifest，但不生成攻击。离线图表绘制工具仍是 CPU 工具，
不添加 GPU 参数。

```bash
conda activate oda
python -B experiments/visualize_predictions.py --help
python -B experiments/visualize_predictions.py --plan-only --dataset coco --devices cuda:0
python -B experiments/visualize_predictions.py --plan-only --dataset voc --targets vfnet_r50 faster_rcnn_r50 --devices cuda:0 cuda:1
```

`--plan-only` 只读取登记声明和原设备 helper 源码，打印 JSON，不创建输出。
它不校验数据集载荷、checkpoint、对抗 manifest 或 GPU 可用性，不导入检测器栈，
不加载模型，不执行推理，不计算 AP。计划成功不等于执行授权或科学结果。

集成负责人确认源码 freeze 并授权 GPU 执行后，可执行小规模干净图像可视化：

```bash
python -B experiments/visualize_predictions.py --dataset coco --target faster_rcnn_r50 --device cuda:0 --max-images 20
```

同一个已有对抗输入的计划示例：

```bash
python -B experiments/visualize_predictions.py --plan-only --dataset voc --split val --adversarial-run outputs/attacks/voc/completed_run --targets vfnet_r50 faster_rcnn_r50 --devices cuda:0 cuda:1
```

将 `outputs/attacks/voc/completed_run` 替换为对应数据集、split 的已完成攻击目录。
只有执行获准后才移除 `--plan-only`。实际执行保留原评估器的 manifest、数据集和
split 校验。当前维护入口只接受 COCO/VOC；BDD100K 不在当前项目范围内。
这些检查不证明全项目发布就绪。

默认仍是一个完整的 Faster R-CNN target job，20 张图像，绘图阈值 0.30，每图最多
绘制 100 个框。选择两卡不会自动增加第二个 job。`--targets` 选择唯一的登记目标，
始终按 canonical registry order 输出。单卡串行执行同一工作集，双卡轮转分配独立的
完整 target job，不进行图像分片、DDP、source ensemble 或攻击预算变更。
每卡必须独立容纳所分配的检测器；显式请求的 GPU 不可用时，在创建输出前报错。

保留旧 `--target` 和 `--device`，包括旧单设备 `cpu`。`--targets` 与 `--target`
互斥，`--devices` 与 `--device` 互斥。`--devices` 只接受一个或两个不同的显式编号，
例如 `cuda:0 cuda:1`，不接受 `cuda`、`cpu` 或逗号拼接字符串。

单目标 metrics 和 overlay 保留原输出布局；多目标放在运行目录的 `targets/TARGET`
下。相邻的 `.dispatch` 后缀目录保存不可覆盖的计划、带哈希的 worker 请求、日志、
归属和终态回收记录。两个目录都必须是新目录，指定 `--output` 时也不能复用空目录。
默认在 `outputs/visualizations` 下使用新 UTC 时间戳。打开成功目标打印的目录，可查看
post-NMS 检测框、类别名称和置信分数。

阈值和框数只影响绘图，不改变 evaluator predictions；有限图像 AP 仍是诊断指标。
全部选中 job 完成且所有自有 worker 已 wait/reap 才算接口执行成功。失败退出非零，
`ERR` 和尚未启动的 `NR` 不编码为零指标。默认允许独立 job 在某个目标失败后继续；
`--stop-on-error` 同时保留遇到图像错误停止的行为，并在 target 失败后中止待执行工作。
中断会终止自有进程组并等待回收。保留部分输出和 `.dispatch` 证据，检查日志；
明确获准重跑时选择新输出目录，不覆盖或自动恢复旧运行。

## Caption 布局

绘图器先完成所有检测框，再画 caption；确定标签位置前预留标题区域。
类别颜色、四位小数置信文本、原图尺寸和比例不变。标签先按确定性规则尝试
锚点附近的位置，再搜索最多 64 x 64 个图像网格起点。明显移动的标签用细引线
连接到检测框锚点，引线在所有文字之前绘制。检测框裁剪、排序、阈值和上限选择
不变，不通过删除检测来降低拥挤程度。

保留原有考虑字形 bearing 的字体拟合逻辑，以及可缩放字体拟合到 8px 的下限。
caption 若无法放进图像，或在有界搜索中找不到无碰撞位置，不画该 caption，
但保留检测框并记录具体限制。新 `caption_layout_limitations` 为空表示这两类条件
均未发生；旧字段只记录 fit failure，旧空列表不代表承诺无碰撞，也不否定原生执行。

Python 调用可指定 `include_layout_rectangles=True`，在返回记录中添加可选的
`layout_rectangles`，不移除旧字段。每个已绘制标签和标题记录原图像素坐标下的
半开区间 `[left, top, right, bottom]`、文字及实际字号；检测标签还记录排序后的
检测索引、检测框和引线。可据此直接检查边界及包含标题在内的非重叠性。
这项绘图修复不改变预测或指标，新诊断图也不自动替换已接受的稿件图身份。
