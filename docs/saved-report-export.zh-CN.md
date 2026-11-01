# 保存主实验报告导出

[English](saved-report-export.md) | [简体中文](saved-report-export.zh-CN.md) | [项目首页](../README.zh-CN.md)

公开 CPU 入口仅接受保存的 **main_transfer** 运行，在新目录导出标签、表格和
可选标量图，不读取预测/图像、不加载检测器、不重新测成本、不重算 AP，
也不接受科学结果。

## 输入与命令

在源码根使用固定 `oda`。不可变输入须有标识 `main_transfer` 的原 `plan.json`、
`records.json`，以及已有 `summary.json`/来源绑定。补充实验、未知协议和旧
通用运行会在创建输出前被拒绝。

```bash
python experiments/reexport_saved_reports.py --help
python experiments/reexport_saved_reports.py --run outputs/experiments/main_transfer/my_coco_run --output outputs/report_exports/my_coco_run_export01 --plots
```

将 `my_coco_run` 替换为确实存在的源运行名，选择不存在的新导出目录。
不需要标量图时去掉 `--plots`。`--native-counts` 仅复制该运行已有逐图计数，
不新测量、不校准物理 BE。`--registry-root` 是只读、受身份约束的注册表选择，
不是更换方法身份的后门。

## 输出与检查

输入 `records.json` 字节保持，生成 `reports/dataset/`、十二行 `evaluations/`
摘要及 `report_export.json`。可选标量图只读旧数值；规范目标/源顺序、标记和
排除源的 BB Mean 保留。官方未定义指标在原 JSON 中保留，派生视图记 NR 并
排除排名；真实数值零仍是零。

诊断显示真实范围/样本，而非全量标签；未知范围保持未知。逻辑更新、检测器/
部分反向、参考和可微视图分别记录，缺失观测或校准 BE 记 NR，不算免费。

成功要求零退出、打印新输出、输入/输出记录哈希相同、绑定原件未变及封存导出
回执。须复核标签和覆盖；失败后保留派生目录，修复后换新目录，不覆盖或合并。
表格导出不等于 AP 回放、独立验收或所有实验已运行。
