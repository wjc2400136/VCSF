# Saved Main-Report Export

[English](saved-report-export.md) | [简体中文](saved-report-export.zh-CN.md) | [Project](../README.md)

This CPU-only public entry accepts saved **main_transfer** runs only. It exports
presentation labels, tables and optional scalar plots to a new directory without
reading predictions/images, loading detectors, measuring new cost or recomputing AP.
It does not accept scientific results.

## Input and Command

Use pinned `oda` from the source root. The immutable input needs its original
`plan.json` identifying `main_transfer`, `records.json`, and available
`summary.json`/source bindings. Supplementary, unknown-protocol and old generic
runs are rejected by this entry before an output is created.

```bash
python experiments/reexport_saved_reports.py --help
python experiments/reexport_saved_reports.py --run outputs/experiments/main_transfer/my_coco_run --output outputs/report_exports/my_coco_run_export01 --plots
```

Replace `my_coco_run` with the exact existing source run name; choose a new,
nonexistent export directory. Omit `--plots` for text/CSV/TeX only.
Optional `--native-counts` copies saved per-image counters from this run without
new measurement or physical-BE calibration. `--registry-root` is a read-only
identity-bound registry selector, not a way to change method identities.

## Outputs and Checks

The exporter preserves input `records.json` bytes and generates
`reports/dataset/`, twelve-row `evaluations/` summaries and
`report_export.json`. Optional scalar plots read saved numbers only.
Canonical target/source order, markers and source-excluded BB Mean are retained.
Undefined official metrics stay raw in JSON, are excluded from rankings and show
as NR in derived views; valid numeric zero stays zero.

Diagnostics display actual scope/sample information rather than full-population
labels. Unknown scope remains unknown. Native logical updates, detector/partial
backwards, references and differentiable views are separate; missing observed
counts or calibrated BE remain NR, never free work.

Success requires zero exit, the printed new output, equal input/output records
hashes, unchanged bound originals and a sealed export receipt. Review labels and
coverage. On failure preserve the derived directory and use a new one after
repair; do not overwrite or merge. An exported table is not AP replay,
independent acceptance or proof that every experiment ran.
