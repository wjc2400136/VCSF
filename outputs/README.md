# Reproduction Outputs

[English](README.md) | [简体中文](README.zh-CN.md) | [Project](../README.md)

Run commands from the repository root in pinned ODA. This source package has no
author output payloads. Follow the [main guide](../docs/current-public-reproduction.md)
and each topic guide for its actual output namespace and success checks.

Direct experiment entries execute by default; `--plan-only` does no model work
but some entries write an immutable plan leaf. Inspect each entry's help.
`--max-images` marks a diagnostic, never formal mAP or canonical checkpoint publication.

Keep `plan.json`, assignments, state, original logs, unrounded `records.json`,
summary, source/input/checkpoint bindings and automatic CSV/TeX/plots together
where the specific entry produces them. Retain all twelve standard metrics.
No missing, failed or unrun result is zero; use NR, ERR or the audited structural skip.

Success requires full intended coverage, zero failed records and independent
saved-output checks, not just exit zero. A source guard pass or CPU plan is not
CUDA execution or scientific acceptance. The package guard does not traverse runtime payloads. Source identity covers
only the shipped source namespaces and the three directory guides.

On failure, preserve the complete leaf and partial evidence. Confirm PID/start
identity before retrying after a lost connection. Repair the actual cause and
execute only an explicitly selected missing subset in a fresh leaf.
There is no automatic resume, overwrite or merge. Payload deletion needs a separate
explicit scope, verified target and backup/recovery decision; this guide authorizes none.
