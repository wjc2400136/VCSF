# Saved Report Export

[Simplified Chinese](saved-report-export.zh-CN.md)

This CPU-only interface repairs presentation scope and compute-unit labels from
existing records. It never evaluates predictions, loads images or detectors,
changes attacks, measures cost, or accepts scientific results.

## Prerequisites

Use the pinned `oda` environment (Python 3.8.20) from the repository root.
The immutable input run must contain `records.json`. Retain `summary.json`
and `plan.json`; final-background exports also use `background_plan.json`.
These files supply known image caps, split and diagnostic identity. Missing
scope stays unknown.

Current preprocessing exports also require `input_binding.json` and the small
bound prepared-view and original generation metadata. Reports are separated by
defense under `reports/defense/` and `evaluations/defense/`; repeated target cells
from different defenses are not merged. The same command supports oblivious and
adaptive saved runs. It preserves input `records.json` byte-for-byte and enriches
only the derived report rows.

Preprocessing transfer tables reuse the maintained runtime reporter, including
its defense/protocol-specific labels and 100-times-raw caption. Undefined COCO
`-1` sentinels become NR only in the non-mutating table/analysis/plot view; they
are excluded from rankings, never converted to zero or negative AP. Valid zero
remains numeric. Original `records.json` bytes and `presentation_records.json`
raw metric values, including `-1`, are preserved. Standard raw COCO-style
single-target summaries keep their existing twelve-row convention.

Generic transfer tables also exclude undefined official metrics before ranking
or calculating BB Mean. This presentation view preserves raw saved JSON, valid
zero values and source-identical white-box cells; an empty black-box subset is NR.

A current-radius run must retain its original `radius_plan.json`. The exporter
copies and binds that small JSON and `records.json`, then calls only the maintained
`current_radius.write_reports` source-by-radius reporter. Generic ablation
tables, scalar plots and `--native-counts` are rejected for this route. Missing
radius selection is an error, never a fallback to a source-collapsed projection.
Training-state table labels use recorded `fullval` and `retained500` roles even
for diagnostics; labels alone do not authorize a full-validation result claim.

Public transfer tables shrink only when needed: width is bounded by
`textwidth`, followed by the existing total-height limit of `0.88 textheight`.
Small panels retain their natural size. This changes only the TeX container,
not the numeric tabular body or the underlying result.

An isolated repair may specify `--registry-root` pointing to the unchanged
frozen project. This is read-only and retains the original source identity guard.
It does not qualify the repaired checkout for detector execution or update its
freeze. Integration and non-numerical source rebinding remain separate decisions.

## Command

```bash
CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 conda run -n oda python -B experiments/reexport_saved_reports.py --run qualification/saved-diagnostic --output qualification/derived-scope01 --plots --native-counts
```

Replace `qualification/saved-diagnostic` with an existing saved run. Choose
a new, nonexistent output directory. For an isolated checkout, append
`--registry-root ../frozen-project`, using the actual frozen registry root.

`--plots` renders scalar-record PNG/PDF figures on CPU without reading attack
trajectories or image/prediction payloads. `--native-counts` reads only saved
`native_cost.jsonl` inside attack groups belonging to the input run. It copies
per-image counters without new measurement or physical calibration. Preprocessing
may reference generation outside the evaluation run only through its bound
input/view provenance. Omit both
flags for the minimal text/CSV/TeX export.

## Output And Interpretation

- `reports/dataset/`: transfer/ablation tables, scalar analysis and runtime accounting.
- `reports/background/`: aggregate and per-control final-background projections.
- `evaluations/`: single-target text/CSV/TeX summaries retaining all twelve raw metrics.
- `reports/dataset/native_counts/`: optional raw per-image counter JSON/CSV/TeX.
- `records.json`: a byte-identical copy of the input records.
- `report_export.json`: input hashes and the label-only export boundary.

Limited or explicitly diagnostic outputs display `PARTIAL DIAGNOSTIC` plus
actual dataset, split, sample N and max_images. N is successfully evaluated images,
not the requested cap. Mixed or missing per-cell N is never replaced by a full
population or zero. Unknown scope displays `SCOPE UNKNOWN`. Explicit
non-diagnostic full-split scope keeps existing scope presentation.

The evaluator accepts optional presentation-only selection context. Current
preprocessing passes its declared protocol/selection: the complete registered
retained-500 population is not smoke, while max-images 1/6 remains diagnostic.
An undeclared image-ID subset is unknown, not diagnostic solely because it is
smaller than the dataset. This context never changes selected IDs or metrics.

Runtime accounting separates declared budget/profile from observed actual mean
counts, with the registered counting unit. LGP early-stop counts are not replaced
by declared 20. VCSF logical updates, partial backbone/neck backwards,
clean-reference rows and adversarial differentiable views are not interchangeable
with complete-detector BE. Auxiliary zero does not imply reference/partial paths
are free. Missing actual counts and calibrated BE remain NR.

Preprocessing CSVs retain `generation_cost_scope`, original generation N and
run hash separately from evaluated N. `reused_original_generation` is not a new
adaptive generation call. `current_adaptive_generation` denotes generation
executed by the saved adaptive run. Original population means are not relabelled
as counts for a smaller evaluated subset. Native logical updates, detector and
partial backwards, references and differentiable views retain separate columns;
missing native means or calibrated BE remain NR.

Paired-cost CSV/TeX now display actual measured N, the bound plan's cap and
diagnostic identity, with method-specific recorded-count units. Training-state
CSV/TeX receive the declared presentation context and report fullval and
retained500 actual N separately. A six-image fullval diagnostic can have a
one-image retained intersection; neither is a 500/5000-image result. This does
not change calibration, training, scheduling or metric calculations.

For `current_paired_cost`, the input type is the original `plan.json` plus
`summary.json` (and available `input_binding.json`/`measurements.json`), not
`records.json`. The exporter copies those inputs byte-for-byte and never creates
an invented records file. Use the same command without `--plots` or
`--native-counts`. A counter whose saved origin is `declared_cap` or
`unavailable` remains visible as saved metadata but its observed actual mean is NR.
The `current_training_state_transfer` records route writes separate fullval and
retained500 tables without prediction replay or detector access.

Native `diagnostic_only=null` remains null in the new native JSON and unknown
in its CSV, independently of the report's known global diagnostic cap. Raw JSON,
twelve-metric ordering, canonical model order, white-box/held-out markers and
existing numerical presentation precision are unchanged.

Success requires exit code zero, `[OUTPUT]` naming the new directory, and
matching input/output `records.json` SHA-256 values. Review labels and input
bindings before use. The output records copy uses the first bound input bytes;
all small bound originals are rechecked before `report_export.json` is sealed.
An input change fails without a sealed receipt; retain that derived directory.
Existing outputs are never overwritten. Preserve failed
exports, resolve the reported input issue, and choose a different new directory
for retry. Do not merge runs/splits or treat diagnostics as formal full-split mAP.
