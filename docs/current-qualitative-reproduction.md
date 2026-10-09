# Current-Main Qualitative Reproduction

[简体中文](current-qualitative-reproduction.zh-CN.md)

Run from the repository root in the pinned `oda` environment. This CPU-only
builder consumes one saved `main_transfer` run. It never loads a detector,
generates attacks, replays AP, downloads payloads or changes the original run.

## Fixed Display

The maintained display is COCO `val`, Faster R-CNN R50 source to Cascade R-CNN
R50 target, image IDs `785, 1503, 2157`, threshold `0.50`, and at most `100`
detections per image. Clean comes first; all ten attack rows follow the declared
`main_transfer` registry order, including OSFD, SFIM-B and VCSF. Each attack's
budget is resolved from that protocol's `method_budget_profiles` with its
registered default as fallback. VCSF uses observed-cost accounting; the other
current-main methods use the compute-matched profile. Mixed budgets are retained
in schema-2 requests and disclosed in the panel and manifest.

Images are not cropped or spatially transformed. Aspect-preserving containment
uses the same display scale for a given image across every row. Predictions,
class labels and confidence scores come from saved post-NMS archives.
These author-preselected COCO examples are illustrative reused choices, not an
independent selection. No fixed VOC examples are invented.

## Inspect Without Inputs

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --help
conda run -n oda python -B experiments/current_saved_qualitative.py --plan-only
```

Without `--main-run`, plan-only prints the display contract and
`display_contract_only_no_inputs_validated`. It reads declarative registry
metadata, not payloads, and does not grant execution or source-freeze admission.

## Validate Saved Inputs

Keep the saved clean images, canonical COCO annotation, attack pixels, generation
metadata, and complete lossless `predictions.json.gz` archives available.
The saved run must have `plan.json` with protocol `main_transfer` and
`summary.json` explicitly recording `max_images: null`, complete status,
zero failed records and zero image failures. All consumed evaluations and
generations must cover the full 5,000-image validation population.

Replace `saved_run` below with your actual saved main-run directory:

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --main-run outputs/experiments/main_transfer/saved_run --plan-only
```

With inputs, plan-only invokes the full loader in a private temporary directory.
It validates all prediction archive bytes and records, hashes, coverage, GT and
selected pixels without writing the requested output directory. The temporary
request is cleaned up; its validation result is not a retained render plan.
The current registry's source-identity checks still apply when loading a main run.

Only these original metric layouts are used:

```text
evaluations/coco/clean/cascade_rcnn_r50/metrics.json
evaluations/coco/faster_rcnn_r50/METHOD/default/cascade_rcnn_r50/metrics.json
attacks/coco/faster_rcnn_r50/METHOD/default/run.json
```

The builder does not read aggregated records or combine runs. It refuses missing,
duplicate, failed, partial, wrong-dataset/split/source, wrong-budget or conflicting
checkpoint/parameter evidence. VCSF must match the maintained public parameter
identity. A method name or AP value alone never qualifies evidence.

## Render PNG

The parent output directory must exist. Choose a fresh destination outside the
original main run; even an existing empty directory is rejected.

```powershell
conda run -n oda python -B experiments/current_saved_qualitative.py --main-run outputs/experiments/main_transfer/saved_run --output outputs/current_qualitative_01
```

Execution is the default and requires both `--main-run` and `--output`.
The output contains `request.json`, `request.sha256`, and
`source_inputs.json` binding this run's original inputs and identities.
`panel/panel.png`, cell PNGs and `panel/manifest.json` are derived display
artifacts. No PDF is produced. This workflow does not replace the manuscript or
manual PDF and does not overwrite canonical author figures.
The output-root `manifest.json` binds the renderer manifest and explicitly
records the new-user scope and non-certification boundary. A failed execution
keeps `failure.json` and its request rather than overwriting or deleting them.

`--max-images 1` is an optional display diagnostic. It shortens the visible
columns only: all three declared inputs and the full 5,000-image evaluation
scope still require validation. It does not turn partial metrics into formal AP.

## Boundaries And Recovery

The result is a new-user saved-input reproduction, never author-certified
figures, scientific acceptance or independent confirmation. Hashes demonstrate
consistency with consumed originals, not independent approval. Original
checkpoint identities are recorded; no checkpoint is loaded or scientifically
requalified.

On missing archives or incomplete metrics, preserve the originals and supply one
complete, correctly registered run. On output-path errors, choose a new path.
Do not merge timestamped runs, fabricate unavailable inputs, reuse an old VCSF
candidate under the current name, or overwrite a failed output. Retain a failed
attempt for inspection and use a new directory for a later attempt.

## Canonical LGP Source Identity

The input run must also retain its original `provenance.json`. The builder checks
the saved `source_manifest` count, seal and unique relative paths, then compares
only the registered canonical LGP executor, its frozen `lgp_*.py` helper family
and directly imported numerical common helpers against the current project.
Expected hashes come from the registry-referenced source freeze, not Python
hash constants. This is a bounded source-map check; historical reporting code
may differ, and no equality of the entire historical project is required.

The original provenance, registry and freeze references and matched numerical
source map are retained in `source_inputs.json`, the root manifest and the
renderer evidence inputs. Missing or old LGP source hashes are rejected even
when the attack name and recorded parameter hashes agree. This prevents treating
old raw LGP output as the current canonical implementation by relabeling it.
It does not confer scientific acceptance or whole-execution equivalence.

For VCSF, the same original run provenance must match the final A10 numerical
source subset recorded in the registry-referenced selection protocol. Its map
seal, current source hashes, complete parameter identity and declared generation
implementation are checked together. The original selection reference and
matched source map are preserved alongside the LGP proof. A same-name,
same-parameter run with missing or different numerical sources is rejected;
unrelated historical reporting hashes may still differ.
