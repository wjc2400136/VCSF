# Final-Only Follow-Up Registration Contract

> Current scope: the owner retained the original bounded five-seed check after
> reconsideration. See the [retention decision](vcsf-final-multiseed-retention-20260917.md)
> for presentation and interpretation limits. The temporary cancellation is
> superseded; the original scope below remains subject to execution admission.

Status: prospective scope and acceptance contract; executable catalogue, input
binding, reuse qualification and execution admission remain pending. This file
does not arm a runner or authorize a new upload payload. It does not change the
hash-bound historical freeze or its historical registration flags.

## Authority and Identity

Follow the [final disposition](vcsf-final-disposition-20260917.md) and the
post-freeze scope in the [nomination](vcsf-final-nomination-20260915.md).
The machine freeze is `vcsf-final-method-freeze-20260917.json`, SHA-256
`a697b1331cf54808633bcfc57d6f07adb630ef41beb7fea709402b749e1418d0`.
Resolve parameters and the entire implementation inventory through its bound
authenticated input report and JSON pointers, not a reconstructed parameter
dictionary or only the three critical implementation files. Require the
registered parameter and inventory hashes before preparing any executable job.

The selected configuration remains common group26 / final A01. No new search,
control promotion, optimizer change or candidate composition is admitted.

## Logical Job Set

Read canonical source and target order from `configs/models.yaml` through the
registry. Every group contains all sixteen targets and all twelve standard
bbox summary metrics. A logical generation key is dataset, split, ordered
image identity, source, configuration/implementation identity and group seed;
report role is not part of that key. Checkpoint, evaluator and input bindings
must also agree before two proposed roles share an observation.

| Report role | Dataset and full split | Source set | Seeds | Groups | Target cells |
| --- | --- | --- | --- | ---: | ---: |
| Final stability | COCO val, 5000 images | Faster R-CNN R50 | 42,43,44,45,46 | 5 | 80 |
| Affected VCSF main | COCO val, 5000 images | Canonical six sources | 42 | 6 | 96 |
| Affected VCSF main | VOC val, 4952 images | Canonical six sources | 42 | 6 | 96 |

The COCO Faster R-CNN seed42 group occurs in two report roles. After logical
deduplication the prospective union is ten COCO groups and six VOC groups:
sixteen groups / 256 target cells, not seventeen / 272. This is a scope count,
not evidence of acceptance or readiness. No BDD main, clean rerun, other method
or seed-expanded ablation control belongs to this job set.

One historical COCO group is a reuse candidate, initially UNQUALIFIED. If and
only if its complete evidence qualifies, fifteen new groups / 240 target cells
remain: nine COCO groups and six VOC groups. The corresponding new generation
count is 74712 images (9 * 5000 + 6 * 4952), not a uniform 5000 per VOC group.
Without that qualification, do not silently rerun or report the candidate as
accepted; resolve the specific failed or missing contract first. Technical
recovery must have its own bounded admission and preserve original evidence.

## Seed42 Dual-Role Qualification

Keep the immutable provenance chain common group26 -> final A01/group1.
One immutable observation can supply COCO stability seed42 and the COCO main
Faster R-CNN row; these are two reporting roles, not independent observations.
Create a separate scoped qualification record rather than changing old
receipts, their eligibility flags, original run roots or the freeze descriptor.

The qualification consumer must authenticate and reconcile:

- The freeze, authenticated full final input report, original common/final
  reuse chain, execution plan, complete runtime map and explicit source bridge.
- Full COCO annotation and clean-image identities, ordered 5000 image IDs,
  full-image/non-diagnostic status, zero failures and original generation audit.
- `selected_position` seeds: image at zero-based position i uses 42+i, not
  42+image_id. Bind ordered IDs, offsets `[[image_id,i],...]`, mapping
  `[[image_id,i,42+i],...]`, and each manifest row's position/attack seed.
- Dataset-specific checkpoint hashes and seventeen role-config slots: one
  source and sixteen targets, retaining source and white-box target roles
  separately. Bind resolved/effective configs, preprocessing, label space,
  NMS/test settings, evaluator and package provenance. Do not broaden existing
  config-normalization exclusions to obtain a match.
- Sixteen target records with all twelve unrounded metrics, exact summary
  order, normalized cells, saved replay and independent audit references.
  Reuse the prior replay's proof; qualification itself does not rerun inference.
- Per-target metrics and prediction artifact/archive hashes, compressed and
  decompressed identity, prediction/image coverage and original attack-root,
  annotation, parameter and checkpoint bindings. Twelve floats alone do not
  qualify a result. Preserve prior payload lifecycle evidence where applicable;
  do not invent present-day payload availability from an old receipt.
- Complete saved-cost provenance where reported, without inferring isolated
  timing, physical backward equivalence or trajectory identity from metric
  agreement. Unknown physical costs stay unknown.

Existing full-panel consumers are evidence sources, not arbitrary single-group
APIs: do not forge a smaller historical execution plan to make A01 pass them.
Verify evidence against its original frozen tree, with the new consumer's own
identity recorded separately from the producer. Reuse decisions are per dataset;
no COCO identity or checkpoint qualification transfers to VOC.

## Statistical and Reporting Contract

Publish all five stability seeds, without dropping a seed based on outcome.
For each target and each of twelve metrics report the five unrounded values,
arithmetic mean, sample standard deviation (ddof=1), minimum and maximum.
Compute each seed's BB Mean over its fifteen non-source targets first, then
summarize those five BB Means; do not substitute variability across targets.
Retain the white-box row and held-out DINO marker in their canonical positions.

These are descriptive results, not a predeclared significance test, confidence
interval, success threshold or independent confirmation. Five seeds do not undo
selection on this validation split. No bootstrap queue is reactivated. Retain
all target disagreements, including Swin, and distinguish exploratory evidence
from frozen-method follow-up. Main rows retain dataset-specific primary metric
definitions; no silent substitution of VOC AP50 and COCO AP.

Generate JSON, CSV and TeX/plots from accepted records. Apply canonical ordering
and the existing precision policy. Seed42 shared across reports is disclosed
as the same observation. Do not fill missing values with zero or publish a
partial panel as a completed main result.

## Execution and Acceptance Gates

The new catalogue must contain only the deduplicated final VCSF jobs, preserving
all roles and the unqualified/qualified/new partition. Machine admission must
bind exact input identities, checkpoints, configurations, implementation,
evaluator, budget and seed mapping before GPU work. Planned hashes or an empty
placeholder are not verified inputs.

Historical COCO/VOC submission entry points contain two methods, clean work and
old freeze schemas; they must not be launched wholesale or silently repointed.
The existing efficacy runner and isolated admission contracts are COCO/source/
seed scoped. Extending them requires a separately validated execution contract,
not weakening historical guards or assuming generic `run_attack` arguments
establish final-method fidelity.

Register equivalent one-device and two-device plans with the same complete jobs.
Use two devices when independently ready jobs and measured resource capacity
permit; preserve ownership and failure containment. Four-device execution is
not automatically admitted by the earlier attribution run's device allowance.
Before execution inspect current processes and disk capacity, predeclare full
payload/prediction retention and failure reserve, and keep isolated cost timing
separate from concurrent throughput. No destructive cleanup is authorized here.

Accept each full group only with complete generation, sixteen target panels,
twelve metrics, immutable hashes and independent checks. Accept the union only
after every registered group/role and qualified reuse matches the catalogue,
failures are resolved under the registered recovery policy, and generated
reports are verified. Static checks or catalogue compilation prove neither
GPU readiness nor numerical acceptance.

Candidate-dependent robustness, training-state, budget and other retained
claims remain subject to the existing replacement-scope review. They are not
silently canceled by this bounded main/stability registration and are not a
blanket rerun instruction. Paper/manual PDFs, citations, bilingual public
guides, whole-project one/two-device acceptance and small-file synchronization
remain separate required deliverables of the active goal.
