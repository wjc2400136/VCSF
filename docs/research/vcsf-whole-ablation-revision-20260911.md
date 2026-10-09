# Bounded Whole-Ablation Revision, 2026-09-11

## Authority And Status

At 12:18:52 UTC the owner accepted completion of the three binary scale-operator
factors, requested another rigorous whole-programme review after the earlier
revision, and requested amendment of the actual active goal. This record
supersedes the prospective staging and closure details of
[the September 9 plan](vcsf-whole-ablation-closure-20260909.md), not its historical
results or the [statistics amendment](vcsf-exploration-statistics-amendment-20260909.md).
This is a bounded research design, not an executable run manifest or acceptance.

The complete objective still includes ablation research, necessary affected
six-source VCSF results, paper and generated figures/tables, references and the
scientific-writing handbook, separate English/Chinese public guides, whole-project
one/two-GPU release validation, and server/local synchronization without large
payloads. Neither a completed engineering gate nor this revision completes it.

## Corrections To The Earlier Design

1. A/B/C/D are an anchor and three one-factor replacements, not a complete
   interpolation-by-placement-by-padding factorial. Preserve their 17 accepted
   operator/width configurations and all outcomes. Complete the missing four
   operator combinations at one fixed half-width, not at every width.
2. Use one common exploration anchor for the remaining modules. Sequential
   module-wise baseline replacement is no longer the default: it makes the
   catalogue's 27-configuration count misleading and mixes conditional effects.
   Record stage recommendations without automatically implementing each winner.
3. Require active, distinct factor levels. Center placement makes the inherited
   independent-offset switch inactive; identity scale also removes geometric
   variation. Do not run or label such rows as correspondence ablations.
4. Treat final attribution as an explicit finite claim-to-control manifest,
   not an open-ended instruction to repair every historical comparison.
   A failed contribution may be withdrawn without inventing a positive result.

## Common Research Contract

Use Faster R-CNN R50 only, COCO val2017 all 5,000 images, exploration seed 42,
the canonical sixteen-target order and the fifteen-target source-excluded BB Mean.
Use unrounded JSON, all twelve evaluator summaries, same-input paired contrasts,
complete factorial contrasts, disclosed physical cost and implementation complexity.
AP is dataset-level: no invented per-image AP. Differences and factorial contrasts
are descriptive, not significance tests or mechanistic proof.

Before each admitted stage bind exact parameters, implementation and input hashes,
ordered IDs/seeds, contrasts, comparison direction, reuse/new partition, cost
accounting, output lifecycle and independent acceptance. A conditional design or
count here does not arm jobs. Partial or failed panels cannot be ranked; missing
evidence is not zero. Reuse requires qualified exact scientific identity and
provenance; matching names, seeds or fixed-state probes alone are insufficient.

## Stage 1: Complete The Operator Factorial

Keep half-width `1/3`, hence scale range `[2/3, 4/3]`, fixed for all eight cells.
All other settings remain the original four-term scale-study background.

| Cell | Image interpolation | Placement | Image padding | Evidence status |
| --- | --- | --- | --- | --- |
| A | bilinear | random | reflect | Existing A_h2, qualify exact reuse |
| B | bilinear | center | reflect | Existing B_h2, qualify exact reuse |
| C | bilinear | random | zero | Existing C_h2, qualify exact reuse |
| D | nearest | random | reflect | Existing D_h2, qualify exact reuse |
| E | bilinear | center | zero | New joint combination |
| F | nearest | center | reflect | New joint combination |
| G | nearest | random | zero | New joint combination |
| H | nearest | center | zero | New joint combination |

Cell letters E-H are prospective labels, not existing registry entries. The new
namespace must not mutate the frozen four-row scale producer or its old hashes.
Image interpolation is the factor; mask interpolation/padding remain nearest/zero.
Preserve the declared RNG draw policy, shared clean/adversarial geometry, schedules
and all other settings. Validate joint branches, gradients and identities in ODA.

Report three main effects, three two-factor interactions and the three-factor
interaction from the complete eight-cell panel, per target and for BB Mean.
Declare factor coding and contrast normalization before execution. Also retain
simple effects to expose mixed signs; do not imply all effects are independent.
For lower attacked AP, a negative candidate-minus-reference difference is favorable.

Only four missing configurations are the planned new scientific cells. The eight
cells require 128 target records in total; four qualified reuses would leave 64
new evaluations and 20,000 generated image instances. These are conditional work
counts, not proof of reuse. A failed historical qualification must be resolved
explicitly, not silently changed into four extra formal runs.

Do not expand to `8 x 5` widths, optimize a separate width for each operator,
or claim global scale-range optimality. The old width study is a retained
four-operator sensitivity study; the new factorial is conditional on width 1/3.
In particular, do not say all old operators preferred 1/3: D's observed grid
minimum was 1/6. The prior C_h2 working decision remains valid history of the
four-construction review, not the winner of an unfinished eight-cell comparison.

After complete acceptance choose one nonzero operator setting as exploration
anchor `B_anchor`, with a written AP/cost/complexity rationale. This choice does
not establish that scaling should remain in the final method; identity remains
an explicit core control. A transparent conservative choice is permitted for
small mixed effects, without inventing a new minimum-effect or all-target gate.

## Stages 2-6: One Common Exploration Anchor

Bind `B_anchor` once: the selected operator at half-width 1/3, cross-stage
levels_two (two backbone plus two neck terms), detector initialization followed
by 19 feature updates, shared geometry, energy weighting, log-mean-exp aggregation,
valid support, centered gradient and momentum `7/10`. Other pinned settings remain
unchanged. This is an exploration reference, not final author promotion.

Every block below retains this background except its declared factors. Complete
each block and record keep/change/reject recommendations, but do not rebase later
blocks on earlier recommendations. Preserve negative and heterogeneous results.

| Block | Cells before within-block deduplication | Interpretation |
| --- | --- | --- |
| Core | 8: identity/nonzero scale x neck/cross-stage x none/detector | Full three-factor design. Initializer is a schedule factor: none has 20 feature updates; detector has 1 detector plus 19 feature updates. |
| Features | 5 unique: cross-stage 1/2/3 levels per stage; neck 2; backbone 2 | Includes the matched-total-two subset: neck 2, backbone 2, cross-stage 1+1. Nested depth and equal-term comparisons answer different questions. |
| Initialization | 5: detector+19, none+19, none+20, random-sign+19, random-sign+20 | Separate equal post-init steps from equal logical-gradient ceiling. Fix random amplitude and independent stream. Report actual forwards/backward paths/time. |
| Geometry | 3: shared, independent offsets, independent geometry | One declared nondegenerate placement/scale background; correspondence disruptions are controls, not automatically candidate improvements. |
| Weight/aggregation | 4: energy/uniform x log-mean-exp/mean | Complete two-factor design, with the same feature terms and support. |
| Support | 2: valid/all model locations | One conditional contrast; no automatic weight-by-support matrix. |
| Optimizer | 4: centering on/off x momentum 0/7/10 | Fixed two-level factorial, independent of later momentum recommendation. |
| Momentum | 6 centered values: 0, 4/10, 55/100, 7/10, 85/100, 1 | One-dimensional sensitivity, not six momentum-by-all-factor combinations. |

With random placement, the current catalogue represents 27 unique configurations
after exact shared-anchor and cross-block overlap deduplication. This is the
remaining common-anchor catalogue, not 27 guaranteed new jobs and not an instruction
to use the old hard-coded A_h2 snapshot without rebinding.

If `B_anchor` uses center placement, retain the geometry question in a declared
diagnostic background `Q`: same interpolation, padding and all other settings,
but random placement. Run the three geometry cells under Q; Q/shared overlaps
the corresponding accepted operator cell only if exact reuse is qualified. The
union is at most 28 common-exploration configurations before historical reuse
(remove two inactive center violations, add three Q cells). These Q contrasts
must remain labelled random-placement diagnostics, not evidence of a correspondence
contribution in a centered final method. No implementation rewrite to invent a
new centered violation is implied. An actual new geometric construction needs
its own bounded owner decision.

Do not claim equal-term controls isolate all representation properties, or that
neck-two versus cross-stage-four alone proves stage complementarity. Likewise,
do not describe the initializer core effect as a pure starting-point effect.
These qualifications remain even when the AP direction is favorable.
The centering switch affects the detector initialization update as well as
feature updates, so its contrast is whole-procedure centering, not feature-only
optimization with identical initialization. Geometry disruption can also alter
effective support; Q does not turn that intervention into pure mechanism isolation.

## Stage 7: One Candidate And One Attribution Pass

After the common-anchor exploration, review the complete evidence jointly and
nominate at most one final candidate `F_final`, either B_anchor or a reasoned
combination of registered settings. Do not assemble it by automatically taking
every marginal winner. State which contributions and interactions will actually
be claimed, which settings are merely engineering choices, and which claims have
been rejected. Register F_final and its entire finite attribution manifest before
running any changed-baseline result. No unseen setting may enter that manifest.

The pass reuses exact qualified rows and refreshes only affected controls needed
for retained claims. Lock the identity-deduplicated attribution set A before the
first F_final evaluation. Its hard resource ceiling is **35 complete configurations**,
including F_final, every auxiliary reference, every counterfactual and factorial
corner, and all reused configurations. New work is only the locked configurations
without exact qualified evidence. This ceiling is a conservative resource boundary,
not a mathematical guarantee that every imaginable final claim is covered, and
not a mandate to run 35 groups. Use only the following already declared families:

| Family | Condition |
| --- | --- |
| Final configuration itself | Always, unless exact evidence is already accepted. |
| Core factorial | Bind active binary levels including the final setting for claimed scale/stage/initializer interactions; preserve the original none/detector exploration separately if final initialization is random. |
| Feature coverage/count | Use only registered original depths/surfaces and relevant matched-count controls; no invented neck-four. Unsupported final-stage complementarity is withdrawn or explicitly narrowed. |
| Initializer/budget | Retain the five declared schedules where a final initializer claim needs them; disclose remaining cost differences. |
| Geometry/objective/support | Use the declared three-way geometry, two-by-two objective and binary support controls only as required; exclude inactive contrasts. A Q diagnostic cannot satisfy a final centered-method claim. |
| Centering/momentum factorial | Use zero versus the selected nonzero value, or the original 7/10 comparator when final momentum is zero. No repeated six-value search. |
| Operator factorial | Only if a retained operator-interaction claim is explicitly made in the changed final background; otherwise cite the original conditional factorial and do not rerun it. |

Each active configuration must map to an
explicit retained claim or the final candidate itself. Final factor levels,
backgrounds and budget controls must be prospectively checked for distinctness
and interpretability; the numerical ceiling never licenses a confounded design.
Any required design that cannot fit these families is a scope decision, not an
automatic extension. Do not silently drop an unresolved registered claim to
declare completion; record its withdrawal and consequence for the manuscript.
Keep B_anchor/Q exploratory conclusions in a separate report section from
F_final attribution. A question answered conditionally in exploration is not
cancelled merely because its control is not rebound to F_final; it is also not
evidence of a final-method contribution. When the prospective claim/control
table cannot fit the 35-configuration boundary, narrow the claims before results
or request a newly bounded owner decision. No positive missing-corner inference.

After this one pass, keep or reject F_final and narrow unsupported claims. Do not
retune from its outcomes, nominate another untested combination, restart a width
or momentum grid, or rerun the exploratory history. A substantive new candidate
requires an explicitly new bounded owner decision. Rejected scientific claims
are legitimate outcomes; they do not by themselves finish unrelated deliverables.
Permit at most one explicitly planned technical recovery round per admitted
stage, using only the same locked configurations' verified missing subsets in
fresh immutable roots. Failed configurations do not free slots for new candidates.
Persistent failures pause the dependent acceptance for diagnosis and a bounded
recovery decision; they neither authorize endless retries nor prove scientific
rejection. Final-only seeds and affected six-source results are separate frozen
deliverables, not extra selection rounds or part of the seed42 attribution cap.

## Final Freeze, Stability And Delivery

Final replacement/freeze remains an explicit author decision. After freeze,
predeclare final-only seeds 42/43/44/45/46 on the declared single-source scope;
reuse seed 42 only with exact qualified identity. Do not seed-expand every
ablation. Refresh affected VCSF six-source main rows without rerunning unaffected
clean or other methods. Preserve Swin disagreements and canonical target order.

All numerical/model/research checks remain server-only in ODA. Use explicit two
devices for admitted independent ready GPU jobs; isolate physical timing and
dependency-bound work. One/two-GPU support remains a whole-project release gate,
not evidence of bitwise CUDA equivalence or permission to rerun accepted matrices.
Keep logical gradients, reference forwards, partial backward paths, transformed
views, paired time and memory separate; saved timing is not isolated calibration.

No mandatory exploration bootstrap is restored. Retain necessary metric/input
verification and paired comparisons. Additional uncertainty work requires a
concrete retained claim and prospective scope, not resumption of retired queues.
All selection uses reused validation data; neither five seeds nor sixteen targets
make it independent confirmation. DINO remains architecturally held out as a
source, but its reviewed metrics are not untouched selection-validation evidence.

Completion requires accepted complete 5,000-image/16-target evidence for every
admitted formal group, exact hashes and independent checks, generated reports,
claim/control closure, author freeze or explicit rejection disposition, final
stability and affected result disposition, plus every original paper/release/sync
deliverable. This document does not assert any unfinished numerical work passed.

## Implementation And Transfer Boundary

The maintained scale validator currently allows only A/B/C/D; joint E-H settings
are not yet executable. The ablation catalogue still carries a historical A_h2
snapshot. Implement/register the new namespace and explicit B_anchor binding in
an isolated validation workspace before admission; do not widen old frozen runs.

The owner separately approved the fourteen files in
[the real-reuse draft](vcsf-core-real-reuse-draft-20260911.md) at 12:18:18 UTC for
the named isolated-server upload and ODA compilation/focused tests. That approval
is effective, but does not include arbitrary additional payloads or a formal
matrix. Qualification of those bytes may proceed while new-factorial preparation
continues; any later changed bytes need their applicable validation. Old results,
failures and stopped queues remain immutable.

Actual active-goal readback and before/after accounting are recorded under
`outputs/diagnostics/codex_goal_revision_20260911/`. File edits alone do not prove
the app's goal was updated.
