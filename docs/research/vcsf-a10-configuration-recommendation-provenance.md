# VCSF Configuration Decision: A10

Decision date: 2026-09-30. Decision owner: the delegated execution owner.
Record status: complete selection recommendation awaiting independent scientific
review, public implementation qualification and affected final-result acceptance.
This draft does not close final configuration selection. The public method name is
VCSF; A10 is its immutable internal configuration identity, not a second method.

## Decision

Recommend selecting the registered A10 configuration without combining it with A23 or any
other candidate. This replaces A23 as the intended final configuration. Preserve
the earlier A23 decision and source promotion, all A01 evidence, and every
candidate outcome under its original identity. The reasons for reconsideration
are the author's explicit ablation-based selection standard, the complete
23-configuration review, new final-background controls, paired cost measurements,
and the recorded two-source resolution of the adverse backbone-only comparison.

The evidence supports A10 as the most suitable tested complete configuration under the declared
overall attacked-AP criterion and the registered logical-gradient budget in this
retrospective selection programme. This is not a global, all-source, all-metric
or repeated-seed optimum. It is a decision supported by observed complete
comparisons and final-background attribution, not merely the lowest number.

## Criterion And Timing

The original complete-family criterion prioritizes unrounded source-excluded
BB Mean bbox_mAP over twelve-metric voting. Each target has equal weight; the
source-matched target is excluded. The 23 configurations use COCO val2017,
5,000 images, Faster R-CNN R50, seed 42 and the complete sixteen-target panel.
The criterion is retrospective because these outcomes had already been seen.

After F03 was stronger on the Faster source, the A10/F03 supplement registered
an equal-weight mean of the two source-specific fifteen-target BB Means before
observing the new Swin-T results. The dated execution plan records that sequence.
This supplement tests the material single-surface alternative; it is not a
second full-family multi-source search. No post-Swin target weight, secondary
metric vote or resource veto is introduced. Only a real registered budget or
resource violation can block execution; ordinary measured cost does not create
a new exclusion threshold.

## Complete Family And Alternatives

The companion machine-readable decision retains all 23 unrounded primary ranks,
twelve BB metrics, per-target directions, DINO changes, parameter hashes and
saved operation counts. Its full source remains configuration_review.json.
A10 ranks first at 0.07124858402730408, A23 second at 0.07197039233734034 and
A01 third at 0.07203803004672414. A10 improves primary BB AP by 0.0721808310
percentage points versus A23 and 0.0789446019 points versus A01. These small
differences are descriptive; no significance, stability or equivalence is claimed.

Against A01, A10 has lower primary AP on eleven black-box targets and higher AP
on four. Against A23, it has lower AP on ten and higher AP on five. The complete
target table and secondary metrics remain visible, including less favorable
YOLO-family, small/medium-object and recall behavior. A10 retains first primary
rank in each existing leave-one-architecture-family-out composition check;
this is sensitivity to target composition, not independent replication.

The nested A09/A01/A10 comparison selects one, two and three levels per stage
at primary BB AP 0.07826923774945947, 0.07203803004672414 and
0.07124858402730408. Adding the third level brings a small observed improvement
while increasing feature terms. The existing A01 factorials and eight-operator
and common-background studies explain development history; they are not relabeled
as A10 module ablations. Centering and momentum 0.85 are retained as settings
of the selected complete configuration, not independently established novel
contributions or universally optimal parameters.

Existing A01 multi-source COCO, VOC and related results are retained in their
accepted original scope. They establish what that older configuration achieved,
not A10's cross-source or cross-dataset performance. A10's final six-source and
VOC evidence is still required; A01 is not retained merely to avoid those runs.

## Final-Background Attribution

All nine controls retain the exact A10 background except their registered
changes. Their complete sixteen-target and twelve-metric outcomes are preserved.
The Faster-source BB AP values are:

| Setting | BB AP, unrounded | Question |
| --- | --- | --- |
| A10 | 0.07124858402730408 | Full selected design |
| F01 | 0.13548118702589543 | Identity versus sampled paired scale |
| F02 | 0.08201206500266628 | Three-level neck-only versus joint surfaces |
| F03 | 0.06841397146151937 | Three-level backbone-only versus joint surfaces |
| F04 | 0.11500364499716358 | Independent branch offsets |
| F05 | 0.13613064209308068 | Independent complete branch geometry |
| F06 | 0.08681469155447336 | No initialization plus twenty feature updates |
| F07 | 0.08782654837036764 | Random initialization plus twenty feature updates |
| F08 | 0.07375177773665641 | Uniform instead of clean-energy weighting |
| F09 | 0.0726546225097716 | Mean instead of log-mean-exp aggregation |

The substantial identity/independent-geometry and initialization-schedule
contrasts support the paired-view construction and the budgeted detector-plus-
feature optimization arrangement. F08/F09 support their conditional choices,
not a foreground-localization claim, minimax guarantee or unmeasured interaction.
Geometry controls also change content/support overlap. Initialization contrasts
compare one detector plus nineteen feature updates with twenty feature updates,
not identical follow-up length. Stage controls change representation and feature
term count together, so they do not isolate equal-physical-cost stage causality.

F03 is a genuine adverse result, not excluded because it was called a control.
For Faster, A10 BB AP is 0.07124858402730408 and F03 is
0.06841397146151937; F03 is better by 0.2834612566 points in the mean, although
A10 is better on nine of fifteen targets. For Swin, A10 is
0.11017481438587108 and F03 is 0.12259680634847143; A10 is better on all fifteen
black-box targets. The registered equal-source means are 0.09071169920658759
for A10 and 0.09550538890499541 for F03, favoring A10 by 0.4793689698 points.
This justifies retaining the complete joint design for the stated two-source
criterion while rejecting a universal joint-stage benefit. DINO favors A10 on
both sources but participated in offline selection observations, so it cannot
be presented as untouched independent selection evidence.

## Cost And Exact Identity

The paired cost diagnostic uses six warmup and thirty-six measured images per
source, balanced variant order, one exclusive device and synchronized complete
attack-call timing. It accepts 252 calls, including 36 warmup and 216 measured
calls. Mean seconds for A01/A10/A23 are
1.3437586772100378/1.375799018294654/1.3497042587906536 on Faster and
2.3310333507704653/2.3628884838480086/2.330764607578102 on Swin.
A10 versus A01 costs 2.3843820790% and 1.3665670235% more time in this bounded
diagnostic. Its 114 versus 76 feature terms are not a 50% physical-cost increase.
The full saved resource measurements accompany the decision. F03 physical cost,
calibrated complete-detector backward equivalents and kernel FLOPs are unknown;
none is encoded as zero or used to invent a selection veto.

A10 parameter SHA-256 is
2551944131096c1d74011f2eb8d789b858e6dce79748f5b5aada54ea18254328.
Its exact parameters are copied from the registered selection protocol:
epsilon 4/255, step 1/255, twenty logical gradients, detector initialization,
nineteen feature updates, feature_levels [0,1,2], three levels per backbone and
neck, shared scale [2/3,4/3], momentum 0.85, gradient centering enabled,
clean-energy weighting, log-mean-exp aggregation, valid support, bilinear
interpolation, random placement, zero image padding and feature_eps 1e-6.
This is not uncentered three-level A23/A10 splicing.

The executed numerical source map is pinned in the companion decision, including
vcsf_common_anchor_isolated.py SHA-256
b55c7ebe5a307249a8776c0370a8e149044ef2fbc923954e9f1dddccaa1843da
and numerical-map SHA-256
5f7174ae7ac4f0f49c385955d3cfa64bc8c1d1893f1a43d8e08d9895c3792acd.
The historical actual builder/class ancestry governs identity, not the old
alias-derived implementation label. A new complete public execution hash remains
pending; this decision alone does not certify a changed dispatch path.

## Paper Claim Map And Remaining Gates

Organize the paper as core design questions, final-method attribution and
configuration/cost tradeoffs, rather than internal job order. Explain the paired
geometry, joint representation and budgeted initialization arrangement as one
method. Use the accepted final-background contrasts for substantive claims and
the full family/cost tables for selection rationale. Preserve F03 and the actual
secondary tradeoffs in the main discussion where they affect the joint-design
claim; detailed twelve-metric/target tables may reside in the supplement.

The reviewed decision must be fixed before new final-method formal generation. Public source
promotion requires exact parameter and numerical-path qualification. Current
COCO Common-2 evidence may be reused only after the promoted-identity bridge is
accepted. Other affected COCO/VOC six-source, fixed-500 preprocessing, training,
radius, current cost and qualitative evidence must be accepted under this one
identity before final rows and rankings are published. A01 results are history,
not A10 replacements. Corrected-LGP's author-skipped radius records remain
unaccepted. Final submission selection closure, manuscript acceptance, project-
wide device-mode release and external publication remain open.

No new model calls, AP replays, payload deletion or external release is performed
by publication of this decision. Every original outcome and failed root remains
unchanged. If later material evidence overturns this choice, record a new dated
decision and refresh the affected identity; do not silently revise this record.
