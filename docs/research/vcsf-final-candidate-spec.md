[English](vcsf-final-candidate-spec.md) | [Simplified Chinese](vcsf-final-candidate-spec.zh-CN.md)

# VCSF-Attack Frozen Development-Candidate Specification

Status: frozen development candidate; implementation equivalence verified;
independent formal COCO validation pending.

This document defines the only behavior currently meant by **VCSF-Attack**
(View-Consistent Cross-Stage Feature-Survival Attack). It freezes the method
after the completed development ablations and the independent six-source
implementation preflight. It does not register the method globally and does
not promote development metrics to formal results.

## Authorities and scope

- Executable implementation: `src/lgp/attacks/vcsf_final_candidate.py`.
- Declarative configuration: `configs/attacks/vcsf_final_candidate.yaml`.
- Machine-readable freeze: `docs/research/vcsf-final-candidate-freeze.json`.
- Completed development evidence:
  `outputs/experiments/svfta_cross_stage_claim_confirmation/20260824T132745Z`.
- Scientific interpretation:
  `docs/svfta-cross-stage-claim-scientific-audit.md`.
- Prior-art boundary:
  `docs/research/svfta-cross-stage-prior-art-boundary.md`.
- Implementation preflight:
  `outputs/diagnostics/vcsf_final_candidate_preflight/20260826T024730Z/structural_equivalence.json`.

The implementation is intentionally independent of the research-ablation
class hierarchy. The maintained `SVFTA` registry entry and its source files are
not replaced by this freeze.

## Threat model and budget

For an input image `x` in pixel space, VCSF-Attack produces `x_adv` under

```text
||x_adv - x||_infinity <= 4/255
```

when expressed in normalized image units, equivalently four pixel values on
the 0--255 tensor used by the executor. The fixed comparison schedule uses
twenty complete source-detector backward image-equivalents:

1. one detector-semantic initialization update;
2. nineteen feature-survival updates;
3. zero auxiliary detector forward passes;
4. zero auxiliary detector backward passes;
5. no target-model query, auxiliary surrogate, generator, diffusion model or
   vision-language model.

Ground-truth boxes and labels do not enter the frozen objective. They remain
accepted by the common attack interface but are deliberately ignored because
the selected spatial support is global clean-feature energy.

## Stage 1: detector-semantic initialization

Let `p_j(x)` be differentiable source-detector confidence values. The executor
uses the differentiable post-NMS surface when available and otherwise the
public differentiable pre-NMS fallback. With datatype-safe clipping, it forms

```text
z_j(x) = log p_j(x) - log(1 - p_j(x))
L_init(x) = log sum_j exp(z_j(x)).
```

The first counted backward minimizes `L_init`. This update supplies a
non-stationary detector-semantic displacement before the clean/adversarial
directional feature objective is evaluated. Its optimizer state is discarded
after the update. Discarding that state is the simpler phase boundary, not a
named mechanism or contribution.

## Stage 2: view-consistent cross-stage feature survival

For each of the remaining nineteen updates, one random scale-and-placement
sample `xi_t` is drawn from the fixed interval `[2/3, 4/3]`. The same transform
`T_xi_t` is applied to the clean and current adversarial images. This common
randomness prevents the objective from comparing different sampled views.

The detector is evaluated once on the concatenated pair. The frozen feature
surface contains six maps:

- configured neck levels 0, 1 and 2;
- the final three backbone feature maps.

For level `l` and location `u`, define clean energy and channel directions as

```text
e_l(u) = mean_c F_l(T_xi_t(x))[c,u]^2
fhat_l(u) = F_l(T_xi_t(x))[:,u] / ||F_l(T_xi_t(x))[:,u]||_2
ahat_l(u) = F_l(T_xi_t(x_adv))[:,u] / ||F_l(T_xi_t(x_adv))[:,u]||_2.
```

The per-level surviving directional evidence is

```text
c_l = sum_u e_l(u) <fhat_l(u), ahat_l(u)>
      / max(sum_u e_l(u), 1e-6).
```

The six levels are combined without a learned weight or mixing coefficient:

```text
L_feature = log((1/6) sum_l exp(c_l)).
```

Minimizing this smooth bottleneck suppresses the strongest surviving
clean-aligned direction across the joint backbone/neck surface. The phrase
"survival" refers to this measured clean-aligned directional evidence; it does
not imply a causal interpretation of internal detector semantics.

## Update rule

For each image gradient `g_t`, the executor removes its per-channel spatial
mean and normalizes by the image-wide mean absolute magnitude:

```text
gbar_t = (g_t - mean_HW(g_t)) / max(mean_CHW(|g_t - mean_HW(g_t)|), 1e-12).
```

The initialization update uses `sign(gbar_0)` once and discards its state. The
feature phase starts fresh and uses ordinary first-moment momentum:

```text
m_t = gbar_t                              for the first feature update
m_t = 0.7 m_(t-1) + gbar_t                thereafter
delta_t = Project_[−4,+4](delta_(t-1) - sign(m_t)).
```

Projection also enforces the valid 0--255 image range. Momentum is a standard
optimizer choice, not an innovation claim.

## Frozen configuration and parameter roles

| Field | Value | Role |
|---|---:|---|
| `eps` | `4/255` | comparison protocol |
| `iterations` | `20` | comparison protocol |
| `step_size` | `1/255` | comparison protocol |
| `feature_levels` | `[0,1,2]` | frozen method constant |
| `feature_eps` | `1e-6` | numerical stability only |
| `scale_min` | `2/3` | frozen method constant |
| `scale_max` | `4/3` | frozen method constant |
| `momentum` | `0.7` | frozen optimizer constant |

There is no uncertainty gate, second-moment state, risk carryover, risk check
interval, object-mask coefficient, layer-mixing coefficient, temperature or
edge/incidence parameter in the frozen candidate.

## Development evidence and contribution hierarchy

All values below are unrounded raw-mAP paired differences from the completed
1,000-image development protocol after excluding the six source-matched
white-box cells. Lower is better.

| Question | Paired delta | Coverage | Interpretation |
|---|---:|---:|---|
| scale at the cross-stage surface | `-0.12374351581002638` | 90/90 cells, 6/6 sources, 16/16 targets | dominant retained ingredient |
| cross-stage surface under scale | `-0.022803203951081555` | 86/90, 6/6, 16/16 | material primary objective evidence |
| one-step initializer | `-0.015002332557435922` | 90/90, 6/6, 16/16 | material supporting principle |
| reset versus carryover | `-0.002151926360121533` | 74/90, 6/6, 16/16 | broad but non-material; not a contribution |
| scale-by-surface interaction | `+0.009471140039330521` | negative on 23/90, 0/6, 4/16 | antagonistic; no synergy claim |

The paper-level hierarchy is therefore:

1. one primary detector-specific objective: view-consistent cross-stage global
   feature-survival suppression;
2. one supporting initialization principle: a single counted smooth detector
   log-odds update before feature optimization;
3. one empirical contribution: broad six-source/sixteen-target evaluation
   under a matched twenty-backward budget.

The unequal effect sizes are reported directly. They do not require each
retained choice to contribute equally; they require each named method choice
to be material under the preregistered development rule.

## Novelty and attribution boundary

The reviewed literature already contains transformation-based transfer,
synchronized feature-input views, multi-layer feature attacks, detector
multi-stage objectives, cosine/directional feature losses, staged
output-to-feature optimization, momentum, random symmetry breaking and phase
reset behavior. None of those atoms may be described as independently novel.

The bounded distinction supported by the review is the specific detector
integration frozen here: one counted smooth detector-log-odds displacement,
followed by nineteen synchronized-scale updates over a global clean-energy
weighted six-level backbone-plus-neck directional-survival bottleneck.

Permitted verbs include "formulate", "instantiate", "study" and "evaluate".
Do not use an unqualified "first", "novel scale transform", "novel cosine
loss", "novel multi-layer attack", "novel reset", or a scale/cross-stage
synergy claim. The targeted review is not an exhaustive novelty or patent
search and is not a legal opinion.

## Evidence boundary and next gate

The completed matrix is development-only, single-seed evidence. The
implementation preflight establishes behavioral equivalence and six-source
structural compatibility; it does not estimate AP. VCSF-Attack remains absent
from the global registry and main transfer table.

Before publication-strength performance wording or registry replacement, the
frozen candidate requires one independent, unchanged 5,000-image COCO
`val2017` confirmation under the canonical sixteen-target panel. That future
run may validate or reject the frozen candidate, but it must not tune its
parameters or revise this specification from formal-split outcomes.
