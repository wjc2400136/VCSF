# VCSF

**Feature-based adversarial transfer across object detectors**

[English](README.md) | [简体中文](README.zh-CN.md)

VCSF studies how perturbations generated on one object detector transfer to
other detector architectures. The current configuration is A10, selected through
the recorded 23-configuration comparison and final-background analysis. This
repository currently distributes scientific metadata and documentation; the
complete executable package remains a separate pending software release.

## Project Overview

| | Scope |
| --- | --- |
| Task | Transfer-based adversarial attacks on object detection |
| Source panel | Six white-box detectors |
| Target panel | Sixteen detectors across two-stage, YOLO, dense/point-based and query/set-based families |
| Main datasets | COCO val2017 and Pascal VOC2007 test |
| Main perturbation bound | L-infinity, 4/255 |
| Method identity | One VCSF method name; the complete A10 configuration, with no per-target configuration selection |

The research includes main transfer comparisons, controlled ablations,
preprocessing, victim training-state and perturbation-budget studies.
The scope above describes the research programme, not a claim that a complete
executable release is already available here.

## Method At A Glance

VCSF combines detector-based initialization with feature updates across backbone
and neck stages. The selected A10 configuration uses shared clean/adversarial
geometry, energy weighting, log-mean-exp aggregation, momentum and gradient
centering. Three levels are selected in each stage, giving six feature terms.

| Component | Selected A10 setting |
| --- | --- |
| Schedule | Detector initialization followed by 19 feature updates |
| Feature selection | Three selected layers per stage, six backbone/neck terms in total |
| Scale range | [2/3, 4/3] |
| Geometry | Bilinear interpolation, random placement, zero padding and valid support |
| Momentum | 0.85 |

Read the [method and interpretation guide](docs/en/vcsf-method.md) for the
selection boundary and scientific limitations. The current identity record
separately binds the selected parameters and nine numerical source files; a
parameter hash alone is not a result-acceptance certificate.

## Available Materials

- [Current A10 selection status and complete parameters](vcsf-selection-status.json)
- [Current A10 numerical source identity](vcsf-current-scientific-identity.json)
- [Previous A23 selection record](docs/history/previous-a23-selection-status.json)
- [Historical A01 scientific identity](vcsf-scientific-identity.json)
- [Historical implementation file hashes](historical-implementation-inventory.json)
- [English method guide](docs/en/vcsf-method.md)
- [中文方法说明](docs/zh_CN/vcsf-method.md)

The selected A10 parameter SHA-256 is:

```text
2551944131096c1d74011f2eb8d789b858e6dce79748f5b5aada54ea18254328
```

The historical A01 identity and implementation inventory remain unchanged.
They identify the original producer, not A10, and are not a validation
certificate for a future public code package.

## Reproduction Status

**This publication contains project documentation and scientific
metadata, not the complete executable reproduction package.**

The current internal source candidate has its own qualified interfaces and
separate accepted result evidence. This metadata publication does not include
that source package or replace its original acceptance records. Third-party
redistribution conditions and final delivery still need closure before the
complete executable package can be published.
No dataset, model checkpoint, private execution log or unaccepted result is
included in this publication. Configuration selection does not itself admit a
formal run or transfer any A01 or A23 result to A10.

The corrected-LGP five-radius numerical inputs and their manuscript comparison
are complete, with 32/255 kept as a separate stress setting. One/two-device
dispatch validation covers all 37 declared maintained paths at their recorded
roles; it does not mean every default study or CLI option ran on native GPUs.
See the [current evidence and delivery status](current-evidence-status.json).
The current working manuscript also restores identity-matched cost records for
ten methods on two common sources and two conditional design comparisons.
Its accepted nine-file source archive is synchronized to the author's private
cloud manuscript project. These are artifact and interpretation updates,
not new model experiments or a controlled baseline speed ranking.
These updates do not publish the private executable package.

## Scientific Scope

A10 was selected from the original retrospective 23-configuration Faster R-CNN
panel on reused COCO val2017, prioritizing unrounded source-excluded attacked
BB Mean AP. The registered final-background controls and a specific two-source
A10/backbone-only comparison informed the recorded decision. The latter is not
a two-source ranking of all 23 configurations. Target-specific disagreements,
negative results and actual cost differences remain part of the evidence.

DINO was excluded from attack generation but observed during offline COCO
configuration selection. Neither DINO nor this reused validation split is an
untouched independent configuration-selection holdout. These metadata records
establish no global optimum, statistical significance, measured speedup or
exactly equal physical compute. BDD100K is outside the current maintained scope.

## License

A project-wide software license has not yet been selected. Third-party
license texts and scoped modification notices have been prepared in a
separate source copy, without changing scientific behavior. The author has
authorized continuing publication preparation; the specific project license
and remaining distribution conditions are distinct from that authorization.
The complete executable package is not published by this metadata update.
