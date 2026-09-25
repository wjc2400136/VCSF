# VCSF

**Feature-based adversarial transfer across object detectors**

[English](README.md) | [简体中文](README.zh-CN.md)

VCSF studies how perturbations generated on one object detector transfer to
other detector architectures. This publication contains historical A01 method
metadata and the current A23 configuration selection. A23 is not yet an
accepted implementation or result identity.

## Project Overview

| | Scope |
| --- | --- |
| Task | Transfer-based adversarial attacks on object detection |
| Source panel | Six white-box detectors |
| Target panel | Sixteen detectors across two-stage, YOLO, dense/point-based and query/set-based families |
| Main datasets | COCO val2017 and Pascal VOC2007 test |
| Main perturbation bound | L-infinity, 4/255 |
| Method identity | One VCSF method name; A23 selected pending implementation promotion, with no per-target configuration selection |

The research includes main transfer comparisons, controlled ablations, seed
stability, preprocessing, victim training-state and perturbation-budget studies.
The scope above describes the research programme, not a claim that a complete
executable release is already available here.

## Method At A Glance

VCSF combines detector-based initialization with feature updates across backbone
and neck stages. The selected A23 configuration uses shared clean/adversarial
geometry, energy weighting, log-mean-exp aggregation and momentum, without
gradient centering in either initialization or feature updates.

| Component | Selected A23 setting, pending promotion |
| --- | --- |
| Schedule | Detector initialization followed by 19 feature updates |
| Feature selection | Two selected layers per stage, four backbone/neck terms in total |
| Scale range | [2/3, 4/3] |
| Geometry | Bilinear interpolation, random placement, zero padding and valid support |
| Momentum | 0.85 |

Read the [method and interpretation guide](docs/en/vcsf-method.md) for the
selection boundary and scientific limitations. The selected parameters are not
an accepted executing-code identity or a final result.

## Available Materials

- [Current A23 selection status and complete parameters](vcsf-selection-status.json)
- [Historical A01 scientific identity](vcsf-scientific-identity.json)
- [Historical implementation file hashes](historical-implementation-inventory.json)
- [English method guide](docs/en/vcsf-method.md)
- [中文方法说明](docs/zh_CN/vcsf-method.md)

The selected A23 parameter SHA-256 is:

```text
5079f4495309ec03b4650a4f5a19f291115c84b7155e7f50d15b2086b3de9afd
```

The historical A01 identity and implementation inventory remain unchanged.
They identify the original producer, not A23, and are not a validation
certificate for a future public code package.

## Reproduction Status

**This initial publication contains project documentation and scientific
metadata, not the complete executable reproduction package.**

Implementation files, beginner-oriented setup instructions and verified result
artifacts remain pending source, licensing, portability and scientific checks.
No dataset, model checkpoint, private execution log or unaccepted result is
included in this publication. A23 selection alone does not admit a formal run
or transfer any A01 result to A23.

## Scientific Scope

A23 was selected from a retrospective 23-configuration, single-source panel on
reused COCO val2017. This is not independent confirmation. A10 had the panel's
lower primary attacked BB AP with more feature terms; secondary outcomes were
mixed. Target-specific disagreements and negative results must be retained.
These materials establish no A23 six-source or VOC result, measured speedup,
statistical equivalence, cross-source optimality or exactly equal physical
compute.

## License

A project-wide software license has not yet been granted. Source provenance
and applicable third-party notices are being reviewed before code publication.
