# VCSF

**Feature-based adversarial transfer across object detectors**

[English](README.md) | [简体中文](README.zh-CN.md)

VCSF studies how perturbations generated on one object detector transfer to
other detector architectures. This project brings together a frozen method
configuration, controlled comparisons and reproducibility resources for
object-detection transfer attacks.

## Project Overview

| | Scope |
| --- | --- |
| Task | Transfer-based adversarial attacks on object detection |
| Source panel | Six white-box detectors |
| Target panel | Sixteen detectors across two-stage, YOLO, dense/point-based and query/set-based families |
| Main datasets | COCO val2017 and Pascal VOC2007 test |
| Main perturbation bound | L-infinity, 4/255 |
| Method identity | One frozen VCSF configuration; no per-target configuration selection |

The research includes main transfer comparisons, controlled ablations, seed
stability, preprocessing, victim training-state and perturbation-budget studies.
The scope above describes the research programme, not a claim that a complete
executable release is already available here.

## Method At A Glance

VCSF combines detector-based initialization with feature updates across backbone
and neck stages. The frozen configuration uses shared clean/adversarial geometry,
energy weighting, log-mean-exp aggregation, gradient centering and momentum.

| Component | Frozen setting |
| --- | --- |
| Schedule | Detector initialization followed by 19 feature updates |
| Feature selection | Two selected layers per stage, four backbone/neck terms in total |
| Scale range | [2/3, 4/3] |
| Geometry | Bilinear interpolation, random placement, zero padding and valid support |
| Momentum | 0.85 |

Read the [method and interpretation guide](docs/en/vcsf-method.md) for the
complete identity and scientific limitations. This configuration is not claimed
to be universally optimal or superior to every ablation control and target.

## Available Materials

- [Scientific identity and complete resolved parameters](vcsf-scientific-identity.json)
- [Historical implementation file hashes](historical-implementation-inventory.json)
- [English method guide](docs/en/vcsf-method.md)
- [中文方法说明](docs/zh_CN/vcsf-method.md)

The parameter SHA-256 is:

```text
158acaf594c74b4f6a9c20fe28d6c2e38ef6ab9230b85f6d0b0b8b7e93dd7e9e
```

The historical implementation inventory identifies the original producer. It is
not a checksum or validation certificate for a future public code package.

## Reproduction Status

**This initial publication contains project documentation and scientific
metadata, not the complete executable reproduction package.**

Implementation files, beginner-oriented setup instructions and verified result
artifacts will be added after their source, licensing and portability checks.
No dataset, model checkpoint, private execution log or unaccepted result is
included in this publication. Updates will preserve the frozen method identity
and distinguish diagnostic checks from formal experimental acceptance.

## Scientific Scope

The selected configuration was observed during exploration before final
attribution controls were registered. Reusing a validation split and reporting
multiple seeds do not establish independent confirmation. Target-specific
disagreements and negative results must be retained. No claim of statistical
significance or exactly equal physical compute is made by these materials.

## License

A project-wide software license has not yet been granted. Source provenance
and applicable third-party notices are being reviewed before code publication.
