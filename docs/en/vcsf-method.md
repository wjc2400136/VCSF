# VCSF Configuration Selection

[English](vcsf-method.md) | [简体中文](../zh_CN/vcsf-method.md)

VCSF remains one method name. The current selection is A23, with complete
parameters in [the selection status record](../../vcsf-selection-status.json)
and parameter SHA-256
`5079f4495309ec03b4650a4f5a19f291115c84b7155e7f50d15b2086b3de9afd`.
This selects a configuration, not an accepted implementation or result. The
earlier A01/common-group-26 identity is retained as historical evidence, not a
second public method or a source of A23 results.

## Configuration

| Setting | Selected A23 value, pending promotion |
| --- | --- |
| Perturbation bound | 4/255 in the main comparison |
| Step size | 1/255 |
| Update schedule | Detector initialization followed by 19 feature updates |
| Momentum | 85/100; initialization momentum is discarded before feature updates |
| Feature selection | Parent levels [0,1,2], with two selected layers per stage and four backbone/neck terms in total |
| Geometry | Shared clean/adversarial geometry; scale range [2/3,4/3]; bilinear interpolation, random placement and zero padding |
| Feature objective | Energy weighting, log-mean-exp aggregation and valid support |
| Gradient processing | No gradient centering in initialization or feature updates |

The table explains the selection; it is not a replacement parameter file. The
selection JSON preserves resolved numeric values without rounding. Its
`hash_encoding` field specifies the canonical parameter serialization. A
parameter hash does not bind executing code, inputs, checkpoints or results.

## Provenance

The [historical A01 identity record](../../vcsf-scientific-identity.json)
retains parameter SHA-256
`158acaf594c74b4f6a9c20fe28d6c2e38ef6ab9230b85f6d0b0b8b7e93dd7e9e`.
The [historical implementation inventory](../../historical-implementation-inventory.json)
contains all 326 file identities bound by that original freeze. Its canonical
SHA-256 is
`0541e0c6236dab892d5321531061049e04810077d0bac62635eff6916346237a`.
These are projections of the original A01 producer, not checksums of the
selected A23 implementation or original acceptance receipts. Operator paths
and private execution records are not included. Neither historical file was
changed to represent A23.

## Interpretation

The A23 selection comes from a retrospective 23-configuration panel on one
Faster R-CNN source, seed 42, using all 5,000 COCO val2017 images and sixteen
targets. Reusing that validation split does not establish independent
confirmation. A10 had lower attacked BB AP on the panel's primary metric but
used more feature terms, while secondary outcomes were mixed. This is a
provisional efficacy/complexity choice, not a proof of optimality. Target and
secondary-metric disagreements remain part of the evidence.

No claim is made of statistical equivalence to A01, measured speedup from
removing centering, cross-source optimality, or exact equality of physical
compute. Earlier operator results at a different optimization background are
not final-momentum interaction evidence. Additional reference forwards, partial
backward paths, time and memory require their own reported accounting.

No A23 six-source, VOC, preprocessing, training-state, radius or cost result is
accepted here. This metadata package does not contain executable reproduction,
accepted A23 result tables, a software license grant or project-wide device
validation.
