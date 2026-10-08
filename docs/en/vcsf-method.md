# VCSF Configuration Selection

[English](vcsf-method.md) | [简体中文](../zh_CN/vcsf-method.md)

VCSF remains one method name. The current selection is A10, with complete
parameters in [the selection status record](../../vcsf-selection-status.json)
and parameter SHA-256
`2551944131096c1d74011f2eb8d789b858e6dce79748f5b5aada54ea18254328`.
The recorded configuration selection is closed. The current numerical source
identity is projected separately in [the A10 identity record](../../vcsf-current-scientific-identity.json).
This documentation update does not publish the executable source or accept new results. The
earlier A01/common-group-26 identity is retained as historical evidence, not a
second public method or a source of A10 results. The prior A23 record is retained
unchanged in [the historical projection](../history/previous-a23-selection-status.json).

## Configuration

| Setting | Selected A10 value |
| --- | --- |
| Perturbation bound | 4/255 in the main comparison |
| Step size | 1/255 |
| Update schedule | Detector initialization followed by 19 feature updates |
| Momentum | 85/100; initialization momentum is discarded before feature updates |
| Feature selection | Parent levels [0,1,2], with three selected layers per stage and six backbone/neck terms in total |
| Geometry | Shared clean/adversarial geometry; scale range [2/3,4/3]; bilinear interpolation, random placement and zero padding |
| Feature objective | Energy weighting, log-mean-exp aggregation and valid support |
| Gradient processing | Gradient centering in initialization and feature updates |

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
selected A10 implementation or original acceptance receipts. Operator paths
and private execution records are not included. Neither historical file was
changed to represent A10.

## Interpretation

The A10 selection comes from the original retrospective 23-configuration panel on one
Faster R-CNN source, seed 42, using all 5,000 COCO val2017 images and sixteen
targets. Reusing that validation split does not establish independent
confirmation. The recorded criterion prioritizes unrounded source-excluded
attacked BB Mean AP; A10 has the lowest value within that complete family.
The Faster-source final-background controls retain the full A10 background.
A later predeclared equal-source A10/backbone-only supplement addresses the
specific adverse Faster-source alternative, not all 23 configurations.
The full configuration uses more feature terms, and target/secondary-metric
disagreements and negative results remain visible. This is not global,
all-source, all-metric or repeated-seed optimality.

No claim is made of statistical significance, equivalence to A01, measured
speedup, cross-source optimality, or exact equality of physical
compute. Earlier operator results at a different optimization background are
not final-momentum interaction evidence. Additional reference forwards, partial
backward paths, time and memory require their own reported accounting.

DINO is excluded from attack generation, not from offline COCO configuration
selection. It is not an untouched independent selection holdout. Historical
A01 ablations and seed results remain tied to their original background.

The internal A10 result families and public source candidate have separate
scoped acceptance records. This metadata package neither replaces those
records nor contains their executable reproduction or raw result tables.
It grants no project-wide software license and makes no full public-release
or all-entry native GPU validation claim. Current datasets are COCO and VOC;
BDD100K is not a remaining experimental or release dependency.
