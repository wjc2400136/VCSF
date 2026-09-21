# VCSF Scientific Identity

[English](vcsf-method.md) | [简体中文](../zh_CN/vcsf-method.md)

VCSF denotes one frozen configuration. Its complete resolved parameters are in
[the scientific identity record](../../vcsf-scientific-identity.json), with
parameter SHA-256
`158acaf594c74b4f6a9c20fe28d6c2e38ef6ab9230b85f6d0b0b8b7e93dd7e9e`.
The research identifiers `A01` and common group `26` refer to this same selected
configuration, not separate public methods.

## Configuration

| Setting | Frozen value |
| --- | --- |
| Perturbation bound | 4/255 in the main comparison |
| Step size | 1/255 |
| Update schedule | Detector initialization followed by 19 feature updates |
| Momentum | 85/100; initialization momentum is discarded before feature updates |
| Feature selection | Parent levels [0,1,2], with two selected layers per stage and four backbone/neck terms in total |
| Geometry | Shared clean/adversarial geometry; scale range [2/3,4/3]; bilinear interpolation, random placement and zero padding |
| Feature objective | Energy weighting, log-mean-exp aggregation and valid support |
| Gradient processing | Whole-procedure gradient centering |

The table explains the configuration; it is not a replacement parameter file.
The JSON preserves the original resolved numeric values without rounding or
reconstructing them from this table. Its `hash_encoding` field specifies the
canonical serialization used for the parameter and inventory hashes.

## Provenance

The [historical implementation inventory](../../historical-implementation-inventory.json)
contains all 326 file identities bound by the original freeze. Its canonical
SHA-256 is
`0541e0c6236dab892d5321531061049e04810077d0bac62635eff6916346237a`.
This is an inventory of the original producer, not a checksum of a current
distribution or proof that a historical public runner executes the same method.
The identity record binds the original freeze and parameter-report hashes and
the exact JSON pointers used for extraction. It is a public projection, not the
original acceptance receipt. Operator paths and private execution records are
not included.

## Interpretation

The retained configuration does not dominate every ablation control or target.
It was observed in exploration before the final attribution controls were
registered. Single-source attribution and reuse of the full validation split
do not establish independent confirmation. DINO was not a white-box source,
but its target metrics were inspected. Target and secondary-metric disagreements
remain part of the evidence.

No claim is made of independently positive centering efficacy, monotonic benefit
from depth, pure correspondence causality, universal layer/stage optimality,
optimal momentum, statistical significance or exact equality of physical
compute. Earlier operator results at a different optimization background are
not final-momentum interaction evidence. Additional reference forwards, partial
backward paths, time and memory require their own reported accounting.

This metadata package does not itself contain executable reproduction, accepted
result tables, a software license grant or project-wide device validation.
