# VCSF Method

[English](vcsf-method.md) | [简体中文](../zh_CN/vcsf-method.md)

VCSF is the method name used in the manuscript, experiment registry and reports.
The complete implementation settings are in
[the configuration](../../configs/attacks/vcsf.yaml), with parameter SHA-256
`2551944131096c1d74011f2eb8d789b858e6dce79748f5b5aada54ea18254328`.
The [source identity](../../vcsf-current-scientific-identity.json) and
[complete package manifest](../../public-package.json) bind the implementation
and its dependencies. Internal record identifiers are provenance, not additional
method names.

## Configuration

| Setting | Value |
| --- | --- |
| Main perturbation bound | 4/255 |
| Step size | 1/255 |
| Update schedule | One detector-based initialization followed by nineteen feature updates |
| Momentum | 85/100; initialization momentum is reset before feature updates |
| Feature selection | Three backbone-tail and three neck maps, using the registered parent-level lists |
| Geometry | Shared clean/adversarial geometry, scale range [2/3,4/3], bilinear interpolation, random placement and zero padding |
| Feature objective | Clean-feature energy weighting, log-mean-exp aggregation and valid support |
| Gradient processing | Gradient centering in initialization and feature updates |

This table explains the method; it does not replace the configuration file.
Execution retains the complete parameters, gradient paths, reference forwards and
counting units. Feature backwards and reference work must not be described as
free operations or automatically equated to complete-detector backwards.

## Experiment Scope

The public workflow evaluates COCO and Pascal VOC main comparisons, with six
independent sources and sixteen targets in canonical order. See the
[main guide](../current-public-reproduction.md) for input preparation, one/two-GPU
plans, execution and recovery.

Configuration sensitivity used the declared COCO validation population for
retrospective selection. It is not independent confirmation of a configuration
selected with the same data. Its full settings and outcomes remain in the
[research provenance record](../research/vcsf-a10-configuration-recommendation-provenance.md).
Historical records keep their original identities and are not relabelled as
results of a changed implementation.

## Verification

The package guard checks the implementation/configuration/source identities.
CPU checks and bounded diagnostics have their recorded scope; they do not prove
every full workload, checkpoint or scientific claim. New runs require qualified
data/weights, full intended coverage and separate result acceptance.
Publication of project-owned source follows the
[licence scope](../licensing/license-scope.md), not a grant for data, weights or
manuscript assets.
