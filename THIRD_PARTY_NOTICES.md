# Third-Party Notices

[简体中文](THIRD_PARTY_NOTICES.zh-CN.md)

This distribution retains the complete maintained source package. Project-owned
source is licensed under GPL-3.0-only as set out in [the licence scope](docs/licensing/license-scope.md).
Third-party terms apply to their corresponding material; including a licence text
does not relicense every third-party portion. Scientific implementations and settings
are unchanged; this notice is not an independent-authorship or legal-clearance warranty.

## NumbOD

`src/lgp/attacks/numbod.py` contains the recorded local NumbOD adaptation,
including locally expressed Haar slicing and detector integration rather than
a byte-identical copy of the upstream files. The retained adapted material is
covered by the [original MIT notice](docs/third_party/licenses/NumbOD-MIT.txt):
Copyright (c) 2024 CGCL-codes.

Source: [NumbOD, commit 0b22cbfb020ea20730fafd23e8e4ebe33b3adda7](https://github.com/CGCL-codes/NumbOD/tree/0b22cbfb020ea20730fafd23e8e4ebe33b3adda7),
particularly `attack.py` and `DWT.py`.

## LGP and OpenMMLab-Derived Portions

The recorded attribution covers the derived portions of these local files:
`src/lgp/attacks/lgp.py`, `lgp_assignment.py`, `lgp_fidelity.py`,
`lgp_heatmap.py`, `lgp_query.py`, `lgp_roi.py`, `lgp_source.py`,
`lgp_vfnet.py`, and `lgp_yolo.py` in the same directory. They combine the
registered LGP formulas and source-surface adaptations with local benchmark
orchestration. This is not an assertion that every line in those files is
third-party code.

The narrow registered source scope is the `mmdet/` subtree of
[LGP, commit fce86da91f2dc4a69cc69751806f0caae80e51a3](https://github.com/liguopeng0923/LGP/tree/fce86da91f2dc4a69cc69751806f0caae80e51a3/mmdet),
including `adv/attacks/LGP.py`, `adv/utils/FBS.py`, and
`adv/models/losses/logit_loss_adv.py`. Its
[original Apache-2.0 text](docs/third_party/licenses/LGP-mmdet-Apache-2.0.txt)
retains the OpenMMLab 2018-2023 attribution. This scope does not imply that
the upstream repository root or the entire local project is Apache-licensed.
The corresponding local files carry prominent modification notices.

## OSFD

`src/lgp/attacks/osfd.py` retains the recorded adaptation of the OSFD
random-rotation/Gaussian-blur components and their local detector integration.
Sources: `attack/base/RRB.py` and `attack/ours/OSFD.py` at
[OSFD, commit 3744caf69e60b46012a6895c095f18c33db491a9](https://github.com/wakuwu/OSFD/tree/3744caf69e60b46012a6895c095f18c33db491a9).
The [unchanged upstream GPL version 3 text](docs/third_party/licenses/GPL-3.0-upstream.txt)
is included. Neither a later-version option nor a software copyright holder
is inferred from the licence document's example appendix or its own copyright.
The local file carries a dated modification notice. The integrated source package
is distributed under GPL version 3, with the project's own grant specified as
GPL-3.0-only; the recorded upstream version scope remains unchanged.

## Local Re-Expressions and Shared Helpers

The bounded origin inspection of `src/lgp/attacks/tog.py` and `afog.py`
found release-informed local mathematical re-expressions, not byte-identical
whole-file copies. Port labels alone do not prove copied protected expression,
and this observation does not prove independent authorship or legal clearance.
Their scholarly and pinned implementation references remain in
`configs/attacks/reference.yaml`; no method is excluded or rewritten here.

Expressive origin was not established for all inspected portions of
`src/lgp/attacks/common.py`, `src/lgp/adapters/openmmlab.py`, and
`src/lgp/attacks/lgp_semantic.py`. This index does not attach a fictional
whole-file upstream licence or copyright to those shared helpers. The bounded
scope and source hashes are preserved in
[the origin record](docs/third_party/origin-scope.json).

## Installed Framework Dependencies

The package references, but does not bundle, the pinned framework source,
configuration trees, wheels, model weights, or datasets. MMDetection 3.0.0
uses its [upstream Apache-2.0 licence](https://github.com/open-mmlab/mmdetection/blob/v3.0.0/LICENSE).
MMYOLO 0.6.0 provides the [upstream GPL version 3 text](https://github.com/open-mmlab/mmyolo/blob/v0.6.0/LICENSE),
which is byte-identical to the included GPL text. Separate installation alone
does not settle the licensing of an integrated combined work. The frozen
environment is not changed by this notice preparation.

## Modification Date and Distribution Scope

The file-level notice additions are dated 2026-10-09. Earlier benchmark
adaptations predate that notice date; an unrecorded historical adaptation date
is not invented. These additions change comments and source hashes, not Python
syntax trees or numerical behavior. Original scientific run identities and
historical byte-pinned protocols remain attached to their original bytes.

The third-party licence texts are reproduced without modification. The project's
own code grant is stated separately in [the licence scope](docs/licensing/license-scope.md). No dataset
or photograph permissions are inferred from code licences. The source package
contains no manuscript/photo payload.
