"""Bound common-anchor constructor and full numerical-source mapping for reuse."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

from .vcsf_core_numerical_mapping import _builder, _dump, _source_bytes
from .vcsf_research_plan import canonical_hash, require


IMPLEMENTATION = "src/lgp/attacks/vcsf_common_anchor_isolated.py"


def map_common_anchor_numerical_sources(current_root, current_sources, original_root, original,
        parameters, projection=None):
    """Map code only; original input/result and cost qualification remain separate."""
    from .vcsf_core_numerical_mapping import map_numerical_sources

    return map_numerical_sources(current_root, current_sources, original_root, original,
        parameters, projection=projection, current_family="common_anchor")


def verify_common_anchor_constructor(root, sources):
    """Verify the type guard and inherited methods under caller-bound imports.

    The caller separately binds the running source loader, complete numerical
    driver, original inputs/results and historical reuse assessment.
    """
    from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorCandidate
    from ..attacks.vcsf_research_isolated import VCSFResearchCandidate
    from ..attacks.vcsf_scale_isolated import VCSFScaleCandidate

    root = Path(root)
    raw = _source_bytes(root, IMPLEMENTATION, sources[IMPLEMENTATION])
    tree = ast.parse(raw)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == "VCSFCommonAnchorCandidate"]
    require(len(classes) == 1, "Missing unique common-anchor candidate class")
    actual = deepcopy(classes[0])
    if actual.body and isinstance(actual.body[0], ast.Expr) \
            and isinstance(actual.body[0].value, ast.Constant) \
            and isinstance(actual.body[0].value.value, str):
        actual.body.pop(0)
    expected = ast.parse('''
class VCSFCommonAnchorCandidate(VCSFScaleCandidate):
    implementation_path = "src/lgp/attacks/vcsf_common_anchor_isolated.py"
    method_name = "VCSF isolated common-anchor ablation"
    def __init__(self, adapter, config: VCSFCommonAnchorConfig):
        if not isinstance(config, VCSFCommonAnchorConfig):
            raise ValueError("Common anchor requires VCSFCommonAnchorConfig")
        super().__init__(adapter, config)
''').body[0]
    require(_dump(actual) == _dump(expected),
        "Common-anchor constructor or inherited numerical class boundary changed")
    candidate = VCSFCommonAnchorCandidate
    require(Path(inspect.getfile(candidate)).resolve() == (root / IMPLEMENTATION).resolve()
        and candidate.__bases__ == (VCSFScaleCandidate,)
        and candidate.__call__ is VCSFScaleCandidate.__call__ is VCSFResearchCandidate.__call__
        and set(candidate.__dict__) <= {"__module__", "__doc__", "implementation_path", "method_name", "__init__"},
        "Imported common-anchor class is not the bound inherited update implementation")
    builder = _builder(root, sources, "common_anchor")
    evidence = dict(status="common_anchor_constructor_checked_not_full_numerical_mapping",
        implementation=IMPLEMENTATION, implementation_sha256=sources[IMPLEMENTATION],
        constructor_ast_sha256=canonical_hash(_dump(actual)), builder=builder,
        type_guard="VCSFCommonAnchorConfig", unchanged_parent_arguments=["adapter", "config"],
        inherited_update_loop="VCSFResearchCandidate.__call__",
        complete_driver_verified=False, original_input_equivalence_assessed=False,
        historical_reuse_qualified=False, formal_execution_admission=False,
        trajectory_identity_claim=False, physical_cost_inheritance=False,
        model_calls=0, AP_evaluations=0)
    evidence["constructor_evidence_sha256"] = canonical_hash(evidence)
    return evidence
