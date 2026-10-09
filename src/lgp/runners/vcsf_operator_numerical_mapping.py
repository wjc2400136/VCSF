"""Bound operator constructor and full numerical-source mapping for original reuse."""
from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from pathlib import Path

from .vcsf_core_numerical_mapping import _builder, _dump, _source_bytes
from .vcsf_research_plan import canonical_hash, require


IMPLEMENTATION = "src/lgp/attacks/vcsf_operator_factorial_isolated.py"


def map_operator_numerical_sources(current_root, current_sources, original_root, original,
        parameters, projection=None):
    """Verify the actual operator dispatch; input/result qualification stays separate."""
    from .vcsf_core_numerical_mapping import map_numerical_sources

    return map_numerical_sources(current_root, current_sources, original_root, original,
        parameters, projection=projection, current_family="operator")


def verify_operator_constructor(root, sources):
    """Check the type guard and unchanged parent call under caller-bound imports.

    The caller must separately bind the running source loader and verify the
    complete numerical driver, original inputs/results and reuse assessment.
    """
    from ..attacks.vcsf_operator_factorial_isolated import VCSFOperatorFactorialCandidate
    from ..attacks.vcsf_research_isolated import VCSFResearchCandidate
    from ..attacks.vcsf_scale_isolated import VCSFScaleCandidate

    root = Path(root)
    raw = _source_bytes(root, IMPLEMENTATION, sources[IMPLEMENTATION])
    tree = ast.parse(raw)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
        and node.name == "VCSFOperatorFactorialCandidate"]
    require(len(classes) == 1, "Missing unique operator candidate class")
    actual = deepcopy(classes[0])
    if actual.body and isinstance(actual.body[0], ast.Expr) \
            and isinstance(actual.body[0].value, ast.Constant) \
            and isinstance(actual.body[0].value.value, str):
        actual.body.pop(0)
    expected = ast.parse('''
class VCSFOperatorFactorialCandidate(VCSFScaleCandidate):
    implementation_path = "src/lgp/attacks/vcsf_operator_factorial_isolated.py"
    method_name = "VCSF isolated four-term operator factorial at half-width 1/3"
    def __init__(self, adapter, config: VCSFOperatorFactorialConfig):
        if not isinstance(config, VCSFOperatorFactorialConfig):
            raise ValueError("Operator factorial requires VCSFOperatorFactorialConfig")
        super().__init__(adapter, config)
''').body[0]
    require(_dump(actual) == _dump(expected),
        "Operator constructor or inherited numerical class boundary changed")
    candidate = VCSFOperatorFactorialCandidate
    require(Path(inspect.getfile(candidate)).resolve() == (root / IMPLEMENTATION).resolve()
        and candidate.__bases__ == (VCSFScaleCandidate,)
        and candidate.__call__ is VCSFScaleCandidate.__call__ is VCSFResearchCandidate.__call__
        and set(candidate.__dict__) <= {"__module__", "__doc__", "implementation_path", "method_name", "__init__"},
        "Imported operator class is not the bound inherited update implementation")
    builder = _builder(root, sources, "operator")
    evidence = dict(status="operator_constructor_checked_not_full_numerical_mapping",
        implementation=IMPLEMENTATION, implementation_sha256=sources[IMPLEMENTATION],
        constructor_ast_sha256=canonical_hash(_dump(actual)), builder=builder,
        type_guard="VCSFOperatorFactorialConfig", unchanged_parent_arguments=["adapter", "config"],
        inherited_update_loop="VCSFResearchCandidate.__call__",
        complete_driver_verified=False, original_input_equivalence_assessed=False,
        historical_reuse_qualified=False, formal_execution_admission=False,
        trajectory_identity_claim=False, physical_cost_inheritance=False,
        model_calls=0, AP_evaluations=0)
    evidence["constructor_evidence_sha256"] = canonical_hash(evidence)
    return evidence
