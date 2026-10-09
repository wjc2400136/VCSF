"""Scoped numerical-source mapping for original scale/research result reuse."""
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
from pathlib import Path

from .vcsf_research_plan import canonical_hash, require
from .vcsf_stage_evidence import _bound_json


DRIVER = "src/lgp/runners/attack.py"
COMMON_SOURCES = (
    "src/lgp/runners/evaluate.py", "src/lgp/attacks/__init__.py",
    "src/lgp/attacks/vcsf_final_candidate.py", "src/lgp/attacks/vcsf_research_isolated.py",
    "src/lgp/attacks/common.py", "src/lgp/attacks/base.py", "src/lgp/adapters/openmmlab.py",
    "src/lgp/modeling.py", "src/lgp/data/coco.py", "src/lgp/metrics.py",
    "src/lgp/runtime_config.py", "src/lgp/io.py", "configs/models.yaml", "configs/datasets/coco.yaml",
)
_MODES = {
    "research": "layered_efficacy",
    "scale": "single_source_scale_efficacy",
    "core": "single_source_ablation_executor_smoke",
    "operator": "single_source_operator_factorial_execution",
    "common_anchor": "single_source_common_anchor_execution",
}
_DISPATCH = {
    "research": ("vcsf_efficacy_isolation.resolve_research_efficacy", "vcsf_research_isolation.build_research_preflight"),
    "scale": ("vcsf_scale_execution_isolation.resolve_scale_efficacy", "vcsf_scale_isolation.build_scale_preflight"),
    "core": ("vcsf_ablation_executor_isolation.resolve_executor_smoke", "vcsf_ablation_preflight_isolation.build_ablation_preflight"),
    "operator": ("vcsf_operator_execution_isolation.resolve_operator_execution", "vcsf_operator_execution_isolation.build_operator_execution"),
    "common_anchor": ("vcsf_common_anchor_execution_isolation.resolve_common_anchor_execution", "vcsf_common_anchor_execution_isolation.build_common_anchor_execution"),
}


def _dump(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


class _Mode(ast.NodeTransformer):
    def __init__(self, mode, helper=False):
        self.mode, self.helper = mode, helper

    def visit_Name(self, node):
        return ast.Constant(self.mode) if self.helper and node.id == "mode" else node

    def visit_Call(self, node):
        if _dump(node) == _dump(ast.parse('isolated_research.get("mode")', mode="eval").body):
            return ast.Constant(self.mode)
        return self.generic_visit(node)

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return node
        try:
            left, right = ast.literal_eval(node.left), ast.literal_eval(node.comparators[0])
        except (ValueError, TypeError, SyntaxError):
            return node
        if isinstance(node.ops[0], ast.Eq):
            return ast.Constant(left == right)
        if isinstance(node.ops[0], ast.In) and isinstance(right, (tuple, list)):
            return ast.Constant(left in right)
        return node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant) and type(node.test.value) is bool:
            return node.body if node.test.value else node.orelse
        return node


def _dispatch(tree, family):
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_research_components"]
    if not helpers:
        require(family == "research", "Missing original numerical dispatch helper")
        return _DISPATCH[family]
    require(len(helpers) == 1, "Duplicate numerical dispatch helper")
    signature = ast.parse("def _research_components(mode):\n    pass\n").body[0]
    require(_dump(helpers[0].args) == _dump(signature.args) and not helpers[0].decorator_list
        and helpers[0].returns is None, "Numerical dispatch signature or import-time behavior changed")
    helper = _Mode(_MODES[family], helper=True).visit(deepcopy(helpers[0]))
    imported = {}
    for node in helper.body:
        if isinstance(node, ast.ImportFrom):
            require(node.level == 2 and node.module.startswith("attacks."), "Unexpected dispatch import")
            for alias in node.names:
                require(alias.asname is None, "Unexpected dispatch alias")
                imported[alias.name] = node.module[len("attacks."):] + "." + alias.name
        elif isinstance(node, ast.Return):
            require(isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2
                and all(isinstance(v, ast.Name) and v.id in imported for v in node.value.elts),
                "Dispatch does not return the bound resolver and builder")
            resolved = tuple(imported[v.id] for v in node.value.elts)
            require(resolved == _DISPATCH[family], "Different resolver/builder selected")
            return resolved
        else:
            raise RuntimeError("Unrecognized executable dispatch statement")
    raise RuntimeError("Numerical dispatch has no return")


class _DriverBoundary(ast.NodeTransformer):
    """Remove only verified resolver wiring; retain the complete numerical body."""
    def __init__(self, mode):
        self.mode = mode

    def visit_FunctionDef(self, node):
        if node.name == "_research_components":
            return None
        if node.name != "run_attack":
            return node
        node = self.generic_visit(node)
        return node

    def visit_ImportFrom(self, node):
        permitted = {
            "attacks.vcsf_research_isolation": {"resolve_research_preflight", "build_research_preflight"},
            "attacks.vcsf_efficacy_isolation": {"resolve_research_efficacy"},
        }
        if node.level == 2 and node.module in permitted and all(
                n.asname is None and n.name in permitted[node.module] for n in node.names):
            return None
        return node

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1:
            return node
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "research_resolver" and isinstance(node.value, ast.Name):
            require(node.value.id in ("resolve_research_preflight", "resolve_research_efficacy"),
                "Unknown legacy resolver assignment")
            return None
        if isinstance(target, ast.Tuple) and [n.id if isinstance(n, ast.Name) else None for n in target.elts] \
                == ["research_resolver", "research_builder"]:
            require(isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_research_components" and len(node.value.args) == 1
                and isinstance(node.value.args[0], ast.Constant) and node.value.args[0].value == self.mode
                and not node.value.keywords,
                "Numerical dispatch assignment changed")
            return None
        if isinstance(target, ast.Name) and target.id == "compatibility":
            require(isinstance(node.value, ast.IfExp) and isinstance(node.value.body, ast.Dict),
                "Research compatibility boundary changed")
            for index, key in enumerate(node.value.body.keys):
                if isinstance(key, ast.Constant) and key.value == "reason":
                    reason = node.value.body.values[index]
                    allowed = (ast.IfExp, ast.Constant, ast.BoolOp, ast.And, ast.Or,
                        ast.Compare, ast.Is, ast.IsNot, ast.Eq, ast.In, ast.Name, ast.Load, ast.Tuple)
                    require(all(isinstance(n, allowed) for n in ast.walk(reason)),
                        "Compatibility prose contains executable side effects")
                    node.value.body.values[index] = ast.Constant("record_only_compatibility_reason")
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "build_research_preflight":
            require([n.id if isinstance(n, ast.Name) else None for n in node.args] == ["adapter", "parameters"]
                and not node.keywords, "Legacy constructor arguments changed")
            node.func = ast.Name(id="research_builder", ctx=ast.Load())
        return node


def driver_projection(raw, family):
    require(family in _MODES, "Unknown numerical driver family")
    tree = ast.parse(raw)
    dispatch = _dispatch(tree, family)
    require(sum(isinstance(n, ast.FunctionDef) and n.name == "run_attack" for n in tree.body) == 1,
        "Missing unique complete attack driver")
    specialized = _Mode(_MODES[family]).visit(deepcopy(tree))
    projected = _DriverBoundary(_MODES[family]).visit(specialized)
    return dict(source_sha256=hashlib.sha256(raw if isinstance(raw, bytes) else raw.encode()).hexdigest(),
        numerical_ast_sha256=hashlib.sha256(_dump(projected).encode()).hexdigest(),
        bound_mode=_MODES[family], resolver=dispatch[0], builder=dispatch[1],
        normalization="known_mode_and_resolver_wiring_only_preserve_full_numerical_driver",
        compatibility_reason_is_record_only=True, executable_source_modified=False)


def _source_bytes(root, relative, expected):
    path = Path(root)/relative
    require(path == path.resolve() and not any(p.is_symlink() for p in (path, *path.parents)),
        "Numerical source must be a plain original file")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Numerical source bytes changed: " + relative)
    return raw


def _builder(root, sources, family):
    qualified = _DISPATCH[family][1]
    module, name = qualified.split(".")
    relative = "src/lgp/attacks/" + module + ".py"
    tree = ast.parse(_source_bytes(root, relative, sources[relative]))
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(found) == 1, "Missing unique original numerical builder")
    candidate, config = {"research": ("VCSFResearchCandidate", "VCSFResearchConfig"),
        "scale": ("VCSFScaleCandidate", "VCSFScaleConfig"),
        "core": ("VCSFAblationCandidate", "VCSFAblationConfig"),
        "operator": ("VCSFOperatorFactorialCandidate", "VCSFOperatorFactorialConfig"),
        "common_anchor": ("VCSFCommonAnchorCandidate", "VCSFCommonAnchorConfig")}[family]
    expected = ast.parse("def {}(adapter, parameters):\n    return {}(adapter, {}.from_mapping(dict(parameters)))\n".format(
        name, candidate, config)).body[0]
    require(_dump(found[0]) == _dump(expected), "Numerical builder performs additional or changed operations")
    expected_module = {"research": "vcsf_research_isolated", "scale": "vcsf_scale_isolated",
        "core": "vcsf_ablation_isolated", "operator": "vcsf_operator_factorial_isolated",
        "common_anchor": "vcsf_common_anchor_isolated"}[family]
    symbols = {candidate, config}
    bindings = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                require(alias.name != "*", "Builder namespace cannot use wildcard imports")
                if (alias.asname or alias.name.split(".")[0]) in symbols:
                    require(isinstance(node, ast.ImportFrom) and node.level == 1
                        and node.module == expected_module and alias.asname is None
                        and alias.name in symbols, "Numerical builder class import identity differs")
                    bindings.append(alias.name)
        elif isinstance(node, ast.FunctionDef):
            require(node.name not in symbols and not node.decorator_list,
                "Builder class name is rebound or decorated at import")
            require(not any(isinstance(n, ast.Global) and symbols.intersection(n.names) for n in ast.walk(node)),
                "Builder class name may be rebound by a module function")
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None:
                    ast.literal_eval(default)
        elif isinstance(node, ast.Assign):
            require(all(isinstance(t, ast.Name) and t.id not in symbols for t in node.targets),
                "Builder class name is rebound by module assignment")
            ast.literal_eval(node.value)
        else:
            require(isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str), "Unexpected executable builder module statement")
    require(sorted(bindings) == sorted(symbols), "Numerical builder class imports are missing or duplicated")
    return dict(file=relative, sha256=sources[relative], function=name,
        candidate=candidate, config=config, complete_parameter_mapping=True)


def map_numerical_sources(current_root, current_sources, original_root, original, parameters, projection=None,
        *, current_family="core"):
    """Compare bound source/config objects; this does not assess inputs or results."""
    from ..attacks.vcsf_ablation_isolated import VCSFAblationCandidate, VCSFAblationConfig
    from ..attacks.vcsf_research_isolated import VCSFResearchCandidate, VCSFResearchConfig
    from ..attacks.vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig

    require(current_family in ("core", "operator", "common_anchor"), "Unsupported current numerical family")
    if current_family == "common_anchor":
        from ..attacks.vcsf_common_anchor_isolated import VCSFCommonAnchorConfig
        from .vcsf_common_anchor_numerical_mapping import verify_common_anchor_constructor

        constructor_evidence = verify_common_anchor_constructor(current_root, current_sources)
        current_config_type = VCSFCommonAnchorConfig
        current_constructor = "VCSFCommonAnchorCandidate"
    elif current_family == "operator":
        from ..attacks.vcsf_operator_factorial_isolated import VCSFOperatorFactorialConfig
        from .vcsf_operator_numerical_mapping import verify_operator_constructor

        constructor_evidence = verify_operator_constructor(current_root, current_sources)
        current_config_type = VCSFOperatorFactorialConfig
        current_constructor = "VCSFOperatorFactorialCandidate"
    else:
        constructor_evidence = None
        current_config_type = VCSFAblationConfig
        current_constructor = "VCSFAblationCandidate"

    identity = original["implementation"]
    families = {
        ("VCSFResearchCandidate", "VCSFResearchConfig"): ("research", VCSFResearchConfig),
        ("VCSFScaleCandidate", "VCSFScaleConfig"): ("scale", VCSFScaleConfig),
    }
    key = (identity["class_name"], identity["config_class_name"])
    require(key in families, "Original numerical family is not mapped")
    family, config_type = families[key]
    old_sources = original["source_sha256"]
    old_parameters = original["parameters"]
    if family == "research":
        require(isinstance(projection, dict) and projection["original_identity_sha256"] == canonical_hash(original)
            and projection["original_parameters_sha256"] == canonical_hash(old_parameters)
            and canonical_hash(projection["projected_parameters"]) == canonical_hash(parameters),
            "Original research reuse needs its already reconstructed exact anchor projection")
        require(projection["explicit_historical_geometry_defaults"] == dict(
            image_interpolation="bilinear", image_padding="reflect", placement="random"),
            "Different original anchor geometry")
    else:
        require(canonical_hash(old_parameters) == canonical_hash(parameters),
            "Different operator parameters need a separately qualified numerical mapping")
    preserved = {}
    for name in COMMON_SOURCES:
        require(name in old_sources and old_sources[name] == current_sources.get(name),
            "Original numerical dependency differs: " + name)
        _source_bytes(original_root, name, old_sources[name])
        _source_bytes(current_root, name, current_sources[name])
        preserved[name] = old_sources[name]
    scale_name = "src/lgp/attacks/vcsf_scale_isolated.py"
    if family == "scale":
        require(old_sources[scale_name] == current_sources[scale_name], "Original scale operator changed")
        _source_bytes(original_root, scale_name, old_sources[scale_name])
    else:
        require(projection["original_mapping"]["unchanged_scientific_sha256"] == {
            name: current_sources[name] for name in projection["original_mapping"]["unchanged_scientific_sha256"]},
            "Previously accepted anchor mapping no longer has the same numerical source")
        plans = projection["mapping_plan_references"]
        require(plans and all(_bound_json(ref)["runtime_sha256"][scale_name] == current_sources[scale_name]
            for ref in plans), "Accepted original anchor used another scale geometry implementation")
    _source_bytes(current_root, scale_name, current_sources[scale_name])
    if current_family == "core":
        _source_bytes(current_root, "src/lgp/attacks/vcsf_ablation_isolated.py",
            current_sources["src/lgp/attacks/vcsf_ablation_isolated.py"])
    old_config, new_config = config_type.from_mapping(old_parameters), current_config_type.from_mapping(parameters)
    old_config.validate()
    new_config.validate()
    require(canonical_hash(asdict(old_config)) == canonical_hash(old_parameters)
        and canonical_hash(asdict(new_config)) == canonical_hash(parameters)
        and not old_config.is_reference() and not new_config.is_reference(),
        "Numerical parameter mapping or reference shortcut differs")
    if current_family == "core":
        require(VCSFAblationCandidate.__call__ is VCSFScaleCandidate.__call__ is VCSFResearchCandidate.__call__
        and VCSFAblationCandidate.__init__ is VCSFScaleCandidate.__init__ is VCSFResearchCandidate.__init__
        and VCSFAblationCandidate.__bases__ == (VCSFScaleCandidate,)
        and set(VCSFAblationCandidate.__dict__) <= {"__module__", "__doc__", "implementation_path", "method_name"},
        "Numerical constructor, reference shortcut or update loop differs")
    old_builder, current_builder = _builder(original_root, old_sources, family), _builder(current_root, current_sources, current_family)
    old_driver = driver_projection(_source_bytes(original_root, DRIVER, old_sources[DRIVER]), family)
    current_driver = driver_projection(_source_bytes(current_root, DRIVER, current_sources[DRIVER]), current_family)
    require(old_driver["numerical_ast_sha256"] == current_driver["numerical_ast_sha256"],
        "Numerical attack driver differs beyond the declared resolver boundary")
    result = dict(status="mapped_original_numerical_path_pending_full_reuse_assessment",
        original_identity_sha256=canonical_hash(original), parameters_sha256=canonical_hash(parameters),
        original_family=family, unchanged_sources=preserved, original_driver=old_driver,
        current_driver=current_driver, original_builder=old_builder, current_builder=current_builder,
        original_reference_shortcut=False, current_reference_shortcut=False,
        current_constructor=current_constructor, inherited_update_loop="VCSFResearchCandidate.__call__",
        current_input_equivalence_assessed=False, formal_reuse_qualified=False,
        output_byte_identity_claim=False, trajectory_identity_claim=False, physical_cost_inheritance=False,
        new_model_calls=0, new_AP_evaluations=0, source_or_determinism_settings_changed=False)
    if constructor_evidence is not None:
        result["constructor_evidence"] = constructor_evidence
    result["mapping_sha256"] = canonical_hash(result)
    return result
