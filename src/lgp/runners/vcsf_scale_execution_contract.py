"""Bound scale-stage execution with original-identity historical reuse."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import tempfile

from ..attacks.vcsf_research_isolated import VCSFResearchCandidate
from ..attacks.vcsf_scale_isolated import VCSFScaleCandidate, VCSFScaleConfig
from ..io import file_digest
from .vcsf_efficacy_contract import child, read_bound, require, runtime_snapshot as base_snapshot, verify_environment
from .vcsf_research_plan import canonical_hash
from .vcsf_scale_plan import PROTOCOL, compile_scale_plan


EXECUTION_PROTOCOL = "vcsf_single_source_scale_execution"
ENTRYPOINT = "experiments/vcsf_single_source_scale.py"
ALIAS = "vcsf_scale_isolated"
CANDIDATE_ID = PROTOCOL
MODE = "single_source_scale_efficacy"
ALLOW_EMPTY_LANES = True
EXTRA_FILES = (ENTRYPOINT, "tools/prepare_vcsf_scale_execution.py",
    "tools/audit_vcsf_scale_execution.py", "tests/test_vcsf_scale_execution.py",
    "tests/test_vcsf_scale_stage_transition.py", "tools/audit_vcsf_scale_workers.py",
    "tests/test_vcsf_scale_workers.py", "experiments/vcsf_scale_operator_points.py",
    "experiments/audit_vcsf_scale_operator_points.py", "tools/run_vcsf_scale_operator_points.py",
    "tools/audit_vcsf_scale_operator_points.py", "tests/test_vcsf_scale_operator_points.py",
    "tests/test_vcsf_scale_point_stage.py", "tools/run_vcsf_scale_points.py",
    "tools/audit_vcsf_scale_points.py", "tools/audit_vcsf_scale_group.py",
    "tools/audit_vcsf_scale_analysis.py", "experiments/vcsf_scale_remaining_points.py",
    "experiments/audit_vcsf_scale_remaining_points.py", "tests/test_vcsf_scale_remaining_points.py",
    "tests/test_vcsf_scale_width_transition.py")
AUTHORITIES = ("docs/research/vcsf-single-source-amendment-20260909.md",
    "docs/research/vcsf-whole-ablation-amendment-20260909.md",
    "docs/research/vcsf-whole-ablation-closure-20260909.md",
    "docs/research/vcsf-exploration-statistics-amendment-20260909.md")
UNCHANGED_SCIENTIFIC_FILES = ("src/lgp/attacks/vcsf_final_candidate.py",
    "src/lgp/attacks/vcsf_research_isolated.py", "src/lgp/attacks/common.py",
    "src/lgp/attacks/base.py", "src/lgp/adapters/openmmlab.py", "src/lgp/modeling.py",
    "src/lgp/data/coco.py", "src/lgp/runners/evaluate.py", "src/lgp/metrics.py",
    "src/lgp/runtime_config.py", "src/lgp/io.py", "configs/models.yaml", "configs/datasets/coco.yaml")


def runtime_snapshot(root):
    result = base_snapshot(root)
    result.update({name: file_digest(child(root, name)) for name in EXTRA_FILES + AUTHORITIES})
    return dict(sorted(result.items()))


def balanced_lanes(groups, sources, devices):
    require(sources == ["faster_rcnn_r50"] and 1 <= len(devices) <= 2
        and len(set(devices)) == len(devices)
        and all(isinstance(d, str) and d.startswith("cuda:") and d[5:].isdigit() for d in devices),
        "Select one or two distinct physical CUDA devices for the single source")
    require(groups and all(g["source"] == sources[0] for g in groups)
        and len({g["group_id"] for g in groups}) == len(groups), "Duplicate or foreign single-source job")
    return [groups[index::len(devices)] for index in range(len(devices))]


def execution_groups(plan, max_images):
    groups = [g for g in plan["groups"] if g["group_id"] in plan["new_group_ids"]]
    require(groups, "No new complete group is scheduled")
    if max_images is not None:
        require(type(max_images) is int and 0 < max_images < 5000, "Invalid diagnostic image limit")
    return groups


def scheduled_stages(plan, selected):
    require(plan["stage"] in ("baseline_and_identity", "operators_at_anchor_width", "remaining_widths")
        and [g["group_id"] for g in selected] == plan["new_group_ids"],
        "Select the complete job set of a separately bound supported stage")
    return [dict(name=plan["stage"], group_ids=[g["group_id"] for g in selected])]


def execution_metadata(plan, plan_sha256, group, max_images):
    return dict(protocol=PROTOCOL, execution_protocol=EXECUTION_PROTOCOL,
        study="vcsf_single_source_scale", variant=group["variant"], seed=group["seed"],
        group_id=group["group_id"], execution_plan_sha256=plan_sha256,
        prepared_plan_sha256=plan["prepared_plan_sha256"], runtime_snapshot_sha256=plan["runtime_snapshot_sha256"],
        selection_namespace=plan["selection_namespace"], independent_confirmation=False,
        smoke_max_images=max_images, formal_metrics_eligible=False)


def normalized_model_config(value):
    if isinstance(value, dict):
        return {key: normalized_model_config(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalized_model_config(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return str(Path(value).resolve())
    return value


def bind_model_configs(registry, historical):
    from mmengine.config import Config
    from ..runtime_config import build_runtime_config

    group = historical['group']
    prefix = '/groups/{:06d}/'.format(group['group_id'])
    roles = [('source', group['source'], 'attack/model/runtime_config.py')]
    roles += [('target', target, 'evaluations/' + target + '/model/runtime_config.py') for target in group['targets']]
    result = []
    with tempfile.TemporaryDirectory(prefix='lgp-scale-config-') as temporary:
        for role, model, suffix in roles:
            paths = [path for path in historical['checked_input_sha256'] if path.endswith(prefix + suffix)]
            require(len(paths) == 1, 'Missing historical resolved config: ' + role + ':' + model)
            path = Path(paths[0])
            require(file_digest(path) == historical['checked_input_sha256'][str(path)], 'Historical config bytes changed')
            original = Config.fromfile(str(path)).to_dict()
            current = build_runtime_config(registry.model(model), registry.dataset('coco'),
                Path(temporary) / role / model, mode='test', test_split='val', dump_config=False).to_dict()
            # Work directories differ by construction; every scientific config field remains checked.
            original.pop('work_dir', None)
            current.pop('work_dir', None)
            old_hash = canonical_hash(normalized_model_config(original))
            require(old_hash == canonical_hash(normalized_model_config(current)),
                'Actual resolved model/preprocessing/evaluator config changed: ' + role + ':' + model)
            result.append(dict(role=role, model=model, original_file=str(path),
                original_file_sha256=file_digest(path), effective_config_sha256=old_hash,
                normalization='exclude_work_dir_resolve_absolute_paths_json_sequences'))
    return result


def map_anchor(registry, historical, old_plan, anchor):
    require(historical.get("status") == "independently_verified_historical_group_in_original_scope"
        and historical.get("errors") == [] and historical.get("scoped_group_evidence_accepted") is True
        and historical.get("complete_images") == 5000
        and historical.get("official_twelve_metric_replay_cells") == 16
        and historical.get("existing_blackbox_bootstrap_cells") == 15
        and historical.get("verified_worker_invocations") == 17,
        "The historical four-term group lacks qualified complete evidence")
    old = historical["group"]
    require(old["variant"] == "levels_two" and old["source"] == anchor["source"] == "faster_rcnn_r50"
        and old["seed"] == anchor["seed"] == 42 and old["images"] == anchor["images"] == 5000
        and old["targets"] == anchor["targets"] == registry.target_ids()
        and old["parameters"]["levels_per_stage"] == 2,
        "Historical anchor is not the exact four-term single-source setting")
    require(old == old_plan["groups"][old["group_id"] - 1]
        and old["parameters_sha256"] == canonical_hash(old["parameters"]), "Historical group relabelled")
    implicit = dict(image_interpolation="bilinear", image_padding="reflect", placement="random")
    require(canonical_hash(dict(old["parameters"], **implicit)) == canonical_hash(anchor["parameters"])
        and canonical_hash(anchor["parameters"]) == anchor["parameters_sha256"],
        "New anchor differs beyond explicit historical geometry defaults")
    config = VCSFScaleConfig.from_mapping(anchor["parameters"])
    config.validate()
    require(not config.is_reference() and asdict(config) == anchor["parameters"]
        and VCSFScaleCandidate.__call__ is VCSFResearchCandidate.__call__
        and file_digest(registry.root / "src/lgp/attacks/vcsf_scale_isolated.py")
            == registry.protocols[EXECUTION_PROTOCOL]["historical_anchor_geometry_implementation_sha256"],
        "Anchor update implementation or reference shortcut changed")
    preserved = {}
    for name in UNCHANGED_SCIENTIFIC_FILES:
        actual = file_digest(child(registry.root, name))
        require(actual == old_plan["runtime_sha256"].get(name), "Historical scientific runtime differs: " + name)
        preserved[name] = actual
    require([c["target"] for c in historical["cells"]] == anchor["targets"]
        and all(c["source"] == anchor["source"] and c["seed"] == 42
            and c["parameters_sha256"] == old["parameters_sha256"]
            and c["image_ids_sha256"] == old_plan["image_ids_sha256"] for c in historical["cells"]),
        "Historical observations have another source, configuration or image set")
    return dict(status="qualified_original_group_mapped_to_exact_effective_anchor",
        original_group_id=old["group_id"], destination_group_id=anchor["group_id"],
        original_variant=old["variant"], destination_variant=anchor["variant"],
        original_parameters_sha256=old["parameters_sha256"], destination_parameters_sha256=anchor["parameters_sha256"],
        explicit_historical_geometry_defaults=implicit, unchanged_scientific_sha256=preserved,
        original_execution_protocol=old_plan["protocol"], destination_execution_protocol=EXECUTION_PROTOCOL,
        scope="same_effective_single_group_conditions_not_same_whole_study_protocol",
        source_group_receipt_sha256=historical["source_group_receipt_sha256"],
        seed_schedule="selected_position", input_transform=None,
        inherited_intervals=False, inherited_analysis_family=False, physical_cost_inheritance=False,
        bitwise_trajectory_equivalence_claimed=False,
        scientific_acceptance=False, independent_confirmation=False)


def compile_execution_plan(registry, historical_path, historical_hash, old_plan_path, old_plan_hash):
    historical = read_bound(historical_path, historical_hash)
    old = read_bound(old_plan_path, old_plan_hash)
    require(historical["execution_plan_sha256"] == old_plan_hash, "Historical audit binds another runtime plan")
    prepared = compile_scale_plan(registry)
    definition = registry.protocols[EXECUTION_PROTOCOL]
    require(definition["parent_protocol"] == PROTOCOL and definition["first_stage"] == "baseline_and_identity"
        and definition["first_stage_variants"] == ["A_h2", "A_h0"]
        and definition["payload_retention"]["mode"] == "keep_all"
        and definition["payload_retention"]["deletion_authorized"] is False,
        "First-stage execution or non-destructive lifecycle scope changed")
    stage = prepared["stages"][0]
    require(stage["stage"] == definition["first_stage"] and stage["variants"] == definition["first_stage_variants"],
        "First-stage scientific job set changed")
    groups = [g for g in prepared["groups"] if g["variant"] in stage["variants"]]
    require([g["variant"] for g in groups] == stage["variants"], "First-stage canonical group order changed")
    mapping = map_anchor(registry, historical, old, groups[0])
    model_configs = bind_model_configs(registry, historical)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=registry.root, text=True).strip()
    require(head == old["base_commit"], "Historical and destination base revisions differ")
    snapshot = runtime_snapshot(registry.root)
    require(prepared["targets"] == old["targets"] == registry.target_ids(), "Destination detector panel differs")
    analysis = definition["first_stage_analysis"]
    require(analysis["coefficients"] == {"A_h2": 1, "A_h0": -1}
        and analysis["family_size"] == 1 and analysis["seed_estimand"] == "conditional_on_seed42_not_resampled"
        and analysis["replicates"] == old["analysis"]["uncertainty"]["replicates"]
        and analysis["seed"] == old["analysis"]["uncertainty"]["seed"]
        and analysis["historical_intervals_reused"] is False, "First-stage paired estimand changed")
    return dict(schema_version=1, status="prepared_scale_stage_pending_admission", protocol=EXECUTION_PROTOCOL,
        parent_protocol=PROTOCOL, definition=definition, definition_sha256=canonical_hash(definition),
        stage=stage["stage"], prepared_plan=prepared, prepared_plan_sha256=canonical_hash(prepared),
        groups=groups, new_group_ids=[groups[1]["group_id"]], reused_group_ids=[groups[0]["group_id"]],
        new_groups=1, new_images=5000, new_evaluations=16, sources=prepared["sources"], targets=prepared["targets"],
        historical_audit_file=str(Path(historical_path).resolve()), historical_audit_sha256=historical_hash,
        historical_plan_file=str(Path(old_plan_path).resolve()), historical_plan_sha256=old_plan_hash,
        historical_mapping=mapping, checkpoint_sha256=old["checkpoint_sha256"],
        model_config_bindings=model_configs,
        annotation_sha256=old["annotation_sha256"], image_ids_sha256=old["image_ids_sha256"],
        packages=old["packages"], base_commit=head, runtime_sha256=snapshot,
        runtime_snapshot_sha256=canonical_hash(snapshot), analysis=analysis, analysis_sha256=canonical_hash(analysis),
        selection_namespace=registry.protocols[PROTOCOL]["selection_namespace"],
        payload_authorization=dict(explicit_owner_confirmation=False, deletion_authorized=False,
            historical_roots_allowed=False, failed_or_partial_groups_allowed=False),
        runner_armed=False, old_run_restart=False, independent_confirmation=False, scientific_acceptance=False,
        formal_metrics_eligible=False, automatic_promotion=False)


def compile_operator_execution_plan(registry, previous_audit_path, previous_audit_hash,
        previous_cost_path, previous_cost_hash, decision_path, decision_hash):
    from .vcsf_scale_plan import compile_stage_plan
    from .vcsf_scale_stage_transition import bind_first_stage

    previous = bind_first_stage(registry, previous_audit_path, previous_audit_hash,
        previous_cost_path, previous_cost_hash, decision_path, decision_hash)
    first = read_bound(previous['first_execution_plan_file'], previous['first_execution_plan_sha256'])
    plan = compile_execution_plan(registry, first['historical_audit_file'], first['historical_audit_sha256'],
        first['historical_plan_file'], first['historical_plan_sha256'])
    stage = compile_stage_plan(registry, 'operators_at_anchor_width', ['cuda:0', 'cuda:1'])
    groups = stage['previous_stage_groups'] + stage['groups']
    require([g['variant'] for g in groups] == ['A_h2', 'A_h0', 'B_h2', 'C_h2', 'D_h2']
        and canonical_hash(plan['prepared_plan']) == stage['prepared_plan_sha256'],
        'Operator-stage scientific configuration set changed')
    plan.update(stage=stage['stage'], groups=groups,
        new_group_ids=[g['group_id'] for g in stage['groups']],
        reused_group_ids=[g['group_id'] for g in stage['previous_stage_groups']],
        new_groups=3, new_images=15000, new_evaluations=48,
        analysis=stage['analysis'], analysis_sha256=stage['analysis_sha256'],
        previous_stage=previous, previous_stage_sha256=canonical_hash(previous),
        previous_analysis_audit_file=str(Path(previous_audit_path).resolve()),
        previous_analysis_audit_sha256=previous_audit_hash,
        previous_cost_audit_file=str(Path(previous_cost_path).resolve()), previous_cost_audit_sha256=previous_cost_hash,
        previous_decision_file=str(Path(decision_path).resolve()), previous_decision_sha256=decision_hash)
    return plan


def verify_execution_plan(registry, plan):
    require(plan.get("status") == "prepared_scale_stage_pending_admission"
        and plan.get("protocol") == EXECUTION_PROTOCOL and plan.get("runner_armed") is False,
        "Unexpected scale execution identity")
    require(plan.get('stage') in ('baseline_and_identity', 'operators_at_anchor_width', 'remaining_widths'),
        'Unknown scale execution stage')
    if plan.get('stage') in ('operators_at_anchor_width', 'remaining_widths'):
        if plan.get('schema_version') == 2:
            require(plan.get('analysis_mode') == 'point_estimate', 'Point execution lost its analysis identity')
            if plan['stage'] == 'remaining_widths':
                from .vcsf_scale_width_transition import compile_remaining_execution_plan as compiler
            else:
                from .vcsf_scale_point_stage import compile_point_execution_plan as compiler
        else:
            require(plan.get('schema_version') == 1 and plan['stage'] == 'operators_at_anchor_width',
                'Unsupported scale execution version')
            compiler = compile_operator_execution_plan
        expected = compiler(registry, plan['previous_analysis_audit_file'],
            plan['previous_analysis_audit_sha256'], plan['previous_cost_audit_file'], plan['previous_cost_audit_sha256'],
            plan['previous_decision_file'], plan['previous_decision_sha256'])
    else:
        expected = compile_execution_plan(registry, plan["historical_audit_file"], plan["historical_audit_sha256"],
            plan["historical_plan_file"], plan["historical_plan_sha256"])
    require(canonical_hash(plan) == canonical_hash(expected), "Scale execution contract or runtime changed")
    return plan


def verify_admission(registry, plan, plan_hash, path, digest, max_images):
    receipt = read_bound(path, digest)
    mode = "formal" if max_images is None else "smoke"
    require(receipt.get("status") == "independently_verified_single_source_scale_execution"
        and receipt.get("errors") == [] and receipt.get("execution_plan_sha256") == plan_hash
        and receipt.get("runtime_snapshot_sha256") == plan["runtime_snapshot_sha256"]
        and receipt.get("analysis_sha256") == plan["analysis_sha256"]
        and receipt.get("historical_audit_sha256") == plan["historical_audit_sha256"]
        and receipt.get("auditor_sha256") == file_digest(registry.root / "tools/audit_vcsf_scale_execution.py")
        and mode in receipt.get("allowed_modes", []) and receipt.get("independent_confirmation") is False,
        "Scale stage lacks a matching independent launch admission")
    if mode == "smoke":
        require(type(max_images) is int and 0 < max_images <= receipt["smoke_max_images"], "Smoke limit exceeds admission")
    else:
        require(receipt.get("all_launch_gates_passed") is True and receipt.get("payload_deletion_authorized") is False,
            "Formal launch gates or non-destructive lifecycle binding missing")
    if plan['stage'] in ('operators_at_anchor_width', 'remaining_widths'):
        require(receipt.get('stage') == plan['stage']
            and receipt.get('previous_stage_sha256') == plan['previous_stage_sha256']
            and all(type(receipt.get(key)) is int and receipt[key] == expected for key, expected in
                (('logical_groups', len(plan['groups'])), ('reused_groups', len(plan['reused_group_ids'])),
                 ('new_groups', plan['new_groups']), ('new_evaluations', plan['new_evaluations']))),
            'Successor-stage admission lost its prior acceptance or complete new job set')
    require(receipt.get("bound_evidence"), "Admission has no execution evidence")
    for item in receipt["bound_evidence"]:
        read_bound(item["file"], item["sha256"])
    return receipt
