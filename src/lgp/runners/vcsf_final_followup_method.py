"""Read-only frozen method inspection, not an execution or reuse admission."""
from dataclasses import asdict
import inspect
from pathlib import Path

from ..attacks.vcsf_common_anchor_isolated import (
    VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig,
)
from ..attacks.vcsf_research_isolated import VCSFResearchCandidate
from ..attacks.vcsf_scale_isolated import VCSFScaleCandidate
from ..io import file_digest
from .vcsf_final_followup_plan import prepare_final_followup_plan
from .vcsf_research_plan import canonical_hash, require


METHOD_FILES = (
    'src/lgp/attacks/base.py',
    'src/lgp/attacks/common.py',
    'src/lgp/attacks/vcsf_final_candidate.py',
    'src/lgp/attacks/vcsf_research_isolated.py',
    'src/lgp/attacks/vcsf_scale_isolated.py',
    'src/lgp/attacks/vcsf_common_anchor_isolated.py',
)


def inspect_frozen_followup_method(registry, freeze_bytes, input_report_bytes):
    """Authenticate producer identity and inspect the unchanged numerical chain.

    This loads no detector and creates no attack instance. Source bytes on disk
    and config parsing do not establish loaded-process or trajectory equivalence.
    """
    plan = prepare_final_followup_plan(
        registry, freeze_bytes, input_report_bytes, ['cuda:0']
    )
    root = Path(registry.root).resolve(strict=True)
    checked = {}
    for name in METHOD_FILES:
        path = (root / name).resolve(strict=True)
        require(root in path.parents, 'Method source escapes the project root')
        digest = file_digest(path)
        require(digest == plan['producer_runtime_sha256'].get(name),
            'Frozen numerical method source differs: ' + name)
        checked[name] = digest

    for cls in (VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig,
                VCSFScaleCandidate, VCSFResearchCandidate):
        module_path = Path(inspect.getfile(cls)).resolve(strict=True)
        require(root in module_path.parents
            and module_path.relative_to(root).as_posix() in checked,
            'Imported numerical class comes from another source tree')
    require(VCSFCommonAnchorCandidate.__call__ is VCSFResearchCandidate.__call__
        and VCSFScaleCandidate.__call__ is VCSFResearchCandidate.__call__,
        'Frozen inherited update implementation differs')
    config = VCSFCommonAnchorConfig.from_mapping(plan['parameters'])
    config.validate()
    resolved_hash = canonical_hash(asdict(config))
    require(resolved_hash == canonical_hash(plan['parameters'])
        and not config.is_reference(),
        'Final parameters changed during config resolution or use reference shortcut')
    return dict(
        schema_version=1,
        status='frozen_method_disk_and_config_checked_not_execution_qualified',
        freeze_sha256=plan['freeze_sha256'],
        parameters_sha256=resolved_hash,
        producer_inventory_sha256=canonical_hash(plan['producer_runtime_sha256']),
        checked_method_source_sha256=checked,
        candidate_class='lgp.attacks.vcsf_common_anchor_isolated.VCSFCommonAnchorCandidate',
        config_class='lgp.attacks.vcsf_common_anchor_isolated.VCSFCommonAnchorConfig',
        update_class='lgp.attacks.vcsf_research_isolated.VCSFResearchCandidate',
        loaded_process_binding_verified=False,
        dataset_dispatch_verified=False,
        numerical_bridge_accepted=False,
        checkpoint_qualification_accepted=False,
        reuse_accepted=False,
        formal_execution_admission=False,
        runner_armed=False,
        model_calls=0,
    )
