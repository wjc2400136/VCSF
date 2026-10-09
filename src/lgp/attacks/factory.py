from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple, Type

from ..adapters import OpenMMLabAdapter
from .afog import AFOG, AFOGConfig
from .augtrans import AugTrans, AugTransConfig
from .corrupting_attention import CorruptingAttention, CorruptingAttentionConfig
from .hifa import HIFA, HIFAConfig
from .lgp import LGP, LGPConfig
from .mlfadv import MLFAdv, MLFAdvConfig
from .naa import NAA, NAAConfig
from .numbod import NumbOD, NumbODConfig
from .osfd import OSFD, OSFDConfig
from .sfim import SFIM, SFIMConfig
from .svfta import SVFTA, SVFTAConfig
from .tog import TOG, TOGConfig
from .vcsf_common_anchor_isolated import VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig


ATTACK_TYPES: Dict[str, Tuple[Type[Any], Type[Any]]] = {
    "vcsf": (VCSFCommonAnchorCandidate, VCSFCommonAnchorConfig),
    "tog": (TOG, TOGConfig),
    "lgp": (LGP, LGPConfig),
    "osfd": (OSFD, OSFDConfig),
    "afog": (AFOG, AFOGConfig),
    "numbod": (NumbOD, NumbODConfig),
    "svfta": (SVFTA, SVFTAConfig),
    "sfim_b": (SFIM, SFIMConfig),
    "augtrans": (AugTrans, AugTransConfig),
    "corrupting_attention": (CorruptingAttention, CorruptingAttentionConfig),
    "hifa": (HIFA, HIFAConfig),
    "mlfadv": (MLFAdv, MLFAdvConfig),
    "naa": (NAA, NAAConfig),
}


def available_attacks() -> Tuple[str, ...]:
    return tuple(ATTACK_TYPES)


def build_attack(
    attack_id: str,
    adapter: OpenMMLabAdapter,
    parameters: Mapping[str, Any],
) -> Any:
    """Build one local attack with strict, typed parameter validation."""
    if attack_id == "vcsf":
        from .vcsf_public import build_public_vcsf

        return build_public_vcsf(adapter, parameters)
    try:
        attack_type, config_type = ATTACK_TYPES[attack_id]
    except KeyError as exc:
        raise ValueError(
            "No local executor for '{}'. Choose one of: {}".format(
                attack_id, ", ".join(ATTACK_TYPES)
            )
        ) from exc
    config = config_type.from_mapping(dict(parameters))
    return attack_type(adapter, config)
