"""Exact A23 candidate for validation; not registered for public execution."""
from __future__ import annotations

from dataclasses import asdict

from .vcsf_common_anchor_isolated import (
    VCSFCommonAnchorCandidate,
    VCSFCommonAnchorConfig,
)


A23_PARAMETERS_SHA256 = (
    "5079f4495309ec03b4650a4f5a19f291115c84b7155e7f50d15b2086b3de9afd"
)


class VCSFA23Candidate(VCSFCommonAnchorCandidate):
    implementation_path = "src/lgp/attacks/vcsf_a23_isolated.py"
    method_name = "VCSF selected A23 candidate (validation only)"

    def __init__(self, adapter, config: VCSFCommonAnchorConfig):
        from ..runners.vcsf_research_plan import canonical_hash

        if (type(config) is not VCSFCommonAnchorConfig
                or canonical_hash(asdict(config)) != A23_PARAMETERS_SHA256):
            raise ValueError("A23 candidate requires the selected complete parameter map")
        super().__init__(adapter, config)
