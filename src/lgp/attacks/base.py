from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch


@dataclass
class AttackOutput:
    adversarial_bgr: torch.Tensor
    perturbation: torch.Tensor
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def linf_pixel(self) -> float:
        return float(self.perturbation.detach().abs().max().item())
