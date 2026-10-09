"""Stable evaluation of the registered LGP no-object BCE contracts."""
import torch
import torch.nn.functional as F


def lgp_no_object_terms(logits, kind):
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("LGP no-object terms require finite real logits")
    if kind == 'sigmoid_classes':
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise ValueError("LGP sigmoid-class logits must be nonempty NxC")
        success = (logits <= 0).all(dim=1)
    elif kind == 'objectness':
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits[:, 0]
        if logits.ndim != 1 or logits.shape[0] == 0:
            raise ValueError("LGP objectness logits must be nonempty N or Nx1")
        success = logits <= 0
    else:
        raise ValueError("Unknown LGP no-object contract")
    # Each tracked row, including repeated rows, contributes once to this sum.
    return F.softplus(logits).sum(), success
