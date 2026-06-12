import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def compute_stream_importance(s_grid: torch.Tensor, alpha: float = 0.7, beta: float = 0.3) -> torch.Tensor:
    
    p = s_grid.abs().pow(2)
    mean_energy = p.mean(dim=1)
    temporal_var = p.var(dim=1, unbiased=False)
    return alpha * mean_energy + beta * temporal_var


def rank_based_stream_mapping(
    s_grid: torch.Tensor,
    H_alloc: torch.Tensor,
    importance: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

    B, Ns, Nfft = s_grid.shape
    Xs_mapped = torch.zeros_like(s_grid)
    q = H_alloc.abs().pow(2)
    imp_order_all = []
    ch_order_all = []

    for b in range(B):
        imp_order = torch.argsort(importance[b], descending=True)
        ch_order = torch.argsort(q[b], descending=True)
        Xs_mapped[b, :, ch_order] = s_grid[b, :, imp_order]
        imp_order_all.append(imp_order)
        ch_order_all.append(ch_order)

    imp_order_all = torch.stack(imp_order_all, dim=0)
    ch_order_all = torch.stack(ch_order_all, dim=0)

    aux = {
        "imp_order": imp_order_all,
        "ch_order": ch_order_all,
        "q": q,
    }
    return Xs_mapped, aux


def undo_rank_based_stream_mapping(
    X_hat: torch.Tensor,
    imp_order: torch.Tensor,
    ch_order: torch.Tensor,
) -> torch.Tensor:

    B, Ns, Nfft = X_hat.shape
    s_hat = torch.zeros_like(X_hat)

    for b in range(B):
        s_hat[b, :, imp_order[b]] = X_hat[b, :, ch_order[b]]

    return s_hat


# Clipping distortion
def clip_ofdm_signal(x: torch.Tensor, clip_ratio: float, eps: float = 1e-8) -> torch.Tensor:

    if clip_ratio is None or clip_ratio <= 0:
        return x

    power = x.abs().pow(2).mean(dim=tuple(range(1, x.dim())), keepdim=True) + eps
    amp_thresh = clip_ratio * torch.sqrt(power)

    amp = x.abs()
    phase = x / (amp + eps)

    amp_clipped = torch.minimum(amp, amp_thresh)
    x_clip = amp_clipped * phase

    power_after = x_clip.abs().pow(2).mean(dim=tuple(range(1, x_clip.dim())), keepdim=True) + eps
    x_clip = x_clip * torch.sqrt(power / power_after)

    return x_clip