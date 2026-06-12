from typing import Dict, Tuple
import torch
import torch.nn.functional as F


def real_to_complex_symbols(z: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    B = z.size(0)
    flat = z.reshape(B, -1)
    orig_len = flat.size(1)

    padded = False
    if orig_len % 2 != 0:
        flat = F.pad(flat, (0, 1))
        padded = True

    s = torch.complex(flat[:, 0::2], flat[:, 1::2])  # [B, Nsym]

    meta = {
        "orig_shape": z.shape,
        "orig_len": orig_len,
        "padded": padded,
        "num_complex": s.size(1),
    }
    return s, meta


def complex_symbols_to_real(shat: torch.Tensor, meta: Dict) -> torch.Tensor:
    B = shat.size(0)
    flat_real = torch.stack([shat.real, shat.imag], dim=-1).reshape(B, -1)

    flat_real = flat_real[:, :meta["orig_len"]]
    zhat = flat_real.reshape(meta["orig_shape"])
    return zhat


def build_exponential_pdp(n_taps: int, delay_decay: float, device, dtype) -> torch.Tensor:

    idx = torch.arange(n_taps, device=device, dtype=torch.float32)
    p = torch.exp(-idx / delay_decay)
    p = p / p.sum()
    return p.to(dtype)


def sample_rayleigh_tdl(
    batch_size: int,
    n_taps: int,
    delay_decay: float,
    device,
    dtype=torch.complex64,
) -> torch.Tensor:

    pdp = build_exponential_pdp(n_taps, delay_decay, device, torch.float32)  # [L]
    std = torch.sqrt(pdp / 2.0).view(1, n_taps)  # [1, L]
    hr = torch.randn(batch_size, n_taps, device=device) * std
    hi = torch.randn(batch_size, n_taps, device=device) * std
    h = torch.complex(hr, hi).to(dtype)
    return h


def add_awgn_complex(x: torch.Tensor, snr_db: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    B = x.size(0)

    if snr_db.dim() == 0:
        snr_db = snr_db.expand(B)
    elif snr_db.dim() == 1 and snr_db.size(0) == 1:
        snr_db = snr_db.expand(B)

    sig_pow = x.abs().pow(2).mean(dim=tuple(range(1, x.dim())))  # [B]
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = sig_pow / snr_lin  # [B]

    std = torch.sqrt(noise_var / 2.0)
    while std.dim() < x.dim():
        std = std.unsqueeze(-1)

    nr = torch.randn_like(x.real) * std
    ni = torch.randn_like(x.imag) * std
    noise = torch.complex(nr, ni)

    y = x + noise
    return y, noise_var.view(B, 1, 1)


def apply_tdl_channel_same(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:

    B, Nsym, T = x.shape
    L = h.size(1)
    y = torch.zeros_like(x)

    for l in range(L):
        coeff = h[:, l].view(B, 1, 1)
        if l == 0:
            y = y + coeff * x
        else:
            y[:, :, l:] = y[:, :, l:] + coeff * x[:, :, :-l]
    return y