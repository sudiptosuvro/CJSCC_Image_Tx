import math
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn


TensorOrInfo = Union[
    torch.Tensor,
    Tuple[torch.Tensor, Dict[str, torch.Tensor]],
]


class AWGN(nn.Module):
    def __init__(self, iq_interleaved: bool = True):
        super().__init__()
        self.iq_interleaved = iq_interleaved

    def forward(self, x: torch.Tensor, snr_db, return_info: bool = False) -> TensorOrInfo:
        device, dtype = x.device, x.dtype
        B = x.shape[0]

        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        if N0.dim() == 0:
            noise_var = N0
            N0_info = N0.expand(B)
        else:
            noise_var = N0.view((B,) + (1,) * (x.dim() - 1))
            N0_info = N0

        if self.iq_interleaved and x.dim() == 2 and x.shape[1] % 2 == 0:
            std = torch.sqrt(noise_var / 2.0)
        else:
            std = torch.sqrt(noise_var)

        y = x + torch.randn_like(x) * std

        if return_info:
            info = {
                "h_real": torch.ones(B, device=device, dtype=dtype),
                "h_imag": torch.zeros(B, device=device, dtype=dtype),
                "h_abs": torch.ones(B, device=device, dtype=dtype),
                "h_power": torch.ones(B, device=device, dtype=dtype),
                "noise_var": N0_info,
            }
            return y, info

        return y


class RayleighFlatFading(nn.Module):
    def __init__(self, per_symbol: bool = False):
        super().__init__()
        self.per_symbol = per_symbol

    def forward(self, x: torch.Tensor, snr_db, return_info: bool = False) -> TensorOrInfo:
        assert x.dim() == 2, "RayleighFlatFading expects x with shape [B, N]"
        B, N = x.shape
        assert N % 2 == 0, "Input length must be even for I/Q pairing"

        device, dtype = x.device, x.dtype
        M = N // 2

        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        if self.per_symbol:
            hr = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
            hi = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
        else:
            hr0 = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            hi0 = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            hr = hr0.expand(B, M)
            hi = hi0.expand(B, M)

        h = torch.complex(hr, hi)
        h_power_sym = h.real.pow(2) + h.imag.pow(2)
        h_abs_sym = torch.sqrt(h_power_sym.clamp_min(1e-12))

        x_iq = x.view(B, M, 2)
        x_complex = torch.complex(x_iq[..., 0], x_iq[..., 1])

        if N0.dim() == 0:
            std = torch.sqrt(N0 / 2.0)
            N0_info = N0.expand(B)
        else:
            std = torch.sqrt(N0.view(B, 1).expand(B, M) / 2.0)
            N0_info = N0

        nr = torch.randn(B, M, device=device, dtype=dtype) * std
        ni = torch.randn(B, M, device=device, dtype=dtype) * std
        noise = torch.complex(nr, ni)

        y_complex = h * x_complex + noise
        y = torch.stack([y_complex.real, y_complex.imag], dim=-1).reshape(B, N)

        if return_info:
            if self.per_symbol:
                h_real = h.real.mean(dim=1)
                h_imag = h.imag.mean(dim=1)
                h_abs = h_abs_sym.mean(dim=1)
                h_power = h_power_sym.mean(dim=1)
            else:
                h_real = h.real[:, 0]
                h_imag = h.imag[:, 0]
                h_abs = h_abs_sym[:, 0]
                h_power = h_power_sym[:, 0]

            info = {
                "h_real": h_real,
                "h_imag": h_imag,
                "h_abs": h_abs,
                "h_power": h_power,
                "noise_var": N0_info,
            }
            return y, info

        return y


class RicianChannel(nn.Module):
    def __init__(self, K: float = 5.0, per_symbol: bool = False):
        super().__init__()
        self.K = float(K)
        self.per_symbol = per_symbol

    def forward(self, x: torch.Tensor, snr_db, return_info: bool = False) -> TensorOrInfo:
        assert x.dim() == 2, "RicianChannel expects x with shape [B, N]"
        B, N = x.shape
        assert N % 2 == 0, "Input length must be even for I/Q pairing"

        device, dtype = x.device, x.dtype
        M = N // 2

        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        los_mag = math.sqrt(self.K / (self.K + 1.0))
        nlos_mag = math.sqrt(1.0 / (self.K + 1.0))

        if self.per_symbol:
            theta = 2.0 * math.pi * torch.rand(B, M, device=device, dtype=dtype)
            gr = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
            gi = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
        else:
            theta = torch.zeros(B, M, device=device, dtype=dtype)
            gr0 = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            gi0 = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            gr = gr0.expand(B, M)
            gi = gi0.expand(B, M)
        
        h_los = torch.complex(torch.cos(theta), torch.sin(theta)) * los_mag
        h_nlos = torch.complex(gr, gi) * nlos_mag

        h = h_los + h_nlos
        h_power_sym = h.real.pow(2) + h.imag.pow(2)
        h_abs_sym = torch.sqrt(h_power_sym.clamp_min(1e-12))

        x_iq = x.view(B, M, 2)
        x_complex = torch.complex(x_iq[..., 0], x_iq[..., 1])

        if N0.dim() == 0:
            std = torch.sqrt(N0 / 2.0)
            N0_info = N0.expand(B)
        else:
            std = torch.sqrt(N0.view(B, 1).expand(B, M) / 2.0)
            N0_info = N0

        nr = torch.randn(B, M, device=device, dtype=dtype) * std
        ni = torch.randn(B, M, device=device, dtype=dtype) * std
        noise = torch.complex(nr, ni)

        y_complex = h * x_complex + noise
        y = torch.stack([y_complex.real, y_complex.imag], dim=-1).reshape(B, N)

        if return_info:
            if self.per_symbol:
                h_real = h.real.mean(dim=1)
                h_imag = h.imag.mean(dim=1)
                h_abs = h_abs_sym.mean(dim=1)
                h_power = h_power_sym.mean(dim=1)
            else:
                h_real = h.real[:, 0]
                h_imag = h.imag[:, 0]
                h_abs = h_abs_sym[:, 0]
                h_power = h_power_sym[:, 0]

            info = {
                "h_real": h_real,
                "h_imag": h_imag,
                "h_abs": h_abs,
                "h_power": h_power,
                "noise_var": N0_info,
            }
            return y, info

        return y