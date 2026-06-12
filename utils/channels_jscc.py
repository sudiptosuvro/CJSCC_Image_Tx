import math
import torch
import torch.nn as nn


class AWGN(nn.Module):
    def __init__(self, iq_interleaved: bool = True):
        super().__init__()
        self.iq_interleaved = iq_interleaved

    def forward(self, x: torch.Tensor, snr_db) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        if N0.dim() == 0:
            nv = N0
        else:
            nv = N0.view((x.shape[0],) + (1,) * (x.dim() - 1))
        std = torch.sqrt(nv / 2.0) if (self.iq_interleaved and x.dim() == 2 and x.shape[1] % 2 == 0) else torch.sqrt(nv)
        return x + torch.randn_like(x) * std


class RayleighFlatFading(nn.Module):
    def __init__(self, equalize: bool = True, per_symbol: bool = False,
                 eps: float = 1e-8, eq: str = "zf"):
        super().__init__()
        self.equalize = equalize
        self.per_symbol = per_symbol
        self.eps = float(eps)
        self.eq = eq.lower()

    def forward(self, x: torch.Tensor, snr_db):
        assert x.dim() == 2
        B, N = x.shape
        assert N % 2 == 0
        device, dtype = x.device, x.dtype
        M = N // 2

        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        if self.per_symbol:
            hr = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
            hi = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
        else:
            hr = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            hi = torch.randn(B, 1, device=device, dtype=dtype) / math.sqrt(2.0)
            hr = hr.expand(B, M)
            hi = hi.expand(B, M)

        h = torch.complex(hr, hi)
        h2 = (h.real**2 + h.imag**2)
        x2 = x.view(B, M, 2)
        xc = torch.complex(x2[..., 0], x2[..., 1])

        if N0.dim() == 0:
            std = torch.sqrt(N0 / 2.0)
            nr = torch.randn(B, M, device=device, dtype=dtype) * std
            ni = torch.randn(B, M, device=device, dtype=dtype) * std
        else:
            std = torch.sqrt(N0.view(B, 1).expand(B, M) / 2.0)
            nr = torch.randn(B, M, device=device, dtype=dtype) * std
            ni = torch.randn(B, M, device=device, dtype=dtype) * std

        n = torch.complex(nr, ni)
        y = h * xc + n

        if self.equalize:
            if self.eq == "zf":
                denom = h2.clamp_min(self.eps)
                y = y * torch.conj(h) / denom
            elif self.eq == "mmse":
                # denom = |h|^2 + N0
                if N0.dim() == 0:
                    denom = (h2 + N0).clamp_min(self.eps)
                else:
                    denom = (h2 + N0.view(B, 1).expand(B, M)).clamp_min(self.eps)
                y = y * torch.conj(h) / denom
            else:
                raise ValueError("eq must be 'zf' or 'mmse'")

        y_iq = torch.stack([y.real, y.imag], dim=-1).reshape(B, N)
        return y_iq


class RicianChannel(nn.Module):
    def __init__(self, K: float = 5.0, equalize: bool = True, per_symbol: bool = False,
                 eps: float = 1e-8, eq: str = "zf"):
        super().__init__()
        self.K = float(K)
        self.equalize = equalize
        self.per_symbol = per_symbol
        self.eps = float(eps)
        self.eq = eq.lower()

    def forward(self, x: torch.Tensor, snr_db):
        assert x.dim() == 2
        B, N = x.shape
        assert N % 2 == 0
        device, dtype = x.device, x.dtype
        M = N // 2

        snr_db = torch.as_tensor(snr_db, device=device, dtype=dtype)
        N0 = 10.0 ** (-snr_db / 10.0)

        los_mag = math.sqrt(self.K / (self.K + 1.0))
        nlos_mag = math.sqrt(1.0 / (self.K + 1.0))

        if self.per_symbol:
            theta = 2.0 * math.pi * torch.rand(B, M, device=device, dtype=dtype)
        else:
            theta = 2.0 * math.pi * torch.rand(B, 1, device=device, dtype=dtype).expand(B, M)

        h_los = torch.complex(torch.cos(theta), torch.sin(theta)) * los_mag
        gr = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
        gi = torch.randn(B, M, device=device, dtype=dtype) / math.sqrt(2.0)
        h_nlos = torch.complex(gr, gi) * nlos_mag
        h = h_los + h_nlos

        h2 = (h.real**2 + h.imag**2)

        x2 = x.view(B, M, 2)
        xc = torch.complex(x2[..., 0], x2[..., 1])

        if N0.dim() == 0:
            std = torch.sqrt(N0 / 2.0)
            nr = torch.randn(B, M, device=device, dtype=dtype) * std
            ni = torch.randn(B, M, device=device, dtype=dtype) * std
        else:
            std = torch.sqrt(N0.view(B, 1).expand(B, M) / 2.0)
            nr = torch.randn(B, M, device=device, dtype=dtype) * std
            ni = torch.randn(B, M, device=device, dtype=dtype) * std
        n = torch.complex(nr, ni)

        y = h * xc + n

        if self.equalize:
            if self.eq == "zf":
                denom = h2.clamp_min(self.eps)
                y = y * torch.conj(h) / denom
            elif self.eq == "mmse":
                if N0.dim() == 0:
                    denom = (h2 + N0).clamp_min(self.eps)
                else:
                    denom = (h2 + N0.view(B, 1).expand(B, M)).clamp_min(self.eps)
                y = y * torch.conj(h) / denom
            else:
                raise ValueError("eq must be 'zf' or 'mmse'")

        y_iq = torch.stack([y.real, y.imag], dim=-1).reshape(B, N)
        return y_iq