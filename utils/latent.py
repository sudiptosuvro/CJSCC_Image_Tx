import math
import torch
import torch.nn as nn


class BWMask(nn.Module):
    def __init__(self, mode: str = "prefix"):
        super().__init__()
        if mode not in ("prefix", "random"):
            raise ValueError(f"Unsupported BWMask mode: {mode}")
        self.mode = mode

    def forward(self, z: torch.Tensor, rho: float):
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0, 1], got {rho}")

        if z.dim() != 4:
            raise ValueError(f"Expected z with shape (B, C, H, W), got {tuple(z.shape)}")

        B, C, H, W = z.shape
        K = max(1, min(C, int(round(float(rho) * C))))

        mask = z.new_zeros(1, C, 1, 1)

        if self.mode == "prefix":
            mask[:, :K, :, :] = 1.0

        elif self.mode == "random":
            idx = torch.randperm(C, device=z.device)[:K]
            mask[:, idx, :, :] = 1.0

        z_masked = z * mask
        return z_masked, mask


class PowerNorm(nn.Module):
    def __init__(self, target_power: float = 1.0, per_sample: bool = True,
                 eps: float = 1e-8, iq_interleaved: bool = True):
        super().__init__()
        self.target_power = float(target_power)
        self.per_sample = bool(per_sample)
        self.eps = float(eps)
        self.iq_interleaved = bool(iq_interleaved)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:
            B, N = z.shape

            if self.iq_interleaved and (N % 2 == 0):
                M = N // 2
                z2 = z.view(B, M, 2)
                p_sym = (z2[..., 0] ** 2 + z2[..., 1] ** 2)

                if self.per_sample:
                    p = p_sym.mean(dim=1, keepdim=True)
                else:
                    p = p_sym.mean()

                scale = math.sqrt(self.target_power) / torch.sqrt(p + self.eps)
                return z * scale

            if self.per_sample:
                p = (z ** 2).mean(dim=1, keepdim=True)
            else:
                p = (z ** 2).mean()
            scale = torch.sqrt(self.target_power / (p + self.eps))
            return z * scale

        elif z.dim() == 4:
            if self.per_sample:
                p = (z ** 2).mean(dim=(1, 2, 3), keepdim=True)
            else:
                p = (z ** 2).mean()
            scale = torch.sqrt(self.target_power / (p + self.eps))
            return z * scale

        else:
            raise ValueError(f"Unsupported z shape {z.shape}, expected (B,N) or (B,C,H,W)")