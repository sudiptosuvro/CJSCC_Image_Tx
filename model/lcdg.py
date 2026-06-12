from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(in_dim: int, hidden: int, out_dim: int,
              act=nn.SiLU, dropout: float = 0.0, depth: int = 1,) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), act()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


def _zero_init_last_linear(module: nn.Module) -> None:
    for m in reversed(list(module.modules())):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
            break


class ECA(nn.Module):
    """
    Efficient Channel Attention → Global avg-pool → 1-D conv across channels → sigmoid weights.
    """
    def __init__(self, channels: int, k_size: int = 3) -> None:
        super().__init__()
        if k_size % 2 == 0:
            k_size += 1
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size,
                                padding=(k_size - 1) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.adaptive_avg_pool2d(x, 1)
        y = y.squeeze(-1).transpose(1, 2)
        y = self.conv1d(y)
        y = y.transpose(1, 2).unsqueeze(-1)
        return torch.sigmoid(y)


class LightSpatialGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fuse = nn.Conv2d(2, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(x)


class CSIEmbedder(nn.Module):
    """
    Separate SNR and rho projection.
    """
    def __init__(self, csi_dim: int, emb_dim: int = 16) -> None:
        super().__init__()
        self.snr_proj = nn.Linear(4, emb_dim // 2, bias=True)
        self.rho_proj = nn.Linear(2, emb_dim // 2, bias=True)
        self.act      = nn.SiLU()

    def forward(self, csi: torch.Tensor) -> torch.Tensor:
        snr_feats = torch.cat([csi[:, 0:1], csi[:, 3:6]], dim=1)
        rho_feats = csi[:, 1:3]
        snr_emb   = self.act(self.snr_proj(snr_feats))
        rho_emb   = self.act(self.rho_proj(rho_feats))
        return torch.cat([snr_emb, rho_emb], dim=1)


@dataclass
class LCDGConfig:
    csi_dim:  int
    channels: int
    eca_k:      int   = 3
    csi_hidden: int   = 32
    groups:     int   = 8
    emb_dim:    int   = 16
    dropout: float = 0.0
    init_spatial_scale: float = 0.10
    init_channel_scale: float = 0.10
    init_csi_scale:     float = 0.05
    init_mask_scale:    float = 0.06


class LCDG(nn.Module):
    """
    Lightweight CSI-aware Dynamic Gate.

    Applies CSI-conditioned spatial attention, channel attention, and a mask-survival prior 
    to adapt intermediate feature maps under varying SNR and BW ratios.
    """

    def __init__(self, cfg: LCDGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        C = cfg.channels
        G = cfg.groups
        E = cfg.emb_dim

        if C % G != 0:
            raise ValueError(
                f"LCDG: channels={C} must be divisible by groups={G}. "
            )

        # CSI embedder
        self.csi_embedder = CSIEmbedder(cfg.csi_dim, emb_dim=E)

        # Spatial Gate
        self.spatial_gate = LightSpatialGate()
        self.spatial_cond = _make_mlp(E, cfg.csi_hidden, 2,
                                       dropout=cfg.dropout, depth=2)
        _zero_init_last_linear(self.spatial_cond)

        # Channel Gate 
        self.eca = ECA(C, k_size=cfg.eca_k)
        self.channel_cond = _make_mlp(E, cfg.csi_hidden, 2 * G,
                                       dropout=cfg.dropout, depth=1)
        _zero_init_last_linear(self.channel_cond)

        # Learnable modulation scales
        self.spatial_scale = nn.Parameter(torch.tensor(cfg.init_spatial_scale))
        self.channel_scale = nn.Parameter(torch.tensor(cfg.init_channel_scale))
        self.csi_scale_s   = nn.Parameter(torch.tensor(cfg.init_csi_scale))
        self.csi_scale_c   = nn.Parameter(torch.tensor(cfg.init_csi_scale))
        self.mask_scale    = nn.Parameter(torch.tensor(cfg.init_mask_scale))

        # Fixed positional prior
        idx = torch.linspace(0.0, 1.0, C).view(1, C, 1, 1)
        self.register_buffer("channel_position", idx, persistent=False)

    # SNR-aware mask-survival prior
    def _mask_survival_prior(self, rho: torch.Tensor,
                              snr_norm: torch.Tensor, C: int) -> torch.Tensor:
        B = rho.size(0)
        rho_clamped = rho.clamp(1.0 / 48.0, 1.0)
        rho_v = rho_clamped.view(B, 1, 1, 1)
        pos   = self.channel_position[:, :C]
        prior = torch.sigmoid((rho_v - pos) / 0.08) - 0.5
        snr_gate = torch.sigmoid(-snr_norm).view(B, 1, 1, 1)
        rho_gate = torch.sigmoid(-(rho_v - 0.12) / 0.02)
    
        return prior * snr_gate * rho_gate

    def forward(self, x: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        G     = self.cfg.groups
        gsize = C // G

        assert C == self.cfg.channels, \
            f"LCDG channel mismatch: cfg.channels={self.cfg.channels}, got {C}"
        assert csi.shape == (B, self.cfg.csi_dim), \
            f"CSI shape: expected ({B},{self.cfg.csi_dim}), got {tuple(csi.shape)}"

        snr_norm = csi[:, 0]
        rho      = csi[:, 2].clamp(1e-6, 1.0)
        rho      = torch.where(rho > 1e-7, rho,
                               torch.full_like(rho, 1.0 / 12.0))

        emb = self.csi_embedder(csi)

        # 1. Static spatial attention
        avg_map = x.mean(dim=1, keepdim=True)
        max_map, _ = x.max(dim=1, keepdim=True)
        S_s_static = torch.sigmoid(self.spatial_gate(torch.cat([avg_map, max_map], dim=1)))

        # 2. CSI spatial correction
        cond_s  = self.spatial_cond(emb)
        gamma_s = cond_s[:, 0].view(B, 1, 1, 1)
        beta_s  = cond_s[:, 1].view(B, 1, 1, 1)
        S_s_csi = torch.tanh(gamma_s + beta_s)
        csi_s = self.csi_scale_s.clamp(0.0, 0.15)
        S_s   = (S_s_static + csi_s * S_s_csi).clamp(0.0, 1.0)
        s_scale = self.spatial_scale.clamp(0.0, 0.30)
        x_s = x * (1.0 + s_scale * (S_s - 0.5))

        # 3. Static channel attention
        S_c_static = self.eca(x_s)

        # 4. CSI channel correction
        cond_c  = self.channel_cond(emb)
        gamma_g = cond_c[:, :G]
        beta_g  = cond_c[:, G:]
        gamma_c = gamma_g.repeat_interleave(gsize, dim=1).view(B, C, 1, 1)
        beta_c  = beta_g.repeat_interleave(gsize, dim=1).view(B, C, 1, 1)
        S_c_csi = torch.tanh(gamma_c + beta_c)
        csi_c   = self.csi_scale_c.clamp(0.0, 0.15)
        S_c     = (S_c_static + csi_c * S_c_csi).clamp(0.0, 1.0)
        c_scale = self.channel_scale.clamp(0.0, 0.30)
        x_sc = x_s * (1.0 + c_scale * (S_c - 0.5))

        # 5. SNR-aware mask-survival prior
        mask_prior = self._mask_survival_prior(rho, snr_norm, C)
        m_scale    = self.mask_scale.clamp(0.0, 0.10)
        x_sc = x_sc * (1.0 + m_scale * mask_prior)

        return x_sc


# Ablation helpers
class PassThrough(nn.Module):
    def forward(self, x: torch.Tensor, csi: torch.Tensor = None) -> torch.Tensor:
        return x


class NullCSI_LCDG(nn.Module):
    def __init__(self, lcdg: LCDG) -> None:
        super().__init__()
        self.lcdg = lcdg

    def forward(self, x: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        return self.lcdg(x, torch.zeros_like(csi))
