from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn

from .enc_dec import SCFBEncoder, DSCFBEncoder, DSCFBDecoder
from .lcdg import LCDG, LCDGConfig, PassThrough
from utils.channels_jscc import AWGN, RayleighFlatFading, RicianChannel
from utils.latent import BWMask, PowerNorm


class MinMax01ToM11(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

class MinMaxM11To01(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5


def make_channel(channel_type: str, rician_K: float = 5.0,
                 equalize: bool = True, per_symbol: bool = False):
    ct = channel_type.lower()
    if ct == "awgn":     return AWGN()
    if ct == "rayleigh": return RayleighFlatFading(equalize=equalize,
                                                    per_symbol=per_symbol)
    if ct == "rician":   return RicianChannel(K=rician_K, equalize=equalize,
                                               per_symbol=per_symbol)
    raise ValueError(f"Unknown channel_type={channel_type}")


@dataclass
class MASCConfig:
    # Core dims
    in_channels:     int = 3
    latent_channels: int = 192

    enc_k:  Tuple[int, int, int, int, int] = (9, 5, 5, 5, 5)
    enc_s:  Tuple[int, int, int, int, int] = (2, 2, 1, 1, 1)
    dec_k:  Tuple[int, int, int, int, int] = (5, 5, 5, 5, 5)
    dec_up: Tuple[int, int, int, int, int] = (1, 1, 1, 2, 2)

    # CSI
    csi_dim: int = 16

    # BW
    bwmask_mode:      Literal["prefix", "random"] = "prefix"
    target_power:     float = 1.0
    power_per_sample: bool  = True

    # Channel
    channel_type:      str   = "awgn"
    rician_K:          float = 5.0
    equalize:          bool  = True
    per_symbol_fading: bool  = False

    # Image scaling
    use_img_m11: bool = False

    ablation_mode:         Literal["full", "no_lcdg", "no_csi"] = "full"
    add_compensating_conv: bool = False


def _make_gate(cfg: MASCConfig, channels: int) -> nn.Module:
    if cfg.ablation_mode == "no_lcdg":
        return PassThrough()
    return LCDG(LCDGConfig(
        csi_dim=cfg.csi_dim,
        channels=channels,
    ))


class _CompensatingConv(nn.Module):
    """
    Optional param-matched conv for no_lcdg baseline.
    """
    def __init__(self, C: int) -> None:
        super().__init__()
        mid = max(C // 4, 16)
        self.net = nn.Sequential(
            nn.Conv2d(C, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, C, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


# Encoder
class EncoderStack(nn.Module):
    def __init__(self, cfg: MASCConfig) -> None:
        super().__init__()
        C = cfg.latent_channels
        self.sflm  = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.comps = nn.ModuleList()

        # Block 1 — standard conv
        self.sflm.append(SCFBEncoder(
            cfg.in_channels, C, k=cfg.enc_k[0], stride=cfg.enc_s[0],
            gdn=None, use_res=False,
            use_mix=False)) 

        # Block 2 — depthwise-separable
        self.sflm.append(DSCFBEncoder(
            C, C, k=cfg.enc_k[1], stride=cfg.enc_s[1],
            gdn=None, use_res=False))

        # Blocks 3-5 — depthwise-separable with stride 1
        for i in range(2, 5):
            self.sflm.append(DSCFBEncoder(
                C, C, k=cfg.enc_k[i], stride=cfg.enc_s[i],
                gdn=None, use_res=True))

        for _ in range(5):
            self.gates.append(_make_gate(cfg, C))
            if cfg.ablation_mode == "no_lcdg" and cfg.add_compensating_conv:
                self.comps.append(_CompensatingConv(C))
            else:
                self.comps.append(nn.Identity())

    def forward(self, x: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        for blk, gate, comp in zip(self.sflm, self.gates, self.comps):
            x = blk(x)
            x = gate(x, csi)
            x = comp(x)
        return x


class DecoderStack(nn.Module):
    def __init__(self, cfg: MASCConfig) -> None:
        super().__init__()
        C = cfg.latent_channels
        self.sflm  = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.comps = nn.ModuleList()

        # Blocks 1-3 — depthwise-separable
        for i in range(3):
            self.sflm.append(DSCFBDecoder(
                C, C, k=cfg.dec_k[i], upsample=cfg.dec_up[i],
                igdn=None, use_res=(cfg.dec_up[i] == 1),
                use_mix=True))

        # Blocks 4-5 — depthwise-separable with upsampling
        for i in range(3, 5):
            self.sflm.append(DSCFBDecoder(
                C, C, k=3, upsample=cfg.dec_up[i],
                igdn=None, use_res=(cfg.dec_up[i] == 1),
                use_mix=False))

        for _ in range(5):
            self.gates.append(_make_gate(cfg, C))
            if cfg.ablation_mode == "no_lcdg" and cfg.add_compensating_conv:
                self.comps.append(_CompensatingConv(C))
            else:
                self.comps.append(nn.Identity())

        self.to_rgb = nn.Conv2d(C, cfg.in_channels,
                                kernel_size=1, padding=0, bias=True)

    def forward(self, z: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        for blk, gate, comp in zip(self.sflm, self.gates, self.comps):
            z = blk(z)
            z = gate(z, csi)
            z = comp(z)
        return self.to_rgb(z)


# CJSCC
class CJSCC(nn.Module):
    def __init__(self, cfg: MASCConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.enc    = EncoderStack(cfg)
        self.dec    = DecoderStack(cfg)
        self.bwmask = BWMask(mode=cfg.bwmask_mode)
        self.pnorm  = PowerNorm(target_power=cfg.target_power,
                                per_sample=cfg.power_per_sample)
        self.channel = make_channel(cfg.channel_type, cfg.rician_K,
                                    cfg.equalize, cfg.per_symbol_fading)

        self.use_img_m11 = bool(cfg.use_img_m11)
        self.img_norm   = MinMax01ToM11() if self.use_img_m11 else nn.Identity()
        self.img_denorm = MinMaxM11To01() if self.use_img_m11 else nn.Identity()

    def forward(self, x: torch.Tensor, *, snr_db, rho: float,
                csi: torch.Tensor):
        if self.cfg.ablation_mode == "no_csi":
            csi = torch.zeros_like(csi)

        x0 = self.img_norm(x)
        z  = self.enc(x0, csi)
        z_shape = z.shape
        z, mask = self.bwmask(z, rho)
        zf = z.flatten(1)
        assert zf.shape[1] % 2 == 0, \
            f"Channel expects even latent length, got {zf.shape[1]}"

        zf     = self.pnorm(zf)
        zf_hat = self.channel(zf, snr_db)
        z_hat = zf_hat.view(z_shape)
        z_hat = z_hat * mask
        x_hat = self.dec(z_hat, csi)
        x_hat = self.img_denorm(x_hat)

        return x_hat, {"mask": mask, "z_shape": z_shape}