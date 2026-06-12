from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.channels import AWGN, RayleighFlatFading, RicianChannel
from model.enc_dec import SCFBEncoder, DSCFBEncoder, DSCFBDecoder
from utils.latent import BWMask, PowerNorm
from lcdg import LCDG, LCDGConfig, PassThrough

class MinMax01ToM11(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

class MinMaxM11To01(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5


def make_channel(channel_type: str, rician_K: float = 5.0,
                 per_symbol: bool = False):
    ct = channel_type.lower()
    if ct == "awgn":
        return AWGN()
    if ct == "rayleigh":
        return RayleighFlatFading(per_symbol=per_symbol)
    if ct == "rician":
        return RicianChannel(K=rician_K, per_symbol=per_symbol)
    raise ValueError(f"Unknown channel_type={channel_type}")

@dataclass
class MASCConfig:
    # Core dims
    in_channels:     int = 3
    latent_channels: int = 144

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

class LearnedComplexEqualizer(nn.Module):
    def __init__(self, csi_dim: int, hidden: int = 96, width: int = 64, depth: int = 4):
        super().__init__()

        self.csi_embed = nn.Sequential(
            nn.Linear(csi_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

        self.global_head = nn.Linear(hidden, 6)
        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        self.film = nn.Linear(hidden, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        layers = []
        layers.append(nn.Conv1d(2, width, kernel_size=1, bias=True))
        layers.append(nn.SiLU())

        for _ in range(depth):
            layers += [
                nn.Conv1d(width, width, kernel_size=5, padding=2, groups=width, bias=False),
                nn.Conv1d(width, width, kernel_size=1, bias=True),
                nn.SiLU(),
            ]

        layers.append(nn.Conv1d(width, 2, kernel_size=1, bias=True))
        self.residual_net = nn.Sequential(*layers)
        nn.init.zeros_(self.residual_net[-1].weight)
        nn.init.zeros_(self.residual_net[-1].bias)
        self.global_scale = nn.Parameter(torch.tensor(-2.0))
        self.res_scale = nn.Parameter(torch.tensor(-2.0))

    def forward(self, y_flat: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        B, N = y_flat.shape
        assert N % 2 == 0, "I/Q vector length must be even"

        M = N // 2
        y = y_flat.view(B, M, 2)

        yr = y[..., 0]
        yi = y[..., 1]

        emb = self.csi_embed(csi)

        p = self.global_head(emb)

        a  = p[:, 0].view(B, 1)
        b  = p[:, 1].view(B, 1)
        c  = p[:, 2].view(B, 1)
        d  = p[:, 3].view(B, 1)
        br = p[:, 4].view(B, 1)
        bi = p[:, 5].view(B, 1)

        rr = a * yr - b * yi + br
        ri = c * yr + d * yi + bi

        global_corr = torch.stack([rr, ri], dim=-1).reshape(B, N)
        y_global = y_flat + 0.3 * torch.sigmoid(self.global_scale) * global_corr

        s = y_global.view(B, M, 2).transpose(1, 2)

        feat = self.residual_net[0](s)
        feat = self.residual_net[1](feat)

        gamma_beta = self.film(emb)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None]
        beta = beta[:, :, None]

        feat = feat * (1.0 + gamma) + beta

        for layer in self.residual_net[2:-1]:
            feat = layer(feat)

        residual = self.residual_net[-1](feat)
        residual = residual.transpose(1, 2).reshape(B, N)

        y_out = y_global + 0.3 * torch.sigmoid(self.res_scale) * residual

        return y_out
        

class CSIToFiLM(nn.Module):
    def __init__(self, csi_dim: int, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(csi_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * channels),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, csi):
        gamma, beta = self.net(csi).chunk(2, dim=1)
        return gamma[:, :, None, None], beta[:, :, None, None]


class LatentRecoveryBlock(nn.Module):
    def __init__(self, channels: int, csi_dim: int, hidden: int = 64):
        super().__init__()
        mid = max(channels // 2, 64)

        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pw1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv2d(mid, channels, 1, bias=True)

        self.film = CSIToFiLM(csi_dim, channels, hidden)
        self.scale = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.pw2.weight)
        nn.init.zeros_(self.pw2.bias)

    def forward(self, x, csi):
        gamma, beta = self.film(csi)

        r = self.dw(x)
        r = r * (1.0 + gamma) + beta
        r = self.pw1(r)
        r = self.act(r)
        r = self.pw2(r)

        return x + 0.2 * torch.sigmoid(self.scale) * r


class LatentRecoveryNet(nn.Module):
    def __init__(self, channels: int, csi_dim: int, depth: int = 6, hidden: int = 64):
        super().__init__()
        self.blocks = nn.ModuleList([
            LatentRecoveryBlock(channels, csi_dim, hidden)
            for _ in range(depth)
        ])

    def forward(self, z, csi):
        for blk in self.blocks:
            z = blk(z, csi)
        return z


class EncoderStack(nn.Module):
    def __init__(self, cfg: MASCConfig) -> None:
        super().__init__()
        C = cfg.latent_channels
        self.sflm  = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.comps = nn.ModuleList()

        self.sflm.append(SCFBEncoder(
            cfg.in_channels, C, k=cfg.enc_k[0], stride=cfg.enc_s[0],
            gdn=None, use_res=False,
            use_mix=False)) 

        self.sflm.append(DSCFBEncoder(
            C, C, k=cfg.enc_k[1], stride=cfg.enc_s[1],
            gdn=None, use_res=False))

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

        for i in range(3):
            self.sflm.append(DSCFBDecoder(
                C, C, k=cfg.dec_k[i], upsample=cfg.dec_up[i],
                igdn=None, use_res=(cfg.dec_up[i] == 1),
                use_mix=True))

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

        self.refine = nn.Sequential(
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, kernel_size=1, bias=True),
            nn.SiLU(),
            nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C, bias=False),
            nn.Conv2d(C, C, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.refine_scale = nn.Parameter(torch.tensor(-3.0))
        self.to_rgb = nn.Conv2d(C, cfg.in_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, z: torch.Tensor, csi: torch.Tensor) -> torch.Tensor:
        for blk, gate, comp in zip(self.sflm, self.gates, self.comps):
            z = blk(z)
            z = gate(z, csi, is_decoder=True)
            z = comp(z)
        z = z + 0.1 * torch.sigmoid(self.refine_scale) * self.refine(z)
        return self.to_rgb(z)


class CJSCC(nn.Module):
    def __init__(self, cfg: MASCConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.enc    = EncoderStack(cfg)
        self.dec    = DecoderStack(cfg)
        self.bwmask = BWMask(mode=cfg.bwmask_mode)
        self.pnorm  = PowerNorm(target_power=cfg.target_power,
                                per_sample=cfg.power_per_sample)
        self.channel = make_channel(
            cfg.channel_type,
            rician_K=cfg.rician_K,
            per_symbol=cfg.per_symbol_fading
        )
        
        self.iq_equalizer = LearnedComplexEqualizer(
            csi_dim=cfg.csi_dim,
            hidden=96,
            width=64,
            depth=4,
        )
        self.latent_recovery = LatentRecoveryNet(
            channels=cfg.latent_channels,
            csi_dim=cfg.csi_dim,
            depth=8,
            hidden=96,
        )

        self.use_img_m11 = bool(cfg.use_img_m11)
        self.img_norm   = MinMax01ToM11() if self.use_img_m11 else nn.Identity()
        self.img_denorm = MinMaxM11To01() if self.use_img_m11 else nn.Identity()

    def apply_feature_mask(self, z: torch.Tensor, rho: float):
        B, C, H, W = z.shape
        K = max(1, min(C, int(round(float(rho) * C))))

        mask_c = torch.zeros(C, device=z.device, dtype=z.dtype)
        if self.cfg.bwmask_mode == "prefix":
            mask_c[:K] = 1.0
        elif self.cfg.bwmask_mode == "random":
            idx = torch.randperm(C, device=z.device)[:K]
            mask_c[idx] = 1.0
        else:
            raise ValueError(f"Unknown bwmask_mode={self.cfg.bwmask_mode}")

        mask = mask_c.view(1, C, 1, 1)
        return z * mask, mask

    def forward(self, x: torch.Tensor, *, snr_db, rho: float,
                csi: torch.Tensor):
        
        if self.cfg.ablation_mode == "no_csi":
            csi = torch.zeros_like(csi)

        x0 = self.img_norm(x)

        csi_enc = csi.clone()
        csi_enc[:, 6:11] = 0.0
        
        z = self.enc(x0, csi_enc)
        z_shape = z.shape

        z, mask = self.apply_feature_mask(z, rho)

        zf = z.flatten(1)
        assert zf.shape[1] % 2 == 0, \
            f"Channel expects even latent length, got {zf.shape[1]}"

        zf     = self.pnorm(zf)
        zf_hat, ch_info = self.channel(zf, snr_db, return_info=True)

        csi_dec = csi.clone()
        
        if self.cfg.ablation_mode != "no_csi":
            csi_dec[:, 6] = ch_info["h_real"]
            csi_dec[:, 7] = ch_info["h_imag"]
            csi_dec[:, 8] = torch.log1p(ch_info["h_abs"])
            csi_dec[:, 9] = torch.log1p(ch_info["h_power"])
            csi_dec[:, 10] = torch.log1p(ch_info["noise_var"])
        
        # I/Q-domain learned equalization before reshaping
        if self.cfg.ablation_mode != "no_lcdg":
            zf_hat = self.iq_equalizer(zf_hat, csi_dec)
        
        z_hat = zf_hat.view(z_shape)
        z_hat = z_hat * mask
        
        # Latent-domain recovery after reshaping
        if self.cfg.ablation_mode != "no_lcdg":
            z_hat = self.latent_recovery(z_hat, csi_dec)
        
        x_hat = self.dec(z_hat, csi_dec)
        x_hat = self.img_denorm(x_hat)

        return x_hat, {"mask": mask, "z_shape": z_shape}