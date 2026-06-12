import os, json, math, random, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.system_model import EncoderStack, DecoderStack, MASCConfig
from utils.latent import PowerNorm, BWMask
from OFDM.core.ofdm import OFDMConfig, OFDMImportanceAwareMiddleRankedRobust
from OFDM.core.channel_estimator import ResidualChannelEstimatorRobust


class CJSCC_OFDM(nn.Module):

    def __init__(self, cfg: MASCConfig, ofdm_cfg: OFDMConfig, clip_ratio: Optional[float] = None):
        super().__init__()
        self.cfg = cfg
        self.enc = EncoderStack(cfg)
        self.dec = DecoderStack(cfg)
        self.ofdm_cfg = ofdm_cfg
        self.bwmask = BWMask(mode=cfg.bwmask_mode)
        self.pnorm = PowerNorm(
            target_power=cfg.target_power,
            per_sample=cfg.power_per_sample,
            iq_interleaved=True,
        )

        self.ofdm_middle = OFDMImportanceAwareMiddleRankedRobust(
            cfg=ofdm_cfg,
            estimator=ResidualChannelEstimatorRobust(nfft=ofdm_cfg.nfft),
            clip_ratio=clip_ratio,
        )

        self.use_img_m11 = bool(cfg.use_img_m11)

    def img_norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0 if self.use_img_m11 else x

    def img_denorm(self, x: torch.Tensor) -> torch.Tensor:
        return (x + 1.0) * 0.5 if self.use_img_m11 else x

    def forward(
        self,
        x: torch.Tensor,
        *,
        snr_db: torch.Tensor,
        rho: float,
        csi: torch.Tensor,
        use_perfect_csi: bool = False,
        use_perfect_alloc_csi: bool = True,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        x0 = self.img_norm(x)

        # semantic encoder
        z = self.enc(x0, csi)
        z_shape = z.shape

        # latent bandwidth + power normalization
        z, mask = self.bwmask(z, rho)
        
        zf = z.flatten(1)
        assert zf.shape[1] % 2 == 0, f"Expected even latent length, got {zf.shape[1]}"
        zf = self.pnorm(zf)
        z = zf.view(z_shape)

        # OFDM middle
        z_hat, aux_mid = self.ofdm_middle(
            z,
            snr_db=snr_db,
            use_perfect_csi=use_perfect_csi,
            use_perfect_alloc_csi=use_perfect_alloc_csi,
            h=h,
        )
        z_hat = z_hat * mask
        
        # semantic decoder
        x_hat = self.dec(z_hat, csi)
        x_hat = self.img_denorm(x_hat)

        aux = {
            "mask": mask,
            **aux_mid,
        }
        return x_hat, aux