import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from OFDM.core.ofdm_helpers import (
    real_to_complex_symbols,
    complex_symbols_to_real,
    sample_rayleigh_tdl,
    add_awgn_complex,
    apply_tdl_channel_same,
)

from OFDM.core.channel_estimator import (
    ResidualChannelEstimator,
)

from OFDM.core.subcarrier_allocator import (
    compute_stream_importance,
    rank_based_stream_mapping,
    undo_rank_based_stream_mapping,
    clip_ofdm_signal,
)


# Configuration
@dataclass
class OFDMConfig:
    nfft: int = 64             # Number of subcarriers in OFDM
    cp_len: int = 16           # Length of cyclic prefix
    n_pilot: int = 1           # Number of pilot OFDM symbols
    n_taps: int = 8            # Number of multipath components
    delay_decay: float = 4.0   # Exponential power delay profile PDP
    pilot_value: complex = 1 + 0j
    equalizer: str = "mmse"
    estimator: str = "ls"
    eps: float = 1e-8


class OFDMChannel(nn.Module):

    def __init__(self, cfg: OFDMConfig):
        super().__init__()
        self.cfg = cfg

    def make_pilot_symbols(self, B: int, device, dtype) -> torch.Tensor:

        pilot = torch.full(
            (B, self.cfg.n_pilot, self.cfg.nfft),
            fill_value=self.cfg.pilot_value,
            dtype=dtype,
            device=device,
        )
        return pilot

    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        B = z.size(0)
        device = z.device

        # Step 1: latent -> complex symbols
        s, latent_meta = real_to_complex_symbols(z)
        dtype = s.dtype

        # Step 2: pack into OFDM data symbols
        num_data_symbols = s.size(1)
        Ns = math.ceil(num_data_symbols / self.cfg.nfft)
        total_slots = Ns * self.cfg.nfft

        if total_slots > num_data_symbols:
            s = F.pad(s, (0, total_slots - num_data_symbols))

        Xs = s.view(B, Ns, self.cfg.nfft)

        # Step 3: create pilot OFDM symbols
        Xp = self.make_pilot_symbols(B, device, dtype=dtype)
        Xf = torch.cat([Xp, Xs], dim=1)

        # Step 4: OFDM modulation
        xt = torch.fft.ifft(Xf, dim=-1)
        xcp = torch.cat([xt[..., -self.cfg.cp_len:], xt], dim=-1)

        # Step 5: multipath channel
        if h is None:
            h = sample_rayleigh_tdl(
                batch_size=B,
                n_taps=self.cfg.n_taps,
                delay_decay=self.cfg.delay_decay,
                device=device,
                dtype=dtype)
        ycp = apply_tdl_channel_same(xcp, h)
        ycp, noise_var = add_awgn_complex(ycp, snr_db)

        # Step 6: Receiver OFDM
        yt = ycp[..., self.cfg.cp_len:self.cfg.cp_len + self.cfg.nfft]
        Yf = torch.fft.fft(yt, dim=-1)
        Yp = Yf[:, :self.cfg.n_pilot, :]
        Ys = Yf[:, self.cfg.n_pilot:, :]
        Hf_true = torch.fft.fft(h, n=self.cfg.nfft, dim=-1)

        meta = {
            "latent_meta": latent_meta,
            "num_data_symbols": num_data_symbols,
            "Ns": Ns,
            "total_slots": total_slots,
        }

        return {
            "Yp": Yp,
            "Ys": Ys,
            "Xp": Xp,
            "Xs": Xs,
            "Hf_true": Hf_true,
            "h_time": h,
            "noise_var": noise_var,
            "meta": meta,
        }


class ChannelEstimator(nn.Module):

    def __init__(self, cfg: OFDMConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, Yp: torch.Tensor, Xp: torch.Tensor) -> torch.Tensor:
        H_ls = Yp / (Xp + self.cfg.eps)
        H_hat = H_ls.mean(dim=1)
        return H_hat


class Equalizer(nn.Module):

    def __init__(self, cfg: OFDMConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        Ys: torch.Tensor,
        H_hat: torch.Tensor,
        noise_var: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        eq_mode = mode if mode is not None else self.cfg.equalizer
        H = H_hat.unsqueeze(1)

        if eq_mode == "zf":
            X_hat = Ys / (H + self.cfg.eps)
            return X_hat

        if eq_mode == "mmse":
            if noise_var is None:
                raise ValueError("noise_var is required for MMSE equalization.")

            denom = H.abs().pow(2) + noise_var
            X_hat = torch.conj(H) / (denom + self.cfg.eps) * Ys
            return X_hat

        raise ValueError(f"Unsupported equalizer mode: {eq_mode}")


class InverseMapper(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X_hat: torch.Tensor, meta: Dict) -> torch.Tensor:

        B = X_hat.size(0)
        s_hat = X_hat.reshape(B, -1)
        s_hat = s_hat[:, :meta["num_data_symbols"]]
        z_hat = complex_symbols_to_real(s_hat, meta["latent_meta"])
        return z_hat


class OFDMJSCCMiddle(nn.Module):

    def __init__(self, cfg: OFDMConfig):
        super().__init__()
        self.channel = OFDMChannel(cfg)
        self.estimator = ChannelEstimator(cfg)
        self.equalizer = Equalizer(cfg)
        self.inverse_mapper = InverseMapper()
        self.cfg = cfg

    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        use_perfect_csi: bool = False,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ch = self.channel(z, snr_db, h=h)

        if use_perfect_csi:
            H_hat = ch["Hf_true"]
        else:
            H_hat = self.estimator(ch["Yp"], ch["Xp"])

        X_hat = self.equalizer(
            Ys=ch["Ys"],
            H_hat=H_hat,
            noise_var=ch["noise_var"],
            mode=self.cfg.equalizer,
        )

        z_hat = self.inverse_mapper(X_hat, ch["meta"])

        aux = {
            **ch,
            "H_hat": H_hat,
            "X_hat": X_hat,
            "z_hat": z_hat,
        }
        return z_hat, aux


class OFDMJSCCMiddleLearnableCE(nn.Module):
    def __init__(self, cfg: OFDMConfig, use_learnable_ce: bool = True):
        super().__init__()
        self.channel = OFDMChannel(cfg)
        self.ls_estimator = ChannelEstimator(cfg)
        self.equalizer = Equalizer(cfg)
        self.inverse_mapper = InverseMapper()

        self.cfg = cfg
        self.use_learnable_ce = use_learnable_ce
        self.learnable_ce = ResidualChannelEstimator(
            nfft=cfg.nfft,
            hidden_channels=64,
            n_blocks=3
        )

    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        use_perfect_csi: bool = False,
        h: torch.Tensor = None,
    ):
        ch = self.channel(z, snr_db, h=h)

        if use_perfect_csi:
            H_hat = ch["Hf_true"]
        else:
            if self.use_learnable_ce:
                H_hat = self.learnable_ce(ch["Yp"], ch["Xp"])
            else:
                H_hat = self.ls_estimator(ch["Yp"], ch["Xp"])

        X_hat = self.equalizer(
            Ys=ch["Ys"],
            H_hat=H_hat,
            noise_var=ch["noise_var"],
            mode=self.cfg.equalizer,
        )

        z_hat = self.inverse_mapper(X_hat, ch["meta"])

        aux = {
            **ch,
            "H_hat": H_hat,
            "X_hat": X_hat,
            "z_hat": z_hat,
        }
        return z_hat, aux


class OFDMImportanceAwareMiddleRankedRobust(nn.Module):
    def __init__(
        self,
        cfg: OFDMConfig,
        estimator: Optional[nn.Module] = None,
        clip_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.channel = OFDMChannel(cfg)
        self.estimator = estimator if estimator is not None else ChannelEstimator(cfg)
        self.equalizer = Equalizer(cfg)
        self.clip_ratio = clip_ratio

    def forward(
        self,
        z: torch.Tensor,
        snr_db: torch.Tensor,
        use_perfect_csi: bool = False,
        h: Optional[torch.Tensor] = None,
        use_perfect_alloc_csi: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        B = z.size(0)

        # latent -> complex symbols
        s, latent_meta = real_to_complex_symbols(z)
        Nsym_orig = s.size(1)

        Ns = math.ceil(Nsym_orig / self.cfg.nfft)
        total_slots = Ns * self.cfg.nfft

        if total_slots > Nsym_orig:
            pad_len = total_slots - Nsym_orig
            s = torch.cat([s, torch.zeros(B, pad_len, device=s.device, dtype=s.dtype)], dim=1)

        s_grid = s.view(B, Ns, self.cfg.nfft)

        # pilot symbols
        Xp = self.channel.make_pilot_symbols(B, z.device, dtype=s.dtype)

        # channel realization
        if h is None:
            h = sample_rayleigh_tdl(
                batch_size=B,
                n_taps=self.cfg.n_taps,
                delay_decay=self.cfg.delay_decay,
                device=z.device,
                dtype=s.dtype,
            )

        Hf_true = torch.fft.fft(h, n=self.cfg.nfft, dim=-1)
        H_alloc = Hf_true if use_perfect_alloc_csi else Hf_true

        importance = compute_stream_importance(s_grid)

        Xs, map_aux = rank_based_stream_mapping(
            s_grid=s_grid,
            H_alloc=H_alloc,
            importance=importance,
        )

        Xf = torch.cat([Xp, Xs], dim=1)

        # OFDM TX
        xt = torch.fft.ifft(Xf, dim=-1)
        xcp = torch.cat([xt[..., -self.cfg.cp_len:], xt], dim=-1)

        # optional clipping
        xcp = clip_ofdm_signal(xcp, self.clip_ratio)

        # channel + noise
        ycp = apply_tdl_channel_same(xcp, h)
        ycp, noise_var = add_awgn_complex(ycp, snr_db)

        # RX frontend
        yt = ycp[..., self.cfg.cp_len:self.cfg.cp_len + self.cfg.nfft]
        Yf = torch.fft.fft(yt, dim=-1)

        Yp = Yf[:, :self.cfg.n_pilot, :]
        Ys = Yf[:, self.cfg.n_pilot:, :]

        # CE for equalization
        if use_perfect_csi:
            H_hat = Hf_true
        else:
            try:
                H_hat = self.estimator(Yp, Xp, snr_db)
            except TypeError:
                H_hat = self.estimator(Yp, Xp)

        X_hat = self.equalizer(
            Ys=Ys,
            H_hat=H_hat,
            noise_var=noise_var,
            mode=self.cfg.equalizer,
        )

        s_hat_grid = undo_rank_based_stream_mapping(
            X_hat=X_hat,
            imp_order=map_aux["imp_order"],
            ch_order=map_aux["ch_order"],
        )

        s_hat = s_hat_grid.reshape(B, total_slots)[:, :Nsym_orig]
        z_hat = complex_symbols_to_real(s_hat, latent_meta)

        aux = {
            "Hf_true": Hf_true,
            "H_hat": H_hat,
            "importance": importance,
            "q": map_aux["q"],
            "imp_order": map_aux["imp_order"],
            "ch_order": map_aux["ch_order"],
            "noise_var": noise_var,
            "z_hat": z_hat,
        }
        return z_hat, aux