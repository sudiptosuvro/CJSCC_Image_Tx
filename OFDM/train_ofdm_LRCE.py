import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_lowres import make_cifar10_loaders
from model.system_model import MASCConfig, CJSCC
from utils.metrics import psnr_torch, ssim_torch
from OFDM.core.ofdm import OFDMConfig, OFDMJSCCMiddleLearnableCE

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)
os.makedirs("./runs", exist_ok=True)


def sample_snr(B: int, snr_min: float, snr_max: float, device):
    return torch.empty(B, device=device).uniform_(snr_min, snr_max)


def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.view_as_real(a) - torch.view_as_real(b)) ** 2)

def build_csi(snr_vals, rho, cfg, channel_type="rayleigh", rho_min=1/24):
    B = snr_vals.size(0)
    device = snr_vals.device

    csi = torch.zeros(B, cfg.csi_dim, device=device, dtype=torch.float32)

    csi[:, 0] = (snr_vals - 12.5) / 12.5
    rho_tensor = torch.full((B,), float(rho), device=device)
    csi[:, 1] = torch.log(rho_tensor / rho_min + 1e-8)
    csi[:, 2] = rho_tensor

    if channel_type == "awgn":
        csi[:, 3] = 1.0
    elif channel_type == "rayleigh":
        csi[:, 4] = 1.0
    elif channel_type == "rician":
        csi[:, 5] = 1.0

    return csi


def build_highres_baseline():
    ckpt = torch.load("../runs/cifar10_C192/rayleigh/cifar10_C192_rayleigh_full_seed0_best.pt", map_location=device)
    cfg = MASCConfig(**ckpt["cfg"])
    assert cfg.latent_channels == 192, f"Expected C=192 checkpoint, got C={cfg.latent_channels}"

    model = CJSCC(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    encoder = model.enc.to(device)
    decoder = model.dec.to(device)
    encoder.eval()
    decoder.eval()

    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    ofdm_cfg = OFDMConfig(
        nfft=64,
        cp_len=16,
        n_pilot=1,
        n_taps=8,
        delay_decay=4.0,
        equalizer="mmse",
    )

    ofdm_learnable = OFDMJSCCMiddleLearnableCE(ofdm_cfg).to(device)
    return cfg, ofdm_cfg, encoder, decoder, ofdm_learnable


def train_one_epoch(
    encoder,
    decoder,
    ofdm_learnable,
    train_loader,
    optimizer,
    cfg,
    snr_min=0.0,
    snr_max=25.0,
    lambda_z=0.25,
):
    encoder.eval()
    decoder.eval()
    ofdm_learnable.train()

    total_h = 0.0
    total_z = 0.0
    total_x = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(train_loader, desc="Train", leave=False):
        x = x.to(device)
        B = x.size(0)

        snr_vals = sample_snr(B, snr_min, snr_max, device)
        snr_db = snr_vals[:1]
        csi_snr = snr_db.expand(B)
        
        rho = 1.0
        csi = build_csi(csi_snr, rho, cfg, channel_type="rayleigh")

        with torch.no_grad():
            z = encoder(x, csi)

        z_hat, aux = ofdm_learnable(z, snr_db=snr_db, use_perfect_csi=False)
        x_hat = decoder(z_hat, csi)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        h_loss = complex_mse(aux["H_hat"], aux["Hf_true"])
        z_loss = torch.mean((z - z_hat) ** 2)
        loss = h_loss + lambda_z * z_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_h += h_loss.item()
        total_z += z_loss.item()
        total_x += torch.mean((x - x_hat) ** 2).item()
        total_psnr += psnr_torch(x_hat, x).mean().item()
        total_ssim += ssim_torch(x_hat, x).mean().item()
        n += 1

    return {
        "h_loss": total_h / max(n, 1),
        "z_loss": total_z / max(n, 1),
        "x_loss": total_x / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def eval_one_epoch(
    encoder,
    decoder,
    ofdm_learnable,
    val_loader,
    cfg,
    snr_val=5.0,
    lambda_z=0.25,
):
    encoder.eval()
    decoder.eval()
    ofdm_learnable.eval()

    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    total_h = 0.0
    total_z = 0.0
    total_x = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(val_loader, desc="Eval", leave=False):
        x = x.to(device)
        B = x.size(0)
        rho = 1.0
        snr_vals = torch.full((B,), snr_val, device=device)
        csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

        z = encoder(x, csi)
        z_hat, aux = ofdm_learnable(z, snr_db=snr_db, use_perfect_csi=False)
        x_hat = decoder(z_hat, csi)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        h_loss = complex_mse(aux["H_hat"], aux["Hf_true"])
        z_loss = torch.mean((z - z_hat) ** 2)

        total_h += h_loss.item()
        total_z += z_loss.item()
        total_x += torch.mean((x - x_hat) ** 2).item()
        total_psnr += psnr_torch(x_hat, x).mean().item()
        total_ssim += ssim_torch(x_hat, x).mean().item()
        n += 1

    return {
        "h_loss": total_h / max(n, 1),
        "z_loss": total_z / max(n, 1),
        "x_loss": total_x / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


def main():
    cfg, ofdm_cfg, encoder, decoder, ofdm_learnable = build_highres_baseline()

    train_loader, val_loader, test_loader = make_cifar10_loaders(
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )

    optimizer = torch.optim.Adam(ofdm_learnable.learnable_ce.parameters(), lr=1e-4)

    best_val = float("inf")
    no_improve = 0
    patience = 10
    min_delta = 1e-4
    min_epochs = 15

    save_path = "./runs/ofdm_LE.pt"

    for epoch in range(150):
        train_stats = train_one_epoch(
            encoder, decoder, ofdm_learnable, train_loader, optimizer, cfg,
            snr_min=0.0, snr_max=25.0, lambda_z=0.25
        )

        val_stats = eval_one_epoch(
            encoder, decoder, ofdm_learnable, val_loader, cfg,
            snr_val=5.0, lambda_z=0.25
        )

        print(
            f"[Learnable CE HighRes] Epoch {epoch+1:02d} | "
            f"train h={train_stats['h_loss']:.6f} | "
            f"train z={train_stats['z_loss']:.6f} | "
            f"train PSNR={train_stats['psnr']:.3f} | "
            f"val h={val_stats['h_loss']:.6f} | "
            f"val z={val_stats['z_loss']:.6f} | "
            f"val PSNR={val_stats['psnr']:.3f} | "
            f"val SSIM={val_stats['ssim']:.4f}"
        )

        score = val_stats["h_loss"] + 0.25 * val_stats["z_loss"]

        if score < best_val - min_delta:
            best_val = score
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": ofdm_learnable.state_dict(),
                    "val_score": best_val,
                    "cfg": cfg.__dict__,
                    "ofdm_cfg": ofdm_cfg.__dict__,
                },
                save_path,
            )
            print(f"Saved best high-res learnable CE checkpoint to {save_path}")
        else:
            no_improve += 1
            if epoch + 1 >= min_epochs and no_improve >= patience:
                print("Early stopping high-res learnable CE.")
                break


if __name__ == "__main__":
    main()