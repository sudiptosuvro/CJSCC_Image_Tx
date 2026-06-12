import os, math, sys
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_lowres import make_cifar10_loaders
from model.system_model import MASCConfig, CJSCC
from utils.metrics import mse_torch, psnr_torch, ssim_torch

from OFDM.core.ofdm import OFDMConfig
from OFDM.core.OFDM_system import CJSCC_OFDM

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

os.makedirs("./runs", exist_ok=True)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def sample_snr(B: int, snr_min: float, snr_max: float, device):
    return torch.empty(B, device=device).uniform_(snr_min, snr_max)

def sample_rho() -> float:
    rho_choices = [1/24, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 1.0]
    idx = torch.randint(0, len(rho_choices), (1,)).item()
    return float(rho_choices[idx])

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


def build_ofdm_system(
    semantic_ckpt_path: str,
    clip_ratio: float | None = None,
):
    ckpt = torch.load(semantic_ckpt_path, map_location=device)
    cfg = MASCConfig(**ckpt["cfg"])
    assert cfg.latent_channels == 192, f"Expected C=192 checkpoint, got C={cfg.latent_channels}"

    base = CJSCC(cfg).to(device)
    base.load_state_dict(ckpt["state_dict"], strict=True)
    base.eval()

    ofdm_cfg = OFDMConfig(
        nfft=64,
        cp_len=16,
        n_pilot=1,
        n_taps=8,
        delay_decay=4.0,
        equalizer="mmse",
    )

    model_ofdm = CJSCC_OFDM(cfg, ofdm_cfg, clip_ratio=clip_ratio).to(device)

    # initialize encoder/decoder from pretrained semantic backbone
    model_ofdm.enc.load_state_dict(base.enc.state_dict(), strict=True)
    model_ofdm.dec.load_state_dict(base.dec.state_dict(), strict=True)

    return model_ofdm, cfg, ofdm_cfg


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    cfg,
    device,
    snr_min: float = 0.0,
    snr_max: float = 25.0,
    lambda_z: float = 0.25,
    lambda_h: float = 0.5,
):
    model.train()

    total_loss = 0.0
    total_x = 0.0
    total_z = 0.0
    total_h = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(train_loader, desc="Train", leave=False):
        x = x.to(device)
        B = x.size(0)

        rho = sample_rho()
        snr_vals = sample_snr(B, snr_min, snr_max, device)
        snr_db = snr_vals[:1]
        csi_snr = snr_db.expand(B)
        csi = build_csi(csi_snr, rho, cfg, channel_type="rayleigh")
        x_hat, aux = model(
            x,
            snr_db=snr_db,
            rho=rho,
            csi=csi,
            use_perfect_csi=False,
            use_perfect_alloc_csi=True,
        )

        x_loss = torch.mean((x - x_hat) ** 2)
        z_loss = torch.mean((aux["z_hat"] - aux["z_hat"].detach()) ** 2) * 0.0
        h_loss = torch.mean(torch.abs(aux["H_hat"] - aux["Hf_true"]) ** 2)

        loss = x_loss + lambda_h * h_loss + lambda_z * z_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            x_eval = torch.clamp(x_hat, 0.0, 1.0)
            total_psnr += psnr_torch(x_eval, x).mean().item()
            total_ssim += ssim_torch(x_eval, x).mean().item()

        total_loss += loss.item()
        total_x += x_loss.item()
        total_z += z_loss.item()
        total_h += h_loss.item()
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "x_loss": total_x / max(n, 1),
        "z_loss": total_z / max(n, 1),
        "h_loss": total_h / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def eval_one_epoch(
    model,
    val_loader,
    cfg,
    device,
    rho: float = 1.0,
    snr_val: float = 5.0,
):
    model.eval()

    total_x = 0.0
    total_h = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    for x, _ in tqdm(val_loader, desc="Eval", leave=False):
        x = x.to(device)
        B = x.size(0)
        snr_vals = torch.full((B,), snr_val, device=device)
        csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

        x_hat, aux = model(
            x,
            snr_db=snr_db,
            rho=rho,
            csi=csi,
            use_perfect_csi=False,
            use_perfect_alloc_csi=True,
        )

        x_eval = torch.clamp(x_hat, 0.0, 1.0)

        x_loss = torch.mean((x - x_eval) ** 2)
        h_loss = torch.mean(torch.abs(aux["H_hat"] - aux["Hf_true"]) ** 2)

        total_x += x_loss.item()
        total_h += h_loss.item()
        total_psnr += psnr_torch(x_eval, x).mean().item()
        total_ssim += ssim_torch(x_eval, x).mean().item()
        n += 1

    return {
        "x_loss": total_x / max(n, 1),
        "h_loss": total_h / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


def main():
    model, cfg, ofdm_cfg = build_ofdm_system(
        semantic_ckpt_path="../checks/cifar10_C192/rayleigh/cifar10_C192_rayleigh_full_seed0_best.pt",
        clip_ratio=None,   # later set e.g. 1.4 for clipping-aware training
    )

    train_loader, val_loader, test_loader = make_cifar10_loaders(
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    ###################################
    # Stage B: decoder adaptation only
    ###################################
    set_requires_grad(model.enc, False)
    set_requires_grad(model.ofdm_middle, False)
    set_requires_grad(model.dec, True)

    optimizer = torch.optim.Adam(model.dec.parameters(), lr=1e-4)

    best_val = float("inf")
    no_improve = 0
    patience = 15

    save_path_stageB = "./runs/cjscc_ofdm_stageB_final.pt"

    for epoch in range(200):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            snr_min=0.0,
            snr_max=25.0,
            lambda_z=0.0,
            lambda_h=0.0,
        )

        val_stats = eval_one_epoch(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            rho=1/24,
            snr_val=5.0,
        )

        print(
            f"[Stage B] Epoch {epoch+1:02d} | "
            f"train x={train_stats['x_loss']:.6f} | "
            f"train PSNR={train_stats['psnr']:.3f} | "
            f"val x={val_stats['x_loss']:.6f} | "
            f"val PSNR={val_stats['psnr']:.3f} | "
            f"val SSIM={val_stats['ssim']:.4f}"
        )

        if val_stats["x_loss"] < best_val:
            best_val = val_stats["x_loss"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "val_x_loss": best_val,
                    "cfg": cfg.__dict__,
                    "ofdm_cfg": ofdm_cfg.__dict__,
                },
                save_path_stageB,
            )
            print(f"Saved best Stage-B checkpoint to {save_path_stageB}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping Stage B.")
                break

    ###########################################
    # Stage C: decoder + robust CE fine-tuning
    ###########################################
    ckpt = torch.load(save_path_stageB, map_location=device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    set_requires_grad(model.enc, False)
    set_requires_grad(model.dec, True)
    set_requires_grad(model.ofdm_middle, False)
    set_requires_grad(model.ofdm_middle.estimator, True)

    optimizer = torch.optim.Adam([
        {"params": model.dec.parameters(), "lr": 5e-5},
        {"params": model.ofdm_middle.estimator.parameters(), "lr": 1e-4},
    ])

    best_val = float("inf")
    no_improve = 0
    patience = 10

    save_path_stageC = "./runs/cjscc_ofdm_stageC_final.pt"

    for epoch in range(250):
        train_stats = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            snr_min=0.0,
            snr_max=25.0,
            lambda_z=0.0,
            lambda_h=0.5,
        )

        val_stats = eval_one_epoch(
            model=model,
            val_loader=val_loader,
            cfg=cfg,
            device=device,
            rho=1/24,
            snr_val=5.0,
        )

        print(
            f"[Stage C] Epoch {epoch+1:02d} | "
            f"train x={train_stats['x_loss']:.6f} | "
            f"train h={train_stats['h_loss']:.6f} | "
            f"train PSNR={train_stats['psnr']:.3f} | "
            f"val x={val_stats['x_loss']:.6f} | "
            f"val h={val_stats['h_loss']:.6f} | "
            f"val PSNR={val_stats['psnr']:.3f} | "
            f"val SSIM={val_stats['ssim']:.4f}"
        )

        if val_stats["x_loss"] < best_val:
            best_val = val_stats["x_loss"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "val_x_loss": best_val,
                    "cfg": cfg.__dict__,
                    "ofdm_cfg": ofdm_cfg.__dict__,
                },
                save_path_stageC,
            )
            print(f"Saved best Stage-C checkpoint to {save_path_stageC}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping Stage C.")
                break


if __name__ == "__main__":
    main()