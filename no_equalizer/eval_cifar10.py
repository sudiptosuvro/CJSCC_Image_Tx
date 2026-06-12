import os
import sys
import json
import argparse
from pathlib import Path
from fractions import Fraction

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

for p in [str(ROOT_DIR), str(THIS_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model import CJSCC, MASCConfig, make_channel
from data.data_lowres import make_cifar10_loaders
from utils.metrics import psnr_torch, ssim_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_csi(snr_db: torch.Tensor, rho: float, csi_dim: int, channel_type: str):
    B = snr_db.size(0)
    dev = snr_db.device
    dtype = snr_db.dtype

    rho_t = torch.full((B,), float(rho), device=dev, dtype=dtype)
    snr_norm = (snr_db - 12.5) / 12.5
    rho_norm = torch.log(rho_t / (1.0 / 48.0) + 1e-8)

    csi = torch.zeros(B, csi_dim, device=dev, dtype=dtype)
    csi[:, 0] = snr_norm
    csi[:, 1] = rho_norm
    csi[:, 2] = rho_t

    ct = channel_type.lower()
    if ct == "awgn":
        csi[:, 3] = 1.0
    elif ct == "rayleigh":
        csi[:, 4] = 1.0
    elif ct == "rician":
        csi[:, 5] = 1.0
    else:
        raise ValueError(f"Unknown channel_type={channel_type}")

    return csi


def load_model(ckpt_path: str):
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(ckpt["cfg"])

    cfg_dict.pop("equalize", None)
    cfg_dict.pop("add_compensating_conv", None)

    cfg = MASCConfig(**cfg_dict)
    model = CJSCC(cfg).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)

    print(f"Loaded: {ckpt_path}")
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    print(f"Best PSNR: {ckpt.get('best_psnr', float('nan')):.3f}")
    print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    print(f"csi_dim={cfg.csi_dim}, latent_channels={cfg.latent_channels}")

    model.eval()
    return model, cfg


@torch.no_grad()
def eval_point(model, loader, snr_db: float, rho: float, csi_dim: int, channel_type: str):
    model.eval()

    model.cfg.channel_type = channel_type
    model.channel = make_channel(
        channel_type,
        rician_K=model.cfg.rician_K,
        per_symbol=model.cfg.per_symbol_fading,
    ).to(device)

    psnr_vals, ssim_vals, mse_vals = [], [], []

    for x, _ in loader:
        x = x.to(device).float()
        B = x.size(0)

        snr = torch.full((B,), float(snr_db), device=device, dtype=x.dtype)
        csi = build_csi(snr, rho, csi_dim, channel_type)

        x_hat, _ = model(x, snr_db=snr, rho=float(rho), csi=csi)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        psnr_vals.append(psnr_torch(x_hat, x).cpu())
        ssim_vals.append(ssim_torch(x_hat, x).cpu())
        mse_vals.append(((x_hat - x) ** 2).flatten(1).mean(1).cpu())

    return {
        "psnr": torch.cat(psnr_vals).mean().item(),
        "ssim": torch.cat(ssim_vals).mean().item(),
        "mse": torch.cat(mse_vals).mean().item(),
    }


def sweep_snr(model, loader, snr_list, rho, csi_dim, channels):
    out = {}
    for ch in channels:
        psnr, ssim, mse = [], [], []
        print(f"\nSNR sweep | channel={ch} | rho={rho}")
        for snr in snr_list:
            r = eval_point(model, loader, snr, rho, csi_dim, ch)
            psnr.append(r["psnr"])
            ssim.append(r["ssim"])
            mse.append(r["mse"])
            print(f"  SNR={snr:5.1f} | PSNR={r['psnr']:.3f} | SSIM={r['ssim']:.4f} | MSE={r['mse']:.6f}")
        out[ch] = {"snr": snr_list, "psnr": psnr, "ssim": ssim, "mse": mse}
    return out


def sweep_rho(model, loader, rho_list, snr_fixed, csi_dim, channels):
    out = {}
    for ch in channels:
        psnr, ssim, mse = [], [], []
        print(f"\nrho sweep | channel={ch} | SNR={snr_fixed}")
        for rho in rho_list:
            r = eval_point(model, loader, snr_fixed, rho, csi_dim, ch)
            psnr.append(r["psnr"])
            ssim.append(r["ssim"])
            mse.append(r["mse"])
            print(f"  rho={rho:.4f} | PSNR={r['psnr']:.3f} | SSIM={r['ssim']:.4f} | MSE={r['mse']:.6f}")
        out[ch] = {"rho": rho_list, "psnr": psnr, "ssim": ssim, "mse": mse}
    return out


def plot_2x3(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    channels = ["awgn", "rayleigh", "rician"]
    labels = {"awgn": "AWGN", "rayleigh": "Rayleigh", "rician": "Rician"}
    markers = {"awgn": "o", "rayleigh": "s", "rician": "^"}

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # Column 1: SNR, rho=1/6
    for ch in channels:
        d = results["snr_rho_1_6"][ch]
        axes[0, 0].plot(d["snr"], d["psnr"], marker=markers[ch], linewidth=2, label=labels[ch])
        axes[1, 0].plot(d["snr"], d["ssim"], marker=markers[ch], linewidth=2, label=labels[ch])

    # Column 2: SNR, rho=1/12
    for ch in channels:
        d = results["snr_rho_1_12"][ch]
        axes[0, 1].plot(d["snr"], d["psnr"], marker=markers[ch], linewidth=2, label=labels[ch])
        axes[1, 1].plot(d["snr"], d["ssim"], marker=markers[ch], linewidth=2, label=labels[ch])

    # Column 3: rho, SNR=5
    for ch in channels:
        d = results["rho_snr_5"][ch]
        axes[0, 2].plot(d["rho"], d["psnr"], marker=markers[ch], linewidth=2, label=labels[ch])
        axes[1, 2].plot(d["rho"], d["ssim"], marker=markers[ch], linewidth=2, label=labels[ch])

    titles = [
        "PSNR vs SNR at ρ = 1/6",
        "PSNR vs SNR at ρ = 1/12",
        "PSNR vs ρ at SNR = 5 dB",
        "SSIM vs SNR at ρ = 1/6",
        "SSIM vs SNR at ρ = 1/12",
        "SSIM vs ρ at SNR = 5 dB",
    ]

    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
        ax.tick_params(labelsize=9)

    axes[0, 0].set_ylabel("PSNR (dB)")
    axes[0, 1].set_ylabel("PSNR (dB)")
    axes[0, 2].set_ylabel("PSNR (dB)")
    axes[1, 0].set_ylabel("SSIM")
    axes[1, 1].set_ylabel("SSIM")
    axes[1, 2].set_ylabel("SSIM")

    axes[0, 0].set_xlabel("SNR (dB)")
    axes[1, 0].set_xlabel("SNR (dB)")
    axes[0, 1].set_xlabel("SNR (dB)")
    axes[1, 1].set_xlabel("SNR (dB)")
    axes[0, 2].set_xlabel("Bandwidth ratio ρ")
    axes[1, 2].set_xlabel("Bandwidth ratio ρ")

    for ax in [axes[0, 2], axes[1, 2]]:
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: str(Fraction(v).limit_denominator(48)))
        )

    fig.suptitle("CIFAR-10 Proposed CJSCC Evaluation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    pdf_path = os.path.join(out_dir, "proposed_eval_2x3.pdf")
    png_path = os.path.join(out_dir, "proposed_eval_2x3.png")

    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, default="../data")
    ap.add_argument("--out_dir", type=str, default="./runs/final_eval")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    print(f"Device: {device}")

    _, _, test_loader = make_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model, cfg = load_model(args.ckpt)

    snr_list = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25]
    rho_list = [1/48, 1/24, 1/12, 1/10, 1/8, 1/6, 0.2, 0.25, 1/3, 0.5, 2/3, 0.8, 1.0]
    channels = ["awgn", "rayleigh", "rician"]

    results = {
        "snr_rho_1_6": sweep_snr(model, test_loader, snr_list, 1/6, cfg.csi_dim, channels),
        "snr_rho_1_12": sweep_snr(model, test_loader, snr_list, 1/12, cfg.csi_dim, channels),
        "rho_snr_5": sweep_rho(model, test_loader, rho_list, 5.0, cfg.csi_dim, channels),
    }

    os.makedirs(args.out_dir, exist_ok=True)

    json_path = os.path.join(args.out_dir, "proposed_eval_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_2x3(results, args.out_dir)

    print(f"Saved JSON: {json_path}")
    print("Evaluation complete.")


if __name__ == "__main__":
    main()