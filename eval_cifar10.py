import os
import json
import argparse
from fractions import Fraction

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.system_model import MASCConfig, CJSCC, make_channel
from data.data_lowres import make_cifar10_loaders
from utils.metrics import psnr_torch, ssim_torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")


# CSI builder
def build_csi(snr_db: torch.Tensor, rho: float,
              csi_dim: int, channel_type: str) -> torch.Tensor:
    B     = snr_db.size(0)
    dev   = snr_db.device
    dtype = snr_db.dtype

    rho_t    = torch.full((B,), float(rho), device=dev, dtype=dtype)
    snr_norm = (snr_db - 12.5) / 12.5
    rho_norm = torch.log(rho_t / (1.0 / 48.0) + 1e-8)

    csi = torch.zeros(B, csi_dim, device=dev, dtype=dtype)
    csi[:, 0] = snr_norm
    csi[:, 1] = rho_norm
    csi[:, 2] = rho_t

    ct = channel_type.lower()
    if   ct == "awgn":     csi[:, 3] = 1.0
    elif ct == "rayleigh": csi[:, 4] = 1.0
    elif ct == "rician":   csi[:, 5] = 1.0
    else:
        raise ValueError(f"Unknown channel_type={channel_type!r}")
    return csi


# Load model
def load_model(ckpt_path: str) -> tuple:
    assert os.path.isfile(ckpt_path), f"Not found: {ckpt_path}"
    ckpt     = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["ablation_mode"] = "full"
    cfg_dict.pop("add_compensating_conv", None)
    cfg   = MASCConfig(**cfg_dict)
    model = CJSCC(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    epoch  = ckpt.get("epoch", "?")
    best_p = ckpt.get("best_psnr", float("nan"))
    trained_ch = cfg_dict.get("channel_type", "unknown")
    print(f"  Loaded {os.path.basename(ckpt_path)}  "
          f"epoch={epoch}  best_psnr={best_p:.3f}  "
          f"trained_on={trained_ch}  C={cfg.latent_channels}")
    return model, cfg


@torch.no_grad()
def eval_point(model, cfg, loader,
               snr_db: float, rho: float,
               eval_channel: str) -> dict:
    model.eval()
    model.cfg.channel_type = eval_channel
    model.channel = make_channel(
        eval_channel,
        rician_K=cfg.rician_K,
        equalize=cfg.equalize,
        per_symbol=cfg.per_symbol_fading,
    ).to(device)

    psnr_vals, ssim_vals = [], []
    for x, _ in loader:
        x   = x.to(device).float()
        B   = x.size(0)
        snr = torch.full((B,), float(snr_db), device=device, dtype=x.dtype)
        csi = build_csi(snr, rho, cfg.csi_dim, eval_channel)

        x_hat, _ = model(x, snr_db=snr, rho=float(rho), csi=csi)
        x_hat    = torch.clamp(x_hat, 0.0, 1.0)
        psnr_vals.append(psnr_torch(x_hat, x).cpu())
        ssim_vals.append(ssim_torch(x_hat, x).cpu())

    return {
        "psnr": round(torch.cat(psnr_vals).mean().item(), 4),
        "ssim": round(torch.cat(ssim_vals).mean().item(), 5),
    }


def run_sweeps(model, cfg, loader,
               snr_list, rho_list,
               rho_fixed_list, snr_fixed,
               eval_channel) -> dict:
    result = {"snr_sweeps": {}, "rho_sweep": {}}

    # SNR sweep at each fixed rho
    for rho_fixed in rho_fixed_list:
        frac = Fraction(rho_fixed).limit_denominator()
        key  = f"rho_{frac}"
        print(f"    SNR sweep @ rho={frac}  eval_ch={eval_channel}")
        psnr_s, ssim_s = [], []
        for snr in snr_list:
            s = eval_point(model, cfg, loader, snr, rho_fixed, eval_channel)
            psnr_s.append(s["psnr"])
            ssim_s.append(s["ssim"])
        result["snr_sweeps"][key] = {
            "rho_value": float(rho_fixed),
            "snr":  snr_list,
            "psnr": psnr_s,
            "ssim": ssim_s,
        }

    # rho sweep at fixed SNR
    key = f"snr_{snr_fixed}"
    print(f"    rho sweep @ SNR={snr_fixed} dB  eval_ch={eval_channel}")
    psnr_r, ssim_r = [], []
    for rho in rho_list:
        s = eval_point(model, cfg, loader, snr_fixed, rho, eval_channel)
        psnr_r.append(s["psnr"])
        ssim_r.append(s["ssim"])
    result["rho_sweep"][key] = {
        "snr_value": float(snr_fixed),
        "rho":  [round(r, 6) for r in rho_list],
        "psnr": psnr_r,
        "ssim": ssim_r,
    }

    return result


# Plot
TRAIN_COLORS = {
    "awgn":     "#1f77b4",   # blue
    "rayleigh": "#d62728",   # red
    "rician":   "#2ca02c",   # green
}

EVAL_STYLES = {
    "awgn":     dict(linestyle="-",  marker="o"),   # solid circle
    "rayleigh": dict(linestyle="--", marker="s"),   # dashed square
    "rician":   dict(linestyle="-.", marker="^"),   # dash-dot triangle
}

TRAIN_LABELS = {
    "awgn":     "AWGN-trained",
    "rayleigh": "Rayleigh-trained",
    "rician":   "Rician-trained",
}

EVAL_LABELS = {
    "awgn":     "AWGN",
    "rayleigh": "Rayleigh",
    "rician":   "Rician",
}


def _style_ax(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title,   fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=9)


def _make_legend(fig, train_channels, eval_channels):
    handles = []
    labels  = []

    for tr_ch in train_channels:
        for ev_ch in eval_channels:
            sty = {**EVAL_STYLES[ev_ch],
                   "color": TRAIN_COLORS[tr_ch],
                   "linewidth": 1.8, "markersize": 5}
            line, = plt.plot([], [], **sty)
            handles.append(line)
            labels.append(f"{TRAIN_LABELS[tr_ch]} → {EVAL_LABELS[ev_ch]}")

    fig.legend(handles, labels,
               loc="upper center", ncol=3,
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.0))


def plot_combined(all_results: dict,
                  snr_list: list, rho_list: list,
                  rho_fixed_list: list, snr_fixed: float,
                  out_dir: str) -> str:
    
    train_channels = list(all_results.keys())
    eval_channels  = ["awgn", "rayleigh", "rician"]
    rho_keys       = [f"rho_{Fraction(r).limit_denominator()}"
                      for r in rho_fixed_list]
    rho_key_snr    = f"snr_{snr_fixed}"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.subplots_adjust(top=0.88)

    for tr_ch in train_channels:
        color = TRAIN_COLORS[tr_ch]
        for ev_ch in eval_channels:
            sty = {**EVAL_STYLES[ev_ch],
                   "color": color, "linewidth": 1.8, "markersize": 5}
            data = all_results[tr_ch][ev_ch]

            # Col 0: SNR sweep @ rho_fixed_list[0]
            if rho_keys[0] in data["snr_sweeps"]:
                d = data["snr_sweeps"][rho_keys[0]]
                axes[0, 0].plot(d["snr"], d["psnr"], **sty)
                axes[1, 0].plot(d["snr"], d["ssim"], **sty)

            # Col 1: SNR sweep @ rho_fixed_list[1]
            if rho_keys[1] in data["snr_sweeps"]:
                d = data["snr_sweeps"][rho_keys[1]]
                axes[0, 1].plot(d["snr"], d["psnr"], **sty)
                axes[1, 1].plot(d["snr"], d["ssim"], **sty)

            # Col 2: rho sweep @ snr_fixed
            if rho_key_snr in data["rho_sweep"]:
                d = data["rho_sweep"][rho_key_snr]
                axes[0, 2].plot(d["rho"], d["psnr"], **sty)
                axes[1, 2].plot(d["rho"], d["ssim"], **sty)

    # Style axes
    frac0 = Fraction(rho_fixed_list[0]).limit_denominator()
    frac1 = Fraction(rho_fixed_list[1]).limit_denominator()

    _style_ax(axes[0, 0], "SNR (dB)", "PSNR (dB)",
              f"PSNR vs SNR @ CR={frac0}")
    _style_ax(axes[0, 1], "SNR (dB)", "PSNR (dB)",
              f"PSNR vs SNR @ CR={frac1}")
    _style_ax(axes[0, 2], "Compression ratio ρ", "PSNR (dB)",
              f"PSNR vs Compression Ratio @ SNR={snr_fixed} dB")
    _style_ax(axes[1, 0], "SNR (dB)", "SSIM",
              f"SSIM vs SNR @ CR={frac0}")
    _style_ax(axes[1, 1], "SNR (dB)", "SSIM",
              f"SSIM vs SNR @ CR={frac1}")
    _style_ax(axes[1, 2], "Compression ratio ρ", "SSIM",
              f"SSIM vs Compression Ratio @ SNR={snr_fixed} dB")

    # Rho axis as fractions
    for row in range(2):
        axes[row, 2].xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda v, _: str(Fraction(v).limit_denominator(12))
            )
        )

    _make_legend(fig, train_channels, eval_channels)

    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "cross_channel_combined.pdf")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Combined PDF → {pdf_path}")
    return pdf_path


def main():
    ap = argparse.ArgumentParser(
        description="CIFAR-10 cross-channel generalization"
    )
    ap.add_argument("--ckpt_dir",  type=str,
                    default="./checks/",
                    help="Directory containing awgn/ rayleigh/ rician/ subfolders")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--out_dir",   type=str,
                    default="./outs/cifar10")
    ap.add_argument("--C",         type=int, default=192,
                    help="Latent channels — used to build checkpoint filename")
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--batch_size",type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Sweep config
    snr_list       = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 23, 25]
    rho_list       = [1/48, 1/24, 1/12, 1/10, 1/8, 1/6,
                      0.2, 0.25, 1/3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    rho_fixed_list = [1/6, 1/12]
    snr_fixed      = 5.0

    train_channels = ["awgn", "rayleigh", "rician"]
    eval_channels  = ["awgn", "rayleigh", "rician"]

    # Data
    print("\nLoading CIFAR-10 test set ...")
    _, _, test_loader = make_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=(str(device) == "cuda"),
    )
    print(f"  Test samples: {len(test_loader.dataset)}")

    # Load checkpoints
    print("\nLoading checkpoints ...")
    models = {}
    for tr_ch in train_channels:
        ckpt_path = os.path.join(
            args.ckpt_dir, tr_ch,
            f"cifar10_C{args.C}_{tr_ch}_full_seed{args.seed}_best.pt"
        )
        if not os.path.isfile(ckpt_path):
            print(f"  [WARN] Not found: {ckpt_path} — skipping")
            continue
        model, cfg = load_model(ckpt_path)
        models[tr_ch] = (model, cfg)

    if not models:
        raise RuntimeError("No checkpoints found. Check --ckpt_dir and --C.")

    all_results = {}

    for tr_ch, (model, cfg) in models.items():
        print(f"\n{'='*60}")
        print(f"  Trained on: {tr_ch.upper()}")
        print(f"{'='*60}")
        all_results[tr_ch] = {}

        for ev_ch in eval_channels:
            print(f"\n  Evaluating on: {ev_ch.upper()}")
            sweeps = run_sweeps(
                model, cfg, test_loader,
                snr_list=snr_list,
                rho_list=rho_list,
                rho_fixed_list=rho_fixed_list,
                snr_fixed=snr_fixed,
                eval_channel=ev_ch,
            )
            all_results[tr_ch][ev_ch] = sweeps

        json_path = os.path.join(args.out_dir, f"{tr_ch}.json")
        with open(json_path, "w") as f:
            json.dump({tr_ch: all_results[tr_ch]}, f, indent=2)
        print(f"\n  Per-channel JSON → {json_path}")

    combined_json = os.path.join(args.out_dir, "combined.json")
    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Combined JSON → {combined_json}")

    # Plot
    plot_combined(
        all_results=all_results,
        snr_list=snr_list,
        rho_list=rho_list,
        rho_fixed_list=rho_fixed_list,
        snr_fixed=snr_fixed,
        out_dir=args.out_dir,
    )

    print(f"\nDone. All outputs → {args.out_dir}")


if __name__ == "__main__":
    main()