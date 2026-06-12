import os
import sys
import json
import argparse
from pathlib import Path
from fractions import Fraction

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from model.system_model import CJSCC, MASCConfig, make_channel
from data.data_highres import make_div2k_kodak_loaders
from utils.metrics import psnr_torch, ssim_torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def build_csi(snr_db, rho, csi_dim, channel_type):
    B = snr_db.size(0)
    dev = snr_db.device
    dtype = snr_db.dtype

    rho_t = torch.full((B,), float(rho), device=dev, dtype=dtype)
    snr_norm = (snr_db - 12.5) / 12.5
    rho_norm = torch.log(rho_t / (1.0 / 24.0) + 1e-8)

    csi = torch.zeros(B, csi_dim, device=dev, dtype=dtype)
    csi[:, 0] = snr_norm
    csi[:, 1] = rho_norm
    csi[:, 2] = rho_t

    channel_type = channel_type.lower()
    if channel_type == "awgn":
        csi[:, 3] = 1.0
    elif channel_type == "rayleigh":
        csi[:, 4] = 1.0
    elif channel_type == "rician":
        csi[:, 5] = 1.0
    else:
        raise ValueError(f"Unknown channel_type={channel_type}")

    return csi


def load_full_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = dict(ckpt["cfg"])
    cfg_dict["ablation_mode"] = "full"
    cfg_dict.pop("add_compensating_conv", None)

    cfg = MASCConfig(**cfg_dict)
    model = CJSCC(cfg).to(device)

    missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)

    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")

    model.eval()

    print(f"Loaded: {ckpt_path}")
    print(f"Epoch: {ckpt.get('epoch', '?')}")
    print(f"Best PSNR: {ckpt.get('best_psnr', float('nan')):.3f}")
    print(f"Latent C: {cfg.latent_channels}")
    print(f"CSI dim: {cfg.csi_dim}")

    return model, cfg


@torch.no_grad()
def eval_point(model, loader, snr_db, rho, csi_dim, channel_type):
    model.eval()

    model.cfg.channel_type = channel_type
    model.channel = make_channel(
        channel_type,
        rician_K=model.cfg.rician_K,
        equalize=model.cfg.equalize,
        per_symbol=model.cfg.per_symbol_fading,
    ).to(device)

    psnr_vals = []
    ssim_vals = []
    mse_vals = []

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


def run_eval(model, loader, cfg, snr_list, rho_list, fixed_rho_list, fixed_snr_list, eval_channels):

    all_results = {}

    for ch in eval_channels:
        print(f"\n{'=' * 70}")
        print(f"Evaluating full model on {ch.upper()} channel")
        print(f"{'=' * 70}")

        ch_results = {
            "snr_sweep": {},
            "rho_sweep": {},
        }

        # SNR sweep: PSNR/SSIM vs SNR at fixed rho
        for rho_fixed in fixed_rho_list:
            rho_frac = Fraction(rho_fixed).limit_denominator()
            key = f"rho_{rho_frac}"

            print(f"\nSNR sweep @ rho = {rho_frac}")

            psnr_list = []
            ssim_list = []
            mse_list = []

            for snr in snr_list:
                result = eval_point(
                    model=model,
                    loader=loader,
                    snr_db=snr,
                    rho=rho_fixed,
                    csi_dim=cfg.csi_dim,
                    channel_type=ch,
                )

                psnr_list.append(round(result["psnr"], 4))
                ssim_list.append(round(result["ssim"], 5))
                mse_list.append(round(result["mse"], 8))

                print(
                    f"SNR={snr:>4} dB | rho={str(rho_frac):<5} | "
                    f"PSNR={result['psnr']:.3f} | SSIM={result['ssim']:.5f}"
                )

            ch_results["snr_sweep"][key] = {
                "rho_value": float(rho_fixed),
                "rho_fraction": str(rho_frac),
                "snr": snr_list,
                "psnr": psnr_list,
                "ssim": ssim_list,
                "mse": mse_list,
            }

        # rho sweep: PSNR/SSIM vs rho at fixed SNR
        for snr_fixed in fixed_snr_list:
            key = f"snr_{snr_fixed}"

            print(f"\nrho sweep @ SNR = {snr_fixed} dB")

            psnr_list = []
            ssim_list = []
            mse_list = []

            for rho in rho_list:
                rho_frac = Fraction(rho).limit_denominator()

                result = eval_point(
                    model=model,
                    loader=loader,
                    snr_db=snr_fixed,
                    rho=rho,
                    csi_dim=cfg.csi_dim,
                    channel_type=ch,
                )

                psnr_list.append(round(result["psnr"], 4))
                ssim_list.append(round(result["ssim"], 5))
                mse_list.append(round(result["mse"], 8))

                print(
                    f"SNR={snr_fixed:>4} dB | rho={str(rho_frac):<5} | "
                    f"PSNR={result['psnr']:.3f} | SSIM={result['ssim']:.5f}"
                )

            ch_results["rho_sweep"][key] = {
                "snr_value": float(snr_fixed),
                "rho": [round(float(r), 8) for r in rho_list],
                "rho_fraction": [
                    str(Fraction(r).limit_denominator()) for r in rho_list
                ],
                "psnr": psnr_list,
                "ssim": ssim_list,
                "mse": mse_list,
            }

        all_results[ch] = ch_results

    return all_results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checks/div2k",
    )

    parser.add_argument(
        "--data_root",
        type=str,
        default="../data_set",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outs/div2k",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("\nLoading Kodak/DIV2K test loader ...")
    _, _, test_loader = make_div2k_kodak_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Test images: {len(test_loader.dataset)}")

    print("\nLoading full model checkpoint ...")
    model, cfg = load_full_model(args.ckpt)

    # Full SNR sweep
    snr_sweep_list = list(range(0, 26, 1))   # 0, 1, 2, ..., 25
    
    # Full rho sweep
    rho_sweep_list = [
        1/48, 1/24, 1/12, 1/10, 1/8, 1/6,
        0.2, 0.25, 1/3, 0.4, 0.5, 0.6,
        0.7, 0.8, 1.0
    ]
    
    # Fixed points
    fixed_rho_list = [1/6, 1/12, 1/24]
    fixed_snr_list = [0, 5, 10]
    eval_channels = ["awgn", "rayleigh", "rician"]

    results = run_eval(
        model=model,
        loader=test_loader,
        cfg=cfg,
        snr_list=snr_sweep_list,
        rho_list=rho_sweep_list,
        fixed_rho_list=fixed_rho_list,
        fixed_snr_list=fixed_snr_list,
        eval_channels=eval_channels,
    )

    json_path = os.path.join(args.out_dir, "full_crosschannel_eval.json")

    payload = {
        "checkpoint": args.ckpt,
        "trained_channel": "awgn",
        "eval_channels": eval_channels,
    
        "snr_sweep_list": snr_sweep_list,
        "rho_sweep_list": [float(r) for r in rho_sweep_list],
    
        "fixed_rho_list": [float(r) for r in fixed_rho_list],
        "fixed_rho_fraction": [
            str(Fraction(r).limit_denominator()) for r in fixed_rho_list
        ],
    
        "fixed_snr_list": fixed_snr_list,
    
        "metrics": ["psnr", "ssim", "mse"],
        "results": results,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved JSON result to:")
    print(json_path)


if __name__ == "__main__":
    main()