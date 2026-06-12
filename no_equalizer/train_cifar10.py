import os
import json
import time
import random
import argparse
from fractions import Fraction
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

import sys

THIS_DIR  = Path(__file__).resolve().parent
ROOT_DIR  = THIS_DIR.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.data_lowres import make_cifar10_loaders
from utils.metrics import psnr_torch, ssim_torch
from model import MASCConfig, CJSCC, make_channel
import model as model_module
import utils.channels
import lcdg as lcdg_module

print("TRAIN loaded model.py from:", model_module.__file__, flush=True)
print("TRAIN loaded lcdg.py from:", lcdg_module.__file__, flush=True)
print("TRAIN loaded channels.py from:", channels.__file__, flush=True)


def get_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_or_fraction(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float(Fraction(x))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sample_snr_db(B: int, snr_min: float, snr_max: float, device, dtype) -> torch.Tensor:
    return torch.empty(B, device=device, dtype=dtype).uniform_(snr_min, snr_max)


def sample_rho(rho_min: float, rho_max: float) -> float:
    return float(random.uniform(rho_min, rho_max))


def sample_channel_type(channel_mode: str) -> str:
    if channel_mode == "random":
        return random.choice(["awgn", "rayleigh", "rician"])
    return channel_mode


def build_csi_from_snr_rho_channel(
    snr_db: torch.Tensor,
    rho: float,
    csi_dim: int,
    channel_type: str,
) -> torch.Tensor:
    
    B      = snr_db.size(0)
    device = snr_db.device
    dtype  = snr_db.dtype

    rho_tensor = torch.full((B,), float(rho), device=device, dtype=dtype)
    snr_norm   = (snr_db - 12.5) / 12.5
    rho_norm   = torch.log(rho_tensor / (1.0 / 48.0) + 1e-8)

    csi = torch.zeros(B, csi_dim, device=device, dtype=dtype)
    csi[:, 0] = snr_norm
    csi[:, 1] = rho_norm
    csi[:, 2] = rho_tensor

    ct = channel_type.lower()
    if   ct == "awgn":     csi[:, 3] = 1.0
    elif ct == "rayleigh": csi[:, 4] = 1.0
    elif ct == "rician":   csi[:, 5] = 1.0
    else:
        raise ValueError(f"Unknown channel_type={channel_type!r}. Use: awgn | rayleigh | rician")

    return csi


def save_checkpoint(path: str, cfg, args, epoch: int,
                    model, opt, best_psnr: float, n_params: int) -> None:
    torch.save({
        "cfg":        cfg.__dict__,
        "args":       vars(args),
        "epoch":      epoch,
        "state_dict": model.state_dict(),
        "opt_state":  opt.state_dict(),
        "best_psnr":  best_psnr,
        "n_params":   n_params,
    }, path)


def load_checkpoint(path: str, model, opt=None, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if opt is not None and "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
    epoch      = ckpt.get("epoch", 0)
    best_psnr  = ckpt.get("best_psnr", -1e9)
    return epoch, best_psnr


@torch.no_grad()
def evaluate(
    model, loader, device, *,
    snr_db_eval: float,
    rho_eval: float,
    csi_dim: int,
    eval_channel: str,
) -> dict:
    model.eval()

    model.cfg.channel_type = eval_channel
    model.channel = make_channel(
        eval_channel,
        rician_K=model.cfg.rician_K,
        per_symbol=model.cfg.per_symbol_fading,
    ).to(device)

    mse_sum = psnr_sum = ssim_sum = 0.0
    n = 0

    for x, _ in loader:
        x   = x.to(device).float()
        B   = x.size(0)
        snr = torch.full((B,), float(snr_db_eval), device=device, dtype=x.dtype)
        csi = build_csi_from_snr_rho_channel(snr, rho_eval, csi_dim, eval_channel)

        x_hat, _ = model(x, snr_db=snr, rho=float(rho_eval), csi=csi)
        x_hat    = torch.clamp(x_hat, 0.0, 1.0)

        mse_sum  += ((x_hat - x) ** 2).flatten(1).mean(dim=1).sum().item()
        psnr_sum += psnr_torch(x_hat, x).sum().item()
        ssim_sum += ssim_torch(x_hat, x).sum().item()
        n        += B

    return {
        "mse":  mse_sum  / n,
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
    }



@torch.no_grad()
def eval_snr_sweep(
    model, loader, device,
    snr_range: list, rho: float, csi_dim: int, channel: str,
) -> dict:

    results = {}
    for snr in snr_range:
        s = evaluate(model, loader, device,
                     snr_db_eval=snr, rho_eval=rho,
                     csi_dim=csi_dim, eval_channel=channel)
        results[snr] = {"psnr": round(s["psnr"], 4), "ssim": round(s["ssim"], 5)}
        print(f"  SNR={snr:5.1f} dB  |  PSNR={s['psnr']:.3f}  |  SSIM={s['ssim']:.4f}")
    return results


@torch.no_grad()
def eval_rho_sweep(
    model, loader, device,
    rho_range: list, snr_db: float, csi_dim: int, channel: str,
) -> dict:

    results = {}
    for rho in rho_range:
        s = evaluate(model, loader, device,
                     snr_db_eval=snr_db, rho_eval=rho,
                     csi_dim=csi_dim, eval_channel=channel)
        results[rho] = {"psnr": round(s["psnr"], 4), "ssim": round(s["ssim"], 5)}
        print(f"  rho={rho:.4f}  |  PSNR={s['psnr']:.3f}  |  SSIM={s['ssim']:.4f}")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="CIFAR-10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--batch_size",   type=int,   default=128)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--save_dir",     type=str,   default="./runs/cifar10")
    ap.add_argument("--resume_path", type=str, default=None)

    # Training schedule
    ap.add_argument("--epochs",       type=int,   default=50,
                    help="Set to 0 with --resume best to run sweeps only.")
    ap.add_argument("--patience",     type=int,   default=100,
                    help="Early stopping patience in epochs.")
    ap.add_argument("--min_delta",    type=float, default=1e-4,
                    help="Minimum PSNR improvement to count as progress.")
    ap.add_argument("--lr",           type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--grad_clip",    type=float, default=1.0,
                    help="Max gradient norm.")

    # Channel conditions
    ap.add_argument("--channel",      type=str,   default="random",
                    choices=["awgn", "rayleigh", "rician", "random"])
    ap.add_argument("--eval_channel", type=str,   default="rayleigh",
                    choices=["awgn", "rayleigh", "rician"])
    ap.add_argument("--rician_K",     type=float, default=5.0)
    ap.add_argument("--snr_min",      type=float, default=0.0)
    ap.add_argument("--snr_max",      type=float, default=25.0)
    ap.add_argument("--rho_min",      type=parse_float_or_fraction, default="1/48")
    ap.add_argument("--rho_max",      type=parse_float_or_fraction, default=1.0)

    # Eval checkpoint
    ap.add_argument("--eval_snr",     type=float, default=5.0,
                    help="Fixed SNR (dB) used for the per-epoch val checkpoint.")
    ap.add_argument("--eval_rho",     type=parse_float_or_fraction, default=0.5,
                    help="Fixed rho used for the per-epoch val checkpoint.")

    # Model architecture
    ap.add_argument("--latent_channels", type=int, default=96)
    ap.add_argument("--csi_dim",         type=int, default=16)
    ap.add_argument("--bwmask_mode",     type=str, default="prefix",
                    choices=["prefix", "random"])
    ap.add_argument("--use_img_m11",     action="store_true")

    # Ablation
    ap.add_argument("--ablation", type=str, default="full",
                    choices=["full", "no_lcdg", "no_csi"]))
    ap.add_argument("--no_compensating_conv", action="store_true",
                    help="no_lcdg only.")

    ap.add_argument("--resume", type=str, default=None,
                    choices=["last", "best"],
                    help="Resume from last or best checkpoint before training.")

    # Post-training sweeps
    ap.add_argument("--eval_snr_sweep", action="store_true",
                    help="After training: PSNR/SSIM vs SNR on val set.")
    ap.add_argument("--eval_rho_sweep", action="store_true",
                    help="After training: PSNR/SSIM vs rho on val set.")

    args = ap.parse_args(argv) if argv is not None else ap.parse_args()
    set_seed(args.seed)

    device = get_device()
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_cifar10_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    cfg = MASCConfig(
        in_channels=3,
        latent_channels=args.latent_channels,
        enc_k=(9, 5, 5, 5, 5),
        enc_s=(2, 2, 1, 1, 1),
        dec_k=(5, 5, 5, 5, 5),
        dec_up=(1, 1, 1, 2, 2),
        csi_dim=args.csi_dim,
        channel_type=("rayleigh" if args.channel == "random" else args.channel),
        rician_K=args.rician_K,
        bwmask_mode=args.bwmask_mode,
        use_img_m11=bool(args.use_img_m11),
        ablation_mode=args.ablation,
        add_compensating_conv=False,
    )

    model = CJSCC(cfg).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("Model parameter count:", n_params, flush=True)
    
    print("Has iq_equalizer:", hasattr(model, "iq_equalizer"), flush=True)
    print("Has latent_recovery:", hasattr(model, "latent_recovery"), flush=True)
    print("Has decoder refine:", hasattr(model.dec, "refine"), flush=True)
    opt      = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_name  = (f"cifar10_C{args.latent_channels}"
                 f"_{args.channel}_{args.ablation}_seed{args.seed}")
    ckpt_last = os.path.join(args.save_dir, f"{run_name}_last.pt")
    ckpt_best = os.path.join(args.save_dir, f"{run_name}_best.pt")
    meta_path = os.path.join(args.save_dir, f"{run_name}_meta.json")

    with open(meta_path, "w") as f:
        json.dump({**vars(args), "n_params": n_params}, f, indent=2, default=str)

    start_epoch = 0
    best_psnr   = -1e9

    if args.resume_path is not None:
        resume_path = args.resume_path
    elif args.resume is not None:
        resume_path = ckpt_best if args.resume == "best" else ckpt_last
    else:
        resume_path = None
    
    if resume_path is not None:
        if os.path.isfile(resume_path):
            _, _ = load_checkpoint(resume_path, model, opt=None, device=device)
            start_epoch = 0
            best_psnr = -1e9
            print(f"Loaded model weights from {resume_path}")
        else:
            print(f"[WARNING] Resume checkpoint not found: {resume_path}. Starting from scratch.")

    print(f"\nDevice      : {device}")
    print(f"Ablation    : {args.ablation}  |  Params: {n_params:,}")
    print(f"Latent C    : {args.latent_channels}  |  Latent spatial: 8×8")
    print(f"Channel     : {args.channel}"
          f"  |  SNR~U[{args.snr_min}, {args.snr_max}] dB"
          f"  |  rho~U[{args.rho_min:.4f}, {args.rho_max}]")
    print(f"Eval point  : SNR={args.eval_snr} dB, rho={args.eval_rho}, ch={args.eval_channel}")
    print(f"Train size  : {len(train_loader.dataset)}"
          f"  |  Val size: {len(val_loader.dataset)}"
          f"  |  Test size: {len(test_loader.dataset)}")
    print(f"Saving to   : {args.save_dir}\n")

    no_improve_count = 0

    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        model.train()
        t0       = time.time()
        loss_sum = 0.0
        n_seen   = 0

        pbar = tqdm(
            train_loader,
            desc=f"[{args.ablation}] epoch {epoch}/{start_epoch + args.epochs}",
        )

        for x, _ in pbar:
            x  = x.to(device).float()
            B  = x.size(0)

            snr_db        = sample_snr_db(B, args.snr_min, args.snr_max, device, x.dtype)
            rho           = sample_rho(args.rho_min, args.rho_max)
            batch_channel = sample_channel_type(args.channel)

            model.cfg.channel_type = batch_channel
            model.channel = make_channel(
                batch_channel,
                rician_K=args.rician_K,
                per_symbol=cfg.per_symbol_fading,
            ).to(device)

            csi = build_csi_from_snr_rho_channel(snr_db, rho, args.csi_dim, batch_channel)

            x_hat, _ = model(x, snr_db=snr_db, rho=rho, csi=csi)
            x_hat    = torch.clamp(x_hat, 0.0, 1.0)

            loss = ((x_hat - x) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()

            loss_sum += loss.item() * B
            n_seen   += B

            pbar.set_postfix(
                loss=f"{loss_sum / max(1, n_seen):.5f}",
                rho=f"{rho:.3f}",
                ch=batch_channel[:3],
            )

        eval_stats = evaluate(
            model, val_loader, device,
            snr_db_eval=args.eval_snr,
            rho_eval=args.eval_rho,
            csi_dim=args.csi_dim,
            eval_channel=args.eval_channel,
        )

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}  |  "
            f"train_mse={loss_sum / max(1, n_seen):.6f}  |  "
            f"val@({args.eval_snr}dB, rho={args.eval_rho})  "
            f"MSE={eval_stats['mse']:.6f}  "
            f"PSNR={eval_stats['psnr']:.3f}  "
            f"SSIM={eval_stats['ssim']:.4f}  |  "
            f"time={dt:.1f}s"
        )

        save_checkpoint(ckpt_last, cfg, args, epoch, model, opt, best_psnr, n_params)

        if eval_stats["psnr"] > best_psnr + args.min_delta:
            best_psnr = eval_stats["psnr"]
            save_checkpoint(ckpt_best, cfg, args, epoch, model, opt, best_psnr, n_params)
            print(f"  ✓ New best PSNR={best_psnr:.3f}  →  {ckpt_best}")
            no_improve_count = 0
        else:
            no_improve_count += 1
            print(f"  No improvement for {no_improve_count}/{args.patience} epochs")
        
        if no_improve_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    if args.eval_snr_sweep or args.eval_rho_sweep:
        print(f"\nLoading best checkpoint for sweeps: {ckpt_best}")
        load_checkpoint(ckpt_best, model, device=device)
    
    if args.eval_snr_sweep:
        snr_range = [0, 2, 5, 8, 10, 12, 15, 20, 25]
        print(f"\n=== SNR sweep  |  ablation={args.ablation}"
              f"  |  rho={args.eval_rho}  |  ch={args.eval_channel} ===")
        snr_results = eval_snr_sweep(
            model, val_loader, device,
            snr_range=snr_range,
            rho=args.eval_rho,
            csi_dim=args.csi_dim,
            channel=args.eval_channel,
        )
        out = os.path.join(args.save_dir, f"{run_name}_snr_sweep.json")
        with open(out, "w") as f:
            json.dump(snr_results, f, indent=2)
        print(f"  Saved → {out}")
    
    if args.eval_rho_sweep:
        rho_range = [1/12, 1/6, 1/4, 1/3, 0.5, 2/3, 1.0]
        print(f"\n=== rho sweep  |  ablation={args.ablation}"
              f"  |  SNR={args.eval_snr} dB  |  ch={args.eval_channel} ===")
        rho_results = eval_rho_sweep(
            model, val_loader, device,
            rho_range=rho_range,
            snr_db=args.eval_snr,
            csi_dim=args.csi_dim,
            channel=args.eval_channel,
        )
        out = os.path.join(args.save_dir, f"{run_name}_rho_sweep.json")
        with open(out, "w") as f:
            json.dump({str(round(k, 6)): v for k, v in rho_results.items()}, f, indent=2)
        print(f"  Saved → {out}")

    print(f"\nDone.")
    print(f"Last : {ckpt_last}")
    print(f"Best : {ckpt_best}")
    print(f"Meta : {meta_path}")


if __name__ == "__main__":
    main()