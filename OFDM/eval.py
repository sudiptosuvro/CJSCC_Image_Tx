import sys
import json
from pathlib import Path

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_lowres import make_cifar10_loaders
from model.system_model import MASCConfig, CJSCC
from utils.metrics import psnr_torch, ssim_torch
from utils.latent import PowerNorm
from OFDM.core.ofdm import OFDMConfig, OFDMJSCCMiddle, OFDMJSCCMiddleLearnableCE
from OFDM.core.OFDM_system import CJSCC_OFDM

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

os.makedirs("./runs", exist_ok=True)
os.makedirs("./runs/figures", exist_ok=True)
os.makedirs("./runs/results", exist_ok=True)


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
    else:
        raise ValueError(f"Unknown channel_type={channel_type}")

    return csi

def build_baselines():
    ckpt = torch.load(
        "../runs/cifar10_C192/rayleigh/cifar10_C192_rayleigh_full_seed0_best.pt",
        map_location=device
    )
    cfg = MASCConfig(**ckpt["cfg"])
    assert cfg.latent_channels == 192, f"Expected C=192, got C={cfg.latent_channels}"
    
    model = CJSCC(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    encoder = model.enc
    decoder = model.dec
    encoder.eval()
    decoder.eval()

    ofdm_cfg = OFDMConfig(
        nfft=64,
        cp_len=16,
        n_pilot=1,
        n_taps=8,
        delay_decay=4.0,
        equalizer="mmse",
    )

    ofdm_std = OFDMJSCCMiddle(ofdm_cfg).to(device)
    ofdm_std.eval()

    ofdm_learnable = OFDMJSCCMiddleLearnableCE(ofdm_cfg).to(device)
    ckpt_ce = torch.load("./runs/ofdm_LE.pt", map_location=device)
    ofdm_learnable.load_state_dict(ckpt_ce["model_state"], strict=True)
    ofdm_learnable.eval()

    return cfg, ofdm_cfg, encoder, decoder, ofdm_std, ofdm_learnable


def build_ranked_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg = MASCConfig(**ckpt["cfg"])
    ofdm_cfg = OFDMConfig(**ckpt["ofdm_cfg"])

    model_ranked = CJSCC_OFDM(cfg, ofdm_cfg, clip_ratio=None).to(device)
    model_ranked.load_state_dict(ckpt["model_state"], strict=True)
    model_ranked.eval()

    return model_ranked, cfg, ofdm_cfg

def complex_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.view_as_real(a) - torch.view_as_real(b)) ** 2)


@torch.no_grad()
def eval_baseline_model_over_loader(
    encoder,
    decoder,
    ofdm_middle,
    data_loader,
    device,
    cfg,
    snr_val=5.0,
):
    encoder.eval()
    decoder.eval()
    ofdm_middle.eval()

    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    total_h = 0.0
    total_z = 0.0
    total_x = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(data_loader, desc=f"Eval baseline @{snr_val} dB", leave=False):
        x = x.to(device)
        B = x.size(0)
        rho = 1.0
        snr_vals = torch.full((B,), float(snr_val), device=device)
        csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

        z = encoder(x, csi)
        z_hat, aux = ofdm_middle(z, snr_db=snr_db, use_perfect_csi=False)
        x_hat = decoder(z_hat, csi)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        h_loss = complex_mse(aux["H_hat"], aux["Hf_true"]).item()
        z_loss = torch.mean((z - z_hat) ** 2).item()
        x_loss = torch.mean((x - x_hat) ** 2).item()
        psnr = psnr_torch(x_hat, x).mean().item()
        ssim = ssim_torch(x_hat, x).mean().item()

        total_h += h_loss
        total_z += z_loss
        total_x += x_loss
        total_psnr += psnr
        total_ssim += ssim
        n += 1

    return {
        "h_mse": total_h / max(n, 1),
        "z_mse": total_z / max(n, 1),
        "x_mse": total_x / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def eval_baseline_model_over_loader_rho(
    encoder,
    decoder,
    ofdm_middle,
    data_loader,
    device,
    cfg,
    snr_val=5.0,
    rho=1/6,
):
    encoder.eval()
    decoder.eval()
    ofdm_middle.eval()

    pnorm = PowerNorm(
        target_power=cfg.target_power,
        per_sample=cfg.power_per_sample,
        iq_interleaved=True,
    ).to(device)

    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    total_h = 0.0
    total_z = 0.0
    total_x = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(data_loader, desc=f"Eval baseline rho={rho:.4f}", leave=False):
        x = x.to(device)
        B = x.size(0)
        snr_vals = torch.full((B,), float(snr_val), device=device)
        csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

        # semantic encoder
        z = encoder(x, csi)
        z_shape = z.shape

        Bz, C, H, W = z.shape
        K = max(1, min(C, int(round(float(rho) * C))))
        
        mask_c = torch.zeros(C, device=z.device, dtype=z.dtype)
        if cfg.bwmask_mode == "prefix":
            mask_c[:K] = 1.0
        elif cfg.bwmask_mode == "random":
            idx = torch.randperm(C, device=z.device)[:K]
            mask_c[idx] = 1.0
        else:
            raise ValueError(f"Unknown bwmask_mode={cfg.bwmask_mode}")
        
        mask = mask_c.view(1, C, 1, 1)
        
        z_masked = z * mask
        
        zf = z_masked.flatten(1)
        zf = pnorm(zf)
        z_masked = zf.view(z_shape)

        # OFDM middle baseline
        z_hat, aux = ofdm_middle(z_masked, snr_db=snr_db, use_perfect_csi=False)
        z_hat = z_hat * mask
        
        # semantic decoder
        x_hat = decoder(z_hat, csi)
        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        h_loss = complex_mse(aux["H_hat"], aux["Hf_true"]).item()
        z_loss = torch.mean((z_masked - z_hat) ** 2).item()
        x_loss = torch.mean((x - x_hat) ** 2).item()
        psnr = psnr_torch(x_hat, x).mean().item()
        ssim = ssim_torch(x_hat, x).mean().item()

        total_h += h_loss
        total_z += z_loss
        total_x += x_loss
        total_psnr += psnr
        total_ssim += ssim
        n += 1

    return {
        "h_mse": total_h / max(n, 1),
        "z_mse": total_z / max(n, 1),
        "x_mse": total_x / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def eval_ranked_model_over_loader(
    model_ranked,
    data_loader,
    device,
    cfg,
    snr_val=5.0,
    rho=1/6,
):
    model_ranked.eval()

    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    total_h = 0.0
    total_z = 0.0
    total_x = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0

    for x, _ in tqdm(data_loader, desc=f"Eval ranked @{snr_val} dB", leave=False):
        x = x.to(device)
        B = x.size(0)
        snr_vals = torch.full((B,), float(snr_val), device=device)
        csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

        # compute original latent explicitly for latent-MSE comparison
        x_in = model_ranked.img_norm(x)
        z = model_ranked.enc(x_in, csi)

        x_hat, aux = model_ranked(
            x,
            snr_db=snr_db,
            rho=rho,
            csi=csi,
            use_perfect_csi=False,
            use_perfect_alloc_csi=True,
        )

        x_hat = torch.clamp(x_hat, 0.0, 1.0)

        h_loss = complex_mse(aux["H_hat"], aux["Hf_true"]).item()
        z_loss = torch.mean((z - aux["z_hat"]) ** 2).item()
        x_loss = torch.mean((x - x_hat) ** 2).item()
        psnr = psnr_torch(x_hat, x).mean().item()
        ssim = ssim_torch(x_hat, x).mean().item()

        total_h += h_loss
        total_z += z_loss
        total_x += x_loss
        total_psnr += psnr
        total_ssim += ssim
        n += 1

    return {
        "h_mse": total_h / max(n, 1),
        "z_mse": total_z / max(n, 1),
        "x_mse": total_x / max(n, 1),
        "psnr": total_psnr / max(n, 1),
        "ssim": total_ssim / max(n, 1),
    }


@torch.no_grad()
def eval_ranked_model_rho_sweep(
    model_ranked,
    data_loader,
    device,
    cfg,
    snr_val=5.0,
    rho_list=None,
):
    if rho_list is None:
        rho_list = [1/12, 1/6, 1/3, 1/2, 1.0]

    model_ranked.eval()
    snr_db = torch.tensor([snr_val], device=device, dtype=torch.float32)

    results = {
        "rho": [],
        "h_mse": [],
        "z_mse": [],
        "x_mse": [],
        "psnr": [],
        "ssim": [],
    }

    for rho in rho_list:
        total_h = 0.0
        total_z = 0.0
        total_x = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        n = 0

        for x, _ in tqdm(data_loader, desc=f"Eval rho={rho:.4f}", leave=False):
            x = x.to(device)
            B = x.size(0)
            snr_vals = torch.full((B,), float(snr_val), device=device)
            csi = build_csi(snr_vals, rho, cfg, channel_type="rayleigh")

            x_in = model_ranked.img_norm(x)
            z = model_ranked.enc(x_in, csi)

            x_hat, aux = model_ranked(
                x,
                snr_db=snr_db,
                rho=rho,
                csi=csi,
                use_perfect_csi=False,
                use_perfect_alloc_csi=True,
            )

            x_hat = torch.clamp(x_hat, 0.0, 1.0)

            h_loss = complex_mse(aux["H_hat"], aux["Hf_true"]).item()
            z_loss = torch.mean((z - aux["z_hat"]) ** 2).item()
            x_loss = torch.mean((x - x_hat) ** 2).item()
            psnr = psnr_torch(x_hat, x).mean().item()
            ssim = ssim_torch(x_hat, x).mean().item()

            total_h += h_loss
            total_z += z_loss
            total_x += x_loss
            total_psnr += psnr
            total_ssim += ssim
            n += 1

        results["rho"].append(rho)
        results["h_mse"].append(total_h / max(n, 1))
        results["z_mse"].append(total_z / max(n, 1))
        results["x_mse"].append(total_x / max(n, 1))
        results["psnr"].append(total_psnr / max(n, 1))
        results["ssim"].append(total_ssim / max(n, 1))

    return results


def save_results_json(results: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON results to {filepath}")

def save_plot_pdf(x, curves, xlabel, ylabel, title, filepath):
    plt.figure(figsize=(7, 5))
    for y, label, marker in curves:
        plt.plot(x, y, marker=marker, label=label)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {filepath}")


def main():
    cfg_base, ofdm_cfg, encoder, decoder, ofdm_std, ofdm_learnable = build_baselines()

    # Ranked checkpoints
    model_stageB, cfg_stageB, _ = build_ranked_from_ckpt(
        "./runs/cjscc_ofdm_stageB_final.pt"
    )
    model_stageC2, cfg_stageC2, _ = build_ranked_from_ckpt(
        "./runs/cjscc_ofdm_stageC_final.pt"
    )

    _, _, test_loader = make_cifar10_loaders(
        batch_size=64,
        num_workers=0,
        pin_memory=False,
    )

    snr_list = [0, 3, 6, 9, 12, 15, 18, 21, 23, 25]

    snr_rho_list = [1/24, 1/12, 1/6]

    results = {
        "snr": snr_list,
        "rho_list": [float(r) for r in snr_rho_list],
        "data": {}
    }
    
    for rho_fixed in snr_rho_list:
        rho_key = f"rho_{rho_fixed:.6f}"
    
        results["data"][rho_key] = {
            "std_h": [], "std_z": [], "std_x": [], "std_psnr": [], "std_ssim": [],
            "learn_h": [], "learn_z": [], "learn_x": [], "learn_psnr": [], "learn_ssim": [],
            "stageB_h": [], "stageB_z": [], "stageB_x": [], "stageB_psnr": [], "stageB_ssim": [],
            "stageC2_h": [], "stageC2_z": [], "stageC2_x": [], "stageC2_psnr": [], "stageC2_ssim": [],
        }
    
        for snr in snr_list:
            print(f"\n===== Evaluating rho={rho_fixed:.6f}, SNR = {snr} dB =====")
    
            out_std = eval_baseline_model_over_loader_rho(
                encoder, decoder, ofdm_std, test_loader, device, cfg_base,
                snr_val=float(snr), rho=rho_fixed
            )
    
            out_learn = eval_baseline_model_over_loader_rho(
                encoder, decoder, ofdm_learnable, test_loader, device, cfg_base,
                snr_val=float(snr), rho=rho_fixed
            )
    
            out_stageB = eval_ranked_model_over_loader(
                model_stageB, test_loader, device, cfg_stageB,
                snr_val=float(snr), rho=rho_fixed
            )
    
            out_stageC2 = eval_ranked_model_over_loader(
                model_stageC2, test_loader, device, cfg_stageC2,
                snr_val=float(snr), rho=rho_fixed
            )
    
            bucket = results["data"][rho_key]
    
            bucket["std_h"].append(out_std["h_mse"])
            bucket["std_z"].append(out_std["z_mse"])
            bucket["std_x"].append(out_std["x_mse"])
            bucket["std_psnr"].append(out_std["psnr"])
            bucket["std_ssim"].append(out_std["ssim"])
    
            bucket["learn_h"].append(out_learn["h_mse"])
            bucket["learn_z"].append(out_learn["z_mse"])
            bucket["learn_x"].append(out_learn["x_mse"])
            bucket["learn_psnr"].append(out_learn["psnr"])
            bucket["learn_ssim"].append(out_learn["ssim"])
    
            bucket["stageB_h"].append(out_stageB["h_mse"])
            bucket["stageB_z"].append(out_stageB["z_mse"])
            bucket["stageB_x"].append(out_stageB["x_mse"])
            bucket["stageB_psnr"].append(out_stageB["psnr"])
            bucket["stageB_ssim"].append(out_stageB["ssim"])
    
            bucket["stageC2_h"].append(out_stageC2["h_mse"])
            bucket["stageC2_z"].append(out_stageC2["z_mse"])
            bucket["stageC2_x"].append(out_stageC2["x_mse"])
            bucket["stageC2_psnr"].append(out_stageC2["psnr"])
            bucket["stageC2_ssim"].append(out_stageC2["ssim"])
    
            print(
                f"rho={rho_fixed:.6f} | SNR={snr:>2} dB | "
                f"LS PSNR={out_std['psnr']:.3f}, "
                f"Learn PSNR={out_learn['psnr']:.3f}, "
                f"StageB PSNR={out_stageB['psnr']:.3f}, "
                f"StageC PSNR={out_stageC2['psnr']:.3f}"
            )

    save_results_json(results, "./runs/results/ofdm_snr_sweep_final.json")

    # Rho sweep at fixed SNR
    rho_list = [1/24, 1/12, 1/8, 1/6, 1/4, 1/3, 1/2, 1.0]

    rho_results = {
        "rho": [],
        "std_h": [], "std_z": [], "std_x": [], "std_psnr": [], "std_ssim": [],
        "learn_h": [], "learn_z": [], "learn_x": [], "learn_psnr": [], "learn_ssim": [],
        "stageB_h": [], "stageB_z": [], "stageB_x": [], "stageB_psnr": [], "stageB_ssim": [],
        "stageC2_h": [], "stageC2_z": [], "stageC2_x": [], "stageC2_psnr": [], "stageC2_ssim": [],
        "snr_fixed_db": 5.0,
    }

    print("\n===== Rho sweep @ 5 dB =====")
    for rho in rho_list:
        print(f"\n===== Evaluating rho = {rho:.4f} =====")

        out_std = eval_baseline_model_over_loader_rho(
            encoder, decoder, ofdm_std, test_loader, device, cfg_base,
            snr_val=5.0, rho=rho
        )

        out_learn = eval_baseline_model_over_loader_rho(
            encoder, decoder, ofdm_learnable, test_loader, device, cfg_base,
            snr_val=5.0, rho=rho
        )

        out_stageB = eval_ranked_model_over_loader(
            model_stageB, test_loader, device, cfg_stageB,
            snr_val=5.0, rho=rho
        )

        out_stageC2 = eval_ranked_model_over_loader(
            model_stageC2, test_loader, device, cfg_stageC2,
            snr_val=5.0, rho=rho
        )

        rho_results["rho"].append(rho)

        rho_results["std_h"].append(out_std["h_mse"])
        rho_results["std_z"].append(out_std["z_mse"])
        rho_results["std_x"].append(out_std["x_mse"])
        rho_results["std_psnr"].append(out_std["psnr"])
        rho_results["std_ssim"].append(out_std["ssim"])

        rho_results["learn_h"].append(out_learn["h_mse"])
        rho_results["learn_z"].append(out_learn["z_mse"])
        rho_results["learn_x"].append(out_learn["x_mse"])
        rho_results["learn_psnr"].append(out_learn["psnr"])
        rho_results["learn_ssim"].append(out_learn["ssim"])

        rho_results["stageB_h"].append(out_stageB["h_mse"])
        rho_results["stageB_z"].append(out_stageB["z_mse"])
        rho_results["stageB_x"].append(out_stageB["x_mse"])
        rho_results["stageB_psnr"].append(out_stageB["psnr"])
        rho_results["stageB_ssim"].append(out_stageB["ssim"])

        rho_results["stageC2_h"].append(out_stageC2["h_mse"])
        rho_results["stageC2_z"].append(out_stageC2["z_mse"])
        rho_results["stageC2_x"].append(out_stageC2["x_mse"])
        rho_results["stageC2_psnr"].append(out_stageC2["psnr"])
        rho_results["stageC2_ssim"].append(out_stageC2["ssim"])

        print(
            f"rho={rho:.4f} | "
            f"LS PSNR={out_std['psnr']:.3f}, "
            f"Learn PSNR={out_learn['psnr']:.3f}, "
            f"StageB PSNR={out_stageB['psnr']:.3f}, "
            f"StageC PSNR={out_stageC2['psnr']:.3f}"
        )
        print(
            f"           | "
            f"LS SSIM={out_std['ssim']:.4f}, "
            f"Learn SSIM={out_learn['ssim']:.4f}, "
            f"StageB SSIM={out_stageB['ssim']:.4f}, "
            f"StageC SSIM={out_stageC2['ssim']:.4f}"
        )

    save_results_json(rho_results, "./runs/results/ofdm_rho_sweep_final.json")

    return results, rho_results


if __name__ == "__main__":
    results, rho_results = main()