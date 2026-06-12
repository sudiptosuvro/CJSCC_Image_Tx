# 📡 CJSCC_Image_Tx

**Deep Joint Source-Channel Coding for Semantic-aware Wireless Image Transmission**

This repository contains the implementation details of a DeepJSCC framework for robust image transmission over fading channels including OFDM.

---

## 🗂️ Repository Structure

```
CJSCC_Image_Tx/
├── OFDM/                       # OFDM modulation, channel estimation & equalization
│   ├── core/
│   │   ├── OFDM_system.py        # End-to-end OFDM system pipeline
│   │   ├── channel_estimator.py  # LRCE
│   │   ├── ofdm.py               # OFDM modulation/demodulation
│   │   ├── ofdm_helpers.py       # Helper functions for OFDM processing
│   │   └── subcarrier_allocator.py # Subcarrier allocation strategies
│   ├── eval.py                  # Evaluation script for OFDM pipeline
│   ├── train_ofdm.py             # Training script (standard OFDM)
│   ├── train_ofdm_LRCE.py         # Training script (Channel Estimator)
│   ├── run_eval.sh
│   ├── run_train_ofdm.sh
│   └── run_train_ofdm_LRCE.sh
│
├── data/                        # Dataset loading utilities
│   ├── data_highres.py          # High-resolution dataset loader (DIV2K+KODAK)
│   └── data_lowres.py           # Low-resolution dataset loader (CIFAR-10)
│
├── model/                        # Core JSCC model architectures
│   ├── enc_dec.py                # Encoder / Decoder networks
│   ├── lcdg.py                    # Lightweight CSI-aware Dynamic Gating
│   └── system_model.py            # Full end-to-end system model
│
├── no_equalizer/                  # Baseline variant without channel equalization
│   ├── lcdg.py
│   ├── model.py
│   ├── train_cifar10.py
│   ├── eval_cifar10.py
│   └── run_*.sh                   # Multiple training run configurations
│
├── utils/                          # Shared utility modules
│   ├── channels.py                  # Wireless channel models AWGN, Rayleigh, and Rician
│   ├── channels_jscc.py              # Channel models with equalizer
│   ├── latent.py                       # BW Masking and Power Normalization
│   └── metrics.py                       # MSE / PSNR / SSIM
│
├── train_cifar10.py               # Main training script — CIFAR-10
├── train_div2k.py                 # Main training script — DIV2K
├── eval_cifar10.py                 # Evaluation — CIFAR-10
├── eval_div2k.py                    # Evaluation — KODAK
├── run_train_cifar10.sh
├── run_train_div2k.sh
├── run_eval_cifar10.sh
├── run_eval_div2k.sh
│
├── LICENSE
└── README.md
```

---

## ⚙️ Requirements

This project was developed and tested with the following environment:

| Component | Version |
|-----------|---------|
| Python    | `3.10.20` |
| PyTorch   | `2.5.1+cu121` |
| CUDA      | `12.1` |

### Installation

```bash
# Create a clean environment
conda create -n cjscc python=3.10.20 -y
conda activate cjscc

# Install PyTorch with CUDA 12.1 support
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> 💡 **Note:** A GPU with CUDA 12.1 support is strongly recommended for training. Evaluation scripts can run on CPU but will be significantly slower.

---

## 🚀 Quick Start

### 1️⃣ Training

**CIFAR-10 (low-resolution)**
```bash
bash run_train_cifar10.sh
```

**DIV2K (high-resolution)**
```bash
bash run_train_div2k.sh
```

**OFDM-based pipeline (Stage-C)**
```bash
cd OFDM
bash run_train_ofdm.sh
```

**OFDM-based pipeline with Channel Estimation (LRCE)**
```bash
cd OFDM
bash run_train_ofdm_LRCE.sh
```

**No-Equalizer baseline**
```bash
cd no_equalizer
bash run_first.sh   # and subsequent run_*.sh scripts for additional configs
```

---

### 2️⃣ Evaluation

**CIFAR-10**
```bash
bash run_eval_cifar10.sh
```

**DIV2K**
```bash
bash run_eval_div2k.sh
```

**OFDM pipeline**
```bash
cd OFDM
bash run_eval.sh
```

**No-Equalizer baseline**
```bash
cd no_equalizer
bash run_eval.sh
```

---

## 🧠 Model Overview

- **`model/enc_dec.py`** — DeepJSCC encoder/decoder networks that map images to channel symbols and back.
- **`model/lcdg.py`** — Dynamic gating with channel state information (CSI).
- **`model/system_model.py`** — Combines encoder, decoder, and channel models into an end-to-end trainable pipeline.
- **`OFDM/core/`** — Implements OFDM modulation/demodulation, subcarrier allocation, and channel estimation.
- **`utils/channels.py`** & **`utils/channels_jscc.py`** — Define wireless channel models (AWGN, Rayleigh, and Rician fading) used during training and evaluation.

---

## 📊 Datasets

- **CIFAR-10** — Used for low-resolution image transmission experiments (`data/data_lowres.py`).
- **DIV2K** — Used for high-resolution image transmission experiments (`data/data_highres.py`).

Make sure to update dataset paths in the corresponding `run_*.sh` scripts or config sections of the training scripts before running.

---

## 📈 Metrics

Evaluation is performed using standard image quality metrics implemented in `utils/metrics.py`, including:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

---

## 🔁 Reproducibility Checklist

- ✅ Fix random seeds in training scripts for deterministic runs.
- ✅ Use the exact dependency versions listed in [Requirements](#️-requirements).
- ✅ Verify dataset paths and channel SNR settings in `run_*.sh` scripts before launching experiments.
- ✅ Log training configurations (saved automatically if enabled in training scripts) for traceability.
