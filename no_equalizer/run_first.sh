#!/bin/bash
#PBS -N cjscc_c256_s1_awgn
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=80gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs checks/stage1_awgn_c256

LOGFILE="logs/stage1_awgn_c256_${PBS_JOBID}.log"

python -u train_cifar10.py \
  --data_root ../data \
  --ablation full \
  --latent_channels 256 \
  --channel awgn \
  --eval_channel awgn \
  --epochs 150 \
  --patience 20 \
  --batch_size 128 \
  --num_workers 4 \
  --lr 2e-4 \
  --seed 0 \
  --snr_min 10 --snr_max 25 \
  --rho_min 1/6 --rho_max 1.0 \
  --eval_snr 15 \
  --eval_rho 1/3 \
  --csi_dim 16 \
  --bwmask_mode prefix \
  --grad_clip 1.0 \
  --save_dir checks/stage1_awgn_c256 \
  > "$LOGFILE" 2>&1
