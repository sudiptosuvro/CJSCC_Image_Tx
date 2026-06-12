#!/bin/bash
#PBS -N cjscc_c256_s5_hard
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=16:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs checks/stage5_random_hard_c256

LOGFILE="logs/stage5_random_hard_c256_${PBS_JOBID}.log"

python -u train_cifar10.py \
  --data_root ../data \
  --ablation full \
  --latent_channels 256 \
  --channel random \
  --eval_channel rayleigh \
  --epochs 200 \
  --patience 30 \
  --batch_size 128 \
  --num_workers 4 \
  --lr 2e-5 \
  --seed 0 \
  --snr_min 2 --snr_max 25 \
  --rho_min 1/24 --rho_max 1.0 \
  --eval_snr 5 \
  --eval_rho 1/6 \
  --csi_dim 16 \
  --bwmask_mode prefix \
  --grad_clip 1.0 \
  --resume_path checks/stage4_random_c256/cifar10_C256_random_full_seed0_best.pt \
  --save_dir checks/stage5_random_hard_c256 \
  > "$LOGFILE" 2>&1
