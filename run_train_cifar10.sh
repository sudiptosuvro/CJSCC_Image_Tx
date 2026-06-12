#!/bin/bash
#PBS -N cjscc_cifar10
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs checks/cifar10

LOGFILE="logs/cjscc_cifar10_${PBS_JOBID}.log"

python -u train_cifar10.py \
  --data_root ./data \
  --save_dir ./checks/cifar10 \
  --epochs 500 \
  --early_stop_patience 20 \
  --batch_size 128 \
  --num_workers 4 \
  --latent_channels 192 \
  --channel rayleigh \
  --eval_channel rayleigh \
  --snr_min 0 \
  --snr_max 25 \
  --rho_min 1/48 \
  --rho_max 1.0 \
  --eval_snr 5 \
  --eval_rho 0.5 \
  --ablation full \
  --bwmask_mode prefix \
  --seed 0 \
  > "$LOGFILE" 2>&1