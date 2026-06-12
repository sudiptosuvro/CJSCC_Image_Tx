#!/bin/bash
#PBS -N cjscc_div2k
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=80gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs checks/div2k

LOGFILE="logs/cjscc_div2k_${PBS_JOBID}.log"

python -u train_div2k.py \
  --data_root ./data_set \
  --save_dir ./checks/div2k \
  --epochs 1000 \
  --early_stop_patience 20 \
  --batch_size 4 \
  --latent_channels 192 \
  --channel awgn \
  --eval_channel awgn \
  --snr_min 0 \
  --snr_max 25 \
  --rho_min 1/24 \
  --rho_max 1.0 \
  --eval_snr 5 \
  --eval_rho 0.5 \
  --ablation full \
  --seed 0
  > "$LOGFILE" 2>&1