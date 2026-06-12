#!/bin/bash
#PBS -N eval_c192_final
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs checks/eval

LOGFILE="logs/eval_fine_tuned_${PBS_JOBID}.log"

python -u eval_cifar10.py \
  --ckpt checks/stage6_random_final_c256/cifar10_C256_random_full_seed0_best.pt \
  --data_root ../data \
  --out_dir checks/eval_may27 \
  --batch_size 128 \
  --num_workers 4 \
  > "$LOGFILE" 2>&1
