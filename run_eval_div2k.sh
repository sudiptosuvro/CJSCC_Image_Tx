#!/bin/bash
#PBS -N eval_div2k
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs outs/div2k

LOGFILE="logs/eval_div2k_${PBS_JOBID}.log"

python -u eval_div2k.py \
  --ckpt_dir ./checks/div2k \
  --data_root ./data_set \
  --out_dir ./outs/div2k \
  --C 192 \
  --seed 0 \
  --batch_size 16 \
  > "$LOGFILE" 2>&1