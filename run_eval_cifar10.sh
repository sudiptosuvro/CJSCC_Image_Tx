#!/bin/bash
#PBS -N eval_cifar10_cross
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs outs/cifar10

LOGFILE="logs/eval_cifar10_cross_${PBS_JOBID}.log"

python -u eval_cifar10.py \
  --ckpt_dir ./checks \
  --data_root ./data \
  --out_dir ./outs/cifar10 \
  --C 192 \
  --seed 0 \
  --batch_size 128 \
  > "$LOGFILE" 2>&1