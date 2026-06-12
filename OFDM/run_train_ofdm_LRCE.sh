#!/bin/bash
#PBS -N ofdm_LE_cifar
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=64gb
#PBS -l walltime=36:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"
echo "Starting at: $(date)"

mkdir -p logs
mkdir -p runs

python -u train_ofdm_LRCE.py \
  | tee "logs/train_ofdm_LE_cifar_${PBS_JOBID}.log"

echo "Finished at: $(date)"