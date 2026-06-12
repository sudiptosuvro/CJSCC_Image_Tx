#!/bin/bash
#PBS -N ofdm_train
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=80gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -m abe

cd "$PBS_O_WORKDIR" || exit 1

echo "============================================"
echo "OFDM training job started"
echo "Job ID:      $PBS_JOBID"
echo "Node:        $(hostname)"
echo "Start time:  $(date)"
echo "Working dir: $(pwd)"
echo "============================================"

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

# Create logs directory
mkdir -p logs

# Log file
LOGFILE="logs/ofdm_train_${PBS_JOBID}.log"

# Run training
python -u train_ofdm.py 2>&1 | tee -a "$LOGFILE"

echo "============================================"
echo "Job finished at $(date)"
echo "============================================"