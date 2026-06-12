#!/bin/bash
#PBS -N eval_ofdm_train
#PBS -q gpu
#PBS -l select=1:ncpus=4:ngpus=1:mem=16gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -m ae

cd $PBS_O_WORKDIR

source ~/.bashrc
conda activate /scratch/$USER/conda_envs/venvs

mkdir -p logs

LOGFILE="logs/eval_${PBS_JOBID}.log"

echo "Job started on $(hostname)" | tee -a "$LOGFILE"
echo "Start time: $(date)" | tee -a "$LOGFILE"
echo "Working dir: $(pwd)" | tee -a "$LOGFILE"

python -u eval.py 2>&1 | tee -a "$LOGFILE"

status=${PIPESTATUS[0]}

echo "End time: $(date)" | tee -a "$LOGFILE"
echo "Exit status: $status" | tee -a "$LOGFILE"

exit $status