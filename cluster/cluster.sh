#!/bin/bash
#SBATCH --job-name=SYKRE
#SBATCH --array=1-36%3
#SBATCH --output=logs/job_%A_%a.log
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G

mkdir -p logs

# Read the Nth line from parameter file
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" parameters.txt)

echo "================================================"
echo "Job Array ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Parameters: $PARAMS"
echo "Running on host: $(hostname)"
echo "================================================"

julia --project=@. Cluster.jl $PARAMS 2>&1 | tee logs/julia_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
