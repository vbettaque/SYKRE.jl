#!/bin/bash
#SBATCH --job-name=SYKRE
#SBATCH --array=1-29
#SBATCH --output=logs/job_%A_%a.txt
#SBATCH --error=logs/job_%A_%a.err
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

julia --project=@. Cluster.jl $PARAMS