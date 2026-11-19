#!/bin/bash
#SBATCH --job-name=embed
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=jobs/logs/%x_%j.out
# Stage 1. Finished slides are skipped, so resubmitting resumes a killed job.
#   mkdir -p jobs/logs && sbatch jobs/embed.slurm
# Adapt the module/conda lines to your cluster.
module load cuda/12.8.1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wsisurv
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HUB_OFFLINE=1   # weights downloaded once beforehand (compute nodes often have no internet)
python scripts/embed_slides.py configs/embed.yaml
