#!/bin/bash
#SBATCH --job-name=survival
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=jobs/logs/%x_%j.out
#   sbatch jobs/train.slurm configs/survival_abmil.yaml
module load cuda/12.8.1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wsisurv
cd "$SLURM_SUBMIT_DIR"
python scripts/baseline_cox.py --config configs/baseline_cox.yaml
python scripts/train_survival.py --config "${1:-configs/survival_abmil.yaml}"
