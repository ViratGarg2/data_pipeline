#!/bin/bash
#SBATCH --job-name=cs336-train
#SBATCH --output=/home2/mehulag022/logs/train_%j.out
#SBATCH --error=/home2/mehulag022/logs/train_%j.err
#SBATCH --nodelist=gnode047
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# ── Setup ──────────────────────────────────────────────────
echo "=============================="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPU IDs:   $CUDA_VISIBLE_DEVICES"
echo "Num GPUs:  $(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)"
echo "Start:     $(date)"
echo "=============================="

# Activate virtual environment
source /home2/mehulag022/venv2/bin/activate

# Working directory
cd /home2/mehulag022/assignment4-data/cs336-basics

# Show GPU info
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ── Train ──────────────────────────────────────────────────
python scripts/train.py --config-name=experiment/your_data

echo "=============================="
echo "End:       $(date)"
echo "=============================="
