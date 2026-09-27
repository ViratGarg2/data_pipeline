#!/bin/bash
# Build token-matched filtered / unfiltered data for shards already processed
# into ViratGarg/pile_data/processed, and upload both to HF.
# Tasks 0-8 cover shards 0-179; tasks 9-10 cover 180-209 (submit those after
# the 180-209 pipeline run has finished uploading).
#SBATCH --job-name=a4_matched_reuse
#SBATCH --partition=u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/logs/matched_reuse_%A_%a.out
set -euo pipefail

TASKS=(0-19 20-39 40-59 60-71 80-99 100-114 116-135 136-155 156-179 180-194 195-209)
SHARDS=${TASKS[$SLURM_ARRAY_TASK_ID]}

FILTERED_REPO=ViratGarg/pile_filtered_13B
UNFILTERED_REPO=ViratGarg/pile_unfiltered_13B
STAGE=/ssd_scratch/mehulag022/a4_matched/reuse_${SLURM_ARRAY_TASK_ID}

source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data
export OMP_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
rm -rf "$STAGE" && mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "task ${SLURM_ARRAY_TASK_ID}: shards ${SHARDS} on $(hostname)"

python -u cs336_data/prepare_matched_data.py \
  --shards "$SHARDS" --filtered-hf-repo ViratGarg/pile_data --filtered-hf-prefix processed \
  --stage-dir "$STAGE/stage" \
  --filtered-repo "$FILTERED_REPO" --unfiltered-repo "$UNFILTERED_REPO"
