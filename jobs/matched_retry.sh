#!/bin/bash
# Rebuild and re-upload shards whose upload stalled in matched_reuse_shards.sh
# (tasks 2 and 7 hit the time limit). Shards 40-59 already have their filtered
# side on HF, so only the unfiltered side is uploaded for them.
#SBATCH --job-name=a4_matched_retry
#SBATCH --partition=u22
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/logs/matched_retry_%A_%a.out
set -euo pipefail

# shards:sides per array task, 5 shards each so every upload stays small.
TASKS=(40-44:unfiltered 45-49:unfiltered 50-54:unfiltered 55-59:unfiltered
       136-140:filtered,unfiltered 141-145:filtered,unfiltered 146-150:filtered,unfiltered 151-155:filtered,unfiltered)
IFS=: read -r SHARDS SIDES <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"

STAGE=/ssd_scratch/mehulag022/a4_matched/retry_${SLURM_ARRAY_TASK_ID}
source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data
export OMP_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
rm -rf "$STAGE" && mkdir -p "$STAGE"
trap 'rm -rf "$STAGE"' EXIT
echo "task ${SLURM_ARRAY_TASK_ID}: shards ${SHARDS}, upload ${SIDES} on $(hostname)"

python -u cs336_data/prepare_matched_data.py \
  --shards "$SHARDS" --filtered-hf-repo ViratGarg/pile_data --filtered-hf-prefix processed \
  --stage-dir "$STAGE/stage" --upload-sides "$SIDES" \
  --filtered-repo ViratGarg/pile_filtered_13B --unfiltered-repo ViratGarg/pile_unfiltered_13B
