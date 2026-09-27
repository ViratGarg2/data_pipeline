#!/bin/bash
# Run new Pile shards through the pipeline, then build token-matched filtered /
# unfiltered data for them and upload both to HF (see prepare_matched_data.py).
#SBATCH --job-name=a4_matched_new
#SBATCH --partition=u22
#SBATCH --array=0-7
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/logs/matched_new_%A_%a.out
set -euo pipefail

# start:count per array task: the gaps in pile_data/processed (72-79, 115), then 210-269.
TASKS=(72:8 115:1 210:10 220:10 230:10 240:10 250:10 260:10)
IFS=: read -r START COUNT <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
END=$((START + COUNT - 1))

FILTERED_REPO=ViratGarg/pile_filtered_13B
UNFILTERED_REPO=ViratGarg/pile_unfiltered_13B
STAGE=/ssd_scratch/mehulag022/a4_matched/new_${SLURM_ARRAY_TASK_ID}

source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data
export OMP_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1
rm -rf "$STAGE" && mkdir -p "$STAGE" metrics
trap 'rm -rf "$STAGE"' EXIT
echo "task ${SLURM_ARRAY_TASK_ID}: shards ${START}-${END} on $(hostname)"

# Same filter settings as the 180-209 run (run_pipeline.sh), plus a metrics report.
python -u cs336_data/process_html_pipeline.py \
  --start-index "$START" --max-files "$COUNT" \
  --no-dedup --batch-size 64 --no-content-classification \
  --workers "$SLURM_CPUS_PER_TASK" \
  --output-dir "$STAGE/pipeline_out" \
  --metrics-json "metrics/pipeline_shards_${START}-${END}.json"

python -u cs336_data/prepare_matched_data.py \
  --shards "${START}-${END}" --filtered-dir "$STAGE/pipeline_out" --stage-dir "$STAGE/stage" \
  --filtered-repo "$FILTERED_REPO" --unfiltered-repo "$UNFILTERED_REPO"
