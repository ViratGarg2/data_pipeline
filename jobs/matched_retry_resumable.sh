#!/bin/bash
# Rebuild and upload shards with an upload that survives interruption: the stage
# dir is only removed after the uploads succeed, so resubmitting with
# --nodelist=<same node> resumes instead of rebuilding. Each upload attempt runs
# under a timeout and is retried, because on a slow link some upload connections
# hang forever; upload_large_folder keeps its progress in the folder, so every
# retry continues where the last one stopped.
#   sbatch --export=ALL,SHARDS=40-49,SIDES=unfiltered jobs/matched_retry_resumable.sh
#SBATCH --job-name=a4_matched_resume
#SBATCH --partition=u22
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/logs/matched_resume_%j.out
set -euo pipefail

SHARDS=${SHARDS:-136-155}
SIDES=${SIDES:-filtered,unfiltered}
STAGE=/ssd_scratch/mehulag022/a4_matched/resume_${SHARDS}
source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data
# Xet uploads hang from Ada; the classic LFS path works at full link speed.
export OMP_NUM_THREADS=1 HF_HUB_DISABLE_PROGRESS_BARS=1 HF_HUB_DISABLE_XET=1
echo "shards ${SHARDS} on $(hostname), stage ${STAGE}"

if [ ! -f "$STAGE/.built" ]; then
  rm -rf "$STAGE" && mkdir -p "$STAGE"
  python -u cs336_data/prepare_matched_data.py \
    --shards "$SHARDS" --filtered-hf-repo ViratGarg/pile_data --filtered-hf-prefix processed \
    --stage-dir "$STAGE/stage" --no-upload \
    --filtered-repo ViratGarg/pile_filtered_13B --unfiltered-repo ViratGarg/pile_unfiltered_13B
  touch "$STAGE/.built"
else
  echo "found built shards in $STAGE, resuming upload"
fi

for SIDE in ${SIDES//,/ }; do
  REPO=ViratGarg/pile_${SIDE}_13B
  for ATTEMPT in $(seq 1 20); do
    echo "Uploading ${SIDE} -> ${REPO} (attempt ${ATTEMPT})"
    if timeout 30m python -u -c "import sys; sys.path.insert(0, '.'); from cs336_data.prepare_matched_data import upload; upload('${REPO}', '${STAGE}/stage/${SIDE}')"; then
      break
    fi
    [ "$ATTEMPT" -eq 20 ] && { echo "upload of ${SIDE} failed 20 times; stage kept at ${STAGE}"; exit 1; }
    sleep 30
  done
done

rm -rf "$STAGE"
echo "done"
