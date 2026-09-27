#!/bin/bash
# One pipeline benchmark configuration (see jobs/submit_benchmarks.sh). Output goes
# to node-local scratch and is discarded; only the metrics JSON is kept.
#SBATCH --partition=u22
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/logs/bench_%x_%j.out
set -euo pipefail
: "${NAME:?}" "${WORKERS:?}" "${START:?}" "${NFILES:?}" "${ROWS:?}"
EXTRA=${EXTRA:-}
STAGE=/ssd_scratch/mehulag022/a4_bench/${NAME}_${SLURM_JOB_ID}
source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data
export OMP_NUM_THREADS=1
rm -rf "$STAGE" && mkdir -p "$STAGE" metrics/bench
trap 'rm -rf "$STAGE"' EXIT
echo "bench ${NAME}: workers=${WORKERS} shards=${START}+${NFILES} rows/shard=${ROWS} extra='${EXTRA}' on $(hostname)"
if [ "$NAME" = "profile" ]; then
  python -m cProfile -o "$STAGE/profile.out" cs336_data/process_html_pipeline.py \
    --start-index "$START" --max-files "$NFILES" --max-rows "$ROWS" --workers 1 \
    --batch-size 64 --no-dedup --no-content-classification --output-dir "$STAGE/out" > /dev/null
  python -c "import pstats; pstats.Stats('$STAGE/profile.out').sort_stats('tottime').print_stats(35)" > metrics/bench/profile_tottime.txt
  python -c "import pstats; pstats.Stats('$STAGE/profile.out').sort_stats('cumulative').print_stats(45)" > metrics/bench/profile_cumulative.txt
else
  # shellcheck disable=SC2086
  python -u cs336_data/process_html_pipeline.py \
    --start-index "$START" --max-files "$NFILES" --max-rows "$ROWS" --workers "$WORKERS" \
    --batch-size 64 --output-dir "$STAGE/out" --metrics-json "metrics/bench/${NAME}.json" $EXTRA
fi
