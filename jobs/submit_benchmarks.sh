#!/bin/bash
# Pipeline benchmarks on shards 300+ (not used for the training data):
#   scale_w*   worker scaling, 16 shards x 20k rows, production filter settings
#   stage_*    one change at a time vs. production settings, 8 workers, 8 shards x 20k rows
#   profile    cProfile of a single worker on 3k rows
set -euo pipefail
cd /home2/mehulag022/assignment4-data
PROD="--no-dedup --no-content-classification"
submit() {  # name workers start nfiles rows extra
  local mem=$(( $2 * 4 + 4 ))
  sbatch --job-name="bench_$1" --cpus-per-task="$2" --mem="${mem}G" \
    --export=ALL,NAME="$1",WORKERS="$2",START="$3",NFILES="$4",ROWS="$5",EXTRA="$6" jobs/benchmark_pipeline.sh
}
for w in 1 2 4 8 16; do submit "scale_w${w}" "$w" 300 16 20000 "$PROD"; done
submit stage_production   8 300 8 20000 "$PROD"
submit stage_no_pii       8 300 8 20000 "$PROD --no-pii"
submit stage_no_langid    8 300 8 20000 "$PROD --no-langid"
submit stage_no_gopher    8 300 8 20000 "$PROD --no-quality-filter"
submit stage_no_qclf      8 300 8 20000 "$PROD --no-quality-classifier"
submit stage_plus_content 8 300 8 20000 "--no-dedup"
submit stage_plus_dedup   8 300 8 20000 "--no-content-classification"
submit profile            1 300 1 3000  ""
