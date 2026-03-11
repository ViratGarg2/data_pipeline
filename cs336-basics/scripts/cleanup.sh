#!/bin/bash

# Run only on gnode047
# use command crontab -e to edit cron jobs and add the following line to run this script every day at 2am:
# */5 * * * /home2/mehulag022/assignment4-data/cs336-basics/scripts/cleanup.sh >> /home2/mehulag022/assignment4-data/cs336-basics/scripts/cleanup.log 2>&1
HOST=$(hostname)
if [[ "$HOST" != "gnode047" ]]; then
    echo "Not running on gnode047. Exiting."
    exit 0
fi

TARGET_DIR="/ssd_scratch/virat/model_checkpoints"

cd "$TARGET_DIR" || exit 1

echo "Running cleanup at $(date)"

# Find the latest checkpoint directory
LATEST_STEP=$(ls -d step_* 2>/dev/null | sort -V | tail -n 1)

if [[ -z "$LATEST_STEP" ]]; then
    echo "No step directories found."
    exit 0
fi

echo "Keeping latest checkpoint: $LATEST_STEP"

# Delete all other step_* directories
for dir in step_*; do
    if [[ "$dir" != "$LATEST_STEP" ]]; then
        echo "Deleting $dir"
        rm -rf "$dir"
    fi
done

# Ensure model_config.json exists
if [[ ! -f model_config.json ]]; then
    echo "Warning: model_config.json missing!"
fi

echo "Cleanup complete."