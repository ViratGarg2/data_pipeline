#!/bin/bash
# Launch one arm of the filtered-vs-unfiltered comparison with the assignment's
# training setup: global batch 256 sequences x 512 tokens x 100k steps.
# Per-device micro-batch and gradient accumulation are derived from the visible
# GPUs so the global batch stays the same on any hardware.
#
#   scripts/launch_matched_run.sh filtered            # or: unfiltered
#   MICRO_BATCH=16 scripts/launch_matched_run.sh unfiltered ++training.train_steps=200
# (use ++ for overrides: the experiment configs only list the keys they change)
set -euo pipefail

ARM=${1:?usage: launch_matched_run.sh filtered|unfiltered [hydra overrides...]}
shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
[ -f "$ROOT/.venv/bin/activate" ] && source "$ROOT/.venv/bin/activate"
export A4_DATA_DIR=${A4_DATA_DIR:-$ROOT/data}
export A4_OUT_DIR=${A4_OUT_DIR:-$ROOT/runs}
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$ROOT/cs336-basics${PYTHONPATH:+:$PYTHONPATH}"
GLOBAL_BATCH=256
MICRO_BATCH=${MICRO_BATCH:-32}

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
if (( GLOBAL_BATCH % (MICRO_BATCH * NGPU) != 0 )); then
  echo "global batch $GLOBAL_BATCH is not divisible by micro-batch $MICRO_BATCH x $NGPU GPUs" >&2; exit 1
fi
ACCUM=$(( GLOBAL_BATCH / (MICRO_BATCH * NGPU) ))
# bf16 needs Ampere or newer; older GPUs fall back to fp16 with loss scaling.
DTYPE=$(python -c "import torch; print('bfloat16' if torch.cuda.is_bf16_supported() else 'float16')")

mkdir -p "$A4_OUT_DIR"
cd "$ROOT/cs336-basics"
echo "arm=$ARM gpus=$NGPU micro_batch=$MICRO_BATCH grad_accum=$ACCUM dtype=$DTYPE out=$A4_OUT_DIR/${ARM}_13B"
ARGS=(--config-name="experiment/${ARM}_13B"
      ++training.train_batch_size="$MICRO_BATCH"
      ++training.gradient_accumulation_steps="$ACCUM"
      ++training.dtype="$DTYPE" "$@")
if (( NGPU > 1 )); then
  exec torchrun --standalone --nproc_per_node="$NGPU" scripts/train.py "${ARGS[@]}"
else
  exec python scripts/train.py "${ARGS[@]}"
fi
