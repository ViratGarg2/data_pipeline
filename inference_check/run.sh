#!/bin/bash
set -euo pipefail
source /home2/mehulag022/venv/bin/activate
python -c "import einx, einops, jaxtyping" 2>/dev/null || pip install -q -c /home2/mehulag022/assignment4-data/smoke/constraints.txt einx einops jaxtyping 2>&1 | grep -v rotobuf | tail -2 || true
S=/ssd_scratch/mehulag022/prev_model; mkdir -p $S
echo "node $(hostname), gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python -u /home2/mehulag022/assignment4-data/inference_check/infer_prev.py $S 2>&1 | grep -viE "warn|FutureWarning|torch.load" || true
