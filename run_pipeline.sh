#!/bin/bash
set -euo pipefail

#SBATCH --job-name=kg_repair
#SBATCH --partition=u22
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-UNSET}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-UNSET}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-UNSET}"
echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-UNSET}"
scontrol show job -d "${SLURM_JOB_ID}" | egrep -i "ReqTRES|AllocTRES|MinMemory"

if [ -f /sys/fs/cgroup/memory.max ]; then
  echo "CGROUP_MEMORY_MAX=$(cat /sys/fs/cgroup/memory.max)"
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  echo "CGROUP_MEMORY_MAX=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source /home2/mehulag022/venv2/bin/activate || true

python -u process_html_pipeline.py \
  --max-files 20 \
  --no-dedup \
  --batch-size 64 \
  --push-to-hf \
  --start-index 39 \
  --hf-repo-id ViratGarg/pile_data \
  --hf-path-prefix processed \
  --no-content-classification \
