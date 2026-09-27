#!/bin/bash
set -euo pipefail
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=kg_repair
#SBATCH --partition=u22
#SBATCH --mem=48G
#SBATCH --ntasks=1
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

source /home2/mehulag022/venv/bin/activate || true

python -u dedup_jsonl_folder_to_hf.py \
  --max-files 20 \
  --max-rows 1000000 \
  --hf-path-prefix processed_full_comp \
  --hf-repo-id ViratGarg/deduce_pile \
  --input_folder "/data3/dataset/eleuther-ai-the-pile-v-1/pile/train" \
  --workers "${SLURM_CPUS_PER_TASK:-1}" \


