#!/bin/bash
set -euo pipefail

#SBATCH --job-name=kg_repair
#SBATCH --partition=u22
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --nodelist=gnode047

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

python hf_tokenisation.py \
    --repo allenai/paloma \
    --config c4_100_domains \
    --split val \
    --output /ssd_scratch/virat/paloma_c4_100_val.bin
  # --start-file 59 \