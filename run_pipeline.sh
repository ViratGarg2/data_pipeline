#!/bin/bash
# #SBATCH directives must come before the first command or sbatch ignores them.
#SBATCH --job-name=a4_pipeline
#SBATCH --partition=u22
#SBATCH --mem=48G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home2/mehulag022/assignment4-data/slurm-%j.out
set -euo pipefail

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

source /home2/mehulag022/venv/bin/activate
cd /home2/mehulag022/assignment4-data

echo "After activation:"
which python
python --version

python -c "import sys; print(sys.executable)"
python -c "import site; print(site.getsitepackages())"
python -c "import psutil; print(psutil.__file__)"


python -u cs336_data/process_html_pipeline.py \
  --max-files 30 \
  --no-dedup \
  --batch-size 64 \
  --push-to-hf \
  --start-index 180 \
  --hf-repo-id ViratGarg/pile_data \
  --hf-path-prefix processed \
  --no-content-classification \
  --workers "${SLURM_CPUS_PER_TASK:-4}"
