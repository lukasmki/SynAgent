#!/bin/bash
# Stage datasets and the base model. CPU ONLY -- no GPU is requested.
#
# The first attempt did this inside the GPU setup job, which left an A40 idle
# for 75 minutes doing network I/O while the condo is capped at 2 nodes. None
# of this work touches a GPU, so it belongs on the CPU condo.
#SBATCH -J cadd-stage
#SBATCH -N 1
#SBATCH -p csd_lr6_192
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH -o /global/home/users/%u/lawrencium/stage_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/stage_%j.out

set -uo pipefail
SCRATCH=/global/scratch/users/$USER

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TMPDIR=/tmp

source /usr/share/lmod/lmod/init/bash
module purge
module load python/3.11.6-gcc-11.4.0      # never pipe this -- it is a shell function
source "$SCRATCH/envs/cadd/bin/activate"

echo "=== node $(hostname) | $(date) ==="
python -c "import sys;print('python',sys.version.split()[0])"

cd ~/lawrencium

echo
echo "=== 1M-row training set ==="
time python 01_prepare_data.py --rows 1000000 --out "$SCRATCH/cadd/train_1M.jsonl"

echo
echo "=== base model (8B, ~16 GB) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model"],
)
print("model at", p)
PY

echo
echo "=== final state ==="
ls -la "$SCRATCH/cadd/"
du -sh "$SCRATCH/hf_cache"
echo "=== STAGE DONE $(date) ==="
