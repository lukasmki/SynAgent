#!/bin/bash
# Build the training environment AND stage the data, on a compute node.
#
# Combined into one job deliberately: queue time dominates, and compute nodes
# have outbound HTTPS (verified: huggingface.co -> 200), so the dataset can be
# pulled here rather than staged from the login node.
#SBATCH -J cadd-setup
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH -o /global/home/users/%u/lawrencium/setup_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/setup_%j.out

set -uo pipefail

SCRATCH=/global/scratch/users/$USER
ENV_DIR=$SCRATCH/envs/cadd
mkdir -p "$SCRATCH"/{envs,cadd,runs/logs,hf_cache,pipcache}

# Keep everything off the 30 GB home quota.
export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export PIP_CACHE_DIR=$SCRATCH/pipcache
export TMPDIR=/tmp                      # compute nodes have ~193 GB here

echo "=== node $(hostname) | $(date) ==="

# Batch scripts do not source the module system -- this was the failure in the
# first probe ("python: command not found" after a successful module load).
source /usr/share/lmod/lmod/init/bash
module purge
# Do NOT pipe `module load`. It is a shell function, so piping runs it in a
# subshell and the PATH/env changes are discarded -- the first attempt did
# `module load ... | head -2` and silently kept system python 3.6.8, which
# then could not install torch at all.
module load python/3.11.6-gcc-11.4.0
echo "python: $(which python3) $(python3 --version 2>&1)"

PYVER=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYVER" != "3.11" ]; then
    echo "FATAL: expected python 3.11, got $PYVER -- module load did not take"
    exit 1
fi

# Fresh venv WITHOUT --system-site-packages. The ml/pytorch module pins
# torch 2.3.1 + CUDA 11.8, which modern axolotl will not accept. The driver
# here is 580.x, which supports CUDA 12.x, so a pip-installed cu124 torch is
# both valid and far less constrained.
if [ ! -d "$ENV_DIR" ]; then
    python3 -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools -q

echo
echo "=== installing torch (cu124) ==="
pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || { echo "TORCH INSTALL FAILED"; exit 1; }
python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available())"

echo
echo "=== installing axolotl + deepspeed ==="
pip install -q "axolotl[deepspeed]==0.6.0" || { echo "AXOLOTL INSTALL FAILED"; exit 1; }
pip install -q "huggingface_hub[hf_transfer]" || true

echo
echo "=== versions ==="
python - <<'PY'
import importlib
for m in ["torch","transformers","datasets","accelerate","peft","trl","deepspeed","axolotl"]:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:14s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        print(f"  {m:14s} MISSING ({type(e).__name__})")
PY

echo
echo "=== GPU visible to torch ==="
python -c "import torch;[print(' ',i,torch.cuda.get_device_name(i),f'{torch.cuda.get_device_properties(i).total_memory/1e9:.1f}GB') for i in range(torch.cuda.device_count())]"

echo
echo "=== staging datasets ==="
export HF_HUB_ENABLE_HF_TRANSFER=1
cd ~/lawrencium
python 01_prepare_data.py --rows 2000    --out "$SCRATCH/cadd/smoke.jsonl"    2>&1 | tail -8
echo "---"
python 01_prepare_data.py --rows 1000000 --out "$SCRATCH/cadd/train_1M.jsonl" 2>&1 | tail -8

echo
echo "=== pre-downloading the base model (so the smoke job doesn't) ==="
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("NousResearch/Meta-Llama-3.1-8B-Instruct",
                      allow_patterns=["*.json","*.safetensors","tokenizer*","*.model"])
print("model at", p)
PY

echo
echo "=== final state ==="
ls -la "$SCRATCH/cadd/" 2>/dev/null
du -sh "$SCRATCH/hf_cache" 2>/dev/null
echo "=== SETUP DONE $(date) ==="
