#!/bin/bash
# One-time environment setup for fine-tuning on Lawrencium.
# Run this ON A COMPUTE NODE, not the login node -- building flash-attn and
# friends will get you killed by the login-node resource limits.
#
#   srun -p es1 -A lr_ninjaone -q condo_ninjaone_es1 --gres=gpu:A40:1 \
#        --ntasks=1 --cpus-per-task=16 --time=2:00:00 --pty bash
#   bash 00_setup_env.sh
#
set -euo pipefail

SCRATCH=/global/scratch/users/$USER
ENV_DIR=$SCRATCH/envs/cadd
mkdir -p "$SCRATCH"/{envs,cadd,models,runs,hf_cache}

# Keep HuggingFace off the 30 GB home quota -- an 8B checkpoint plus the
# dataset will blow through it immediately.
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache

module purge
module load ml/pytorch/2.3.1     # brings CUDA with it
module list

python -m venv "$ENV_DIR" --system-site-packages   # reuse the module's torch
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel

# Pinned rather than latest: axolotl moves fast and its config schema changes
# between releases, which is exactly the kind of surprise you do not want
# halfway through a multi-hour allocation.
pip install "axolotl[deepspeed]==0.6.0"
pip install "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" \
            "peft>=0.12" "trl>=0.9" "bitsandbytes>=0.43" "sentencepiece" "protobuf"

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB")
PY

echo
echo "Environment ready: $ENV_DIR"
echo "Activate later with:"
echo "  module load ml/pytorch/2.3.1 && source $ENV_DIR/bin/activate"
