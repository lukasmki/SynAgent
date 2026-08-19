#!/bin/bash
# Smoke test: QLoRA on ONE A40, 50 steps. Proves the whole path works before a
# 4-GPU allocation is spent discovering otherwise.
#SBATCH -J cadd-smoke
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH -o /global/home/users/%u/lawrencium/smoke_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/smoke_%j.out

set -uo pipefail
SCRATCH=/global/scratch/users/$USER

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp

# Batch shells do not initialise Lmod, and `module` must NOT be piped -- it is a
# shell function, so a pipe subshells it and the PATH change is lost.
source /usr/share/lmod/lmod/init/bash
module purge
module load python/3.11.6-gcc-11.4.0
source "$SCRATCH/envs/cadd/bin/activate"

echo "=== node $(hostname) | $(date) ==="
python -c "import torch;print('torch',torch.__version__,'| gpus',torch.cuda.device_count(),'|',torch.cuda.get_device_name(0))"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo
echo "=== training (50 steps, QLoRA) ==="
cd ~/lawrencium
time accelerate launch --num_processes 1 -m axolotl.cli.train smoke.yaml
RC=$?

echo
echo "=== exit $RC | $(date) ==="
ls -la "$SCRATCH/runs/smoke" 2>/dev/null | head
exit $RC
