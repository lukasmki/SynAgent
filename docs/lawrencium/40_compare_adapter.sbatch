#!/bin/bash
#SBATCH --job-name=cadd-eval-qlora
#SBATCH --account=lr_ninjaone
#SBATCH --partition=es1
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A40:1
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
source /usr/share/lmod/lmod/init/bash
source /global/scratch/users/$USER/envs/cadd/bin/activate

export HF_HOME=/global/scratch/users/$USER/hf_cache
export TOKENIZERS_PARALLELISM=false

SCRIPT=$HOME/lawrencium/40_compare_adapter.py
DATA=/global/scratch/users/$USER/cadd/train_1M.jsonl
ADAPTER=/global/scratch/users/$USER/runs/qlora_1M_20260827
OUTPUT=/global/scratch/users/$USER/runs/qlora_1M_20260827/evaluation

nvidia-smi
python "$SCRIPT" \
  --adapter "$ADAPTER" \
  --data "$DATA" \
  --output "$OUTPUT" \
  --sample-size 25 \
  --seed 20260831 \
  --max-new-tokens 256

echo "Evaluation artifacts: $OUTPUT"
