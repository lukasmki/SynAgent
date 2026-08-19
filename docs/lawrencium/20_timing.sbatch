#!/bin/bash
# Timing run: FULL fine-tune of 8B on 4 x A40 with DeepSpeed ZeRO-3, bounded to
# a fixed step count, then extrapolated to whole-dataset epoch time.
#SBATCH -J cadd-timing
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --gres=gpu:A40:4
#SBATCH --ntasks=1              # ONE launcher; accelerate spawns the 4 ranks
#SBATCH --cpus-per-task=64      # A40 nodes have exactly 64 CPUs for 4 GPUs
#SBATCH --time=04:00:00
#SBATCH -o /global/home/users/%u/lawrencium/timing_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/timing_%j.out

set -uo pipefail
SCRATCH=/global/scratch/users/$USER

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/tmp
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

source /usr/share/lmod/lmod/init/bash
module purge
module load python/3.11.6-gcc-11.4.0
source "$SCRATCH/envs/cadd/bin/activate"

echo "=== node $(hostname) | job $SLURM_JOB_ID | $(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| gpus',torch.cuda.device_count())"

cd ~/lawrencium
START=$(date +%s)
accelerate launch \
    --num_processes 4 \
    --num_machines 1 \
    --mixed_precision bf16 \
    --use_fsdp \
    -m axolotl.cli.train timing.yaml
RC=$?
END=$(date +%s)
echo "=== training exit $RC | wall $((END - START))s | $(date) ==="

# Extrapolate from steady-state throughput. Warmup steps are discarded because
# the first several are dominated by compile and allocator churn and would
# flatter the estimate.
python - "$SLURM_JOB_ID" <<'PY'
import re, sys, os, glob
jid = sys.argv[1]
log = f"/global/home/users/{os.environ['USER']}/lawrencium/timing_{jid}.out"
txt = open(log, errors="ignore").read()

secs = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*s/it", txt)]
its  = [float(x) for x in re.findall(r"(\d+\.?\d*)\s*it/s", txt)]
if its and not secs:
    secs = [1.0 / i for i in its if i > 0]
secs = secs[20:] or secs          # drop warmup if we have enough samples

if not secs:
    print("No throughput lines found; inspect the log manually.")
    raise SystemExit(0)

s_per_step = sum(secs) / len(secs)
rows_per_step = 1 * 8 * 4         # micro_batch * grad_accum * gpus
rps = rows_per_step / s_per_step
print(f"steady state : {s_per_step:.2f} s/step over {len(secs)} samples")
print(f"throughput   : {rps:.1f} rows/s  ({rps*3600:,.0f} rows/h)")
print()
print("EXTRAPOLATED EPOCH TIME (4 x A40, full FT, ZeRO-3):")
for n, label in [(1_000_000, "1M rows"), (2_000_000, "2M rows"), (5_856_395, "full 5.86M")]:
    h = n / rps / 3600
    print(f"  {label:>12s}: {h:7.1f} h   ({h/24:5.1f} days)")
PY
exit $RC
