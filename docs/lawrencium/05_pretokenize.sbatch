#!/bin/bash
# Tokenize the 1M-row training set on a CPU node, BEFORE the GPU job runs.
#
# WHY THIS EXISTS
# axolotl tokenizes the dataset on first use and caches it at
# `dataset_prepared_path`. Left to itself, that work happens inside the timing
# job -- 1M rows of tokenization while four A40s sit idle, inside a 4-hour cap.
# Tokenization is pure CPU: no GPU is touched. Doing it here means the GPU job
# starts training on its first minute.
#
# It writes to the SAME dataset_prepared_path the timing job reads, using the
# SAME config file, so axolotl's config hash matches and the cache is a hit.
# If you edit the `datasets:` or `sequence_len` block of timing.yaml, the hash
# changes and this must be re-run.
#
# lr6 + lr_lowprio, because the GPU condo is full and the CPU pool is not.
# lowprio is preemptible; that is fine for a cache-fill -- if it dies, resubmit,
# nothing is lost but time.
#
#SBATCH -J cadd-pretok
#SBATCH -N 1
#SBATCH -p lr6
#SBATCH -A lr_ninjaone
#SBATCH --qos=lr_lowprio
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH -o /global/home/users/%u/lawrencium/pretok_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/pretok_%j.out

set -euo pipefail
SCRATCH=/global/scratch/users/$USER

export HF_HOME=$SCRATCH/hf_cache
export HF_DATASETS_CACHE=$SCRATCH/hf_cache
export TMPDIR=/tmp
# Force CPU. A CPU node has no visible device, but being explicit stops any
# accelerate/torch path from probing for one and erroring during import.
export CUDA_VISIBLE_DEVICES=""
# Tokenizers' own parallelism deadlocks when combined with dataset multiproc.
# Let datasets do the forking; keep the fast tokenizer single-threaded.
export TOKENIZERS_PARALLELISM=false

source /usr/share/lmod/lmod/init/bash
module purge
module load python/3.11.6-gcc-11.4.0
source "$SCRATCH/envs/cadd/bin/activate"

# Trap 3 from the runbook: never pipe `module load`. Assert instead.
python -c 'import sys; assert sys.version_info[:2]==(3,11), sys.version; print("python", sys.version.split()[0])'

echo "host      : $(hostname)"
echo "cpus      : $(nproc)"
echo "start     : $(date)"
echo

cd ~/lawrencium
axolotl preprocess timing.yaml

echo
echo "end       : $(date)"
echo "=== prepared cache ==="
du -sh "$SCRATCH"/cadd/prepared_1M 2>/dev/null || echo "  MISSING -- preprocess did not write the cache"
