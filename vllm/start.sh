#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@90
# Usage: ./start.sh [server ...] [--dry-run]           (on a GPU node, e.g. under salloc)
#        sbatch [sbatch options] start.sh [server ...]  (e.g. sbatch -q debug -t 30 start.sh)
# Starts the servers in config.yaml (all of them by default)

# sbatch runs a copy of this script from the spool dir; Slurm records the original path
if [[ -n $SLURM_JOB_ID && ! -f $(dirname "${BASH_SOURCE[0]}")/launch.py ]]; then
    script=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')
else
    script=${BASH_SOURCE[0]}
fi
cd "$(dirname "$(realpath "$script")")" || exit 1

source .venv/bin/activate

export OMP_NUM_THREADS=8
export HF_HOME=$PWD/.cache/huggingface
export FLASHINFER_WORKSPACE_BASE=$PWD/.cache/flashinfer
export VLLM_CACHE_ROOT=$PWD/.cache/vllm
export TRITON_CACHE_DIR=$PWD/.cache/triton
export SYNAGENT_LOG_DIR=$PWD/logs

exec python launch.py config.yaml "$@"
