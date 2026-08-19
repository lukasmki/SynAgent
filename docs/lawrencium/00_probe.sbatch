#!/bin/bash
# 10-minute probe: confirm the sbatch header is valid for this account and
# that a compute node has what the training scripts assume. Cheap insurance
# before spending an allocation on the environment build.
#SBATCH -J probe
#SBATCH -N 1
#SBATCH -p es1
#SBATCH -A lr_ninjaone
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --gres=gpu:A40:1
#SBATCH --ntasks-per-node 16
#SBATCH --time=00:10:00
#SBATCH -o /global/home/users/%u/lawrencium/probe_%j.out
#SBATCH -e /global/home/users/%u/lawrencium/probe_%j.out

echo "=== NODE: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo
echo "=== CPUS: $(nproc) | MEM: $(free -g | awk '/Mem:/{print $2}') GB ==="
echo
echo "=== FILESYSTEM ==="
echo "scratch:"; df -h /global/scratch 2>/dev/null | tail -1
echo "home:";    df -h "$HOME" 2>/dev/null | tail -1
echo "tmp:";     df -h /tmp | tail -1
touch "/global/scratch/users/$USER/.probe" 2>/dev/null \
  && echo "scratch WRITABLE" && rm -f "/global/scratch/users/$USER/.probe" \
  || echo "scratch NOT writable"
echo
echo "=== MODULE / TORCH ==="
module purge 2>/dev/null
module load ml/pytorch/2.3.1-py3.11.7-mf 2>&1 | head -3
python - <<'PY'
import torch, sys
print("python", sys.version.split()[0])
print("torch", torch.__version__, "| cuda", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.1f} GB")
PY
echo
echo "=== NETWORK: can a compute node reach HuggingFace? ==="
# This decides whether models/datasets can be pulled from inside a job or
# have to be staged in advance from the login node.
curl -s -o /dev/null -w "huggingface.co -> HTTP %{http_code}\n" --max-time 20 https://huggingface.co || echo "NO OUTBOUND HTTPS"
echo
echo "=== DONE ==="
