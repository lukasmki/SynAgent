# Fine-tuning on Lawrencium — runbook

Full fine-tune of an 8B model on `cadd-instruct`, timed on a 1M-row subset to
extrapolate the cost of the full dataset.

**Everything below marked ✅ was verified by running it on the cluster**, not
inferred from documentation. The earlier version of this file contained several
guesses; they were wrong and are corrected here.

---

## Verified cluster facts

| | Value | How known |
|---|---|---|
| Login | `lrc-login.lbl.gov`, MFA = PIN+OTP concatenated | ✅ |
| Account | `lr_ninjaone` | ✅ |
| GPU partition | `es1` | ✅ |
| **QoS** | **`condo_ninjaone_es1`** or `es_lowprio` | ✅ `es_normal` does NOT exist for this account |
| Condo cap | **2 nodes** (`QOSGrpNodeLimit` when exceeded) | ✅ |
| Walltime | **unlimited** (`es1` = infinite, condo has no maxwall) | ✅ |
| A40 node | 4 × A40 46 GB, **64 CPUs**, **503 GB RAM** | ✅ |
| Driver | **580.173.02** → supports CUDA 12.x | ✅ |
| Compute-node internet | **YES** — `huggingface.co → 200` | ✅ |
| Compute-node `/tmp` | **193 GB free** | ✅ |
| Login-node `/tmp` | **FULL** — breaks heredocs and `scp` | ✅ |
| Home | `/global/home/users/$USER`, 30 GB quota | ✅ |
| Scratch | `/global/scratch/users/$USER`, no meaningful quota | ✅ |

---

## Four traps, all hit for real

### 1. `es_normal` doesn't exist for this account
Guessed from public docs. Real QoS list comes from:
```bash
sacctmgr show assoc user=$USER format=account,partition,qos%40
```
Jobs with a bad QoS are rejected at submission.

### 2. Batch scripts don't initialize the module system
`module load` in an sbatch script does nothing until you first:
```bash
source /usr/share/lmod/lmod/init/bash
```
Symptom without it: `python: command not found` *after* an apparently
successful `module load`.

### 3. Never pipe `module load`
```bash
module load python/3.11.6-gcc-11.4.0 2>&1 | head -2   # WRONG
module load python/3.11.6-gcc-11.4.0                  # right
```
`module` is a **shell function**. Piping runs it in a subshell, so the PATH and
env changes are discarded. The symptom is badly misleading: the job silently
keeps system Python 3.6.8 and later dies with
`Could not find a version that satisfies the requirement torch==2.5.1`, which
looks like a network or index failure but is a 10-year-old interpreter.

The setup script now asserts the interpreter is 3.11 immediately after loading,
so this fails in seconds rather than 20 minutes later.

### 4. The login node's `/tmp` is full
Full enough that bash **heredocs fail** (`cannot create temp file for
here-document`) and `scp` dies with `Connection closed`. Workarounds in use:
- transfer files by streaming: `tar czf - dir | ssh lrc 'tar xzf - -C ~/'`
- write scripts locally and ship them; don't compose them on the login node

Worth reporting to `hpcshelp@lbl.gov`.

---

## Don't use the `all` dataset config

The Hub reports 9,712,790 rows for `cadd-instruct`. That number is wrong for
training. `all/` holds **two overlapping exports** and the config globs
`all/*.parquet`, loading both:

| File | Rows | Contents |
|---|---|---|
| `all/train-00000-of-00001.parquet` | 3,856,395 | **stale** — smileyllama + linkllama only |
| `all/train-0000{0..8}-of-00009.parquet` | 5,856,395 | current three-way mix |

Verified from the parquet footers and the `source` column: the single file is
2,330,226 smileyllama + 1,526,169 linkllama with **no synllama rows** — it
predates synllama being added.

**Loading `all` gives smileyllama and linkllama twice, synllama once**, shifting
the mix from 40/26/34 to 48/31/21 and under-weighting synthesis planning by a
third. `01_prepare_data.py` loads the three per-source configs explicitly.

The real dataset is **5,856,395 rows**. Lukas's fix is deleting one file.

---

## Non-interactive access (so automation can drive the cluster)

Lawrencium requires MFA on **every** login; an installed public key is not
sufficient (tested). SSH connection multiplexing does **not** work from Git Bash
— MSYS2 lacks the fd-passing session multiplexing needs, so `ssh -O check`
succeeds while opening a session fails.

What works is LBL's short-lived certificate:

```bash
git clone https://github.com/lbnl-science-it/lrc-scripts.git
cd lrc-scripts && ./request_cert.sh -p lrc      # username, PIN, OTP as SEPARATE prompts
```

> **Bug on Windows.** `request_cert.sh` writes the key via `python3 print()`,
> and Python on Windows emits CRLF, which corrupts the PEM. ssh then fails with
> `error in libcrypto: unsupported`. Fix:
> ```bash
> cd ~/.ssh/ssh_certs && for f in lrc_cert*; do tr -d '\r' < "$f" > t && mv t "$f"; done
> chmod 600 lrc_cert
> ```
> Worth reporting to `lbnl-science-it/lrc-scripts`.

The script also needs `jq` and a real `python3` on PATH.

Certificate lasts 12 hours; re-run to renew.

---

## Order of operations

```bash
# ship the scripts (scp is broken on the login node - stream instead)
tar czf - lawrencium | ssh lrc 'tar xzf - -C ~/'

sbatch 00_probe.sbatch     # 10 min  - sanity check node, GPU, network
sbatch 01_setup.sbatch     # ~1 h    - venv, torch, axolotl, datasets, base model
sbatch 10_smoke.sbatch     # ~1 h    - QLoRA, 1 GPU, 50 steps
sbatch 20_timing.sbatch    # ~2 h    - full FT, 4 A40, ZeRO-3, 200 steps + extrapolation

squeue -u $USER
```

Each stage exists to make the next one's failures cheap. The probe cost 2
seconds and caught the module-init bug that would otherwise have surfaced deep
inside a 4-GPU allocation.

---

## Why the training config looks the way it does

**ZeRO-3, not ZeRO-2.** Full FT of 8B needs ~128 GB: 16 GB bf16 params, 16 GB
gradients, and 96 GB for Adam's two moments plus fp32 master weights
(8B × 12 bytes). Against 46 GB cards all three must shard. Across 4 GPUs that is
~32 GB each, leaving ~14 GB for activations.

**CPU optimizer offload is on by default** in `zero3.json`. 32 GB plus
activations is close enough to 46 GB that a long sequence can OOM hours in, and
the node has 503 GB of RAM going spare. It costs throughput — once a run is
stable with headroom, set `offload_optimizer.device` to `"none"` and re-measure.

**`--ntasks=1` with `--cpus-per-task=64`.** A40 nodes have exactly 64 CPUs for
4 GPUs. It must be ONE task: `accelerate` spawns the four ranks itself, so
requesting four tasks launches four independent trainers each assuming it owns
the node. (The first probe used `--ntasks-per-node 16` and got 16 *tasks*, not
16 CPUs.)

**Timing is bounded to 200 steps**, not a full epoch — running an epoch to learn
how long an epoch takes wastes the measurement. The sbatch tail parses
steady-state `s/it`, discards 20 warmup steps, and extrapolates to 1M / 2M /
full 5.86M.

**Base model is the NousResearch mirror.** `meta-llama/Llama-3.1-8B-Instruct` is
gated and 401s inside a batch job without an accepted licence and `HF_TOKEN`.
Same weights, ungated, one less failure mode you cannot click through from a
compute node.

**Not the `ml/pytorch` module.** It pins torch 2.3.1 + CUDA 11.8, which modern
axolotl rejects. The driver supports CUDA 12.x, so a pip-installed cu124 torch
in a clean venv is both valid and far less constrained.

---

## Files

| File | Purpose |
|---|---|
| `00_probe.sbatch` | 10-min sanity check: GPU, filesystem, network, modules |
| `01_setup.sbatch` | venv + torch + axolotl + datasets + base model |
| `01_prepare_data.py` | JSONL subset, avoiding the stale-shard bug |
| `smoke.yaml` / `10_smoke.sbatch` | QLoRA, 1 GPU, 50 steps |
| `timing.yaml` / `20_timing.sbatch` | full FT, 4 × A40, ZeRO-3, 200 steps |
| `zero3.json` | DeepSpeed config, memory reasoning inline |
