# Checking and evaluating the 1M-row QLoRA model

**Status:** training completed successfully on August 31, 2026.<br>
**SLURM result:** `COMPLETED`, exit code `0:0`.<br>
**Adapter:** `/global/scratch/users/$USER/runs/qlora_1M_20260827`.

## What finished

The production pilot completed one epoch over 1,000,000 training rows using
four NVIDIA A40 GPUs. All 12,090 optimizer steps finished. The final reported
training loss was 0.06389, and held-out evaluation loss fell monotonically from
0.10089 at step 500 to 0.04937 at step 12,000. The run saved a 168 MB LoRA
adapter and retained checkpoints 11,500, 12,000, and 12,090.

This proves the training pipeline ran correctly and optimized held-out loss. It
does not by itself prove that molecular generations or synthesis plans are
chemically better. That requires output-level comparison against the untouched
base model and chemistry-specific evaluation.

## Fast status check

Log in to Lawrencium, then run:

```bash
sacct -j 25306635 --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -X
tail -40 ~/lawrencium/qlora_1m_25306635.out
du -sh /global/scratch/users/$USER/runs/qlora_1M_20260827
ls -lh /global/scratch/users/$USER/runs/qlora_1M_20260827
```

Expected evidence:

```text
State: COMPLETED
ExitCode: 0:0
Steps: 12090/12090
Training Completed!!!
adapter_model.bin: approximately 168 MB
```

## Run a reproducible base-versus-adapter comparison

The branch includes `40_compare_adapter.py` and `40_compare_adapter.sbatch`.
The job uses one A40 and loads the base and fine-tuned models sequentially. It
generates deterministic outputs for the same 25 records and writes both raw
JSONL and a readable Markdown preview.

From the local repository, transfer the scripts without relying on the login
node's full `/tmp`:

```bash
tar czf - docs/lawrencium/40_compare_adapter.py \
  docs/lawrencium/40_compare_adapter.sbatch | \
  ssh lrc 'tar xzf - --strip-components=2 -C ~/lawrencium'
```

On Lawrencium:

```bash
cd ~/lawrencium
sbatch 40_compare_adapter.sbatch
squeue -u $USER
```

After completion:

```bash
cat /global/scratch/users/$USER/runs/qlora_1M_20260827/evaluation/base_vs_qlora_preview.md
wc -l /global/scratch/users/$USER/runs/qlora_1M_20260827/evaluation/base_vs_qlora.jsonl
```

The comparison uses greedy decoding (`do_sample=False`) and a fixed seed so the
base and adapter receive identical prompts. The sampled file is the staged
training subset, so this is a functional smoke comparison, not an unbiased
test-set estimate. For a paper, replace `--data` with a truly held-out JSONL
that was excluded before training.

## Load the adapter in Python

The saved artifact is a LoRA adapter, not a standalone 8B model. Load the same
base checkpoint first and then attach the adapter:

```python
import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_id = "NousResearch/Meta-Llama-3.1-8B-Instruct"
adapter_path = os.path.expandvars(
    "/global/scratch/users/$USER/runs/qlora_1M_20260827"
)

quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id, quantization_config=quant, device_map="auto"
)
model = PeftModel.from_pretrained(base, adapter_path)
model.eval()
```

Keep model loading on a GPU compute node.

## What to measure next

Use a held-out set with source labels and compare the untouched base model and
QLoRA adapter using identical prompts and decoding:

1. Molecule generation: valid-SMILES rate, property error, uniqueness, and
   scaffold diversity.
2. Retrosynthesis: parse success, exact route validity, permutation-aware
   route validity, and analog-aware route validity.
3. Instruction following: schema compliance and required-field completion.
4. Human review: blinded assessment of a stratified sample by a synthetic
   chemist.
5. Efficiency: generation latency, GPU memory, and failure rate.

Do not choose the best-looking checkpoint using the test set. Pick a checkpoint
using validation loss or a predeclared development metric, then report the test
set once. Checkpoint 12,000 has the lowest recorded evaluation loss among the
regular 500-step checkpoints; checkpoint 12,090 is the final training state.

## Recovery and preservation

Scratch is not backed up. Preserve the adapter, configuration, tokenizer, and
trainer state before relying on them for publication:

```bash
RUN=/global/scratch/users/$USER/runs/qlora_1M_20260827
tar -C "$(dirname "$RUN")" -czf "$HOME/qlora_1M_20260827_adapter.tar.gz" \
  qlora_1M_20260827/adapter_model.bin \
  qlora_1M_20260827/adapter_config.json \
  qlora_1M_20260827/config.json \
  qlora_1M_20260827/tokenizer.json \
  qlora_1M_20260827/tokenizer_config.json
sha256sum "$RUN/adapter_model.bin" "$RUN/adapter_config.json"
```

Recorded SHA-256 checksums:

```text
adapter_model.bin    9e3896eb80eba3397911246aaea1eecbc2480059abcd747b2030de8beaf294d4
adapter_config.json  f6c4dc78a255874d6b5092ba30d41b6f171fb028ec57354cc5db15a60ebab981
```
