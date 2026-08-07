# Running SynAgent against local quantized models

Runs SmileyLlama and SynLlama locally as GGUF quants, so the agent's tool calls
work with no vLLM server. Verified working: 7/8 end-to-end pipeline runs.

## 1. Get the models

| Model | Repo | File | Size |
|---|---|---|---|
| SmileyLlama 1B | `lukaskim/SmileyLlama-1B-GGUF` | `SmileyLlama-1B-Q6_K.gguf` | 1.24 GB |
| SynLlama 1B | `lukaskim/SynLlama-1B-GGUF` | `SynLlama-1B-Q6_K.gguf` | 1.24 GB |

Both are GGUF quants of 1B community reproductions (`kysun63/*-reproduced`,
fine-tuned from Llama-3.2-1B-Instruct) — **not** the official THGLab 8B
SmileyLlama. Q2_K / Q4_K_M / Q6_K / Q8_0 / f16 are all published; Q6_K is a good
default at this size, where aggressive quantization costs noticeably more.

```bash
huggingface-cli download lukaskim/SmileyLlama-1B-GGUF SmileyLlama-1B-Q6_K.gguf --local-dir ~/models
huggingface-cli download lukaskim/SynLlama-1B-GGUF   SynLlama-1B-Q6_K.gguf   --local-dir ~/models
```

## 2. Serve them — llama.cpp, not Ollama

**Ollama will not work here.** SynAgent builds the raw prompt itself and posts it
to the legacy `/v1/completions` endpoint. Ollama re-applies a chat template to
that endpoint ([ollama#12636](https://github.com/ollama/ollama/issues/12636),
open), so the prompt gets templated twice and output degrades badly.
`llama-server` passes `/v1/completions` prompts through verbatim — templating
there is scoped to `/v1/chat/completions`.

Grab a build from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases)
(`win-vulkan-x64` gives GPU acceleration on NVIDIA without the CUDA runtime;
`win-cpu-x64` is an 18 MB fallback), then start it in **router mode** — one
process serving both models on one port, selected by the `model=` field:

```bash
llama-server --models-dir ~/models --models-max 2 --models-autoload \
             --port 8080 --host 127.0.0.1 -ngl 99
```

Confirm both registered:

```bash
curl http://127.0.0.1:8080/v1/models
```

You should see `SmileyLlama-1B-Q6_K` and `SynLlama-1B-Q6_K` — the ids come from
the filenames, and those are what `model=` must match.

## 3. Point SynAgent at it

```bash
export VLLM_BASE_URL=http://127.0.0.1:8080/v1   # name is historical; no vLLM involved
export SMILEYLLAMA_MODEL=SmileyLlama-1B-Q6_K
export SYNLLAMA_MODEL=SynLlama-1B-Q6_K
```

## 4. Prompt formats — the part that bites

Both 1B checkpoints are trained on **alpaca** format, not the Llama-3.1 chat
template:

```text
### Instruction:
You love and excel at generating SMILES strings of drug-like molecules

### Input:
Output a SMILES string for a drug like molecule with the following properties: <= 500 Molecular weight, <= 5 logP:

### Response:
```

Two details that are easy to get wrong and both cost output quality:

- **Comparison comes first** — `<= 500 Molecular weight`, not
  `molecular weight <= 500`. The reverse is off-distribution.
- **The trailing colon** after the property list is part of the trained format.

The official 8B `THGLab/Llama-3.1-8B-SmileyLlama-1.1` *does* use the Llama-3.1
chat template. If you swap the 8B in, `generate_molecules` needs its prompt
builder switched back.

SynLlama uses the same alpaca layout with its own long instruction block, and
answers with JSON carrying `<rxn>` templates and `<bb>` building blocks.

## 5. Token budget

`retrosynthesis` uses `max_tokens=1024`. It was 256, which truncates a route
mid-JSON so every response lands in the `parse_error` branch — a two-step
aspirin route already runs past 500 tokens. Complex targets can still overflow
1024; that was the one failure in 8 runs.

## 6. Measured behaviour

On an RTX 3060 Laptop (6 GB, full offload), 8 pipeline runs:

| Metric | Result |
|---|---|
| End-to-end success | 7/8 (88%) |
| SmileyLlama valid SMILES | 8/8 |
| Property constraints satisfied | 20/21 (95%) |
| Route reactants parseable | 21/22 |
| Median wall clock | 14.0 s (SmileyLlama 5.4 s, SynLlama 7.8 s) |

The one failure was SynLlama JSON that didn't parse, on the largest molecule
generated (MW 437.6, five rings).

**What this does and doesn't show.** It shows the agent's tool layer drives both
models end to end and gets structured, parseable chemistry back. It does **not**
show the routes are chemically sound — nothing here checks them against a
building-block catalogue or a chemist's judgement.
