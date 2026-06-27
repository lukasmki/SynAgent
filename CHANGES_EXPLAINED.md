# SynAgent Pipeline Changes — Explained

This document explains every new file and modification introduced in the `SynLlama-SmileyLlama-linkLlama-added-in` branch. It covers what each component does, how the tool calls work, and how everything fits together.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [llm_tools.py — LLM Tool Definitions](#llm_toolspy--llm-tool-definitions)
3. [enaminetool.py — Enamine Database Search](#enaminetoolpy--enamine-database-search)
4. [workflows.py — Composite Enamine + LinkLlama Pipeline](#workflowspy--composite-enamine--linkllama-pipeline)
5. [master.py — Disagreeable Master Agent](#masterpy--disagreeable-master-agent)
6. [tests/ — Mock Test Suite](#tests--mock-test-suite)
7. [pyproject.toml — Dependency Changes](#pyprojecttoml--dependency-changes)
8. [How Tool Calls Work](#how-tool-calls-work)
9. [Environment Variables](#environment-variables)

---

## Pipeline Overview

The full end-to-end pipeline looks like this:

```
User prompt
    │
    ▼
┌─────────────────────────────┐
│  Master Agent (Gemini)      │  ← "Disagreeable advisor" that challenges
│  Decides which tools to use │    the user's choices before executing
└─────────┬───────────────────┘
          │
          ├──► SmileyLlama (8B) ──► Generates novel drug-like molecules
          │                          with property constraints (MW, LogP, etc.)
          │
          ├──► SynLlama (1B) ──► Decomposes molecules into retrosynthetic
          │                       pathways (reaction templates + building blocks)
          │
          ├──► Validation Agent ──► RDKit checks on SMILES, SMARTS, reactions
          │
          ├──► Enamine Search ──► Finds purchasable similar molecules
          │         │
          │         ▼
          ├──► LinkLlama (1B) ──► Designs linker molecules between fragments
          │
          ├──► ChemSpace Agent ──► Building block pricing & availability
          │
          └──► Optimization Agent ──► Route hazard/safety scoring
```

---

## llm_tools.py — LLM Tool Definitions

**File:** `src/synagent/llm_tools.py`

**What it does:** Defines tool call functions for three fine-tuned LLMs. All three are served behind a single vLLM server that exposes an OpenAI-compatible `/v1/completions` endpoint. The file uses the `openai` Python client to send requests, selecting which model to use via the `model` parameter on each request.

### SmileyLlama — Molecule Generation

- **Model:** `THGLab/Llama-3.1-8B-SmileyLlama-1.1` (8 billion parameters)
- **Training data:** ~2 million SMILES strings from the ChEMBL database
- **What it does:** Given pharmaceutical property constraints, it generates valid SMILES strings of novel drug-like molecules
- **Input (Pydantic model `SmileyLlamaInput`):**
  - `mw_range` — Molecular weight (e.g., `"<= 500"`)
  - `logp_range` — Lipophilicity (e.g., `"<= 5"`)
  - `hbd_range` — Hydrogen-bond donors (e.g., `"<= 5"`)
  - `hba_range` — Hydrogen-bond acceptors (e.g., `"<= 10"`)
  - `rotatable_bonds` — (e.g., `"<= 10"`)
  - `fsp3` — Fraction of sp3 carbons (e.g., `">= 0.25"`)
  - `macrocycle` — Boolean, whether to include macrocycles
  - `num_samples` — How many molecules to generate (1–50)
  - `temperature`, `top_p` — Sampling parameters
- **Output:** List of `{"smiles": "CCO", "raw_output": "CCO"}`
- **Prompt format:** Uses the Llama-3.1 chat template with special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`)
- **How `_build_smileyllama_prompt` works:** Collects all non-null property constraints into a comma-separated string. If no constraints, uses a generic "generate a drug-like molecule" prompt. This matches the training format from the SmileyLlama paper.

### SynLlama — Retrosynthetic Pathway Prediction

- **Model:** `SynLlama-1B` (1 billion parameters)
- **Training data:** 2 million reactions across 91 reaction templates
- **What it does:** Takes a target molecule (SMILES) and decomposes it backwards into purchasable building blocks, producing a step-by-step synthesis plan
- **Input (Pydantic model `SynLlamaInput`):**
  - `product_smiles` — The target molecule to synthesize
  - `num_pathways` — Number of diverse routes to generate (1–20)
  - `temperature` — Default 1.5 (high for diversity sampling)
  - `top_p` — Default 0.9 (nucleus sampling for diversity)
- **Output:** `{"product": "CCCO", "pathways": [{"pathway": {...}, "parse_error": false}]}`
- **Prompt format:** Uses `### Instruction:` / `### Input:` / `### Response:` sections. The instruction tells SynLlama to use `<rxn>` tags for reaction templates and `<bb>` tags for building blocks.
- **JSON parsing:** SynLlama outputs JSON but sometimes wraps it in markdown code fences (` ```json `). The code strips those before parsing. If parsing fails entirely, the raw text is preserved with `parse_error: true` so nothing is lost.
- **Why high temperature?** Each call with T=1.5 produces a *different* retrosynthetic disconnection. By generating multiple pathways, downstream validation can pick the best one.

### LinkLlama — Fragment Linker Design

- **Model:** `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` (1 billion parameters)
- **Training data:** Instruction-style JSONL derived from ChEMBL linker data
- **What it does:** Given two molecular fragments with geometric constraints (distance in Ångströms, angle in degrees), it proposes linker molecules to connect them
- **Input (Pydantic model `LinkLlamaInput`):**
  - `fragment1_smiles`, `fragment2_smiles` — Fragments with `[*]` dummy atoms at attachment points
  - `distance_angstrom` — Distance between attachment points
  - `angle_degrees` — Angle between attachment points
  - Linker property constraints: `linker_type` (chain/branched/ring-containing), `rotb_range`, `heavy_atoms_range`
  - Molecule property constraints: `hbd_range`, `hba_range`, `mw_range`, `logp_range`, `tpsa_range`
  - `reasonability` — `"reasonable"` or `"unreasonable"`
  - `num_samples` — 1–100 linker proposals
- **Output:** `{"fragments": [...], "geometry": {...}, "samples": [{"linker": "CCC", "reasoning": "...", "parse_error": false}]}`
- **How `_build_linkllama_prompt` works:** Assembles fragment info, then optionally adds linker property and molecule property sections, ending with reasonability. Follows LinkLlama's `sft_corpus.py` training format.

### Tool Registration — `register_llm_tools(agent)`

This function takes any `pydantic_ai.Agent` and registers all three LLMs as `@agent.tool_plain` decorated functions. "tool_plain" means no dependency injection (no `RunContext`) — the vLLM client is a module-level singleton. The registered tool names are:
- `generate_molecules` (SmileyLlama)
- `retrosynthesis` (SynLlama)
- `design_linker` (LinkLlama)

These are the names the master agent's LLM sees and can decide to call.

---

## enaminetool.py — Enamine Database Search

**File:** `src/synagent/enaminetool.py`

**What it does:** Searches the Enamine REAL database for commercially available molecules similar to (or containing) a query molecule. This is critical for the pipeline because even if SmileyLlama generates a great molecule, you need to find *purchasable* fragments to actually synthesize it.

### Two-tier search strategy

1. **Enamine REST API** (primary) — If `ENAMINE_API_KEY` is set, sends a POST request to `https://api.enamine.net/api/v1/search/similarity` (or `/substructure`). Returns results with Enamine catalog IDs, Tanimoto similarity scores, availability status, and pricing.

2. **Local RDKit fallback** — If the API key is missing or the API call fails (network error, 401, etc.), falls back to computing Morgan fingerprint similarity locally against a cached CSV file at `data/enamine_fragments.csv`. Uses RDKit's `GetMorganFingerprintAsBitVect` with radius=2 and 2048 bits (standard for drug-like molecules), then Tanimoto similarity scoring.

### Modeled after chemspacetool.py

The Enamine tool follows the same pattern as the existing ChemSpace tool:
- `httpx.AsyncClient` for async HTTP requests
- Bearer token authentication
- Pydantic `BaseModel` for input validation
- Graceful error handling with fallback

### Registered tools

- `search_enamine_similarity(smiles, threshold, max_results)` — Find molecules with Tanimoto score above threshold
- `search_enamine_substructure(smiles, max_results)` — Find molecules containing the query as a substructure (API only, no local fallback for substructure)

---

## workflows.py — Composite Enamine + LinkLlama Pipeline

**File:** `src/synagent/workflows.py`

**What it does:** Chains Enamine search and LinkLlama into a single composite workflow. This is the "find purchasable analogs, then design linkers" pipeline.

### How `fragment_linker_workflow` works step by step

1. **Search Enamine** for both fragments — finds commercially available molecules similar to each input fragment
2. **Filter** out error results, fall back to original fragments if no hits found
3. **Generate fragment pairs** — takes top-3 hits for each fragment and creates all combinations (up to 3×3 = 9 pairs)
4. **Run LinkLlama** on each pair — distributes the total `num_linker_samples` across all pairs to keep cost bounded
5. **Deduplicate** linker proposals by SMILES string
6. **Rank results** — purchasable pairs first, then by combined Tanimoto score (higher = closer to original design intent)

### Output structure

```json
{
  "purchasable_fragments": {
    "fragment1_query": "[*]c1ccccc1",
    "fragment1_hits": [...],
    "fragment2_query": "[*]C1CCCCC1",
    "fragment2_hits": [...]
  },
  "linker_proposals": [
    {
      "linker_smiles": "CCC",
      "reasoning": "short chain linker",
      "fragment1": "...", "fragment2": "...",
      "fragment1_tanimoto": 0.9, "fragment2_tanimoto": 0.85,
      "fragment1_enamine_id": "EN100",
      "fragment2_enamine_id": "EN200",
      "purchasable": true
    }
  ],
  "summary": {
    "total_linkers": 15,
    "purchasable_pairs_used": 9,
    "fully_purchasable_linkers": 12
  }
}
```

### Registered as `find_and_link_fragments`

One tool call from the master agent triggers the entire Enamine → LinkLlama pipeline. The agent doesn't need to orchestrate the sub-steps manually.

---

## master.py — Disagreeable Master Agent

**File:** `src/synagent/agents/master.py`

### What changed

The master agent was rewritten from a neutral coordinator to a **"disagreeable advisor"** — it challenges the user's tool choices by default before cooperating.

### The disagreeable persona

- **Default behavior:** When a user proposes a workflow, the master agent pushes back:
  - *"Why SmileyLlama here instead of starting from a known scaffold?"*
  - *"Have you considered running retrosynthesis first to check feasibility?"*
  - *"Generating molecules without property constraints often yields unsynthesizable junk."*
- **Capitulation rule:** After the user provides a reasoned justification (or after 2 rounds of pushback), the agent cooperates fully and executes
- **Purpose:** Forces the user to think through their pipeline decisions, producing better workflows and avoiding wasted compute

### Tool registration

The master agent now has access to 10+ tools across four categories:

| Category | Tools |
|----------|-------|
| LLM Tools | `generate_molecules`, `retrosynthesis`, `design_linker` |
| Search Tools | `search_enamine_similarity`, `search_enamine_substructure`, `call_chemspace_agent` |
| Validation | `call_validation_agent`, `call_optimization_agent` |
| Composite | `find_and_link_fragments`, `full_route_evaluation` |

### Other changes

- Fixed typo: `MASTER_RPOMPT` → `MASTER_PROMPT`
- Improved docstrings on all sub-agent tool functions
- Added pipeline flow documentation in the system prompt so the LLM knows the standard order of operations

---

## tests/ — Mock Test Suite

**Directory:** `tests/`

All tests use `unittest.mock` to mock the vLLM server and Enamine API, so they run without any external services.

### conftest.py — Shared Fixtures

- `mock_openai_client` — Patches the vLLM OpenAI client singleton in `llm_tools.py`
- `make_completion(text)` — Factory that builds a fake OpenAI completion response
- `mock_httpx` — Patches `httpx.AsyncClient` for Enamine API tests

### test_llm_tools.py — LLM Tool Tests

- **Prompt construction tests:** Verify that `_build_smileyllama_prompt` correctly assembles property constraints into natural language. Tests all constraint types individually and in combination.
- **Generation tests:** Mock the vLLM response and verify output shape. Tests valid SMILES, empty responses, and multiple samples.
- **SynLlama tests:** Mock valid JSON pathways and invalid JSON. Verifies `parse_error` flag is set correctly and raw output is preserved on failure.
- **LinkLlama tests:** Mock JSON responses with `linker` and `reasoning` fields. Tests prompt construction with geometry and property constraints.

### test_enaminetool.py — Enamine Search Tests

- **API success:** Mock a 200 response with sample results, verify output structure
- **API failure → local fallback:** Mock an `httpx.HTTPError`, verify results come from `local_cache`
- **No API key:** Verify local fallback is used when `ENAMINE_API_KEY` is empty
- **Substructure without API:** Verify error message when substructure search has no API

### test_workflows.py — Composite Workflow Tests

- **Full workflow:** Mock both Enamine and LinkLlama, verify the composite output has `purchasable_fragments`, `linker_proposals`, and `summary` keys
- **No Enamine hits:** Verify that original fragments are used as fallback and `purchasable: false` is set

### test_master.py — Master Agent Configuration Tests

- **Prompt persona:** Assert the system prompt contains "disagreeable", "challenge", "why", "alternative"
- **Tool registration:** Assert all expected tools are registered on the agent (LLM tools, Enamine tools, workflow tools, sub-agent tools)

---

## pyproject.toml — Dependency Changes

### New dependencies added

| Package | Why |
|---------|-----|
| `openai>=1.0.0` | Client for the vLLM server (OpenAI-compatible API) |
| `httpx>=0.27.0` | Async HTTP client for Enamine REST API (also used by existing chemspacetool) |

### New dev dependencies

| Package | Why |
|---------|-----|
| `pytest>=8.0` | Test runner |
| `pytest-asyncio>=0.24` | Async test support for Enamine and workflow tests |

---

## How Tool Calls Work

The SynAgent pipeline uses **pydantic-ai** for tool orchestration. Here's how a tool call flows:

```
1. User sends a message to the master agent
2. Master agent (Gemini LLM) reads the message and its system prompt
3. Gemini decides which tool to call (e.g., "generate_molecules")
4. pydantic-ai invokes the registered Python function
5. The function validates input via Pydantic, calls the vLLM server
6. vLLM runs the fine-tuned model (e.g., SmileyLlama) and returns text
7. The function parses the response and returns structured data
8. Gemini receives the tool result and decides next steps
   (e.g., "now call retrosynthesis on these molecules")
9. Process repeats until the agent has a complete answer
```

### Tool types in this codebase

- **`@agent.tool_plain`** — No dependency injection. Used for LLM tools (vLLM client is a module singleton) and Enamine search (API key from env).
- **`@agent.tool`** (with `RunContext`) — Used by ChemSpace tools where a token manager needs to be injected via `deps`.

### vLLM serving

All three LLMs are served by a single vLLM server. The `model` parameter in each `completions.create()` call tells vLLM which checkpoint to use. Example startup:

```bash
vllm serve THGLab/Llama-3.1-8B-SmileyLlama-1.1 \
           --served-model-name THGLab/Llama-3.1-8B-SmileyLlama-1.1
```

For multi-model serving, use vLLM's `--model` flag multiple times or a model config file.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_API_KEY` | `EMPTY` | vLLM auth key (usually not needed locally) |
| `SMILEYLLAMA_MODEL` | `THGLab/Llama-3.1-8B-SmileyLlama-1.1` | SmileyLlama model ID |
| `SYNLLAMA_MODEL` | `SynLlama-1B` | SynLlama model ID |
| `LINKLLAMA_MODEL` | `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` | LinkLlama model ID |
| `ENAMINE_API_KEY` | *(empty)* | Enamine REST API key (falls back to local if unset) |
| `ENAMINE_BASE_URL` | `https://api.enamine.net/api/v1` | Enamine API endpoint |
| `CHEMSPACE_API_KEY` | *(from .env)* | ChemSpace API key (existing) |
