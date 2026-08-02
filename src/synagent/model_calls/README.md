# Model Calls

The `model-calls` capability gives a Pydantic AI agent access to three fine-tuned chemistry LLMs (SmileyLlama, SynLlama, LinkLlama), the Enamine REAL database, and a composite fragment-linking workflow. All LLM calls go through a vLLM server exposing an OpenAI-compatible `/v1/completions` endpoint. In the SynAgent pipeline the master agent uses these tools for de novo molecule generation, retrosynthetic planning, linker design, and purchasable fragment discovery.

## Capability class

```python
@dataclass
class ModelCalls(AbstractCapability[AgentDepsT]):
    id = "model-calls"
    description = "Use for molecule generation, retrosynthesis, linker design, and Enamine search."
    defer_loading = True
```

| Attribute | Value | Notes |
|-----------|-------|-------|
| `id` | `"model-calls"` | Identifier used when the master agent selects capabilities |
| `defer_loading` | `True` | Toolset is instantiated only when the capability is first used |
| `get_instructions()` | Pipeline instructions for all 6 tools | Injected into the agent's system prompt when the capability is active |
| `get_toolset()` | `ModelCallsToolset()` | Returns the six tools described below |

### Attaching to an agent

```python
from pydantic_ai import Agent
from synagent.model_calls import ModelCalls

agent = Agent(capabilities=[ModelCalls()])
```

## Tools

| Tool | Model / Source | Input | Output |
|------|---------------|-------|--------|
| `generate_molecules` | SmileyLlama (8B) | property constraints | `list[dict]` — SMILES + raw output |
| `retrosynthesis` | SynLlama (1B) | target SMILES | `dict` — pathways with reactions + building blocks |
| `design_linker` | LinkLlama (1B) | two fragments + geometry + constraints | `dict` — linker samples with reasoning |
| `search_enamine_similarity` | Enamine API / local RDKit | query SMILES + threshold | `dict` — ranked hits with Tanimoto scores |
| `search_enamine_substructure` | Enamine API | query SMILES | `dict` — substructure hits |
| `find_and_link_fragments` | Enamine + LinkLlama | two fragments + geometry | `dict` — purchasable fragments + linker proposals |

### `generate_molecules`

Generates novel drug-like molecules using SmileyLlama, an 8B-parameter LLM fine-tuned on ~2M ChEMBL SMILES.

```python
generate_molecules(
    mw_range: str | None = None,       # e.g. "<= 500"
    logp_range: str | None = None,     # e.g. "<= 5"
    hbd_range: str | None = None,      # e.g. "<= 5"
    hba_range: str | None = None,      # e.g. "<= 10"
    rotatable_bonds: str | None = None,# e.g. "<= 10"
    fsp3: str | None = None,           # e.g. ">= 0.25"
    macrocycle: bool | None = None,
    num_samples: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> list[dict]
```

Returns a list of `{"smiles": "<SMILES>", "raw_output": "<full model text>"}`. All property constraints are optional — omitting them produces an unconditional generation prompt. The prompt follows the SmileyLlama training format using the Llama-3.1 chat template with special tokens.

| Parameter | Range | Default | Notes |
|-----------|-------|---------|-------|
| `num_samples` | 1–50 | 1 | Number of molecules to generate per call |
| `temperature` | 0.0–2.0 | 0.8 | Higher = more diverse but less constrained |
| `top_p` | 0.0–1.0 | 0.95 | Nucleus sampling threshold |

### `retrosynthesis`

Decomposes a target molecule into retrosynthetic pathways using SynLlama, a 1B-parameter LLM fine-tuned on 2M reactions across 91 reaction templates.

```python
retrosynthesis(
    product_smiles: str,
    num_pathways: int = 1,
    temperature: float = 1.5,
    top_p: float = 0.9,
) -> dict
```

Returns `{"product": "<SMILES>", "pathways": [...]}`. Each pathway is either `{"pathway": <parsed JSON>, "parse_error": false}` on success, or `{"raw_output": "<text>", "parse_error": true}` when JSON parsing fails. Raw output is always preserved for debugging.

High temperature (1.5) and top-p (0.9) are the defaults because each call should produce a *different* retrosynthetic disconnection. Generate multiple pathways and filter downstream with the validation capability.

### `design_linker`

Proposes linker molecules between two fragments using LinkLlama, a 1B-parameter LLM fine-tuned on ChEMBL linker data with geometry and property constraints.

```python
design_linker(
    fragment1_smiles: str,          # SMILES with [*] attachment point
    fragment2_smiles: str,          # SMILES with [*] attachment point
    distance_angstrom: float,       # distance between attachment points (Å)
    angle_degrees: float,           # angle between attachment points (°)
    linker_type: str | None = None, # "chain", "branched", "ring-containing"
    rotb_range: str | None = None,
    heavy_atoms_range: str | None = None,
    hbd_range: str | None = None,
    hba_range: str | None = None,
    mw_range: str | None = None,
    logp_range: str | None = None,
    tpsa_range: str | None = None,
    reasonability: str = "reasonable",
    num_samples: int = 10,
    temperature: float = 1.4,
    top_p: float = 0.99,
) -> dict
```

Returns `{"fragments": [...], "geometry": {...}, "samples": [...]}`. Each sample is `{"linker": "<SMILES>", "reasoning": "<text>", "parse_error": false}` or `{"raw_output": "<text>", "parse_error": true}`. The prompt follows LinkLlama's `sft_corpus.py` training format: fragment info → linker properties → molecule properties → reasonability.

### `search_enamine_similarity`

Searches the Enamine REAL database for purchasable molecules similar to a query SMILES.

```python
search_enamine_similarity(
    smiles: str,
    similarity_threshold: float = 0.7,
    max_results: int = 10,
) -> dict
```

Returns `{"query": "<SMILES>", "search_type": "similarity", "results": [...]}`. Each result has `smiles`, `enamine_id`, `tanimoto_score`, `availability`, `price_info`, and `source` (`"enamine_api"` or `"local_cache"`).

**Two-tier search strategy:**

| Tier | Method | When used |
|------|--------|-----------|
| 1 | Enamine REST API | `ENAMINE_API_KEY` is set and API returns 200 |
| 2 | Local RDKit Morgan fingerprints | API key missing or API call fails |

The local fallback uses Morgan fingerprints (radius=2, 2048 bits) and Tanimoto similarity against a cached CSV at `data/enamine_fragments.csv`.

### `search_enamine_substructure`

Searches Enamine for molecules containing the query as a substructure.

```python
search_enamine_substructure(
    smiles: str,
    max_results: int = 10,
) -> dict
```

Same output shape as `search_enamine_similarity`. Requires the Enamine API — no local fallback for substructure search.

### `find_and_link_fragments`

Composite workflow that chains Enamine similarity search with LinkLlama linker design.

```python
find_and_link_fragments(
    fragment1_smiles: str,
    fragment2_smiles: str,
    distance_angstrom: float,
    angle_degrees: float,
    similarity_threshold: float = 0.6,
    max_enamine_results: int = 5,
    num_linker_samples: int = 10,
) -> dict
```

Steps:

1. Search Enamine for purchasable molecules similar to each input fragment
2. Take top-3 hits per fragment, form all pair combinations (up to 9 pairs)
3. Run `design_linker` on each pair, distributing `num_linker_samples` across pairs
4. Deduplicate linker SMILES and rank by purchasability then Tanimoto score

Returns:

```python
{
    "purchasable_fragments": {
        "fragment1_query": str, "fragment1_hits": list,
        "fragment2_query": str, "fragment2_hits": list,
    },
    "linker_proposals": [
        {
            "linker_smiles": str,
            "reasoning": str,
            "fragment1": str, "fragment2": str,
            "fragment1_tanimoto": float, "fragment2_tanimoto": float,
            "fragment1_enamine_id": str, "fragment2_enamine_id": str,
            "purchasable": bool,
        }
    ],
    "summary": {
        "total_linkers": int,
        "purchasable_pairs_used": int,
        "fully_purchasable_linkers": int,
    },
}
```

If no purchasable analogs are found for a fragment, the original fragment SMILES is used as a fallback (with `purchasable: false`).

## Related types

Defined in `src/synagent/model_calls/_models.py`:

| Model | Description |
|-------|-------------|
| `SmileyLlamaInput` | Property constraints for molecule generation (MW, LogP, HBD, HBA, rotatable bonds, Fsp3, macrocycle) |
| `MoleculeResult` | Generated molecule with `smiles` and `raw_output` |
| `SynLlamaInput` | Target SMILES + sampling config for retrosynthesis |
| `PathwayResult` | Single pathway with `pathway` (parsed JSON), `raw_output`, and `parse_error` flag |
| `RetrosynthesisResult` | Aggregated result with `product` and `pathways` list |
| `LinkLlamaInput` | Two fragments + geometry + property constraints for linker design |
| `LinkerSample` | Single linker proposal with `linker`, `reasoning`, and `parse_error` |
| `LinkerResult` | Aggregated result with `fragments`, `geometry`, and `samples` |
| `EnamineSearchInput` | Query SMILES + search type + threshold + max results |
| `EnamineHit` | Single Enamine result with `smiles`, `enamine_id`, `tanimoto_score`, `availability`, `source` |
| `LinkerProposal` | Linker with purchasability metadata from the composite workflow |
| `FragmentLinkerWorkflowResult` | Full workflow result with `purchasable_fragments`, `linker_proposals`, `summary` |

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_API_KEY` | `EMPTY` | vLLM auth key (usually not needed locally) |
| `SMILEYLLAMA_MODEL` | `THGLab/Llama-3.1-8B-SmileyLlama-1.1` | SmileyLlama model ID on vLLM |
| `SYNLLAMA_MODEL` | `SynLlama-1B` | SynLlama model ID on vLLM |
| `LINKLLAMA_MODEL` | `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` | LinkLlama model ID on vLLM |
| `ENAMINE_API_KEY` | *(empty)* | Enamine REST API key; falls back to local RDKit if unset |
| `ENAMINE_BASE_URL` | `https://api.enamine.net/api/v1` | Enamine API endpoint |
