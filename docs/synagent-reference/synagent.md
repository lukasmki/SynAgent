# SynAgent — methodology and workflow

Source: `src/synagent/synagent.py` (capability wiring, orchestrator setup),
`src/synagent/prompts.py` (both personas, verbatim).

SynAgent is a [pydantic-ai](https://ai.pydantic.dev) agent: one orchestrator
LLM, wired to 8 capabilities that each contribute a toolset, an instruction
fragment, and — for two of them (Corrector, Storage) — their own
tool-visibility rule.

## Orchestrator vs. chemistry models — two different things

SynAgent draws a hard line between the model that *drives the loop* and the
models that *do chemistry*:

- **The orchestrator** — the LLM deciding which tool to call next.
  Configurable via `build_model()`: local `qwen3.5` via Ollama (default), or
  a hosted model (Anthropic, Mistral, DeepSeek). It never produces SMILES
  directly except by calling a tool.
- **Three fine-tuned chemistry models**, each served separately via vLLM
  and reached only through the `model_calls` toolset:
  - **SmileyLlama** (8B) — de-novo generation under MW / LogP / HBD / HBA /
    rotatable-bond / Fsp3 / macrocycle constraints
  - **SynLlama** (1B) — target SMILES -> retrosynthetic route (SMARTS
    templates + building blocks)
  - **LinkLlama** (1B) — linker design between two fragments under
    geometry/property constraints

  The orchestrator calls these as tools; it doesn't reproduce their output
  itself. `build_model()`'s own docstring is explicit about why: a general
  chemistry-fluent LLM (e.g. Claude, used for exercising the agent loop
  without a vLLM server) will not reproduce SynLlama's route-JSON grammar or
  SmileyLlama's property-conditioned distribution.

## The 8 capabilities

| Capability | Contributes |
|---|---|
| `ModelCalls` | `generate_molecules`, `retrosynthesis`, `design_linker` (the three fine-tuned models above), plus Enamine REAL similarity/substructure search and a composite `find_and_link_fragments` |
| `SynthesisValidation` | `validate_route`, `validate_smiles`, `validate_reaction_smarts`, `validate_products`, `reverse_reaction` — RDKit-backed checks; the ground truth every other capability reads from or writes to |
| `Corrector` | See `corrector.md`. Gated behind "fix" / "correct" / "repair" |
| `Retrosynthesis` | `retro_search` — template-based retrosynthetic search, distinct from SynLlama's learned `retrosynthesis` call. Only invoked when the user explicitly asks to redesign or find a new route |
| `AnalogueSearch` | `search_building_blocks`, `search_templates`, `search_building_blocks_by_template` — the local Enamine-derived fingerprint database, independent of the corrector's own narrower `search_step_building_blocks` |
| `Scoring` | `score_molecules`, `score_reactions`, `score_paths` — hazard and synthetic-accessibility scoring |
| `Storage` | `save_record`, `get_record`, `list_records`. Gated the same way as the corrector: only used when the user explicitly asks to save or retrieve |
| `SubAgents` | Two delegate agents sharing the orchestrator model: `analogue` (building-block/reaction database + optional Chemspace) and `worker` (general-purpose, validation + analogue search). Chemspace is included only when `CHEMSPACE_API_KEY` is actually set |

`Chemspace` is loaded conditionally (`_chemspace_caps()`): constructing it
eagerly without an API key used to crash agent construction entirely, so
it's now an empty list unless the key is present, for both the top-level
agent and both sub-agents.

## The standard flow this composes into

No single tool orchestrates the following — it emerges from the
capabilities above being called in sequence as a request naturally moves
through it:

```
generate (SmileyLlama)
  -> decompose into route (SynLlama)
  -> validate_route
  -> corrector repairs failures
  -> source building blocks
  -> score
```

Not every request needs every stage. A request to just validate a route
someone already has stops at `validate_route` and reports; nothing
downstream fires unless asked for.

## Two personas, deliberately different defaults

### Deterministic — default
Terse, rule-bound, every clause names a tool and ends "STOP." Exists so a
tool sequence is reproducible — the pipeline's own tests depend on it. This
is what runs in automated/benchmark contexts, including every corrector run
behind the Table 1 numbers in `../chembl-benchmark/`.

### Disagreeable — opt-in
A skeptical-advisor persona that pushes back before executing: questions
vague requests, suggests validation before generation, challenges skipped
steps. Useful for interactive exploration; **actively harmful in automated
tests** (per its own module docstring), since it spends turns arguing
instead of calling tools.

## The pattern that ties it together

The tool-gating seen in the corrector isn't specific to it — it's how the
whole agent is designed to behave. Storage tools are invisible until the
user says save or retrieve. `retro_search` stays off unless redesign is
explicitly requested. The system instructions state the rule at the top
level too:

> Never use a tool unless the user explicitly asked for that action. Never
> copy or retype SMILES or JSON yourself.

The corrector is simply the capability where that rule is enforced
programmatically (hidden tool definitions, via `prepare_tools()`) rather
than by instruction alone.
