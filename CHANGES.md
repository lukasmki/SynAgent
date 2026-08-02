# `synagent-full-pipeline` — what this branch does

Combines three branches so one agent can run the full pipeline:

> SmileyLlama generates → SynLlama plans a route → validation checks it → the corrector fixes it

Three commits. Test suite: **26 passed, 2 skipped** (the skipped two are the
live tier — they need servers that don't exist yet).

---

## Why the branch structure changed

The task was described as combining `SynLlama-SmileyLlama-linkLlama-added-in`
with `corrector.py2`. Those two **cannot** be merged directly: the first is
branched off a pre-refactor `main` and uses flat modules (`agents/master.py`,
`llm_tools.py`); the second uses the pydantic-v2 capability/toolset
architecture. Combining them as-is would be a rewrite.

`model-calls-capability` is the same three-Llama work **already ported to the
current architecture**, sharing a base commit with `corrector.py2`. Merging
those two produces exactly one conflict.

Verification that nothing was lost from the third branch:

| Branch 1 asset | Status |
|---|---|
| `llm_tools.py` (3 model tool calls) | absorbed into `model_calls/_toolset.py`, Pydantic schemas **identical field-for-field** |
| `enaminetool.py` | absorbed — API + local RDKit fallback both present |
| `workflows.py` | absorbed as `find_and_link_fragments` |
| Disagreeable master persona | **was dropped** by the port (reduced to one line) → restored in `prompts.py` |
| `tests/` mocks | **not ported** → restored in `tests/conftest.py`, retargeted to the new module paths |

So all three branches are represented.

---

## Commit 1 — `2199b4b` the merge

One conflict, in the `capabilities=[...]` list of `synagent.py`. Resolved to
the union.

**One non-obvious change was required.** corrector.py2's instructions end with
*"Never use a tool unless the user explicitly asked for that action"* and
enumerate only validation / fixing / retro_search / storage. Merging in
`generate_molecules`, `retrosynthesis` and `design_linker` without saying
anything about them leaves the agent holding tools it will never call. A
GENERATION clause was added, which also disambiguates SynLlama's
`retrosynthesis` tool from the template-search tool `retro_search` — near
identical names, different things.

---

## Commit 2 — `196332b` persona, Claude backend, tests

### `prompts.py`
- `DETERMINISTIC` (default) — corrector.py2's terse rule-bound prompt. The
  pipeline tests depend on it for reproducible tool sequences.
- `DISAGREEABLE` — branch 1's master persona, **opt-in**. An agent that argues
  for two rounds before executing is good interactively and actively breaks
  automated tests, so it is not the default.

### `synagent.py`
`get_agent()` gains `provider` and `persona`. Both existing callers pass
positionally, so this is backward compatible.

```bash
uv run synagent serve --provider anthropic --model claude-sonnet-5
```

`provider="anthropic"` runs the agent loop with **no local server** — useful
for exercising capability wiring, tool gating and subagent routing while the
vLLM port is still being built. It does **not** replace the chemistry models:
Claude will not emit SynLlama's route format or SmileyLlama's
property-conditioned distribution. Real agent loop, mocked chemistry tools.

### `tests/`
| Tier | Needs | Runs today |
|---|---|---|
| `test_model_calls.py` | nothing | yes — 16 tests |
| `test_pipeline.py` wiring / gate / mocked | nothing | yes — 10 tests |
| `test_pipeline.py::TestLivePipeline` | vLLM + orchestrator | skips |

Live assertions are **structural, never exact-match** — these models sample.

---

## Commit 3 — `0a8c503` two real bugs, found by running the tests

Both pre-existing on `corrector.py2`. Neither is visible without executing.

### 1. `get_agent()` was impossible to construct without `CHEMSPACE_API_KEY`

`ChemspaceToolset.__init__` builds `ChemspaceAPI` eagerly, which raises when
the key is unset. corrector.py2 commented `Chemspace()` out of the top-level
capability list for exactly this reason — **but left it in both subagents**, so
the constructor still raised. Anyone without a ChemSpace key could not build
the agent at all, which blocks the entire pipeline.

Chemspace is now included only when the key is present.

### 2. Seven capabilities silently lost `id`, `description` and `defer_loading`

`AnalogueSearch` is a plain class. The other seven are `@dataclass` and
declared those three names as **unannotated** class attributes. The base
`AbstractCapability` declares the same names as dataclass fields defaulting to
`None`/`False`. Unannotated class attributes are not fields, so the inherited
defaults won on every instance:

```python
ModelCalls.id                # "model-calls"
ModelCalls().id              # None     <- what the agent actually saw
ModelCalls().defer_loading   # False, though declared True
```

All seven now carry explicit annotations, guarded by
`test_capability_ids_survive_dataclass`.

> **Behaviour change to watch.** This flips `defer_loading` to actually-True
> for `chemspace`, `model_calls`, `scoring` and `storage`. That is the declared
> intent, but those toolsets now load **lazily** where they previously loaded
> eagerly.

### Correction to commit 2

Commit 2 claimed the fence-stripping in `retrosynthesis()` was buggy. **It is
not.** `str.strip(chars)` does strip characters rather than a substring, but
JSON always begins `{`/`[` and ends `}`/`]`, none of which are in the strip
set, so payloads survive. The `xfail` is replaced by tests pinning the real
behaviour, including the leading-prose case that correctly reports
`parse_error: True`.

---

## The corrector gate — read this before debugging

`Corrector.prepare_tools()` hides **every** corrector tool unless one of

> `fix` · `correct` · `repair` · `search alternative` · `alternative building block`

appears in the **last three user messages**. "Improve the route" gets you no
corrector tools and an agent that looks like it is ignoring you.

`TestLivePipeline` drives a multi-turn conversation whose repair turn literally
contains "fix". That is deliberate — it exercises the real production path.
Relaxing the gate to simplify the test would mean the test stops covering what
ships.

---

## Environment

| Variable | Default |
|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` |
| `SMILEYLLAMA_MODEL` | `THGLab/Llama-3.1-8B-SmileyLlama-1.1` |
| `SYNLLAMA_MODEL` | `SynLlama-1B` |
| `LINKLLAMA_MODEL` | `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` |
| `ENAMINE_API_KEY` | unset — falls back to local RDKit similarity |
| `CHEMSPACE_API_KEY` | unset — Chemspace omitted |
| `ANTHROPIC_API_KEY` | unset — required for `--provider anthropic` |

**Port collision:** `synagent serve` defaults to port 8000, and so does
`VLLM_BASE_URL`. Move one.

---

## Known issues not fixed here

1. **`SubAgents(agents={...})` breaks on the latest `pydantic-ai-harness`.**
   Verified against the locked `pydantic-ai 2.0.0` / `pydantic-ai-harness
   0.4.0`, where the dict form is correct. On the current release it raises
   `AttributeError: 'str' object has no attribute 'resolved_name'` — it now
   expects a list of `SubAgent`. A dependency upgrade will need this rewritten.
2. **Deprecation:** `pydantic_ai_harness.experimental.subagents` has moved to
   `pydantic_ai_harness.subagents`.
3. **`__main__.py` inconsistency (pre-existing):** `serve` defaults to
   `qwen3.5:9b` while `cli` defaults to `google:gemini-3-flash-preview`, but
   `get_agent` points the OpenAI-compatible client at Ollama, so the gemini
   default cannot work as written.
4. **Fixture capture (A1.5) not started** — needs the vLLM server.

---

## Reproducing the test run

```bash
uv sync && uv run pytest -v
```

Verified on Python 3.14.5 with `pydantic-ai==2.0.0`,
`pydantic-ai-harness[codemode]==0.4.0`, `rdkit 2026.03.4`, `FPSim2 0.7.4`
built from `github.com/lukasmki/FPSim2`.
