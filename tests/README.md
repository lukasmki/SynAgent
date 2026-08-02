# SynAgent tests

Three tiers, separated by what infrastructure each needs.

| File | Needs | Runs today |
|---|---|---|
| `test_model_calls.py` | nothing | yes |
| `test_pipeline.py::TestMergeWiring` / `TestCorrectorGate` / `TestMockedPipeline` | nothing | yes |
| `test_pipeline.py::TestLivePipeline` | vLLM + orchestrator | **no — skips** |

```bash
uv run pytest
```

The live tier skips automatically when the servers aren't reachable, so this is
safe to run and safe in CI while the vLLM port is still being built.

## Two model layers, don't conflate them

**Orchestrator** — the LLM driving the agent loop and deciding which tools to
call. Either a local Ollama, or Claude:

```bash
uv run synagent serve --provider anthropic --model claude-sonnet-5
```

`--provider anthropic` needs `ANTHROPIC_API_KEY` and needs no local server. It
is the fastest way to exercise capability wiring, tool gating and subagent
routing.

**Chemistry models** — SmileyLlama, SynLlama, LinkLlama. These are specific
fine-tuned weights behind a vLLM server. **Claude cannot substitute for them.**
It will not emit SynLlama's route format or reproduce SmileyLlama's
property-conditioned distribution. With `--provider anthropic` and no vLLM, you
get a real agent loop over mocked chemistry tools — useful, but not a chemistry
result.

## Environment

| Variable | Default | Used by |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | all three chemistry models |
| `VLLM_API_KEY` | `EMPTY` | vLLM (no real auth locally) |
| `SMILEYLLAMA_MODEL` | `THGLab/Llama-3.1-8B-SmileyLlama-1.1` | generate_molecules |
| `SYNLLAMA_MODEL` | `SynLlama-1B` | retrosynthesis |
| `LINKLLAMA_MODEL` | `THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50` | design_linker |
| `ENAMINE_API_KEY` | unset | Enamine search (falls back to local RDKit) |
| `ANTHROPIC_API_KEY` | unset | `--provider anthropic` |
| `SYNAGENT_MODEL` | `qwen3.5:9b` / `claude-sonnet-5` | live tests |

**Port collision:** `synagent serve` defaults to port 8000 and so does
`VLLM_BASE_URL`. Run the web server on another port, or move vLLM.

## The corrector gate — read before "fixing" a failing test

`Corrector.prepare_tools()` hides **every** corrector tool unless one of

> `fix` · `correct` · `repair` · `search alternative` · `alternative building block`

appears in the **last three user messages**. A prompt that says "repair the
route" works; one that says "improve the route" silently gets no corrector
tools and the agent will look like it is ignoring you.

`TestLivePipeline` therefore drives a multi-turn conversation whose repair turn
literally contains "fix". That is deliberate — it exercises the real production
path. Relaxing the gate to simplify the test would mean the test no longer
covers what actually ships.

## Fence stripping — looks wrong, is actually fine

`ModelCallsToolset.retrosynthesis` strips markdown fences with
`raw_text.strip().strip("\`\`\`json").strip("\`\`\`")`. `str.strip(chars)` strips
*characters*, not a substring, so it removes any leading/trailing `` ` j s o n ``.

That reads like a bug, and it was initially reported as one here. It isn't:
JSON objects and arrays always start with `{` or `[` and end with `}` or `]`,
none of which are in the strip set, so the payload survives. Both the fenced and
unfenced forms are pinned by tests so a future "cleanup" of the strip chain
can't silently change the behaviour.

Where it genuinely gives up is leading prose ("Here is the route:\n```json..."),
which is recorded as `parse_error: True` rather than mangled — the right
outcome. Also pinned.

## Capability metadata — fixed here

Every capability except `AnalogueSearch` is a `@dataclass`, and they declared
`id` / `description` / `defer_loading` as **unannotated** class attributes. The
base `AbstractCapability` declares those same names as dataclass fields
defaulting to `None`/`False`. Unannotated class attributes are not fields, so
the inherited defaults won on every instance:

```python
ModelCalls.id            # "model-calls"
ModelCalls().id          # None      <- before the fix
ModelCalls().defer_loading  # False  <- declared True
```

Seven capabilities were affected. All now carry explicit annotations.
`test_capability_ids_survive_dataclass` guards it.

Note this **flips `defer_loading` to actually-True** for chemspace, model_calls,
scoring and storage — the declared intent, but a genuine runtime behaviour
change: those toolsets now load lazily where they previously loaded eagerly.
