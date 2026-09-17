# SynAgent reference — tools, workflow, methodology

What SynAgent is made of and how it decides what to run. Two parts: the
corrector (one capability, the one most of the ChEMBL benchmark work in
`../chembl-benchmark/` exercises), then SynAgent as a whole.

Everything below is read directly from source, not recalled from memory:
`src/synagent/corrector/_toolset.py`, `src/synagent/corrector/_capability.py`,
`src/synagent/synagent.py`, `src/synagent/prompts.py`, and the other
capabilities' `_toolset.py` files.

| File | What |
|---|---|
| `corrector.md` | The corrector's 8 tools, the gate that hides them, the prescribed fix sequence, and the failure-mode decision tree inside `fix_step` |
| `synagent.md` | All 8 capabilities, the orchestrator/chemistry-model split, the standard pipeline flow, and the two personas |

## The one-paragraph version

SynAgent is a [pydantic-ai](https://ai.pydantic.dev) agent: one orchestrator
LLM (local `qwen3.5` by default, or a hosted model) wired to 8 capabilities,
each contributing tools, instructions, and sometimes its own
tool-visibility rule. Two capabilities — Corrector and Storage — hide their
tools entirely until the user says a trigger word ("fix"/"correct"/"repair",
or "save"/"retrieve"); this isn't a suggestion in the prompt, the tool
definitions are absent from what the model sees. The corrector never asks
the model to retype a SMILES string: every tool reads its input from the
last `ValidationReport` in conversation history rather than from arguments.

## Reproduce / verify

```bash
grep -n "add_function" src/synagent/corrector/_toolset.py   # the 8 tools
sed -n '1,45p' src/synagent/corrector/_capability.py          # the gate + prescribed sequence
sed -n '90,160p' src/synagent/synagent.py                      # capability wiring
cat src/synagent/prompts.py                                    # both personas, verbatim
```
