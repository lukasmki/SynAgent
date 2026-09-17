# The corrector — tools and workflow

Source: `src/synagent/corrector/_toolset.py` (the 8 tools),
`src/synagent/corrector/_capability.py` (the gate and prescribed sequence).

## When it activates — the gate

`Corrector.prepare_tools()` scans the **last 3 user messages** for one of
five trigger phrases:

```
fix · correct · repair · search alternative · alternative building block
```

Outside that window, all 8 corrector tools are removed from the tool list
the model is given — not discouraged by instruction, actually absent. Say
"validate this route" and the corrector doesn't exist as far as the model
can tell. Say "now fix it" and the full toolset appears on the next turn.

This mirrors a rule stated twice elsewhere in the system: capability-level
(`Corrector.description`) and prompt-level (`prompts.py`) — **never use a
tool unless the user explicitly asked for that action.**

## The prescribed sequence

Per `Corrector.get_instructions()`, once triggered the model is told to run
exactly this, once through:

```
fix_building_blocks()
  -> fix_step(N)   once per failed step, no repeats
  -> apply_fixes()
```

If the resulting report still fails the *same* steps, one more round is
allowed (`fix_step` -> `apply_fixes` again), then it stops regardless of
outcome. `apply_fixes` is never called more than once per round. Tools
outside the corrector (`retro_search`, `save_record`,
`search_step_building_blocks`, `search_building_blocks`, `score_molecules`)
stay off-limits unless separately requested.

## How `fix_step` routes by failure mode

`fix_step` is the dispatcher. It reads the failing reaction's
`failure_mode` from the last `ValidationReport` and picks one of three
chains:

### `invalid_template`

```
fix_smarts  ->  fix_template
```

Syntactic repair first (tag-stripping, unescaping, primitive fixes). Only
reaches for the 167-template library search if the SMARTS itself can't be
salvaged as text.

### `no_products` / `wrong_product`

```
fix_template  ->  extract_template_from_reaction  ->  retro-disconnection
```

Escalating cost: try the known library first, then derive a fresh template
via Indigo + rdchiral, then — last resort — reverse the original template
against the product and retry `fix_template` on each resulting precursor
set.

### `invalid_reactant_smiles` / `invalid_product_smiles`

```
fix_smiles
```

Syntactic SMILES repair only — no chemistry search needed for a malformed
string.

## The 8 tools

### `fix_step(step)`
The dispatcher above. Looks up the step in the last `ValidationReport`,
branches on `failure_mode`, and calls the other tools in sequence until one
reports `fixed: True` or all options are exhausted.

### `fix_building_blocks()`
Runs `fix_smiles` on every building block flagged `is_valid: false` in the
last report. No arguments — reads which blocks are invalid from
conversation history.

### `apply_fixes()`
Rebuilds the route from every `fix_step`/`fix_building_blocks` result since
the last report, then re-validates it.

**Building blocks are derived from the rebuilt reactions themselves** —
every reactant that isn't another step's product — rather than patched from
the original `building_blocks` list. (This used to be patched
independently, which let the same molecule get "fixed" two different ways
and silently drift out of sync — fixed; see `../chembl-benchmark/table1_with_synagent.md`
footnote 6 for the measured effect: BB Selection 94.28% -> 98.40%.)

### `fix_smarts(reaction_smarts)`
Three syntactic passes, cheapest first:
1. strip stray XML-style tags/quotes
2. unescape encoded angle brackets
3. fix a known RDKit retro-SMARTS quirk — bare `N`/`C` on the product side
   need `[#7]`/`[#6]`

Stops at the first pass that parses.

### `fix_template(reactants, product)`
Searches a fixed 167-template library (`_COMMON_SMARTS`) for one that turns
the given reactants into the given product, trying every reactant order.
Worst case: 167 templates x every reactant permutation x RDKit's
`RunReactants` — run off the event loop in a worker thread
(`asyncio.to_thread`) because of that cost. (Previously ran inline inside an
`async def` with no `await` — blocked the event loop outright on
complex/many-reactant routes; see the corrector-bugs section of
`../chembl-benchmark/table1_with_synagent.md`.)

### `extract_template_from_reaction(reactants, product)`
When nothing in the library fits: Indigo atom-maps the reactant->product
transformation, rdchiral extracts a fresh retrosynthetic SMARTS from that
mapping, then a self-consistency check confirms the derived template
actually reproduces the product before it's trusted. This is the tool that
**invents** chemistry outside the trained template set — it's why
Template Memorization drops after correction even as route validity goes
up (same tradeoff, documented in the Table 1 write-up).

### `search_step_building_blocks(step, threshold, max_results)`
Cosine-similarity search over a local Enamine-derived building-block
fingerprint database (FPSim2) for replacements to a step's reactants. Only
reachable when the trigger phrase itself mentions an alternative building
block (`search alternative` / `alternative building block`).

### `fix_smiles(smiles)`
Layered SMILES repair:
1. parse as-is
2. strip atom-map numbers and retry
3. partial sanitization
4. bracket/paren/ring-closure completion for strings that look truncated
   (mismatched `[ ]`/`( )` counts, or an odd number of any ring-bond digit)

The completion pass tries appending closers, inserting a missing `]` at
every plausible position, then single-character mutations (dropping or
shifting a paren) combined with the same completion logic — a genuine
constraint search over small edits, not a single heuristic.

## The state-store pattern

The corrector never asks the model to retype a SMILES string or JSON blob.
Every tool either takes arguments the model already has in view, or reads
its own input from the last `ValidationReport` / prior fix results in
conversation history — **the conversation is the state store.** That's also
why the corrector can't be called as a library offline: a repair needs its
own two-turn conversation, one route at a time (see
`../chembl-benchmark/run_repair_frozen.py` for how the benchmark work drove
this programmatically instead of through the UI).
