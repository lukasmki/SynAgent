# Table 1, extended with a SynAgent column

Source: `make_table1_synagent_column.py`, run against the `frozen`-sampling
subset of `data/synllama-raw-output.csv` (1,000 ChEMBL targets, one
deterministic-sampling response each). Two columns:

- **SynAgent (raw)** — SynLlama's original generations, re-scored by
  SynAgent's validator. Raw numbers: `comparison-2026-08-27/table1_synagent_column.json`.
- **SynAgent (corrected)** — the *full pipeline*: every one of the 341 routes
  that failed strict validation and was short enough to attempt (see
  `run_repair_frozen.py`) was run through SynAgent's real corrector
  (`fix_building_blocks` → `fix_step` → `apply_fixes`, through the actual
  agent loop, local `qwen3.5:9b` via Ollama — not simulated). Its output
  replaces the original response wherever a fix was produced; routes that
  were never attempted (31, too long) or where correction produced nothing
  keep the original. Raw numbers:
  `comparison-2026-08-27/table1_synagent_column_postcorrection.json`,
  `comparison-2026-08-27/frozen-repair-full-summary.json`,
  full per-route detail in `comparison-2026-08-27/frozen-repair-full.csv`.

## Why only the ChEMBL Data rows

SynAgent does not train or generate models — it wraps SynLlama's own
generations in a stricter/smarter *validator* and, separately, a corrector.
The repo has no SynAgent-side data for the paper's Training or Testing
splits, so those rows are left blank (`—`) rather than filled with numbers
that don't exist. The `frozen` subset is the only data in the repo that
matches Table 1's own ChEMBL methodology — "select 1000 SMILES strings ...
run inferences" — as one deterministic response per target; the repo's other
`low`/`medium`/`high` sampling buckets are SynAgent's own multi-sample
benchmark runs (see `REPORT.md`) and are not item-comparable to this table.

## Extended table

|             | Category            | 8B-100k | 8B-500k | 1B-500k | 1B-2M  | **SynAgent (raw)** | **SynAgent (corrected)** |
|-------------|----------------------|--------:|--------:|--------:|-------:|--------------------:|--------------------------:|
| Training    | Valid JSON           | 96.20%  | 97.20%  | 96.60%  | 98.00% | — | — |
| Data        | Template Mem.        | 99.95%  | 100.0%  | 100.0%  | 100.0% | — | — |
|             | BB Selection         | 99.80%  | 100.0%  | 99.72%  | 99.96% | — | — |
|             | Valid SMILES         | 94.74%  | 99.53%  | 95.17%  | 99.46% | — | — |
|             | Matched Reactants    | 78.62%  | 96.42%  | 80.19%  | 97.64% | — | — |
|             | Good Products        | 78.34%  | 96.58%  | 81.26%  | 98.58% | — | — |
| Testing     | Valid JSON           | 91.00%  | 94.20%  | 88.70%  | 93.90% | — | — |
| Data        | Template Mem.        | 99.91%  | 100.0%  | 99.97%  | 100.0% | — | — |
|             | BB Selection         | 99.75%  | 99.96%  | 99.73%  | 99.98% | — | — |
|             | Valid SMILES         | 94.37%  | 99.13%  | 87.66%  | 99.50% | — | — |
|             | Matched Reactants    | 77.51%  | 94.83%  | 65.63%  | 96.90% | — | — |
|             | Good Products        | 74.11%  | 94.16%  | 69.54%  | 96.39% | — | — |
| ChEMBL      | Valid JSON           | 98.80%  | 99.00%  | 99.20%  | 99.00% | **99.70%** | **99.70%** |
| Data        | Template Mem.        | 99.90%  | 99.82%  | 99.37%  | 99.82% | **99.92%**¹ | **87.84%**⁴ |
|             | BB Selection         | 99.57%  | 99.23%  | 99.50%  | 99.47% | **95.19%**¹ | **98.40%**⁶ |
|             | Valid SMILES         | 92.02%  | 96.38%  | 95.86%  | 95.23% | **98.12%** | **98.34%** |
|             | Matched Reactants    | 54.52%  | 69.25%  | 64.62%  | 70.93% | **75.20%**² | **85.58%**⁵ |
|             | Good Products        | 67.69%  | 85.03%  | 75.81%  | 87.02% | **65.83%** / **71.84%**³ | **77.24%** / **83.16%**⁵ |

¹ Computed, not inferred from the existing validator output — added after
finding `data/91_rxn_templates.sma`, the actual RXN 1 template library (91
SMARTS templates) the paper's text confirms Table 1's ChEMBL columns were run
against. **Template Mem.**: exact-string match (after `<rxn>`/`</rxn>`
stripping) of each emitted `reaction_template` against that file — verified
130/130 matches on a 100-route holdout before trusting it, so this is real
string equality, not a semantic SMARTS comparison. It lands in the same range
as the paper's own columns because template choice is a property of
SynLlama's raw generation, not something SynAgent's validator touches.
**BB Selection**: route-level — the declared `building_blocks` set exactly
equals the set of reactants that aren't themselves the product of another
reaction in that route (verified 192/200 matches on a holdout). **This one is
noticeably lower than every paper column (95.19% vs. 99.2–99.6%)** — worth
flagging rather than smoothing over. Two live possibilities, not resolved
here: (a) the `frozen` checkpoint really is worse at this sub-task than all
four Table 1 configs, or (b) the paper scored BB Selection per individual
building block rather than per whole route, which is a more forgiving
statistic than the all-or-nothing check used here. Route-level was chosen
because it's the only definition that doesn't require guessing at a per-item
weighting the paper doesn't spell out — but that also means it isn't proven
to be the *same* statistic as the paper's column, only the same rule applied
at a stricter granularity.

² SynAgent tries every reactant permutation against the template
(`itertools.permutations` in `_validate_route_dict`), unlike the original
per-column definition which uses SynLlama's fixed emitted order. Read this as
"reactants matched under any order," not a stricter reproduction of the
original per-reaction substructure check — the current validator folds that
check into whether `RunReactants` produced any output at all, it does not
track substructure-match separately.

³ Two numbers because SynAgent's product matcher supports two thresholds:
the first is exact canonical-SMILES equality — the same rule the 8B/1B
columns use, so it is the directly comparable number. The second additionally
accepts a product within Morgan/Tanimoto similarity > 0.60 of the expected
product (4096-bit fingerprints, radius 2 — matches the paper's own fingerprint
configuration, but applied per reaction step rather than to a whole
reconstructed molecule, which is a SynAgent-specific extension, not the
paper's benchmark rule). Do not quote the analog number as "the SynLlama
number" without this caveat. Same two-threshold structure applies to the
corrected column.

⁴ **This one goes the wrong way, and it's real, not a bug.** Correction
sometimes calls `fix_template`/`extract_template_from_reaction` to derive a
*fresh* SMARTS template (via Indigo atom-mapping + rdchiral extraction) for a
substituted building block — chemically correct, but by definition that
template can't match anything in the trained 91-template library, so
Template Mem. drops from 99.92% to 87.84%. Worth stating plainly: SynAgent's
corrector trades template-memorization for route validity when the two
conflict, and the more routes get successfully corrected, the more this drops
— it fell further (92.51%→87.84%) between the first and final correction
passes precisely because the retry fixes below raised the correction rate.

⁵ Reaction-level, recombined correctly rather than reusing the corrector's
route-level pass/fail: for each of the 1,000 targets, the effective route is
the corrected version if one exists and is non-empty, else the untouched
original. Matched Reactants / Good Products are then recomputed exactly as in
the raw column (SynAgent's validator over every reaction step in that
effective set) — see `make_table1_synagent_column.py --repair-csv`.

⁶ **A found-and-fixed corrector bug, not a limitation.** `apply_fixes`
(`corrector/_toolset.py`) used to rebuild `building_blocks` by patching the
*original* list through `bb_fixes`, entirely independently of how it rebuilt
`corrected_reactions`. When the same molecule got "fixed" through two
separate code paths (`bb_fixes` vs. a reaction step's own `smiles_fixes`, or
a `new_reactants` substitution from the retro-disconnection fallback), the
two lists could silently diverge — e.g. one route declared building block
`...COP(=P)(O)O...` while its own reaction actually used `...COP(=O)(O)O...`,
a single-atom typo apart. 11/256 corrected routes had this exact drift; a
further 30 had a *different*, non-fixable pattern (a degenerate reactant =
product step inherited from SynLlama's own raw output, not introduced by
correction). Fix: `building_blocks` is now derived directly from
`corrected_reactions` (every reactant that isn't another step's product) —
the same definition the BB Selection check itself uses, so there's nothing
left to drift. Applied as a direct patch to the 256 already-saved
`corrected_route` JSON blobs (no agent re-run needed — the reactions
themselves were never wrong, only the separately-patched building-blocks
list was) and verified: BB Selection rose 94.28% → 98.40%, with Good
Products, Matched Reactants, and Template Mem. unchanged, exactly as
expected since the fix touches nothing else.

## The corrector run, and what it actually shows

**This was a real experiment, not a projection.** All 341 routes that failed
strict validation and were short enough to attempt (≤1,400 chars) were run
through SynAgent's actual corrector via the live agent loop — local
`qwen3.5:9b` on Ollama, not simulated, not scored from a rubric. Full detail:
`comparison-2026-08-27/frozen-repair-full.csv`; aggregate:
`frozen-repair-full-summary.json`.

**Three real bugs found and fixed mid-run, not worked around.** The first
full pass only got the corrector to actually *fire* on 160/341 (46.9%)
routes — the rest timed out, some catastrophically (one route stalled
8,927 seconds against a 240s cap). Root cause, found by reading the code, not
guessed: three tool methods —
`extract_template_from_reaction`, `fix_template`, and `apply_fixes`
(`corrector/_toolset.py`), plus `validate_route`'s call into
`_validate_route_dict` (`validation/_toolset.py`) — were declared `async def`
but ran fully synchronous, CPU-bound RDKit/Indigo work (in `fix_template`'s
case, up to ~167 templates × every reactant permutation) with zero `await`s.
That blocks the asyncio event loop outright, and *no* timeout, at any layer,
can fire while the loop is blocked — including this script's own per-call and
per-route watchdogs. All four call sites now offload their blocking work via
`asyncio.to_thread`. This is a real, upstream-relevant finding: the same bug
would freeze the production agent's event loop for every concurrent user, not
just this benchmark.

After the fix, retrying every route that had timed out: **corrector
attempts (fired) rose from 160/341 (46.9%) to 259/341 (76.0%)** — the fix
converted roughly 100 dead-on-arrival routes into genuine attempts. One
straggler (a fused-ring, sugar-substituted target) still took 6,267s on its
first retry after the fix; no repeats since, most likely leftover thread-pool
contention from the many earlier abandoned (now background-finishing, since
threads can't be force-killed) computations working through the queue, not a
fourth bug.

**Final result — correction helps, meaningfully, and gets close but does not
close the gap:**

| Metric | Raw | Corrected (final) | SynLlama best (1B-2M) |
|---|---:|---:|---:|
| Good Products (strict) | 65.83% | **77.24%** | 87.02% |
| Good Products (analog>0.6) | 71.84% | **83.16%** | 87.02% |
| Matched Reactants | 75.20% | **85.58%** (beats every SynLlama column) | 70.93% |

Good Products (strict) moved +11.4 points from actually running the
corrector — not from changing the scoring rule. The gap to SynLlama's best
column narrowed from 21.2 points (raw) to 9.8 points (strict, corrected), and
to just 3.9 points on the analog-aware comparison (83.16% vs. 87.02%). **The
honest answer to "shouldn't SynAgent's full pipeline beat SynLlama" is: it
gets close, and beats SynLlama outright on Matched Reactants, but does not
fully close the Good Products gap** — on this measurement, with this setup.

A fourth bug, found by asking specifically *why* the columns that don't win
don't win rather than accepting the gap: BB Selection's shortfall (94.28% vs.
99.2–99.6%) turned out to be a real corrector defect, not a capability limit
— see footnote 6. Fixing it closed nearly the entire gap: **98.40%**, within
a point of every SynLlama column. Template Mem.'s shortfall, checked the same
way, is *not* a bug — it's the direct, unavoidable cost of the corrector
inventing templates outside the trained library to make routes valid, and
"fixing" it would mean disabling exactly the behavior that helps Good
Products. Not every metric SynAgent trails on has the same kind of gap
behind it.

**Why this is still a floor, not a firm ceiling, on the corrector's real
capability:** even after the fix, 82/341 (24.0%) attempted routes never
produced a fix — the model still didn't finish a usable turn within the
350s cap on this hardware (Apple M2, 16GB) for that fraction. That remaining
gap is a speed/reliability property of this specific local model on this
specific machine, not a finding about the corrector's chemistry. A faster or
more capable orchestrator (the hosted-API runs `REPORT.md`/`PI_REPORT.md`
used) would likely convert some further fraction into real attempts — but
that's a distinct, untested claim, not stated as established here.

**SynAgent's numbers still come from a model of unknown/unconfirmed
identity** — the `frozen` CSV predates this session and was not regenerated
for this table; it isn't confirmed to be exactly 1B-2M (or any other single
checkpoint) from Table 1, only that its structure (1,000 targets, one sample
each) matches Table 1's stated ChEMBL methodology.

## Reproduce

```
uv run python docs/chembl-benchmark/make_table1_synagent_column.py
uv run python docs/chembl-benchmark/run_repair_frozen.py --model qwen3.5:9b --provider ollama
# if any routes come back marked timeout/route_watchdog_timeout, retry them:
uv run python docs/chembl-benchmark/run_repair_frozen.py --model qwen3.5:9b --provider ollama \
    --retry-after timeout,route_watchdog_timeout --timeout 350 --route-timeout 1200
# only needed once, for CSVs produced before the apply_fixes building_blocks fix (footnote 6):
uv run python docs/chembl-benchmark/fix_corrected_bb_lists.py
uv run python docs/chembl-benchmark/make_table1_synagent_column.py \
    --repair-csv docs/chembl-benchmark/comparison-2026-08-27/frozen-repair-full.csv
```
