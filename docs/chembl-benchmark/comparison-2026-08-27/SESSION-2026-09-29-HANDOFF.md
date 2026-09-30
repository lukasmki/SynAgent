# Session handoff — 2026-09-29

Resume point for the corrector-waterfall work on the official SynLlama 1b-2m
output. Everything below is measured, not estimated.

## 1. The baseline reproduces SynLlama's published row exactly

`data/synllama-data/results/table_1_llm_benchmark_91rxns/synllama_reconstruct/llm_benchmark_stats.csv`
is the authors' **own** published statistics, shipped in their data release.
Our `0_before` column equals it to the decimal on all six metrics:

| metric | published | our `0_before` |
|---|---|---|
| Valid JSON | 99.0 | 99.0 |
| Template Mem | 99.82 | 99.82 |
| BB Selection | 99.47 | 99.47 |
| Valid SMILES | 95.23 | 95.23 |
| Matched Reactants | 70.93 | 70.93 |
| Good Products | 87.02 | 87.02 |

This is the most valuable asset in the comparison — it is the proof that
`REPORT.md`'s "scoring is provably identical" claim holds. **Do not perturb the
baseline.** Every gain must live in the corrected column.

Two further published metrics we do *not* yet compute: `total_success_formats`
(97.9) and `total_success_reactions` (56.9). Adding them would extend the match
from six metrics to eight.

## 2. The degeneracy finding (the important one)

`retro_disconnection_all_templates` searches all 91 templates for anything that
reaches the product and takes the **first** hit. The easiest hits barely change
the molecule, and a reactant 93% the size of the product will of course react to
give it — so Matched Reactants and Good Products rise without the chemistry
improving.

Audited against SynLlama's own 1,033 passing reactions as control:

| measure | baseline | tier-4 repairs (163) | tier-6 rescues (9) |
|---|---|---|---|
| single reactant (not a disconnection) | 20% | **44%** | 0% |
| a reactant >=0.85 Tanimoto to product | 3% | **13%** | 44% |
| a reactant >=90% of product heavy atoms | 22% | **59%** | 44% |
| median largest-reactant/product ratio | **0.70** | **0.93** | 0.88 |

This closes the item `REPORT.md:185` lists as not established ("whether the
corrections are chemically correct"). Unscreened, the answer was no.

With a 0.9 heavy-atom screen: **58% of the Good Products lift and 39% of the
Matched Reactants lift were artifact.**

## 3. Which screen to use — settled, but NOT yet measured

The 0.9 heavy-atom ratio is **too blunt and mis-fires.** Two of the ten
rescuable placeholder targets are ordinary acylations:

```
#2  78-heavy product  <-  75-heavy alcohol + CC(=O)O   ratio 0.96
#8  59-heavy product  <-  56-heavy alcohol + CC(=O)O   ratio 0.95
```

Acetylating a hydroxyl on a natural product is real late-stage chemistry, and
the baseline itself does >=0.9 steps in 22% of its reactions. The ratio screen
discards them purely because acetic acid is small.

**Decision taken: screen on `min_reactants=2` instead.** A "disconnection" that
consumes no reagent is a functional-group interconversion, not a synthesis
step, and single-reactant matches are the corrector's actual bad habit (44% vs
20%). Both criteria are implemented and independently switchable.

**This is the run that has not happened yet.** See §7.

## 4. Results so far (all on `data/synllama-official-1b2m.csv`)

Final corrected column, tier 6:

| metric | `0_before` | capped, no screen | uncapped, no screen | uncapped + ratio 0.9 |
|---|---|---|---|---|
| Valid JSON | 99.0 | 99.9 | 99.9 | 99.5 |
| Template Mem | 99.82 | 99.82 | 99.82 | 99.82 |
| BB Selection | 99.47 | 99.59 | 99.70 | 99.70 |
| Valid SMILES | 95.23 | 95.70 | 97.15 | 97.18 |
| Matched Reactants | 70.93 | 76.09 | 81.38 | 77.34 |
| Good Products | 87.02 | 90.70 | 91.08 | 88.74 |

Files: `ablation-denovo-rescue-official1b2m.json`,
`ablation-uncapped-noguard-official1b2m.json`,
`ablation-uncapped-guard90-official1b2m.json`.

`ablation-uncapped-guard90-v2-official1b2m.json` (ratio 0.9 **in-search**)
DID complete. It isolates the in-search fix from the criterion change:

| | Valid JSON | Matched Reactants | Good Products |
|---|---|---|---|
| ratio 0.9, screen OUTSIDE the tool | 99.5 | 77.34 | 88.74 |
| ratio 0.9, screen IN-SEARCH | **99.7** | **77.61** | **88.87** |

The +0.2 Valid JSON is exactly targets #7 and #9 recovering, as predicted.
The fix also pays across the waterfall (tier 4 76.56 -> 76.79, tier 5
77.27 -> 77.51), so moving the screen into the search is a real gain
independent of which criterion is chosen.

## 5. Two bugs found and fixed

**Cosine/Tanimoto threshold.** `_analogue_candidates` gates candidates on
`metric="cosine"` but `_fix_via_product_analogue_retro_sync` reported and
reasoned about **Tanimoto**, computed *after* acceptance and never re-checked.
A bisindole alkaloid matched a carbazole-dicyanobenzene at Tanimoto **0.067**
and was returned as a "similar, purchasable analog". Now re-checked in the
reported metric before accepting.

Consequence: this was the **only** thing that ever moved Template Memorization
(99.82 -> 99.94 in `ablation-reordered-partial-retention-official1b2m.json`).
With the threshold enforced that lift disappears. **Tiers 7-8 of that file are
superseded and should be regenerated or marked.**

**First-hit-wins in the reverse search.** The screen sat *outside* the tool, so
a degenerate first hit was rejected and the tool gave up, even when a genuine
disconnection sat further down the same fragment-set list. The criterion now
goes *into* `_retro_disconnection_all_templates_sync`, which holds a
non-disconnection aside as `fallback` and keeps scanning. Verified: targets #7
(`0.96 -> 0.85`) and #9 (`0.93 -> 0.88`) recover. With both criteria `None` the
behaviour is byte-identical to before.

## 6. Ceilings — what is and is not reachable

**Valid JSON: 99.9% is the ceiling, not 100%.** The 10 failures are not
malformed JSON — the `response` field is the literal 17-byte string
`json format error`, an upstream generation failure with no route to salvage
(so `fix_invalid_json.py`'s truncation salvage does not apply). Of the 10:

- 5 rescued on the first hit
- 2 (#7, #9) rescued once the screen moved into the search
- 2 (#2, #8) rescued only under `min_reactants` — the ratio screen rejects them
- 1 (#5, a vinblastine-type bisindole) **genuinely unreachable**

#5 is proven unreachable from both directions. Reverse: 171 fragment sets
across all 91 templates, zero forward matches, and the fragment cap (200) never
tripped. Forward: of 2,591 catalog building blocks in 30..61 heavy atoms, **zero**
are substructures of the target, best Tanimoto 0.2446, and the largest common
substructure is **15 atoms — 24% of a 62-atom target**, where a one-step
coupling needs a partner carrying ~54. The pieces are not purchasable.

**Template Mem: 100% is not reachable.** Three reactions, two distinct invented
templates, both single-atom interpolations of families that DO exist in RXN1:

- **A** (x2): `[N;$(N-[#6]):3]=[C;$(C=N):1].[N…:2]>>[N:3]-[C:1]-[N+0:2]` —
  guanidine formation. RXN1 has the `C=S` and `C=O` members, not `C=N`.
- **B** (x1): alcohol + all-carbon 5-ring amine. RXN1 has the N-rich ring
  variants, not the all-carbon one.

Targets: `COc1ccc(C2CCCN2C(S)=Nc2cccc(C)c2)cc1OC`,
`CCc1cccc(NC(=N)Nc2c(Cl)cccc2Cl)c1`,
`O=[N+]([O-])c1cccc(C=NNc2cnc3ccccc3n2)c1`.

**REFINED 2026-09-29 by exhaustive search (`session-diagnostics/` probe: all 91
forward on the declared reactants, all 91 in reverse unscreened, partial
retention, plus an RDKit `TautomerEnumerator` comparison).** The earlier
"two are unfixable by any tool" understated one of them:

- **Target 1** `COc1ccc(C2CCCN2C(S)=Nc2cccc(C)c2)cc1OC` — **RXN1 already has the
  reaction.** Isothiocyanate + pyrrolidine; RXN1's `C=S` member fires on these
  exact reactants and gives `...C(=S)N...` (thione), while the declared product
  is `...C(S)=N...` (thiol). `TautomerEnumerator` canonicalizes both to one
  form: same molecule. The model wrote the thiol tautomer, then invented a
  `C=N` template to match its own output. A **representation mismatch, not
  invented chemistry.**
- **Target 2** `CCc1cccc(NC(=N)Nc2c(Cl)cccc2Cl)c1` — genuinely out of set.
  Aniline + carbodiimide -> guanidine; no RXN1 template reaches the product or
  any tautomer of it. Unscreened retro-all finds only
  `['CCc1cccc(NC(=N)Nc2c(O)cccc2Cl)c1']`, a single-reactant Cl->OH
  interconversion — exactly what `min_reactants` rejects.
- **Target 3** `O=[N+]([O-])c1cccc(C=NNc2cnc3ccccc3n2)c1` — no, and the step is
  a **no-op**: the declared reactants are `[the target itself, OCCO]` and the
  declared product is the target. Nothing to swap.

So under the chosen screen Template Mem is **99.82% flat, 0 of 3 repairable**.
Recovering target 1 would need the product comparison to be
tautomer-insensitive rather than canonical-SMILES-exact. That is a change to
the SCORING RULE, which applies to the baseline column too — it would move the
authors' own 99.82 and forfeit §1's exact six-metric reproduction. Not worth
+0.06pp. Report the refinement in words instead.
The only other routes to 100% are expanding RXN1 (changing the ruler, since the
metric is defined as adherence to those 91) or dropping the reactions from the
denominator. Neither is honest. Report 99.82 as an upstream property.

## 7. DONE (2026-09-29, later session) — the min_reactants run

Ran as specified below. Output:
`ablation-uncapped-minreactants2-official1b2m.json`.
**Both advance predictions in this section held.**

| metric | `0_before` | min_reactants=2 (tier 6) |
|---|---|---|
| Valid JSON | 99.0 | **99.9** |
| Template Mem | 99.82 | 99.82 |
| BB Selection | 99.47 | 99.70 |
| Valid SMILES | 95.23 | 97.19 |
| Matched Reactants | 70.93 | **78.82** |
| Good Products | 87.02 | **89.51** |

- Valid JSON hit the predicted 99.9: `routes_with_a_reconstruction_attempt`
  went 420 -> 429, i.e. exactly 9 of the 10 unparseable rows rescued, only the
  bisindole (#5) left. The 2 acylations (#2, #8) the ratio screen rejected do
  come back, as predicted.
- MR/GP land **between** the screened and unscreened runs as predicted
  (77.34/88.74 < **78.82/89.51** < 81.38/91.08).
- Baseline untouched: `0_before` still reproduces the published row on all six.

### Tier-level: the screen hits tier 4 and leaves tier 5 alone

Matched Reactants delta contributed by each tier, across the three screens:

| screen | t3->t4 (retro-all) | t4->t5 (retention) |
|---|---|---|
| none | +6.04 | +0.78 |
| ratio 0.9 in-search | +2.51 | +0.72 |
| **min_reactants=2** | **+3.41** | **+0.90** |

Retro-all's contribution is what the screen cuts (-44% under min_reactants,
-58% under ratio 0.9). Retention's is flat across all three (spread 0.18),
which is structural, not luck: `_partial_reactant_retention_sync` returns
early on <2 reactants and returns `kept + [frag]`, preserving arity, so it can
never emit the single-reactant fix the screen targets. Retention is slightly
*higher* under a screen than without one (+0.90 vs +0.78) because retro-all
runs first in the cascade and leaves it more to work on.

**Good Products: retention is the larger contributor, not retro-all.**
t4->t5 adds +2.97 GP versus retro-all's +1.74. Note ΔGP > ΔMR at tier 5 — that
is expected, not an anomaly: GP's denominator is `successful_reactions`, and
retention repairs steps that already had matched reactants but the wrong
product (`wrong_product` arm), lifting the numerator without the denominator.
So the tool that is structurally immune to the degeneracy screen carries most
of the Good Products lift. That is the honest version of the headline.

### Original spec, as run


```bash
cd /pscratch/sd/s/stefani/SynAgent
.venv/bin/python docs/chembl-benchmark/ablate_corrector_tools.py \
    --source data/synllama-official-1b2m.csv \
    --max-len 100000 \
    --min-reactants 2 \
    --out docs/chembl-benchmark/comparison-2026-08-27/ablation-uncapped-minreactants2-official1b2m.json
```

Takes ~25 min single-job on a compute node. Expected: Valid JSON **99.9%**
(9 of 10 rescued, only the bisindole failing), and Matched Reactants / Good
Products **between** the screened `77.34 / 88.74` and the unscreened
`81.38 / 91.08`, because it stops discarding legitimate acylations across all
163 tier-4 repairs.

Then: `--max-len 100000 --min-reactants 2 --include-analog-tools` to replace the
superseded tiers 7-8 under the fixed similarity threshold.

## 7b. DONE — analog-only column (`--include-analog-tools`)

`ablation-analog-minreactants2-official1b2m.json`, same flags as §7 plus
`--include-analog-tools`. Reported as a SEPARATE column, explicitly **not**
part of the SynLlama comparison: these tools return routes to a *similar*
molecule, not the declared target, so their numbers are not comparable to the
70.93 baseline.

| metric | `0_before` | t6 strict | t7 analogue BB | t8 product-analogue |
|---|---|---|---|---|
| Valid JSON | 99.0 | 99.9 | 99.9 | 99.9 |
| Template Mem | 99.82 | 99.82 | **99.82** | **99.82** |
| BB Selection | 99.47 | 99.7 | 99.7 | 99.7 |
| Valid SMILES | 95.23 | 97.19 | 97.19 | 97.19 |
| Matched Reactants | 70.93 | 78.82 | 81.02 | **81.14** |
| Good Products | 87.02 | 89.51 | 87.08 | **86.95** |

**Template Mem stays 99.82 across all nine tiers.** The old 99.82 -> 99.94 lift
was entirely the cosine/Tanimoto bug (§5); with the threshold re-checked in the
reported metric it is gone. Tiers 7-8 of
`ablation-reordered-partial-retention-official1b2m.json` are **superseded by
this file**.

**The analog tools buy nothing.** MR +2.32, GP **-2.56** -- ending at 86.95,
*below the 87.02 baseline*. Mechanism:
`GP = n_products_strict / successful_reactions`; the analog tools admit
reactions whose reactants match but whose product is a similar molecule rather
than the declared target, inflating the denominator without the numerator. The
MR rise and the GP fall are the same event from two sides. So analog relaxation
degrades strict product correctness below doing nothing -- the strongest
argument yet for keeping these tools opt-in and out of the comparison.

**Caveat:** the run logs `Database was created with RDKit version 2026.03.3 but
installed version is 2026.03.6`. The building-block fingerprint DB predates the
installed RDKit. Touches only the analog similarity lookups, so no number in
tiers 0-6 or in the comparison is affected.

## 8. Other open items

- **`REPORT.md` is dated 2026-09-17** and describes none of this. §4's "not
  established" list needs the chemical-correctness item resolved, and §5's next
  steps are stale.
- **The overview deck** (claude.ai Slides artifact, currently Version 11) shows
  the **ratio-0.9** numbers. Its waterfall and quality-audit slides both move
  once §7 runs. Also corrected there this session: 8B-500k BB Selection read
  99.23%, the authors' file says **99.53%**.
- **`docs/SynAgent-Project-Status.pptx`** is a separate, older 21-slide deck
  (dated Aug 17). Slide 20 still lists chemical correctness as not established.
  Untouched.
- **Environment note:** `python-pptx` and `XlsxWriter` were installed into
  `.venv` this session while inspecting that .pptx.
- **`--max-len` gates only the corrector**, never the baseline, so raising it
  cannot move `0_before`. At the old default of 1400, 93 routes (lengths
  1406-2822) were never corrected and they held 149 of the 226 invalid SMILES
  in the dataset.
- `fix_smiles` repairs **syntax, not chemistry**: +8 matched reactions,
  **+0** good products (1032 -> 1032, exact integer counts). The Good Products
  dip at tier 1 is denominator dilution, not damage. Verified.
- **Reordering the `fix_one_reaction` cascade is a proven no-op.** Of 320 failing
  reactions in the `no_products`/`wrong_product` arm, 23 have more than one tool
  firing and **0** disagree on exactness — all four tools score 100%
  exact-when-they-fire. `TOOL_ORDER` only drives ablation-tier presentation;
  the real order is the `if` cascade.

- **MEASURED 2026-09-29: per-step repair BREAKS route connectivity.**
  `fix_one_reaction` patches each failing step independently, and
  `retro_disconnection_all_templates` discards that step's original reactants
  wholesale. If the step consumed an earlier step's product, that earlier step
  is orphaned. `session-diagnostics/connectivity.py`, full tier-6 waterfall,
  `min_reactants=2`, baseline as control:

  | | |
  |---|---|
  | multi-step parseable routes | 345 |
  | of which the corrector rewrote | 224 |
  | disconnected BEFORE repair (control) | **5** / 345 |
  | disconnected AFTER repair | 69 |
  | **newly introduced by repair** | **64 / 224 = 28.6%** |
  | healed by repair | **0** |

  SynLlama's own routes are essentially connected (5/345); the disconnection is
  ours. 91 orphaned steps, 5.44% of the 1672-reaction denominator. Repair never
  once improves connectivity.

  **Careful reading.** This does NOT make MR 78.82 arithmetically wrong, and
  probably does not inflate the *lift*: the usual pattern is step 1 makes X,
  step 2 consumes X and fails, retro-all rebuilds step 2 from scratch, X is
  orphaned -- but step 1 was passing before and still passes, so it sits in
  both numerators. What it means is that the repaired object is **no longer a
  synthesis route**: a valid final step plus vestigial steps making compounds
  nothing uses. Per-reaction metrics are structurally blind to this
  (`pct(successful_reactions, total_reactions)`,
  `pct(n_products_strict, successful_reactions)` -- every step validated in
  isolation).

  **Consequence for §1.** The two unmatched published metrics,
  `total_success_formats` (97.9) and `total_success_reactions` (56.9), are
  ROUTE-level. Adding them still extends the baseline match from six metrics to
  eight, but it will now cut both ways: our corrected column will look
  materially worse route-level than per-reaction, because a route with an
  orphaned step should not count as a success.

  **NOT established:** whether each orphan step is itself counted as a
  successful reaction (reasoned to be in both columns; not measured).
  **Next measurement:** route-level success -- all steps pass AND the route
  connects -- baseline vs corrected. That is both the honest headline and a
  direct reproduction of the authors' 56.9.
  `partial_reactant_retention` is the structurally safer path: it keeps k-1
  originals, so a kept upstream product preserves connectivity.

Diagnostic scripts for every number above are in `session-diagnostics/`.
