# SynLlama vs SynAgent + SynLlama

**Date:** 2026-08-27  
**Data:** 10,000 published SynLlama paths for 1,000 ChEMBL targets; paired
corrector subset of 50 known strict failures.  
**Code status:** 31 tests passed, 2 live tests skipped. Nothing in this work has
been pushed yet.

## Executive result

The comparison separates model output quality from validation and correction:

| System / rule | Valid | Route-level rate |
|---|---:|---:|
| SynLlama published rule: emitted reactant order, exact product | 3,065 / 10,000 | **30.65%** |
| SynLlama with analog-aware products only | 3,420 / 10,000 | **34.20%** |
| SynAgent validator: any reactant order, exact product | 5,919 / 10,000 | **59.19%** |
| SynAgent validator: any reactant order, analog-aware product | 6,523 / 10,000 | **65.23%** |

The main full-dataset lift is therefore not a better SynLlama generation model.
It is SynAgent interpreting the same outputs more robustly: trying compatible
reactant permutations and accepting a clearly labeled close product analog.

## Product-tool correction

Both `validate_route` and `validate_products` previously required exact
canonical-SMILES equality. They now share one matcher:

1. Check exact canonical-SMILES equality first.
2. If exact matching fails, compute radius-2, 4096-bit Morgan fingerprints.
3. Select the generated product with maximum Tanimoto similarity.
4. Approve it as an analog only when similarity is **strictly greater than
   0.60**.
5. Record `product_match_type`, `matched_product`, and `product_similarity` in
   the structured `ReactionResult`.
6. Exact-only scoring remains available by passing a null threshold.

The 4096-bit setting matches the SynLlama paper. However, the paper compares a
fully reconstructed analog with the original target molecule. Applying the
threshold to an individual reaction-step product is a new SynAgent heuristic;
it should not be presented as the paper's original benchmark rule.

## Paired correction result

The existing DeepSeek corrector experiment contains 50 routes sampled from the
strictly failed pool using seed 42 and a maximum response length of 1,400
characters. Reconstructing the same original sample gives:

| Metric | Before correction | After correction | Net change |
|---|---:|---:|---:|
| SynLlama strict | 0 / 50 (0%) | 7 / 50 (14%) | +7 |
| SynLlama analog-aware | 1 / 50 (2%) | 7 / 50 (14%) | +6 |
| SynAgent strict | 22 / 50 (44%) | 37 / 50 (74%) | +15 |
| SynAgent analog-aware | 26 / 50 (52%) | 39 / 50 (78%) | **+13 net** |

For the primary SynAgent analog-aware comparison, 14 routes changed from fail
to pass, one changed from pass to fail, 25 passed both before and after, and 10
failed both. The exact paired McNemar p-value is **0.00098**. This supports a
real correction effect on this selected sample, but the sample is small and
length-filtered, so it is not yet an unbiased estimate across all 6,935
published failures.

## What can be said to the PI

> On the complete 10,000-route ChEMBL output set, SynAgent's analog-aware and
> reactant-order-aware validation accepts 65.23% of routes, compared with
> 30.65% under the published strict SynLlama rule. Most of that gap comes from
> fixing evaluation behavior rather than changing model generations. In a
> paired 50-route correction study, SynAgent then improved its own analog-aware
> pass rate from 52% before correction to 78% after correction, a net gain of 13
> routes (+26 percentage points; exact paired p=0.00098).

## What should not be claimed

- Do not say the SynLlama model became 34.58 percentage points more accurate.
  The model outputs are unchanged; the validation rules changed.
- Do not present 65.23% as the SynLlama paper's reconstruction rate. It is a
  route-validation result on the supplied 10,000 outputs.
- Do not claim the 78% repaired pass rate generalizes to all failures. The 50
  routes were length-filtered and selected from failures only.
- A Morgan similarity above 0.60 establishes structural similarity, not reaction
  feasibility, biological equivalence, selectivity, or safety.

## What is needed for a paper-scale result

1. Expand the paired correction experiment to at least **400 stratified failing
   routes**. At a proportion near 50%, n=400 gives an approximate 95% margin of
   error of +/-5 percentage points before finite-population correction.
2. Stratify by `no_products`, `wrong_product`, reactant mismatch, invalid
   SMILES, route length, and target scaffold rather than filtering only by
   response length.
3. Repeat the agent correction with multiple deterministic seeds or at least
   three independent runs per route, because tool invocation was observed to be
   nondeterministic.
4. Confirm SynLlama's `sampling_params=frozen` configuration and rerun model
   generation with matched temperature, TopP, model checkpoint, and number of
   samples.
5. Have a synthetic chemist review a blinded subset of exact and analog passes;
   validator success alone is not experimental feasibility.
6. Report runtime, hosted-model cost, parser failures, and negative outcomes,
   not only successful examples.

## Reproducibility files

- `summary.json`: machine-readable aggregate metrics.
- `baseline-10000-path-level.csv`: all four outcomes for every published path.
- `synagent-repair-n50-paired.csv`: paired before/after outcomes.
- `compare_synllama_synagent.py`: complete scoring and sample reconstruction.
- `synagent-vs-synllama-summary.png`: PI-ready figure.

## Paper alignment

The SynLlama paper evaluates 1,000 unseen ChEMBL molecules and reports analog
quality using Tanimoto similarity on 4096-bit Morgan fingerprints. It also
describes the model as a retrosynthesis component followed by a reconstruction
workflow using building blocks and reaction templates. These details motivate
the fingerprint configuration here, but the product-step threshold remains a
SynAgent-specific extension.

- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12047903/
- ACS publication: https://pubs.acs.org/doi/10.1021/acscentsci.5c01285
