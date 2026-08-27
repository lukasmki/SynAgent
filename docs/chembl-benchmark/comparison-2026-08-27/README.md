# SynLlama vs SynAgent + SynLlama evidence package

This directory contains the reproducible data, code, figures, and case studies
for the August 27, 2026 ChEMBL comparison.

![Benchmark summary](synagent-vs-synllama-summary.png)

## Headline results

| Metric | Valid routes | Rate |
|---|---:|---:|
| SynLlama published: emitted order + exact product | 3,065 / 10,000 | 30.65% |
| SynLlama emitted order + analog product | 3,420 / 10,000 | 34.20% |
| SynAgent permutations + exact product | 5,919 / 10,000 | 59.19% |
| SynAgent permutations + analog product | 6,523 / 10,000 | 65.23% |

The same SynLlama outputs are used in all four rows. The first two rows preserve
emitted reactant order. The latter two use SynAgent's validator, which checks
reactant permutations. Analog matching uses radius-2, 4096-bit Morgan
fingerprints and accepts the closest product only when Tanimoto similarity is
strictly greater than 0.60.

## Agent correction pilot

The 50-route DeepSeek run is a genuine agent evaluation: each route was passed
through a Pydantic AI conversation, and the orchestrator chose the validation
and correction tools. Under SynAgent's permutation-plus-analog metric, 26/50
routes passed before correction and 39/50 passed after correction. Fourteen
changed fail-to-pass and one changed pass-to-fail, for a net gain of 13 routes
and an exact paired McNemar p-value of 0.00098.

![Three real correction wins](three-synagent-wins.png)

`winning-cases.json` contains the original and corrected route for every one of
the 15 routes that changed from fail to pass under SynAgent's exact-product
validator. The figure shows three representative cases and the actual tool
sequence recorded by the n=50 batch.

## Full pipeline screenshot

The uncropped UI evidence is
[`../screenshots/full-pipeline-with-repair.png`](../screenshots/full-pipeline-with-repair.png).
That conversation visibly contains:

1. `generate_molecules`
2. `retrosynthesis`
3. `validate_route`
4. `fix_building_blocks`
5. `fix_step`
6. `apply_fixes`

The route moved from two of three reaction steps passing to all three passing.
This is real agent execution, not a direct call to the tool functions.

## Reproduce

From the repository root:

```bash
python docs/chembl-benchmark/compare_synllama_synagent.py
python docs/chembl-benchmark/comparison-2026-08-27/make_comparison_figures.py
python docs/chembl-benchmark/comparison-2026-08-27/make_winning_cases.py
pytest -q
```

## Files

- `PI_REPORT.md` - concise interpretation and PI-safe language.
- `summary.json` - aggregate metrics and method labels.
- `baseline-10000-path-level.csv` - one row per published SynLlama path.
- `synagent-repair-n50-paired.csv` - paired outcomes for the correction pilot.
- `winning-cases.json` - original/corrected routes for exact-product wins.
- `synagent-vs-synllama-summary.{png,pdf}` - presentation figure.
- `three-synagent-wins.{png,pdf}` - case-study figure.

## Paper-readiness limits

This is a strong pilot and methods package, but not yet a complete paper result:

- The correction sample is only n=50 and excludes responses over 1,400
  characters.
- The supplied baseline says `sampling_params=frozen`; the exact generation
  parameters should be confirmed with the SynLlama authors.
- Product-step analog acceptance is a SynAgent extension. The SynLlama paper
  applies 4096-bit Morgan similarity to final reconstructed analogs versus the
  original target.
- A similarity threshold does not establish reaction feasibility, biological
  equivalence, selectivity, or safety.
- A larger stratified correction study and expert chemistry review are needed
  before peer-reviewed claims.

A practical next target is **at least 400 stratified failing routes**, which is
roughly sufficient for a +/-5 percentage-point 95% margin near a 50% outcome
rate. Because tool invocation was nondeterministic in the pilot, paper-quality
evaluation should also repeat each route across multiple agent runs or fixed
seeds.
