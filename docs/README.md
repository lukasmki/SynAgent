# docs — benchmark results, cluster scripts, and the status deck

Supporting material for the work on this branch. Nothing here is imported by
`synagent`; it is evidence and tooling.

| Path | What |
|---|---|
| `SynAgent-Project-Status.pptx` | 21-slide status deck covering all four workstreams |
| `chembl-benchmark/` | ChEMBL route-validity benchmark, agent evidence, figures |
| `lawrencium/` | SLURM scripts and configs for fine-tuning on an HPC cluster |

---

## The headline result

`data/synllama-validate.py` calls `rxn.RunReactants(reactants)` with the
reactant list exactly as the model emitted it. RDKit matches reactants to
template slots **positionally**, so a route listing `(amine, acid)` against a
template written `(acid, amine)` produces nothing and is scored a `reaction`
failure — even though the chemistry is correct.

Re-scoring the same 500 routes:

| Grading | Valid |
|---|---|
| Order as emitted (published criterion) | 141/500 = **28.20%** |
| Any reactant order | 285/500 = **57.00%** |
| Rescued purely by reordering | **144 routes** |

**SynLlama's route validity is likely ~57%, not the published 30.65%.** It also
explains the failure profile: `reaction produced no products` was the largest
bucket at 36.5%, and most of that is ordering rather than chemistry.

Reproduce with `chembl-benchmark/test_order.py`. The head-to-head validator
comparison that surfaced it is `compare_validators.py`.

## Other results

- **Route validity**, SynAgent driving `lukaskim/SynLlama-1B-GGUF` Q6_K:
  121/400 = 30.25%, against the 3,065/10,000 = 30.65% baseline. The quantised
  1B reproduction matches SynLlama proper.
- **Agent orchestration**: three real failing routes driven through the
  pydantic-ai UI with an external orchestrator. `validate_route` fired on all
  three; the full corrector chain (`fix_building_blocks` → `fix_step` →
  `apply_fixes`) on two. Screenshots in `chembl-benchmark/screenshots/`.
- **Scoring is provably comparable**: the validator here is copied verbatim
  from `data/synllama-validate.py` and reproduces its published split exactly
  (3065/10000 = 30.65%).

## Not established

- Whether the corrector's repairs are chemically correct. `apply_fixes`
  completes; nobody has verified the output is better chemistry
- Whether the validity comparison is sampling-matched — the baseline CSV has
  `sampling_params = frozen`, meaning unresolved
- **No model has been fine-tuned.** `lawrencium/` is staged and untested
  against a live scheduler

## Running the benchmark

The scripts expect the CSVs from this repo's `data/` directory copied
alongside them, and a local OpenAI-compatible server for the model calls
(`llama.cpp llama-server` in router mode — see `docs/lawrencium/README.md`
context, and note Ollama cannot serve this; it re-templates `/v1/completions`).

```bash
python run_bench.py --targets 100 --samples 10   # route validity
python test_order.py 500                         # the ordering finding
python compare_validators.py 500                 # validator head-to-head
python make_figures.py && python make_fig4.py    # regenerate figures
```
