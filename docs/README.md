# docs — benchmark results, cluster scripts, and the status deck

Supporting material for the work on this branch. Nothing here is imported by
`synagent`; it is evidence and tooling.

| Path | What |
|---|---|
| `SynAgent-Project-Status.pptx` | 21-slide status deck covering all four workstreams |
| `SynAgent-CADD-Lawrencium-Updated-Project-Report.pdf` | Updated 12-page report covering the pipeline, benchmarks, correction study, dataset findings, and completed Lawrencium smoke run |
| `PI_SYNAGENT_SYNLLAMA_AND_FINETUNING_REPORT.{md,docx,pdf}` | August 30 PI report: every evidence image, the full 10,000-path comparison, genuine agent corrections, and live 1M-row QLoRA status |
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

The earlier 500-route diagnostic suggested ~57%. The completed 10,000-route
analysis now measures **59.19% exact-product validity** when reactant
permutations are allowed, versus 30.65% under the emitted-order rule. This is a
validator result on unchanged SynLlama outputs, not a model-accuracy gain.

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
- **Corrector evaluation**, DeepSeek orchestrator over 50 failing routes:
  corrector tools fired on 48/50 (96%) and 7/50 (14%) became strictly valid.
- **Lawrencium smoke training completed**: a 50-step Llama-3.1-8B QLoRA run
  finished successfully on one NVIDIA A40 and saved a 168 MB adapter.
- **Real 1M-row QLoRA is running**: Lawrencium job `25306635` reached
  9,622/12,090 steps (79.6%) on four A40 GPUs at the August 30 report snapshot.
  Held-out loss decreased from 0.10089 at step 500 to 0.05064 at step 9,500;
  checkpoints are recoverable every 500 steps.
- **Paper evidence package**: the complete 10,000-path comparison, paired n=50
  correction analysis, machine-readable CSV/JSON, figures, workflow, and three
  real correction wins are in
  `chembl-benchmark/comparison-2026-08-27/`.

## Not established

- Whether the corrector's repairs are consistently better chemistry beyond
  the 7/50 routes that passed the same strict validator after repair
- Whether the validity comparison is sampling-matched — the baseline CSV has
  `sampling_params = frozen`, meaning unresolved
- Final generation quality of the 1M-row adapter; the one-epoch QLoRA run was
  still active at the August 30 report snapshot and requires held-out
  generation evaluation after completion

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
