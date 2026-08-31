# SynAgent, SynLlama, and Fine-Tuning Status Report

**Prepared for:** THG Lab project discussion  
**Updated:** August 31, 2026<br>
**Repository scope:** `lukasmki/SynAgent`, branch `synagent-full-pipeline`

## Executive summary

Three connected workstreams are now operational:

1. **SynAgent pipeline:** a real Pydantic AI agent has called the complete
   chemistry workflow: molecule generation, retrosynthesis, validation,
   building-block correction, reaction-step correction, and application of
   fixes.
2. **SynLlama comparison:** all 10,000 supplied SynLlama pathways for 1,000
   ChEMBL targets were rescored under four clearly separated validation rules.
   The same generated pathways score 30.65% under the published strict rule and
   65.23% under SynAgent's reactant-order-aware, analog-aware validator.
3. **Fine-tuning infrastructure:** the Lawrencium environment, 1M-row dataset,
   base model cache, SLURM scripts, 50-step QLoRA smoke test, and 200-step
   four-A40 full-parameter timing run have completed successfully. A separate
   one-epoch 1M-row QLoRA run completed successfully as job `25306635` on four
   A40 GPUs.

The 65.23% result is a **validation-method improvement on unchanged SynLlama
outputs**, not evidence that SynLlama itself generated better chemistry. The
real correction pilot is n=50 and improved SynAgent's paired analog-aware pass
rate from 52% to 78%; it is promising pilot evidence, not yet a paper-scale
generalization claim.

## 1. End-to-end SynAgent workflow

The implemented workflow is:

```text
User request
  -> generate_molecules (SmileyLlama)
  -> retrosynthesis (SynLlama)
  -> validate_route (RDKit reaction execution)
  -> fix_building_blocks
  -> fix_step
  -> apply_fixes
  -> corrected route and structured validation result
```

The orchestrating model selects tools through Pydantic AI. Tool execution was
verified from structural `ToolCallPart` and `ToolReturnPart` records, not merely
from printed text. The uncropped conversation in
`chembl-benchmark/screenshots/full-pipeline-with-repair.png` shows all six tools
and a route moving from two of three valid steps to three of three valid steps.

## 2. Product-validation correction

The product tool previously required exact canonical-SMILES equality. It now:

1. checks exact canonical-SMILES equality;
2. computes radius-2, 4096-bit Morgan fingerprints when exact matching fails;
3. selects the generated product with the highest Tanimoto similarity;
4. approves an analog only when similarity is strictly greater than 0.60; and
5. records the match type, matched product, and similarity in the structured
   result.

Exact-only scoring remains available. The SynLlama paper also uses 4096-bit
Morgan similarity, but it compares the final reconstructed analog with the
target. Applying the threshold to an individual reaction-step product is a new
SynAgent heuristic and must be labeled as such.

## 3. Complete 10,000-path comparison

| Scoring system | Valid routes | Rate |
|---|---:|---:|
| SynLlama: emitted reactant order + exact product | 3,065 / 10,000 | 30.65% |
| SynLlama: emitted order + analog-aware product | 3,420 / 10,000 | 34.20% |
| SynAgent: reactant permutations + exact product | 5,919 / 10,000 | 59.19% |
| SynAgent: permutations + analog-aware product | 6,523 / 10,000 | 65.23% |

The largest difference comes from reactant ordering. RDKit reaction templates
match reactants positionally; SynLlama can emit chemically compatible reactants
in a different order. SynAgent tries compatible permutations before rejecting
the route. Analog matching adds a second, smaller gain.

## 4. Genuine agent-correction pilot

The paired n=50 experiment used real DeepSeek-orchestrated Pydantic AI
conversations. Routes were drawn from the strict-failure pool with seed 42 and
limited to responses no longer than 1,400 characters.

| Metric | Before | After | Net change |
|---|---:|---:|---:|
| SynLlama strict | 0% | 14% | +7 routes |
| SynLlama analog-aware | 2% | 14% | +6 routes |
| SynAgent strict | 44% | 74% | +15 routes |
| SynAgent analog-aware | 52% | 78% | +13 routes / +26 points |

For the primary SynAgent analog-aware comparison, 14 routes changed from fail
to pass, one changed from pass to fail, 25 passed both, and 10 failed both. The
exact paired McNemar p-value is 0.00098. Tool invocation occurred in 48 of 50
runs after retry logic, and seven of the original strict failures became strict
passes. Agent behavior was observed to be nondeterministic, so a paper-quality
study should use fixed settings and repeated trials.

## 5. Image and evidence guide

### Primary PI figures

- `chembl-benchmark/comparison-2026-08-27/synagent-vs-synllama-summary.png`
  presents the four full-dataset rates and the paired n=50 correction result.
- `chembl-benchmark/comparison-2026-08-27/three-synagent-wins.png` presents
  three fail-to-pass routes from the real correction batch. The underlying 15
  exact-product wins are stored in `winning-cases.json`.
- `chembl-benchmark/screenshots/full-pipeline-with-repair.png` is the strongest
  visual proof of actual agent execution because it contains the uncropped
  six-tool conversation and corrected result.

### Pipeline sequence screenshots

- `1_generate.png`: SmileyLlama molecule generation tool call.
- `2_route.png`: SynLlama retrosynthesis tool call and returned pathway.
- `3_validate.png`: deterministic route validation.
- `4_fix.png`: correction-tool invocation.
- `5_full_conversation.png`: conversation overview.
- `full-pipeline-with-repair.png`: complete uncropped evidence.

### Diagnostic figures and examples

- `fig4_ordering.png`: demonstrates the emitted-order versus
  permutation-aware validation finding.
- `fig2_errors.png`: summarizes failure categories rather than hiding negative
  outcomes.
- `fig3_agent_tools.png`: summarizes tool invocation and correction behavior.
- `product_1_validate.png` and `product_2_correct.png`: product-failure example
  before and after correction.
- `reactant_1_validate.png` and `reactant_2_correct.png`: reactant-failure
  example before and after correction.
- `smiles_1_validate.png` and `smiles_2_correct.png`: malformed-SMILES example
  before and after correction.

`fig1_validity.png` is retained as a historical artifact but should not be used
for reporting; its original denominator logic rendered an incorrect 100% bar.
The corrected PI figure is `synagent-vs-synllama-summary.png`.

## 6. Fine-tuning work completed

### Infrastructure and data

- Lawrencium login, modules, Python environment, CUDA/PyTorch, Axolotl, and
  SLURM submission were validated.
- The staged training subset contains exactly 1,000,000 rows (about 1.5 GB).
- The 8B Llama 3.1 Instruct base-model cache is stored on scratch, not the login
  node.
- The QLoRA smoke job completed 50/50 steps with exit code 0 and produced a
  168 MB adapter.
- The four-A40 full-parameter timing job completed 200/200 steps with exit code
  0 and produced seven complete model shards totaling 32.1 GB.

The timing run covered about 6,400 examples, or 1.65% of one 1M-row epoch. Its
average loss moved from 0.600 over the first 20 steps to 0.085 over the final
20. This proves optimization and checkpoint writing worked, but it does not
prove held-out chemistry quality.

### Measured full-parameter throughput

- 0.5 rows/second, or approximately 1,733 rows/hour;
- 1M rows: approximately 577 hours (24 days);
- 2M rows: approximately 48 days; and
- all 5.86M rows: approximately 141 days.

Full-parameter training is bottlenecked by FSDP CPU offload. For that reason,
the real 1M-row experiment uses QLoRA rather than spending approximately 24 days
on a single full-parameter epoch.

## 7. Completed 1M-row QLoRA run

The production-scale pilot completed as Lawrencium job `25306635` with:

- one node and four NVIDIA A40 GPUs;
- one epoch over 1,000,000 rows;
- sequence length 2,048 and sample packing;
- LoRA rank 16, alpha 32, dropout 0.05;
- effective global batch size 32;
- 2,000-example validation split;
- evaluation and checkpoints every 500 steps; and
- a separate output directory:
  `/global/scratch/users/asanil/runs/qlora_1M_20260827`.

The run does not overwrite the smoke adapter or timing model. Its SLURM log is
`/global/home/users/asanil/lawrencium/qlora_1m_25306635.out`.

### Verified final status on August 31

- SLURM state: **COMPLETED**, exit code **0:0**;
- progress: **12,090 / 12,090 steps**, epoch 1.0;
- elapsed allocation time: **2 days, 22 hours, 3 minutes, 47 seconds**;
- training runtime: 251,233 seconds, with 3.953 samples/second and 0.048
  optimizer steps/second;
- final reported training loss: **0.06389**;
- held-out evaluation loss: 0.10089 at step 500, decreasing monotonically to
  **0.04937 at step 12,000**; and
- final adapter: `adapter_model.bin`, approximately 168 MB, with checkpoints at
  steps 11,500, 12,000, and 12,090.

The output directory is approximately 953 MB including the three retained
checkpoints. The error scan found no traceback, CUDA out-of-memory event, NCCL
failure, or runtime error. The monotonic validation-loss decline is strong
evidence that optimization worked, but final chemistry quality still requires
a generation-based comparison with the untouched base model on data excluded
before training. The branch now includes a one-A40 reproducible comparison job
and a step-by-step evaluation guide in `docs/lawrencium/`.

## 8. Claims appropriate for presentation

> On the complete 10,000-route ChEMBL output set, SynAgent's
> reactant-order-aware and analog-aware validator accepts 65.23% of routes,
> compared with 30.65% under the strict emitted-order SynLlama rule. This is an
> evaluation and correction improvement on the same generations. In a genuine
> n=50 agent-correction pilot, SynAgent's analog-aware paired pass rate improved
> from 52% to 78%.

Do not claim that the SynLlama model became 34.58 percentage points more
accurate, that a Tanimoto score proves synthetic feasibility, or that the n=50
repair rate generalizes to every failed route.

## 9. Work needed for a paper-quality result

1. Evaluate at least 400 stratified failed routes, including product,
   reactant, reaction, invalid-SMILES, and route-length strata.
2. Repeat correction trials because tool selection was nondeterministic.
3. Compare the final QLoRA checkpoint against the untouched base model on a
   held-out set using valid-SMILES rate, instruction following, property error,
   route validity, and scaffold diversity.
4. Have a synthetic chemist blindly review a representative subset.
5. Report negative outcomes, parser failures, runtime, and hosted-model cost.
6. Confirm the original SynLlama generation settings before claiming a direct
   model comparison.

## Reproducibility

The branch contains the full 10,000-row path-level CSV, paired n=50 CSV,
machine-readable summary, 15 winning cases, analysis scripts, tests, figures,
and screenshots. The implementation suite passed 33 tests with 2 live tests
skipped. No API keys are stored in the repository.

Key artifacts for review:

- `docs/chembl-benchmark/comparison-2026-08-27/PI_REPORT.md`: concise
  interpretation and presentation-safe language;
- `baseline-10000-path-level.csv`: one scored row for every supplied pathway;
- `synagent-repair-n50-paired.csv`: paired before/after correction outcomes;
- `winning-cases.json`: all 15 exact-product fail-to-pass cases;
- `compare_synllama_synagent.py`: complete reproducible scoring procedure;
- `docs/chembl-benchmark/screenshots/`: genuine pipeline and diagnostic UI
  evidence; and
- `docs/chembl-benchmark/comparison-2026-08-27/`: final figures in PNG and PDF
  formats.

Scientific context:

- SynLlama paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC12047903/
- ACS publication: https://pubs.acs.org/doi/10.1021/acscentsci.5c01285

The QLoRA adapter and completed SLURM accounting are verified. The remaining
paper-critical item is generation-based evaluation on a genuinely held-out,
source-stratified test set.
