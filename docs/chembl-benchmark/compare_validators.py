"""Do SynAgent's validator and SynLlama's validator agree?

The repair harness surfaced a route that SynLlama's validator (data/synllama-validate.py,
the one that produced the 3065/10000 baseline) marks FAILED, while SynAgent's
validate_route marks PASSED. If they disagree systematically then any
"corrector fixed X%" figure measured with SynAgent's own validator is not
comparable to the baseline, and would flatter SynAgent.

This quantifies the disagreement over a sample.
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
csv.field_size_limit(10**7)

from run_bench import classify  # noqa: E402  SynLlama's criterion, verbatim

from synagent.validation._toolset import _validate_route_dict  # noqa: E402


def synagent_verdict(response: str) -> str:
    """SynAgent's own validate_route verdict on the same route JSON."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return "json"
    try:
        rep = _validate_route_dict(data)
        d = rep if isinstance(rep, dict) else rep.model_dump()
        ok = d.get("all_building_blocks_valid") and d.get("all_reactions_passed")
        return "valid" if ok else "failed"
    except Exception as e:
        return f"error:{type(e).__name__}"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rows = []
    with (HERE / "synllama-raw-output.csv").open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            rows.append(row)

    agree = disagree = 0
    matrix = Counter()
    examples = {"syn_pass_lukas_fail": None, "syn_fail_lukas_pass": None}

    for row in rows:
        lukas = classify(row["response"])
        syn = synagent_verdict(row["response"])
        lukas_ok = lukas == "valid"
        syn_ok = syn == "valid"
        matrix[(lukas_ok, syn_ok)] += 1
        if lukas_ok == syn_ok:
            agree += 1
        else:
            disagree += 1
            key = "syn_pass_lukas_fail" if syn_ok else "syn_fail_lukas_pass"
            if examples[key] is None:
                examples[key] = {"target": row["smiles"],
                                 "lukas": lukas, "synagent": syn,
                                 "route": row["response"][:600]}

    tot = len(rows)
    print(f"Compared {tot} routes\n")
    print(f"  agree    : {agree}/{tot} ({agree/tot*100:.1f}%)")
    print(f"  disagree : {disagree}/{tot} ({disagree/tot*100:.1f}%)\n")
    print("  confusion (SynLlama-valid, SynAgent-valid) -> count")
    for (lk, sy), c in sorted(matrix.items(), key=lambda kv: -kv[1]):
        print(f"    SynLlama={'PASS' if lk else 'FAIL'}  "
              f"SynAgent={'PASS' if sy else 'FAIL'}  ->  {c:>4}")

    sp = matrix[(False, True)]
    print(f"\n  SynAgent passes {sp} routes SynLlama's validator rejects "
          f"({sp/tot*100:.1f}% of the sample).")
    print("  Those are routes a 'repair' would be credited for without the")
    print("  baseline validator ever agreeing they are valid.\n")

    for k, ex in examples.items():
        if ex:
            print("=" * 62)
            print(k)
            print("  target  :", ex["target"])
            print("  SynLlama:", ex["lukas"], "| SynAgent:", ex["synagent"])
            print("  route   :", ex["route"][:400])


if __name__ == "__main__":
    main()
