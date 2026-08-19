"""Is reactant ORDER the reason the two validators disagree?

SynLlama's validator calls rxn.RunReactants(reactants) with the list in the
order the model emitted. RDKit matches reactants to template slots positionally,
so if the model lists (amine, acid) for a template written (acid, amine), the
reaction yields nothing and the path is scored a `reaction` failure -- even
though the chemistry is fine.

If that's the mechanism, simply reversing the list should flip the verdict.
"""

import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
csv.field_size_limit(10**7)

from run_bench import classify  # noqa: E402


def best_over_permutations(response: str) -> str:
    """Score a route allowing any ordering of each step's reactants."""
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        return "json"

    steps = data.get("reactions", [])
    orders = []
    for s in steps:
        rs = [r for r in s.get("reactants", []) if r != ""]
        orders.append(list(itertools.permutations(rs)) or [()])

    # Cap the search: routes with many multi-reactant steps explode.
    total = 1
    for o in orders:
        total *= len(o)
    if total > 64:
        return "too_many_permutations"

    for combo in itertools.product(*orders):
        trial = json.loads(response)
        for step, perm in zip(trial["reactions"], combo):
            step["reactants"] = list(perm)
        if classify(json.dumps(trial)) == "valid":
            return "valid"
    return "still_invalid"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    rows = []
    with (HERE / "synllama-raw-output.csv").open(encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            rows.append(row)

    strict = Counter()
    relaxed = Counter()
    rescued = 0

    for row in rows:
        s = classify(row["response"])
        strict[s] += 1
        if s == "valid":
            relaxed["valid"] += 1
            continue
        r = best_over_permutations(row["response"])
        relaxed[r] += 1
        if r == "valid":
            rescued += 1

    tot = len(rows)
    sv = strict["valid"]
    rv = relaxed["valid"]
    print(f"Sample: {tot} routes\n")
    print(f"  strict  (order as emitted) : {sv}/{tot} = {sv/tot*100:.2f}% valid")
    print(f"  relaxed (any reactant order): {rv}/{tot} = {rv/tot*100:.2f}% valid")
    print(f"\n  rescued purely by reordering: {rescued} ({rescued/tot*100:.1f}% of sample)")
    print(f"\n  strict failure buckets: "
          f"{ {k: v for k, v in strict.items() if k != 'valid'} }")
    print(f"  after reordering      : "
          f"{ {k: v for k, v in relaxed.items() if k != 'valid'} }")


if __name__ == "__main__":
    main()
