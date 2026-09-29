"""Classify every Valid-JSON failure in the official 1b-2m output, using the
exact abort paths score_effective_dataset takes."""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
SOURCE = ROOT / "data" / "synllama-official-1b2m.csv"

def classify(target, resp):
    try:
        out = json.loads(resp)
    except Exception as e:
        return "json_parse_error", f"{type(e).__name__}: {e}"
    if not isinstance(out, dict):
        return "not_a_dict", type(out).__name__
    missing = [k for k in ("reactions", "building_blocks") if k not in out]
    if missing:
        return "missing_top_level_key", ",".join(missing)
    for i, rxn in enumerate(out["reactions"], 1):
        miss = [k for k in ("reaction_template", "reactants", "product") if k not in rxn]
        if miss:
            return "reaction_missing_key", f"rxn {i}: {','.join(miss)}"
    for i, rxn in enumerate(out["reactions"], 1):
        raw = str(rxn["reaction_template"])
        if ("<rxn>" in raw) != ("</rxn>" in raw):
            which = "<rxn> only" if "<rxn>" in raw else "</rxn> only"
            return "half_rxn_tag", f"rxn {i}: {which}"
    return "OK", ""

with SOURCE.open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

fails = []
for r in rows:
    kind, detail = classify(r["smiles"], r["response"])
    if kind != "OK":
        fails.append((kind, detail, r))

print(f"{len(rows)} frozen rows, {len(rows)-len(fails)} valid, {len(fails)} FAIL "
      f"({100*(len(rows)-len(fails))/len(rows):.2f}% valid)\n")
from collections import Counter
for k, n in Counter(k for k, _, _ in fails).most_common():
    print(f"  {n:>3}  {k}")
print()
for i, (kind, detail, r) in enumerate(fails, 1):
    resp = r["response"]
    print(f"--- {i}. {kind} | {detail}")
    print(f"    target: {r['smiles'][:80]}")
    print(f"    len(response)={len(resp)}  ends: ...{resp[-70:]!r}")
