import sys, csv, json
from pathlib import Path
from collections import Counter
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M

SRC = ROOT / "data"
# 1. provenance of the placeholder across every CSV we have
print("=== 'json format error' placeholder, per source CSV ===")
for f in sorted(SRC.glob("synllama-*.csv")):
    try:
        with f.open(encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            if "response" not in (rd.fieldnames or []):
                print(f"  {f.name:<44} (no response column)"); continue
            rows = list(rd)
    except Exception as e:
        print(f"  {f.name:<44} ERROR {e}"); continue
    n_ph = sum(1 for r in rows if r["response"].strip() == "json format error")
    print(f"  {f.name:<44} rows={len(rows):>5}  placeholder={n_ph}")

# 2. which templates are outside RXN1, baseline vs fully-corrected
print("\n=== Template Memorization failures ===")
with (SRC / "synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

def offenders(pairs):
    out = []
    for target, resp in pairs:
        try:
            o = json.loads(resp)
        except Exception:
            continue
        if not isinstance(o, dict) or "reactions" not in o or "building_blocks" not in o:
            continue
        for i, rxn in enumerate(o.get("reactions", []), 1):
            if not all(k in rxn for k in ("reaction_template", "reactants", "product")):
                break
            raw = str(rxn["reaction_template"])
            ho, hc = "<rxn>" in raw, "</rxn>" in raw
            if ho and hc:
                t = raw.split("<rxn>")[1].split("</rxn>")[0]
            elif not ho and not hc:
                t = raw
            else:
                break
            if t not in M._RXN1_TEMPLATE_SET:
                out.append((target, i, t))
    return out

base = offenders([(r["smiles"], r["response"]) for r in rows])
print(f"baseline: {len(base)} reactions carry a template outside RXN1")
for t, i, tmpl in base:
    print(f"  target {t[:56]}")
    print(f"    rxn {i}  len={len(tmpl)}  has '>>': {'>>' in tmpl}")
    print(f"    template: {tmpl[:150]}")
    print(f"    parses as SMARTS: ", end="")
    try:
        from rdkit import Chem
        rx = Chem.rdChemReactions.ReactionFromSmarts(tmpl)
        print("yes" if rx is not None else "no")
    except Exception as e:
        print(f"no ({type(e).__name__})")
