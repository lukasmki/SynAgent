"""The reverse search returns its FIRST forward match. When the screen rejects
that one, is there a non-degenerate fragment set further down the same list?"""
import sys, csv
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from synagent.corrector._toolset import _fix_template_sync, _try_parse_smarts, _rxn1_templates
from rdkit import Chem

def all_matches(product_smiles, cap=400):
    """Same loop, but collect every forward match instead of returning the first."""
    pm = Chem.MolFromSmiles(product_smiles)
    if pm is None: return []
    canon = Chem.CanonSmiles(product_smiles)
    n_p = pm.GetNumHeavyAtoms()
    seen, tried, hits = set(), 0, []
    for smarts in _rxn1_templates():
        retro = _try_parse_smarts(">>".join(smarts.split(">>")[::-1]))
        if retro is None: continue
        try: outs = retro.RunReactants((pm,))
        except Exception: continue
        for o in outs:
            frags, ok = [], True
            for m in o:
                try:
                    Chem.SanitizeMol(m); frags.append(Chem.MolToSmiles(m, canonical=True))
                except Exception: ok = False; break
            if not ok or not frags: continue
            key = tuple(sorted(frags))
            if key in seen: continue
            seen.add(key); tried += 1
            if tried > cap: return hits
            fwd = _fix_template_sync(frags, canon)
            if fwd.get("found"):
                big = max((Chem.MolFromSmiles(x).GetNumHeavyAtoms()
                           for x in frags if Chem.MolFromSmiles(x)), default=0)
                hits.append((frags, fwd["template"], big / max(n_p, 1)))
    return hits

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["sampling_params"] == "frozen" and r["response"].strip() == "json format error"]

print(f"{len(rows)} placeholder targets; listing every forward match per target\n")
recoverable = 0
for i, r in enumerate(rows, 1):
    hits = all_matches(r["smiles"])
    if not hits:
        print(f"{i:>2}. NO MATCH AT ALL                      {r['smiles'][:46]}")
        continue
    first = hits[0][2]
    best = min(h[2] for h in hits)
    verdict = "first hit already clean" if first < 0.9 else (
        "RECOVERABLE deeper" if best < 0.9 else "all matches degenerate")
    if first >= 0.9 and best < 0.9: recoverable += 1
    print(f"{i:>2}. matches={len(hits):>3}  first_ratio={first:.2f}  best_ratio={best:.2f}  "
          f"{verdict:<24} {r['smiles'][:40]}")
print(f"\n{recoverable} target(s) the screen rejects but where a clean disconnection exists deeper")
