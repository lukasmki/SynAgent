"""How often does retro_disconnection_all_templates produce a DEGENERATE
'disconnection' -- one reactant that is essentially the product already --
rather than a real breakdown into simpler pieces?"""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as fg
gen = fg.GetMorganGenerator(radius=2, fpSize=2048)

def sim(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if ma is None or mb is None: return None
    return DataStructs.TanimotoSimilarity(gen.GetFingerprint(ma), gen.GetFingerprint(mb))

def heavy(s):
    m = Chem.MolFromSmiles(s)
    return m.GetNumHeavyAtoms() if m else 0

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

cases = []
# the 10 placeholder rescues (tier 6)
for r in rows:
    if r["response"].strip() == "json format error":
        cases.append(("tier6_rescue", r["smiles"]))
# every failing reaction retro_all_templates fires on (tiers 4-6)
for r in rows:
    if r["response"].strip() == "json format error": continue
    try:
        rep = M._validate_route_dict(M._parse_route_json(r["response"]), analog_product_threshold=None)
    except Exception:
        continue
    for rxn in rep.reactions:
        if rxn.status != "passed" and rxn.failure_mode in ("no_products", "wrong_product"):
            cases.append(("tier4_repair", rxn.expected_product))

stats = {"tier6_rescue": [], "tier4_repair": []}
for kind, prod in cases:
    res = M._retro_disconnection_all_templates_sync(prod)
    if not (res and res.get("found")): continue
    rs = res["new_reactants"]
    best = max((sim(x, prod) or 0) for x in rs)
    ratio = max(heavy(x) for x in rs) / max(heavy(prod), 1)
    stats[kind].append((len(rs), best, ratio))

for kind, vals in stats.items():
    if not vals: continue
    n = len(vals)
    one = sum(1 for v in vals if v[0] == 1)
    degen = sum(1 for v in vals if v[1] >= 0.85)
    big = sum(1 for v in vals if v[2] >= 0.9)
    print(f"{kind}: {n} repairs")
    print(f"   single-reactant 'disconnections' : {one} ({100*one/n:.0f}%)")
    print(f"   a reactant >=0.85 Tanimoto to product : {degen} ({100*degen/n:.0f}%)")
    print(f"   a reactant >=90% of product's heavy atoms : {big} ({100*big/n:.0f}%)")
    print(f"   median largest-reactant/product size ratio : "
          f"{sorted(v[2] for v in vals)[n//2]:.2f}")
