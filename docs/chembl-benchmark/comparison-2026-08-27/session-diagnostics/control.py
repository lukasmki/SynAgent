"""Control: what do SynLlama's OWN passing reactions look like on the same
measures? Without this, 0.93 is a number with nothing to compare against."""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as fg
gen = fg.GetMorganGenerator(radius=2, fpSize=2048)

def heavy(s):
    m = Chem.MolFromSmiles(s); return m.GetNumHeavyAtoms() if m else 0
def sim(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    return DataStructs.TanimotoSimilarity(gen.GetFingerprint(ma), gen.GetFingerprint(mb)) \
        if ma and mb else 0.0

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

vals = []
for r in rows:
    if r["response"].strip() == "json format error": continue
    try:
        rep = M._validate_route_dict(M._parse_route_json(r["response"]), analog_product_threshold=None)
    except Exception:
        continue
    for rxn in rep.reactions:
        if rxn.status != "passed": continue
        rs = [x for x in rxn.reactant_smiles if Chem.MolFromSmiles(x)]
        if not rs: continue
        p = rxn.expected_product
        vals.append((len(rs), max(sim(x, p) for x in rs), max(heavy(x) for x in rs)/max(heavy(p),1)))

n = len(vals)
one  = sum(1 for v in vals if v[0] == 1)
degen= sum(1 for v in vals if v[1] >= 0.85)
big  = sum(1 for v in vals if v[2] >= 0.9)
print(f"BASELINE (SynLlama's own passing reactions): {n}")
print(f"   single-reactant                           : {one} ({100*one/n:.0f}%)")
print(f"   a reactant >=0.85 Tanimoto to product     : {degen} ({100*degen/n:.0f}%)")
print(f"   a reactant >=90% of product's heavy atoms : {big} ({100*big/n:.0f}%)")
print(f"   median largest-reactant/product ratio     : {sorted(v[2] for v in vals)[n//2]:.2f}")
