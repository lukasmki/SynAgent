"""For the 10 placeholder targets the corrector currently skips: can the
STRICT tools build a verified route from the bare target SMILES alone?
retro_disconnection_all_templates(product) searches all 91 RXN1 templates in
reverse and needs nothing but the product, so it is the one tool that applies."""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["sampling_params"] == "frozen"
            and r["response"].strip() == "json format error"]
print(f"{len(rows)} placeholder targets\n")

ok = 0
for i, r in enumerate(rows, 1):
    tgt = r["smiles"]
    res = M._retro_disconnection_all_templates_sync(tgt)
    found = bool(res and res.get("found"))
    line = f"{i:>2}. found={found}"
    if found:
        tmpl = res["template"]
        reactants = res["new_reactants"]
        in_rxn1 = tmpl in M._RXN1_TEMPLATE_SET
        prods = M._arrange_and_react(tmpl, [x for x in reactants if Chem.MolFromSmiles(x)])
        exact = False
        if prods:
            pm = Chem.MolFromSmiles(tgt)
            exact = pm is not None and Chem.MolToSmiles(pm) in {
                Chem.MolToSmiles(p) for p in prods if p}
        line += f"  in_RXN1={in_rxn1}  reproduces_target={exact}  n_reactants={len(reactants)}"
        ok += exact and in_rxn1
    line += f"   {tgt[:60]}"
    print(line, flush=True)

print(f"\n{ok}/{len(rows)} targets get a strict, RDKit-verified one-step route")
print(f"-> Valid JSON after correction would be "
      f"{(990+ok)/1000*100:.1f}%  (baseline stays 99.0%)")
