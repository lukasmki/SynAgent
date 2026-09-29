"""Price the single strictly-repairable off-RXN1 reaction. Its route is NOT in
needs_fix (it passes validation), so the corrector never touches it in any
tier -- the swap is disjoint from all other corrector activity and its count
deltas carry over to tier 6 unchanged."""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M

TGT = "CCc1cccc(NC(=N)Nc2c(Cl)cccc2Cl)c1"
with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

swap = {}
for r in rows:
    if r["smiles"] != TGT: continue
    out = json.loads(r["response"])
    for rxn in out["reactions"]:
        raw = str(rxn["reaction_template"])
        t = raw.split("<rxn>")[1].split("</rxn>")[0] if "<rxn>" in raw else raw
        if t in M._RXN1_TEMPLATE_SET: continue
        res = M._retro_disconnection_all_templates_sync(str(rxn["product"]))
        print(f"repair found={res.get('found')} in_RXN1={res['template'] in M._RXN1_TEMPLATE_SET}")
        print(f"  old reactants: {rxn['reactants']}")
        print(f"  new reactants: {res['new_reactants']}")
        rxn["reaction_template"] = res["template"]
        rxn["reactants"] = res["new_reactants"]
    seen, prods = [], {str(x["product"]) for x in out["reactions"]}
    for x in out["reactions"]:
        for q in x["reactants"]:
            if q not in seen: seen.append(q)
    out["building_blocks"] = [q for q in seen if q not in prods]
    swap[TGT] = json.dumps(out)

a = M.score_effective_dataset([(r["smiles"], r["response"]) for r in rows])
b = M.score_effective_dataset([(r["smiles"], swap.get(r["smiles"], r["response"])) for r in rows])
print(f"\n{'':<32}{'baseline':>12}{'+swap':>12}{'delta':>10}")
for k in ["valid_json_percent","template_mem_percent","bb_selection_percent",
          "valid_smiles_percent","matched_reactants_percent","good_products_strict_percent"]:
    print(f"{k:<32}{a[k]:>12}{b[k]:>12}{b[k]-a[k]:>+10.2f}")
for k in ["_n_total_reactions","_n_successful_reactions","_n_products_strict"]:
    print(f"{k:<32}{a[k]:>12}{b[k]:>12}{b[k]-a[k]:>+10d}")
