"""Price the two remaining gaps: the 10th placeholder, and the 3 off-RXN1
reactions. Strict first, analogue only where strict fails."""
import sys, csv, json
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]
PH = "json format error"

def route_json(tmpl, reactants, product):
    return json.dumps({"reactions": [{"reaction_number": 1, "reaction_template": tmpl,
                                      "reactants": reactants, "product": product}],
                       "building_blocks": sorted(set(reactants))})

# ---- layer 1: strict de novo for placeholders ----
strict = {}
for r in rows:
    if r["response"].strip() != PH: continue
    res = M._retro_disconnection_all_templates_sync(r["smiles"])
    if res and res.get("found"):
        strict[r["smiles"]] = route_json(res["template"], res["new_reactants"], r["smiles"])

# ---- layer 2: analogue for the placeholder strict missed ----
analog = {}
for r in rows:
    if r["response"].strip() != PH or r["smiles"] in strict: continue
    res = M._fix_via_product_analogue_retro_sync(r["smiles"], [], "")
    if res and res.get("found"):
        print(f"analogue for unreachable target:")
        print(f"  similarity      : {res.get('product_similarity')}")
        print(f"  matched_product : {str(res.get('matched_product'))[:70]}")
        print(f"  template in RXN1: {res['template'] in M._RXN1_TEMPLATE_SET}")
        # convention: declared product stays the ORIGINAL target
        analog[r["smiles"]] = route_json(res["template"], res["new_reactants"], r["smiles"])

# ---- layer 3: the 3 off-RXN1 reactions ----
OFF = {"COc1ccc(C2CCCN2C(S)=Nc2cccc(C)c2)cc1OC",
       "CCc1cccc(NC(=N)Nc2c(Cl)cccc2Cl)c1",
       "O=[N+]([O-])c1cccc(C=NNc2cnc3ccccc3n2)c1"}
tmpl_fix = {}
print("\noff-RXN1 reaction repair:")
for r in rows:
    if r["smiles"] not in OFF: continue
    out = json.loads(r["response"])
    changed = False
    for rxn in out["reactions"]:
        raw = str(rxn["reaction_template"])
        t = raw.split("<rxn>")[1].split("</rxn>")[0] if "<rxn>" in raw else raw
        if t in M._RXN1_TEMPLATE_SET: continue
        reactants = [str(x).split("<bb>")[-1].split("</bb>")[0] if "<bb>" in str(x) else str(x)
                     for x in rxn["reactants"]]
        prod = str(rxn["product"])
        got, how = None, None
        for nm, res in (("strict_all_templates", M._retro_disconnection_all_templates_sync(prod)),
                        ("analogue_retro", M._fix_via_product_analogue_retro_sync(prod, reactants, t))):
            if res and res.get("found") and res["template"] in M._RXN1_TEMPLATE_SET:
                got, how = res, nm
                break
        print(f"  {r['smiles'][:46]:<46} rxn{rxn['reaction_number']} -> {how or 'UNFIXABLE'}")
        if got:
            rxn["reaction_template"] = got["template"]
            rxn["reactants"] = got.get("new_reactants") or reactants
            changed = True
    if changed:
        seen, prods = [], {str(x["product"]) for x in out["reactions"]}
        for x in out["reactions"]:
            for q in x["reactants"]:
                if q not in seen: seen.append(q)
        out["building_blocks"] = [q for q in seen if q not in prods]
        tmpl_fix[r["smiles"]] = json.dumps(out)

KEYS = ["valid_json_percent", "template_mem_percent", "bb_selection_percent",
        "valid_smiles_percent", "matched_reactants_percent", "good_products_strict_percent"]
def score(overrides):
    return M.score_effective_dataset([(r["smiles"], overrides.get(r["smiles"], r["response"]))
                                      for r in rows])
variants = {
    "0_before": {},
    "A strict de novo": dict(strict),
    "B +analogue 10th": {**strict, **analog},
    "C +template fix": {**strict, **analog, **tmpl_fix},
}
scored = {k: score(v) for k, v in variants.items()}
print(f"\n{'metric':<30}" + "".join(f"{k:>18}" for k in scored))
for k in KEYS:
    print(f"{k:<30}" + "".join(f"{scored[v][k]:>18}" for v in scored))
print(f"\n{'good/matched counts':<30}" + "".join(
    f"{str(scored[v]['_n_products_strict'])+'/'+str(scored[v]['_n_successful_reactions']):>18}"
    for v in scored))
