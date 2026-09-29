"""Where do the invalid SMILES live, which are reachable by fix_smiles's
current dispatch, and how many would canonicalise if we did reach them?"""
import sys, csv, json, asyncio
from pathlib import Path
from collections import Counter
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem

async def main():
    with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    total_mol = 0; invalid = []          # (smiles, failure_mode, role)
    for r in rows:
        try:
            out = json.loads(r["response"])
            rep = M._validate_route_dict(M._parse_route_json(r["response"]), analog_product_threshold=None)
        except Exception:
            continue
        modes = {x.reaction_number: (x.status, x.failure_mode) for x in rep.reactions}
        for rxn in out.get("reactions", []):
            if not all(k in rxn for k in ("reaction_template","reactants","product")): break
            raw = str(rxn["reaction_template"])
            ho, hc = "<rxn>" in raw, "</rxn>" in raw
            if ho and hc: t = raw.split("<rxn>")[1].split("</rxn>")[0]
            elif not ho and not hc: t = raw
            else: break
            if t not in M._RXN1_TEMPLATE_SET: continue
            st, fm = modes.get(rxn.get("reaction_number"), (None, None))
            rs = [str(x).split("<bb>")[-1].split("</bb>")[0] if "<bb>" in str(x) else str(x)
                  for x in rxn["reactants"]]
            for x in rs:
                if x == "": continue
                total_mol += 1
                if Chem.MolFromSmiles(x) is None: invalid.append((x, fm, "reactant"))
            p = str(rxn["product"]); total_mol += 1
            if Chem.MolFromSmiles(p) is None: invalid.append((p, fm, "product"))

    print(f"molecules counted: {total_mol}   invalid: {len(invalid)} "
          f"({100*len(invalid)/total_mol:.2f}%)  -> valid_smiles "
          f"{100*(total_mol-len(invalid))/total_mol:.2f}%\n")

    REACHABLE = {"invalid_reactant_smiles", "invalid_product_smiles"}
    print("invalid molecules by the failure_mode of their reaction:")
    for fm, n in Counter(f for _, f, _ in invalid).most_common():
        mark = "REACHED by fix_smiles" if fm in REACHABLE else "never sent to fix_smiles"
        print(f"  {str(fm):<28}{n:>5}   {mark}")

    ts = M.CorrectorToolset()
    uniq = sorted({s for s, _, _ in invalid})
    res = await ts.fix_smiles(uniq)
    fixable = {k for k, v in res.items() if v.get("valid") and v.get("canonical")}
    print(f"\ndistinct invalid strings: {len(uniq)}   fix_smiles can canonicalise: {len(fixable)}")

    n_fix = sum(1 for s, _, _ in invalid if s in fixable)
    unreach_fix = sum(1 for s, f, _ in invalid if s in fixable and f not in REACHABLE)
    print(f"invalid molecule instances repairable: {n_fix}/{len(invalid)}")
    print(f"   ...of which currently UNREACHABLE by dispatch: {unreach_fix}")
    print(f"\nceiling if every repairable one were applied: "
          f"{100*(total_mol-len(invalid)+n_fix)/total_mol:.2f}%")
    print(f"gain available from fixing dispatch alone:      "
          f"+{100*unreach_fix/total_mol:.2f} pp")

asyncio.run(main())
