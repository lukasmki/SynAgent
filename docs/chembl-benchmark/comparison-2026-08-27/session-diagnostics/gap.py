"""Why is only 0.49pp of a 3.97pp repairable pool realised? Split the invalid
molecules by the two gates in main(): the --max-len cap, and needs_fix."""
import sys, csv, json, asyncio
from pathlib import Path
from collections import Counter
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT/"src")); sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem
MAXLEN = 1400

async def main():
    with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    # replicate needs_fix exactly
    needs, passes = set(), set()
    for r in rows:
        try:
            rep = M._validate_route_dict(M._parse_route_json(r["response"]), analog_product_threshold=None)
            ok = rep.all_building_blocks_valid and rep.all_reactions_passed
        except Exception:
            ok = False
        if ok: passes.add(r["smiles"])
        elif len(r["response"]) <= MAXLEN: needs.add(r["smiles"])

    total_mol = 0; inv = []   # (smiles, target, bucket)
    for r in rows:
        try:
            out = json.loads(r["response"])
        except Exception:
            continue
        if r["smiles"] in passes:      bucket = "route already passes (never corrected)"
        elif r["smiles"] in needs:     bucket = "in needs_fix (corrector runs)"
        else:                          bucket = f"SKIPPED: response > --max-len {MAXLEN}"
        for rxn in out.get("reactions", []):
            if not all(k in rxn for k in ("reaction_template","reactants","product")): break
            raw = str(rxn["reaction_template"]); ho, hc = "<rxn>" in raw, "</rxn>" in raw
            if ho and hc: t = raw.split("<rxn>")[1].split("</rxn>")[0]
            elif not ho and not hc: t = raw
            else: break
            if t not in M._RXN1_TEMPLATE_SET: continue
            mols = [str(x).split("<bb>")[-1].split("</bb>")[0] if "<bb>" in str(x) else str(x)
                    for x in rxn["reactants"]] + [str(rxn["product"])]
            for x in mols:
                if x == "": continue
                total_mol += 1
                if Chem.MolFromSmiles(x) is None: inv.append((x, r["smiles"], bucket))

    ts = M.CorrectorToolset()
    res = await ts.fix_smiles(sorted({s for s, _, _ in inv}))
    fixable = {k for k, v in res.items() if v.get("valid") and v.get("canonical")}

    print(f"total molecules {total_mol}, invalid {len(inv)}\n")
    print(f"{'bucket':<46}{'invalid':>9}{'repairable':>12}{'pp if fixed':>13}")
    for b, n in Counter(x[2] for x in inv).most_common():
        rep_n = sum(1 for s, _, bb in inv if bb == b and s in fixable)
        print(f"{b:<46}{n:>9}{rep_n:>12}{100*rep_n/total_mol:>12.2f}")
    lens = [len(r['response']) for r in rows if r['smiles'] not in passes and r['smiles'] not in needs]
    if lens:
        print(f"\nroutes skipped by the cap: {len(lens)}  "
              f"(response length min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)})")

asyncio.run(main())
