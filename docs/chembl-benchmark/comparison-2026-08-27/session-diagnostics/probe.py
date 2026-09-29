"""Two diagnostics on the strict corrector chain:

A. Is the tier-1 Good Products dip real damage, or denominator dilution?
B. Do the four strict tools in the no_products/wrong_product arm contend?
   If at most one ever fires per reaction, reordering them is a no-op.
"""
import sys, csv, json, asyncio
from pathlib import Path
from collections import Counter

ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import abl_counts as M
from rdkit import Chem

SOURCE = ROOT / "data" / "synllama-official-1b2m.csv"
MAX_LEN = 1400

STRICT = [
    ("fix_template",                      lambda r: M._fix_template_sync(r.reactant_smiles, r.expected_product)),
    ("retro_disconnection",               lambda r: M._retro_disconnection_sync(r.reactant_smiles, r.expected_product, r.reaction_template)),
    ("retro_disconnection_all_templates", lambda r: M._retro_disconnection_all_templates_sync(r.expected_product)),
    ("partial_reactant_retention",        lambda r: M._partial_reactant_retention_sync(r.reactant_smiles, r.expected_product)),
]


def outcome(fix, rxn):
    """Score one tool's proposal exactly as score_effective_dataset would:
    in-RXN1 template, reactants react, declared product recovered exactly."""
    if not fix or not fix.get("found"):
        return None
    tmpl = fix.get("template")
    reactants = fix.get("new_reactants") or rxn.reactant_smiles
    if tmpl is None:
        return None
    in_rxn1 = tmpl in M._RXN1_TEMPLATE_SET
    valid = [r for r in reactants if r and Chem.MolFromSmiles(r) is not None]
    prods = M._arrange_and_react(tmpl, valid) if in_rxn1 else None
    matched = prods is not None
    exact = False
    if matched:
        canon = set()
        for p in prods:
            try:
                canon.add(Chem.MolToSmiles(p))
            except Exception:
                pass
        pm = Chem.MolFromSmiles(rxn.expected_product)
        exact = pm is not None and Chem.MolToSmiles(pm) in canon
    return {"in_rxn1": in_rxn1, "matched": matched, "exact": exact}


async def main():
    with SOURCE.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    needs_fix = {}
    for row in rows:
        try:
            route = M._parse_route_json(row["response"])
            rep = M._validate_route_dict(route, analog_product_threshold=None)
            ok = rep.all_building_blocks_valid and rep.all_reactions_passed
        except Exception:
            ok = False
        if not ok and len(row["response"]) <= MAX_LEN:
            needs_fix[row["smiles"]] = row["response"]
    print(f"{len(rows)} targets, {len(needs_fix)} need correcting\n", flush=True)

    # ---------- A. baseline vs fix_smiles-only, in absolute counts ----------
    print("=== A. is the tier-1 dip damage or dilution? ===", flush=True)
    base = M.score_effective_dataset([(r["smiles"], r["response"]) for r in rows])

    ts = M.CorrectorToolset()
    eff = []
    for row in rows:
        if row["smiles"] in needs_fix:
            c = await M.build_corrected_route(ts, row["response"], {"fix_smiles"})
            eff.append((row["smiles"], json.dumps(c) if c is not None else row["response"]))
        else:
            eff.append((row["smiles"], row["response"]))
    t1 = M.score_effective_dataset(eff)

    for label, d in (("0_before", base), ("1_fix_smiles", t1)):
        print(f"  {label:<14} reactions={d['_n_total_reactions']:>5}  "
              f"matched={d['_n_successful_reactions']:>5} ({d['matched_reactants_percent']}%)  "
              f"good_products={d['_n_products_strict']:>5} ({d['good_products_strict_percent']}%)", flush=True)
    dm = t1['_n_successful_reactions'] - base['_n_successful_reactions']
    dp = t1['_n_products_strict'] - base['_n_products_strict']
    print(f"  delta: matched {dm:+d} reactions, good products {dp:+d} reactions\n", flush=True)

    # ---------- B. contention among the four strict tools ----------
    print("=== B. do the four strict tools contend? ===", flush=True)
    fires = Counter()          # tool -> times it reports found
    exacts = Counter()         # tool -> times its proposal recovers the declared product
    matches = Counter()        # tool -> times its proposal reacts at all
    n_multi = 0                # reactions where >1 tool fires
    n_any = 0                  # reactions where >=1 tool fires
    n_rxn = 0                  # failing reactions in the no_products/wrong_product arm
    combos = Counter()
    disagree = 0               # >1 fires AND they differ on exactness

    for smiles, resp in needs_fix.items():
        try:
            rep = M._validate_route_dict(M._parse_route_json(resp), analog_product_threshold=None)
        except Exception:
            continue
        for rxn in rep.reactions:
            if rxn.status == "passed" or rxn.failure_mode not in ("no_products", "wrong_product"):
                continue
            n_rxn += 1
            got = {}
            for name, fn in STRICT:
                try:
                    o = outcome(fn(rxn), rxn)
                except Exception:
                    o = None
                if o:
                    got[name] = o
                    fires[name] += 1
                    matches[name] += o["matched"]
                    exacts[name] += o["exact"]
            if got:
                n_any += 1
                combos[tuple(sorted(got))] += 1
            if len(got) > 1:
                n_multi += 1
                if len(set(o["exact"] for o in got.values())) > 1:
                    disagree += 1

    print(f"  failing reactions in this arm: {n_rxn}")
    print(f"  at least one tool fires:       {n_any}")
    print(f"  more than one tool fires:      {n_multi}   <-- order only matters here")
    print(f"  ...and they disagree on exact: {disagree}   <-- order changes the SCORE only here\n")
    print(f"  {'tool':<36}{'fires':>7}{'reacts':>8}{'exact':>7}{'exact|fires':>13}")
    for name, _ in STRICT:
        f = fires[name]
        rate = f"{100*exacts[name]/f:.1f}%" if f else "-"
        print(f"  {name:<36}{f:>7}{matches[name]:>8}{exacts[name]:>7}{rate:>13}")
    print("\n  co-firing combinations:")
    for combo, n in combos.most_common():
        print(f"    {n:>5}  {' + '.join(combo)}")

asyncio.run(main())
