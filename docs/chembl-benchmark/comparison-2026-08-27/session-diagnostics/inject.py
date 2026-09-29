"""Score the full effect of rescuing the 10 placeholder rows with strict
de novo retro routes: what it buys on Valid JSON and what it costs elsewhere.
Also probe the analogue path for the one target strict cannot reach."""
import sys, csv, json, inspect
from pathlib import Path
ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import abl_counts as M
from rdkit import Chem

with (ROOT/"data"/"synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

PLACEHOLDER = "json format error"
built = {}
for r in rows:
    if r["response"].strip() != PLACEHOLDER:
        continue
    tgt = r["smiles"]
    res = M._retro_disconnection_all_templates_sync(tgt)
    if res and res.get("found"):
        built[tgt] = json.dumps({
            "reactions": [{
                "reaction_number": 1,
                "reaction_template": res["template"],
                "reactants": res["new_reactants"],
                "product": tgt,
            }],
            "building_blocks": sorted(set(res["new_reactants"])),
        })

base = M.score_effective_dataset([(r["smiles"], r["response"]) for r in rows])
inj  = M.score_effective_dataset([(r["smiles"], built.get(r["smiles"], r["response"])) for r in rows])

KEYS = ["valid_json_percent", "template_mem_percent", "bb_selection_percent",
        "valid_smiles_percent", "matched_reactants_percent", "good_products_strict_percent"]
print(f"rescued {len(built)}/10 placeholder rows with strict de novo routes\n")
print(f"{'metric':<30}{'0_before':>12}{'+de novo':>12}{'delta':>10}")
for k in KEYS:
    b, i = base[k], inj[k]
    print(f"{k:<30}{b:>12}{i:>12}{i-b:>+10.2f}")
print(f"\nraw counts  reactions {base['_n_total_reactions']} -> {inj['_n_total_reactions']}"
      f" | matched {base['_n_successful_reactions']} -> {inj['_n_successful_reactions']}"
      f" | good {base['_n_products_strict']} -> {inj['_n_products_strict']}")

# --- the one strict cannot reach: does the analogue path apply at all? ---
print("\n=== analogue path on the unreachable target ===")
missing = [r["smiles"] for r in rows
           if r["response"].strip() == PLACEHOLDER and r["smiles"] not in built]
sig = inspect.signature(M._fix_via_product_analogue_retro_sync)
print(f"  _fix_via_product_analogue_retro_sync{sig}")
for tgt in missing:
    print(f"  target: {tgt[:70]}")
    try:
        res = M._fix_via_product_analogue_retro_sync(tgt, [], "")
        print(f"    found={bool(res and res.get('found'))}  keys={sorted(res) if res else None}")
        if res and res.get("found"):
            an = res.get("analogue") or res.get("new_product")
            print(f"    analogue product: {str(an)[:70]}")
            print(f"    template in RXN1: {res.get('template') in M._RXN1_TEMPLATE_SET}")
    except Exception as e:
        print(f"    raised {type(e).__name__}: {e}")
