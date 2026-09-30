"""Does per-step repair break ROUTE connectivity?

fix_one_reaction patches each failing step independently, and
retro_disconnection_all_templates discards that step's original reactants
wholesale. If the step consumed an earlier step's product, that earlier step is
now orphaned -- it makes something nothing consumes. Neither headline metric
would notice: both use per-reaction denominators, so every step is validated in
isolation and a disconnected route still scores as improved.

Orphan = a step whose product is neither the route's target nor consumed as a
reactant by any other step.

The baseline is the control. SynLlama's own routes may already be disconnected;
what matters is whether correction INTRODUCES orphans, not the absolute rate.
"""
import sys, csv, json, asyncio
from pathlib import Path

ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "chembl-benchmark"))

import ablate_corrector_tools as A
from synagent.corrector._toolset import CorrectorToolset
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10 ** 7)

ENABLED = set(A.TOOL_ORDER)          # full tier-6 waterfall
MIN_REACTANTS = 2                    # the screen that was chosen


def canon(s):
    if not s:
        return None
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m is not None else s
    except Exception:
        return s


def orphan_steps(route, target):
    """Indices of steps whose product goes nowhere. None if not multi-step."""
    rxns = route.get("reactions") or []
    if len(rxns) < 2:
        return None
    ct = canon(target)
    consumed = set()
    for r in rxns:
        for x in (r.get("reactants") or []):
            consumed.add(canon(x))
    out = []
    for i, r in enumerate(rxns):
        p = canon(r.get("product"))
        if p is None:
            continue
        if p != ct and p not in consumed:
            out.append(i)
    return out


async def main():
    with (ROOT / "data" / "synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    ts = CorrectorToolset()
    n_multi = 0
    base_bad = 0          # multi-step routes already disconnected before repair
    corr_bad = 0          # ... and after repair
    introduced = []       # connected before, orphaned after  <-- the finding
    healed = 0            # orphaned before, connected after
    n_corrected = 0       # multi-step routes the corrector actually rewrote

    for i, row in enumerate(rows, start=1):
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        resp, target = row["response"], row["smiles"]
        try:
            base = A._parse_route_json(resp)
        except Exception:
            continue                       # unparseable -> tier 6's business
        b_orph = orphan_steps(base, target)
        if b_orph is None:
            continue                       # single-step, immune by construction
        n_multi += 1

        corrected = await A.build_corrected_route(
            ts, resp, ENABLED, target_smiles=target,
            ratio_cap=None, min_reactants=MIN_REACTANTS,
        )
        if corrected is None:
            continue                       # route already passed; nothing rewritten
        n_corrected += 1
        c_orph = orphan_steps(corrected, target)
        if c_orph is None:
            continue

        if b_orph:
            base_bad += 1
        if c_orph:
            corr_bad += 1
        if not b_orph and c_orph:
            introduced.append({
                "target": target,
                "n_steps": len(corrected["reactions"]),
                "orphan_step_indices": c_orph,
            })
        if b_orph and not c_orph:
            healed += 1

    print("\n" + "=" * 70)
    print(f"multi-step parseable routes                : {n_multi}")
    print(f"  ... of which the corrector rewrote       : {n_corrected}")
    print(f"disconnected BEFORE repair (control)       : {base_bad}")
    print(f"disconnected AFTER  repair                 : {corr_bad}")
    print(f"  newly INTRODUCED by repair               : {len(introduced)}")
    print(f"  healed by repair                         : {healed}")
    print("=" * 70)
    for c in introduced[:15]:
        print(f"  + {c['n_steps']}-step, orphan idx {c['orphan_step_indices']}  {c['target'][:60]}")

    out = Path(__file__).parent / "connectivity-results.json"
    out.write_text(json.dumps({
        "min_reactants": MIN_REACTANTS,
        "multi_step_routes": n_multi,
        "corrector_rewrote": n_corrected,
        "disconnected_before": base_bad,
        "disconnected_after": corr_bad,
        "introduced_by_repair": len(introduced),
        "healed_by_repair": healed,
        "introduced_cases": introduced,
    }, indent=2))
    print(f"\nwrote {out}")


asyncio.run(main())
