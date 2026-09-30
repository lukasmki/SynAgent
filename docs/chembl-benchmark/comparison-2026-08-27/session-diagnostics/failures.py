"""What is actually FAILING Template Mem, BB Selection and Valid SMILES?

Replicates score_effective_dataset's walk exactly (ablate_corrector_tools.py
~line 580) but records the offending items instead of just counting them.

  template_mem  = (total_reactions - not_in_template) / total_reactions
                  fails when reaction_template is not an exact string match
                  against the 91-template RXN1 set
  bb_selection  = mean over routes of (declared BBs found in reactant_stack)
                  / (declared BBs).  Per-ROUTE average, not per-molecule.
  valid_smiles  = (total_molecules - invalid) / total_molecules over every
                  reactant and product in every reaction

Run on the raw baseline and on the tier-6 corrected routes (min_reactants=2).
"""
import sys, csv, json, asyncio, collections
from pathlib import Path

ROOT = Path("/pscratch/sd/s/stefani/SynAgent")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "docs" / "chembl-benchmark"))

import ablate_corrector_tools as A
from synagent.corrector._toolset import CorrectorToolset
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
csv.field_size_limit(10 ** 7)

ENABLED = set(A.TOOL_ORDER)
MIN_REACTANTS = 2


def strip_bb(x):
    s = str(x)
    return s.split("<bb>")[-1].split("</bb>")[0] if "<bb>" in s else s


def walk(route, target, acc):
    """One route through the scorer's logic, recording failures into acc.

    Mirrors score_effective_dataset exactly: the stack is SEEDED WITH THE
    TARGET, and reaction_template arrives <rxn>-wrapped on raw model output but
    bare on corrected routes (both accepted; exactly one tag = malformed, the
    whole route is dropped).
    """
    reactions = route.get("reactions") or []
    building_blocks = route.get("building_blocks") or []
    reactant_stack = [target]
    for reaction in reactions:
        if not all(k in reaction for k in ("reaction_template", "reactants", "product")):
            acc["aborted_routes"] += 1
            return
        raw = str(reaction["reaction_template"])
        has_open, has_close = "<rxn>" in raw, "</rxn>" in raw
        if has_open and has_close:
            template = raw.split("<rxn>")[1].split("</rxn>")[0]
        elif not has_open and not has_close:
            template = raw
        else:
            acc["aborted_routes"] += 1
            return
        acc["total_reactions"] += 1
        if template not in A._RXN1_TEMPLATE_SET:
            acc["bad_templates"][template] += 1
            acc["bad_template_targets"].setdefault(template, []).append(target)
            continue
        reactants = [strip_bb(r) for r in reaction["reactants"]]
        reactant_stack.extend(reactants)
        product = str(reaction["product"])
        if product in reactant_stack:
            reactant_stack.remove(product)
        acc["total_molecules"] += len(reactants)
        for r in reactants:
            if Chem.MolFromSmiles(r) is None:
                acc["invalid"][r] += 1
                acc["invalid_role"]["reactant"] += 1
            elif r == "":
                acc["total_molecules"] -= 1
        acc["total_molecules"] += 1
        if Chem.MolFromSmiles(product) is None:
            acc["invalid"][product] += 1
            acc["invalid_role"]["product"] += 1

    if building_blocks:
        missing = [b for b in (strip_bb(x) for x in building_blocks)
                   if b not in reactant_stack]
        if missing:
            acc["bb_missing_routes"].append({
                "target": target, "n_declared": len(building_blocks),
                "n_missing": len(missing), "missing": missing[:4],
            })
        acc["bb_values"].append(
            (len(building_blocks) - len(missing)) / len(building_blocks))
    else:
        acc["bb_values"].append(1.0)


def new_acc():
    return {"total_reactions": 0, "total_molecules": 0, "aborted_routes": 0,
            "bad_templates": collections.Counter(), "bad_template_targets": {},
            "invalid": collections.Counter(),
            "invalid_role": collections.Counter(),
            "bb_missing_routes": [], "bb_values": []}


def report(label, acc):
    tr, tm = acc["total_reactions"], acc["total_molecules"]
    n_bad_t = sum(acc["bad_templates"].values())
    n_inv = sum(acc["invalid"].values())
    bb = sum(acc["bb_values"]) / len(acc["bb_values"]) * 100
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    print(f"Template Mem  {100*(tr-n_bad_t)/tr:6.2f}   {n_bad_t} bad reaction(s) "
          f"of {tr}, {len(acc['bad_templates'])} distinct template(s)")
    for t, n in acc["bad_templates"].most_common():
        print(f"    x{n}  {t}")
        for tg in acc["bad_template_targets"][t]:
            print(f"          target: {tg}")
    print(f"\nBB Selection  {bb:6.2f}   {len(acc['bb_missing_routes'])} route(s) "
          f"with >=1 declared BB absent from the reactant stack")
    for r in acc["bb_missing_routes"][:12]:
        print(f"    {r['n_missing']}/{r['n_declared']} missing  target {r['target'][:46]}")
        for m in r["missing"]:
            print(f"          absent BB: {m[:70]}")
    print(f"\naborted routes (malformed/missing keys): {acc['aborted_routes']}")
    print(f"\nValid SMILES  {100*(tm-n_inv)/tm:6.2f}   {n_inv} invalid of {tm} molecules "
          f"({dict(acc['invalid_role'])})")
    print(f"    {len(acc['invalid'])} distinct bad string(s); top 12 by count:")
    for s, n in acc["invalid"].most_common(12):
        print(f"    x{n:<4} {s[:88]!r}")


async def main():
    with (ROOT / "data" / "synllama-official-1b2m.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    base, corr = new_acc(), new_acc()
    ts = CorrectorToolset()
    for i, row in enumerate(rows, start=1):
        if i % 250 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        resp, target = row["response"], row["smiles"]
        try:
            r0 = A._parse_route_json(resp)
        except Exception:
            r0 = None
        if r0 is not None:
            walk(r0, target, base)
        fixed = await A.build_corrected_route(
            ts, resp, ENABLED, target_smiles=target,
            ratio_cap=None, min_reactants=MIN_REACTANTS)
        rc = fixed if fixed is not None else r0
        if rc is not None:
            walk(rc, target, corr)

    report("BASELINE (raw SynLlama 1b-2m)", base)
    report("CORRECTED (tier 6, min_reactants=2)", corr)

    out = Path(__file__).parent / "failures-results.json"
    out.write_text(json.dumps({
        k: {"bad_templates": dict(v["bad_templates"]),
            "bad_template_targets": v["bad_template_targets"],
            "invalid_smiles": dict(v["invalid"]),
            "invalid_role": dict(v["invalid_role"]),
            "bb_missing_routes": v["bb_missing_routes"]}
        for k, v in (("baseline", base), ("corrected", corr))}, indent=2))
    print(f"\nwrote {out}")


asyncio.run(main())
