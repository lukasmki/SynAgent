#!/usr/bin/env python
"""Ablation: how much does each corrector tool actually contribute?

Measured directly against the full frozen ChEMBL subset -- no LLM agent loop
needed, because the underlying fix-finding logic (fix_template,
fix_via_analogue_building_block, retro-disconnection) is pure RDKit search
with no model call in it. That means this doesn't depend on whether a
particular orchestrator (Qwen, Gemma, ...) happens to discover and call each
tool -- it measures each tool's actual fix-finding capability directly,
which the previous cluster run's tool-usage numbers could not do (fix_step
called these internally as plain Python, invisible to any tool-call tracker
built from conversation history -- see docs/synagent-reference/corrector.md
and table1_with_synagent.md for that whole story).

Three progressively larger tool sets, each a strict superset of the last:

  template_only              fix_smarts, fix_template (RXN1's 91 templates)
  template+analogue           + fix_via_analogue_building_block (swap a
                              reactant for a database analogue, retry RXN1)
  template+analogue+retro     + retro_disconnection (reverse the original
                              template against the product, retry RXN1 on
                              the resulting fragments)

fix_building_blocks / fix_smiles (SMILES-level repair) run in every variant
-- that's baseline hygiene all three chains already assume, not part of
what's being ablated here.

Every reconstruction step below mirrors apply_fixes (corrector/_toolset.py)
exactly, function for function, so "what would apply_fixes have produced
under this tool set" is a faithful question, not an approximation.
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
csv.field_size_limit(10**7)

sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger, rdBase  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")

from synagent.corrector._toolset import (  # noqa: E402
    CorrectorToolset,
    _fix_template_sync,
    _fix_via_analogue_sync,
    _retro_disconnection_sync,
    _retro_disconnection_all_templates_sync,
)
from synagent.validation._toolset import (  # noqa: E402
    ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    _parse_route_json,
    _validate_route_dict,
)

VARIANTS = {
    "template_only": {"analogue": False, "retro": False, "broad_retro": False},
    "template+analogue": {"analogue": True, "retro": False, "broad_retro": False},
    "template+analogue+retro": {"analogue": True, "retro": True, "broad_retro": False},
    "template+analogue+retro+broad_retro": {"analogue": True, "retro": True, "broad_retro": True},
}


def strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start):]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


async def fix_one_reaction(
    ts: CorrectorToolset, rxn, use_analogue: bool, use_retro: bool, use_broad_retro: bool = False
) -> dict | None:
    """Try to fix one failed reaction under a given tool set -- same order,
    same functions as fix_step's auto chain, just scoped to this variant."""
    failure = rxn.failure_mode
    template = rxn.reaction_template
    reactants = rxn.reactant_smiles
    product = rxn.expected_product

    if failure == "invalid_template":
        sr = await ts.fix_smarts(template)
        if sr.get("fixed"):
            return {"new_template": sr["smarts"]}
        tr = _fix_template_sync(reactants, product)
        if tr.get("found"):
            return {"new_template": tr["template"]}
        return None

    if failure in ("no_products", "wrong_product"):
        tr = _fix_template_sync(reactants, product)
        if tr.get("found"):
            return {"new_template": tr["template"]}
        if use_analogue:
            ar = _fix_via_analogue_sync(reactants, product)
            if ar.get("found"):
                return {"new_template": ar["template"], "new_reactants": ar["new_reactants"]}
        if use_retro:
            rr = _retro_disconnection_sync(reactants, product, template)
            if rr.get("found"):
                return {"new_template": rr["template"], "new_reactants": rr["new_reactants"]}
        if use_broad_retro:
            rrt = _retro_disconnection_all_templates_sync(product)
            if rrt.get("found"):
                return {"new_template": rrt["template"], "new_reactants": rrt["new_reactants"]}
        return None

    if failure in ("invalid_reactant_smiles", "invalid_product_smiles"):
        smiles_to_fix = reactants if failure == "invalid_reactant_smiles" else [product]
        sr = await ts.fix_smiles(smiles_to_fix)
        fixed = {k: v["canonical"] for k, v in sr.items() if v.get("valid") and v.get("canonical")}
        return {"smiles_fixes": fixed} if fixed else None

    return None


async def build_corrected_route(
    ts: CorrectorToolset, original_response: str, use_analogue: bool, use_retro: bool,
    use_broad_retro: bool = False,
) -> dict | None:
    """apply_fixes-equivalent reconstruction for one route under one tool-set
    variant. Returns None if the route already passes (nothing to correct)
    or can't be parsed at all."""
    try:
        route = _parse_route_json(original_response)
    except Exception:
        return None
    try:
        report = _validate_route_dict(route, analog_product_threshold=None)
    except Exception:
        return None
    if report.all_building_blocks_valid and report.all_reactions_passed:
        return None

    bb_fixes: dict[str, str] = {}
    for bb in report.building_blocks:
        if not bb.is_valid:
            sr = await ts.fix_smiles([bb.smiles])
            r = sr.get(bb.smiles, {})
            if r.get("valid") and r.get("canonical"):
                bb_fixes[bb.smiles] = r["canonical"]

    corrected_reactions = []
    for rxn in report.reactions:
        if rxn.status == "passed":
            corrected_reactions.append({
                "reaction_number": rxn.reaction_number,
                "reaction_template": rxn.reaction_template,
                "reactants": rxn.reactant_smiles,
                "product": rxn.expected_product,
            })
            continue

        fix = await fix_one_reaction(ts, rxn, use_analogue, use_retro, use_broad_retro) or {}
        template = fix.get("new_template") or rxn.reaction_template
        if fix.get("new_reactants"):
            reactants = fix["new_reactants"]
        else:
            smiles_fixes = fix.get("smiles_fixes", {})
            reactants = [
                smiles_fixes.get(r) or bb_fixes.get(r) or r for r in rxn.reactant_smiles
            ]
        smiles_fixes = fix.get("smiles_fixes", {})
        product = (
            smiles_fixes.get(rxn.expected_product)
            or bb_fixes.get(rxn.expected_product)
            or rxn.expected_product
        )
        corrected_reactions.append({
            "reaction_number": rxn.reaction_number,
            "reaction_template": template,
            "reactants": reactants,
            "product": product,
        })

    # Same derivation apply_fixes uses: a building block is a reactant that
    # isn't another step's product.
    seen: list[str] = []
    seen_set: set[str] = set()
    products: set[str] = set()
    for rxn in corrected_reactions:
        for r in rxn["reactants"]:
            if r and r not in seen_set:
                seen_set.add(r)
                seen.append(r)
        if rxn["product"]:
            products.add(rxn["product"])
    corrected_bbs = [r for r in seen if r not in products]

    return {"reactions": corrected_reactions, "building_blocks": corrected_bbs}


def score_effective_dataset(effective_responses: list[str]) -> dict:
    """Same six Table 1 metrics as make_table1_synagent_column.py, over
    whatever route text is passed in per target (original or corrected)."""
    rxn1 = {
        line.strip()
        for line in (ROOT / "data" / "91_rxn_templates.sma").read_text().splitlines()
        if line.strip()
    }

    n_targets = len(effective_responses)
    json_ok = 0
    n_smiles_total = n_smiles_valid = 0
    n_reactions_total = n_reactants_matched = n_products_strict = n_products_analog = 0
    n_templates_total = n_templates_memorized = 0
    n_routes_scored = n_routes_bb_correct = 0

    for resp in effective_responses:
        try:
            route = _parse_route_json(resp)
        except Exception:
            continue
        json_ok += 1

        for rxn in route.get("reactions", []):
            n_templates_total += 1
            if strip(str(rxn.get("reaction_template", "")), "rxn") in rxn1:
                n_templates_memorized += 1

        n_routes_scored += 1
        declared_bbs = {strip(str(b), "bb") for b in route.get("building_blocks", [])}
        all_reactants = {
            strip(str(s), "")
            for rxn in route.get("reactions", [])
            for s in rxn.get("reactants", [])
            if str(s).strip()
        }
        all_products = {
            strip(str(rxn.get("product", "")), "")
            for rxn in route.get("reactions", [])
            if str(rxn.get("product", "")).strip()
        }
        if declared_bbs == (all_reactants - all_products):
            n_routes_bb_correct += 1

        for bb in route.get("building_blocks", []):
            n_smiles_total += 1
            if Chem.MolFromSmiles(strip(str(bb), "bb")) is not None:
                n_smiles_valid += 1
        for rxn in route.get("reactions", []):
            for s in rxn.get("reactants", []):
                if str(s).strip():
                    n_smiles_total += 1
                    if Chem.MolFromSmiles(strip(str(s), "")) is not None:
                        n_smiles_valid += 1
            n_smiles_total += 1
            if Chem.MolFromSmiles(strip(str(rxn.get("product", "")), "")) is not None:
                n_smiles_valid += 1

        try:
            rs = _validate_route_dict(route, analog_product_threshold=None)
            ra = _validate_route_dict(route, analog_product_threshold=ANALOG_PRODUCT_SIMILARITY_THRESHOLD)
        except Exception:
            continue
        for r1, r2 in zip(rs.reactions, ra.reactions):
            n_reactions_total += 1
            if r1.failure_mode not in (
                "invalid_reactant_smiles", "invalid_product_smiles",
                "invalid_template", "no_products",
            ):
                n_reactants_matched += 1
            if r1.status == "passed":
                n_products_strict += 1
            if r2.status == "passed":
                n_products_analog += 1

    def pct(n, d):
        return round(n / d * 100, 2) if d else None

    return {
        "valid_json_percent": pct(json_ok, n_targets),
        "template_mem_percent": pct(n_templates_memorized, n_templates_total),
        "bb_selection_percent": pct(n_routes_bb_correct, n_routes_scored),
        "valid_smiles_percent": pct(n_smiles_valid, n_smiles_total),
        "matched_reactants_percent": pct(n_reactants_matched, n_reactions_total),
        "good_products_strict_percent": pct(n_products_strict, n_reactions_total),
        "good_products_analog_percent": pct(n_products_analog, n_reactions_total),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1400,
                    help="skip failing routes longer than this -- matches "
                         "run_repair_frozen.py's default, for direct comparability")
    ap.add_argument("--out", type=Path,
                    default=HERE / "comparison-2026-08-27" / "ablation-corrector-tools.json")
    args = ap.parse_args()

    source = ROOT / "data" / "synllama-raw-output.csv"
    with source.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    # Which of the 1,000 targets actually need correcting under this pool's
    # length cap (matches frozen_failures() in run_repair_frozen.py).
    needs_fix: dict[str, str] = {}
    for row in rows:
        try:
            route = _parse_route_json(row["response"])
            report = _validate_route_dict(route, analog_product_threshold=None)
            ok = report.all_building_blocks_valid and report.all_reactions_passed
        except Exception:
            ok = False
        if not ok and len(row["response"]) <= args.max_len:
            needs_fix[row["smiles"]] = row["response"]

    print(f"{len(rows)} targets total, {len(needs_fix)} need correcting under "
          f"--max-len {args.max_len}\n", flush=True)

    ts = CorrectorToolset()
    results = {}
    for name, cfg in VARIANTS.items():
        print(f"running variant: {name} ...", flush=True)
        effective = []
        fixed_count = 0
        for i, row in enumerate(rows, start=1):
            original = row["response"]
            if row["smiles"] in needs_fix:
                corrected = await build_corrected_route(
                    ts, original, cfg["analogue"], cfg["retro"], cfg["broad_retro"]
                )
                if corrected is not None:
                    fixed_count += 1
                    effective.append(json.dumps(corrected))
                else:
                    effective.append(original)
            else:
                effective.append(original)
            if i % 200 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

        metrics = score_effective_dataset(effective)
        metrics["routes_with_a_reconstruction_attempt"] = fixed_count
        metrics["routes_needing_fix"] = len(needs_fix)
        results[name] = metrics
        print(f"  -> {metrics}\n", flush=True)

    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("=" * 78)
    print(f"{'metric':<28}", *[f"{k:>16}" for k in VARIANTS], sep="")
    for metric in [
        "valid_json_percent", "template_mem_percent", "bb_selection_percent",
        "valid_smiles_percent", "matched_reactants_percent",
        "good_products_strict_percent", "good_products_analog_percent",
    ]:
        row_vals = [f"{results[v][metric]:>16}" for v in VARIANTS]
        print(f"{metric:<28}", *row_vals, sep="")
    print("=" * 78)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
