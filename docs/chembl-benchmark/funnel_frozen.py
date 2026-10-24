#!/usr/bin/env python
"""Cascading funnel view of the six Table 1 metrics, reordered as requested
for the slide 17 table: Valid JSON -> SMILES -> Reactants -> Products ->
Templates -> BB. Each stage's denominator is routes that survived every
earlier stage -- a route that fails Valid JSON is excluded from every
category after it, a route that fails SMILES is excluded from Reactants
onward, and so on.

This is a DIFFERENT statistic from every other number in this deck. SynLlama's
own algorithm (score_effective_dataset in ablate_corrector_tools.py) scores
each of the six metrics over its own natural population (e.g. Good Products
over successful_reactions, Template Mem. over total_reactions) and never
cascades them through one shared, ordered funnel. This script builds that
funnel fresh: each stage is a route-level ALL-reactions-must-pass check
(a route counts as passing "Reactants" only if every one of its reaction
steps matches reactant count and reacts; passing "Products" only if every
step reproduces its exact declared product; etc.), computed on the frozen
1,000-target local subset of data/synllama-raw-output.csv, raw (uncorrected)
responses only -- the same dataset used throughout this deck, not the
official 1b-2m cluster data.
"""

import csv
import json
import sys
from itertools import permutations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
csv.field_size_limit(10**7)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem, RDLogger, rdBase  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")

from ablate_corrector_tools import (  # noqa: E402
    CorrectorToolset,
    TOOL_ORDER,
    _arrange_and_react,
    _RXN1_TEMPLATE_SET,
    build_corrected_route,
)
from synagent.validation._toolset import _parse_route_json, _validate_route_dict  # noqa: E402

MIN_REACTANTS = 2  # same screen used throughout this deck's corrected-route figures


def _parse_route(resp: str):
    try:
        output = json.loads(resp)
    except Exception:
        return None
    if not isinstance(output, dict) or "reactions" not in output or "building_blocks" not in output:
        return None
    reactions = output["reactions"]
    for reaction in reactions:
        if "reaction_template" not in reaction or "reactants" not in reaction or "product" not in reaction:
            return None
        raw_template = str(reaction["reaction_template"])
        has_open = "<rxn>" in raw_template
        has_close = "</rxn>" in raw_template
        if not ((has_open and has_close) or (not has_open and not has_close)):
            return None
    return output


def _reaction_template(reaction: dict) -> str:
    raw_template = str(reaction["reaction_template"])
    if "<rxn>" in raw_template:
        return raw_template.split("<rxn>")[1].split("</rxn>")[0]
    return raw_template


def _reaction_reactants(reaction: dict) -> list[str]:
    return [
        str(r).split("<bb>")[-1].split("</bb>")[0] if "<bb>" in str(r) else str(r)
        for r in reaction["reactants"]
    ]


def score_funnel(targets_and_responses: list[tuple[str, str]]) -> dict:
    n_targets = len(targets_and_responses)

    survivors = [(t, _parse_route(r)) for t, r in targets_and_responses]
    stage_counts = {}

    # Stage 1: Valid JSON -- parses as a well-formed route dict.
    stage_counts["valid_json"] = {"numerator": sum(1 for _, o in survivors if o is not None), "denominator": n_targets}
    survivors = [(t, o) for t, o in survivors if o is not None]

    # Stage 2: SMILES -- every reactant and product SMILES in the route parses.
    def _all_smiles_valid(output: dict) -> bool:
        for reaction in output["reactions"]:
            for r in _reaction_reactants(reaction):
                if r and Chem.MolFromSmiles(r) is None:
                    return False
            product = str(reaction["product"])
            if Chem.MolFromSmiles(product) is None:
                return False
        return True

    denom = len(survivors)
    passed = [(t, o) for t, o in survivors if _all_smiles_valid(o)]
    stage_counts["smiles"] = {"numerator": len(passed), "denominator": denom}
    survivors = passed

    # Stage 3: Reactants -- every reaction step's reactant count matches its
    # template's arity and the reaction actually fires (SynLlama's own
    # arrange_and_react, exact-count, permutation search).
    def _all_reactants_matched(output: dict) -> bool:
        for reaction in output["reactions"]:
            template = _reaction_template(reaction)
            reactants = [r for r in _reaction_reactants(reaction) if r and Chem.MolFromSmiles(r) is not None]
            if _arrange_and_react(template, reactants) is None:
                return False
        return True

    denom = len(survivors)
    passed = [(t, o) for t, o in survivors if _all_reactants_matched(o)]
    stage_counts["reactants"] = {"numerator": len(passed), "denominator": denom}
    survivors = passed

    # Stage 4: Products -- every reaction step reproduces its exact declared
    # product (canonical-SMILES membership in the actual RDKit output).
    def _all_products_good(output: dict) -> bool:
        for reaction in output["reactions"]:
            template = _reaction_template(reaction)
            reactants = [r for r in _reaction_reactants(reaction) if r and Chem.MolFromSmiles(r) is not None]
            prods = _arrange_and_react(template, reactants)
            product_mol = Chem.MolFromSmiles(str(reaction["product"]))
            if product_mol is None or prods is None:
                return False
            product_canon = Chem.MolToSmiles(product_mol)
            prod_canon_set = set()
            for p in prods:
                try:
                    prod_canon_set.add(Chem.MolToSmiles(p))
                except Exception:
                    continue
            if product_canon not in prod_canon_set:
                return False
        return True

    denom = len(survivors)
    passed = [(t, o) for t, o in survivors if _all_products_good(o)]
    stage_counts["products"] = {"numerator": len(passed), "denominator": denom}
    survivors = passed

    # Stage 5: Templates -- every reaction's template is one of the 91 RXN1
    # templates SynLlama was actually trained on.
    def _all_templates_memorized(output: dict) -> bool:
        for reaction in output["reactions"]:
            if _reaction_template(reaction) not in _RXN1_TEMPLATE_SET:
                return False
        return True

    denom = len(survivors)
    passed = [(t, o) for t, o in survivors if _all_templates_memorized(o)]
    stage_counts["templates"] = {"numerator": len(passed), "denominator": denom}
    survivors = passed

    # Stage 6: BB -- every declared building block is actually used somewhere
    # in the route (reactant_stack membership, same definition as SynLlama's
    # own recycled_bbs, applied as a route-level pass/fail: perfect obedience
    # only, not the fractional average).
    def _bb_selection_perfect(target_smiles: str, output: dict) -> bool:
        reactant_stack = [target_smiles]
        for reaction in output["reactions"]:
            reactants = _reaction_reactants(reaction)
            reactant_stack.extend(reactants)
            product = str(reaction["product"])
            if product in reactant_stack:
                reactant_stack.remove(product)
        building_blocks = output["building_blocks"]
        if not building_blocks:
            return True
        for bb in building_blocks:
            bb_clean = str(bb).split("<bb>")[-1].split("</bb>")[0]
            if bb_clean not in reactant_stack:
                return False
        return True

    denom = len(survivors)
    passed = [(t, o) for t, o in survivors if _bb_selection_perfect(t, o)]
    stage_counts["bb"] = {"numerator": len(passed), "denominator": denom}

    for key, c in stage_counts.items():
        c["percent"] = round(c["numerator"] / c["denominator"] * 100, 2) if c["denominator"] else None

    return stage_counts


async def _build_corrected_responses(rows: list[dict]) -> list[str]:
    """Same corrected-route construction used throughout this deck
    (compare_frozen_three_way.py): rebuild only the routes that fail
    SynAgent's own validator, via the full corrector tool chain."""
    needs_fix: dict[str, str] = {}
    for row in rows:
        try:
            route = _parse_route_json(row["response"])
            report = _validate_route_dict(route, analog_product_threshold=None)
            ok = report.all_building_blocks_valid and report.all_reactions_passed
        except Exception:
            ok = False
        if not ok:
            needs_fix[row["smiles"]] = row["response"]

    ts = CorrectorToolset()
    enabled = set(TOOL_ORDER)
    corrected_responses = []
    for i, row in enumerate(rows, start=1):
        if row["smiles"] in needs_fix:
            corrected = await build_corrected_route(
                ts, row["response"], enabled, target_smiles=row["smiles"],
                min_reactants=MIN_REACTANTS,
            )
            corrected_responses.append(json.dumps(corrected) if corrected else row["response"])
        else:
            corrected_responses.append(row["response"])
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return corrected_responses


def _print_stage_table(label: str, result: dict) -> None:
    print(f"\n{label}")
    print(f"{'stage':<14}{'numerator':>12}{'denominator':>14}{'percent':>10}")
    for stage in ("valid_json", "smiles", "reactants", "products", "templates", "bb"):
        c = result[stage]
        print(f"{stage:<14}{c['numerator']:>12}{c['denominator']:>14}{c['percent']:>10}")


async def main() -> None:
    source = ROOT / "data" / "synllama-raw-output.csv"
    with source.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]
    print(f"{len(rows)} frozen-subset targets\n", flush=True)

    raw_result = score_funnel([(r["smiles"], r["response"]) for r in rows])

    print("building + scoring corrected routes (full tool chain, "
          f"min_reactants={MIN_REACTANTS}) ...", flush=True)
    corrected_responses = await _build_corrected_responses(rows)
    corrected_result = score_funnel(
        [(r["smiles"], resp) for r, resp in zip(rows, corrected_responses)]
    )

    out = HERE / "comparison-2026-08-27" / "funnel_frozen.json"
    out.write_text(json.dumps({"raw": raw_result, "corrected": corrected_result}, indent=2))

    _print_stage_table("raw", raw_result)
    _print_stage_table("corrected", corrected_result)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
