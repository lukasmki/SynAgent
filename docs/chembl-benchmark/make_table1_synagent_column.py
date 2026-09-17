#!/usr/bin/env python
"""Compute a SynAgent column for the ChEMBL-Data rows of SynLlama's Table 1.

Uses the 'frozen' subset of data/synllama-raw-output.csv -- 1,000 ChEMBL
targets, one deterministic-sampling response each -- because that is the
only slice of the repo's data that structurally matches Table 1's own
ChEMBL methodology ("select 1000 SMILES strings ... run inferences"). The
other sampling_params buckets (low/medium/high) are SynAgent's own
multi-sample runs and are NOT paper-comparable at the item level.

All six paper metrics are computed, item-level to match the paper's stated
methodology (percentage of SMILES / reactions / templates, not percentage of
routes, except where noted):

  Valid JSON          - response parses as JSON (route-level, 1/target)
  Template Mem.        - % of emitted reaction_template SMARTS strings that
                          exact-string-match an entry in data/91_rxn_templates.sma
                          -- SynLlama's own RXN 1 template library (91 SMARTS
                          templates; the paper's text confirms RXN 1, not RXN 2,
                          is what Table 1's ChEMBL columns were run on)
  BB Selection          - route-level: declared building_blocks (as a set)
                          exactly equals the set of reactants across all
                          reactions that are not themselves the product of
                          another reaction in the same route
  Valid SMILES         - % of all SMILES strings (building blocks +
                          reactants + products) that RDKit can parse
  Matched Reactants     - % of reaction steps whose reactants produce at
                          least one output under the template (SynAgent
                          tries every reactant permutation, unlike
                          SynLlama's fixed emitted order)
  Good Products         - % of reaction steps whose product matches, both
                          strict (exact canonical SMILES) and analog-aware
                          (Morgan/Tanimoto > 0.60, SynAgent's own extension)

Template Mem. and BB Selection are properties of SynLlama's raw generation
(SynAgent doesn't touch templates or the building_blocks list on the
pre-correction pass), so these two numbers are expected to be close to
whatever SynLlama's own checkpoint would score -- they are not something
SynAgent's validator changes.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "comparison-2026-08-27"
csv.field_size_limit(10**7)

sys.path.insert(0, str(ROOT / "src"))

from synagent.validation._toolset import (  # noqa: E402
    ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    _parse_route_json,
    _validate_route_dict,
)

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")


def strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start) :]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repair-csv", type=Path, default=None,
        help=(
            "Optional path to a run_repair_frozen.py output CSV (target, "
            "corrected_route, ...). When given, every frozen-subset response "
            "is replaced by its corrected_route IF that target appears in the "
            "CSV with a non-empty corrected_route -- i.e. this computes the "
            "post-correction 'full pipeline' column instead of the raw one. "
            "Routes never attempted (too long) or where correction produced "
            "nothing keep their original, unmodified response."
        ),
    )
    args = ap.parse_args()

    source = ROOT / "data" / "synllama-raw-output.csv"
    with source.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]

    if args.repair_csv:
        with args.repair_csv.open(encoding="utf-8", newline="") as fh:
            corrected_by_target = {
                r["target"]: r["corrected_route"]
                for r in csv.DictReader(fh)
                if r.get("corrected_route", "").strip()
            }
        n_substituted = 0
        for row in rows:
            corrected = corrected_by_target.get(row["smiles"])
            if corrected:
                row["response"] = corrected
                n_substituted += 1
        print(f"post-correction mode: substituted {n_substituted} corrected routes "
              f"(of {len(corrected_by_target)} available in {args.repair_csv})\n")

    # RXN 1 -- the 91-SMARTS template library Table 1's ChEMBL columns were
    # generated against (paper text, p.5-6: "select 1000 SMILES strings ...
    # using SynLlama models trained on RXN 1"; RXN 1 = 91 templates, RXN 2 =
    # 115, used only for Supplementary Table S1).
    known_templates = {
        line.strip()
        for line in (ROOT / "data" / "91_rxn_templates.sma").read_text().splitlines()
        if line.strip()
    }

    n_targets = len(rows)
    json_ok = 0

    n_smiles_total = 0
    n_smiles_valid = 0

    n_reactions_total = 0
    n_reactants_matched = 0
    n_products_strict = 0
    n_products_analog = 0

    n_templates_total = 0
    n_templates_memorized = 0

    n_routes_scored = 0
    n_routes_bb_correct = 0

    for row in rows:
        try:
            route = _parse_route_json(row["response"])
        except Exception:
            continue
        json_ok += 1

        # Template Mem.: exact-string match of the emitted SMARTS against the
        # trained RXN 1 library. The model was fine-tuned to reproduce known
        # templates verbatim, so string equality (after tag-stripping) is the
        # right test, not a semantic/canonical SMARTS comparison.
        for rxn in route.get("reactions", []):
            n_templates_total += 1
            template = strip(str(rxn.get("reaction_template", "")), "rxn")
            if template in known_templates:
                n_templates_memorized += 1

        # BB Selection: does the declared building_blocks set exactly equal
        # the set of reactants that are not themselves the product of another
        # reaction in this route? Route-level (all-or-nothing), matching how
        # near-100% BB Selection numbers only make sense if it's scored per
        # complete route rather than per individual building block.
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
        expected_bbs = all_reactants - all_products
        if declared_bbs == expected_bbs:
            n_routes_bb_correct += 1

        # Valid SMILES: every building-block SMILES + every reactant/product
        # SMILES across every reaction step, RDKit-parsed independently
        # (item-level, matches the paper's "percentage of Valid SMILES ...
        # out of all SMILES strings in the responses").
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

        # Matched Reactants / Good Products: run SynAgent's own validator,
        # which tries every reactant permutation against the template.
        try:
            report_strict = _validate_route_dict(route, analog_product_threshold=None)
            report_analog = _validate_route_dict(
                route, analog_product_threshold=ANALOG_PRODUCT_SIMILARITY_THRESHOLD
            )
        except Exception:
            continue

        for r_strict, r_analog in zip(report_strict.reactions, report_analog.reactions):
            n_reactions_total += 1
            # "Matched Reactants": the reactants fit the template well enough
            # that running it produced *some* output (SynAgent doesn't track
            # substructure-match separately from RunReactants success).
            if r_strict.failure_mode not in (
                "invalid_reactant_smiles",
                "invalid_product_smiles",
                "invalid_template",
                "no_products",
            ):
                n_reactants_matched += 1
            if r_strict.status == "passed":
                n_products_strict += 1
            if r_analog.status == "passed":
                n_products_analog += 1

    def pct(num, den):
        return round(num / den * 100, 2) if den else None

    print(f"n targets (frozen subset): {n_targets}")
    print(f"Valid JSON:        {pct(json_ok, n_targets)}%  ({json_ok}/{n_targets})")
    print(f"Template Mem.:     {pct(n_templates_memorized, n_templates_total)}%  ({n_templates_memorized}/{n_templates_total})")
    print(f"BB Selection:      {pct(n_routes_bb_correct, n_routes_scored)}%  ({n_routes_bb_correct}/{n_routes_scored})")
    print(f"Valid SMILES:      {pct(n_smiles_valid, n_smiles_total)}%  ({n_smiles_valid}/{n_smiles_total})")
    print(f"Matched Reactants: {pct(n_reactants_matched, n_reactions_total)}%  ({n_reactants_matched}/{n_reactions_total})")
    print(f"Good Products (strict): {pct(n_products_strict, n_reactions_total)}%  ({n_products_strict}/{n_reactions_total})")
    print(f"Good Products (analog>{ANALOG_PRODUCT_SIMILARITY_THRESHOLD}): {pct(n_products_analog, n_reactions_total)}%  ({n_products_analog}/{n_reactions_total})")

    out = {
        "n_targets": n_targets,
        "subset": "sampling_params == frozen",
        "post_correction": bool(args.repair_csv),
        "valid_json_percent": pct(json_ok, n_targets),
        "template_mem_percent": pct(n_templates_memorized, n_templates_total),
        "template_mem_source": "data/91_rxn_templates.sma (RXN 1, 91 templates)",
        "bb_selection_percent": pct(n_routes_bb_correct, n_routes_scored),
        "bb_selection_method": "route-level exact set match",
        "valid_smiles_percent": pct(n_smiles_valid, n_smiles_total),
        "matched_reactants_percent": pct(n_reactants_matched, n_reactions_total),
        "good_products_strict_percent": pct(n_products_strict, n_reactions_total),
        "good_products_analog_percent": pct(n_products_analog, n_reactions_total),
    }
    OUT.mkdir(exist_ok=True)
    out_name = (
        "table1_synagent_column_postcorrection.json"
        if args.repair_csv else "table1_synagent_column.json"
    )
    (OUT / out_name).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {OUT / out_name}")


if __name__ == "__main__":
    main()
