#!/usr/bin/env python
"""Three-way comparison on the SAME dataset (the 'frozen' subset of
data/synllama-raw-output.csv -- 1,000 ChEMBL targets, one deterministic
response each), so the three columns are only ever differing in scoring
method or correction, never in which molecules they're scored over:

  1. SynLlama's own algorithm, on the raw (uncorrected) responses
       -- score_effective_dataset() / _arrange_and_react(), reimplementing
          SynLlama's calc_benchmark_rxn (ablate_corrector_tools.py)
  2. SynAgent's own validator, on the SAME raw responses
       -- _validate_route_dict(), permutation-tolerant reactant matching,
          item-level Valid SMILES/Template Mem/BB Selection (this file's own
          logic, identical to make_table1_synagent_column.py with no
          --repair-csv)
  3. SynAgent's own validator, on routes corrected by the CURRENT full
     corrector tool chain (all of TOOL_ORDER from ablate_corrector_tools.py,
     min_reactants=2 screen -- the configuration that file's own handoff doc
     settled on)

Columns 2 and 3 share a scoring method on purpose, so the only thing that
differs between them is the corrector's effect -- column 1 is a separate
reference point (how would this same data read under SynLlama's own rule),
not something 2/3 are trying to match.
"""

import asyncio
import csv
import json
import sys
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
    build_corrected_route,
    score_effective_dataset as synllama_score,
)
from synagent.validation._toolset import (  # noqa: E402
    ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    _parse_route_json,
    _validate_route_dict,
)

MIN_REACTANTS = 2  # the screen ablate_corrector_tools.py's handoff doc settled on


def strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start):]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


def synagent_score(responses: list[str]) -> dict:
    """SynAgent's own validator -- identical logic to
    make_table1_synagent_column.py, inlined here so both raw and corrected
    columns are scored by literally the same function."""
    known_templates = {
        line.strip()
        for line in (ROOT / "data" / "91_rxn_templates.sma").read_text().splitlines()
        if line.strip()
    }

    n_targets = len(responses)
    json_ok = 0
    n_smiles_total = n_smiles_valid = 0
    n_reactions_total = n_reactants_matched = n_products_strict = n_products_analog = 0
    n_templates_total = n_templates_memorized = 0
    n_routes_scored = n_routes_bb_correct = 0

    for resp in responses:
        try:
            route = _parse_route_json(resp)
        except Exception:
            continue
        json_ok += 1

        for rxn in route.get("reactions", []):
            n_templates_total += 1
            if strip(str(rxn.get("reaction_template", "")), "rxn") in known_templates:
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
    source = ROOT / "data" / "synllama-raw-output.csv"
    with source.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["sampling_params"] == "frozen"]
    print(f"{len(rows)} frozen-subset targets\n", flush=True)

    raw_responses = [r["response"] for r in rows]

    print("1/3 scoring raw via SynLlama's own algorithm ...", flush=True)
    col_synllama = synllama_score([(r["smiles"], r["response"]) for r in rows])

    print("2/3 scoring raw via SynAgent's validator ...", flush=True)
    col_synagent_raw = synagent_score(raw_responses)

    print("3/3 building + scoring corrected routes (full tool chain, "
          f"min_reactants={MIN_REACTANTS}) ...", flush=True)
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
    col_synagent_corrected = synagent_score(corrected_responses)

    print("extra: scoring the SAME corrected routes via SynLlama's own algorithm ...", flush=True)
    col_synllama_corrected = synllama_score(
        [(r["smiles"], resp) for r, resp in zip(rows, corrected_responses)]
    )

    results = {
        "synllama_algorithm_raw": col_synllama,
        "synagent_validator_raw": col_synagent_raw,
        "synagent_validator_corrected": col_synagent_corrected,
        "synllama_algorithm_corrected": col_synllama_corrected,
    }
    out = HERE / "comparison-2026-08-27" / "compare_frozen_three_way.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 106)
    metrics = [
        "valid_json_percent", "template_mem_percent", "bb_selection_percent",
        "valid_smiles_percent", "matched_reactants_percent",
        "good_products_strict_percent", "good_products_analog_percent",
    ]
    cols = ("synllama_algorithm_raw", "synagent_validator_raw",
            "synagent_validator_corrected", "synllama_algorithm_corrected")
    print(f"{'metric':<28}{'synllama_raw':>16}{'synagent_raw':>16}{'synagent_corrected':>20}{'synllama_corrected':>20}")
    for m in metrics:
        vals = [results[c].get(m) for c in cols]
        print(f"{m:<28}" + "".join(f"{(v if v is not None else '-'):>16}" if i < 2
                                     else f"{(v if v is not None else '-'):>20}"
                                     for i, v in enumerate(vals)))
    print("=" * 106)
    print(f"wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
