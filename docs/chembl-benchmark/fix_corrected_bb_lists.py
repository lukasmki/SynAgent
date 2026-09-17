#!/usr/bin/env python
"""One-time patch for a corrector bug in apply_fixes (corrector/_toolset.py).

apply_fixes used to rebuild `building_blocks` by patching the *original* list
through `bb_fixes`, independently of how it rebuilt `corrected_reactions`.
The same molecule could get "fixed" through two separate code paths --
bb_fixes vs. a reaction step's own smiles_fixes, or a new_reactants
substitution from the retro-disconnection fallback -- and the two lists could
silently diverge (see table1_with_synagent.md footnote 6 for a worked
example: a single-atom typo between a declared building block and the
reactant actually used in the reaction).

The fix (already applied in corrector/_toolset.py) derives building_blocks
directly from corrected_reactions going forward. This script re-derives it
the same way for every corrected_route already saved by run_repair_frozen.py,
so the existing 341-route corrector run doesn't have to be re-run through the
live agent loop -- the reactions themselves were never wrong, only the
separately-patched building-blocks list was.
"""

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10**7)


def strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start):]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


def rederive_building_blocks(route: dict) -> list[str]:
    """A building block is a reactant that isn't another step's product --
    the same definition make_table1_synagent_column.py's BB Selection check
    uses, so there's nothing left to drift once both sides agree."""
    seen: list[str] = []
    seen_set: set[str] = set()
    products: set[str] = set()
    for rxn in route.get("reactions", []):
        for s in rxn.get("reactants", []):
            s = strip(str(s).strip(), "")
            if s and s not in seen_set:
                seen_set.add(s)
                seen.append(s)
        p = strip(str(rxn.get("product", "")).strip(), "")
        if p:
            products.add(p)
    return [s for s in seen if s not in products]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        HERE / "comparison-2026-08-27" / "frozen-repair-full.csv"
    )
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0].keys())

    patched = 0
    for row in rows:
        raw = row.get("corrected_route", "").strip()
        if not raw:
            continue
        try:
            route = json.loads(raw)
        except Exception:
            continue

        old_bbs = [strip(str(b), "bb") for b in route.get("building_blocks", [])]
        new_bbs = rederive_building_blocks(route)
        if old_bbs != new_bbs:
            patched += 1
            route["building_blocks"] = new_bbs
            row["corrected_route"] = json.dumps(route)

    print(f"patched building_blocks in {patched} corrected routes")

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
