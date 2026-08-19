#!/usr/bin/env python
"""Drive SynAgent's pydantic-ai UI on REAL failing SynLlama routes and screenshot it.

This is the part the raw benchmark does not show. run_bench.py calls the
SynLlama model directly over HTTP -- it never touches the agent. Here the
agent itself decides to call validate_route, and then the corrector tools.

Each case is a two-turn conversation:
  turn 1  "validate this route: <real failing route JSON>"   -> validate_route
  turn 2  "fix the failed steps"                             -> fix_building_blocks,
                                                                fix_step, apply_fixes

The word "fix" in turn 2 is REQUIRED, not stylistic. Corrector.prepare_tools()
hides every corrector tool unless one of fix/correct/repair/search alternative
appears in the last three user messages, so a politer phrasing silently yields
no corrector tools at all.
"""

import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
OUT = HERE / "agent-screenshots"
OUT.mkdir(exist_ok=True)

CASES = json.loads((HERE / "failing_examples.json").read_text(encoding="utf-8"))

CORRECTOR_TOOLS = [
    "fix_building_blocks", "fix_step", "apply_fixes",
    "fix_smiles", "fix_smarts", "search_step_building_blocks",
]


def settle(page, needles, deadline_s=420, stable_needed=3):
    """Wait until the named strings appear and the page stops growing."""
    end = time.time() + deadline_s
    last, stable = 0, 0
    while time.time() < end:
        time.sleep(5)
        body = page.inner_text("body")
        if len(body) == last:
            stable += 1
        else:
            stable, last = 0, len(body)
        if all(n in body for n in needles) and stable >= stable_needed:
            break
    return page.inner_text("body")


results = []

with sync_playwright() as p:
    browser = p.chromium.launch()
    for name, case in CASES.items():
        print(f"\n=== case: {name} | target {case['target'][:50]} ===", flush=True)
        page = browser.new_page(viewport={"width": 1280, "height": 2600})
        page.goto("http://127.0.0.1:8000/", wait_until="networkidle")

        # ---- turn 1: validate -------------------------------------------
        prompt1 = (
            "Validate this synthesis route and report every step:\n\n"
            + case["response"]
        )
        page.fill("textarea, input[type=text]", prompt1)
        page.keyboard.press("Enter")
        print("  turn 1 submitted (validate)", flush=True)
        body1 = settle(page, ["validate_route"])
        page.screenshot(path=str(OUT / f"{name}_1_validate.png"), full_page=True)
        called_validate = "validate_route" in body1

        # ---- turn 2: fix ------------------------------------------------
        # "fix" is load-bearing -- see module docstring.
        page.fill("textarea, input[type=text]",
                  "Now fix the failed steps in that route.")
        page.keyboard.press("Enter")
        print("  turn 2 submitted (fix)", flush=True)
        body2 = settle(page, ["apply_fixes"], deadline_s=600)
        page.screenshot(path=str(OUT / f"{name}_2_correct.png"), full_page=True)

        fired = [t for t in CORRECTOR_TOOLS if t in body2]
        rec = {
            "case": name,
            "target": case["target"],
            "called_validate_route": called_validate,
            "corrector_tools_fired": fired,
            "n_corrector_tools": len(fired),
            "screenshots": [f"{name}_1_validate.png", f"{name}_2_correct.png"],
            "transcript_chars": len(body2),
        }
        results.append(rec)
        print(f"  validate_route: {called_validate} | corrector tools: {fired}",
              flush=True)
        page.close()
    browser.close()

(OUT / "agent_runs.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nwrote {OUT / 'agent_runs.json'}")
for r in results:
    print(f"  {r['case']:<10} validate={r['called_validate_route']} "
          f"corrector={r['n_corrector_tools']} tools")
