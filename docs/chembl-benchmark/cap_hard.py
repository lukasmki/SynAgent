"""Full pipeline on a HARD target, so the corrector actually has work to do.

The previous capture came out clean: SmileyLlama produced an easy molecule,
SynLlama routed it correctly, validation passed, and the agent rightly said
there was nothing to fix. To show the repair stage we need a route that
genuinely fails.

Strategy: ask for a large, polycyclic target. In the earlier benchmark the
failures skewed heavily toward big multi-ring molecules (the single failure in
the 8-run pipeline test was MW 437.6 with five rings). Retry with a different
constraint set until validation reports a failure, then fix it.
"""
import json, time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUT = Path("full-pipeline-hard"); OUT.mkdir(exist_ok=True)
TOOLS = ["generate_molecules", "retrosynthesis", "validate_route",
         "fix_building_blocks", "fix_step", "apply_fixes"]

# Progressively harder asks -- more rings, more mass, more stereocentres.
ATTEMPTS = [
    "Generate one drug-like molecule with molecular weight <= 600, at least "
    "four rings, and LogP <= 5.",
    "Generate one drug-like molecule with molecular weight <= 600 containing a "
    "macrocycle.",
    "Generate one drug-like molecule with molecular weight <= 600, Fsp3 >= 0.4, "
    "and at most 3 H-bond donors.",
]

def settle(pg, needles, limit=540, need=3):
    end, last, stable = time.time() + limit, 0, 0
    while time.time() < end:
        time.sleep(5)
        b = pg.inner_text("body")
        stable = stable + 1 if len(b) == last else 0
        last = len(b)
        if all(n in b for n in needles) and stable >= need:
            return b, True
    return pg.inner_text("body"), False

log = []
with sync_playwright() as p:
    br = p.chromium.launch()
    for attempt, gen_prompt in enumerate(ATTEMPTS, start=1):
        print(f"\n### attempt {attempt}: {gen_prompt[:70]}", flush=True)
        pg = br.new_page(viewport={"width": 1280, "height": 3600})
        pg.goto("http://127.0.0.1:8000/", wait_until="domcontentloaded")
        time.sleep(3)

        for name, prompt, needle in [
            ("generate", gen_prompt, "generate_molecules"),
            ("route", "Now produce a retrosynthetic route for that exact molecule.",
             "retrosynthesis"),
            ("validate", "Validate that route and report every step.", "validate_route"),
        ]:
            pg.fill("textarea, input[type=text]", prompt)
            pg.keyboard.press("Enter")
            body, ok = settle(pg, [needle])
            print(f"  [{name}] marker={ok}", flush=True)

        low = body.lower()
        failed = ("failed" in low or "false" in low or "failure mode" in low)
        print(f"  validation reports a failure: {failed}", flush=True)
        if not failed:
            print("  -> route came out clean, trying a harder target", flush=True)
            pg.close()
            continue

        pg.fill("textarea, input[type=text]",
                "Now fix the failed steps: call fix_building_blocks(), then "
                "fix_step(N) for each failed step, then apply_fixes().")
        pg.keyboard.press("Enter")
        body, ok = settle(pg, ["apply_fixes"], limit=720)
        pg.screenshot(path=str(OUT / f"hard_attempt{attempt}_full.png"), full_page=True)
        fired = [t for t in TOOLS if t in body]
        log.append({"attempt": attempt, "prompt": gen_prompt,
                    "apply_fixes_reached": ok, "tools_visible": fired})
        print(f"  [fix] tools: {fired}", flush=True)
        pg.close()
        if ok:
            break
    br.close()
(OUT / "hard_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
print("\n" + json.dumps(log, indent=2))
