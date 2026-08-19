#!/usr/bin/env python
"""Measure what the corrector actually fixes.

THE QUESTION
SynLlama's baseline is 3,065/10,000 = 30.65% valid on the ChEMBL test set.
6,935 paths fail. SynAgent's pitch is that validation + correction recovers
some of those. This measures how many.

WHY IT MUST GO THROUGH THE AGENT
The corrector tools cannot be called standalone. All three take
`ctx: RunContext` and read their inputs out of conversation history:

    fix_step(ctx, step)        -> reads the last ValidationReport
    fix_building_blocks(ctx)   -> reads the last ValidationReport
    apply_fixes(ctx)           -> reads the report AND every fix result

The conversation is the state store, so each route needs its own two-turn
agent conversation. Driven programmatically here rather than through the web
UI, which is both faster and lets us capture tool returns directly instead of
scraping rendered text.

SCORING
`apply_fixes` returns a re-validated ValidationReport of the corrected route.
We rebuild that route into SynLlama's wire format and score it with the SAME
validator used for the baseline (copied verbatim in run_bench.py), so
before/after numbers are directly comparable.

No GPU is used: validation and correction are pure RDKit, and the orchestrator
is a hosted API.
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from run_bench import classify  # noqa: E402  (verbatim baseline validator)

csv.field_size_limit(10**7)

CORRECTOR_TOOLS = {
    "fix_building_blocks", "fix_step", "apply_fixes",
    "fix_smiles", "fix_smarts", "fix_template",
    "extract_template_from_reaction", "search_step_building_blocks",
}


def report_to_route(report: dict) -> dict | None:
    """Rebuild a SynLlama-format route from a ValidationReport.

    The report carries everything needed: per-reaction template, reactants and
    expected product, plus the building-block list. Converting back lets the
    corrected route be scored by the same validator as the original.
    """
    try:
        reactions = [
            {
                "reaction_number": r["reaction_number"],
                "reaction_template": r["reaction_template"],
                "reactants": r["reactant_smiles"],
                "product": r["expected_product"],
            }
            for r in report["reactions"]
        ]
        blocks = [b["smiles"] for b in report["building_blocks"]]
    except (KeyError, TypeError):
        return None
    if not reactions:
        return None
    return {"reactions": reactions, "building_blocks": blocks}


def tool_calls_and_returns(messages) -> tuple[list[str], dict | None]:
    """Pull the tool names the agent invoked, and the last apply_fixes payload."""
    from pydantic_ai.messages import ModelResponse, ModelRequest

    called: list[str] = []
    apply_payload = None
    for m in messages:
        if isinstance(m, ModelResponse):
            for part in m.parts:
                if getattr(part, "part_kind", "") == "tool-call":
                    called.append(part.tool_name)
        elif isinstance(m, ModelRequest):
            for part in m.parts:
                if getattr(part, "part_kind", "") == "tool-return":
                    if part.tool_name == "apply_fixes":
                        c = part.content
                        apply_payload = (
                            c if isinstance(c, dict)
                            else getattr(c, "model_dump", lambda: None)()
                        )
    return called, apply_payload


async def repair_one(agent, route_json: str, target: str, timeout_s: int) -> dict:
    """Two-turn conversation: validate, then fix."""
    rec: dict = {"target": target, "before": classify(route_json)}
    t0 = time.time()
    try:
        r1 = await asyncio.wait_for(
            agent.run(f"Validate this synthesis route and report every step:\n\n{route_json}"),
            timeout=timeout_s,
        )
        history = r1.all_messages()

        # "fix" is required -- Corrector.prepare_tools() hides every corrector
        # tool unless fix/correct/repair appears in the last 3 user messages.
        r2 = await asyncio.wait_for(
            agent.run("Now fix the failed steps in that route.", message_history=history),
            timeout=timeout_s,
        )
        called, payload = tool_calls_and_returns(r2.all_messages())

        rec["tools_called"] = called
        rec["corrector_fired"] = sorted(set(called) & CORRECTOR_TOOLS)
        rec["applied"] = payload is not None

        if payload:
            corrected = report_to_route(payload)
            if corrected:
                rec["after"] = classify(json.dumps(corrected))
                rec["corrected_route"] = json.dumps(corrected)
            else:
                rec["after"] = "no_route_rebuilt"
        else:
            rec["after"] = "no_apply_fixes"
    except asyncio.TimeoutError:
        rec["after"], rec["error"] = "timeout", f"exceeded {timeout_s}s"
    except Exception as e:
        rec["after"], rec["error"] = "exception", f"{type(e).__name__}: {e}"

    rec["seconds"] = round(time.time() - t0, 1)
    rec["repaired"] = rec["before"] != "valid" and rec.get("after") == "valid"
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="failing routes to attempt")
    ap.add_argument("--model", default="mistral-large-latest")
    ap.add_argument("--provider", default="mistral")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=8.0,
                    help="pause between routes; the API rate-limits hard (429)")
    ap.add_argument("--max-len", type=int, default=1400,
                    help="skip very long routes -- they blow the context and the "
                         "route extractor is size-sensitive")
    ap.add_argument("--out", type=Path, default=HERE / "repair-results.csv")
    args = ap.parse_args()

    if not os.environ.get("MISTRAL_API_KEY"):
        raise SystemExit("MISTRAL_API_KEY not set")

    from synagent.synagent import get_agent

    # Sample failing routes only -- the point is to measure recovery.
    pool = []
    with (HERE / "synllama-raw-failed.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if len(row["response"]) <= args.max_len:
                pool.append(row)
    random.Random(args.seed).shuffle(pool)
    sample = pool[: args.n]
    print(f"{len(pool):,} failing routes under {args.max_len} chars; "
          f"attempting {len(sample)}\n", flush=True)

    agent = get_agent(args.model, provider=args.provider)

    fields = ["target", "before", "after", "repaired", "corrector_fired",
              "tools_called", "applied", "seconds", "error", "corrected_route"]
    rows = []
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(sample, start=1):
            rec = await repair_one(agent, row["response"], row["smiles"], args.timeout)
            rec["tools_called"] = "|".join(rec.get("tools_called", []))
            rec["corrector_fired"] = "|".join(rec.get("corrector_fired", []))
            rows.append(rec)
            w.writerow(rec)
            fh.flush()  # survive interruption

            fired = len(rec["corrector_fired"].split("|")) if rec["corrector_fired"] else 0
            fixed = sum(r["repaired"] for r in rows)
            print(f"  {i:>3}/{len(sample)}  {rec['before']:>8} -> {str(rec.get('after')):<14}"
                  f" tools={fired}  repaired so far {fixed}  ({rec['seconds']}s)",
                  flush=True)
            if i < len(sample):
                await asyncio.sleep(args.sleep)

    n = len(rows)
    fired_any = sum(1 for r in rows if r["corrector_fired"])
    applied = sum(1 for r in rows if r.get("applied"))
    repaired = sum(1 for r in rows if r["repaired"])
    errs = [r for r in rows if r.get("error")]

    print(f"\n{'=' * 62}")
    print(f"CORRECTOR REPAIR RATE  (n={n} failing routes)")
    print(f"  corrector tools fired : {fired_any}/{n} ({fired_any/n*100:.1f}%)")
    print(f"  apply_fixes returned  : {applied}/{n} ({applied/n*100:.1f}%)")
    print(f"  FAILING -> VALID      : {repaired}/{n} ({repaired/n*100:.1f}%)")
    if errs:
        print(f"  errors                : {len(errs)} ({errs[0].get('error','')[:60]})")
    print(f"\n  Baseline for context: 3,065/10,000 = 30.65% of paths valid.")
    if repaired:
        lift = 30.65 + (69.35 * repaired / n)
        print(f"  If this rate held across all 6,935 failures, overall validity")
        print(f"  would go 30.65% -> {lift:.1f}%.")
    print("=" * 62)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
