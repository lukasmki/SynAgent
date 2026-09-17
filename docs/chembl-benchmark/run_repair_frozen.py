#!/usr/bin/env python
"""Run the corrector over every eligible failing route in the frozen ChEMBL
subset -- the one-deterministic-sample-per-target slice of
data/synllama-raw-output.csv that matches SynLlama's own Table 1 methodology
(see make_table1_synagent_column.py).

run_repair.py's existing n=50 experiment (repair-deepseek-n50.csv,
PI_REPORT.md) samples failing routes from the full 10,000-path *mixed*
sampling pool (frozen + low + medium + high). That's a fine general repair-
rate estimate, but it can't be plugged into table1_with_synagent.md, which is
scoped to the frozen subset specifically. This script draws its sample from
that same frozen subset instead, and reuses run_repair.py's agent loop
(`repair_one`) unchanged so the two experiments stay methodologically
comparable.

HANGS
A first full run found extract_template_from_reaction (corrector/_toolset.py)
running Indigo automap + rdchiral's template extractor fully synchronously,
with zero awaits -- so it blocks the asyncio event loop, and *no* timeout,
however set, can fire while it runs (the loop itself has to be free to run
the timeout callback). One route hung 8,927s against a 240s per-call cap
before finally returning on its own. That call is now offloaded to a worker
thread (asyncio.to_thread), which lets a wait_for timeout actually fire --
but a Python thread can't be force-killed, so the slow computation still
finishes eventually in the background. Belt and suspenders: this script also
wraps each whole route in an outer --route-timeout watchdog (default 900s,
comfortably above the normal ~200-800s range but far below the observed
hang), so one pathological route can no longer stall the batch.

RESUME
Safe to Ctrl-C and re-run with the same --out: already-attempted targets
(read back from the existing CSV) are skipped, and new rows are appended
rather than overwriting.

RETRY
--retry-after timeout re-attempts every route currently recorded with that
after-status (comma-separated for more than one, e.g.
--retry-after timeout,route_watchdog_timeout) instead of treating it as
done. Their stale rows are dropped from --out first, then they're re-run
through the normal loop -- so a bump in --timeout/--route-timeout actually
gets applied to them, rather than the value from whichever earlier run
first attempted them.

Scored two ways on the way out:
  - run_bench.py's classify() -- SynLlama's original strict per-route rule,
    for continuity with the existing repair-deepseek-n50 numbers.
  - compare_synllama_synagent.py's all_scores() -- strict + analog-aware,
    permutation-tolerant SynAgent scoring, for the Table 1 column.
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
csv.field_size_limit(10**7)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from rdkit import RDLogger, rdBase  # noqa: E402

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")

from run_repair import repair_one  # noqa: E402  (reuse the agent loop verbatim)
from compare_synllama_synagent import all_scores  # noqa: E402  (reuse scoring)
from synagent.validation._toolset import _parse_route_json, _validate_route_dict  # noqa: E402

FIELDS = ["target", "before", "after", "repaired", "corrector_fired",
          "tools_called", "applied", "fix_attempts", "seconds", "error",
          "corrected_route"]


def frozen_failures(max_len: int) -> list[dict]:
    """Every frozen-subset route that fails SynAgent's own strict validator."""
    source = ROOT / "data" / "synllama-raw-output.csv"
    pool = []
    with source.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["sampling_params"] != "frozen":
                continue
            try:
                route = _parse_route_json(row["response"])
                report = _validate_route_dict(route, analog_product_threshold=None)
                ok = report.all_building_blocks_valid and report.all_reactions_passed
            except Exception:
                ok = False
            if not ok and len(row["response"]) <= max_len:
                pool.append(row)
    return pool


def already_done(out: Path) -> set[str]:
    if not out.exists() or out.stat().st_size == 0:
        return set()
    with out.open(encoding="utf-8", newline="") as fh:
        return {row["target"] for row in csv.DictReader(fh)}


def drop_rows_for_retry(out: Path, retry_statuses: set[str]) -> int:
    """Rewrite --out with rows whose after-status is being retried removed.

    Returns how many were dropped (i.e. how many will be re-attempted).
    """
    with out.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    kept = [r for r in rows if r["after"] not in retry_statuses]
    dropped = len(rows) - len(kept)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept)
    return dropped


async def repair_one_guarded(agent, route_json: str, target: str,
                              call_timeout: int, route_timeout: int) -> dict:
    """repair_one, with a hard wall-clock ceiling on the whole route.

    repair_one already caps each individual model call at call_timeout, but
    that only works if the event loop stays free to notice the deadline --
    see the module docstring. This is the second, independent guard: no
    matter what hangs inside, one route can take at most route_timeout
    seconds of this script's wall clock before being abandoned.
    """
    t0 = time.time()
    try:
        return await asyncio.wait_for(
            repair_one(agent, route_json, target, call_timeout),
            timeout=route_timeout,
        )
    except asyncio.TimeoutError:
        return {
            "target": target,
            "before": "unknown",
            "after": "route_watchdog_timeout",
            "repaired": False,
            "tools_called": [],
            "corrector_fired": [],
            "applied": False,
            "fix_attempts": 0,
            "seconds": round(time.time() - t0, 1),
            "error": f"exceeded outer route timeout ({route_timeout}s)",
            "corrected_route": "",
        }


def write_summary(out: Path, pool: list[dict]) -> dict:
    """Recompute the full before/after summary from every row on disk."""
    by_target = {row["smiles"]: row["response"] for row in pool}
    with out.open(encoding="utf-8", newline="") as fh:
        done_rows = list(csv.DictReader(fh))

    n = len(done_rows)
    fired_any = sum(1 for r in done_rows if r["corrector_fired"])
    applied = sum(1 for r in done_rows if r["applied"] in ("True", "true", "1"))
    repaired_strict_synllama = sum(1 for r in done_rows if r["repaired"] in ("True", "true", "1"))

    metric_names = list(all_scores("{}").keys())
    before_counts = {m: 0 for m in metric_names}
    after_counts = {m: 0 for m in metric_names}
    for row in done_rows:
        original = by_target.get(row["target"])
        if original is None:
            continue
        before_scores = all_scores(original)
        corrected = (row.get("corrected_route") or "").strip()
        after_scores = all_scores(corrected) if corrected else None
        for m in metric_names:
            before_counts[m] += before_scores[m]["passed"]
            after_counts[m] += bool(after_scores and after_scores[m]["passed"])

    summary = {
        "n_attempted": n,
        "n_total_eligible": len(pool),
        "corrector_fired": fired_any,
        "apply_fixes_returned": applied,
        "repaired_strict_synllama_rule": repaired_strict_synllama,
        "before_synagent": before_counts,
        "after_synagent": after_counts,
    }
    (out.parent / "frozen-repair-full-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def print_summary(summary: dict) -> None:
    n = summary["n_attempted"]
    print(f"\n{'=' * 62}")
    print(f"FROZEN-SUBSET CORRECTOR REPAIR  "
          f"(n={n}/{summary['n_total_eligible']} eligible routes attempted)")
    print(f"  corrector tools fired      : {summary['corrector_fired']}/{n} "
          f"({summary['corrector_fired']/n*100:.1f}%)")
    print(f"  apply_fixes returned       : {summary['apply_fixes_returned']}/{n} "
          f"({summary['apply_fixes_returned']/n*100:.1f}%)")
    print(f"  strict SynLlama rule fixed : {summary['repaired_strict_synllama_rule']}/{n} "
          f"({summary['repaired_strict_synllama_rule']/n*100:.1f}%)")
    for m, b in summary["before_synagent"].items():
        a = summary["after_synagent"][m]
        print(f"  {m:<18} before {b:>3}/{n} ({b/n*100:5.1f}%)  "
              f"after {a:>3}/{n} ({a/n*100:5.1f}%)")
    print("=" * 62)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = every remaining eligible route")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--timeout", type=int, default=150,
                    help="per model-call cap, seconds (passed into repair_one)")
    ap.add_argument("--route-timeout", type=int, default=900,
                    help="hard wall-clock cap per whole route, seconds -- "
                         "the guard against a single stuck route (see module docstring)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="pause between routes; 0 is fine for a local model")
    ap.add_argument("--max-len", type=int, default=1400,
                    help="skip very long routes -- matches run_repair.py's default")
    ap.add_argument("--out", type=Path,
                    default=HERE / "comparison-2026-08-27" / "frozen-repair-full.csv")
    ap.add_argument("--retry-after", default="",
                    help="comma-separated after-statuses to re-attempt instead of "
                         "treating as done, e.g. timeout,route_watchdog_timeout")
    args = ap.parse_args()

    retry_statuses = {s.strip() for s in args.retry_after.split(",") if s.strip()}
    if retry_statuses and args.out.exists() and args.out.stat().st_size > 0:
        dropped = drop_rows_for_retry(args.out, retry_statuses)
        print(f"retry mode: dropped {dropped} rows with after in {sorted(retry_statuses)} "
              f"-- they'll be re-attempted", flush=True)

    pool = frozen_failures(args.max_len)
    done = already_done(args.out)
    remaining = [r for r in pool if r["smiles"] not in done]
    if done:
        print(f"resuming: {len(done)} already done, {len(remaining)} remaining "
              f"of {len(pool)} eligible", flush=True)
    if args.n:
        remaining = remaining[: args.n]
    print(f"attempting {len(remaining)} routes this run "
          f"(model={args.model}, provider={args.provider}, "
          f"call-timeout={args.timeout}s, route-timeout={args.route_timeout}s)\n",
          flush=True)

    from synagent.synagent import get_agent

    agent = get_agent(args.model, provider=args.provider)

    args.out.parent.mkdir(exist_ok=True)
    write_header = not (args.out.exists() and args.out.stat().st_size > 0)
    with args.out.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        n_repaired_so_far = 0
        for i, row in enumerate(remaining, start=1):
            rec = await repair_one_guarded(
                agent, row["response"], row["smiles"], args.timeout, args.route_timeout
            )
            rec["tools_called"] = "|".join(rec.get("tools_called", []) or [])
            rec["corrector_fired"] = "|".join(rec.get("corrector_fired", []) or [])
            w.writerow(rec)
            fh.flush()  # survive interruption

            fired = len(rec["corrector_fired"].split("|")) if rec["corrector_fired"] else 0
            n_repaired_so_far += bool(rec.get("repaired"))
            print(f"  {i:>4}/{len(remaining)}  {rec['before']:>8} -> {str(rec.get('after')):<22}"
                  f" tools={fired}  repaired so far {n_repaired_so_far}  ({rec['seconds']}s)",
                  flush=True)
            if i < len(remaining) and args.sleep:
                await asyncio.sleep(args.sleep)

    summary = write_summary(args.out, pool)
    print_summary(summary)
    print(f"wrote {args.out}")
    print(f"wrote {args.out.parent / 'frozen-repair-full-summary.json'}")


if __name__ == "__main__":
    asyncio.run(main())
