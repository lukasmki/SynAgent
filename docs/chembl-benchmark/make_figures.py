#!/usr/bin/env python
"""Generate figures for the ChEMBL benchmark report.

Reads whatever data exists so it can be re-run as the benchmark fills in:
  synllama-raw-output.csv   the 10k-path SynLlama baseline
  synagent-pilot-100.csv    our run (partial file is fine)
  agent-screenshots/agent_runs.json   corrector-chain results
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

csv.field_size_limit(10**7)
HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
from run_bench import classify  # noqa: E402

INK, GRID = "#22252a", "#d8dce2"
OK, BAD, ACC = "#3f8f5f", "#b4483c", "#4a6fa5"
BUCKETS = ["valid", "reaction", "reactant", "product", "smiles", "json", "other"]


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def load_baseline() -> Counter:
    c = Counter()
    with (HERE / "synllama-raw-output.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c[classify(row["response"])] += 1
    return c


def load_ours() -> Counter:
    """Our run's results.

    The harness only writes its CSV at completion, and this run was stopped at
    400/1000 to free the GPU (it was sitting at 98 C). So the per-row error
    breakdown was never written and only the aggregate survives, taken from the
    last progress line in pilot.log:

        400/1000  valid  121 (30.25%)

    Fixing the harness to append incrementally is on the list -- losing the
    breakdown to a stop is a real flaw, not an acceptable trade.
    """
    p = HERE / "synagent-pilot-100.csv"
    c = Counter()
    if p.exists():
        with p.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c[row["result"]] += 1
        return c

    # Aggregate-only fallback from the stopped run.
    log = HERE / "pilot.log"
    if log.exists():
        import re
        last = None
        for line in log.read_text(errors="ignore").splitlines():
            m = re.search(r"(\d+)/1000\s+valid\s+(\d+)", line)
            if m:
                last = m
        if last:
            total, valid = int(last.group(1)), int(last.group(2))
            c["valid"] = valid
            c["_unbroken_failures"] = total - valid
    return c


def total(c: Counter) -> int:
    """Total paths, including the aggregate-only failure sentinel.

    `_unbroken_failures` holds the failures from a run that was stopped before
    the per-row CSV was written, so it MUST be in the denominator. Filtering it
    out (as an earlier version did, to skip 'private' keys) made the denominator
    equal the valid count and rendered the bar at 100.0% -- visually claiming
    the 1B reproduction tripled the baseline.
    """
    return sum(c.values())


def fig_validity(base: Counter, ours: Counter) -> None:
    bt, ot = total(base), total(ours)
    labels = [f"SynLlama proper\n(n={bt:,})"]
    vals = [base["valid"] / bt * 100]
    cols = [ACC]
    if ot:
        labels.append(f"SynAgent + 1B GGUF\n(n={ot:,})")
        vals.append(ours["valid"] / ot * 100)
        cols.append(OK)

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=170)
    bars = ax.bar(labels, vals, color=cols, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("Routes passing full validation (%)", fontsize=10, color=INK)
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("Route validity on the same 1,000 ChEMBL targets",
                 fontsize=12, fontweight="bold", color=INK, pad=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_validity.png", bbox_inches="tight")
    plt.close(fig)


def fig_errors(base: Counter, ours: Counter) -> None:
    bt, ot = total(base), total(ours)
    # Only plot our breakdown if we actually have one -- the stopped run has
    # aggregate counts only, and inventing a split would be worse than omitting it.
    have_breakdown = any(ours.get(b, 0) for b in BUCKETS if b != "valid")
    fails = [b for b in BUCKETS if b != "valid"]
    bvals = [base.get(b, 0) / bt * 100 for b in fails]

    fig, ax = plt.subplots(figsize=(7.4, 4.2), dpi=170)
    x = range(len(fails))
    w = 0.38 if have_breakdown else 0.55
    ax.bar([i - w / 2 for i in x] if have_breakdown else list(x), bvals, w,
           label=f"SynLlama proper (n={bt:,})", color=ACC)
    if have_breakdown:
        ovals = [ours.get(b, 0) / ot * 100 for b in fails]
        ax.bar([i + w / 2 for i in x], ovals, w,
               label=f"SynAgent + 1B GGUF (n={ot:,})", color=OK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        ["reaction\nno products", "reactants\nmismatch", "wrong\nproduct",
         "invalid\nSMILES", "bad\nJSON", "other"],
        fontsize=8.5, color=INK,
    )
    ax.set_ylabel("% of all paths", fontsize=10, color=INK)
    ax.set_title("Failure mode breakdown", fontsize=12, fontweight="bold",
                 color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_errors.png", bbox_inches="tight")
    plt.close(fig)


def fig_agent() -> None:
    p = HERE / "agent-screenshots" / "agent_runs.json"
    if not p.exists():
        return
    runs = json.loads(p.read_text(encoding="utf-8"))
    names = [r["case"] for r in runs]
    validated = [1 if r["called_validate_route"] else 0 for r in runs]
    corrected = [r["n_corrector_tools"] for r in runs]

    fig, ax = plt.subplots(figsize=(6.6, 3.8), dpi=170)
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], validated, 0.4, label="validate_route called",
           color=ACC)
    ax.bar([i + 0.2 for i in x], corrected, 0.4,
           label="corrector tools fired", color=OK)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}\nerror case" for n in names], fontsize=9, color=INK)
    ax.set_ylabel("count", fontsize=10, color=INK)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_title("Agent tool invocation on real failing routes",
                 fontsize=12, fontweight="bold", color=INK, pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_agent_tools.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("scoring baseline (10k paths)...", flush=True)
    base = load_baseline()
    ours = load_ours()
    ot = sum(ours.values())
    print(f"baseline valid: {base['valid']}/{sum(base.values())}")
    print(f"ours     valid: {ours.get('valid',0)}/{ot}" if ot else "ours: no data yet")

    fig_validity(base, ours)
    fig_errors(base, ours)
    fig_agent()

    stats = {
        "baseline": {k: base.get(k, 0) for k in BUCKETS},
        "baseline_total": sum(base.values()),
        "ours": {k: ours.get(k, 0) for k in BUCKETS},
        "ours_total": ot,
    }
    (HERE / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nwrote figures to {FIG}")
    for f in sorted(FIG.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
