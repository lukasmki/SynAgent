"""Figure 4: the reactant-ordering finding.

Two panels -- the validity gap created by positional matching, and the
one-directional validator disagreement that revealed it.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from pathlib import Path  # noqa: E402

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

INK, GRID = "#22252a", "#d8dce2"
STRICT, RELAX, BAD = "#b4483c", "#3f8f5f", "#8a8f98"


def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), dpi=170)

# --- panel 1: strict vs order-insensitive -------------------------------
vals = [28.20, 57.00]
bars = a1.bar(
    ["Order must match\ntemplate slots\n(published criterion)",
     "Any reactant order\n(chemistry)"],
    vals, color=[STRICT, RELAX], width=0.5,
)
for b, v in zip(bars, vals):
    a1.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
            ha="center", fontsize=13, fontweight="bold", color=INK)
a1.annotate("", xy=(1, 55), xytext=(0, 30),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
a1.text(0.5, 44, "144 routes rescued\nby reordering alone",
        ha="center", fontsize=9.5, color=INK, style="italic")
a1.set_ylabel("SynLlama routes scored valid (%)", fontsize=10, color=INK)
a1.set_ylim(0, 72)
a1.set_title("Same routes, two grading rules", fontsize=12,
             fontweight="bold", color=INK, pad=12)
style(a1)

# --- panel 2: validator disagreement ------------------------------------
labels = ["both\nFAIL", "both\nPASS", "SynLlama FAIL\nSynAgent PASS",
          "SynLlama PASS\nSynAgent FAIL"]
counts = [215, 141, 144, 0]
cols = [BAD, BAD, STRICT, BAD]
bars = a2.bar(labels, counts, color=cols, width=0.6)
for b, v in zip(bars, counts):
    a2.text(b.get_x() + b.get_width() / 2, v + 4, str(v),
            ha="center", fontsize=11, fontweight="bold", color=INK)
a2.set_ylabel("routes (n=500)", fontsize=10, color=INK)
a2.set_ylim(0, 250)
a2.set_title("Validator disagreement runs one way only", fontsize=12,
             fontweight="bold", color=INK, pad=12)
a2.tick_params(axis="x", labelsize=8)
style(a2)

fig.tight_layout()
out = FIG / "fig4_ordering.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
