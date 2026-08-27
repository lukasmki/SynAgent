from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path(__file__).parent

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "figure.dpi": 180,
    }
)


def label_bars(ax, bars, suffix="%"):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.1,
            f"{value:.2f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)

names = [
    "SynLlama\npublished",
    "SynLlama\nanalog-aware",
    "SynAgent\nstrict",
    "SynAgent\nanalog-aware",
]
values = [30.65, 34.20, 59.19, 65.23]
colors = ["#64748b", "#3b82f6", "#f59e0b", "#16a34a"]
bars = axes[0].bar(names, values, color=colors, width=0.68)
axes[0].set_title("Route-level validity on 10,000 SynLlama paths")
axes[0].set_ylabel("Valid routes (%)")
axes[0].set_xlabel("Validation rule")
axes[0].set_ylim(0, 75)
axes[0].grid(axis="y", alpha=0.2)
label_bars(axes[0], bars)
axes[0].text(
    0.01,
    -0.25,
    "Published: emitted reactant order + exact product.\n"
    "SynAgent: reactant permutations; analog-aware adds 4096-bit Morgan/Tanimoto > 0.60.",
    transform=axes[0].transAxes,
    fontsize=8,
    color="#475569",
)

before = [0, 2, 44, 52]
after = [14, 14, 74, 78]
x = range(4)
width = 0.34
before_bars = axes[1].bar(
    [i - width / 2 for i in x],
    before,
    width,
    label="Before correction",
    color="#94a3b8",
)
after_bars = axes[1].bar(
    [i + width / 2 for i in x], after, width, label="After correction", color="#16a34a"
)
axes[1].set_title("Paired corrector evaluation on 50 strict failures")
axes[1].set_ylabel("Valid routes (%)")
axes[1].set_xlabel("Validation rule")
axes[1].set_xticks(
    list(x),
    ["SynLlama\nstrict", "SynLlama\nanalog", "SynAgent\nstrict", "SynAgent\nanalog"],
)
axes[1].set_ylim(0, 90)
axes[1].grid(axis="y", alpha=0.2)
axes[1].legend(frameon=False, loc="upper left")
label_bars(axes[1], before_bars, suffix="%")
label_bars(axes[1], after_bars, suffix="%")
axes[1].text(
    0.01,
    -0.25,
    "Same 50 routes before/after DeepSeek-driven correction.\n"
    "SynAgent analog-aware: 52% to 78%; net +13 routes; exact McNemar p=0.00098.",
    transform=axes[1].transAxes,
    fontsize=8,
    color="#475569",
)

fig.suptitle("SynAgent + SynLlama benchmark summary", fontsize=16, fontweight="bold")
fig.savefig(
    OUT / "synagent-vs-synllama-summary.png", bbox_inches="tight", facecolor="white"
)
fig.savefig(
    OUT / "synagent-vs-synllama-summary.pdf", bbox_inches="tight", facecolor="white"
)
print(OUT / "synagent-vs-synllama-summary.png")
