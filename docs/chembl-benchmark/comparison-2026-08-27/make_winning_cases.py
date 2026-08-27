"""Build auditable case studies from the real n=50 SynAgent correction run."""

import csv
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
ROOT = BENCH.parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))

from synagent.validation._toolset import (  # noqa: E402
    _parse_route_json,
    _validate_route_dict,
)

csv.field_size_limit(10**7)


def reconstruct_original_sample(n=50, seed=42, max_len=1400):
    pool = []
    with (DATA / "synllama-raw-failed.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if len(row["response"]) <= max_len:
                pool.append(row)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def status(route_json: str) -> dict:
    report = _validate_route_dict(
        _parse_route_json(route_json), analog_product_threshold=None
    )
    return {
        "passed": report.all_building_blocks_valid and report.all_reactions_passed,
        "steps_passed": sum(step.status == "passed" for step in report.reactions),
        "steps_total": len(report.reactions),
        "failure_modes": [
            step.failure_mode for step in report.reactions if step.failure_mode
        ],
    }


def main():
    originals = reconstruct_original_sample()
    repairs = list(
        csv.DictReader(
            (BENCH / "repair-deepseek-n50.csv").open(encoding="utf-8", newline="")
        )
    )
    cases = []
    for index, (original, repair) in enumerate(zip(originals, repairs, strict=True), 1):
        if original["smiles"] != repair["target"]:
            raise ValueError(f"Sample mismatch at row {index}")
        corrected = repair.get("corrected_route", "").strip()
        if not corrected:
            continue
        before = status(original["response"])
        after = status(corrected)
        if not before["passed"] and after["passed"]:
            cases.append(
                {
                    "sample_index": index,
                    "target": original["smiles"],
                    "before": before,
                    "after": after,
                    "tools_called": repair["tools_called"].split("|"),
                    "fix_attempts": int(repair["fix_attempts"]),
                    "seconds": float(repair["seconds"]),
                    "original_route": json.loads(original["response"]),
                    "corrected_route": json.loads(corrected),
                }
            )

    (HERE / "winning-cases.json").write_text(
        json.dumps(cases, indent=2), encoding="utf-8"
    )

    shown = cases[:3]
    fig, axes = plt.subplots(len(shown), 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(
        "Three real SynAgent correction wins",
        fontsize=18,
        fontweight="bold",
    )
    for row_index, case in enumerate(shown):
        molecule_ax, evidence_ax = axes[row_index]
        molecule_ax.axis("off")
        molecule = Chem.MolFromSmiles(case["target"])
        molecule_image = Draw.MolToImage(molecule, size=(700, 360))
        molecule_ax.imshow(molecule_image)
        molecule_ax.set_title(
            f"Case {case['sample_index']}: target molecule", fontweight="bold"
        )

        evidence_ax.axis("off")
        before = case["before"]
        after = case["after"]
        tools = " -> ".join(case["tools_called"])
        text = (
            f"SynAgent exact validator before: FAIL\n"
            f"Failure modes: {', '.join(before['failure_modes']) or 'route-level failure'}\n"
            f"Steps passing: {before['steps_passed']}/{before['steps_total']}\n\n"
            f"SynAgent tools called:\n{tools}\n\n"
            f"SynAgent exact validator after: PASS\n"
            f"Steps passing: {after['steps_passed']}/{after['steps_total']}\n"
            f"Fix attempts: {case['fix_attempts']} | Runtime: {case['seconds']:.1f}s"
        )
        evidence_ax.text(
            0.02,
            0.96,
            text,
            va="top",
            ha="left",
            fontsize=11,
            linespacing=1.45,
            bbox={
                "boxstyle": "round,pad=0.8",
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
            },
        )

    fig.text(
        0.5,
        -0.015,
        "Source: repair-deepseek-n50.csv. Routes re-scored with exact-product validation after reconstruction.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    fig.savefig(
        HERE / "three-synagent-wins.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        HERE / "three-synagent-wins.pdf", bbox_inches="tight", facecolor="white"
    )
    print(f"strict wins: {len(cases)}; displayed: {len(shown)}")


if __name__ == "__main__":
    main()
