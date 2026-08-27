#!/usr/bin/env python
"""Compare SynLlama alone with SynAgent + SynLlama under paired validators.

Four route-level metrics are reported so product analog acceptance is never
confused with the published benchmark:

1. SynLlama published: emitted reactant order + exact product.
2. SynLlama analog-aware: emitted reactant order + Morgan/Tanimoto > 0.60.
3. SynAgent strict: any reactant order + exact product.
4. SynAgent analog-aware: any reactant order + 4096-bit Morgan/Tanimoto > 0.60.

The n=50 corrector sample is reconstructed from the original deterministic
sampling procedure and evaluated before/after under all four metrics.
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdChemReactions

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
OUT = HERE / "comparison-2026-08-27"
OUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "src"))

from synagent.validation._toolset import (  # noqa: E402
    ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    _match_product,
    _parse_route_json,
    _validate_route_dict,
)

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")
csv.field_size_limit(10**7)


def _strip(value: str, tag: str) -> str:
    start, end = f"<{tag}>", f"</{tag}>"
    if value.startswith(start):
        value = value[len(start) :]
    if value.endswith(end):
        value = value[: -len(end)]
    return value


def score_emitted_order(response: str, analog_threshold: float | None) -> dict:
    """Score with SynLlama's original emitted-order validator plus match detail."""
    result = {
        "passed": False,
        "bucket": "other",
        "analog_steps": 0,
        "max_similarity": None,
    }
    try:
        data = json.loads(response)
        for bb in data["building_blocks"]:
            if Chem.MolFromSmiles(_strip(str(bb), "bb")) is None:
                result["bucket"] = "smiles"
                return result

        similarities = []
        for reaction in data["reactions"]:
            template = _strip(str(reaction["reaction_template"]), "rxn")
            rxn = rdChemReactions.ReactionFromSmarts(template)
            rxn.Initialize()

            reactants = [
                Chem.MolFromSmiles(smiles)
                for smiles in reaction["reactants"]
                if smiles != ""
            ]
            expected = str(reaction["product"])
            if (
                any(mol is None for mol in reactants)
                or Chem.MolFromSmiles(expected) is None
            ):
                result["bucket"] = "smiles"
                return result

            for reactant in reactants:
                if not any(
                    reactant.HasSubstructMatch(pattern)
                    for pattern in rxn.GetReactants()
                ):
                    result["bucket"] = "reactant"
                    return result

            products = []
            for product_tuple in rxn.RunReactants(reactants):
                for mol in product_tuple:
                    try:
                        Chem.SanitizeMol(mol)
                        smiles = Chem.MolToSmiles(
                            mol, canonical=True, ignoreAtomMapNumbers=True
                        )
                        if smiles not in products:
                            products.append(smiles)
                    except Exception:
                        continue
            if not products:
                result["bucket"] = "reaction"
                return result

            matched, match_type, _, similarity = _match_product(
                expected, products, analog_threshold
            )
            if similarity is not None:
                similarities.append(similarity)
            if not matched:
                result["bucket"] = "product"
                result["max_similarity"] = max(similarities, default=None)
                return result
            if match_type == "analog":
                result["analog_steps"] += 1

        result.update(
            passed=True,
            bucket="valid",
            max_similarity=max(similarities, default=None),
        )
        return result
    except json.JSONDecodeError:
        result["bucket"] = "json"
    except Exception:
        result["bucket"] = "other"
    return result


def score_synagent(response: str, analog_threshold: float | None) -> dict:
    """Score with SynAgent's production validator (reactant permutations enabled)."""
    try:
        report = _validate_route_dict(
            _parse_route_json(response), analog_product_threshold=analog_threshold
        )
    except json.JSONDecodeError:
        return {"passed": False, "bucket": "json", "analog_steps": 0}
    except Exception:
        return {"passed": False, "bucket": "other", "analog_steps": 0}

    if report.all_building_blocks_valid and report.all_reactions_passed:
        return {
            "passed": True,
            "bucket": "valid",
            "analog_steps": sum(
                reaction.product_match_type == "analog" for reaction in report.reactions
            ),
        }
    if not report.all_building_blocks_valid:
        return {"passed": False, "bucket": "smiles", "analog_steps": 0}
    failure = next(
        (
            reaction.failure_mode
            for reaction in report.reactions
            if reaction.status == "failed"
        ),
        "other",
    )
    buckets = {
        "invalid_reactant_smiles": "smiles",
        "invalid_product_smiles": "smiles",
        "invalid_template": "other",
        "no_products": "reaction",
        "wrong_product": "product",
    }
    return {"passed": False, "bucket": buckets.get(failure, "other"), "analog_steps": 0}


def all_scores(response: str) -> dict:
    return {
        "synllama_strict": score_emitted_order(response, None),
        "synllama_analog": score_emitted_order(
            response, ANALOG_PRODUCT_SIMILARITY_THRESHOLD
        ),
        "synagent_strict": score_synagent(response, None),
        "synagent_analog": score_synagent(
            response, ANALOG_PRODUCT_SIMILARITY_THRESHOLD
        ),
    }


def score_full_baseline() -> tuple[list[dict], dict]:
    rows = []
    counts = {name: Counter() for name in all_scores("{}")}
    source = DATA / "synllama-raw-output.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    for index, row in enumerate(source_rows, start=1):
        scores = all_scores(row["response"])
        record = {
            "index": index,
            "target": row["smiles"],
            "sampling_params": row["sampling_params"],
        }
        for name, score in scores.items():
            counts[name][score["bucket"]] += 1
            record[f"{name}_passed"] = score["passed"]
            record[f"{name}_bucket"] = score["bucket"]
            record[f"{name}_analog_steps"] = score["analog_steps"]
        rows.append(record)
        if index % 500 == 0:
            print(f"baseline {index:,}/{len(source_rows):,}", flush=True)

    summary = {
        "n": len(rows),
        "analog_threshold": ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
        "metrics": {
            name: {
                "valid": counter["valid"],
                "valid_percent": round(counter["valid"] / len(rows) * 100, 2),
                "buckets": dict(counter),
            }
            for name, counter in counts.items()
        },
    }
    return rows, summary


def reconstruct_repair_sample(n: int = 50, seed: int = 42, max_len: int = 1400):
    pool = []
    with (DATA / "synllama-raw-failed.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if len(row["response"]) <= max_len:
                pool.append(row)
    random.Random(seed).shuffle(pool)
    return pool[:n]


def score_repair_sample() -> tuple[list[dict], dict]:
    original = reconstruct_repair_sample()
    with (HERE / "repair-deepseek-n50.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        repaired = list(csv.DictReader(handle))
    if len(original) != len(repaired):
        raise ValueError("Repair CSV length does not match reconstructed sample")

    output = []
    metric_names = list(all_scores("{}").keys())
    before_counts = Counter()
    after_counts = Counter()
    for index, (before_row, after_row) in enumerate(
        zip(original, repaired, strict=True), 1
    ):
        if before_row["smiles"] != after_row["target"]:
            raise ValueError(f"Repair sample mismatch at row {index}")
        before_scores = all_scores(before_row["response"])
        corrected_route = after_row.get("corrected_route", "").strip()
        after_scores = all_scores(corrected_route) if corrected_route else None
        record = {
            "index": index,
            "target": before_row["smiles"],
            "tools_called": after_row.get("tools_called", ""),
            "applied": after_row.get("applied", ""),
            "error": after_row.get("error", ""),
        }
        for name in metric_names:
            before_pass = before_scores[name]["passed"]
            after_pass = bool(after_scores and after_scores[name]["passed"])
            record[f"before_{name}_passed"] = before_pass
            record[f"after_{name}_passed"] = after_pass
            before_counts[name] += before_pass
            after_counts[name] += after_pass
        output.append(record)

    summary = {
        "n": len(output),
        "selection": "seed=42, failing routes <=1400 chars",
        "metrics": {
            name: {
                "before_valid": before_counts[name],
                "after_valid": after_counts[name],
                "absolute_gain": after_counts[name] - before_counts[name],
                "after_valid_percent": round(after_counts[name] / len(output) * 100, 2),
            }
            for name in metric_names
        },
    }
    return output, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    baseline_rows, baseline_summary = score_full_baseline()
    repair_rows, repair_summary = score_repair_sample()
    write_csv(OUT / "baseline-10000-path-level.csv", baseline_rows)
    write_csv(OUT / "synagent-repair-n50-paired.csv", repair_rows)
    summary = {
        "method": {
            "synllama": "emitted reactant order",
            "synagent": "all reactant permutations",
            "strict_product": "canonical SMILES equality",
            "analog_product": "Morgan radius=2, 4096-bit Tanimoto > 0.60",
        },
        "baseline": baseline_summary,
        "repair": repair_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
