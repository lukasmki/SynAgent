#!/usr/bin/env python
"""Run SynAgent's retrosynthesis tool over the same ChEMBL targets SynLlama was
tested on, and score with SynLlama's own validator so the numbers are comparable.

BASELINE (from data/synllama-raw-{output,valid,failed}.csv in the SynAgent repo)
    10,000 paths = 1,000 targets x 10 samples
     3,065 valid  (30.65%)
     6,935 failed (69.35%)

WHAT IS AND ISN'T COMPARABLE
The validator here is copied verbatim from data/synllama-validate.py, so the
pass criterion is identical: JSON parses, every building block and SMILES is
RDKit-valid, the reactants substructure-match the template, the reaction
actually runs, and the stated product is among the generated products.

The MODEL is not the same. The baseline came from SynLlama proper; this runs
lukaskim/SynLlama-1B-GGUF Q6_K, a quantised community reproduction. So a gap in
either direction is a statement about that checkpoint, not about SynAgent's
plumbing. Worth stating plainly whenever the number is quoted.

Prompt construction is SynAgent's own, taken from ModelCallsToolset.retrosynthesis.
"""

import argparse
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdChemReactions

RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.*")

csv.field_size_limit(10**7)

HERE = Path(__file__).parent
BASE_URL = "http://127.0.0.1:8080/v1/completions"
MODEL = "SynLlama-1B-Q6_K"

# Verbatim from SynAgent's model_calls toolset.
SYNLLAMA_SYSTEM = (
    "### Instruction:\n"
    "You are an expert synthetic organic chemist. Your task is to "
    "design a synthesis pathway for a given target molecule using "
    "common and reliable reaction templates and building blocks. "
    "Follow these instructions:\n\n"
    "1. **Input the SMILES String:** Read in the SMILES string of "
    "the target molecule and identify common reaction templates "
    "that can be applied.\n\n"
    "2. **Decompose the Target Molecule:** Use the identified "
    "reaction templates to decompose the target molecule into "
    "different intermediates.\n\n"
    "3. **Check for Building Blocks:** For each intermediate:\n"
    "    - Identify if it is a building block. If it is, wrap it "
    "in <bb> and </bb> tags and save it for later use.\n"
    "    - If it is not a building block, apply additional reaction "
    "templates to further decompose it into building blocks.\n\n"
    "4. **Document Reactions:** For each reaction documented in the "
    "output, wrap the reaction template in <rxn> and </rxn> tags.\n\n"
    "5. **Repeat the Process:** Continue this process until all "
    "intermediates are decomposed into building blocks, and document "
    "each step clearly in a structured JSON format.\n\n"
)


# ── validator: copied from data/synllama-validate.py, unchanged ───────────


class SmilesError(ValueError): ...
class ReactantError(ValueError): ...
class ReactionError(ValueError): ...
class ProductsError(ValueError): ...


def validate(response: str) -> None:
    data = json.loads(response)

    for bb in data["building_blocks"]:
        if bb.startswith("<bb>"):
            bb = bb[4:]
        if bb.endswith("</bb>"):
            bb = bb[:-5]
        if Chem.MolFromSmiles(bb) is None:
            raise SmilesError("Invalid building block SMILES")

    for reaction in data["reactions"]:
        tmpl = reaction["reaction_template"]
        if tmpl.startswith("<rxn>"):
            tmpl = tmpl[5:]
        if tmpl.endswith("</rxn>"):
            tmpl = tmpl[:-6]

        rxn = rdChemReactions.ReactionFromSmarts(tmpl)
        rxn.Initialize()

        reactants = [Chem.MolFromSmiles(m) for m in reaction["reactants"] if m != ""]
        product = Chem.MolFromSmiles(reaction["product"])
        if any(m is None for m in reactants) or product is None:
            raise SmilesError("Invalid smiles")

        for reactant in reactants:
            for template in rxn.GetReactants():
                if reactant.HasSubstructMatch(template):
                    break
            else:
                raise ReactantError("Provided reactants don't match template")

        products = [
            Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
            for p in rxn.RunReactants(reactants)
            for m in p
        ]
        if not products:
            raise ReactionError("Reaction produced no products")

        expected = Chem.CanonSmiles(
            Chem.MolToSmiles(product, canonical=True, ignoreAtomMapNumbers=True)
        )
        for prod in products:
            try:
                if expected == Chem.CanonSmiles(prod):
                    break
            except Exception:
                continue
        else:
            raise ProductsError("Expected product not in provided products")


def classify(response: str) -> str:
    """Return 'valid' or the error bucket, matching the baseline's categories."""
    try:
        validate(response)
        return "valid"
    except json.JSONDecodeError:
        return "json"
    except SmilesError:
        return "smiles"
    except ReactantError:
        return "reactant"
    except ReactionError:
        return "reaction"
    except ProductsError:
        return "product"
    except Exception:
        return "other"


# ── generation ───────────────────────────────────────────────────────────


def generate(smiles: str, temperature: float, top_p: float, timeout: int = 180) -> str:
    prompt = (
        f"{SYNLLAMA_SYSTEM}"
        f"### Input:\n"
        f"Provide a synthetic pathway for this SMILES string: {smiles}\n\n"
        f"### Response:\n"
    )
    body = json.dumps(
        {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": 1024,   # 256 truncates routes mid-JSON
            "temperature": temperature,
            "top_p": top_p,
            "stop": ["### Input:", "### Instruction:"],
        }
    ).encode()
    req = urllib.request.Request(
        BASE_URL, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["text"]


def clean(raw: str) -> str:
    """Strip markdown fences the model sometimes emits around the JSON."""
    t = raw.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.replace("```json", "").replace("```", "")
    i, j = t.find("{"), t.rfind("}")
    return t[i : j + 1] if i != -1 and j > i else t


def one(args) -> dict:
    smiles, sample_idx, temperature, top_p = args
    t0 = time.time()
    try:
        raw = generate(smiles, temperature, top_p)
        resp = clean(raw)
        bucket = classify(resp)
        err = ""
    except Exception as e:
        raw, resp, bucket, err = "", "", "request", f"{type(e).__name__}: {e}"
    return {
        "smiles": smiles,
        "sample": sample_idx,
        "response": resp,
        "result": bucket,
        "error": err,
        "seconds": round(time.time() - t0, 1),
    }


def guard_single_instance() -> None:
    """Refuse to start if another run is already going.

    Two instances were once launched 52 seconds apart -- one via nohup that
    appeared to have failed, one via Start-Process. They interleaved into the
    same log (producing two different values for the same checkpoint) and
    halved each other's throughput competing for the server. Cheap to prevent.
    """
    lock = HERE / ".bench.lock"
    if lock.exists():
        age = time.time() - lock.stat().st_mtime
        if age < 3600:
            raise SystemExit(
                f"Another run appears active (lock {age/60:.0f} min old).\n"
                f"If that is wrong, delete {lock}"
            )
    lock.write_text(str(time.time()))


def main() -> None:
    guard_single_instance()
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=int, default=50,
                    help="How many of the 1000 ChEMBL targets to run. 0 = all.")
    ap.add_argument("--samples", type=int, default=10,
                    help="Paths per target. Baseline used 10.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent requests; match llama-server --parallel.")
    ap.add_argument("--temperature", type=float, default=1.5)   # SynAgent default
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=HERE / "synagent-output.csv")
    args = ap.parse_args()

    # Same target list the baseline used, in file order.
    with (HERE / "synllama-raw-output.csv").open(encoding="utf-8") as f:
        seen, targets = set(), []
        for row in csv.DictReader(f):
            s = row["smiles"]
            if s not in seen:
                seen.add(s)
                targets.append(s)
    print(f"{len(targets)} unique ChEMBL targets in the baseline file", flush=True)

    if args.targets:
        random.Random(args.seed).shuffle(targets)
        targets = targets[: args.targets]

    jobs = [
        (s, i, args.temperature, args.top_p)
        for s in targets
        for i in range(args.samples)
    ]
    print(f"running {len(targets)} targets x {args.samples} samples "
          f"= {len(jobs)} generations, {args.workers} workers\n", flush=True)

    # Write incrementally, not at the end. A previous run was stopped at
    # 400/1000 to free an overheating GPU and the entire per-row error
    # breakdown was lost, leaving only the aggregate from the progress log.
    # A long run must survive being interrupted.
    fields = ["smiles", "sample", "response", "result", "error", "seconds"]
    rows, t0 = [], time.time()
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for n, rec in enumerate(pool.map(one, jobs), start=1):
                rows.append(rec)
                writer.writerow(rec)
                if n % 25 == 0 or n == len(jobs):
                    fh.flush()          # partial file must be readable mid-run
                    ok = sum(r["result"] == "valid" for r in rows)
                    el = time.time() - t0
                    print(f"  {n:>5}/{len(jobs)}  valid {ok:>4} ({ok/n*100:5.2f}%)  "
                          f"{el/n:.1f}s/gen  eta {(len(jobs)-n)*el/n/60:5.1f}m",
                          flush=True)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["result"]] = counts.get(r["result"], 0) + 1
    valid = counts.get("valid", 0)

    print(f"\n{'=' * 58}")
    print(f"SynAgent + SynLlama-1B-GGUF Q6_K")
    print(f"  valid: {valid}/{len(rows)} = {valid/len(rows)*100:.2f}%")
    print(f"\n  error breakdown:")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        if k != "valid":
            print(f"    {k:<10} {v:>5}  ({v/len(rows)*100:5.2f}%)")
    print(f"\nBASELINE (SynLlama proper, 10k paths): 3065/10000 = 30.65%")
    print(f"{'=' * 58}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
