"""SmileyLlama -> SynLlama pipeline, driven through SynAgent's own toolset.

This calls ModelCallsToolset.generate_molecules() and .retrosynthesis() —
the exact functions the agent exposes as tools — rather than hitting the
server directly, so a pass here means the agent's tool layer works end to end
against the local quantized models.

Records every trial to trials.json for later write-up.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("VLLM_BASE_URL", "http://127.0.0.1:8080/v1")
os.environ.setdefault("SMILEYLLAMA_MODEL", "SmileyLlama-1B-Q6_K")
os.environ.setdefault("SYNLLAMA_MODEL", "SynLlama-1B-Q6_K")

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

from synagent.model_calls import ModelCallsToolset  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT = Path(__file__).parent / "trials.json"

# Each trial uses a different constraint set so the runs aren't near-duplicates.
CONSTRAINTS = [
    {"mw_range": "<= 500", "logp_range": "<= 5", "hbd_range": "<= 5"},
    {"mw_range": "<= 400", "logp_range": "<= 3"},
    {"mw_range": "<= 300", "hba_range": "<= 5", "rotatable_bonds": "<= 7"},
    {"mw_range": "<= 500", "logp_range": "<= 4", "fsp3": "> 0.4"},
    {"mw_range": "<= 600", "logp_range": "<= 5", "hbd_range": "<= 3"},
    {"mw_range": "<= 400", "logp_range": "<= 5", "hba_range": "<= 10"},
    {"mw_range": "<= 500", "hbd_range": "<= 3", "rotatable_bonds": "<= 10"},
    {"mw_range": "<= 300", "logp_range": "<= 3"},
]


def describe(smiles: str) -> dict:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"valid": False}
    return {
        "valid": True,
        "canonical": Chem.MolToSmiles(m),
        "mw": round(Descriptors.MolWt(m), 1),
        "logp": round(Descriptors.MolLogP(m), 2),
        "hbd": Descriptors.NumHDonors(m),
        "hba": Descriptors.NumHAcceptors(m),
        "rotb": Descriptors.NumRotatableBonds(m),
        "rings": m.GetRingInfo().NumRings(),
        "heavy_atoms": m.GetNumHeavyAtoms(),
    }


def summarize_route(pathway: dict) -> dict:
    """Pull step count, templates and building blocks out of a SynLlama route."""
    reactions = pathway.get("reactions") or pathway.get("steps") or []
    templates, reactants, products = [], [], []
    for rxn in reactions:
        t = rxn.get("reaction_template") or rxn.get("template") or ""
        templates.append(t.replace("<rxn>", "").replace("</rxn>", ""))
        for r in rxn.get("reactants", []) or []:
            if r:
                reactants.append(r.replace("<bb>", "").replace("</bb>", ""))
        p = rxn.get("product")
        if p:
            products.append(p)
    valid_reactants = [r for r in reactants if Chem.MolFromSmiles(r) is not None]
    return {
        "n_steps": len(reactions),
        "templates": templates,
        "reactants": reactants,
        "n_reactants": len(reactants),
        "n_valid_reactants": len(valid_reactants),
        "products": products,
    }


async def trial(toolset: ModelCallsToolset, idx: int, constraints: dict) -> dict:
    rec: dict = {
        "trial": idx,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "constraints": constraints,
    }
    t0 = time.time()

    # --- Stage 1: SmileyLlama ------------------------------------------------
    mols = await toolset.generate_molecules(temperature=0.9, **constraints)
    rec["smiley_seconds"] = round(time.time() - t0, 1)
    raw = mols[0]["raw_output"] if mols else ""
    smiles = mols[0]["smiles"] if mols else ""
    rec["smiley_raw"] = raw[:400]
    rec["smiles"] = smiles
    rec["molecule"] = describe(smiles)

    if not rec["molecule"]["valid"]:
        rec["status"] = "FAIL: SmileyLlama produced invalid SMILES"
        rec["total_seconds"] = round(time.time() - t0, 1)
        return rec

    # --- Stage 2: SynLlama ---------------------------------------------------
    t1 = time.time()
    route = await toolset.retrosynthesis(
        product_smiles=rec["molecule"]["canonical"], temperature=0.7, top_p=0.9
    )
    rec["synllama_seconds"] = round(time.time() - t1, 1)

    pw = route["pathways"][0]
    rec["parse_error"] = pw.get("parse_error", False)
    if pw.get("parse_error"):
        rec["synllama_raw"] = (pw.get("raw_output") or "")[:600]
        rec["status"] = "FAIL: SynLlama route did not parse as JSON"
    else:
        rec["route"] = summarize_route(pw["pathway"])
        if rec["route"]["n_steps"] == 0:
            rec["status"] = "FAIL: route parsed but contained no reaction steps"
        else:
            rec["status"] = "SUCCESS"

    rec["total_seconds"] = round(time.time() - t0, 1)
    return rec


async def main(target_successes: int = 3, max_trials: int = 8) -> None:
    toolset = ModelCallsToolset()
    trials, successes = [], 0

    for i in range(max_trials):
        c = CONSTRAINTS[i % len(CONSTRAINTS)]
        print(f"\n{'=' * 68}\nTRIAL {i + 1}  constraints={c}", flush=True)
        rec = await trial(toolset, i + 1, c)
        trials.append(rec)

        print(f"  SMILES : {rec.get('smiles', '')[:70]}", flush=True)
        mol = rec.get("molecule", {})
        if mol.get("valid"):
            print(
                f"  props  : MW={mol['mw']} logP={mol['logp']} "
                f"HBD={mol['hbd']} HBA={mol['hba']} rings={mol['rings']}",
                flush=True,
            )
        if "route" in rec:
            r = rec["route"]
            print(
                f"  route  : {r['n_steps']} steps, "
                f"{r['n_valid_reactants']}/{r['n_reactants']} reactants valid",
                flush=True,
            )
        print(f"  {rec['status']}  ({rec['total_seconds']}s)", flush=True)

        if rec["status"] == "SUCCESS":
            successes += 1
            if successes >= target_successes:
                print(f"\nReached {successes} successful trials.", flush=True)
                break

    OUT.write_text(
        json.dumps(
            {
                "generated": datetime.now().isoformat(timespec="seconds"),
                "endpoint": os.environ["VLLM_BASE_URL"],
                "models": {
                    "smileyllama": os.environ["SMILEYLLAMA_MODEL"],
                    "synllama": os.environ["SYNLLAMA_MODEL"],
                },
                "n_trials": len(trials),
                "n_success": successes,
                "trials": trials,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n{successes}/{len(trials)} succeeded. Wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
