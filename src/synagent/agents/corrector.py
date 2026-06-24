import json
from itertools import permutations

from pydantic_ai import Agent
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdChemReactions

from .validation import _COMMON_SMARTS, auto_validate_reaction

RDLogger.DisableLog("rdApp.*")

import logfire

logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()


SYSTEM_PROMPT = """You are SynAgent's route corrector.

## Input format
You receive either:
(a) A full route in the format <SMILES>,<difficulty>,<JSON> (the JSON may have doubled
    double-quotes "" from CSV escaping) — this is the same format the validation agent
    parses. When you get this, call correct_route with the ENTIRE raw input string,
    unmodified, exactly as given. Do not parse the JSON yourself and do not extract
    individual reactants/products by hand — correct_route does all of that internally.
(b) A single isolated reaction step: a list of reactant SMILES and an expected product
    SMILES. When you get this, call suggest_reactant_fix with exactly those inputs.

## Rules
Do not assess the chemistry yourself. Do not invent a fix, a mechanism, or a class of
intermediate. Do not fill in gaps the tool left unresolved. Report exactly what the tool
returned for every step, including steps it could not fix — say "unfixable" and quote
the tool's message verbatim rather than proposing your own chemical reasoning.
The tools already re-validate every proposed fix before returning it, so trust their
output completely; never second-guess or soften a "could not be fixed" result into a
plausible-sounding story.

When you call correct_route, also check the "chain_consistent" field and any
"chain_warning" on individual steps. A step being individually "fixed" does not mean the
whole route works — if chain_consistent is false, say so explicitly and report which
step(s) are now disconnected from the rest of the route, quoting the chain_warning
verbatim. Do not claim the route is valid overall unless all_resolved and
chain_consistent are both true.""".strip()

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type=str,
)


def _reverse_smarts(smarts: str) -> str:
    """Swap the two sides of a forward reaction SMARTS to get a retro-template.
    Only valid for atom-economical templates (every atom in the reactants is
    mapped into the product) — true for all entries in _COMMON_SMARTS."""
    lhs, rhs = smarts.split(">>")
    return f"{rhs}>>{lhs}"


def _fingerprint(mol: Chem.Mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(_fingerprint(mol_a), _fingerprint(mol_b))


@agent.tool_plain
def suggest_reactant_fix(
    reactant_smiles: list[str], expected_product: str
) -> dict:
    """Diagnoses a failed reaction step and proposes a corrected reactant list.

    Strategy: if the reaction is already valid, say so. Otherwise, apply the reverse
    of every template in the common SMARTS library to the expected product to generate
    candidate "correct" precursor fragments. Match each candidate fragment to the given
    reactants by Tanimoto similarity. Whichever given reactant has the worst match is
    flagged as the likely corrupted one and replaced with its matching candidate
    fragment. The proposed fix is re-validated with auto_validate_reaction before being
    returned, so a returned fix is guaranteed to actually produce the expected product.

    Args:
        reactant_smiles (list[str]): The (possibly wrong) reactant SMILES from the failed step
        expected_product (str): The expected product SMILES

    Returns:
        dict: {
            "already_valid": bool,
            "fix_found": bool,
            "message": str,
            "corrected_reactants": list[str] | None,
            "matched_smarts": str | None,
        }
    """
    reactant_smiles = [s for s in reactant_smiles if s.strip()]
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactants):
        return {
            "already_valid": False,
            "fix_found": False,
            "message": "`reactant_smiles` contains invalid SMILES strings",
            "corrected_reactants": None,
            "matched_smarts": None,
        }

    product_mol = Chem.MolFromSmiles(expected_product)
    if product_mol is None:
        return {
            "already_valid": False,
            "fix_found": False,
            "message": "`expected_product` is not a valid SMILES string",
            "corrected_reactants": None,
            "matched_smarts": None,
        }

    is_valid, msg = auto_validate_reaction(reactant_smiles, expected_product)
    if is_valid:
        return {
            "already_valid": True,
            "fix_found": False,
            "message": msg,
            "corrected_reactants": None,
            "matched_smarts": None,
        }

    best_fix = None
    for smarts in _COMMON_SMARTS:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(_reverse_smarts(smarts))
            if rxn is None:
                continue
        except Exception:
            continue

        try:
            outputs_list = rxn.RunReactants((product_mol,))
        except Exception:
            continue

        for outputs in outputs_list:
            if len(outputs) != len(reactants):
                continue
            try:
                candidates = [Chem.MolFromSmiles(Chem.MolToSmiles(m, ignoreAtomMapNumbers=True)) for m in outputs]
            except Exception:
                continue
            if any(c is None for c in candidates):
                continue

            # find the assignment of candidates -> given reactants maximizing similarity
            best_perm = None
            best_score = -1.0
            for perm in permutations(range(len(candidates))):
                score = sum(
                    _similarity(reactants[i], candidates[perm[i]])
                    for i in range(len(reactants))
                )
                if score > best_score:
                    best_score = score
                    best_perm = perm

            per_reactant_sim = [
                _similarity(reactants[i], candidates[best_perm[i]]) for i in range(len(reactants))
            ]
            unchanged_flags = [sim >= 0.999 for sim in per_reactant_sim]
            corrected = [
                reactants[i] if unchanged_flags[i] else candidates[best_perm[i]]
                for i in range(len(reactants))
            ]
            corrected_smiles = [Chem.MolToSmiles(m, canonical=True) for m in corrected]

            ok, _ = auto_validate_reaction(corrected_smiles, expected_product)
            if ok:
                # Prefer fixes that leave the most reactants untouched (minimal edit),
                # then break ties by how close the changed reactant(s) are to the originals.
                num_unchanged = sum(unchanged_flags)
                score = (num_unchanged, min(per_reactant_sim))
                if best_fix is None or score > best_fix["score"]:
                    best_fix = {
                        "smarts": smarts,
                        "corrected_smiles": corrected_smiles,
                        "score": score,
                        "num_unchanged": num_unchanged,
                    }

    if best_fix is None:
        return {
            "already_valid": False,
            "fix_found": False,
            "message": "No retro-template produced a confirmed valid fix for this product",
            "corrected_reactants": None,
            "matched_smarts": None,
        }

    num_changed = len(reactant_smiles) - best_fix["num_unchanged"]
    return {
        "already_valid": False,
        "fix_found": True,
        "message": (
            f"Proposed corrected reactants {best_fix['corrected_smiles']} "
            f"({num_changed}/{len(reactant_smiles)} reactant(s) changed) "
            f"confirmed valid via SMARTS: {best_fix['smarts']}"
        ),
        "corrected_reactants": best_fix["corrected_smiles"],
        "matched_smarts": best_fix["smarts"],
    }


def _strip_tag(s: str, open_tag: str, close_tag: str) -> str:
    if s.startswith(open_tag):
        s = s[len(open_tag):]
    if s.endswith(close_tag):
        s = s[: -len(close_tag)]
    return s


def _parse_route_input(text: str) -> dict:
    """Parses the <SMILES>,<difficulty>,<JSON> route format (with CSV-doubled
    double-quotes) into a dict with "reactions" and "building_blocks"."""
    text = text.strip()
    parts = text.split(",", 2)
    if len(parts) < 3:
        raise ValueError("Input does not match <SMILES>,<difficulty>,<JSON> format")
    _, _, json_blob = parts
    json_blob = json_blob.strip()
    if json_blob.startswith('"') and json_blob.endswith('"'):
        json_blob = json_blob[1:-1]
    json_blob = json_blob.replace('""', '"')
    return json.loads(json_blob)


@agent.tool_plain
def correct_route(route_input: str) -> dict:
    """Parses a full <SMILES>,<difficulty>,<JSON> retrosynthesis route, validates every
    reaction step with auto_validate_reaction, and calls suggest_reactant_fix for any
    step that fails. This does all JSON parsing and per-step chaining deterministically
    in Python — no chemistry or parsing judgment calls are left to the caller.

    Args:
        route_input (str): The raw route input exactly as given by the user,
            i.e. "<SMILES>,<difficulty>,<JSON>" (JSON may have doubled double-quotes)

    Returns:
        dict: {
            "steps": [
                {
                    "reaction_number": int | None,
                    "status": "valid" | "fixed" | "unfixable" | "skipped_invalid_smiles",
                    "original_reactants": list[str],
                    "product": str,
                    "corrected_reactants": list[str] | None,
                    "matched_smarts": str | None,
                    "message": str,
                },
                ...
            ],
            "all_resolved": bool,
            "error": str | None,
        }
    """
    try:
        data = _parse_route_input(route_input)
    except Exception as e:
        return {
            "steps": [],
            "all_resolved": False,
            "error": f"Could not parse route input: {e}",
        }

    reactions = data.get("reactions", [])
    if not reactions:
        return {
            "steps": [],
            "all_resolved": False,
            "error": "No 'reactions' list found in parsed input",
        }

    steps = []
    all_resolved = True
    for reaction in reactions:
        num = reaction.get("reaction_number")
        product = reaction.get("product", "")
        raw_reactants = reaction.get("reactants", [])
        reactants = [
            _strip_tag(r, "<bb>", "</bb>") for r in raw_reactants if r and r.strip()
        ]

        if Chem.MolFromSmiles(product) is None or any(
            Chem.MolFromSmiles(r) is None for r in reactants
        ):
            all_resolved = False
            steps.append({
                "reaction_number": num,
                "status": "skipped_invalid_smiles",
                "original_reactants": reactants,
                "product": product,
                "corrected_reactants": None,
                "matched_smarts": None,
                "message": "Reactant or product SMILES could not be parsed",
            })
            continue

        is_valid, msg = auto_validate_reaction(reactants, product)
        if is_valid:
            steps.append({
                "reaction_number": num,
                "status": "valid",
                "original_reactants": reactants,
                "product": product,
                "corrected_reactants": None,
                "matched_smarts": None,
                "message": msg,
            })
            continue

        fix = suggest_reactant_fix(reactants, product)
        if fix["fix_found"]:
            steps.append({
                "reaction_number": num,
                "status": "fixed",
                "original_reactants": reactants,
                "product": product,
                "corrected_reactants": fix["corrected_reactants"],
                "matched_smarts": fix["matched_smarts"],
                "message": fix["message"],
            })
        else:
            all_resolved = False
            steps.append({
                "reaction_number": num,
                "status": "unfixable",
                "original_reactants": reactants,
                "product": product,
                "corrected_reactants": None,
                "matched_smarts": None,
                "message": fix["message"],
            })

    # Chain-consistency check: a "fixed" step's corrected reactants should still be
    # produced by some other step in the route (or be a listed building block) —
    # otherwise the fix solves that step in isolation but breaks the multi-step chain,
    # since whatever produced the original reactant no longer feeds into this step.
    known_smiles = set()
    for bb in data.get("building_blocks", []):
        bb_clean = _strip_tag(bb, "<bb>", "</bb>")
        mol = Chem.MolFromSmiles(bb_clean)
        if mol is not None:
            known_smiles.add(Chem.MolToSmiles(mol, canonical=True))
    for step in steps:
        mol = Chem.MolFromSmiles(step["product"])
        if mol is not None:
            known_smiles.add(Chem.MolToSmiles(mol, canonical=True))

    chain_consistent = True
    for step in steps:
        if step["status"] != "fixed":
            continue
        orphans = []
        for r in step["corrected_reactants"]:
            mol = Chem.MolFromSmiles(r)
            canon = Chem.MolToSmiles(mol, canonical=True) if mol else r
            if canon not in known_smiles:
                orphans.append(r)
        if orphans:
            chain_consistent = False
            step["chain_warning"] = (
                f"Corrected reactant(s) {orphans} are not produced by any other step "
                f"or listed building block in this route — fixing this step in isolation "
                f"breaks the multi-step chain. The step(s) that previously fed into this "
                f"one need to be corrected too (or removed if no longer needed)."
            )

    return {
        "steps": steps,
        "all_resolved": all_resolved,
        "chain_consistent": chain_consistent,
        "error": None,
    }
