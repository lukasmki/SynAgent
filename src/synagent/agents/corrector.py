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

You are given a reaction step that failed validation: a list of reactant SMILES and an
expected product SMILES. Your only job is to call suggest_reactant_fix with exactly those
inputs and report the tool result. Do not assess the chemistry yourself, do not invent a
fix, and do not second-guess the tool's output — it already re-validates any proposed fix
before returning it.

If the tool reports no fix found, say so plainly and report the message it returned.""".strip()

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
