from collections import Counter
from itertools import permutations

from pydantic_ai import Agent, RunContext
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi

from synagent.models import SynLlamaReport
from pydantic_ai.models.google import GoogleModel
import logfire

logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()


# Setup the agent
SYSTEM_PROMPT = """You are SynAgent, a rigorous retrosynthesis verification agent.

## Input format
Your input always looks like this:
  <SMILES>,<difficulty>,<JSON>

Example:
  CCO,medium,{"reactions":[{"reaction_number":1,"reactants":["CC=O","[H][H]"],"product":"CCO"}],"building_blocks":["CC=O"]}

The JSON may use doubled double-quotes (e.g. ""reactions"") due to CSV escaping — treat "" as a single ".
Always parse the JSON yourself to extract the reactions list. Never ask the user to re-provide data.

## Validation steps
For each reaction in the reactions list:
1. Read "reactants" (list of SMILES) and "product" (SMILES) directly from the JSON.
2. Skip any reaction where a reactant is an empty string "".
3. Call is_valid_smiles on all SMILES to confirm they are valid.
4. Call auto_validate_reaction with just the reactant SMILES list and expected product. This tool tries all common reaction SMARTS automatically — you do NOT need to provide or construct a SMARTS string.

## Output
Report whether each reaction step is valid and give a final verdict on the full route.""".strip()

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type= str,
)

async def main():
    result = await agent.run()
    print(result.output)



# Define a tool
@agent.tool_plain
def is_valid_smiles(smiles: list[str]) -> dict[str, bool]:
    """Checks validity of SMILES strings

    Args:
        smiles (list[str]): list of SMILES to check

    Returns:
        dict[str, bool]: {smiles: is_valid}
    """
    return {s: Chem.MolFromSmiles(s) is not None for s in smiles}


_COMMON_SMARTS = [
    "[NH1;R:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]",       # secondary ring amine + acid
    "[NH2:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]",          # primary amine + acid
    "[NH:1].[C:2](=[O:3])Cl>>[N:1][C:2]=[O:3]",             # amine + acid chloride
    "[NH:1].[C:2](=[O:4])[O:3]>>[N:1][C:2]=[O:4]",           # amine + ester
    "[NH:1].[CH2:2]Br>>[N:1][CH2:2]",                        # N-alkylation (bromide)
    "[NH:1].[CH2:2]Cl>>[N:1][CH2:2]",                        # N-alkylation (chloride)
    "[c:1][OH]>>[c:1]Cl",                                     # phenol → aryl chloride
    "[C:1][OH]>>[C:1]Cl",                                     # alcohol → alkyl chloride
    "[c:1]F.[NH:2]>>[c:1][N:2]",                             # SNAr fluoride
    "[c:1]Cl.[NH:2]>>[c:1][N:2]",                            # SNAr chloride
    "[c:1][Cl].[c:3][C:2]#N>>[c:1][C:2](=O)[c:3]",          # aryl ketone via C-CN activation (Cl)
    "[c:1][Br].[c:3][C:2]#N>>[c:1][C:2](=O)[c:3]",           # aryl ketone via C-CN activation (Br)
    "[C:1][Br].[c:3][C:2]#N>>[C:1][C:2](=O)[c:3]",           # aliphatic C-Br + nitrile → ketone
    "[C:1](=O)[OH].[OH:2]>>[C:1](=O)[O:2]",                   # esterification (acid + alcohol)
    "[C:1](=O)[OH].[SH:2]>>[C:1](=O)[S:2]",                   # thioesterification
    "[CH1:1]>>[C:1]Br",                                        # alpha-bromination (CH)
    "[CH2:1]>>[C:1]Br",                                        # alpha-bromination (CH2)
    "[c:1][H]>>[c:1]Br",                                       # electrophilic aromatic bromination
    "[c:1][H]>>[c:1]Cl",                                       # electrophilic aromatic chlorination
    "[C:1]=O.[NH2:2]>>[C:1][N:2]",                            # reductive amination (primary amine)
    "[C:1]=O.[NH:2]>>[C:1][N:2]",                             # reductive amination (secondary amine)
    "[C:1](=O)[OH].[NH2:2]>>[C:1](=O)[N:2]",                  # amide from acid + primary amine
    "[C:1](=O)[OH].[NH1:2]>>[C:1](=O)[N:2]",                  # amide from acid + secondary amine
    "[c:1]Br.[NH2:2]>>[c:1][N:2]",                            # Buchwald-Hartwig primary amine (Br)
    "[c:1]Br.[NH1:2]>>[c:1][N:2]",                            # Buchwald-Hartwig secondary amine (Br)
    "[c:1]I.[NH2:2]>>[c:1][N:2]",                             # Buchwald-Hartwig primary amine (I)
    "[c:1]I.[NH1:2]>>[c:1][N:2]",                             # Buchwald-Hartwig secondary amine (I)
    "[c:1]Cl.[OH:2]>>[c:1][O:2]",                             # SNAr oxygen nucleophile
    "[C:1]Br.[OH:2]>>[C:1][O:2]",                             # SN2 O-alkylation
    "[C:1]Br.[NH:2]>>[C:1][N:2]",                             # SN2 N-alkylation (Br)
    "[C:1]Cl.[NH:2]>>[C:1][N:2]",                             # SN2 N-alkylation (Cl)
    "[C:1]=O.[OH:2]>>[C:1][O:2]",                             # carbonyl reduction / hemiacetal
    "[CH2:1][OH]>>[C:1]#N",                                    # primary alcohol -> nitrile (oxidation/oxime/dehydration)
]


def _atom_formula(mol: Chem.Mol) -> Counter:
    """Return heavy-atom + implicit-H counts as a Counter."""
    c: Counter = Counter()
    for atom in mol.GetAtoms():
        c[atom.GetAtomicNum()] += 1
        c[0] += atom.GetTotalNumHs()  # key 0 = hydrogen
    return c


# Common leaving-group formulas (atomic_num: count)
_LEAVING_GROUPS: list[Counter] = [
    Counter({17: 1, 0: 1}),   # HCl
    Counter({9:  1, 0: 1}),   # HF
    Counter({35: 1, 0: 1}),   # HBr
    Counter({8:  1, 0: 2}),   # H2O
    Counter({8:  2, 0: 2}),   # 2×OH (e.g. oxalic acid loss)
]


def _check_intramolecular(reactant_mol: Chem.Mol, product_mol: Chem.Mol) -> bool:
    """Return True if product could plausibly arise from reactant via intramolecular
    ring-forming cyclization: product gains ≥1 ring and the formula difference
    matches a common leaving group."""
    r_rings = rdMolDescriptors.CalcNumRings(reactant_mol)
    p_rings = rdMolDescriptors.CalcNumRings(product_mol)
    if p_rings <= r_rings:
        return False
    r_counts = _atom_formula(reactant_mol)
    p_counts = _atom_formula(product_mol)
    for lg in _LEAVING_GROUPS:
        if r_counts == p_counts + lg:
            return True
    return False


@agent.tool_plain
def auto_validate_reaction(
    reactant_smiles: list[str], expected_product: str
) -> tuple[bool, str]:
    """Validates a reaction by automatically trying common reaction SMARTS patterns.
    Use this instead of is_valid_reaction — you do not need to provide a SMARTS string.

    Args:
        reactant_smiles (list[str]): Reactant SMILES (skip empty strings)
        expected_product (str): Expected product SMILES

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    reactant_smiles = [s for s in reactant_smiles if s.strip()]
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactants):
        return False, "`reactant_smiles` contains invalid SMILES strings"

    expected_mol = Chem.MolFromSmiles(expected_product)
    if expected_mol is None:
        return False, "`expected_product` is not a valid SMILES string"

    canon_product = Chem.MolToSmiles(expected_mol, canonical=True)
    expected_inchi = MolToInchi(expected_mol)

    for smarts in _COMMON_SMARTS:
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if rxn is None:
                continue
        except Exception:
            continue
        for perm in permutations(reactants):
            try:
                for products in rxn.RunReactants(tuple(perm)):
                    for mol in products:
                        try:
                            smi = Chem.MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True)
                            if smi == canon_product:
                                return True, f"Reaction is valid (matched with SMARTS: {smarts})"
                            if MolToInchi(mol) == expected_inchi:
                                return True, f"Reaction is valid (matched with SMARTS: {smarts})"
                        except Exception:
                            continue
            except Exception:
                continue

    # Intramolecular cyclization check: single reactant that ring-closes onto itself
    for mol in reactants:
        if _check_intramolecular(mol, expected_mol):
            return True, "Reaction is valid (intramolecular cyclization: ring count increases and formula matches leaving-group loss)"

    return False, "No common reaction pattern produced the expected product"


@agent.tool_plain
def is_valid_reaction(
    reaction_smarts: str, reactant_smiles: list[str], expected_product: str
) -> tuple[bool, str]:
    """Runs the reaction on the given reactants and
    checks if the expected product is formed

    Args:
        reaction_smarts (str): Reaction SMARTS
        reactant_smiles (list[str]): Reactant SMILES
        expected_product (str): Product SMILES

    Returns:
        tuple[bool, str]: (is_valid, message)
    """
    # parse the reaction SMARTS
    try:
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
        if rxn is None:
            return False, "`reaction_smarts` could not be parsed"
    except Exception:
        return False, "`reaction_smarts` could not be parsed"

    # Validate reactants
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactants):
        return False, "`reactant_smiles` contains invalid SMILES strings"

    # Run the reaction trying all reactant orderings (RDKit matches templates positionally)
    all_products = []
    for perm in permutations(reactants):
        all_products.extend([
            Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
            for p in rxn.RunReactants(tuple(perm))
            for m in p
        ])
    products = list(set(all_products))
    if not products:
        return False, "Reaction produced no products"

    # Check for expected product using canonical SMILES and InChI fallback
    canon_product = Chem.CanonSmiles(expected_product)
    expected_inchi = MolToInchi(Chem.MolFromSmiles(expected_product))
    for product in products:
        if canon_product == product:
            return True, f"Reaction produced expected product {expected_product}"
        prod_mol = Chem.MolFromSmiles(product)
        if prod_mol and MolToInchi(prod_mol) == expected_inchi:
            return True, f"Reaction produced expected product {expected_product}"
    return (
        False,
        f"Reaction did not produce expected product, instead got {products}",
    )
