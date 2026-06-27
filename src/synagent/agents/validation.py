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
    "[C:1][OH]>>[C:1]Br",                                     # alcohol → alkyl bromide (e.g. PBr3, HBr/H2SO4)
    "[c:1]F.[NH:2]>>[c:1][N:2]",                             # SNAr fluoride (secondary amine)
    "[c:1]Cl.[NH:2]>>[c:1][N:2]",                            # SNAr chloride (secondary amine)
    "[c:1]F.[NH2:2]>>[c:1][N:2]",                            # SNAr fluoride (primary amine)
    "[c:1]Cl.[NH2:2]>>[c:1][N:2]",                           # SNAr chloride (primary amine)
    "[c:1][Cl].[c:3][C:2]#N>>[c:1][C:2](=O)[c:3]",          # aryl ketone via C-CN activation (Cl)
    "[c:1][Br].[c:3][C:2]#N>>[c:1][C:2](=O)[c:3]",           # aryl ketone via C-CN activation (Br)
    "[C:1][Br].[c:3][C:2]#N>>[C:1][C:2](=O)[c:3]",           # aliphatic C-Br + aryl nitrile → ketone
    "[C:1]Cl.[c:3][C:2]#N>>[C:1][C:2](=O)[c:3]",             # aliphatic C-Cl + aryl nitrile → ketone
    "[C:1]I.[c:3][C:2]#N>>[C:1][C:2](=O)[c:3]",              # aliphatic C-I + aryl nitrile → ketone
    "[c:1]Br.[C:3][C:2]#N>>[c:1][C:2](=O)[C:3]",             # aryl Grignard (from Br) + alkyl nitrile → ketone
    "[c:1]Cl.[C:3][C:2]#N>>[c:1][C:2](=O)[C:3]",             # aryl Grignard (from Cl) + alkyl nitrile → ketone
    "[c:1]I.[C:3][C:2]#N>>[c:1][C:2](=O)[C:3]",              # aryl Grignard (from I) + alkyl nitrile → ketone
    "[C,c:1]Br.[C,c:3]Br>>[C,c:1][C,c:3]",                    # generic cross-coupling of two halides (Suzuki/Negishi/Kumada-type, abstracted)
    "[C,c:1]Br.[C,c:3]Br>>[$([CX4]),c:1][$([CX4]),c:3]",      # same, but bond-side restricted to sp3/aromatic — safe for blind discovery (excludes splitting at alkene/vinyl carbons)
    "[C:1](=O)[OH].[OH:2]>>[C:1](=O)[O:2]",                   # esterification (acid + alcohol)
    "[C:1](=O)[OH].[SH:2]>>[C:1](=O)[S:2]",                   # thioesterification
    "[CH1:1]>>[C:1]Br",                                        # alpha-bromination (CH)
    "[CH2:1]>>[C:1]Br",                                        # alpha-bromination (CH2)
    "[cH:1]>>[c:1]Br",                                          # electrophilic aromatic bromination
    "[cH:1]>>[c:1]Cl",                                          # electrophilic aromatic chlorination
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
    "[CH2:1][OH]>>[C:1](F)(F)Br",                              # primary alcohol -> bromodifluoromethyl (deoxy-halofluorination, e.g. DAST/NBS-type reagent combination)
    "[c:1]I>>[c:1]C#N",                                        # aryl cyanation (Pd/Cu-catalyzed, e.g. Zn(CN)2 or CuCN)
    "[c:1]Br>>[c:1]C#N",                                       # aryl cyanation (Pd/Cu-catalyzed, e.g. Zn(CN)2 or CuCN)
    "[c:1]Cl>>[c:1]C#N",                                       # aryl cyanation (Pd-catalyzed, e.g. Zn(CN)2)
    "[c:1][Br,I].[c:3]B(O)O>>[c:1][c:3]",                       # Suzuki-Miyaura coupling (aryl halide + arylboronic acid)
    "[c:1][Br,I].[CH:2]#[C:3]>>[c:1][C:2]#[C:3]",               # Sonogashira coupling (aryl halide + terminal alkyne)
    "[#6:4][C:2](=[O:3])Cl.[cH:1]>>[c:1][C:2](=[O:3])[#6:4]",     # Friedel-Crafts acylation (arene + acid chloride, R must be carbon to exclude amide/acid/ester carbonyls)
    "[C:1](=O)Cl.[OH:2]>>[C:1](=O)[O:2]",                       # esterification (acid chloride + alcohol)
    "[C:1](=O)[NH2:2]>>[C:1]#[N:2]",                            # primary amide -> nitrile (dehydration)
    "[NH:1].[S:2](=O)(=O)Cl>>[N:1][S:2](=O)(=O)",               # sulfonamide formation (amine + sulfonyl chloride)
    "[NH2:1].[N:2]=C=O>>[N:1]C(=O)[N:2]",                       # urea formation (amine + isocyanate)
    "[c:1]([OH:7])[c:2][C:3](=[O:8])[CH3:4].[#6:6][CH:5]=[O:9]>>[O:7]1[CH:5]([#6:6])[CH2:4][C:3](=[O:8])[c:2][c:1]1",  # flavanone/chromanone synthesis (ortho-hydroxyacetophenone + aldehyde, Claisen-Schmidt + oxa-Michael cyclization)
    "[CH2:1][C:2](=[O:3]).[#6:6][CH:5]=[O:9]>>[C:1](=[C:5][#6:6])[C:2](=[O:3])",  # Claisen-Schmidt condensation (ketone with alpha-CH2 + aldehyde -> benzylidene/chalcone-type enone)
    "[C:1]=[C:2]>>[C:1]1O[C:2]1",                               # epoxidation (alkene + peracid, e.g. mCPBA)
    "[C:1]1O[C:2]1.[NH2:3]>>[N:3][C:1][C:2][OH]",               # epoxide ring-opening by primary amine -> amino alcohol
    "[NH2:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",          # Boc protection (primary amine + Boc-Cl/Boc2O, abstracted)
    "[NH:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",           # Boc protection (secondary amine)
    "[CH2:1]([C:2]=O)[C:3]=O.[#6:4][CH:5]=[O:6]>>[C:1](=[C:5][#6:4])([C:2]=O)[C:3]=O",  # Knoevenagel condensation (active methylene + aldehyde)
    "[C:1](=O)[CH2:2][CH2:3][C:4](=O)>>[c:1]1[cH:2][cH:3][c:4]o1",  # Paal-Knorr furan synthesis (1,4-diketone cyclodehydration)
    "O=[CH:1]-[c:2].[C:3]-[CH:4]=P(-c1:c:c:c:c:c:1)(-c1:c:c:c:c:c:1)-c1:c:c:c:c:c:1>>[C:3]-[CH:4]=[CH:1]-[c:2]",  # Wittig reaction (aldehyde + triphenylphosphonium ylide -> alkene; extracted+verified, see extract_template_from_reaction)
    "Br-[c:1](:[c:2]):[c:3].[O;H0;D1;+0:4]=[CH:5]-[c:6]>>[OH:4]-[CH:5](-[c:6])-[c:1](:[c:2]):[c:3]",  # Grignard addition to aldehyde -> secondary alcohol (extracted+verified)
]


# Subset of _COMMON_SMARTS trusted for BLIND retrosynthetic discovery (no given building
# blocks to anchor against) — i.e. design_route's unconstrained fallback only. Templates
# requiring a specific, distinguishing functional-group combination (amide bond, ester,
# nitrile, a primary CH2OH, a boronic acid, a terminal alkyne, an isocyanate, a sulfonyl
# chloride, an aromatic halide + amine, an epoxide, a Boc carbamate, a doubly-activated
# methylene, a furan ring) are unlikely to misattribute an unrelated bond elsewhere in
# the molecule. Epoxide ring-OPENING is deliberately excluded even though epoxidation
# (forming one) is included — the resulting 1,2-amino-alcohol pattern is too generic to
# trust without an anchor, unlike the rare epoxide ring itself. Excluded: anything
# generic enough to "explain" almost any
# bond — bare alcohol<->halide swaps, alpha-halogenation, BOTH two-halide coupling
# variants (even the bond-side-restricted one: tried it, it still let multiple Br
# couplings compound on the same atom across steps, producing a geminal dibromide as a
# "building block" — not safe for multi-step blind chaining), SN2/SNAr O-alkylation,
# carbonyl reduction, reductive amination, ketone-via-nitrile-activation,
# Friedel-Crafts alkylation — these remain safe when anchored
# to known building blocks (the constrained search) but too permissive to trust without
# that anchor.
_DISCOVERY_SAFE_SMARTS = [
    "[NH1;R:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]",
    "[NH2:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]",
    "[NH:1].[C:2](=[O:3])Cl>>[N:1][C:2]=[O:3]",
    "[NH:1].[C:2](=[O:4])[O:3]>>[N:1][C:2]=[O:4]",
    "[c:1]F.[NH:2]>>[c:1][N:2]",
    "[c:1]F.[NH2:2]>>[c:1][N:2]",
    "[C:1](=O)[OH].[OH:2]>>[C:1](=O)[O:2]",
    "[C:1](=O)[OH].[SH:2]>>[C:1](=O)[S:2]",
    "[cH:1]>>[c:1]Br",
    "[cH:1]>>[c:1]Cl",
    "[C:1](=O)[OH].[NH2:2]>>[C:1](=O)[N:2]",
    "[C:1](=O)[OH].[NH1:2]>>[C:1](=O)[N:2]",
    "[c:1]Br.[NH2:2]>>[c:1][N:2]",
    "[c:1]Br.[NH1:2]>>[c:1][N:2]",
    "[c:1]I.[NH2:2]>>[c:1][N:2]",
    "[c:1]I.[NH1:2]>>[c:1][N:2]",
    "[CH2:1][OH]>>[C:1]#N",
    "[CH2:1][OH]>>[C:1](F)(F)Br",
    "[c:1][Br,I].[c:3]B(O)O>>[c:1][c:3]",
    "[c:1][Br,I].[CH:2]#[C:3]>>[c:1][C:2]#[C:3]",
    "[#6:4][C:2](=[O:3])Cl.[cH:1]>>[c:1][C:2](=[O:3])[#6:4]",
    "[C:1](=O)Cl.[OH:2]>>[C:1](=O)[O:2]",
    "[C:1](=O)[NH2:2]>>[C:1]#[N:2]",
    "[NH:1].[S:2](=O)(=O)Cl>>[N:1][S:2](=O)(=O)",
    "[NH2:1].[N:2]=C=O>>[N:1]C(=O)[N:2]",
    "[c:1]([OH:7])[c:2][C:3](=[O:8])[CH3:4].[#6:6][CH:5]=[O:9]>>[O:7]1[CH:5]([#6:6])[CH2:4][C:3](=[O:8])[c:2][c:1]1",
    "[CH2:1][C:2](=[O:3]).[#6:6][CH:5]=[O:9]>>[C:1](=[C:5][#6:6])[C:2](=[O:3])",
    "[C:1]=[C:2]>>[C:1]1O[C:2]1",
    "[NH2:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",
    "[NH:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",
    "[CH2:1]([C:2]=O)[C:3]=O.[#6:4][CH:5]=[O:6]>>[C:1](=[C:5][#6:4])([C:2]=O)[C:3]=O",
    "[C:1](=O)[CH2:2][CH2:3][C:4](=O)>>[c:1]1[cH:2][cH:3][c:4]o1",
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
