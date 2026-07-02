"""Reaction SMARTS library and auto-validation logic shared across capabilities."""

from collections import Counter
from itertools import permutations

from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi


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
    "[NH2:1].[N:2]=C=O>>[N:1]C(=O)[N:2]",                       # urea formation (primary amine + isocyanate)
    "[NH1:1].[N:2]=C=O>>[N:1]C(=O)[N:2]",                       # urea formation (secondary amine + isocyanate)
    "[c:1]([OH:7])[c:2][C:3](=[O:8])[CH3:4].[#6:6][CH:5]=[O:9]>>[O:7]1[CH:5]([#6:6])[CH2:4][C:3](=[O:8])[c:2][c:1]1",  # flavanone/chromanone synthesis
    "[C:1](=[O:3])[CH2:2].[CH1:4]=[O:5]>>[C:1](=[O:3])[CH1:2][C:4]([OH:5])",       # aldol condensation (ketone α-CH2 + aldehyde → β-hydroxy ketone)
    "[CH2:1][C:2](=[O:3]).[#6:6][CH:5]=[O:9]>>[C:1](=[C:5][#6:6])[C:2](=[O:3])",  # Claisen-Schmidt condensation
    "[C:1]=[C:2]>>[C:1]1O[C:2]1",                               # epoxidation (alkene + peracid, e.g. mCPBA)
    "[C:1]1O[C:2]1.[NH2:3]>>[N:3][C:1][C:2][OH]",               # epoxide ring-opening by primary amine -> amino alcohol
    "[NH2:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",          # Boc protection (primary amine)
    "[NH:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",           # Boc protection (secondary amine)
    "[CH2:1]([C:2]=O)[C:3]=O.[#6:4][CH:5]=[O:6]>>[C:1](=[C:5][#6:4])([C:2]=O)[C:3]=O",  # Knoevenagel condensation
    "[C:1](=O)[CH2:2][CH2:3][C:4](=O)>>[c:1]1[cH:2][cH:3][c:4]o1",  # Paal-Knorr furan synthesis
    "O=[CH:1]-[c:2].[C:3]-[CH:4]=P(-c1:c:c:c:c:c:1)(-c1:c:c:c:c:c:1)-c1:c:c:c:c:c:1>>[C:3]-[CH:4]=[CH:1]-[c:2]",  # Wittig reaction
    "Br-[c:1](:[c:2]):[c:3].[O;H0;D1;+0:4]=[CH:5]-[c:6]>>[OH:4]-[CH:5](-[c:6])-[c:1](:[c:2]):[c:3]",  # Grignard addition to aldehyde -> secondary alcohol
    "[CH2;D2;+0:1][C:2]=[O:3].[CH3][CH2]O[C:6](=[O:7])O[CH2][CH3]>>[CH;D3;+0:1]([C:2]=[O:3])[C:6](=[O:7])OCC",  # alpha-carbethoxylation (cyclic ketone + diethyl carbonate)
    "CCOC(=O)-[CH;+0:1](-[C:2](=[O:3])-[#6:9])-[#6:4].[#7;a:7]:[c:8](-[NH2;D1;+0:12]):[nH;D2;+0:10]:[#7;a:11]>>[#7;a:7]:[c:8]1:[nH;D2;+0:12]:[c;H0;D3;+0](:[c;H0;D3;+0:2](:[c;H0;D3;+0:1](=[O;H0;D1;+0]):[n;H0;D3;+0:10]:1:[#7;a:11])-[#6:9])-[#6:4]",  # triazolopyrimidinone cyclocondensation
    "O=[C;H0;D3;+0:1](-[C:2]=[CH;D2;+0:3]-[c:4]:[cH;D2;+0:5]:[c:6])-[c:7]:[c;H0;D3;+0:8](:[c:9])-[CH2;D2;+0:10]-[C:11].[#7;a:12]:[c:13](-[NH2;D1;+0:14]):[nH;D2;+0:15]:[#7;a:16]>>[#7;a:12]:[c:13]1:[n;H0;D3;+0:15](:[#7;a:16])-[CH;D3;+0:1](-[c:7]:[cH;D2;+0:8]:[c:9])-[C:2]=[C;H0;D3;+0:3](-[NH;D2;+0:14]-1)-[c:4]:[c;H0;D3;+0:5](:[c:6])-[CH2;D2;+0:10]-[C:11]",  # triazolopyrimidine cyclization
]


# Subset trusted for blind retrosynthetic discovery (no given building blocks to anchor
# against). Excludes templates generic enough to misattribute unrelated bonds — safe only
# when anchored to known building blocks. See corrector._toolset._search_route_discovery.
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
    "[NH1:1].[N:2]=C=O>>[N:1]C(=O)[N:2]",
    "[c:1]([OH:7])[c:2][C:3](=[O:8])[CH3:4].[#6:6][CH:5]=[O:9]>>[O:7]1[CH:5]([#6:6])[CH2:4][C:3](=[O:8])[c:2][c:1]1",
    "[C:1](=[O:3])[CH2:2].[CH1:4]=[O:5]>>[C:1](=[O:3])[CH1:2][C:4]([OH:5])",       # aldol condensation
    "[CH2:1][C:2](=[O:3]).[#6:6][CH:5]=[O:9]>>[C:1](=[C:5][#6:6])[C:2](=[O:3])",
    "[C:1]=[C:2]>>[C:1]1O[C:2]1",
    "[NH2:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",
    "[NH:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C",
    "[CH2:1]([C:2]=O)[C:3]=O.[#6:4][CH:5]=[O:6]>>[C:1](=[C:5][#6:4])([C:2]=O)[C:3]=O",
    "[C:1](=O)[CH2:2][CH2:3][C:4](=O)>>[c:1]1[cH:2][cH:3][c:4]o1",
    "[CH2;D2;+0:1][C:2]=[O:3].[CH3][CH2]O[C:6](=[O:7])O[CH2][CH3]>>[CH;D3;+0:1]([C:2]=[O:3])[C:6](=[O:7])OCC",
    "CCOC(=O)-[CH;+0:1](-[C:2](=[O:3])-[#6:9])-[#6:4].[#7;a:7]:[c:8](-[NH2;D1;+0:12]):[nH;D2;+0:10]:[#7;a:11]>>[#7;a:7]:[c:8]1:[nH;D2;+0:12]:[c;H0;D3;+0](:[c;H0;D3;+0:2](:[c;H0;D3;+0:1](=[O;H0;D1;+0]):[n;H0;D3;+0:10]:1:[#7;a:11])-[#6:9])-[#6:4]",
    "O=[C;H0;D3;+0:1](-[C:2]=[CH;D2;+0:3]-[c:4]:[cH;D2;+0:5]:[c:6])-[c:7]:[c;H0;D3;+0:8](:[c:9])-[CH2;D2;+0:10]-[C:11].[#7;a:12]:[c:13](-[NH2;D1;+0:14]):[nH;D2;+0:15]:[#7;a:16]>>[#7;a:12]:[c:13]1:[n;H0;D3;+0:15](:[#7;a:16])-[CH;D3;+0:1](-[c:7]:[cH;D2;+0:8]:[c:9])-[C:2]=[C;H0;D3;+0:3](-[NH;D2;+0:14]-1)-[c:4]:[c;H0;D3;+0:5](:[c:6])-[CH2;D2;+0:10]-[C:11]",
]


def _atom_formula(mol: Chem.Mol) -> Counter:
    """Return heavy-atom + implicit-H counts as a Counter."""
    c: Counter = Counter()
    for atom in mol.GetAtoms():
        c[atom.GetAtomicNum()] += 1
        c[0] += atom.GetTotalNumHs()
    return c


_LEAVING_GROUPS: list[Counter] = [
    Counter({17: 1, 0: 1}),   # HCl
    Counter({9:  1, 0: 1}),   # HF
    Counter({35: 1, 0: 1}),   # HBr
    Counter({8:  1, 0: 2}),   # H2O
    Counter({8:  2, 0: 2}),   # 2×OH
]


def _check_intramolecular(reactant_mol: Chem.Mol, product_mol: Chem.Mol) -> bool:
    """True if product could plausibly arise from reactant via intramolecular
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


def auto_validate_reaction(
    reactant_smiles: list[str], expected_product: str
) -> tuple[bool, str]:
    """Validates a reaction by automatically trying all common reaction SMARTS patterns.

    Args:
        reactant_smiles (list[str]): Reactant SMILES (empty strings are skipped)
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

    for mol in reactants:
        if _check_intramolecular(mol, expected_mol):
            return True, "Reaction is valid (intramolecular cyclization: ring count increases and formula matches leaving-group loss)"

    return False, "No common reaction pattern produced the expected product"
