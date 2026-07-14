import json
import math
import os
import re
import sys
from itertools import permutations

import httpx
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdChemReactions

from synagent.validation._smarts import _COMMON_SMARTS, auto_validate_reaction

try:
    from rdchiral.main import rdchiralRunText
    from rdchiral.template_extractor import extract_from_reaction as _rdchiral_extract
except ImportError:
    rdchiralRunText = None
    _rdchiral_extract = None

try:
    from indigo import Indigo
except ImportError:
    Indigo = None

try:
    import rdkit as _rdkit_pkg
    _sa_score_dir = os.path.join(os.path.dirname(_rdkit_pkg.__file__), "Contrib", "SA_Score")
    if _sa_score_dir not in sys.path:
        sys.path.append(_sa_score_dir)
    import sascorer as _sascorer
except ImportError:
    _sascorer = None

RDLogger.DisableLog("rdApp.*")

_DIOXOLANE_PATT = Chem.MolFromSmarts("[C;X4:1]1[O:2]CC[O:3]1")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _reverse_smarts(smarts: str) -> str:
    lhs, rhs = smarts.split(">>")
    return f"{rhs}>>{lhs}"


def _fingerprint(mol: Chem.Mol):
    # useChirality=True: without it, enantiomers score as 1.0 (identical)
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)


def _similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(_fingerprint(mol_a), _fingerprint(mol_b))


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _sa_score(smiles: str) -> float | None:
    """Ertl/Schuffenhauer SA_Score (~1 easy to ~10 hard). Returns None if unavailable."""
    if _sascorer is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(_sascorer.calculateScore(mol))
    except Exception:
        return None


def _strip_unbalanced_map_numbers(smarts: str) -> str:
    """Remove atom map numbers that appear on only one side of >> (required by rdchiral)."""
    lhs, rhs = smarts.split(">>")
    lhs_maps = set(re.findall(r":(\d+)\]", lhs))
    rhs_maps = set(re.findall(r":(\d+)\]", rhs))
    unbalanced = (lhs_maps - rhs_maps) | (rhs_maps - lhs_maps)

    def strip(side: str) -> str:
        for n in unbalanced:
            side = re.sub(rf":{n}\]", "]", side)
        return side

    return f"{strip(lhs)}>>{strip(rhs)}"


def _rdchiral_decompose(reverse_smarts: str, product_smiles: str) -> list[list[str]]:
    """Fallback decomposition via rdchiral for templates that crash bare RDKit."""
    if rdchiralRunText is None:
        return []
    try:
        balanced = _strip_unbalanced_map_numbers(reverse_smarts)
        outcomes = rdchiralRunText(balanced, product_smiles)
    except Exception:
        return []
    return [outcome.split(".") for outcome in outcomes]


def _decompose_dioxolane(product_smiles: str) -> list[dict]:
    """Programmatically retro-open 1,3-dioxolane (ketal/acetal) rings.

    Template 167's reversed SMARTS fails for polycyclic ketal carbons because
    RDKit ring-closure matching breaks when the ketal C participates in 3+ rings.
    This function bypasses SMARTS entirely: it finds the dioxolane substructure,
    removes both C-O bonds to the ketal carbon with RWMol, appends a new C=O, and
    validates the resulting open-chain intermediate forward via auto_validate_reaction.
    """
    if _DIOXOLANE_PATT is None:
        return []
    mol = Chem.MolFromSmiles(product_smiles)
    if mol is None:
        return []
    matches = mol.GetSubstructMatches(_DIOXOLANE_PATT)
    if not matches:
        return []

    candidates: list[dict] = []
    seen: set[str] = set()

    for match in matches:
        # match indices for [C;X4:1]1[O:2]CC[O:3]1:
        #   0=ketal_C, 1=O(first), 2=CH2, 3=CH2, 4=O(second)
        ketal_c_idx, o1_idx, o2_idx = match[0], match[1], match[4]
        try:
            rw = Chem.RWMol(mol)
            rw.RemoveBond(ketal_c_idx, o1_idx)
            rw.RemoveBond(ketal_c_idx, o2_idx)

            for o_idx in (o1_idx, o2_idx):
                o_atom = rw.GetAtomWithIdx(o_idx)
                o_atom.SetNoImplicit(False)
                o_atom.SetNumExplicitHs(0)

            ketal_c_atom = rw.GetAtomWithIdx(ketal_c_idx)
            ketal_c_atom.SetNumExplicitHs(0)
            ketal_c_atom.SetNoImplicit(False)

            new_o_idx = rw.AddAtom(Chem.Atom(8))
            rw.AddBond(ketal_c_idx, new_o_idx, Chem.BondType.DOUBLE)

            Chem.SanitizeMol(rw)
            open_smi = Chem.MolToSmiles(rw, canonical=True, ignoreAtomMapNumbers=True)
        except Exception:
            continue

        if not open_smi:
            continue
        open_canon = _canon(open_smi)
        if open_canon is None or open_canon in seen or open_canon == _canon(product_smiles):
            continue
        seen.add(open_canon)

        ok, _ = auto_validate_reaction([open_canon], product_smiles)
        if not ok:
            continue

        candidates.append({
            "smarts": "[C;X4:1]1[O:2]CC[O:3]1>>[C:1]=O.[OH:2]CC[OH:3]",
            "precursors": [open_canon],
        })

    return candidates


def _decompose_one_step(product_smiles: str, smarts_list: list[str] = _COMMON_SMARTS) -> list[dict]:
    """Try every reverse SMARTS template on product_smiles; return only candidates
    whose forward reaction is confirmed valid by auto_validate_reaction."""
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return []

    candidates = []
    for smarts in smarts_list:
        reverse_smarts = _reverse_smarts(smarts)
        raw_precursor_lists: list[list[str]] = []
        engine_crashed = False

        try:
            rxn = rdChemReactions.ReactionFromSmarts(reverse_smarts)
            if rxn is None:
                continue
        except Exception:
            continue
        try:
            outputs_list = rxn.RunReactants((product_mol,))
            for outputs in outputs_list:
                try:
                    raw_precursor_lists.append(
                        [Chem.MolToSmiles(m, ignoreAtomMapNumbers=True) for m in outputs]
                    )
                except Exception:
                    continue
        except Exception:
            engine_crashed = True

        if engine_crashed:
            raw_precursor_lists.extend(_rdchiral_decompose(reverse_smarts, product_smiles))

        for raw_precursors in raw_precursor_lists:
            precursor_smiles = [_canon(s) for s in raw_precursors]
            if any(s is None for s in precursor_smiles):
                continue

            precursor_mols = [Chem.MolFromSmiles(s) for s in precursor_smiles]
            if any(
                pm is None or sum(a.GetNumRadicalElectrons() for a in pm.GetAtoms()) != 0
                for pm in precursor_mols
            ):
                continue

            ok, _ = auto_validate_reaction(precursor_smiles, product_smiles)
            if not ok:
                continue

            key = tuple(sorted(precursor_smiles))
            if any(tuple(sorted(c["precursors"])) == key for c in candidates):
                continue
            candidates.append({"smarts": smarts, "precursors": precursor_smiles})

    for cand in _decompose_dioxolane(product_smiles):
        key = tuple(sorted(cand["precursors"]))
        if not any(tuple(sorted(c["precursors"])) == key for c in candidates):
            candidates.append(cand)

    return candidates


_MAX_SEARCH_NODES = 2000


def _closest_known_block(
    target_c: str, known_mols: dict[str, Chem.Mol], min_similarity: float
) -> tuple[str | None, float]:
    target_mol = Chem.MolFromSmiles(target_c)
    if target_mol is None or not known_mols:
        return None, 0.0
    best_smi, best_sim = None, 0.0
    for smi, mol in known_mols.items():
        sim = _similarity(target_mol, mol)
        if sim > best_sim:
            best_smi, best_sim = smi, sim
    if best_sim >= min_similarity:
        return best_smi, best_sim
    return None, best_sim


# ---------------------------------------------------------------------------
# MCTS search
# ---------------------------------------------------------------------------

class _MCTSNode:
    __slots__ = ("smiles", "depth_remaining", "candidates", "is_known", "is_terminal", "visits", "total_value")

    def __init__(self, smiles: str, depth_remaining: int):
        self.smiles = smiles
        self.depth_remaining = depth_remaining
        self.candidates: list[dict] | None = None
        self.is_known = False
        self.is_terminal = False
        self.visits = 0
        self.total_value = 0.0


def _mcts_heuristic_value(
    smiles: str, known_set: set[str], known_mols: dict[str, Chem.Mol]
) -> float:
    if smiles in known_set:
        return 10.0
    sa = _sa_score(smiles)
    sa_term = max(0.0, (10.0 - sa) / 10.0 * 5.0) if sa is not None else 2.5
    _, sim = _closest_known_block(smiles, known_mols, min_similarity=0.0)
    return sa_term + sim * 5.0


def _mcts_expand(
    node: _MCTSNode,
    known_set: set[str],
    budget: list[int],
    node_cache: dict[tuple[str, int], "_MCTSNode"],
    smarts_list: list[str],
) -> None:
    if node.candidates is not None:
        return
    if node.smiles in known_set:
        node.is_known = True
        node.candidates = []
        return
    if node.depth_remaining <= 0 or budget[0] <= 0:
        node.is_terminal = True
        node.candidates = []
        return
    budget[0] -= 1

    node.candidates = []
    for cand in _decompose_one_step(node.smiles, smarts_list=smarts_list):
        precursor_nodes = []
        for p in cand["precursors"]:
            key = (p, node.depth_remaining - 1)
            if key not in node_cache:
                node_cache[key] = _MCTSNode(p, node.depth_remaining - 1)
            precursor_nodes.append(node_cache[key])
        node.candidates.append({
            "smarts": cand["smarts"],
            "precursors": cand["precursors"],
            "nodes": precursor_nodes,
            "visits": 0,
            "value_sum": 0.0,
        })
    if not node.candidates:
        node.is_terminal = True


def _mcts_select_candidate(node: _MCTSNode, exploration_c: float = 1.4) -> dict | None:
    if not node.candidates:
        return None
    best, best_score = None, -float("inf")
    for cand in node.candidates:
        if cand["visits"] == 0:
            score = float("inf")
        else:
            exploit = cand["value_sum"] / cand["visits"]
            explore = exploration_c * math.sqrt(math.log(max(node.visits, 1)) / cand["visits"])
            score = exploit + explore
        if score > best_score:
            best_score, best = score, cand
    return best


def _mcts_simulate(
    node: _MCTSNode,
    known_set: set[str],
    known_mols: dict[str, Chem.Mol],
    budget: list[int],
    node_cache: dict[tuple[str, int], "_MCTSNode"],
    smarts_list: list[str],
) -> float:
    _mcts_expand(node, known_set, budget, node_cache, smarts_list)

    if node.is_known:
        value = 10.0
    elif not node.candidates:
        value = _mcts_heuristic_value(node.smiles, known_set, known_mols)
    else:
        cand = _mcts_select_candidate(node)
        bottleneck = min(
            cand["nodes"],
            key=lambda n: _mcts_heuristic_value(n.smiles, known_set, known_mols),
        )
        value = _mcts_simulate(bottleneck, known_set, known_mols, budget, node_cache, smarts_list)
        cand["visits"] += 1
        cand["value_sum"] += value

    node.visits += 1
    node.total_value += value
    return value


def _mcts_extract_route(
    node: _MCTSNode,
    known_set: set[str],
    known_mols: dict[str, Chem.Mol],
    visited: frozenset[str],
    min_block_similarity: float,
    swap_registry: dict[str, dict],
    allow_swap: bool = True,
) -> tuple[list[dict], int] | None:
    if node.smiles in known_set:
        return [], 0
    if node.smiles in visited:
        return None

    if node.candidates:
        next_visited = visited | {node.smiles}
        ranked = sorted(
            node.candidates,
            key=lambda c: -(c["value_sum"] / c["visits"] if c["visits"] > 0 else -1),
        )
        for cand in ranked:
            sub_steps: list[dict] = []
            sub_swaps = 0
            feasible = True
            for child in cand["nodes"]:
                sub = _mcts_extract_route(
                    child, known_set, known_mols, next_visited, min_block_similarity, swap_registry,
                )
                if sub is None:
                    feasible = False
                    break
                sub_steps.extend(sub[0])
                sub_swaps += sub[1]
            if feasible:
                sub_steps.append({
                    "reactants": cand["precursors"],
                    "product": node.smiles,
                    "matched_smarts": cand["smarts"],
                })
                return sub_steps, sub_swaps

    if not allow_swap:
        return None

    block, sim = _closest_known_block(node.smiles, known_mols, min_block_similarity)
    if block is not None:
        swap_registry[node.smiles] = {
            "needed_smiles": node.smiles,
            "suggested_building_block": block,
            "similarity": round(sim, 3),
        }
        return [], 1
    return None


def _mcts_search_route(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int,
    iterations: int,
    min_block_similarity: float,
    smarts_list: list[str],
) -> tuple[list[dict] | None, list[dict]]:
    target_c = _canon(target_smiles)
    if target_c is None:
        return None, []
    known_set = {c for c in (_canon(s) for s in known_smiles) if c}
    known_mols = {smi: Chem.MolFromSmiles(smi) for smi in known_set}

    node_cache: dict[tuple[str, int], _MCTSNode] = {}
    root_key = (target_c, max_depth)
    root = _MCTSNode(target_c, max_depth)
    node_cache[root_key] = root
    budget = [_MAX_SEARCH_NODES]

    for _ in range(iterations):
        if budget[0] <= 0:
            break
        _mcts_simulate(root, known_set, known_mols, budget, node_cache, smarts_list)

    swap_registry: dict[str, dict] = {}
    result = _mcts_extract_route(
        root, known_set, known_mols, frozenset(), min_block_similarity, swap_registry, allow_swap=False,
    )
    if result is None:
        return None, []
    steps, _ = result

    produced = {step["product"] for step in steps}
    leaves = {r for step in steps for r in step["reactants"] if r not in known_set and r not in produced}
    swaps = [swap_registry[leaf] for leaf in leaves if leaf in swap_registry]
    return steps, swaps


# ---------------------------------------------------------------------------
# DFS retrosynthesis search
# ---------------------------------------------------------------------------

def _search_route(
    target_smiles: str,
    known_set: set[str],
    known_mols: dict[str, Chem.Mol],
    max_depth: int,
    visited: frozenset[str],
    budget: list[int],
    cache: dict[tuple[str, int], tuple[list[dict], int] | None],
    swap_registry: dict[str, dict],
    min_block_similarity: float,
    allow_swap: bool = True,
) -> tuple[list[dict], int] | None:
    target_c = _canon(target_smiles)
    if target_c is None:
        return None
    if target_c in known_set:
        return [], 0

    cache_key = (target_c, max_depth)
    if cache_key in cache and allow_swap:
        return cache[cache_key]

    if target_c in visited or budget[0] <= 0:
        if allow_swap:
            cache[cache_key] = None
        return None
    budget[0] -= 1

    next_visited = visited | {target_c}
    best: tuple[list[dict], int] | None = None

    if max_depth > 0:
        for cand in _decompose_one_step(target_c):
            sub_steps: list[dict] = []
            sub_swaps = 0
            feasible = True
            for precursor in cand["precursors"]:
                sub = _search_route(
                    precursor, known_set, known_mols, max_depth - 1, next_visited,
                    budget, cache, swap_registry, min_block_similarity,
                )
                if sub is None:
                    feasible = False
                    break
                sub_steps.extend(sub[0])
                sub_swaps += sub[1]
            if not feasible:
                continue

            this_step = {
                "reactants": cand["precursors"],
                "product": target_c,
                "matched_smarts": cand["smarts"],
            }
            total_steps = sub_steps + [this_step]
            total = (total_steps, sub_swaps)
            if best is None or (total[1], len(total[0])) < (best[1], len(best[0])):
                best = total

    if allow_swap:
        block, sim = _closest_known_block(target_c, known_mols, min_block_similarity)
        if block is not None:
            swap_candidate = ([], 1)
            if best is None or (swap_candidate[1], 0) < (best[1], len(best[0])):
                best = swap_candidate
                swap_registry[target_c] = {
                    "needed_smiles": target_c,
                    "suggested_building_block": block,
                    "similarity": round(sim, 3),
                }

    if allow_swap:
        cache[cache_key] = best
    return best


def _search_route_discovery(
    target_smiles: str,
    max_depth: int,
    visited: frozenset[str],
    budget: list[int],
    cache: dict[tuple[str, int], tuple[list[dict], list[str]]],
) -> tuple[list[dict], list[str]] | None:
    """Decompose with NO building-block list, using _COMMON_SMARTS."""
    target_c = _canon(target_smiles)
    if target_c is None:
        return [], [target_smiles]

    if target_c in visited:
        return None  # cycle — infeasible

    cache_key = (target_c, max_depth)
    if cache_key in cache:
        return cache[cache_key]

    if budget[0] <= 0:
        return [], [target_c]
    budget[0] -= 1

    next_visited = visited | {target_c}
    best: tuple[list[dict], list[str]] | None = None

    if max_depth > 0:
        for cand in _decompose_one_step(target_c, smarts_list=_COMMON_SMARTS):
            sub_steps: list[dict] = []
            sub_leaves: list[str] = []
            feasible = True
            for precursor in cand["precursors"]:
                sub = _search_route_discovery(
                    precursor, max_depth - 1, next_visited, budget, cache,
                )
                if sub is None:
                    feasible = False
                    break
                sub_steps.extend(sub[0])
                sub_leaves.extend(sub[1])
            if not feasible:
                continue

            this_step = {
                "reactants": cand["precursors"],
                "product": target_c,
                "matched_smarts": cand["smarts"],
            }
            total = (sub_steps + [this_step], sub_leaves)
            if best is None or (len(total[1]), len(total[0])) < (len(best[1]), len(best[0])):
                best = total

    if best is None:
        best = [], [target_c]

    cache[cache_key] = best
    return best


def _design_route_once(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int,
    min_block_similarity: float,
) -> tuple[list[dict] | None, list[dict]]:
    known_set = {c for c in (_canon(s) for s in known_smiles) if c}
    known_mols = {smi: Chem.MolFromSmiles(smi) for smi in known_set}
    cache: dict[tuple[str, int], tuple[list[dict], int] | None] = {}
    swap_registry: dict[str, dict] = {}
    budget = [_MAX_SEARCH_NODES]

    result = _search_route(
        target_smiles, known_set, known_mols, max_depth, frozenset(), budget, cache,
        swap_registry, min_block_similarity, allow_swap=False,
    )
    route_steps = result[0] if result is not None else None
    if route_steps is None:
        return None, []

    produced = {step["product"] for step in route_steps}
    leaves = {
        r for step in route_steps for r in step["reactants"]
        if r not in known_set and r not in produced
    }
    swaps = [swap_registry[leaf] for leaf in leaves if leaf in swap_registry]
    return route_steps, swaps


# ---------------------------------------------------------------------------
# Reaction conditions lookup
# ---------------------------------------------------------------------------

_REACTION_CONDITIONS: dict[str, str] = {
    "[c:1]Br.[NH2:2]>>[c:1][N:2]": "Buchwald-Hartwig amination: Pd2(dba)3 or Pd(OAc)2 + a phosphine ligand (BINAP, XPhos, or similar), Cs2CO3 or K3PO4 base, toluene or dioxane, 80-110C.",
    "[c:1]Br.[NH1:2]>>[c:1][N:2]": "Buchwald-Hartwig amination: Pd2(dba)3 or Pd(OAc)2 + a phosphine ligand (BINAP, XPhos, or similar), Cs2CO3 or K3PO4 base, toluene or dioxane, 80-110C.",
    "[c:1]I.[NH2:2]>>[c:1][N:2]": "Buchwald-Hartwig amination: Pd2(dba)3 or Pd(OAc)2 + a phosphine ligand (BINAP, XPhos, or similar), Cs2CO3 or K3PO4 base, toluene or dioxane, 80-110C.",
    "[c:1]I.[NH1:2]>>[c:1][N:2]": "Buchwald-Hartwig amination: Pd2(dba)3 or Pd(OAc)2 + a phosphine ligand (BINAP, XPhos, or similar), Cs2CO3 or K3PO4 base, toluene or dioxane, 80-110C.",
    "[c:1][Br,I].[c:3]B(O)O>>[c:1][c:3]": "Suzuki-Miyaura coupling: Pd(PPh3)4 or Pd(dppf)Cl2, K2CO3 or Cs2CO3 base, dioxane/water or THF/water, 60-90C.",
    "[c:1][Br,I].[CH:2]#[C:3]>>[c:1][C:2]#[C:3]": "Sonogashira coupling: Pd(PPh3)2Cl2 + CuI co-catalyst, amine base (Et3N or iPr2NH), THF or DMF, room temp to 60C.",
    "[#6:4][C:2](=[O:3])Cl.[cH:1]>>[c:1][C:2](=[O:3])[#6:4]": "Friedel-Crafts acylation: AlCl3 (stoichiometric or excess), DCM or CS2, 0C to room temp.",
    "[NH:1].[S:2](=O)(=O)Cl>>[N:1][S:2](=O)(=O)": "Sulfonamide formation: mild base (pyridine, Et3N, or aqueous NaOH/Na2CO3), DCM or THF, 0C to room temp.",
    "[NH2:1].[N:2]=C=O>>[N:1]C(=O)[N:2]": "Urea formation from isocyanate + amine: often no catalyst needed, DCM or THF, 0C to room temp.",
    "[NH1;R:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]": "Amide coupling: a coupling reagent (EDC/HOBt, HATU, or T3P) + base (DIPEA), DMF or DCM, room temp — or via the acid chloride/ester route.",
    "[NH2:1].[C:2](=[O:3])[OH]>>[N:1][C:2]=[O:3]": "Amide coupling: a coupling reagent (EDC/HOBt, HATU, or T3P) + base (DIPEA), DMF or DCM, room temp — or via the acid chloride/ester route.",
    "[C:1]=[C:2]>>[C:1]1O[C:2]1": "Epoxidation: mCPBA, DCM, 0C to room temp (or H2O2/catalyst for industrial scale).",
    "[NH2:1].[Cl]C(=O)OC(C)(C)C>>[N:1]C(=O)OC(C)(C)C": "Boc protection: Boc2O (di-tert-butyl dicarbonate) is more common than Boc-Cl in practice, with a base (Et3N, DMAP, or aqueous NaOH), DCM/THF/water, room temp.",
    "[c:1]([OH:7])[c:2][C:3](=[O:8])[CH3:4].[#6:6][CH:5]=[O:9]>>[O:7]1[CH:5]([#6:6])[CH2:4][C:3](=[O:8])[c:2][c:1]1": "Flavanone synthesis: base or acid catalysis (NaOH/EtOH or piperidine for the initial Claisen-Schmidt, then acid- or base-catalyzed oxa-Michael cyclization), often one-pot.",
}


# ---------------------------------------------------------------------------
# Module-level tool implementations (called by the toolset class methods)
# ---------------------------------------------------------------------------

def synthetic_accessibility_score(smiles: str) -> dict:
    score = _sa_score(smiles)
    if score is None:
        return {
            "smiles": smiles,
            "sa_score": None,
            "interpretation": "Could not score — invalid SMILES or scorer unavailable.",
        }
    if score < 3:
        interpretation = "Looks straightforward — common fragments, low complexity."
    elif score < 6:
        interpretation = "Moderate complexity — plausible but not trivial."
    else:
        interpretation = "Looks synthetically demanding — unusual fragments and/or high structural complexity."
    return {"smiles": smiles, "sa_score": round(score, 2), "interpretation": interpretation}


def design_route_mcts(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int = 4,
    iterations: int = 300,
    min_block_similarity: float = 0.6,
) -> dict:
    route_steps, swaps = _mcts_search_route(
        target_smiles, known_smiles, max_depth, iterations, min_block_similarity, _COMMON_SMARTS,
    )

    if route_steps is None:
        return {
            "route_found": False,
            "steps": [],
            "recommended_building_blocks": [],
            "building_block_swaps": [],
            "message": (
                f"MCTS search ({iterations} iterations, max_depth={max_depth}) found no "
                f"fully-validated route to the given building blocks. This target may "
                f"require chemistry outside the template library, more iterations, a "
                f"deeper search, or a lower min_block_similarity."
            ),
        }

    recheck_failures = []
    for step in route_steps:
        ok, msg = auto_validate_reaction(step["reactants"], step["product"])
        if not ok:
            recheck_failures.append(f"{step['reactants']} -> {step['product']}: {msg}")
    if recheck_failures:
        return {
            "route_found": False,
            "steps": [],
            "recommended_building_blocks": [],
            "building_block_swaps": [],
            "message": f"MCTS extraction returned an unconfirmed step — discarded. Failures: {recheck_failures}",
        }

    for i, step in enumerate(route_steps, start=1):
        step["reaction_number"] = i

    swap_note = ""
    if swaps:
        swap_note = (
            " NOTE: this route is only valid if the following building block "
            "substitution(s) are made — the given building block does NOT match "
            f"exactly: {swaps}."
        )

    return {
        "route_found": True,
        "steps": route_steps,
        "recommended_building_blocks": [],
        "building_block_swaps": swaps,
        "message": (
            f"MCTS search found a {len(route_steps)}-step route in {iterations} "
            f"iterations; every step independently re-confirmed via "
            f"auto_validate_reaction.{swap_note}"
        ),
    }


def canonicalize_smiles(smiles: list[str]) -> dict[str, str | None]:
    """Canonicalize and validate a list of SMILES strings using RDKit.

    For each input, returns the canonical SMILES if parseable, or an error
    string prefixed with 'ERROR:' if RDKit cannot parse it.

    Args:
        smiles: List of SMILES strings to canonicalize.

    Returns:
        dict mapping each input SMILES to its canonical form or an error string.
    """
    result = {}
    for smi in smiles:
        if not smi or not smi.strip():
            result[smi] = "ERROR: empty string"
            continue
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            result[smi] = f"ERROR: RDKit could not parse '{smi}'"
        else:
            result[smi] = Chem.MolToSmiles(mol, canonical=True)
    return result


async def literature_lookup(query: str, max_results: int = 5) -> dict:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.crossref.org/works",
                params={"query": query, "rows": max_results},
            )
            response.raise_for_status()
            data = response.json()
    except Exception as e:
        return {"query": query, "results": [], "error": str(e)}

    results = []
    for item in data.get("message", {}).get("items", []):
        title_list = item.get("title", [])
        title = title_list[0] if title_list else "(no title)"
        doi = item.get("DOI")
        year = None
        date_parts = item.get("published-print", item.get("published-online", {})).get("date-parts")
        if date_parts and date_parts[0]:
            year = date_parts[0][0]
        results.append({
            "title": title,
            "doi": doi,
            "year": year,
            "url": f"https://doi.org/{doi}" if doi else None,
        })
    return {"query": query, "results": results, "error": None}


async def pubchem_lookup(smiles: str) -> dict:
    import urllib.parse
    empty = {
        "smiles": smiles, "found": False, "cid": None, "synonyms": [],
        "pubmed_ids": [], "patent_ids": [], "registry_ids": [], "error": None,
    }
    quoted = urllib.parse.quote(smiles, safe="")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            cid_resp = await client.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quoted}/cids/JSON"
            )
            if cid_resp.status_code == 404:
                return empty
            cid_resp.raise_for_status()
            cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
            if not cids:
                return empty
            cid = cids[0]

            async def xrefs(kind: str) -> list[str]:
                r = await client.get(
                    f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/{kind}/JSON"
                )
                if r.status_code == 404:
                    return []
                r.raise_for_status()
                info = r.json().get("InformationList", {}).get("Information", [{}])[0]
                return [str(x) for x in info.get(kind, [])]

            syn_resp = await client.get(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
            )
            if syn_resp.status_code == 404:
                synonyms = []
            else:
                syn_resp.raise_for_status()
                syn_info = syn_resp.json().get("InformationList", {}).get("Information", [{}])[0]
                synonyms = syn_info.get("Synonym", [])
            pubmed_ids = await xrefs("PubMedID")
            patent_ids = await xrefs("PatentID")
            registry_ids = await xrefs("RegistryID")
    except Exception as e:
        return {**empty, "error": str(e)}

    return {
        "smiles": smiles,
        "found": True,
        "cid": cid,
        "synonyms": synonyms,
        "pubmed_ids": pubmed_ids,
        "patent_ids": patent_ids,
        "registry_ids": registry_ids,
        "error": None,
    }


def extract_template_from_reaction(reactant_smiles: list[str], product_smiles: str) -> dict:
    if Indigo is None or _rdchiral_extract is None:
        return {
            "forward_smarts": None, "self_consistent": False,
            "error": "Indigo and/or rdchiral's template extractor is not installed.",
        }

    try:
        indigo = Indigo()
        unmapped = f"{'.'.join(reactant_smiles)}>>{product_smiles}"
        rxn = indigo.loadReaction(unmapped)
        rxn.automap("discard")
        mapped = rxn.smiles()
        mapped_reactants, mapped_products = mapped.split(">>")
    except Exception as e:
        return {"forward_smarts": None, "self_consistent": False, "error": f"atom mapping failed: {e}"}

    try:
        extracted = _rdchiral_extract({
            "reactants": mapped_reactants, "products": mapped_products, "_id": "candidate",
        })
        retro_smarts = extracted.get("reaction_smarts") if isinstance(extracted, dict) else None
        if not retro_smarts:
            return {"forward_smarts": None, "self_consistent": False, "error": "template extraction found no reacting atoms"}
    except Exception as e:
        return {"forward_smarts": None, "self_consistent": False, "error": f"template extraction failed: {e}"}

    retro_lhs, retro_rhs = retro_smarts.split(">>")
    forward_smarts = f"{retro_rhs}>>{retro_lhs}"

    try:
        rxn_check = rdChemReactions.ReactionFromSmarts(forward_smarts)
        reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        expected = Chem.CanonSmiles(product_smiles)
        self_consistent = False
        for perm in permutations(reactant_mols):
            for products in rxn_check.RunReactants(perm):
                for m in products:
                    try:
                        Chem.SanitizeMol(m)
                        if Chem.CanonSmiles(Chem.MolToSmiles(m, ignoreAtomMapNumbers=True)) == expected:
                            self_consistent = True
                    except Exception:
                        continue
    except Exception as e:
        return {"forward_smarts": forward_smarts, "self_consistent": False, "error": f"self-check failed: {e}"}

    return {"forward_smarts": forward_smarts, "self_consistent": self_consistent, "error": None}



def reaction_conditions(matched_smarts: str) -> dict:
    return {
        "matched_smarts": matched_smarts,
        "conditions": _REACTION_CONDITIONS.get(matched_smarts),
    }


def design_route(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int = 4,
    min_block_similarity: float = 0.6,
) -> dict:
    accepted_blocks = list(known_smiles)
    recommended_additions: list[str] = []
    route_steps: list[dict] | None = None
    swaps: list[dict] = []

    for _ in range(3):
        route_steps, swaps = _design_route_once(
            target_smiles, accepted_blocks, max_depth, min_block_similarity,
        )
        if route_steps is None or not swaps:
            break
        new_adds = [
            s["needed_smiles"] for s in swaps if s["needed_smiles"] not in accepted_blocks
        ]
        if not new_adds:
            break
        accepted_blocks.extend(new_adds)
        recommended_additions.extend(new_adds)

    if route_steps is None:
        discovery_cache: dict[tuple[str, int], tuple[list[dict], list[str]]] = {}
        discovery_budget = [_MAX_SEARCH_NODES]
        discovery_result = _search_route_discovery(
            target_smiles, max_depth, frozenset(), discovery_budget, discovery_cache,
        )
        discovery_steps, discovery_leaves = discovery_result if discovery_result else ([], [])
        target_c = _canon(target_smiles)

        if discovery_steps and discovery_leaves != [target_c]:
            for i, step in enumerate(discovery_steps, start=1):
                step["reaction_number"] = i
            leaves_list = sorted(set(discovery_leaves))
            return {
                "route_found": True,
                "steps": discovery_steps,
                "recommended_building_blocks": leaves_list,
                "building_block_swaps": [],
                "message": (
                    f"The given building blocks could not reach this target at all — "
                    f"this route was found instead by decomposing the target on its own, "
                    f"using only reaction templates specific enough to trust without a "
                    f"given building block to anchor against. Found a "
                    f"{len(discovery_steps)}-step route requiring these building blocks "
                    f"instead: {leaves_list}. Every step was independently confirmed via "
                    f"auto_validate_reaction. Review before treating as resolved — this "
                    f"was discovered without your input on what's actually available."
                ),
            }

        return {
            "route_found": False,
            "steps": [],
            "recommended_building_blocks": [],
            "building_block_swaps": [],
            "message": (
                f"No fully-validated route to the given building blocks was found "
                f"within {max_depth} step(s) using the available reaction templates, "
                f"even allowing building-block substitutions down to "
                f"{min_block_similarity:.2f} similarity. A discovery-only search "
                f"(ignoring the given building blocks, using only templates specific "
                f"enough to trust blindly) also found no plausible disconnection. This "
                f"target may require chemistry outside the template library, a deeper "
                f"search (raise max_depth), or a lower min_block_similarity."
            ),
        }

    for i, step in enumerate(route_steps, start=1):
        step["reaction_number"] = i

    swap_note = ""
    if recommended_additions:
        swap_note = (
            " NOTE: the originally given building blocks were not sufficient on their "
            "own — this route additionally requires sourcing: "
            f"{recommended_additions} (each independently confirmed as a real precursor "
            "during the search, not just a similar-looking substitute)."
        )
    if swaps:
        swap_note += (
            " UNRESOLVED: even after auto-accepting the above, the search still could "
            f"not find an exact match for: {swaps} — treat this route as unconfirmed "
            "until that gap is addressed."
        )

    return {
        "route_found": True,
        "steps": route_steps,
        "recommended_building_blocks": recommended_additions,
        "building_block_swaps": swaps,
        "message": (
            f"Found a {len(route_steps)}-step route; "
            f"every step was independently confirmed via auto_validate_reaction during "
            f"the search. Still pending an independent re-check before this can be "
            f"treated as resolved.{swap_note}"
        ),
    }


def compare_search_strategies(
    target_smiles: str, known_smiles: list[str], max_depth: int = 4,
) -> dict:
    dfs_result = design_route(target_smiles, known_smiles, max_depth=max_depth)
    mcts_result = design_route_mcts(target_smiles, known_smiles, max_depth=max_depth)

    agree = (
        dfs_result["route_found"] == mcts_result["route_found"]
        and len(dfs_result["steps"]) == len(mcts_result["steps"])
    )
    if dfs_result["route_found"] and mcts_result["route_found"]:
        if agree:
            summary = f"Both engines found a {len(dfs_result['steps'])}-step route."
        else:
            summary = (
                f"Engines DISAGREE: DFS found {len(dfs_result['steps'])} step(s), "
                f"MCTS found {len(mcts_result['steps'])} step(s) — review both before "
                f"trusting either as the only/best answer."
            )
    elif dfs_result["route_found"] != mcts_result["route_found"]:
        found_by = "DFS" if dfs_result["route_found"] else "MCTS"
        summary = f"Only {found_by} found a route — the other reported route_found: false."
    else:
        summary = "Neither engine found a route."

    return {
        "dfs_result": dfs_result,
        "mcts_result": mcts_result,
        "agree": agree,
        "summary": summary,
    }


def _strip_tag(s: str, open_tag: str, close_tag: str) -> str:
    if s.startswith(open_tag):
        s = s[len(open_tag):]
    if s.endswith(close_tag):
        s = s[: -len(close_tag)]
    return s


def _parse_route_input(text: str) -> dict:
    text = text.strip()
    parts = text.split(",", 2)
    if len(parts) < 3:
        raise ValueError("Input does not match <SMILES>,<difficulty>,<JSON> format")
    _, _, json_blob = parts
    json_blob = json_blob.strip()

    # Try multiple unescaping strategies to handle LLM re-encoding artifacts.
    # Qwen tends to add an extra layer of "" wrapping around the JSON blob.
    candidates = []

    # normal: "{""k"":...}" → strip one outer quote pair, unescape ""
    if json_blob.startswith('"') and json_blob.endswith('"'):
        candidates.append(json_blob[1:-1].replace('""', '"'))

    # double-wrap: ""{""k"":...}"" or ""{""k"":...}"
    if json_blob.startswith('""'):
        stripped = json_blob[2:]
        if stripped.endswith('""'):
            stripped = stripped[:-2]
        elif stripped.endswith('"'):
            stripped = stripped[:-1]
        candidates.append(stripped.replace('""', '"'))

    # fallback: unescape without stripping
    candidates.append(json_blob.replace('""', '"'))
    # last resort: raw blob
    candidates.append(json_blob)

    last_err = None
    for blob in candidates:
        try:
            return json.loads(blob)
        except json.JSONDecodeError as e:
            last_err = e
    raise ValueError(f"Could not parse route input: {last_err}")


def suggest_reactant_fix(
    reactant_smiles: list[str], expected_product: str
) -> dict:
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
            f"via reverse SMARTS: {best_fix['smarts']} — pending an independent "
            f"re-check before this can be treated as resolved."
        ),
        "corrected_reactants": best_fix["corrected_smiles"],
        "matched_smarts": best_fix["smarts"],
    }


def correct_route(route_input: str) -> dict:
    try:
        data = _parse_route_input(route_input)
    except Exception as e:
        return {
            "steps": [],
            "all_resolved": False,
            "error": f"Could not parse route input: {e}",
        }

    if not isinstance(data, dict):
        return {
            "steps": [],
            "all_resolved": False,
            "error": (
                "Parsed route JSON must be an object with \"reactions\" and "
                f"\"building_blocks\" keys, got {type(data).__name__} instead"
            ),
        }

    reactions = data.get("reactions", [])
    if not reactions:
        return {
            "steps": [],
            "all_resolved": False,
            "error": "No 'reactions' list found in parsed input",
        }
    if not isinstance(reactions, list) or any(not isinstance(r, dict) for r in reactions):
        return {
            "steps": [],
            "all_resolved": False,
            "error": "\"reactions\" must be a list of objects, each with \"reaction_number\", \"reactants\", and \"product\" keys",
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
    # produced by some other step or listed as a building block.
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


# ---------------------------------------------------------------------------
# Toolset class
# ---------------------------------------------------------------------------

class CorrectorToolset(FunctionToolset[AgentDepsT]):
    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.canonicalize_smiles, name="canonicalize_smiles")
        self.add_function(self.synthetic_accessibility_score, name="synthetic_accessibility_score")
        self.add_function(self.design_route_mcts, name="design_route_mcts")
        # self.add_function(self.literature_lookup, name="literature_lookup")
        self.add_function(self.pubchem_lookup, name="pubchem_lookup")
        self.add_function(self.extract_template_from_reaction, name="extract_template_from_reaction")
        self.add_function(self.reaction_conditions, name="reaction_conditions")
        self.add_function(self.compare_search_strategies, name="compare_search_strategies")
        self.add_function(self.design_route, name="design_route")
        self.add_function(self.suggest_reactant_fix, name="suggest_reactant_fix")
        self.add_function(self.correct_route, name="correct_route")

    async def synthetic_accessibility_score(self, smiles: str) -> dict:
        """Scores how easy or hard a molecule looks to synthesize using RDKit's SA_Score
        (Ertl & Schuffenhauer, 2009) — a rule-based heuristic, NOT a prediction of
        whether a specific route exists. Use to compare candidates or flag a proposed
        building block that looks suspiciously complex.

        Args:
            smiles (str): The molecule to score.

        Returns:
            dict: {"smiles": str, "sa_score": float | None, "interpretation": str}
        """
        return synthetic_accessibility_score(smiles)

    async def design_route_mcts(
        self,
        target_smiles: str,
        known_smiles: list[str],
        max_depth: int = 4,
        iterations: int = 300,
        min_block_similarity: float = 0.6,
    ) -> dict:
        """Alternative to design_route using MCTS (UCB1 selection, heuristic evaluation,
        backpropagation). Uses SAScore + similarity-to-known-blocks as the evaluation
        heuristic. Every returned step is independently re-confirmed via
        auto_validate_reaction — the heuristic only affects which route is searched for.
        May find a DIFFERENT valid route from design_route, or miss one DFS would reach.

        Args:
            target_smiles (str): The target molecule to decompose.
            known_smiles (list[str]): SMILES already available as building blocks.
            max_depth (int): Maximum number of reaction steps to search.
            iterations (int): Number of MCTS cycles — spent adaptively not exhaustively.
            min_block_similarity (float): Minimum Tanimoto similarity to accept a
                building-block substitution as a last resort.

        Returns:
            dict: {"route_found": bool, "steps": [...],
                "recommended_building_blocks": [...],
                "building_block_swaps": [...], "message": str}
        """
        return design_route_mcts(
            target_smiles, known_smiles, max_depth, iterations, min_block_similarity
        )

    def canonicalize_smiles(self, smiles: list[str]) -> dict[str, str | None]:
        """Canonicalize and validate SMILES strings using RDKit. Call this first if any
        SMILES look malformed, non-canonical, or came from an untrusted source. Returns
        the canonical form for valid SMILES, or an 'ERROR:' prefixed message for ones
        RDKit cannot parse.

        Args:
            smiles (list[str]): SMILES strings to canonicalize.

        Returns:
            dict mapping each input to its canonical SMILES or an error string.
        """
        return canonicalize_smiles(smiles)

    async def literature_lookup(self, query: str, max_results: int = 5) -> dict:
        """Searches real published chemistry literature via CrossRef (free, no API key).
        Call this when design_route/correct_route come up empty and you suspect the
        target needs a named reaction not in the template library. Returns actual paper
        titles/DOIs/years — never a fabricated citation. This only finds CANDIDATE
        literature; chemistry still needs independent verification.

        Args:
            query (str): Specific functional-group/reaction-class keywords
                (e.g. "3-amino-1,2,4-triazole benzylidene cyclocondensation
                dihydropyrimidine"), not a SMILES string.
            max_results (int): Maximum number of results to return.

        Returns:
            dict: {"query": str, "results": [{"title": str, "doi": str,
                "year": int | None, "url": str}, ...], "error": str | None}
        """
        return await literature_lookup(query, max_results)

    async def pubchem_lookup(self, smiles: str) -> dict:
        """Checks whether a molecule is a known compound via PubChem's free public REST
        API. A known compound implies some route exists. Returns CID, synonyms, and
        cross-references. Never fabricates a CID — returns found=False if not in
        PubChem rather than inventing a record.

        Args:
            smiles (str): SMILES of the molecule to look up.

        Returns:
            dict: {"smiles": str, "found": bool, "cid": int | None,
                "synonyms": list[str], "pubmed_ids": list[str],
                "patent_ids": list[str], "registry_ids": list[str],
                "error": str | None}
        """
        return await pubchem_lookup(smiles)

    async def extract_template_from_reaction(
        self, reactant_smiles: list[str], product_smiles: str
    ) -> dict:
        """Derives a candidate reaction template from ONE real, correct example reaction
        using Indigo atom-mapping + rdchiral template extraction. Does NOT add anything
        to the template library — only proposes a candidate. Self-validates against the
        single given example, but that does NOT prove generality. Re-test against at
        least one other example via correct_route/design_route before trusting.

        Args:
            reactant_smiles (list[str]): Reactants of one real, correct example.
            product_smiles (str): Product of that same example.

        Returns:
            dict: {"forward_smarts": str | None, "self_consistent": bool,
                "error": str | None}
        """
        return extract_template_from_reaction(reactant_smiles, product_smiles)

    async def reaction_conditions(self, matched_smarts: str) -> dict:
        """Looks up typical real-world reagents/conditions for a template that
        design_route or auto_validate_reaction matched. Informational only —
        well-established textbook conditions, not verified per-substrate. A missing
        entry means this template is not yet annotated, not that no conditions exist.

        Args:
            matched_smarts (str): The exact SMARTS string from a tool's
                "matched_smarts" field.

        Returns:
            dict: {"matched_smarts": str, "conditions": str | None}
        """
        return reaction_conditions(matched_smarts)

    async def compare_search_strategies(
        self,
        target_smiles: str,
        known_smiles: list[str],
        max_depth: int = 4,
    ) -> dict:
        """Runs BOTH design_route (exhaustive DFS) and design_route_mcts (heuristic MCTS)
        on the same target/building blocks and reports results side by side. The two
        engines can disagree — use this when you want to verify they agree before trusting
        a specific route.

        Args:
            target_smiles (str): The target molecule to decompose.
            known_smiles (list[str]): SMILES already available as building blocks.
            max_depth (int): Maximum number of reaction steps to search.

        Returns:
            dict: {"dfs_result": dict, "mcts_result": dict, "agree": bool,
                "summary": str}
        """
        return compare_search_strategies(target_smiles, known_smiles, max_depth)

    async def design_route(
        self,
        target_smiles: str,
        known_smiles: list[str],
        max_depth: int = 4,
        min_block_similarity: float = 0.6,
    ) -> dict:
        """Searches for a complete, fully-validated multi-step retrosynthetic route from
        target_smiles down to the given building blocks. Uses recursive backward search
        with immediate auto_validate_reaction confirmation at every step — no invalid
        decomposition is ever used. Returns the shortest fully-validated route found.
        Auto-resolves building-block mismatches by similarity over up to 3 rounds.

        Call this when you need a brand-new multi-step sequence — suggest_reactant_fix
        and correct_route only patch a single already-given step.

        Args:
            target_smiles (str): The target molecule to decompose.
            known_smiles (list[str]): SMILES already available as building blocks.
            max_depth (int): Maximum number of reaction steps to search.
            min_block_similarity (float): Minimum Tanimoto similarity to accept a
                building-block substitution on the first pass.

        Returns:
            dict: {"route_found": bool, "steps": [...],
                "recommended_building_blocks": list[str],
                "building_block_swaps": [...], "message": str}
        """
        return design_route(target_smiles, known_smiles, max_depth, min_block_similarity)

    async def suggest_reactant_fix(
        self, reactant_smiles: list[str], expected_product: str
    ) -> dict:
        """Diagnoses a failed reaction step and proposes a corrected reactant list.
        If already valid, says so. Otherwise applies every reverse template in the SMARTS
        library to the expected product, matches candidates to the given reactants by
        Tanimoto similarity, and re-validates the proposed fix. A returned fix is
        guaranteed to actually produce the expected product — but still needs
        re-validation by the validation agent before treating as resolved.

        Args:
            reactant_smiles (list[str]): The (possibly wrong) reactant SMILES.
            expected_product (str): The expected product SMILES.

        Returns:
            dict: {"already_valid": bool, "fix_found": bool, "message": str,
                "corrected_reactants": list[str] | None, "matched_smarts": str | None}
        """
        return suggest_reactant_fix(reactant_smiles, expected_product)

    async def correct_route(self, route_input: str) -> dict:
        """Parses a full <SMILES>,<difficulty>,<JSON> retrosynthesis route, validates
        every reaction step with auto_validate_reaction, and calls suggest_reactant_fix
        for any step that fails. Also checks chain consistency — a "fixed" step may
        break the multi-step chain if its corrected reactants are not produced elsewhere
        in the route. All JSON parsing and per-step logic is handled deterministically
        in Python; no chemistry judgment is left to the caller.

        Args:
            route_input (str): The raw route input exactly as given, i.e.
                "<SMILES>,<difficulty>,<JSON>" (JSON may have doubled double-quotes).

        Returns:
            dict: {"steps": [...], "all_resolved": bool,
                "chain_consistent": bool, "error": str | None}
        """
        return correct_route(route_input)
