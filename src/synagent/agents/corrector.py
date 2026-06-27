import json
import math
import os
import re
import sys
from itertools import permutations

import httpx
from pydantic_ai import Agent
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdChemReactions

from .validation import _COMMON_SMARTS, _DISCOVERY_SAFE_SMARTS, auto_validate_reaction
from ..chemspacetool import ChemspaceDeps, check_building_block_economics
from ..tokenmanager import ChemspaceTokenManager

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

import logfire

logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()


SYSTEM_PROMPT = """You are SynAgent's route corrector.

## Input format
You receive one of:
(a) A full route in the format <SMILES>,<difficulty>,<JSON> (the JSON may have doubled
    double-quotes "" from CSV escaping) — this is the same format the validation agent
    parses. When you get this, call correct_route with the ENTIRE raw input string,
    unmodified, exactly as given. Do not parse the JSON yourself and do not extract
    individual reactants/products by hand — correct_route does all of that internally.
(b) A single isolated reaction step: a list of reactant SMILES and an expected product
    SMILES. When you get this, call suggest_reactant_fix with exactly those inputs.
(c) A request to design or redesign a COMPLETE multi-step route from a target molecule
    down to a given set of building blocks (e.g. because a single-step fix left the
    route disconnected, or because no route exists yet). When you get this, call
    design_route with the target SMILES and the building block SMILES — do not attempt
    to invent a multi-step sequence yourself by reasoning about mechanisms; design_route
    does an exhaustive backward search over the same template library and only returns
    a route where every single step is already confirmed valid.

If design_route/correct_route come up empty and you suspect the target needs a real
named reaction not in the template library, call literature_lookup with specific
functional-group/reaction-class keywords BEFORE telling the user it's unresolvable —
this returns real paper titles/DOIs, never a fabricated citation. A literature hit is
only a lead, not proof: still verify any reaction it suggests by mass balance (do the
atoms add up?) before treating it as a real answer, the same way every other template in
this system was verified before being trusted.

You can also call synthetic_accessibility_score on a candidate molecule (e.g. a proposed
building block or intermediate) to flag whether it looks suspiciously complex — this is
a rule-based heuristic (RDKit's SA_Score), not a guarantee a route exists or works, so
never use it as the sole basis for accepting or rejecting a route.

## Rules
Do not assess the chemistry yourself. Do not invent a fix, a mechanism, or a class of
intermediate. Do not fill in gaps the tool left unresolved. Report exactly what the tool
returned for every step, including steps it could not fix — say "unfixable" and quote
the tool's message verbatim rather than proposing your own chemical reasoning.
The tools use chemistry checks internally only to search for and rank candidate fixes —
that internal search is NOT a substitute for proper validation. You are proposing a
modification, not certifying one. Always frame your output as a suggestion that still
needs to be re-validated by the validation agent before anyone treats it as resolved;
never claim a fix is "confirmed" or "guaranteed valid" yourself.

When you call correct_route, also check the "chain_consistent" field and any
"chain_warning" on individual steps. A step being individually "fixed" does not mean the
whole route works — if chain_consistent is false, say so explicitly and report which
step(s) are now disconnected from the rest of the route, quoting the chain_warning
verbatim. Do not claim the route is valid overall — that determination belongs to the
validation agent's re-check, not to you, even if all_resolved and chain_consistent are
both true.""".strip()

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
    # useChirality=True matters: without it, two enantiomers score as 1.0 (identical),
    # which would let the building-block swap-matching logic (_closest_known_block)
    # silently treat the wrong stereoisomer as a perfect substitute.
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)


def _similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(_fingerprint(mol_a), _fingerprint(mol_b))


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _sa_score(smiles: str) -> float | None:
    """Ertl/Schuffenhauer synthetic accessibility score (RDKit's bundled SA_Score
    contrib implementation) — a fragment-frequency + complexity-penalty heuristic, NOT
    a learned model. Scale ~1 (easy/simple, common fragments) to ~10 (hard/complex,
    unusual ring systems, many stereocenters). Returns None if the SMILES is invalid or
    the scorer isn't available — callers must not treat None as a score of 0."""
    if _sascorer is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(_sascorer.calculateScore(mol))
    except Exception:
        return None


@agent.tool_plain
def synthetic_accessibility_score(smiles: str) -> dict:
    """Scores how easy or hard a molecule looks to synthesize, using RDKit's bundled
    SA_Score (Ertl & Schuffenhauer, 2009) — a rule-based heuristic combining fragment
    commonality (via a precomputed fragment-frequency table) with a structural
    complexity penalty (ring systems, stereocenters, macrocycles). This is NOT a
    prediction of whether a specific route exists or works — design_route's confirmed
    steps are the source of truth for that. Use this to compare CANDIDATES (e.g. which
    of several valid disconnections looks more synthetically reasonable) or to flag a
    proposed building block that looks suspiciously complex for a "simple starting
    material."

    Args:
        smiles (str): The molecule to score.

    Returns:
        dict: {
            "smiles": str,
            "sa_score": float | None,  # ~1 (easy) to ~10 (hard); None if invalid SMILES
            "interpretation": str,
        }
    """
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


def _strip_unbalanced_map_numbers(smarts: str) -> str:
    """rdchiral requires atom map numbers introduced or lost across a reaction (present
    on only one side of >>) to be unmapped — our _COMMON_SMARTS entries don't always
    follow that convention (they're written for plain RDKit, which doesn't care). Strips
    map numbers that don't appear on both sides, so any template can be handed to
    rdchiral without manually rewriting it."""
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
    """Fallback decomposition engine for when bare RDKit's RunReactants crashes (e.g. the
    kekulization failures we hit with certain fused heteroaromatic rings). rdchiral
    handles many of these edge cases more gracefully — but it's slower to initialize, so
    this is only tried when the primary engine fails, never run unconditionally. Returns
    an empty list (never raises) if rdchiral isn't installed, can't parse the template,
    or finds nothing — callers should treat this exactly like "no candidates found"."""
    if rdchiralRunText is None:
        return []
    try:
        balanced = _strip_unbalanced_map_numbers(reverse_smarts)
        outcomes = rdchiralRunText(balanced, product_smiles)
    except Exception:
        return []
    results = []
    for outcome in outcomes:
        results.append(outcome.split("."))
    return results


def _decompose_one_step(product_smiles: str, smarts_list: list[str] = _COMMON_SMARTS) -> list[dict]:
    """Tries every reverse SMARTS template on product_smiles. Only returns candidates
    where the forward template, re-applied to the proposed precursors, is independently
    confirmed (via auto_validate_reaction) to actually regenerate the product — so every
    candidate returned here is a single chemically-validated reaction step, never a
    guess.

    smarts_list defaults to the full _COMMON_SMARTS library (used by the constrained
    search, which is anchored to known building blocks so an overly-generic template
    match just won't connect to anything real). Pass _DISCOVERY_SAFE_SMARTS instead for
    blind discovery with no building blocks to anchor against — see that list's comment
    for why the generic templates are excluded there."""
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

        # rdchiral is slower to initialize, so it's only tried as a fallback when bare
        # RDKit's reaction engine actually crashes (e.g. kekulization failures on certain
        # fused heteroaromatic rings) — never run unconditionally alongside it.
        if engine_crashed:
            raw_precursor_lists.extend(_rdchiral_decompose(reverse_smarts, product_smiles))

        for raw_precursors in raw_precursor_lists:
            precursor_smiles = [_canon(s) for s in raw_precursors]
            if any(s is None for s in precursor_smiles):
                continue

            # RDKit's raw reaction-output Mol objects don't always have radical character
            # computed yet (that only becomes apparent after a SMILES round-trip), so
            # re-parse the canonical SMILES and check THOSE for malformed/radical atoms
            # left over from an under-specified template match — not a real molecule.
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

    return candidates


_MAX_SEARCH_NODES = 2000


def _closest_known_block(
    target_c: str, known_mols: dict[str, Chem.Mol], min_similarity: float
) -> tuple[str | None, float]:
    """Finds the known building block most structurally similar to target_c. Used only
    as a last-resort fallback when no template-based synthesis closes the gap — never
    pretends a similar-but-different molecule IS the target."""
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


class _MCTSNode:
    """A molecule-to-be-solved node in the retrosynthesis AND/OR tree. Each "candidate"
    in self.candidates is an OR-choice (one possible single-step decomposition); each
    candidate's precursors are an AND-requirement (every one of them must itself resolve
    for that candidate to count as solved)."""

    __slots__ = ("smiles", "depth_remaining", "candidates", "is_known", "is_terminal", "visits", "total_value")

    def __init__(self, smiles: str, depth_remaining: int):
        self.smiles = smiles
        self.depth_remaining = depth_remaining
        self.candidates: list[dict] | None = None  # set on first expand()
        self.is_known = False
        self.is_terminal = False
        self.visits = 0
        self.total_value = 0.0


def _mcts_heuristic_value(
    smiles: str, known_set: set[str], known_mols: dict[str, Chem.Mol]
) -> float:
    """Heuristic node value in [0, 10] used in place of a trained value network: higher
    means "looks more synthetically accessible / closer to something already
    available." This is a heuristic, not a guarantee — every step in the FINAL extracted
    route is still independently re-confirmed via auto_validate_reaction before being
    returned, exactly like the DFS search. Combines SAScore (lower = simpler, scaled to
    0-5) with similarity to the closest known building block (scaled to 0-5, regardless
    of the swap-acceptance threshold — this is just a heuristic signal for search
    prioritization, not an acceptance decision)."""
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
    """Expands a node once: finds all single-step decomposition candidates and creates
    (or reuses, via node_cache, to collapse identical subtrees) the child node for each
    precursor. No-op if already expanded."""
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
    """UCB1 selection among a node's already-expanded candidates — balances exploiting
    the best-scoring candidate so far against exploring under-visited ones. An unvisited
    candidate always wins (infinite UCB1 score), so every candidate gets tried at least
    once before any gets revisited."""
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
    """One MCTS iteration: select->expand->evaluate->backpropagate, recursing into the
    single LEAST-accessible precursor of the selected candidate (the bottleneck most
    worth exploring further) rather than every precursor, to keep each iteration cheap.
    Returns the value backpropagated to this node (and updates its visit/value stats)."""
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
    """After the MCTS search budget is spent, greedily extracts the best fully-resolved
    route found: at each node, tries candidates in descending order of average
    backpropagated value, recursing into ALL of that candidate's precursors (a true
    AND-requirement, unlike the single-bottleneck simulation), falling back to the same
    honest similarity-based swap as the DFS search if no candidate fully resolves.
    allow_swap is False only for the original target itself (the top-level call) — same
    reasoning as the DFS search: substituting the WHOLE target for "something similar"
    doesn't synthesize the target, so that would be a false "route found" rather than an
    honest substitution of an intermediate precursor reached partway through real
    decomposition."""
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
    """Real MCTS retrosynthesis search (UCB1 selection + heuristic evaluation +
    backpropagation) as an alternative to the brute-force DFS-with-budget in
    _search_route. Uses SAScore + similarity-to-known-blocks as the node-evaluation
    heuristic in place of a trained value network (no training data/infra available) —
    directionally the same idea as AiZynthFinder/ASKCOS's MCTS planners, substantially
    smaller in scope. Returns (route_steps, swaps), matching _design_route_once's
    contract, so design_route can use either engine interchangeably."""
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
    """Recursive backward search: returns (steps, swap_count) — the forward-direction
    steps that fully decompose target_smiles down to entries in known_set, and how many
    of its leaves needed a building-block substitution — or None if no such route was
    found within max_depth steps using the available template library.

    Candidates are ranked by (swap_count, step_count): a longer route using only real,
    exact chemistry always beats a shorter route that leans on a fuzzy building-block
    substitution — swapping is a last resort, never a shortcut preferred for brevity.

    If no real synthesis step can be found for a node (decomposition fails or depth is
    exhausted) but a known building block is structurally very similar, that node is
    accepted as a leaf via a flagged SUBSTITUTION rather than failing outright — recorded
    in swap_registry, never silently treated as identical to what was actually given.
    allow_swap is False only for the original target itself (the top-level call) —
    substituting the WHOLE target for "something similar" doesn't synthesize the target,
    so that would be a false "route found" rather than an honest substitution of an
    intermediate precursor reached partway through real decomposition."""
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
        # Always compare a direct substitution against the best decomposition found —
        # never just a fallback used only when decomposition fails entirely. A direct
        # swap (0 extra steps) can be strictly better than decomposing one level deeper
        # only to swap an even-more-basic precursor (1+ extra steps, same swap count).
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
    """Decomposes target_smiles with NO building-block list at all, using ONLY
    _DISCOVERY_SAFE_SMARTS (never the full _COMMON_SMARTS) — used as a last resort when
    design_route's constrained search fails entirely because the given building blocks
    are completely unrelated to what the target structurally requires. Restricting to the
    discovery-safe subset matters: the generic templates (catch-all coupling, bare
    alcohol<->halide swaps, alpha-halogenation) are loose enough to "explain" cutting
    almost any bond in a molecule, since splitting-then-recombining via the same generic
    template is tautologically self-confirming — fine when anchored to real given
    building blocks, unsafe when discovering from scratch.

    Whenever no further safe template applies (or the search budget runs out), the
    current molecule is accepted unconditionally as a required raw material — never a
    similarity guess, since there is nothing to compare against in this mode. Returns
    None only for a CYCLE (a candidate decomposing back to a molecule already on this
    path) — rejected as infeasible, never accepted as if "you need the target itself"
    were a valid raw material. Ranks candidates by (fewest required raw materials, fewest
    steps)."""
    target_c = _canon(target_smiles)
    if target_c is None:
        return [], [target_smiles]

    if target_c in visited:
        return None  # cycle — infeasible for this path, never cached (path-dependent)

    cache_key = (target_c, max_depth)
    if cache_key in cache:
        return cache[cache_key]

    if budget[0] <= 0:
        return [], [target_c]
    budget[0] -= 1

    next_visited = visited | {target_c}
    best: tuple[list[dict], list[str]] | None = None

    if max_depth > 0:
        for cand in _decompose_one_step(target_c, smarts_list=_DISCOVERY_SAFE_SMARTS):
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
    """Single search pass. Returns (route_steps, swaps) for the given known_smiles, with
    swaps filtered down to only the leaves actually used in the winning route_steps."""
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


@agent.tool_plain
def design_route_mcts(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int = 4,
    iterations: int = 300,
    min_block_similarity: float = 0.6,
) -> dict:
    """Alternative to design_route's brute-force DFS: a real MCTS search (UCB1
    selection, heuristic evaluation, backpropagation) that prioritizes exploring
    promising branches instead of exhaustively trying every template at every node.
    Directionally the same idea as production retrosynthesis planners like
    AiZynthFinder/ASKCOS, which use MCTS guided by a trained value network — this uses
    SAScore + similarity-to-known-blocks as the evaluation heuristic instead, since
    there's no training data/infra here for a learned network. That makes this a
    heuristic-guided search, not a guarantee of finding the same (or best) route
    design_route's exhaustive DFS would — for small/well-covered targets the two will
    often agree, but MCTS may find a DIFFERENT valid route, or miss one DFS would
    eventually reach with enough budget, since it spends its iteration budget exploring
    where the heuristic looks promising rather than everywhere.

    Every step in the returned route is independently re-confirmed via
    auto_validate_reaction before being returned (identical guarantee to design_route) —
    the heuristic only affects WHICH route is searched for and found, never whether a
    returned step is actually valid.

    Args:
        target_smiles (str): The target molecule to decompose.
        known_smiles (list[str]): SMILES already available as building blocks.
        max_depth (int): Maximum number of reaction steps to search.
        iterations (int): Number of MCTS select/expand/simulate/backpropagate cycles —
            roughly analogous to design_route's search budget, but spent adaptively
            rather than exhaustively.
        min_block_similarity (float): Minimum Tanimoto similarity to a known block
            required to accept a building-block substitution as a last resort.

    Returns:
        dict: same shape as design_route's return value — {"route_found": bool,
            "steps": [...], "recommended_building_blocks": [...],
            "building_block_swaps": [...], "message": str}.
    """
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
        # should be unreachable (every candidate is validated during the search itself),
        # but never trust a search algorithm's output without re-checking — if this ever
        # fires, it means a real bug in the MCTS extraction, not a chemistry problem.
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


@agent.tool_plain
async def literature_lookup(query: str, max_results: int = 5) -> dict:
    """Searches real published chemistry literature via CrossRef (free, no API key) —
    call this when design_route/correct_route come up empty and you suspect the target
    needs a real named reaction not in the template library, instead of guessing a
    mechanism. Returns actual paper titles/DOIs/years, never a fabricated citation —
    if CrossRef returns nothing or the request fails, says so plainly rather than
    inventing a plausible-sounding reference.

    This only finds CANDIDATE literature to verify against — it does NOT confirm that a
    paper's reaction actually matches your target. After finding a promising result, the
    chemistry still needs to be independently verified (e.g. via mass balance and
    mechanism, the way design_route's own templates are verified before being trusted).

    Args:
        query (str): Search terms — works best as specific functional-group/reaction-class
            keywords (e.g. "3-amino-1,2,4-triazole benzylidene cyclocondensation
            dihydropyrimidine"), not a SMILES string.
        max_results (int): Maximum number of results to return.

    Returns:
        dict: {
            "query": str,
            "results": [
                {"title": str, "doi": str, "year": int | None, "url": str},
                ...
            ],  # empty list if nothing found or the lookup failed
            "error": str | None,
        }
    """
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


@agent.tool_plain
async def pubchem_lookup(smiles: str) -> dict:
    """Checks whether a molecule (target, intermediate, or proposed building block) is a
    KNOWN compound via PubChem's free public REST API — call this alongside
    literature_lookup when a target has no working route, to find independent evidence
    the compound is real/synthesizable (a known compound implies *some* route exists,
    even if our template library can't find it) and to get cross-references (PubMed,
    patent, registry IDs) for further literature search.

    Note: Reaxys and CAS/SciFinder have NO public API reachable from a normal
    institutional web login — there is nothing this tool (or any tool) can call there.
    PubChem is the only structure-searchable database with a genuinely free, keyless API.

    Never fabricates a CID or reference: if PubChem has no record, returns found=False
    plainly rather than inventing one.

    Args:
        smiles (str): SMILES of the molecule to look up.

    Returns:
        dict: {
            "smiles": str,
            "found": bool,
            "cid": int | None,
            "synonyms": list[str],       # includes the systematic IUPAC-style name if present
            "pubmed_ids": list[str],
            "patent_ids": list[str],
            "registry_ids": list[str],   # vendor/library catalog IDs; a real numeric CAS RN
                                          # would appear here too, but most hits are NOT CAS RNs
            "error": str | None,
        }
    """
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


@agent.tool_plain
def extract_template_from_reaction(reactant_smiles: list[str], product_smiles: str) -> dict:
    """Derives a candidate reaction template from ONE real, correct example reaction —
    e.g. one found via literature_lookup, or one given directly — instead of hand-writing
    SMARTS by trial and error. Pipeline: Indigo (lightweight, rule-based, no ML model)
    auto-maps the atoms between reactants and product, then rdchiral extracts a SMARTS
    template from the mapping.

    This does NOT add anything to the template library — it only PROPOSES a candidate.
    The candidate is self-validated here (does the extracted template, reapplied, still
    produce the exact given product?) but that only proves it's consistent with the ONE
    example it came from — it has NOT been checked for generalizing to other molecules
    of the same reaction class, which is exactly the kind of check that caught real
    problems with hand-written templates this session (the alkene-exploit, the
    gem-dibromide, the Friedel-Crafts/amide false-positive). Before treating a candidate
    as trustworthy, re-test it against at least one OTHER real example of the same
    reaction class via auto_validate_reaction, the same way every template in
    _COMMON_SMARTS was verified.

    Args:
        reactant_smiles (list[str]): The reactants of one real, correct example reaction.
        product_smiles (str): The product of that same example reaction.

    Returns:
        dict: {
            "forward_smarts": str | None,  # in the same "reactants>>product" convention
                # as _COMMON_SMARTS — None if extraction failed
            "self_consistent": bool,  # True if reapplying the extracted template to the
                # given reactants reproduces the given product
            "error": str | None,
        }
    """
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

    # rdchiral extracts product>>reactants (retro); _COMMON_SMARTS stores the forward
    # direction (reactants>>product), so flip it before returning.
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


@agent.tool_plain
async def check_building_blocks_economics(smiles_csv: str) -> dict:
    """Real availability + pricing check for a list of building blocks (e.g. the leaves
    of a design_route result), via ChemSpace. A molecule being structurally "terminal" in
    a search isn't useful if nobody actually sells it, or if the only listed price is for
    a multi-week-lead-time specialty reagent — this surfaces that before you commit to a
    route. Never invents a price: a vendor offer with no listed price is skipped, not
    treated as free; a failed lookup is reported as unconfirmed, never silently treated
    as either available or unavailable.

    Args:
        smiles_csv (str): Comma-separated building block SMILES to check.

    Returns:
        dict: {"results": [{"smiles": str, "available": bool, "vendor_count": int,
            "cheapest": {"price_usd": float, "pack_size": str, "vendor": str,
            "lead_time_days": int, "purity": float} | None, "error": str | None}, ...]}
    """
    smiles_list = [s.strip() for s in smiles_csv.split(",") if s.strip()]
    mgr = ChemspaceTokenManager()
    deps = ChemspaceDeps(mgr=mgr)
    results = [await check_building_block_economics(deps, s) for s in smiles_list]
    return {"results": results}


# Typical reagents/conditions for a subset of _COMMON_SMARTS — informational only, NOT
# verified the way the templates themselves are (no mass-balance/mechanism check makes
# sense for a conditions string). These are well-established, textbook conditions for
# well-known named reactions; treat as a helpful default to look up further, not as a
# guarantee that these exact conditions will work for a specific substrate.
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


@agent.tool_plain
def reaction_conditions(matched_smarts: str) -> dict:
    """Looks up typical real-world reagents/conditions for a template that design_route
    or auto_validate_reaction matched (the "matched_smarts" field in their output) — our
    template library models WHICH atoms rearrange, not HOW (catalyst, solvent,
    temperature). This is informational only: well-established textbook conditions for
    well-known named reactions, not verified per-substrate the way the templates
    themselves are mass-balance/mechanism checked. Don't treat a missing entry as "this
    reaction has no real conditions" — it just means this specific template isn't yet
    annotated.

    Args:
        matched_smarts (str): The exact SMARTS string from a tool's "matched_smarts" field.

    Returns:
        dict: {"matched_smarts": str, "conditions": str | None}
    """
    return {
        "matched_smarts": matched_smarts,
        "conditions": _REACTION_CONDITIONS.get(matched_smarts),
    }


@agent.tool_plain
def compare_search_strategies(
    target_smiles: str, known_smiles: list[str], max_depth: int = 4,
) -> dict:
    """Runs BOTH design_route (exhaustive DFS) and design_route_mcts (heuristic-guided
    MCTS) on the same target/building blocks and reports both results side by side. The
    two engines can disagree — MCTS spends its budget where the heuristic looks
    promising rather than everywhere, so it can find a different (still independently
    validated) route, or miss one DFS would reach. Use this when you want to know
    whether they agree before trusting either one's specific route, rather than running
    just one and assuming it found the best (or only) answer.

    Args:
        target_smiles (str): The target molecule to decompose.
        known_smiles (list[str]): SMILES already available as building blocks.
        max_depth (int): Maximum number of reaction steps to search.

    Returns:
        dict: {
            "dfs_result": dict,   # design_route's full return value
            "mcts_result": dict,  # design_route_mcts's full return value
            "agree": bool,        # same route_found AND same step count
            "summary": str,
        }
    """
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


@agent.tool_plain
def design_route(
    target_smiles: str,
    known_smiles: list[str],
    max_depth: int = 4,
    min_block_similarity: float = 0.6,
) -> dict:
    """Autonomously searches for a complete, fully-validated multi-step retrosynthetic
    route from target_smiles down to the given known/building-block SMILES, and
    automatically resolves any building-block mismatch it finds along the way.

    Strategy: recursive backward search. At each node, every reverse template in the
    common SMARTS library is applied to the current molecule; each candidate set of
    precursors is immediately confirmed via auto_validate_reaction (the same check used
    everywhere else), so an invalid decomposition is never used. The search recurses on
    any precursor not already in known_smiles, up to max_depth steps deep, and returns
    the SHORTEST fully-validated route found. This is the only tool that can construct a
    brand-new multi-step sequence — suggest_reactant_fix/correct_route only patch a
    single already-given step.

    If a node can't be reached by any real synthesis step but is structurally similar
    (Tanimoto >= min_block_similarity) to one of known_smiles, the first pass flags that
    as a SUBSTITUTION rather than failing outright. design_route then automatically
    re-runs the search with the EXACT molecule the search determined it actually needs
    (not just "something similar") added as an accepted building block, and repeats this
    up to a few rounds until the route is fully resolved with no remaining substitutions,
    or no further progress is possible. The final result always lists which building
    blocks beyond the originally given ones were required, in "recommended_building_blocks"
    — never silently presented as if the originally given blocks alone were sufficient.

    Args:
        target_smiles (str): The target molecule to decompose.
        known_smiles (list[str]): SMILES already available as building blocks (the
            search stops at any molecule matching one of these).
        max_depth (int): Maximum number of reaction steps to search before giving up.
        min_block_similarity (float): Minimum Tanimoto similarity (0-1) to a known block
            required to accept a building-block substitution on the first pass. Higher =
            more conservative (e.g. 0.6 only matches blocks differing by a small edit).

    Returns:
        dict: {
            "route_found": bool,
            "steps": [
                {
                    "reaction_number": int,
                    "reactants": list[str],
                    "product": str,
                    "matched_smarts": str,
                },
                ...
            ],  # in forward (precursor-first) order; empty if route_found is False
            "recommended_building_blocks": list[str],  # exact molecules beyond the
                # originally given known_smiles that this route actually requires —
                # empty if the original building blocks were already sufficient
            "building_block_swaps": [...],  # any remaining unresolved substitution(s)
                # after all auto-resolve rounds — present only if convergence stalled
            "message": str,
        }
    """
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
            break  # no progress possible — same swap(s) would recur
        accepted_blocks.extend(new_adds)
        recommended_additions.extend(new_adds)

    if route_steps is None:
        # The given building blocks could not reach the target at all — not even via a
        # similarity-based substitution. As a last resort, decompose the target with NO
        # building-block constraint, using ONLY the discovery-safe template subset (never
        # the full library — see _DISCOVERY_SAFE_SMARTS for why), to see if the target
        # has SOME structurally plausible disconnection regardless of what was given.
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
            f"via reverse SMARTS: {best_fix['smarts']} — pending an independent "
            f"re-check before this can be treated as resolved."
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
