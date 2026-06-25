import json
from itertools import permutations

from pydantic_ai import Agent
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdChemReactions

from .validation import _COMMON_SMARTS, _DISCOVERY_SAFE_SMARTS, auto_validate_reaction

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
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _similarity(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    return DataStructs.TanimotoSimilarity(_fingerprint(mol_a), _fingerprint(mol_b))


def _canon(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


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
            try:
                precursor_smiles = [
                    _canon(Chem.MolToSmiles(m, ignoreAtomMapNumbers=True)) for m in outputs
                ]
            except Exception:
                continue
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
