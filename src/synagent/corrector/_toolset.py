import asyncio
import functools
import re
from itertools import permutations
from pathlib import Path

import json

from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdChemReactions

from synagent.validation._toolset import (
    ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    _match_product,
    _morgan_generator,
)

try:
    from rdchiral.template_extractor import extract_from_reaction as _rdchiral_extract
except ImportError:
    _rdchiral_extract = None

try:
    from indigo import Indigo
except ImportError:
    Indigo = None

RDLogger.DisableLog("rdApp.*")


@functools.lru_cache(maxsize=1)
def _rxn1_templates() -> tuple[str, ...]:
    """SynLlama's own RXN 1 template library -- 91 SMARTS templates, the set
    Table 1's ChEMBL columns were trained/benchmarked against (paper text,
    "select 1000 SMILES strings ... using SynLlama models trained on RXN 1";
    RXN 2's 115 templates cover a different set of models, reported only in
    the paper's Supplementary Table S1).

    fix_template and fix_via_analogue_building_block search this library, not
    the broader corrector-curated _COMMON_SMARTS (167 templates): a template
    found in _COMMON_SMARTS is *not* memorized in SynLlama's sense -- checked
    empirically, 0 of 43 corrector fixes sourced from _COMMON_SMARTS on the
    ChEMBL frozen-subset benchmark happened to also be in RXN1. Searching
    RXN1 directly means anything found here provably doesn't cost Template
    Memorization, at the cost of a smaller, less general library (measured:
    ~15% recovery rate on a sample of routes that needed the broader search,
    vs ~22% against _COMMON_SMARTS -- see
    docs/chembl-benchmark/table1_with_synagent.md footnote 6 for context).
    """
    path = Path(__file__).resolve().parents[3] / "data" / "91_rxn_templates.sma"
    return tuple(line.strip() for line in path.read_text().splitlines() if line.strip())


def _try_parse_smarts(smarts: str) -> rdChemReactions.ChemicalReaction | None:
    """Try to parse a reaction SMARTS, returning None on failure."""
    try:
        rxn = rdChemReactions.ReactionFromSmarts(smarts)
        if rxn is not None:
            rxn.Initialize()
            return rxn
    except Exception:
        pass
    return None


def _extract_template_from_reaction_sync(
    reactant_smiles: list[str], product_smiles: str
) -> dict:
    """Synchronous body of extract_template_from_reaction -- runs in a worker
    thread (see the async wrapper) because Indigo automap and rdchiral's
    template extractor are blocking C-extension/pure-Python calls with no
    async equivalent."""
    if Indigo is None or _rdchiral_extract is None:
        return {
            "fixed": False,
            "smarts": None,
            "self_consistent": False,
            "message": "Indigo and/or rdchiral not installed. Try search_building_blocks to find an alternative building block.",
        }

    try:
        indigo = Indigo()
        unmapped = f"{'.'.join(reactant_smiles)}>>{product_smiles}"
        rxn_obj = indigo.loadReaction(unmapped)
        rxn_obj.automap("discard")
        mapped = rxn_obj.smiles()
        mapped_reactants, mapped_products = mapped.split(">>")
    except Exception as e:
        return {
            "fixed": False,
            "smarts": None,
            "self_consistent": False,
            "message": f"Atom mapping failed: {e}. Try search_building_blocks to find an alternative building block.",
        }

    try:
        extracted = _rdchiral_extract({
            "reactants": mapped_reactants,
            "products": mapped_products,
            "_id": "candidate",
        })
        retro = extracted.get("reaction_smarts") if isinstance(extracted, dict) else None
        if not retro or ">>" not in retro:
            return {
                "fixed": False,
                "smarts": None,
                "self_consistent": False,
                "message": "Template extraction found no reacting atoms. Try search_building_blocks to find an alternative building block.",
            }
    except Exception as e:
        return {
            "fixed": False,
            "smarts": None,
            "self_consistent": False,
            "message": f"Template extraction failed: {e}. Try search_building_blocks to find an alternative building block.",
        }

    retro_lhs, retro_rhs = retro.split(">>")
    forward = f"{retro_rhs}>>{retro_lhs}"
    rxn = _try_parse_smarts(forward)
    if rxn is None:
        return {
            "fixed": False,
            "smarts": forward,
            "self_consistent": False,
            "message": "Extracted SMARTS could not be parsed. Try search_building_blocks to find an alternative building block.",
        }

    # Self-consistency check
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    canon_product = Chem.CanonSmiles(product_smiles)
    self_consistent = False
    for perm in permutations(reactant_mols):
        for outputs in rxn.RunReactants(perm):
            for mol in outputs:
                try:
                    Chem.SanitizeMol(mol)
                    if Chem.MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True) == canon_product:
                        self_consistent = True
                except Exception:
                    continue

    return {
        "fixed": True,
        "smarts": forward,
        "self_consistent": self_consistent,
        "message": (
            "Fresh SMARTS derived via Indigo + rdchiral."
            + (" Self-consistent: produces the expected product." if self_consistent
               else " Not self-consistent — template does not reproduce the expected product. Try fix_template.")
        ),
    }


def _fix_template_sync(reactant_smiles: list[str], product_smiles: str) -> dict:
    """Synchronous body of fix_template -- runs in a worker thread (see the
    async wrapper): tries every RXN1 template x every reactant permutation,
    up to ~91 x n! RDKit calls in the worst case."""
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return {"found": False, "template": None, "message": f"Invalid product SMILES: {product_smiles}"}

    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactant_mols):
        return {"found": False, "template": None, "message": "One or more reactant SMILES could not be parsed."}

    canon_product = Chem.CanonSmiles(product_smiles)

    for smarts in _rxn1_templates():
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smarts)
            if rxn is None:
                continue
        except Exception:
            continue
        try:
            for perm in permutations(reactant_mols):
                for outputs in rxn.RunReactants(perm):
                    for mol in outputs:
                        try:
                            Chem.SanitizeMol(mol)
                            if Chem.MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True) == canon_product:
                                return {
                                    "found": True,
                                    "template": smarts,
                                    "reactants": reactant_smiles,
                                    "product": canon_product,
                                    "message": "Template found and validated — reactants produce the expected product.",
                                }
                        except Exception:
                            continue
        except Exception:
            continue

    return {
        "found": False,
        "template": None,
        "message": "No template in the library produces the expected product from these reactants.",
    }


def _analogue_candidates(smiles: str, threshold: float, max_candidates: int) -> list[str]:
    """Similar building blocks from the local Enamine-derived database, cheapest
    first. Constructs its own FPSim2Engine per call, matching
    search_step_building_blocks's existing pattern."""
    from pathlib import Path

    from FPSim2.FPSim2 import FPSim2Engine

    moldb = Path(__file__).parent.parent / "analogues" / "data" / "building_blocks.h5"
    if not moldb.exists():
        return []
    engine = FPSim2Engine(str(moldb), in_memory_fps=True)
    hits = engine.similarity(smiles, threshold, metric="cosine", n_workers=4, mol_format="smiles")
    return [s for s in engine.get_strings(hits)[:max_candidates] if s != smiles]


def _fix_via_analogue_sync(
    reactant_smiles: list[str],
    product_smiles: str,
    similarity_threshold: float = 0.6,
    max_candidates_per_reactant: int = 5,
    analog_product_threshold: float | None = ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
) -> dict:
    """Try swapping one reactant at a time for a commercially-available analogue,
    then search SynLlama's own RXN1 template library again with the
    substitution in place. Unlike extract_template_from_reaction, every
    template tried here is one of the 91 SynLlama was actually trained on --
    nothing invented, nothing from a library we curated ourselves -- so a fix
    found this way costs nothing on Template Memorization. The tradeoff: since
    the reactant changed, the achieved product is generally an *analogue* of
    the original target, not the exact molecule, so matching is analog-aware
    by default (accept a Morgan/Tanimoto > threshold match, not only exact).
    """
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return {"found": False, "message": f"Invalid product SMILES: {product_smiles}"}

    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactant_mols):
        return {"found": False, "message": "One or more reactant SMILES could not be parsed."}

    for idx in range(len(reactant_smiles)):
        candidates = _analogue_candidates(
            reactant_smiles[idx], similarity_threshold, max_candidates_per_reactant
        )
        for candidate in candidates:
            candidate_mol = Chem.MolFromSmiles(candidate)
            if candidate_mol is None:
                continue
            trial_reactants = list(reactant_mols)
            trial_reactants[idx] = candidate_mol

            for smarts in _rxn1_templates():
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(smarts)
                    if rxn is None:
                        continue
                except Exception:
                    continue

                actual_products: list[str] = []
                try:
                    for perm in permutations(trial_reactants):
                        for outputs in rxn.RunReactants(perm):
                            for mol in outputs:
                                try:
                                    Chem.SanitizeMol(mol)
                                    smi = Chem.MolToSmiles(
                                        mol, canonical=True, ignoreAtomMapNumbers=True
                                    )
                                    if smi not in actual_products:
                                        actual_products.append(smi)
                                except Exception:
                                    continue
                except Exception:
                    continue

                if not actual_products:
                    continue

                matched, match_type, matched_product, similarity = _match_product(
                    product_smiles, actual_products, analog_product_threshold
                )
                if matched:
                    new_reactants = list(reactant_smiles)
                    new_reactants[idx] = candidate
                    return {
                        "found": True,
                        "new_reactants": new_reactants,
                        "template": smarts,
                        "substituted_index": idx,
                        "original_reactant": reactant_smiles[idx],
                        "analogue_reactant": candidate,
                        "match_type": match_type,
                        "matched_product": matched_product,
                        "product_similarity": similarity,
                        "message": (
                            f"Swapped reactant {idx} for an analogue ({candidate}); "
                            f"known template produces a {match_type} match "
                            f"(similarity={similarity:.4f})."
                        ),
                    }

    return {
        "found": False,
        "message": (
            "No analogue substitution against the known template library produces "
            "the expected product or an approved analog. Try extract_template_from_reaction."
        ),
    }


def _already_analog_passing(reactant_smiles: list[str], template: str, product_smiles: str) -> bool:
    """True if the UNTOUCHED original reactants+template already produce an
    analog-acceptable match to the declared product -- i.e. the same
    RunReactants + _match_product check _validate_route_dict itself uses
    (validation/_toolset.py), just for one reaction in isolation.

    A "wrong_product" failure only means the match failed under STRICT
    (exact) comparison; the untouched combination can still already be
    analog-passing. fix_via_product_analogue_retro must check this before it
    fires -- otherwise it can overwrite an already-analog-passing reaction
    with a *different* candidate product found via a different similarity
    metric (FPSim2 cosine on the building-block database vs. Morgan/Tanimoto
    here), net negative on Good Products (analog) even though every other
    tool in the chain is a strict superset. Confirmed empirically: adding
    this tool without the guard raised Matched Reactants +8.52 and left
    Good Products (strict) unchanged, but *dropped* Good Products (analog)
    -0.78 on the full frozen set.
    """
    rxn = _try_parse_smarts(template)
    if rxn is None:
        return False
    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactant_mols):
        return False
    actual_products: list[str] = []
    try:
        for perm in permutations(reactant_mols):
            for outputs in rxn.RunReactants(perm):
                for mol in outputs:
                    try:
                        Chem.SanitizeMol(mol)
                        smi = Chem.MolToSmiles(mol, canonical=True, ignoreAtomMapNumbers=True)
                        if smi not in actual_products:
                            actual_products.append(smi)
                    except Exception:
                        continue
    except Exception:
        return False
    if not actual_products:
        return False
    found, *_ = _match_product(product_smiles, actual_products, ANALOG_PRODUCT_SIMILARITY_THRESHOLD)
    return found


def _retro_disconnection_sync(
    reactant_smiles: list[str],
    product_smiles: str,
    template: str,
    template_search=_fix_template_sync,
) -> dict:
    """Reverse `template`, apply it to the product to get candidate precursor
    fragment sets, then try `template_search` (same signature as
    _fix_template_sync: (reactants, product) -> dict) on each set.

    Extracted as its own function -- rather than left inline inside
    fix_step -- so fix_step's retro_disconnection method and any offline
    ablation/analysis script call the exact same logic instead of two
    implementations that can silently drift apart (the way apply_fixes's
    building_blocks reconstruction once did against the reactions it was
    supposed to match).
    """
    retro_smarts = ">>".join(template.split(">>")[::-1])
    try:
        product_mol = Chem.MolFromSmiles(product_smiles)
        retro_rxn = rdChemReactions.ReactionFromSmarts(retro_smarts)
        retro_rxn.Initialize()
    except Exception:
        return {"found": False, "new_reactants": None, "template": None,
                "message": "Could not reverse the original template for retro-disconnection."}

    seen: set[tuple] = set()
    try:
        for outputs in retro_rxn.RunReactants((product_mol,)):
            frags: list[str] = []
            ok = True
            for m in outputs:
                try:
                    Chem.SanitizeMol(m)
                    frags.append(Chem.MolToSmiles(m, canonical=True))
                except Exception:
                    ok = False
                    break
            if not ok or not frags:
                continue
            key = tuple(sorted(frags))
            if key in seen:
                continue
            seen.add(key)
            tr = template_search(frags, product_smiles)
            if tr.get("found"):
                return {
                    "found": True,
                    "new_reactants": frags,
                    "template": tr["template"],
                    "message": (f"Original reactants could not produce the product. "
                                f"Retro disconnection found alternative precursors: {frags}"),
                }
    except Exception:
        pass

    return {"found": False, "new_reactants": None, "template": None,
            "message": "Retro disconnection found no working alternative precursor set."}


def _retro_disconnection_all_templates_sync(
    product_smiles: str,
    template_search=_fix_template_sync,
    max_fragment_sets: int = 200,
) -> dict:
    """Like _retro_disconnection_sync, but reverses every RXN1 template
    against the product instead of only the one already attached to the
    failing step.

    _retro_disconnection_sync can only ever recover from bad REACTANTS: it
    keeps the step's original template and asks whether reversing that same
    template yields precursors template_search accepts. If the template
    itself was the wrong one for this product -- the common case in
    practice, see the diagnostic note below -- reversing it can never surface
    the right precursors, no matter how good template_search is. This
    instead reverses every one of the 91 templates in turn and re-searches
    the library forward against each distinct fragment set, so a wrong
    original template selection no longer blocks retrosynthesis. Still
    nothing outside RXN1, so a fix found here costs nothing on Template
    Memorization -- a wider search of the same library, not a different one.

    Diagnostic note (docs/chembl-benchmark/comparison-2026-08-27/frozen-repair-cluster-sample.csv,
    a 25-route Gemma cluster sample instrumented to log fix_step's internal
    failure_mode/method): 26/36 failed steps were no_products, and
    _retro_disconnection_sync (single template) fixed 0 of them. A standalone
    check of this function against 21 real no_products cases from the same
    dataset recovered 5/21 (23.8%).
    """
    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return {"found": False, "new_reactants": None, "template": None,
                "message": f"Invalid product SMILES: {product_smiles}"}
    canon_product = Chem.CanonSmiles(product_smiles)

    seen: set[tuple] = set()
    tried = 0
    for smarts in _rxn1_templates():
        retro_smarts = ">>".join(smarts.split(">>")[::-1])
        retro_rxn = _try_parse_smarts(retro_smarts)
        if retro_rxn is None:
            continue
        try:
            outputs_list = retro_rxn.RunReactants((product_mol,))
        except Exception:
            continue
        for outputs in outputs_list:
            frags: list[str] = []
            ok = True
            for m in outputs:
                try:
                    Chem.SanitizeMol(m)
                    frags.append(Chem.MolToSmiles(m, canonical=True))
                except Exception:
                    ok = False
                    break
            if not ok or not frags:
                continue
            key = tuple(sorted(frags))
            if key in seen:
                continue
            seen.add(key)
            tried += 1
            if tried > max_fragment_sets:
                return {"found": False, "new_reactants": None, "template": None,
                        "fragment_sets_tried": tried - 1,
                        "message": (f"Exhausted {max_fragment_sets} retro-derived fragment "
                                    "sets across the RXN1 library without a forward match.")}
            fwd = template_search(frags, canon_product)
            if fwd.get("found"):
                return {
                    "found": True,
                    "new_reactants": frags,
                    "template": fwd["template"],
                    "source_retro_template": smarts,
                    "fragment_sets_tried": tried,
                    "message": (f"Retro-decomposed the product with a different RXN1 template "
                                f"than the original step used, then found a forward match "
                                f"using {frags}."),
                }

    return {"found": False, "new_reactants": None, "template": None,
            "fragment_sets_tried": tried,
            "message": ("No RXN1 template's retro-decomposition of the product led to a "
                        "valid forward match, across all templates tried.")}


def _fix_via_product_analogue_retro_sync(
    product_smiles: str,
    reactant_smiles: list[str],
    template: str,
    similarity_threshold: float = 0.6,
    max_candidates: int = 10,
) -> dict:
    """Fixes a step by relaxing the TARGET, not the ingredients: finds
    commercially-available molecules similar to the expected product, then
    runs the full retro_disconnection_all_templates search aiming at each
    candidate in turn instead of the original product.

    Every other repair path in this file keeps the original target product
    fixed and only ever varies what feeds into it (fix_template,
    fix_via_analogue_building_block, retro_disconnection,
    retro_disconnection_all_templates). That means a product with NO RXN1
    retrosynthetic path -- see retro_disconnection_all_templates's diagnostic
    note, 215/341 frozen-subset routes as of this session -- is unfixable by
    any of them, no matter how the search is widened, because the goal itself
    is unreachable within the library. This instead treats the target as
    negotiable: a molecule one similarity-database hop away from X might have
    a perfectly good RXN1 route even when X itself does not.

    Necessarily analog-scored, not strict -- the molecule actually built is a
    close relative of the original target, not the target itself, by
    construction (same tradeoff as fix_via_analogue_building_block, just
    applied to the product side of the reaction instead of the reactant
    side).

    Requires reactant_smiles/template (the step's UNTOUCHED originals) purely
    to guard against a real regression found empirically: if the original
    reactants+template already produce an analog-acceptable match to the
    product -- a "wrong_product" failure only means it failed the STRICT
    check -- searching for a different candidate can replace an
    already-analog-passing reaction with a worse one, since candidates come
    from a different similarity metric (FPSim2 cosine on the building-block
    database) than the one scoring uses (Morgan/Tanimoto). See
    _already_analog_passing's docstring for the measured impact.
    """
    if _already_analog_passing(reactant_smiles, template, product_smiles):
        return {"found": False,
                "message": ("Original reactants+template already analog-pass the declared "
                             "product (just not strict) -- declined to search for a different, "
                             "possibly worse target.")}

    product_mol = Chem.MolFromSmiles(product_smiles)
    if product_mol is None:
        return {"found": False, "message": f"Invalid product SMILES: {product_smiles}"}
    target_fp = _morgan_generator.GetFingerprint(product_mol)

    candidates = _analogue_candidates(product_smiles, similarity_threshold, max_candidates)
    for candidate in candidates:
        candidate_mol = Chem.MolFromSmiles(candidate)
        if candidate_mol is None:
            continue
        rrt = _retro_disconnection_all_templates_sync(candidate)
        if not rrt.get("found"):
            continue
        similarity = DataStructs.TanimotoSimilarity(
            target_fp, _morgan_generator.GetFingerprint(candidate_mol)
        )
        return {
            "found": True,
            "new_reactants": rrt["new_reactants"],
            "template": rrt["template"],
            "matched_product": candidate,
            "product_similarity": similarity,
            "message": (f"Original product has no RXN1 retrosynthetic path. Found one to a "
                        f"similar, purchasable analog instead ({candidate}, "
                        f"similarity={similarity:.4f}): {rrt['message']}"),
        }

    return {
        "found": False,
        "message": ("No RXN1 route found to the original product or to any of its "
                     f"{len(candidates)} nearest purchasable analogs."),
    }


def _strip_tags(s: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", s).strip()
    # Strip surrounding quotes that Qwen sometimes adds
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _parse_smiles_list(val: "list[str] | str") -> list[str]:
    """Handle reactant_smiles passed as either a real list or a JSON-encoded string."""
    import json
    if isinstance(val, list):
        return val
    val = val.strip()
    try:
        parsed = json.loads(val)
        if isinstance(parsed, list):
            return [str(s) for s in parsed]
    except Exception:
        pass
    # Single SMILES string
    return [val.strip('"').strip("'")]


def _likely_truncated(smi: str) -> bool:
    """Heuristic: returns True if the SMILES looks cut off mid-structure."""
    if smi.count("[") != smi.count("]"):
        return True
    if smi.count("(") != smi.count(")"):
        return True
    tokens = re.findall(r"%\d{2}|\d", re.sub(r"\[.*?\]", "", smi))
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    if any(v % 2 != 0 for v in counts.values()):
        return True
    return False


def _try_complete(smi: str) -> str | None:
    """Append missing closing brackets/parens/ring tokens and try to parse."""
    square_depth = 0
    paren_depth = 0
    ring_opens: set[str] = set()
    i = 0
    while i < len(smi):
        c = smi[i]
        if c == "[":
            square_depth += 1
        elif c == "]":
            square_depth = max(0, square_depth - 1)
        elif c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth = max(0, paren_depth - 1)
        elif c == "%" and i + 2 < len(smi) and smi[i + 1 : i + 3].isdigit():
            tok = smi[i : i + 3]
            ring_opens.discard(tok) if tok in ring_opens else ring_opens.add(tok)
            i += 2
        elif c.isdigit() and square_depth == 0:
            ring_opens.discard(c) if c in ring_opens else ring_opens.add(c)
        i += 1

    suffix = "]" * square_depth + ")" * paren_depth + "".join(sorted(ring_opens))
    mol = Chem.MolFromSmiles(smi + suffix)
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def _attempt_bracket_completion(smi: str) -> str | None:
    """Try to fix a SMILES with missing/misplaced brackets and parentheses."""
    # Pass 1: append missing closers to end
    result = _try_complete(smi)
    if result is not None:
        return result

    # Pass 2: missing ] inside the string — find unclosed [, try inserting ] after each char
    bracket_start = None
    depth = 0
    for i, c in enumerate(smi):
        if c == "[":
            depth += 1
            if depth == 1:
                bracket_start = i
        elif c == "]":
            depth -= 1
            if depth == 0:
                bracket_start = None

    if bracket_start is not None:
        for insert_pos in range(bracket_start + 2, len(smi) + 1):
            candidate = smi[:insert_pos] + "]" + smi[insert_pos:]
            result = _try_complete(candidate)
            if result is not None:
                return result

    # Pass 3+4: try single-character mutations combined with bracket/paren completion
    # 3a: remove each ( or ) one at a time
    # 3b: move each ( one position to the right (catches n1(c=O) → n1c(=O) type errors)
    mutations: list[str] = []
    for i, c in enumerate(smi):
        if c in ("(", ")"):
            mutations.append(smi[:i] + smi[i + 1:])
        if c == "(" and i + 1 < len(smi):
            mutations.append(smi[:i] + smi[i + 1] + "(" + smi[i + 2:])

    for candidate in mutations:
        result = _try_complete(candidate)
        if result is not None:
            return result
        # also try pass-2 bracket insertion on the mutated candidate
        b_start = None
        d = 0
        for j, ch in enumerate(candidate):
            if ch == "[":
                d += 1
                if d == 1:
                    b_start = j
            elif ch == "]":
                d -= 1
                if d == 0:
                    b_start = None
        if b_start is not None:
            for insert_pos in range(b_start + 2, len(candidate) + 1):
                c2 = candidate[:insert_pos] + "]" + candidate[insert_pos:]
                result = _try_complete(c2)
                if result is not None:
                    return result

    return None


_REPORT_TOOLS = {"validate_route", "apply_fixes"}


def _get_last_validation_report(messages: list):
    """Scan conversation history for the most recent validate_route or apply_fixes result."""
    from synagent.validation._models import ValidationReport

    for msg in reversed(messages):
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name not in _REPORT_TOOLS:
                continue
            content = part.content
            try:
                if isinstance(content, ValidationReport):
                    return content
                if isinstance(content, dict):
                    return ValidationReport.model_validate(content)
                if isinstance(content, str):
                    return ValidationReport.model_validate_json(content)
                return ValidationReport.model_validate(json.loads(json.dumps(content, default=str)))
            except Exception:
                pass
    return None


def _get_fix_results_since_report(messages: list) -> tuple[dict, dict]:
    """Scan messages after the last validation report for fix_building_blocks and fix_step results.

    Returns:
        bb_fixes: {old_smiles -> canonical_smiles}
        step_fixes: {step_number -> fix_result_dict}
    """
    # Find index of the most recent report tool call
    report_pos = None
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name in _REPORT_TOOLS:
                    report_pos = i

    bb_fixes: dict = {}
    step_fixes: dict = {}
    start = (report_pos + 1) if report_pos is not None else 0

    for msg in messages[start:]:
        if not isinstance(msg, ModelRequest):
            continue
        for part in msg.parts:
            if not isinstance(part, ToolReturnPart):
                continue
            try:
                content = part.content
                if isinstance(content, str):
                    content = json.loads(content)
                elif not isinstance(content, dict):
                    content = json.loads(json.dumps(content, default=str))
            except Exception:
                continue

            if part.tool_name in ("fix_building_blocks", "fix_smiles"):
                # fix_building_blocks: {"results": {old_smi: {"canonical": ..., "valid": ...}}}
                # fix_smiles:          {old_smi: {"canonical": ..., "valid": ...}}
                results = content.get("results", content)
                if isinstance(results, dict):
                    for old_smi, res in results.items():
                        if isinstance(res, dict) and res.get("valid") and res.get("canonical"):
                            bb_fixes[old_smi] = res["canonical"]

            elif part.tool_name == "fix_step":
                step = content.get("step")
                if step is not None and content.get("fixed"):
                    step_fixes[int(step)] = content

    return bb_fixes, step_fixes


class CorrectorToolset(FunctionToolset[AgentDepsT]):
    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.fix_step, name="fix_step")
        self.add_function(self.fix_building_blocks, name="fix_building_blocks")
        self.add_function(self.apply_fixes, name="apply_fixes")
        self.add_function(self.search_step_building_blocks, name="search_step_building_blocks")
        self.add_function(self.fix_smarts, name="fix_smarts")
        self.add_function(self.extract_template_from_reaction, name="extract_template_from_reaction")
        self.add_function(self.fix_template, name="fix_template")
        self.add_function(self.fix_smiles, name="fix_smiles")
        self.add_function(self.fix_via_analogue_building_block, name="fix_via_analogue_building_block")
        self.add_function(self.retro_disconnection_all_templates, name="retro_disconnection_all_templates")
        self.add_function(self.fix_via_product_analogue_retro, name="fix_via_product_analogue_retro")

    async def fix_step(
        self, ctx: RunContext[AgentDepsT], step: int, method: str | None = None
    ) -> dict:
        """Fix a failed reaction step using the ValidationReport already in the conversation.

        Reads the most recent validate_route result automatically — no SMILES copying needed.

        By default (method omitted or "auto"), runs the full fix chain for the
        step's failure_mode automatically, trying each option in order until
        one works:
          - invalid_template  → fix_smarts → fix_template
          - no_products / wrong_product → fix_template → fix_via_analogue_building_block
                                          → retro_disconnection → retro_disconnection_all_templates
                                          → fix_via_product_analogue_retro
          - invalid_reactant_smiles / invalid_product_smiles → fix_smiles

        Pass method to try exactly one option yourself instead of the full
        chain — call fix_step again with the next method if the first didn't
        fix it. Useful for orchestrators that reason better making one
        decision at a time than trusting a hidden multi-step chain. Valid
        values depend on the step's current failure_mode:
          - invalid_template: "fix_smarts", "fix_template"
          - no_products / wrong_product: "fix_template", "fix_via_analogue_building_block",
            "retro_disconnection", "retro_disconnection_all_templates",
            "fix_via_product_analogue_retro"
          - invalid_reactant_smiles / invalid_product_smiles: "fix_smiles"

        extract_template_from_reaction is never tried by fix_step, auto or
        explicit — it invents a template outside SynLlama's trained library.
        Call it directly (not through fix_step) if you specifically want that.

        Args:
            step (int): Reaction number to fix (as shown in the ValidationReport).
            method (str | None): One specific sub-tool to try, or None/"auto"
                for the full automatic chain (see above).

        Returns:
            dict with fixed=True/False, method (whichever sub-tool actually
            fixed it, or was tried and didn't), new_template/new_reactants/
            smiles_fixes as applicable, and message.
        """
        report = _get_last_validation_report(ctx.messages)
        if report is None:
            return {"fixed": False, "step": step,
                    "message": "No ValidationReport found. Call validate_route first."}

        rxn = next((r for r in report.reactions if r.reaction_number == step), None)
        if rxn is None:
            available = [r.reaction_number for r in report.reactions]
            return {"fixed": False, "step": step,
                    "message": f"Step {step} not in ValidationReport. Available: {available}"}

        if rxn.status == "passed":
            return {"fixed": True, "step": step, "message": f"Step {step} already passes."}

        failure = rxn.failure_mode
        template = rxn.reaction_template
        reactants = rxn.reactant_smiles
        product = rxn.expected_product
        method = (method or "auto").strip().lower()

        # --- invalid_template: fix_smarts -> fix_template ---
        if failure == "invalid_template":
            valid = {"auto", "fix_smarts", "fix_template"}
            if method not in valid:
                return {"fixed": False, "step": step, "failure_mode": failure,
                        "message": f"method must be one of {sorted(valid)} for invalid_template."}

            if method in ("auto", "fix_smarts"):
                sr = await self.fix_smarts(template)
                if sr.get("fixed"):
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "fix_smarts", "new_template": sr["smarts"],
                            "message": sr["message"]}
                if method == "fix_smarts":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "fix_smarts", "message": sr.get("message")}

            tr = await self.fix_template(reactants, product)
            return {"fixed": tr.get("found", False), "step": step, "failure_mode": failure,
                    "method": "fix_template", "new_template": tr.get("template"),
                    "message": tr.get("message")}

        # --- no_products / wrong_product: fix_template -> analogue swap -> retro ---
        #
        # extract_template_from_reaction deliberately excluded from every path
        # here: it invents a template outside SynLlama's trained library,
        # which fixes more routes but can never count as "memorized" -- see
        # docs/chembl-benchmark/table1_with_synagent.md footnote 4/6 and the
        # corrector-bugs section for the measured Template Mem. cost (99.92% ->
        # 87.84% raw-vs-corrected on the version that also searched the broader
        # corrector-curated _COMMON_SMARTS library). fix_template and the
        # analogue swap only ever select from SynLlama's own RXN1 templates
        # (_rxn1_templates()), so nothing here can lower Template Mem.
        # The tool itself is left registered (still directly callable) in case
        # a future run wants it back -- only fix_step (auto or explicit
        # method) skips it.
        if failure in ("no_products", "wrong_product"):
            valid = {"auto", "fix_template", "fix_via_analogue_building_block",
                     "retro_disconnection", "retro_disconnection_all_templates",
                     "fix_via_product_analogue_retro"}
            if method not in valid:
                return {"fixed": False, "step": step, "failure_mode": failure,
                        "message": f"method must be one of {sorted(valid)} for {failure}."}

            if method in ("auto", "fix_template"):
                tr = await self.fix_template(reactants, product)
                if tr.get("found"):
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "fix_template", "new_template": tr["template"],
                            "message": tr.get("message")}
                if method == "fix_template":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "fix_template", "message": tr.get("message")}

            if method in ("auto", "fix_via_analogue_building_block"):
                # Swap one reactant for a commercially-available analogue and
                # search the known library again. Costs nothing on Template
                # Memorization -- the tradeoff is the achieved product is an
                # analogue of the original target, not the exact molecule, so
                # this only counts under analog-aware scoring.
                ar = await self.fix_via_analogue_building_block(reactants, product)
                if ar.get("found"):
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "fix_via_analogue_building_block",
                            "new_template": ar["template"], "new_reactants": ar["new_reactants"],
                            "message": ar.get("message")}
                if method == "fix_via_analogue_building_block":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "fix_via_analogue_building_block", "message": ar.get("message")}

            if method in ("auto", "retro_disconnection"):
                # Reverse the template, apply to product, try the known
                # library again on each suggested precursor set. Shared with
                # any offline ablation script via _retro_disconnection_sync
                # so this can't drift from what fix_step actually does.
                rr = await asyncio.to_thread(
                    _retro_disconnection_sync, reactants, product, template
                )
                if rr.get("found"):
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "retro_disconnection",
                            "new_reactants": rr["new_reactants"],
                            "new_template": rr["template"],
                            "message": rr["message"]}
                if method == "retro_disconnection":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "retro_disconnection", "message": rr["message"]}

            if method in ("auto", "retro_disconnection_all_templates"):
                # retro_disconnection above only recovers from bad REACTANTS --
                # it keeps the step's original template. This instead reverses
                # every RXN1 template against the product, so a wrong original
                # template choice no longer blocks retrosynthesis. Shared with
                # any offline ablation script via
                # _retro_disconnection_all_templates_sync.
                rrt = await self.retro_disconnection_all_templates(product)
                if rrt.get("found"):
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "retro_disconnection_all_templates",
                            "new_reactants": rrt["new_reactants"],
                            "new_template": rrt["template"],
                            "message": rrt["message"]}
                if method == "retro_disconnection_all_templates":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "retro_disconnection_all_templates", "message": rrt["message"]}

            if method in ("auto", "fix_via_product_analogue_retro"):
                # Every path above keeps the original target product fixed
                # and only varies what feeds into it. This instead relaxes
                # the target itself: finds a purchasable molecule similar to
                # the product and retro-searches for a route to THAT. Only
                # ever analog-scored -- the molecule built is a close
                # relative of the target, not the target itself -- and only
                # reached once every strict-preserving option has failed.
                par = await self.fix_via_product_analogue_retro(product, reactants, template)
                if par.get("found"):
                    # Deliberately no new_product/new_target key: apply_fixes
                    # must keep the ORIGINAL declared product so this route is
                    # scored analog-only, not strict -- see apply_fixes's own
                    # comment. matched_product/product_similarity are
                    # diagnostic only.
                    return {"fixed": True, "step": step, "failure_mode": failure,
                            "method": "fix_via_product_analogue_retro",
                            "new_reactants": par["new_reactants"],
                            "new_template": par["template"],
                            "matched_product": par["matched_product"],
                            "product_similarity": par.get("product_similarity"),
                            "message": par["message"]}
                if method == "fix_via_product_analogue_retro":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "fix_via_product_analogue_retro", "message": par["message"]}

            return {"fixed": False, "step": step, "failure_mode": failure,
                    "message": ("fix_template, fix_via_analogue_building_block, retro_disconnection, "
                                "retro_disconnection_all_templates, and fix_via_product_analogue_retro "
                                "all failed within the known template library. The route step may need "
                                "extract_template_from_reaction (not tried automatically -- would invent "
                                "a template outside the trained library) or manual redesign.")}

        # --- invalid SMILES ---
        if failure in ("invalid_reactant_smiles", "invalid_product_smiles"):
            valid = {"auto", "fix_smiles"}
            if method not in valid:
                return {"fixed": False, "step": step, "failure_mode": failure,
                        "message": f"method must be one of {sorted(valid)} for {failure}."}
            smiles_to_fix = reactants if failure == "invalid_reactant_smiles" else [product]
            sr = await self.fix_smiles(smiles_to_fix)
            fixed_any = any(v.get("valid") for v in sr.values())
            return {"fixed": fixed_any, "step": step, "failure_mode": failure,
                    "method": "fix_smiles", "smiles_fixes": sr,
                    "message": "SMILES corrected." if fixed_any else "Could not fix SMILES."}

        return {"fixed": False, "step": step,
                "message": f"Unhandled failure_mode: {failure}"}

    async def fix_building_blocks(self, ctx: RunContext[AgentDepsT]) -> dict:
        """Fix all invalid building blocks from the last ValidationReport.

        Reads the most recent validate_route result automatically — no SMILES copying needed.
        Runs fix_smiles on every building block where is_valid=False.

        Returns:
            dict with per-building-block results and a summary.
        """
        report = _get_last_validation_report(ctx.messages)
        if report is None:
            return {"message": "No ValidationReport found. Call validate_route first."}

        invalid = [bb for bb in report.building_blocks if not bb.is_valid]
        if not invalid:
            return {"fixed_count": 0, "total": 0,
                    "message": "All building blocks are valid — nothing to fix."}

        results = {}
        for bb in invalid:
            fix = await self.fix_smiles([bb.smiles])
            results[bb.smiles] = fix.get(bb.smiles, {"valid": False, "message": "No result"})

        fixed = sum(1 for r in results.values() if r.get("valid"))
        return {"fixed_count": fixed, "total": len(invalid), "results": results,
                "message": f"Fixed {fixed}/{len(invalid)} invalid building blocks."}

    async def apply_fixes(self, ctx: RunContext[AgentDepsT]) -> dict:
        """Apply all fix results from this session to the route and re-validate.

        Reads the last ValidationReport and all fix_building_blocks / fix_step results
        from conversation history automatically — no SMILES copying needed.

        Returns a new ValidationReport reflecting the corrected route. Subsequent
        fix_step / fix_building_blocks calls will read from this new report, so the
        fix loop always advances forward on the corrected path.
        """
        from synagent.validation._models import ValidationReport
        from synagent.validation._toolset import _validate_route_dict

        report = _get_last_validation_report(ctx.messages)
        if report is None:
            return ValidationReport(
                reactions=[], building_blocks=[], target_molecule="unknown",
                all_building_blocks_valid=False, all_reactions_passed=False,
                suggested_fixes=["No ValidationReport found. Call validate_route first."],
            )

        bb_fixes, step_fixes = _get_fix_results_since_report(ctx.messages)

        # --- Build corrected reactions list ---
        corrected_reactions = []
        for rxn in report.reactions:
            fix = step_fixes.get(rxn.reaction_number, {})

            # Template: use new_template if fix found one
            template = fix.get("new_template") or rxn.reaction_template

            # Reactants: prefer new_reactants, otherwise apply smiles_fixes then bb_fixes
            if fix.get("new_reactants"):
                reactants = fix["new_reactants"]
            else:
                smiles_fixes = fix.get("smiles_fixes", {})
                reactants = [
                    smiles_fixes.get(r, {}).get("canonical") or bb_fixes.get(r) or r
                    for r in rxn.reactant_smiles
                ]

            # Product: apply smiles_fixes if available, otherwise bb_fixes.
            # Deliberately NEVER overwritten by fix_via_product_analogue_retro's
            # matched_product -- the declared product must stay the ORIGINAL
            # target so _validate_route_dict compares actual output against
            # it and correctly falls into analog-only scoring, same
            # convention as fix_via_analogue_building_block. Setting it to
            # the matched analog would make the route self-consistent and
            # wrongly count as strict for a target we deliberately did not
            # reproduce.
            smiles_fixes = fix.get("smiles_fixes", {})
            product = (
                smiles_fixes.get(rxn.expected_product, {}).get("canonical")
                or bb_fixes.get(rxn.expected_product)
                or rxn.expected_product
            )

            corrected_reactions.append({
                "reaction_number": rxn.reaction_number,
                "reaction_template": template,
                "reactants": reactants,
                "product": product,
            })

        # --- Derive building blocks from the corrected reactions themselves ---
        #
        # Previously this patched the *original* building_blocks list through
        # bb_fixes independently of corrected_reactions above. bb_fixes and the
        # per-reaction smiles_fixes/new_reactants can each "fix" the same
        # underlying molecule to a slightly different SMILES (a typo'd atom, a
        # different canonicalization) -- and reactants introduced via
        # fix_step's retro-disconnection fallback (new_reactants) never made it
        # into building_blocks at all. Either way the two lists silently
        # drifted apart. A building block is, by definition, a reactant that
        # isn't another step's product -- so derive it from corrected_reactions
        # directly and there is nothing left to drift.
        seen_reactants: list[str] = []
        seen_reactants_set: set[str] = set()
        all_products: set[str] = set()
        for rxn in corrected_reactions:
            for r in rxn["reactants"]:
                if r and r not in seen_reactants_set:
                    seen_reactants_set.add(r)
                    seen_reactants.append(r)
            if rxn["product"]:
                all_products.add(rxn["product"])
        corrected_bbs = [r for r in seen_reactants if r not in all_products]

        corrected_route = {"reactions": corrected_reactions, "building_blocks": corrected_bbs}
        # Same combinatorial reactant-permutation cost as validate_route --
        # offload it (see extract_template_from_reaction for why an unwrapped
        # call here defeats every timeout upstream, including run_repair.py's).
        return await asyncio.to_thread(_validate_route_dict, corrected_route)

    async def search_step_building_blocks(
        self, ctx: RunContext[AgentDepsT], step: int, threshold: float = 0.6, max_results: int = 10
    ) -> dict:
        """Find commercially available alternative building blocks for a reaction step.

        Reads reactant SMILES for the given step from the last ValidationReport automatically
        — no SMILES copying needed. Searches the local building block database for similar
        molecules.

        Args:
            step (int): Reaction step number whose reactants to search alternatives for.
            threshold (float): Similarity threshold (0–1). Lower = more results.
            max_results (int): Maximum alternatives to return per reactant.

        Returns:
            dict mapping each reactant SMILES to a list of similar building blocks.
        """
        report = _get_last_validation_report(ctx.messages)
        if report is None:
            return {"message": "No ValidationReport found. Call validate_route first."}

        rxn = next((r for r in report.reactions if r.reaction_number == step), None)
        if rxn is None:
            available = [r.reaction_number for r in report.reactions]
            return {"message": f"Step {step} not found. Available steps: {available}"}

        try:
            from pathlib import Path
            from FPSim2.FPSim2 import FPSim2Engine
            moldb = Path(__file__).parent.parent / "analogues" / "data" / "building_blocks.h5"
            if not moldb.exists():
                return {"message": f"Building block database not found at {moldb}"}
            engine = FPSim2Engine(str(moldb), in_memory_fps=True)
        except ImportError:
            return {"message": "FPSim2 not installed — cannot search building blocks."}
        except Exception as e:
            return {"message": f"Failed to load building block database: {e}"}

        results = {}
        for smi in rxn.reactant_smiles:
            try:
                hits = engine.similarity(smi, threshold, metric="cosine",
                                         n_workers=1, mol_format="smiles")
                results[smi] = engine.get_strings(hits)[:max_results]
            except Exception as e:
                results[smi] = [f"Search failed: {e}"]

        return {"step": step, "reactants_searched": rxn.reactant_smiles,
                "alternatives": results,
                "message": f"Found alternatives for {len(results)} reactants in step {step}."}

    async def fix_smarts(
        self,
        reaction_smarts: str,
    ) -> dict:
        """Tries to repair a reaction SMARTS string that failed validate_reaction_smarts
        using syntactic fixes: XML tag stripping, encoding cleanup, atom primitive fixes.
        If this fails, call extract_template_from_reaction next.
        Use when failure_mode is 'invalid_template'.

        Args:
            reaction_smarts (str): The broken reaction SMARTS string.

        Returns:
            dict: {"fixed": bool, "smarts": str | None, "method": str, "message": str}
        """
        attempts = []

        # 1. Strip XML tags and surrounding quotes
        cleaned = _strip_tags(reaction_smarts)
        attempts.append(("tag_strip", cleaned))

        # 2. Fix escaped angle brackets
        unescaped = cleaned.replace("\\u003e", ">").replace("%3E", ">").replace("%3e", ">")
        if unescaped != cleaned:
            attempts.append(("unescape", unescaped))

        # 3. Uppercase N/C in product side → [#7]/[#6] (known RDKit retro-SMARTS issue)
        fixed_primitives = re.sub(r">>([^>]+)$", lambda m: ">>" + re.sub(
            r"\b([NC])\b", lambda a: "[#7]" if a.group(1) == "N" else "[#6]", m.group(1)
        ), unescaped)
        if fixed_primitives != unescaped:
            attempts.append(("primitive_fix", fixed_primitives))

        for method, candidate in attempts:
            if ">>" not in candidate:
                continue
            rxn = _try_parse_smarts(candidate)
            if rxn is not None:
                return {
                    "fixed": True,
                    "smarts": candidate,
                    "method": method,
                    "message": f"SMARTS repaired via {method}.",
                }

        return {
            "fixed": False,
            "smarts": None,
            "method": "none",
            "message": (
                "Syntactic fixes could not repair this SMARTS. "
                "Call fix_template with reactant_smiles and product_smiles to find a library template instead."
            ),
        }

    async def extract_template_from_reaction(
        self,
        reactant_smiles: list[str],
        product_smiles: str,
    ) -> dict:
        """Derives a reaction SMARTS template for a new building block + product combination
        using Indigo atom-mapping and rdchiral template extraction. Call this after
        search_building_blocks has found an alternative building block, to derive a template
        that works with the new reactants and expected product.

        Args:
            reactant_smiles (list[str]): Reactant SMILES including the new building block.
            product_smiles (str): Expected product SMILES.

        Returns:
            dict: {"fixed": bool, "smarts": str | None, "self_consistent": bool, "message": str}
        """
        reactant_smiles = _parse_smiles_list(reactant_smiles)
        product_smiles = _strip_tags(str(product_smiles))

        # Indigo atom-mapping and rdchiral's template extractor are synchronous,
        # CPU-bound, and on some inputs pathologically slow (or effectively
        # hung) -- observed hangs past an hour on a single call. Run them off
        # the event loop so a slow/stuck extraction can't block every other
        # coroutine (including any asyncio.wait_for timeout a caller set on
        # this call -- that timeout can only fire when the loop gets control
        # back, which never happens if this runs inline).
        return await asyncio.to_thread(
            _extract_template_from_reaction_sync, reactant_smiles, product_smiles
        )

    async def fix_template(
        self,
        reactant_smiles: list[str],
        product_smiles: str,
    ) -> dict:
        """Fixes a failed reaction step by searching SynLlama's own RXN1 template
        library (91 templates) for a valid SMARTS that produces the expected
        product from the given reactants. Use when validate_products fails due
        to an invalid or missing reaction template. Every template tried here
        is one SynLlama was actually trained on, so a fix found this way never
        costs anything on Template Memorization scoring.

        Args:
            reactant_smiles (list[str]): Reactant SMILES for the failed step.
            product_smiles (str): Expected product SMILES for the failed step.

        Returns:
            dict: {"found": bool, "template": str | None, "reactants": list,
                   "product": str, "message": str}
        """
        reactant_smiles = _parse_smiles_list(reactant_smiles)
        product_smiles = _strip_tags(str(product_smiles))

        # Tries every one of the 91 RXN1 templates x every reactant
        # permutation x RunReactants -- worst case ~500 RDKit calls,
        # fully synchronous. Offload it (see extract_template_from_reaction
        # for why a blocked event loop defeats any asyncio timeout).
        return await asyncio.to_thread(
            _fix_template_sync, reactant_smiles, product_smiles
        )

    async def fix_via_analogue_building_block(
        self,
        reactant_smiles: list[str],
        product_smiles: str,
    ) -> dict:
        """Fixes a failed reaction step by swapping one reactant for a similar,
        commercially-available building block, then searching SynLlama's own
        RXN1 template library again with that substitution. Use after
        fix_template fails: this only ever uses templates SynLlama was
        actually trained on, so it costs nothing on Template Memorization --
        extract_template_from_reaction invents new SMARTS instead, which
        fixes more routes but can't be "memorized" by definition and is not
        part of the automatic fix_step chain for that reason.

        The tradeoff: swapping a reactant generally changes the exact product, so
        a fix found here matches the original target as an *analogue*
        (Morgan/Tanimoto similarity), not always an exact match.

        Args:
            reactant_smiles (list[str]): Reactant SMILES for the failed step.
            product_smiles (str): Expected product SMILES for the failed step.

        Returns:
            dict: {"found": bool, "new_reactants": list | None, "template": str | None,
                   "match_type": "exact" | "analog" | None, "product_similarity": float | None,
                   "message": str}
        """
        reactant_smiles = _parse_smiles_list(reactant_smiles)
        product_smiles = _strip_tags(str(product_smiles))

        # Same combinatorial cost profile as fix_template, times up to 5
        # candidate analogues per reactant position -- offload it for the same
        # reason (see extract_template_from_reaction).
        return await asyncio.to_thread(
            _fix_via_analogue_sync, reactant_smiles, product_smiles
        )

    async def retro_disconnection_all_templates(self, product_smiles: str) -> dict:
        """Fixes a failed reaction step by retrosynthetic search across the whole
        RXN1 library: reverses every one of the 91 templates against the
        product, then re-searches the same 91-template library forward
        against each distinct candidate precursor set. Use after fix_template,
        fix_via_analogue_building_block, and retro_disconnection (fix_step's
        single-template retro fallback) all fail -- retro_disconnection can
        only recover from bad reactants because it keeps the step's original
        template; this instead asks whether a DIFFERENT RXN1 template was the
        right one for this product all along. Every template tried here is
        one of the 91 SynLlama was trained on, so a fix found this way never
        costs anything on Template Memorization.

        Args:
            product_smiles (str): Expected product SMILES for the failed step.

        Returns:
            dict: {"found": bool, "new_reactants": list | None, "template": str | None,
                   "source_retro_template": str | None, "fragment_sets_tried": int,
                   "message": str}
        """
        product_smiles = _strip_tags(str(product_smiles))

        # Worst case ~91 retro applications x up to 200 forward fix_template
        # searches (each itself ~91 templates x reactant permutations) --
        # fully synchronous RDKit work. Offload it for the same reason as
        # fix_template (see extract_template_from_reaction).
        return await asyncio.to_thread(
            _retro_disconnection_all_templates_sync, product_smiles
        )

    async def fix_via_product_analogue_retro(
        self, product_smiles: str, reactant_smiles: list[str], template: str
    ) -> dict:
        """Fixes a failed reaction step by relaxing the TARGET product itself,
        not the ingredients: finds commercially-available molecules similar
        to the expected product, then runs the full retro_disconnection_all_templates
        search aiming at each candidate instead of the original. Use after
        fix_template, fix_via_analogue_building_block, retro_disconnection,
        and retro_disconnection_all_templates all fail -- those all keep the
        original target product fixed and only vary the ingredients feeding
        into it, so a product with genuinely no RXN1 retrosynthetic path is
        unfixable by any of them, no matter how the ingredient search is
        widened. This instead asks whether a close, purchasable relative of
        the target has a route, when the target itself does not.

        Declines to run at all (found=False) when the step's UNTOUCHED
        original reactants+template already produce an analog-acceptable
        match to the product -- a "wrong_product" failure only means it
        failed the STRICT check, and searching for a different candidate
        could replace an already-good analog match with a worse one.

        Every template considered is still one of the 91 SynLlama was trained
        on -- costs nothing on Template Memorization -- but the molecule
        actually built is a similar analog of the original target, not the
        target itself, so a fix found here only ever counts under
        analog-aware scoring, never strict (same tradeoff as
        fix_via_analogue_building_block, applied to the product side instead
        of the reactant side).

        Args:
            product_smiles (str): Expected product SMILES for the failed step.
            reactant_smiles (list[str]): The step's UNTOUCHED original reactants
                (used only for the already-analog-passing guard, never varied).
            template (str): The step's UNTOUCHED original template (same).

        Returns:
            dict: {"found": bool, "new_reactants": list | None, "template": str | None,
                   "matched_product": str | None, "product_similarity": float | None,
                   "message": str}
        """
        product_smiles = _strip_tags(str(product_smiles))
        reactant_smiles = _parse_smiles_list(reactant_smiles)
        template = _strip_tags(str(template))

        # Up to 10 candidate analog products, each its own full
        # retro_disconnection_all_templates search (~91 retro applications x
        # up to 200 forward searches) -- the most expensive tool in the
        # chain, only reached once everything cheaper has failed. Offload it
        # for the same reason as fix_template.
        return await asyncio.to_thread(
            _fix_via_product_analogue_retro_sync, product_smiles, reactant_smiles, template
        )

    async def fix_smiles(self, smiles: list[str]) -> dict[str, dict]:
        """Tries to parse and canonicalize SMILES strings. For invalid ones, attempts
        common fixes: removing atom map numbers, partial sanitization. Detects likely
        truncation. Use when SMILES in the route fail validate_smiles.

        Args:
            smiles (list[str]): SMILES strings to fix.

        Returns:
            dict mapping each input to
            {"canonical": str | None, "valid": bool, "truncated": bool, "message": str}
        """
        result = {}
        for smi in smiles:
            clean = smi.strip()

            mol = Chem.MolFromSmiles(clean)
            if mol is not None:
                result[smi] = {
                    "canonical": Chem.MolToSmiles(mol, canonical=True),
                    "valid": True,
                    "truncated": False,
                    "message": "Valid SMILES.",
                }
                continue

            # Try removing atom map numbers
            no_maps = re.sub(r":\d+", "", clean)
            mol = Chem.MolFromSmiles(no_maps)
            if mol is not None:
                result[smi] = {
                    "canonical": Chem.MolToSmiles(mol, canonical=True),
                    "valid": True,
                    "truncated": False,
                    "message": "Fixed by removing atom map numbers.",
                }
                continue

            # Try partial sanitization
            try:
                mol = Chem.MolFromSmiles(clean, sanitize=False)
                if mol is not None:
                    Chem.SanitizeMol(mol, catchErrors=True)
                    canon = Chem.MolToSmiles(mol, canonical=True)
                    if canon:
                        result[smi] = {
                            "canonical": canon,
                            "valid": True,
                            "truncated": False,
                            "message": "Fixed via partial sanitization.",
                        }
                        continue
            except Exception:
                pass

            # Try bracket/paren/ring completion for truncated SMILES
            completed = _attempt_bracket_completion(clean)
            if completed is not None:
                result[smi] = {
                    "canonical": completed,
                    "valid": True,
                    "truncated": True,
                    "message": f"Fixed by completing missing brackets/closures: '{completed}'.",
                }
                continue

            truncated = _likely_truncated(clean)
            msg = (
                f"Likely truncated — could not complete missing brackets automatically. "
                f"Please supply the correct SMILES for '{smi}'."
                if truncated
                else f"Could not parse '{smi}' — invalid valence, bad aromaticity, or malformed notation."
            )
            result[smi] = {"canonical": None, "valid": False, "truncated": truncated, "message": msg}
        return result

