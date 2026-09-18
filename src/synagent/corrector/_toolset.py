import asyncio
import functools
import re
from itertools import permutations
from pathlib import Path

import json

from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

from synagent.validation._toolset import ANALOG_PRODUCT_SIMILARITY_THRESHOLD, _match_product

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
                                          → retro_disconnection
          - invalid_reactant_smiles / invalid_product_smiles → fix_smiles

        Pass method to try exactly one option yourself instead of the full
        chain — call fix_step again with the next method if the first didn't
        fix it. Useful for orchestrators that reason better making one
        decision at a time than trusting a hidden multi-step chain. Valid
        values depend on the step's current failure_mode:
          - invalid_template: "fix_smarts", "fix_template"
          - no_products / wrong_product: "fix_template", "fix_via_analogue_building_block",
            "retro_disconnection"
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
            valid = {"auto", "fix_template", "fix_via_analogue_building_block", "retro_disconnection"}
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
                # Reverse the template, apply to product, try fix_template
                # with each suggested precursor set.
                retro_smarts = ">>".join(template.split(">>")[::-1])
                try:
                    product_mol = Chem.MolFromSmiles(product)
                    retro_rxn = rdChemReactions.ReactionFromSmarts(retro_smarts)
                    retro_rxn.Initialize()
                    seen: set[tuple] = set()
                    for outputs in retro_rxn.RunReactants((product_mol,)):
                        frags = []
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
                        tr2 = await self.fix_template(frags, product)
                        if tr2.get("found"):
                            return {"fixed": True, "step": step, "failure_mode": failure,
                                    "method": "retro_disconnection",
                                    "new_reactants": frags,
                                    "new_template": tr2["template"],
                                    "message": (f"Original reactants could not produce the product. "
                                                f"Retro disconnection found alternative precursors: {frags}")}
                except Exception:
                    pass
                if method == "retro_disconnection":
                    return {"fixed": False, "step": step, "failure_mode": failure,
                            "method": "retro_disconnection",
                            "message": "Retro disconnection found no working alternative precursor set."}

            return {"fixed": False, "step": step, "failure_mode": failure,
                    "message": ("fix_template, fix_via_analogue_building_block, and retro disconnection "
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

            # Product: apply smiles_fixes if available, otherwise bb_fixes
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

