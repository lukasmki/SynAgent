import re
from itertools import permutations

import json

from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.tools import AgentDepsT, RunContext
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions

from synagent.validation._smarts import _COMMON_SMARTS

try:
    from rdchiral.template_extractor import extract_from_reaction as _rdchiral_extract
except ImportError:
    _rdchiral_extract = None

try:
    from indigo import Indigo
except ImportError:
    Indigo = None

RDLogger.DisableLog("rdApp.*")


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
    # Unbalanced square brackets
    if smi.count("[") != smi.count("]"):
        return True
    # Unbalanced parentheses
    if smi.count("(") != smi.count(")"):
        return True
    # Unpaired ring-closure digits: each digit 1-9 and %nn must appear an even number of times
    # Simple check: collect all ring-closure tokens
    tokens = re.findall(r"%\d{2}|\d", re.sub(r"\[.*?\]", "", smi))
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    if any(v % 2 != 0 for v in counts.values()):
        return True
    return False


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

            if part.tool_name == "fix_building_blocks":
                for old_smi, res in content.get("results", {}).items():
                    if res.get("valid") and res.get("canonical"):
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

    async def fix_step(self, ctx: RunContext[AgentDepsT], step: int) -> dict:
        """Fix a failed reaction step using the ValidationReport already in the conversation.

        Reads the most recent validate_route result automatically — no SMILES copying needed.
        Runs the correct fix chain for the step's failure_mode:
          - invalid_template  → fix_smarts → fix_template
          - no_products / wrong_product → fix_template → extract_template_from_reaction
                                          → retro disconnection fallback
          - invalid_reactant_smiles / invalid_product_smiles → fix_smiles

        Args:
            step (int): Reaction number to fix (as shown in the ValidationReport).

        Returns:
            dict with fixed=True/False, new_template or smiles_fixes, and message.
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

        # --- invalid_template: try fix_smarts then fix_template ---
        if failure == "invalid_template":
            sr = await self.fix_smarts(template)
            if sr.get("fixed"):
                return {"fixed": True, "step": step, "failure_mode": failure,
                        "method": "fix_smarts", "new_template": sr["smarts"],
                        "message": sr["message"]}
            tr = await self.fix_template(reactants, product)
            return {"fixed": tr.get("found", False), "step": step, "failure_mode": failure,
                    "method": "fix_template", "new_template": tr.get("template"),
                    "message": tr.get("message")}

        # --- no_products / wrong_product: fix_template → extract → retro ---
        if failure in ("no_products", "wrong_product"):
            tr = await self.fix_template(reactants, product)
            if tr.get("found"):
                return {"fixed": True, "step": step, "failure_mode": failure,
                        "method": "fix_template", "new_template": tr["template"],
                        "message": tr.get("message")}

            er = await self.extract_template_from_reaction(reactants, product)
            if er.get("fixed"):
                return {"fixed": True, "step": step, "failure_mode": failure,
                        "method": "extract_template_from_reaction", "new_template": er["smarts"],
                        "message": er.get("message")}

            # Retro disconnection fallback: reverse the template, apply to product,
            # try fix_template with each suggested precursor set
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
                                "method": "retro_disconnection+fix_template",
                                "new_reactants": frags,
                                "new_template": tr2["template"],
                                "message": (f"Original reactants could not produce the product. "
                                            f"Retro disconnection found alternative precursors: {frags}")}
            except Exception:
                pass

            return {"fixed": False, "step": step, "failure_mode": failure,
                    "message": ("fix_template, extract_template_from_reaction, and retro disconnection "
                                "all failed. The route step may need to be redesigned.")}

        # --- invalid SMILES ---
        if failure in ("invalid_reactant_smiles", "invalid_product_smiles"):
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

        # --- Build corrected building blocks list ---
        corrected_bbs = [bb_fixes.get(bb.smiles, bb.smiles) for bb in report.building_blocks]

        # --- Build corrected reactions list ---
        corrected_reactions = []
        for rxn in report.reactions:
            fix = step_fixes.get(rxn.reaction_number, {})

            # Template: use new_template if fix found one
            template = fix.get("new_template") or rxn.reaction_template

            # Reactants: prefer new_reactants, otherwise apply smiles_fixes per-reactant
            if fix.get("new_reactants"):
                reactants = fix["new_reactants"]
            else:
                smiles_fixes = fix.get("smiles_fixes", {})
                reactants = [
                    smiles_fixes.get(r, {}).get("canonical") or r
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

        corrected_route = {"reactions": corrected_reactions, "building_blocks": corrected_bbs}
        return _validate_route_dict(corrected_route)

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

    async def fix_template(
        self,
        reactant_smiles: list[str],
        product_smiles: str,
    ) -> dict:
        """Fixes a failed reaction step by searching the template library for a valid
        SMARTS template that produces the expected product from the given reactants.
        Use when validate_products fails due to an invalid or missing reaction template.

        Args:
            reactant_smiles (list[str]): Reactant SMILES for the failed step.
            product_smiles (str): Expected product SMILES for the failed step.

        Returns:
            dict: {"found": bool, "template": str | None, "reactants": list,
                   "product": str, "message": str}
        """
        reactant_smiles = _parse_smiles_list(reactant_smiles)
        product_smiles = _strip_tags(str(product_smiles))

        product_mol = Chem.MolFromSmiles(product_smiles)
        if product_mol is None:
            return {"found": False, "template": None, "message": f"Invalid product SMILES: {product_smiles}"}

        reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        if any(m is None for m in reactant_mols):
            return {"found": False, "template": None, "message": "One or more reactant SMILES could not be parsed."}

        canon_product = Chem.CanonSmiles(product_smiles)

        for smarts in _COMMON_SMARTS:
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

            # Detect likely truncation: unbalanced brackets/parens or unpaired ring closures
            truncated = _likely_truncated(clean)
            msg = (
                f"Likely truncated — string appears cut off mid-structure. "
                f"The route needs to be re-generated for '{smi}'."
                if truncated
                else f"Could not parse '{smi}' — invalid valence, bad aromaticity, or malformed notation."
            )
            result[smi] = {"canonical": None, "valid": False, "truncated": truncated, "message": msg}
        return result

