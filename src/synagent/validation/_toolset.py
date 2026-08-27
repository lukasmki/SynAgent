import json
import os
import re
import sys
from itertools import permutations

from pydantic_ai import FunctionToolset
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from rdkit import Chem, DataStructs
from rdkit.Chem import rdChemReactions, rdFingerprintGenerator

from synagent.validation._models import (
    BuildingBlockResult,
    ReactionResult,
    ValidationReport,
)

# SA scorer — optional, loaded once at import time
try:
    import rdkit as _rdkit_pkg

    _sa_score_dir = os.path.join(
        os.path.dirname(_rdkit_pkg.__file__), "Contrib", "SA_Score"
    )
    if _sa_score_dir not in sys.path:
        sys.path.append(_sa_score_dir)
    import sascorer as _sascorer
except Exception:
    _sascorer = None


ANALOG_PRODUCT_SIMILARITY_THRESHOLD = 0.60
_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def _strip_tags(s: str) -> str:
    cleaned = re.sub(r"<[^>]+>", "", s).strip()
    if len(cleaned) >= 2 and cleaned[0] in ('"', "'") and cleaned[-1] == cleaned[0]:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _sa_score(smiles: str) -> float | None:
    if _sascorer is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(_sascorer.calculateScore(mol))
    except Exception:
        return None


def _extract_csv_target(raw: str) -> str | None:
    """Extract the target SMILES from the first field of a CSV row like: SMILES,params,{json}."""
    raw = raw.strip()
    if raw.startswith("{") or raw.startswith('"'):
        return None
    idx = raw.find(",")
    if idx > 0:
        candidate = raw[:idx].strip()
        if " " not in candidate and len(candidate) > 5:
            return candidate
    return None


def _extract_route_from_messages(messages: list) -> str:
    """Scan recent user messages for a route JSON string and return the first match."""
    for msg in reversed(messages):
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    content = str(part.content)
                    # Accept messages that look like they contain a route JSON
                    if '"reactions"' in content or '""reactions""' in content:
                        return content
    return ""


def _match_product(
    expected_product: str,
    actual_products: list[str],
    analog_threshold: float | None = ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
) -> tuple[bool, str | None, str | None, float | None]:
    """Match an expected product exactly, then optionally by Morgan similarity.

    Exact canonical-SMILES equality always wins. When ``analog_threshold`` is not
    None, the closest actual product is accepted as an analog only when its
    radius-2, 4096-bit Morgan-fingerprint Tanimoto similarity is strictly above
    the threshold. Returning the match type and score keeps analog-aware results
    auditable and allows benchmarks to retain the original strict metric.
    """
    expected_mol = Chem.MolFromSmiles(expected_product)
    if expected_mol is None:
        return False, None, None, None

    expected_canon = Chem.MolToSmiles(
        expected_mol, canonical=True, ignoreAtomMapNumbers=True
    )
    valid_products: list[tuple[str, Chem.Mol]] = []
    for product in actual_products:
        product_mol = Chem.MolFromSmiles(product)
        if product_mol is None:
            continue
        product_canon = Chem.MolToSmiles(
            product_mol, canonical=True, ignoreAtomMapNumbers=True
        )
        if product_canon == expected_canon:
            return True, "exact", product_canon, 1.0
        valid_products.append((product_canon, product_mol))

    if analog_threshold is None or not valid_products:
        return False, None, None, None
    if not 0.0 <= analog_threshold <= 1.0:
        raise ValueError("analog_threshold must be between 0.0 and 1.0")

    expected_fp = _morgan_generator.GetFingerprint(expected_mol)
    best_product: str | None = None
    best_similarity = -1.0
    for product_canon, product_mol in valid_products:
        similarity = float(
            DataStructs.TanimotoSimilarity(
                expected_fp, _morgan_generator.GetFingerprint(product_mol)
            )
        )
        if similarity > best_similarity:
            best_product = product_canon
            best_similarity = similarity

    score = round(best_similarity, 4)
    if best_similarity > analog_threshold:
        return True, "analog", best_product, score
    return False, None, best_product, score


def _validate_route_dict(
    route: dict,
    analog_product_threshold: float | None = ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
) -> ValidationReport:
    """Core validation logic — takes a parsed route dict, returns a ValidationReport."""
    reactions_data: list[dict] = route.get("reactions", [])
    bb_data: list = route.get("building_blocks", [])

    target = ""
    if reactions_data:
        target = _strip_tags(str(reactions_data[0].get("product", "")))

    bb_results: list[BuildingBlockResult] = []
    for bb in bb_data:
        smi = _strip_tags(str(bb))
        is_valid = Chem.MolFromSmiles(smi) is not None
        bb_results.append(
            BuildingBlockResult(
                smiles=smi,
                is_valid=is_valid,
                suggested_fix="fix_smiles" if not is_valid else None,
            )
        )

    rxn_results: list[ReactionResult] = []
    for i, rxn_data in enumerate(reactions_data):
        step_num = rxn_data.get("reaction_number", i + 1)
        template = _strip_tags(str(rxn_data.get("reaction_template", "")))
        reactants = [
            _strip_tags(str(s)) for s in rxn_data.get("reactants", []) if str(s).strip()
        ]
        product = _strip_tags(str(rxn_data.get("product", "")))

        invalid_reactants = [s for s in reactants if Chem.MolFromSmiles(s) is None]
        if invalid_reactants:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=[],
                    status="failed",
                    failure_mode="invalid_reactant_smiles",
                    suggested_fix="fix_smiles",
                )
            )
            continue

        if Chem.MolFromSmiles(product) is None:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=[],
                    status="failed",
                    failure_mode="invalid_product_smiles",
                    suggested_fix="fix_smiles",
                )
            )
            continue

        try:
            rxn_obj = rdChemReactions.ReactionFromSmarts(template)
            if rxn_obj is None:
                raise ValueError("null reaction")
            rxn_obj.Initialize()
        except Exception:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=[],
                    status="failed",
                    failure_mode="invalid_template",
                    suggested_fix="fix_smarts",
                )
            )
            continue

        reactant_mols = [Chem.MolFromSmiles(s) for s in reactants]
        actual_products: list[str] = []

        for perm in permutations(reactant_mols):
            for outputs in rxn_obj.RunReactants(perm):
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

        found, match_type, matched_product, product_similarity = _match_product(
            product, actual_products, analog_product_threshold
        )

        if not actual_products:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=[],
                    status="failed",
                    failure_mode="no_products",
                    suggested_fix="fix_template",
                )
            )
        elif not found:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=actual_products,
                    status="failed",
                    failure_mode="wrong_product",
                    suggested_fix="fix_template",
                    matched_product=matched_product,
                    product_similarity=product_similarity,
                )
            )
        else:
            rxn_results.append(
                ReactionResult(
                    reaction_number=step_num,
                    reaction_template=template,
                    reactant_smiles=reactants,
                    expected_product=product,
                    actual_products=actual_products,
                    status="passed",
                    failure_mode=None,
                    suggested_fix=None,
                    product_match_type=match_type,
                    matched_product=matched_product,
                    product_similarity=product_similarity,
                )
            )

    target_sa = None
    if target:
        score = _sa_score(target)
        if score is not None:
            target_sa = round(score, 2)

    issues: list[str] = []
    for bb in bb_results:
        if not bb.is_valid:
            issues.append(f"Building block has invalid SMILES: '{bb.smiles}'")
    for rxn in rxn_results:
        if rxn.status == "failed":
            if rxn.suggested_fix == "fix_smarts":
                issues.append(
                    f"Step {rxn.reaction_number}: reaction template cannot be parsed"
                )
            elif rxn.suggested_fix == "fix_template":
                issues.append(
                    f"Step {rxn.reaction_number}: template does not produce expected product"
                )
            elif rxn.suggested_fix == "fix_smiles":
                issues.append(
                    f"Step {rxn.reaction_number}: reactant or product SMILES is invalid"
                )
    if target_sa is not None and target_sa > 6.0:
        issues.append(
            f"Target SA score {target_sa} is high — may be difficult to synthesize"
        )
    suggested_fixes = issues

    return ValidationReport(
        reactions=rxn_results,
        building_blocks=bb_results,
        target_molecule=target,
        target_sa_score=target_sa,
        all_building_blocks_valid=all(bb.is_valid for bb in bb_results),
        all_reactions_passed=all(r.status == "passed" for r in rxn_results),
        suggested_fixes=suggested_fixes,
    )


def _parse_route_json(raw: str) -> dict:
    """Parse route JSON from a raw string, handling CSV-wrapped and double-escaped forms."""
    raw = raw.strip()

    # Try direct JSON parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from within the string (handles CSV prefix/suffix)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_str = raw[start : end + 1]
        # Unescape CSV-style doubled double-quotes
        json_str = json_str.replace('""', '"')
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse route JSON (first 200 chars): {raw[:200]}")


class SynthesisValidationToolset(FunctionToolset[AgentDepsT]):
    """Toolset for SMILES and reaction SMARTS validation."""

    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.validate_route, name="validate_route")
        self.add_function(self.validate_smiles, name="validate_smiles")
        self.add_function(
            self.validate_reaction_smarts, name="validate_reaction_smarts"
        )
        self.add_function(self.validate_products, name="validate_products")
        self.add_function(self.reverse_reaction, name="reverse_reaction")
        self.add_function(self.create_report, name="create_report")

    async def validate_route(
        self, ctx: RunContext[AgentDepsT], route_json: str = "from_message"
    ) -> ValidationReport:
        """Validates a complete synthesis route.

        Pass route_json="from_message" (the default) to automatically extract the
        route from the user's most recent message — no copying needed.
        Only pass the actual JSON string if it is short and self-contained.

        Args:
            route_json (str): "from_message" (default) to auto-extract from the
                conversation, or the raw route JSON / CSV row string.

        Returns:
            ValidationReport with per-step results and a suggested_fixes list.
        """
        # Auto-extract from the last user message when the model passes "from_message"
        # or any other short placeholder — this avoids Qwen having to copy long strings.
        if (
            not route_json
            or len(route_json) < 80
            or route_json.strip().lower()
            in (
                "from_message",
                "from message",
                "see above",
                "above",
                "the route",
                "route",
            )
        ):
            route_json = _extract_route_from_messages(ctx.messages)

        try:
            route = _parse_route_json(route_json)
        except Exception as exc:
            return ValidationReport(
                reactions=[],
                building_blocks=[],
                target_molecule="unknown",
                all_building_blocks_valid=False,
                all_reactions_passed=False,
                suggested_fixes=[f"Route JSON could not be parsed: {exc}"],
            )

        return _validate_route_dict(route)

    async def validate_smiles(self, smiles: list[str]) -> dict[str, bool]:
        """Checks the validity of SMILES strings

        Args:
            smiles (list[str]): list of SMILES to check

        Returns:
            dict[str, bool]: {smiles: is_valid}
        """
        return {s: Chem.MolFromSmiles(_strip_tags(s)) is not None for s in smiles}

    async def validate_reaction_smarts(
        self, reaction_smarts: list[str]
    ) -> dict[str, bool | str]:
        """Checks the validity of reaction SMARTS strings

        Args:
            reaction_smarts (list[str]): list of reaction SMARTS to check

        Returns:
            dict[str, bool | str]: {reaction_smarts: is_valid | error}
        """
        result = {}
        for rs in reaction_smarts:
            clean = _strip_tags(rs)
            try:
                rxn = rdChemReactions.ReactionFromSmarts(clean)
                rxn.Initialize()
            except ValueError as e:
                result[rs] = str(e)
                continue
            result[rs] = True
        return result

    async def validate_products(
        self,
        reaction_smarts: str,
        reactant_smiles: list[str],
        expected_product: str | None,
        analog_similarity_threshold: float | None = ANALOG_PRODUCT_SIMILARITY_THRESHOLD,
    ) -> tuple[bool, str]:
        """Runs the reaction on the given reactants and checks if the expected product is formed.
        Use this to re-validate a single step after a fix has been applied.

        Args:
            reaction_smarts (str): Reaction SMARTS
            reactant_smiles (list[str]): Reactant SMILES
            expected_product (str | None): Product SMILES
            analog_similarity_threshold (float | None): Accept the closest actual
                product as an analog when its Morgan/Tanimoto score is strictly
                above this value. Use null for exact-match-only validation.

        Returns:
            tuple[bool, str]: (is_valid, message)
        """
        try:
            rxn = rdChemReactions.ReactionFromSmarts(_strip_tags(reaction_smarts))
            rxn.Initialize()
        except ValueError:
            return False, "`reaction_smarts` could not be parsed."

        reactants = [Chem.MolFromSmiles(_strip_tags(s)) for s in reactant_smiles]
        if any(m is None for m in reactants):
            return False, "`reactant_smiles` contains invalid SMILES strings."

        products = [
            Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
            for p in rxn.RunReactants(reactants)
            for m in p
        ]
        if not products:
            return False, "Reaction produced no products."

        if expected_product is not None:
            matched, match_type, matched_product, similarity = _match_product(
                _strip_tags(expected_product), products, analog_similarity_threshold
            )
            if matched and match_type == "exact":
                return True, f"Reaction produced expected product: {expected_product}"
            if matched:
                return True, (
                    "Reaction produced an approved product analog: "
                    f"{matched_product} (Morgan/Tanimoto={similarity:.4f}, "
                    f"threshold>{analog_similarity_threshold:.2f})"
                )
            similarity_note = (
                f"; closest Morgan/Tanimoto={similarity:.4f}"
                if similarity is not None
                else ""
            )
            return False, (
                f"Reaction did not produce expected product or an approved analog; "
                f"got {products}{similarity_note}"
            )
        return True, f"Reaction produced products: {products}"

    async def reverse_reaction(self, reaction_smarts: list[str]) -> list[str]:
        """Returns the reaction smarts with the reactant and product patterns reversed.

        Args:
            reaction_smarts (list[str]): Reaction to reverse in reaction SMARTS format

        Returns:
            list[str]: Reversed reaction SMARTS
        """
        return [">>".join(rs.split(">>")[::-1]) for rs in reaction_smarts]

    async def create_report(self, report: ValidationReport) -> ValidationReport:
        """Composes the results into a final validation report.

        Args:
            report (ValidationReport): Summary of validation results.

        Returns:
            ValidationReport
        """
        return report
