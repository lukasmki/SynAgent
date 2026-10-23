from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.validation._toolset import SynthesisValidationToolset


@dataclass
class SynthesisValidation(AbstractCapability[AgentDepsT]):
    id: str = field(default="synthesis-validation", kw_only=True)
    description: str = field(default="Use for synthesis path validation.", kw_only=True)
    defer_loading: bool = field(default=True, kw_only=True)

    def get_instructions(self) -> str:
        return (
            "Validate a synthesis route with these tools. Report exactly what the "
            "tools return. If you edit a SMILES or template, validate the edited "
            "version; a step passes only on the version that was validated.\n\n"
            "Procedure:\n"
            "1. Collect every molecule in the route: building blocks, "
            "intermediates, and the target. Check them in a single "
            "validate_smiles call.\n"
            "2. Check all reaction templates in a single "
            "validate_reaction_smarts call.\n"
            "3. For each step, call validate_products with the template, the "
            "step's reactants, and its expected product.\n"
            "   - Templates must run forward (reactants>>product). If a step "
            "produces no products, the template may be written retrosynthetically: "
            "flip it with reverse_reaction and retry once.\n"
            "   - Reactants must be listed in the same order as the template's "
            "reactant patterns. If a multi-reactant step produces no products, "
            "retry once with the reactants reordered.\n"
            "   - Still validate later steps when an earlier step fails.\n"
            "4. Finish by calling create_report, then summarize which steps "
            "failed and why.\n\n"
            "Report fields:\n"
            "- building_blocks lists only the starting materials, each as an "
            'object: {"smiles": ..., "is_valid": true/false}.\n'
            "- Every reaction entry needs reaction_number, reaction_template, "
            "reactant_smiles, expected_product, actual_products, status, and "
            "failure_mode.\n"
            "- Number reactions in forward order, so the last step forms the "
            "target.\n"
            "- reaction_template is the template that was actually run, "
            "including a reversed one.\n"
            "- expected_product is the product the route lists for the step.\n"
            "- actual_products lists the products named in the tool message.\n"
            '- status is "passed" or "failed".\n'
            "- failure_mode is JSON null for a passed step, otherwise one of:\n"
            "  invalid_template: the template could not be parsed\n"
            "  invalid_reactants: a reactant SMILES is invalid\n"
            "  no_products: the template did not match the reactants\n"
            "  product_mismatch: products formed, but not the expected one\n"
            "- If the input contains several routes, create one report per route."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return SynthesisValidationToolset()
