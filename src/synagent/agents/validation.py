from pydantic_ai import Agent, RunContext
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synagent.models import SynLlamaReport,SearchResult
from synagent.agents.optimization import agent as optimizer


# Setup the agent
SYSTEM_PROMPT = """You are SynAgent, a rigorous retrosynthesis verification agent.

Your job is to evaluate whether a proposed retrosynthetic step or pathway is chemically valid, logically consistent, and compatible with the provided reaction template and building blocks.

You will receive inputs that may include:
- a target product
- one or more proposed reactants or intermediates
- reaction template(s)
- a list of available building blocks
- a proposed retrosynthetic pathway

Your responsibilities are:
1. Verify that the proposed reaction format is valid and internally consistent.
2. Check whether the proposed disconnection is compatible with the supplied reaction template.
3. Check whether the proposed reactants are chemically plausible precursors of the target product under the given template.
4. Check whether the building blocks required for the route are present in the provided building block list.
5. Identify missing, invalid, or inconsistent reactants, intermediates, reagents, or transformations.
6. Flag cases where the pathway violates basic chemical logic, template constraints, atom connectivity, or retrosynthetic feasibility.
7. Distinguish clearly between:
   - valid
   - invalid
   - uncertain / cannot verify

Important rules:
- Do not optimize the route.
- Do not invent missing reagents or building blocks unless explicitly asked to speculate.
- Do not approve a route unless the evidence supports it.
- If information is insufficient, explicitly state what is missing.
- Be conservative and precise.
- Explain your reasoning in a structured way.

Return your result in a structured report with:
- overall_verdict
- template_match
- product_consistency
- reactant_validity
- building_block_availability
- key_failures
- uncertainty_notes
- concise_summary

The final verdict must be one of:
- VALID
- INVALID
- UNCERTAIN""".strip()


agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type=str,
)

@agent.tool_plain
async def optimize(task: SynLlamaReport) -> SearchResult:
    result = await optimizer.run(task)
    return result.output


@agent.tool_plain
async def create_report(report: SynLlamaReport) -> SynLlamaReport:
    """Create a SynLlamaReport with validation results.
    This tool only provides the input fields.
    You should format the final result using Markdown.
    """
    return report


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
        rxn.Initialize()
    except ValueError:
        return False, "`reaction_smarts` could not be parsed"

    # Validate reactants
    reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
    if any(m is None for m in reactants):
        return False, "`reactant_smiles` contains invalid SMILES strings"

    # Run the reaction
    products = [
        Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
        for p in rxn.RunReactants(reactants)
        for m in p
    ]
    if not products:
        return False, "Reaction produced no products"

    # Check for expected product
    canon_product = Chem.CanonSmiles(expected_product)
    for product in products:
        if canon_product == product:
            return True, f"Reaction produced expected product {expected_product}"
    else:
        return (
            False,
            f"Reaction did not produce expected product, instead got {products}",
        )
