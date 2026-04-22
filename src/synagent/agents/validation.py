from pydantic_ai import Agent, RunContext
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synagent.models import SynLlamaReport, OptimizationReport
from synagent.agents.optimization import agent as optimizer
from synagent.agents.optimization import optimize_route
from pydantic_ai.models.google import GoogleModel
import logfire

logfire.configure()
logfire.instrument_pydantic_ai()


# Setup the agent
SYSTEM_PROMPT = """You are SynAgent, a rigorous retrosynthesis verification agent.

First validate the proposed route.
If validation is complete and an optimization step is requested or useful,
call the `optimize` tool, and return OptimizationReport,
after return the optimizationreport, generate a markdown report""".strip()

agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type=[SynLlamaReport,OptimizationReport],
)

WRITER_PROMPT = """
    You are a scientific report writer.
    Given an optimization result, write a clear markdown report.
    Use markdown headings, bullet points, and short explanations.
    Do not return JSON
    """.strip()

'''writer = Agent(
    model = GoogleModel('gemini-3-flash-preview'),
    output_type = str,
    system_prompt = WRITER_PROMPT,
)'''

@agent.tool_plain
async def optimize(task: SynLlamaReport) -> OptimizationReport:
    result = await optimize_route(task)
    '''writer_input = f"""
    write a markdown report based on this optimization result. Optimization result: {result.model_dump_json(indent=2)}
    """.strip
    writer_result = await writer.run(writer_input)'''
    return result


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
