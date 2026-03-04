from rdkit import Chem
from rdkit.Chem import rdChemReactions
from pydantic_ai import Agent

SYSTEM_PROMPT = """
You are a helpful AI agent.
""".strip()

# Setup the agent
agent = Agent(
    system_prompt=SYSTEM_PROMPT,
    output_type=str,
)


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
def validate_reaction(
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

    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    rxn.Initialize()

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
