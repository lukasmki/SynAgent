from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synagent.validation._models import ValidationReport


class SynthesisValidationToolset(FunctionToolset[AgentDepsT]):
    """Toolset for SMILES and reaction SMARTS validation."""

    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.validate_smiles, name="validate_smiles")
        self.add_function(
            self.validate_reaction_smarts, name="validate_reaction_smarts"
        )
        self.add_function(self.validate_products, name="validate_products")
        self.add_function(self.reverse_reaction, name="reverse_reaction")
        self.add_function(self.create_report, name="create_report")

    async def validate_smiles(self, smiles: list[str]) -> dict[str, bool]:
        """Checks the validity of SMILES strings

        Args:
            smiles (list[str]): list of SMILES to check

        Returns:
            dict[str, bool]: {smiles: is_valid}
        """
        return {s: Chem.MolFromSmiles(s) is not None for s in smiles}

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
            try:
                rxn = rdChemReactions.ReactionFromSmarts(rs)
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
    ) -> tuple[bool, str]:
        """Runs the reaction on the given reactants and
        checks if the expected product is formed.
        If no expected product is given, the products are returned.

        Args:
            reaction_smarts (str): Reaction SMARTS
            reactant_smiles (list[str]): Reactant SMILES
            expected_product (str | None): Product SMILES

        Returns:
            tuple[bool, str]: (is_valid, message)
        """
        # parse the reaction SMARTS
        try:
            rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
            rxn.Initialize()
        except ValueError:
            return False, "`reaction_smarts` could not be parsed."

        # Validate reactants
        reactants = [Chem.MolFromSmiles(s) for s in reactant_smiles]
        if any(m is None for m in reactants):
            return False, "`reactant_smiles` contains invalid SMILES strings."

        # Run the reaction
        products = [
            Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
            for p in rxn.RunReactants(reactants)
            for m in p
        ]
        if not products:
            return False, "Reaction produced no products."

        # Check for expected product
        if expected_product is not None:
            canon_product = Chem.CanonSmiles(expected_product)
            for product in products:
                if canon_product == product:
                    return (
                        True,
                        f"Reaction produced expected product: {expected_product}",
                    )
            else:
                return (
                    False,
                    f"Reaction did not produce expected product, instead got {products}",
                )
        else:
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
