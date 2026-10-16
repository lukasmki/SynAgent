from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synagent.validation._models import ValidationReport
from synagent.validation.rdkit_logging import capture_rdkit_log


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

    async def validate_smiles(self, smiles: list[str]) -> dict[str, bool | str]:
        """Checks the validity of SMILES strings

        Args:
            smiles (list[str]): Plain SMILES strings, one molecule each,
                without tags or labels. Pass every molecule in a single call.
                Example: ["CC(=O)O", "CN", "CNC(C)=O"]

        Returns:
            dict[str, bool | str]: {smiles: True | error}
        """
        result = {}
        for s in smiles:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmiles(s)
            result[s] = True if mol is not None else log.annotate("Invalid SMILES.")
        return result

    async def validate_reaction_smarts(
        self, reaction_smarts: list[str]
    ) -> dict[str, bool | str]:
        """Checks the validity of reaction SMARTS strings

        Args:
            reaction_smarts (list[str]): Bare reaction SMARTS strings of the
                form "reactants>>products". Remove any <rxn> and </rxn> tags
                from the pathway output first; tags make the template unparseable.
                Example: ["[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]"]

        Returns:
            dict[str, bool | str]: {reaction_smarts: is_valid | error}
        """
        result = {}
        for rs in reaction_smarts:
            with capture_rdkit_log() as log:
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(rs)
                    rxn.Initialize()
                except ValueError as e:
                    result[rs] = log.annotate(str(e))
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
            reaction_smarts (str): One bare forward reaction SMARTS
                ("reactants>>product"), without <rxn> tags.
                Example: "[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]"
            reactant_smiles (list[str]): One SMILES per reactant pattern, in
                the same order as the patterns left of ">>".
                Example: ["CC(=O)O", "CN"] (acid first, then amine)
            expected_product (str | None): SMILES of the product this step
                should form, e.g. "CNC(C)=O". Pass null to list the products.

        Returns:
            tuple[bool, str]: (is_valid, message)
        """
        # parse the reaction SMARTS
        with capture_rdkit_log() as log:
            try:
                rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
                rxn.Initialize()
            except ValueError as e:
                return False, log.annotate(
                    f"`reaction_smarts` could not be parsed: {e}"
                )

        # Validate reactants
        reactants = []
        errors = []
        for s in reactant_smiles:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmiles(s)
            if mol is None:
                errors.append(log.annotate(f"{s!r} is invalid."))
            reactants.append(mol)
        if errors:
            return False, "\n".join(
                ["`reactant_smiles` contains invalid SMILES strings:", *errors]
            )

        # Validate the expected product
        canon_product = None
        if expected_product is not None:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmiles(expected_product)
            if mol is None:
                return False, log.annotate("`expected_product` is invalid SMILES.")
            canon_product = Chem.MolToSmiles(mol, ignoreAtomMapNumbers=True)

        # Run the reaction
        with capture_rdkit_log() as log:
            try:
                products = [
                    Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
                    for p in rxn.RunReactants(reactants)
                    for m in p
                ]
            except Exception as e:
                return False, log.annotate(str(e))

        if not products:
            return False, log.annotate("Reaction produced no products.")

        # Check for expected product
        if canon_product is not None:
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
            reaction_smarts (list[str]): Bare reaction SMARTS strings to flip,
                without <rxn> tags.
                Example: ["[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OH].[N;H2:3]"]
                returns ["[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]"]

        Returns:
            list[str]: Reversed reaction SMARTS
        """

        return [">>".join(rs.split(">>")[::-1]) for rs in reaction_smarts]

    async def create_report(self, report: ValidationReport) -> ValidationReport:
        """Composes the results into a final validation report.
        Build it only from earlier tool outputs, using the same bare SMILES
        and SMARTS strings that were passed to the tools, without tags.

        Each entry of building_blocks is an object:
        {"smiles": "CN", "is_valid": true}
        Each entry of reactions is an object with all of these fields:
        {"reaction_number": 1,
         "reaction_template": "[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]",
         "reactant_smiles": ["CC(=O)O", "CN"],
         "expected_product": "CNC(C)=O",
         "actual_products": ["CNC(C)=O"],
         "status": "passed",
         "failure_mode": null}

        Args:
            report (ValidationReport): Summary of validation results.

        Returns:
            ValidationReport
        """
        return report
