import json
import logging

from rdkit import Chem
from rdkit.Chem import rdChemReactions

logging.basicConfig(level=logging.CRITICAL)


class SmilesError(ValueError): ...


class ReactantError(ValueError): ...


class ProductsError(ValueError): ...


def validate(row: dict):
    try:
        data = json.loads(row["response"])
    except json.JSONDecodeError as e:
        raise e

    reactions = data["reactions"]
    for reaction in reactions:
        rxn_template = reaction["reaction_template"]
        if rxn_template.startswith("<rxn>"):
            rxn_template = rxn_template[5:]
        if rxn_template.endswith("</rxn>"):
            rxn_template = rxn_template[:-6]

        # Load reaction
        rxn = rdChemReactions.ReactionFromSmarts(rxn_template)
        rxn.Initialize()

        # Load reactants and products
        reactants = [Chem.MolFromSmiles(m) for m in reaction["reactants"]]
        product = Chem.MolFromSmiles(reaction["product"])
        if any(m is None for m in reactants) or product is None:
            raise SmilesError("Invalid smiles")

        for reactant in reactants:
            reactant: Chem.Mol
            for template in rxn.GetReactants():
                match = reactant.HasSubstructMatch(template)
                if match:
                    break
            else:
                raise ReactantError()

        product_smiles = Chem.MolToSmiles(
            product, canonical=True, ignoreAtomMapNumbers=True
        )
        products = [
            Chem.MolToSmiles(m, canonical=True, ignoreAtomMapNumbers=True)
            for p in rxn.RunReactants(reactants)
            for m in p
        ]
        if product_smiles not in products:
            raise ProductsError()
