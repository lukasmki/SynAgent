from collections.abc import Iterable, Sequence
from html import escape
from typing import Self

from pydantic import BaseModel, Field


class SynLlamaReaction(BaseModel):
    "A single reaction step"

    reaction_number: int = Field(description="Step number")
    reaction_template: str = Field(description="Reaction SMARTS format")
    reactants: list[str] = Field(description="SMILES format")
    product: str = Field(description="SMILES format")


class SynLlamaFormat(BaseModel):
    "Retrosyntheis path in SynLlama format"

    reactions: list[SynLlamaReaction]
    building_blocks: list[str] = Field(description="Building blocks in SMILES format")

    # Plain methods, not fields: the JSON schema the model sees stays the same.

    @classmethod
    def from_steps(cls, steps: Iterable[tuple[str, Sequence[str], str]]) -> Self:
        """Build a route from `(template, reactants, product)` steps in forward order.

        Steps are numbered from 1. The building blocks are the reactants that no
        earlier step produces, in the order they first appear.
        """
        reactions: list[SynLlamaReaction] = []
        building_blocks: list[str] = []
        products: set[str] = set()
        for number, (template, reactants, product) in enumerate(steps, start=1):
            reactions.append(
                SynLlamaReaction(
                    reaction_number=number,
                    reaction_template=template,
                    reactants=list(reactants),
                    product=product,
                )
            )
            for smiles in reactants:
                if smiles not in products and smiles not in building_blocks:
                    building_blocks.append(smiles)
            products.add(product)
        return cls(reactions=reactions, building_blocks=building_blocks)

    @classmethod
    def from_json(cls, text: str | bytes) -> Self:
        """Read a route in SynLlama JSON, as `synagent run generation` prints it."""
        return cls.model_validate_json(text)

    def _repr_html_(self) -> str:
        steps = "".join(
            f"<div><b>Step {r.reaction_number}</b><br>{_draw_reaction(r)}</div>"
            for r in self.reactions
        )
        blocks = ", ".join(f"<code>{escape(s)}</code>" for s in self.building_blocks)
        return f"{steps}<p>Building blocks: {blocks}</p>"


def _draw_reaction(reaction: SynLlamaReaction) -> str:
    """SVG of `reactants>>product`, or the SMILES as text when RDKit can't parse it."""
    from rdkit.Chem import rdChemReactions
    from rdkit.Chem.Draw import rdMolDraw2D

    from synagent.validation.rdkit_logging import capture_rdkit_log

    smiles = f"{'.'.join(reaction.reactants)}>>{reaction.product}"
    with capture_rdkit_log():
        try:
            rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
        except ValueError:
            return f"<code>{escape(smiles)}</code>"
    drawer = rdMolDraw2D.MolDraw2DSVG(700, 180)
    drawer.DrawReaction(rxn)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
