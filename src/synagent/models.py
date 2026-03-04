from typing import Annotated
from pydantic import BaseModel

type Smiles = Annotated[str, "Molecule SMILES string"]
type ReactionSmarts = Annotated[str, "Reaction SMARTS string"]


class SynLlamaReaction(BaseModel):
    reaction_number: int
    reaction_template: ReactionSmarts
    reactants: list[Smiles]
    product: Smiles


class SynLlamaFormat(BaseModel):
    reactions: list[SynLlamaReaction]
    building_blocks: list[Smiles]
