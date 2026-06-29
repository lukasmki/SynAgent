from pydantic import BaseModel


class BuildingBlockSearchResult(BaseModel):
    smiles: str
    similarity: float


class ReactionSearchResult(BaseModel):
    reaction_smarts: str
    similarity: float
