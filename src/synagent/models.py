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
