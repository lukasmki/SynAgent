from typing import Annotated, Literal

from pydantic import BaseModel, Field

type Smiles = Annotated[str, "Molecule SMILES string"]
type ReactionSmarts = Annotated[str, "Reaction SMARTS string"]

###################
# SynLlama format #
###################


class SynLlamaReaction(BaseModel):
    "A single reaction step"

    reaction_number: int
    reaction_template: ReactionSmarts
    reactants: list[Smiles]
    product: Smiles


class SynLlamaFormat(BaseModel):
    "Retrosyntheis path in SynLlama format"

    reactions: list[SynLlamaReaction]
    building_blocks: list[Smiles]


###################
# SynAgent Report #
###################


class BuildingBlockResult(BaseModel):
    smiles: Smiles
    is_valid: bool


class ReactionResult(BaseModel):
    reaction_number: int
    reaction_template: ReactionSmarts
    reactant_smiles: list[Smiles]
    expected_product: Smiles
    actual_products: list[Smiles] = Field(
        description="Canonical SMILES of all products actually produced by the template. "
        "Empty if the reaction produced no products."
    )
    status: Literal["passed", "failed"]
    failure_mode: Literal[
        "invalid_reactants", "no_products", "product_mismatch", "invalid_template", None
    ] = Field(
        default=None,
        description="Null on success. Structured failure category for downstream error handling.",
    )


class SynLlamaReport(BaseModel):
    reactions: list[ReactionResult]
    building_blocks: list[BuildingBlockResult]

    target_molecule: Smiles
    all_building_blocks_valid: bool = Field(
        description="True only if every building block passed SMILES validation. "
        "Fast-path flag for downstream agents to short-circuit on invalid inputs."
    )
    all_reactions_passed: bool = Field(
        description="True only if every reaction step produced the expected product."
    )

###################
# Optimizer report #
###################

class BuildingBlockAssessment(BaseModel):
    price: str | None = None
    harzard: str | None = None
    smiles: BuildingBlockResult

class OptimizationReport(BaseModel):
    is_optimizable: bool
    summary: str
    building_block_assessments: list[BuildingBlockAssessment]
    recommended_actions:list[str]