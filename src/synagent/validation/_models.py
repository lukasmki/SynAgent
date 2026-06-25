from typing import Literal

from pydantic import BaseModel, Field


class BuildingBlockResult(BaseModel):
    smiles: str
    name: str | None = None
    is_valid: bool


class ReactionResult(BaseModel):
    reaction_number: int
    reaction_template: str
    reactant_smiles: list[str]
    expected_product: str
    actual_products: list[str] = Field(
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


class ValidationReport(BaseModel):
    reactions: list[ReactionResult]
    building_blocks: list[BuildingBlockResult]

    target_molecule: str
    all_building_blocks_valid: bool = Field(
        description="True only if every building block passed SMILES validation. "
        "Fast-path flag for downstream agents to short-circuit on invalid inputs."
    )
    all_reactions_passed: bool = Field(
        description="True only if every reaction step produced the expected product."
    )
