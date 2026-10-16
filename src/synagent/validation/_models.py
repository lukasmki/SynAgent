from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BuildingBlockResult(BaseModel):
    smiles: str = Field(description="Building block SMILES.")
    name: str | None = None
    is_valid: bool = Field(description="Result of validate_smiles.")


class ReactionResult(BaseModel):
    reaction_number: int = Field(description="Step number in forward order, from 1.")
    reaction_template: str = Field(description="Reaction SMARTS that was run.")
    reactant_smiles: list[str] = Field(
        description="Reactant SMILES, in template order."
    )
    expected_product: str = Field(
        description="Product SMILES the route lists for this step."
    )
    actual_products: list[str] = Field(
        description="Canonical SMILES of all products actually produced by the template. "
        "Empty if the reaction produced no products."
    )
    status: Literal["passed", "failed"]
    failure_mode: str | None = Field(
        default=None,
        description="Null on success. Otherwise one of invalid_template, "
        "invalid_reactants, no_products, product_mismatch.",
    )

    @field_validator("failure_mode", mode="before")
    @classmethod
    def _null_string(cls, value: object) -> object:
        # Models often write the string "null" instead of JSON null.
        if isinstance(value, str) and value.strip().lower() in ("", "null", "none"):
            return None
        return value


class ValidationReport(BaseModel):
    reactions: list[ReactionResult] = Field(
        description="One entry per reaction step, in forward order."
    )
    building_blocks: list[BuildingBlockResult] = Field(
        description="One object per starting material, not plain SMILES strings."
    )

    target_molecule: str
    all_building_blocks_valid: bool = Field(
        description="True only if every building block passed SMILES validation. "
        "Fast-path flag for downstream agents to short-circuit on invalid inputs."
    )
    all_reactions_passed: bool = Field(
        description="True only if every reaction step produced the expected product."
    )
