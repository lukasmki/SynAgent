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
    name: str | None = None
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
# Optimizer report (search) #
###################

class PriceLookupResult(BaseModel):
    smiles: str
    query_used: str
    price_text: str | None = None
    estimated_price_usd: float | None = None
    currency: str | None = None
    vendor_hint: str | None = None
    evidence: str

class HazardLookupResult(BaseModel):
    smiles: str
    query_used: str
    hazard_flags: list[str] = []
    hazard_score: float = Field(default=0.5, ge=0, le=1)  # higher = safer
    evidence: str

class AvailabilityLookupResult(BaseModel):
    smiles: str
    query_used: str
    availability: Literal["available", "likely_available", "unclear", "not_found"]
    vendor_hints: list[str] = []
    evidence: str

###################
# route level #
###################

class RouteCostResult(BaseModel):
    per_block: list[PriceLookupResult]
    total_estimated_cost_usd: float | None = None
    missing_price_count: int
    cost_score: float = Field(ge=0, le=1)

class RouteSafetyResult(BaseModel):
    per_block: list[HazardLookupResult]
    average_hazard_score: float = Field(ge=0, le=1)
    flagged_blocks: list[str] = []
    safety_score: float = Field(ge=0, le=1)

class BuildingBlockEvaluation(BaseModel):
    smiles: str
    name: str | None = None
    valid: bool
    price: PriceLookupResult | None = None
    hazard: HazardLookupResult | None = None
    availability: AvailabilityLookupResult | None = None

class OptimizationScore(BaseModel):
    cost_score: float = Field(ge=0, le=1)
    safety_score: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)

class OptimizationReport(BaseModel):
    is_optimizable: bool
    summary: str
    target_molecule: str
    issues_found: list[str]
    building_block_evaluations: list[BuildingBlockEvaluation]
    route_cost: RouteCostResult
    route_safety: RouteSafetyResult
    score: OptimizationScore
    recommended_actions: list[str]


