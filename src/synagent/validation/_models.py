from typing import Literal

from pydantic import BaseModel, Field


class BuildingBlockResult(BaseModel):
    smiles: str
    name: str | None = None
    is_valid: bool
    suggested_fix: Literal["fix_smiles", "fix_building_block"] | None = Field(
        default=None,
        description=(
            "Set to 'fix_smiles' if the SMILES cannot be parsed by RDKit. "
            "Set to 'fix_building_block' if the SMILES is valid but the building block "
            "appears too complex or unlikely to be commercially available. "
            "Null if the building block is valid."
        ),
    )


class ReactionResult(BaseModel):
    reaction_number: int
    reaction_template: str
    reactant_smiles: list[str]
    expected_product: str
    actual_products: list[str] = Field(
        description="Canonical SMILES of all products actually produced by the template. "
        "Empty if the reaction produced no products."
    )
    product_match_type: Literal["exact", "analog"] | None = Field(
        default=None,
        description=(
            "How the expected product matched an actual product. 'exact' means "
            "canonical SMILES equality; 'analog' means 4096-bit Morgan-fingerprint Tanimoto "
            "similarity passed the configured threshold. Null when no product matched."
        ),
    )
    matched_product: str | None = Field(
        default=None,
        description="Actual product selected as the exact or closest analog match.",
    )
    product_similarity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "4096-bit Morgan-fingerprint Tanimoto similarity between expected_product and "
            "matched_product. Exact matches are reported as 1.0."
        ),
    )
    status: Literal["passed", "failed"]
    failure_mode: (
        Literal[
            "invalid_template",
            "no_products",
            "wrong_product",
            "invalid_reactant_smiles",
            "invalid_product_smiles",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Null on success. One of: "
            "'invalid_template' — SMARTS could not be parsed; "
            "'no_products' — template produced no products; "
            "'wrong_product' — product did not match expected; "
            "'invalid_reactant_smiles' — a reactant SMILES is unparseable; "
            "'invalid_product_smiles' — the expected product SMILES is unparseable."
        ),
    )
    suggested_fix: Literal["fix_smarts", "fix_template", "fix_smiles"] | None = Field(
        default=None,
        description=(
            "Tool to call to fix this step. "
            "'fix_smarts' if failure_mode is 'invalid_template' (SMARTS failed to parse). "
            "'fix_template' if failure_mode is 'no_products' or 'wrong_product' (SMARTS parsed but gave wrong result). "
            "'fix_smiles' if failure_mode is 'invalid_reactant_smiles' or 'invalid_product_smiles'. "
            "Null if the step passed."
        ),
    )


class ValidationReport(BaseModel):
    reactions: list[ReactionResult]
    building_blocks: list[BuildingBlockResult]

    target_molecule: str
    target_sa_score: float | None = Field(
        default=None,
        description=(
            "SA score of the target molecule (1 = easy, 10 = very hard to synthesize). "
            "Scores above 6 suggest poor synthesizability — include 'fix_target' in suggested_fixes."
        ),
    )
    all_building_blocks_valid: bool = Field(
        description="True only if every building block passed SMILES validation."
    )
    all_reactions_passed: bool = Field(
        description="True only if every reaction step produced the expected product."
    )
    suggested_fixes: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered action list for the user. Each entry names the fix tool and what to pass. "
            "Examples: "
            "'fix_template: step 1 — reactants [...], product [...]'; "
            "'fix_smiles: building block Cc1csc(N)n1 (likely truncated)'; "
            "'fix_building_block: O=C(Nc1ccc(O)cc1)c1cccc(C(F)(F)F)c1 (too complex)'; "
            "'fix_target: SA score 7.2 — find synthesizable analogue'. "
            "Empty list if everything passed."
        ),
    )
