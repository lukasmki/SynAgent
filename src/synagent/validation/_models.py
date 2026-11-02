from html import escape
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    import polars as pl


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

    # Plain properties and methods, not fields: the JSON schema the model sees
    # stays the same.

    @property
    def passed(self) -> bool:
        """Every building block is valid and every step formed its product.

        Checks the per-step results as well as the summary flags, which the model
        fills in and can get wrong.
        """
        return (
            self.all_building_blocks_valid
            and self.all_reactions_passed
            and all(b.is_valid for b in self.building_blocks)
            and not self.failed_steps
        )

    @property
    def failed_steps(self) -> list[ReactionResult]:
        return [r for r in self.reactions if r.status == "failed"]

    def to_polars(self) -> "pl.DataFrame":
        """One row per reaction step."""
        import polars as pl

        return pl.DataFrame([r.model_dump() for r in self.reactions])

    def _repr_html_(self) -> str:
        steps_passed = len(self.reactions) - len(self.failed_steps)
        blocks_valid = sum(b.is_valid for b in self.building_blocks)
        summary = (
            f"<p><b>Route {'PASSED' if self.passed else 'FAILED'}</b>: "
            f"{steps_passed}/{len(self.reactions)} steps passed, "
            f"{blocks_valid}/{len(self.building_blocks)} building blocks valid</p>"
        )
        rows = "".join(
            "<tr>"
            + "".join(
                f"<td>{escape(str(cell))}</td>"
                for cell in (
                    r.reaction_number,
                    r.reaction_template,
                    " + ".join(r.reactant_smiles),
                    r.expected_product,
                    ", ".join(r.actual_products) or "-",
                    r.status if r.failure_mode is None else r.failure_mode,
                )
            )
            + "</tr>"
            for r in self.reactions
        )
        header = "".join(
            f"<th>{h}</th>"
            for h in ("step", "template", "reactants", "expected", "actual", "status")
        )
        return f"{summary}<table><tr>{header}</tr>{rows}</table>"
