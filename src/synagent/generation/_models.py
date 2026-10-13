"""Pydantic models for the three fine-tuned LLMs."""

from typing import Literal

from pydantic import BaseModel, Field

# ── SmileyLlama ──────────────────────────────────────────────────────────


class SmileyLlamaInput(BaseModel):
    """Property constraints for de novo molecule generation.

    All constraint fields are optional; omit them for unconstrained generation.
    """

    mw_range: str | None = Field(
        default=None,
        description="Molecular weight range, e.g. '<= 500', '300-500', '> 700'.",
    )
    logp_range: str | None = Field(
        default=None,
        description="LogP (lipophilicity) range, e.g. '<= 5', '1-3'.",
    )
    hbd_range: str | None = Field(
        default=None,
        description="Hydrogen-bond donor count range, e.g. '<= 5'.",
    )
    hba_range: str | None = Field(
        default=None,
        description="Hydrogen-bond acceptor count range, e.g. '<= 10'.",
    )
    rotatable_bonds: str | None = Field(
        default=None,
        description="Rotatable bonds range, e.g. '<= 10'.",
    )
    fsp3: str | None = Field(
        default=None,
        description="Fraction of sp3 carbons, e.g. '>= 0.25'.",
    )
    macrocycle: bool | None = Field(
        default=None,
        description="Whether the molecule should contain a macrocycle.",
    )
    num_samples: int = Field(default=1, ge=1, le=50)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)

    def build_prompt(self) -> str:
        """Construct the user prompt matching SmileyLlama's training format.

        Property phrasing is "<comparison> <name>" — e.g. "<= 500 Molecular
        weight" — with the comparison FIRST. This is the order the model card
        documents and the order the model was trained on; the reverse
        ("molecular weight <= 500", which this used to emit) is off-distribution
        and measurably weakens constraint adherence.

        The trailing colon after the property list is also part of the trained
        format, not a typo.
        """
        props = [
            f"{value} {label}"
            for value, label in (
                (self.mw_range, "Molecular weight"),
                (self.logp_range, "logP"),
                (self.hbd_range, "H-bond donors"),
                (self.hba_range, "H-bond acceptors"),
                (self.rotatable_bonds, "Rotatable bonds"),
                (self.fsp3, "Fraction sp3"),
            )
            if value
        ]
        if self.macrocycle is True:
            props.append("a macrocycle")
        elif self.macrocycle is False:
            props.append("no macrocycles")
        if props:
            return (
                "Output a SMILES string for a drug like molecule with the "
                f"following properties: {', '.join(props)}:"
            )
        return "Output a SMILES string for a drug like molecule:"


class MoleculeResult(BaseModel):
    """A single generated molecule from SmileyLlama."""

    smiles: str = Field(description="Generated SMILES string.")
    raw_output: str = Field(description="Full raw model output.")


# ── SynLlama ─────────────────────────────────────────────────────────────


class SynLlamaInput(BaseModel):
    """Target molecule for retrosynthetic pathway prediction.

    High temperature and top-p produce diverse candidate pathways for
    downstream filtering.
    """

    target_smiles: str = Field(
        description="SMILES string of the target molecule to decompose."
    )
    num_pathways: int = Field(default=1, ge=1, le=20)
    temperature: float = Field(default=1.5)
    top_p: float = Field(default=0.9)

    def build_prompt(self) -> str:
        """Construct the user prompt matching SynLlama's training format."""
        return (
            f"Provide a synthetic pathway for this SMILES string: {self.target_smiles}"
        )


class PathwayResult(BaseModel):
    """A single retrosynthetic pathway from SynLlama."""

    pathway: dict | None = Field(
        default=None,
        description="Parsed JSON pathway with reactions and building blocks.",
    )
    raw_output: str | None = Field(
        default=None,
        description="Raw model output, present when JSON parsing fails.",
    )
    parse_error: bool = Field(
        description="True if the model output could not be parsed as JSON."
    )


class RetrosynthesisResult(BaseModel):
    """Result of retrosynthetic analysis."""

    product: str
    pathways: list[PathwayResult]


# ── LinkLlama ────────────────────────────────────────────────────────────


class LinkLlamaInput(BaseModel):
    """Two fragments, their attachment geometry, and optional linker constraints."""

    fragment1_smiles: str = Field(
        description="SMILES of fragment 1 (with [*] dummy atom at attachment point)."
    )
    fragment2_smiles: str = Field(
        description="SMILES of fragment 2 (with [*] dummy atom at attachment point)."
    )
    distance_angstrom: float = Field(
        description="Distance between attachment points in Angstroms."
    )
    angle_degrees: float = Field(
        description="Angle between attachment points in degrees."
    )
    linker_type: Literal["chain", "branched", "ring-containing"] | None = Field(
        default=None,
        description="Desired linker topology.",
    )
    rotb_range: str | None = Field(
        default=None, description="Rotatable bonds constraint."
    )
    heavy_atoms_range: str | None = Field(
        default=None, description="Heavy atom count constraint."
    )
    hbd_range: str | None = Field(default=None, description="H-bond donors constraint.")
    hba_range: str | None = Field(
        default=None, description="H-bond acceptors constraint."
    )
    mw_range: str | None = Field(
        default=None, description="Molecular weight constraint."
    )
    logp_range: str | None = Field(default=None, description="LogP constraint.")
    tpsa_range: str | None = Field(default=None, description="TPSA constraint.")
    reasonability: Literal["reasonable", "unreasonable"] = Field(default="reasonable")
    num_samples: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=1.4)
    top_p: float = Field(default=0.99)

    def build_prompt(self) -> str:
        """Construct the LinkLlama prompt from fragments, geometry, and constraints."""
        fragment_info = (
            f"Fragment 1 (SMILES: {self.fragment1_smiles}) and "
            f"Fragment 2 (SMILES: {self.fragment2_smiles}). "
            "The distance between the attachment points is "
            f"{self.distance_angstrom:.2f} Angstroms, "
            f"and the angle between them is {self.angle_degrees:.2f} degrees."
        )
        linker_props = [
            f"{label}: {value}"
            for label, value in (
                ("Linker type", self.linker_type),
                ("Rotatable bonds", self.rotb_range),
                ("Heavy atoms", self.heavy_atoms_range),
            )
            if value
        ]
        mol_props = [
            f"{label} {value}"
            for label, value in (
                ("H-bond donors", self.hbd_range),
                ("H-bond acceptors", self.hba_range),
                ("Molecular weight", self.mw_range),
                ("LogP", self.logp_range),
                ("TPSA", self.tpsa_range),
            )
            if value
        ]
        sections = [fragment_info]
        if linker_props:
            sections.append("Linker properties: " + "; ".join(linker_props) + ".")
        if mol_props:
            sections.append(
                "Desired molecule properties: " + "; ".join(mol_props) + "."
            )
        sections.append(f"Reasonability: {self.reasonability}.")
        return "\n".join(sections)


class LinkerSample(BaseModel):
    """A single linker proposal from LinkLlama."""

    linker: str = Field(default="", description="Linker SMILES.")
    reasoning: str = Field(default="", description="Chemical rationale from the model.")
    raw_output: str | None = Field(
        default=None, description="Raw text if JSON parsing failed."
    )
    parse_error: bool = False


class LinkerResult(BaseModel):
    """Result of linker design."""

    fragments: list[str]
    geometry: dict
    samples: list[LinkerSample]
