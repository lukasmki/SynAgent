"""Pydantic models for the three fine-tuned LLMs and Enamine search."""

from typing import Literal

from pydantic import BaseModel, Field


# ── SmileyLlama ──────────────────────────────────────────────────────────


class SmileyLlamaInput(BaseModel):
    """Property constraints for de novo molecule generation via SmileyLlama (8B).

    SmileyLlama generates novel drug-like SMILES strings satisfying
    pharmaceutical property constraints. All constraint fields are optional;
    omitting them produces an unconditional generation prompt.
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


class MoleculeResult(BaseModel):
    """A single generated molecule from SmileyLlama."""

    smiles: str = Field(description="Generated SMILES string.")
    raw_output: str = Field(description="Full raw model output.")


# ── SynLlama ─────────────────────────────────────────────────────────────


class SynLlamaInput(BaseModel):
    """Input for retrosynthetic pathway prediction via SynLlama (1B).

    SynLlama decomposes a target molecule into purchasable building blocks
    and reaction templates (SMARTS). High temperature (1.5) and top-p (0.9)
    produce diverse candidate pathways for downstream filtering.
    """

    product_smiles: str = Field(
        description="SMILES string of the target molecule to decompose."
    )
    num_pathways: int = Field(default=1, ge=1, le=20)
    temperature: float = Field(default=1.5)
    top_p: float = Field(default=0.9)


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
    """Input for fragment linker design via LinkLlama (1B).

    LinkLlama proposes chemically reasonable linkers connecting two molecular
    fragments given geometric constraints (distance, angle) and optional
    property constraints.
    """

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
    rotb_range: str | None = Field(default=None, description="Rotatable bonds constraint.")
    heavy_atoms_range: str | None = Field(default=None, description="Heavy atom count constraint.")
    hbd_range: str | None = Field(default=None, description="H-bond donors constraint.")
    hba_range: str | None = Field(default=None, description="H-bond acceptors constraint.")
    mw_range: str | None = Field(default=None, description="Molecular weight constraint.")
    logp_range: str | None = Field(default=None, description="LogP constraint.")
    tpsa_range: str | None = Field(default=None, description="TPSA constraint.")
    reasonability: Literal["reasonable", "unreasonable"] = Field(default="reasonable")
    num_samples: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=1.4)
    top_p: float = Field(default=0.99)


class LinkerSample(BaseModel):
    """A single linker proposal from LinkLlama."""

    linker: str = Field(default="", description="Linker SMILES.")
    reasoning: str = Field(default="", description="Chemical rationale from the model.")
    raw_output: str | None = Field(default=None, description="Raw text if JSON parsing failed.")
    parse_error: bool = False


class LinkerResult(BaseModel):
    """Result of linker design."""

    fragments: list[str]
    geometry: dict
    samples: list[LinkerSample]


# ── Enamine search ───────────────────────────────────────────────────────


class EnamineSearchInput(BaseModel):
    """Input for Enamine REAL database search."""

    smiles: str = Field(description="Query SMILES string.")
    search_type: Literal["similarity", "substructure"] = Field(default="similarity")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_results: int = Field(default=10, ge=1, le=100)


class EnamineHit(BaseModel):
    """A single hit from Enamine search."""

    smiles: str
    enamine_id: str = ""
    tanimoto_score: float | None = None
    availability: str = "unknown"
    price_info: str | None = None
    source: str = Field(description="'enamine_api' or 'local_cache'.")


# ── Fragment linker workflow ─────────────────────────────────────────────


class LinkerProposal(BaseModel):
    """A linker proposal with purchasability metadata."""

    linker_smiles: str
    reasoning: str = ""
    fragment1: str
    fragment2: str
    fragment1_tanimoto: float | None = None
    fragment2_tanimoto: float | None = None
    fragment1_enamine_id: str = ""
    fragment2_enamine_id: str = ""
    purchasable: bool = False


class FragmentLinkerWorkflowResult(BaseModel):
    """Result of the composite Enamine → LinkLlama workflow."""

    purchasable_fragments: dict
    linker_proposals: list[LinkerProposal]
    summary: dict
