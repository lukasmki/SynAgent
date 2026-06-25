"""
Tool call functions for the three fine-tuned LLMs in the SynAgent pipeline:

1. SmileyLlama  — de novo molecule generation with property constraints
2. SynLlama     — retrosynthetic pathway prediction
3. LinkLlama    — fragment linker design with geometry/property constraints

All three models are served via a vLLM-compatible OpenAI endpoint.
Set the base URL with the VLLM_BASE_URL env var (default: http://localhost:8000/v1).
"""

from __future__ import annotations

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import SynLlamaFormat


# ---------------------------------------------------------------------------
# vLLM / OpenAI-compatible client
# ---------------------------------------------------------------------------

# All three LLMs (SmileyLlama, SynLlama, LinkLlama) are served behind a single
# vLLM server exposing an OpenAI-compatible /v1/completions endpoint.
# The model selection happens per-request via the `model` parameter.
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")  # vLLM default; no real auth needed locally

# Shared client instance — reused across all tool calls to avoid connection overhead.
# Tests mock this object via monkeypatch on `synagent.llm_tools._client`.
_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SmileyLlama  —  de novo molecule generation
# ═══════════════════════════════════════════════════════════════════════════
#
# Model : THGLab/Llama-3.1-8B-SmileyLlama-1.1  (8B params)
# Paper : https://arxiv.org/abs/2409.02231
#
# Fine-tuned Llama-3.1-8B-Instruct on ~2M ChEMBL SMILES.
# Generates a single SMILES string satisfying user-specified
# pharmaceutical property constraints.
# ═══════════════════════════════════════════════════════════════════════════

SMILEYLLAMA_MODEL = os.getenv(
    "SMILEYLLAMA_MODEL", "THGLab/Llama-3.1-8B-SmileyLlama-1.1"
)

SMILEYLLAMA_SYSTEM = (
    "You love and excel at generating SMILES strings of drug-like molecules."
)


class SmileyLlamaInput(BaseModel):
    """Property constraints for de novo molecule generation."""

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
    num_samples: int = Field(
        default=1,
        ge=1,
        le=50,
        description="Number of molecules to generate.",
    )
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)


def _build_smileyllama_prompt(params: SmileyLlamaInput) -> str:
    """Construct the user-facing prompt string from property constraints.

    Matches the training format from the SmileyLlama paper: properties are
    listed as a comma-separated natural language string after the instruction.
    If no constraints are provided, uses the unconditional generation prompt.
    """
    props: list[str] = []
    if params.mw_range:
        props.append(f"molecular weight {params.mw_range}")
    if params.logp_range:
        props.append(f"LogP {params.logp_range}")
    if params.hbd_range:
        props.append(f"hydrogen-bond donors {params.hbd_range}")
    if params.hba_range:
        props.append(f"hydrogen-bond acceptors {params.hba_range}")
    if params.rotatable_bonds:
        props.append(f"rotatable bonds {params.rotatable_bonds}")
    if params.fsp3:
        props.append(f"Fsp3 {params.fsp3}")
    if params.macrocycle is True:
        props.append("containing a macrocycle")
    elif params.macrocycle is False:
        props.append("without macrocycles")

    if props:
        constraint_str = ", ".join(props)
        return (
            f"Output a SMILES string for a drug like molecule with the "
            f"following properties: {constraint_str}"
        )
    return "Output a SMILES string for a drug like molecule:"


def smileyllama_generate(params: SmileyLlamaInput) -> list[dict]:
    """Generate drug-like molecules using SmileyLlama.

    SmileyLlama is an 8B-parameter LLM (fine-tuned Llama-3.1-8B-Instruct)
    trained on ~2M ChEMBL molecules. Given pharmaceutical property
    constraints (MW, LogP, HBD, HBA, rotatable bonds, Fsp3, macrocycle),
    it generates valid SMILES strings of novel drug-like molecules.

    Use this tool when you need to propose new candidate molecules that
    satisfy specific drug-likeness criteria. Each call can generate
    multiple diverse candidates by setting num_samples > 1.

    Args:
        params: SmileyLlamaInput with property constraints and sampling config.

    Returns:
        List of dicts, each with 'smiles' (str) and 'raw_output' (str).
    """
    prompt = _build_smileyllama_prompt(params)
    results = []

    for _ in range(params.num_samples):
        # Use the Llama-3.1 chat template format for the prompt.
        # vLLM's /v1/completions endpoint accepts raw text prompts,
        # so we manually construct the special tokens.
        response = _client.completions.create(
            model=SMILEYLLAMA_MODEL,
            prompt=(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{SMILEYLLAMA_SYSTEM}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            max_tokens=256,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=["<|eot_id|>"],  # Stop at end-of-turn token
        )
        raw = response.choices[0].text.strip()
        # Model sometimes outputs extra text after the SMILES; take only the first token
        results.append({"smiles": raw.split()[0] if raw else "", "raw_output": raw})

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SynLlama  —  retrosynthetic pathway prediction
# ═══════════════════════════════════════════════════════════════════════════
#
# Model : SynLlama 1B  (fine-tuned on 2M reactions, 91 templates)
# Decomposes a target molecule into purchasable building blocks via
# validated retrosynthetic disconnections.
#
# Output is a JSON with reaction steps (SMARTS templates) and building
# blocks (SMILES).  If parsing fails, raw text is returned with a
# parse_error flag.
# ═══════════════════════════════════════════════════════════════════════════

SYNLLAMA_MODEL = os.getenv("SYNLLAMA_MODEL", "SynLlama-1B")

SYNLLAMA_SYSTEM_PROMPT = (
    "### Instruction:\n"
    "You are an expert synthetic organic chemist. Your task is to "
    "design a synthesis pathway for a given target molecule using "
    "common and reliable reaction templates and building blocks. "
    "Follow these instructions:\n\n"
    "1. **Input the SMILES String:** Read in the SMILES string of "
    "the target molecule and identify common reaction templates "
    "that can be applied.\n\n"
    "2. **Decompose the Target Molecule:** Use the identified "
    "reaction templates to decompose the target molecule into "
    "different intermediates.\n\n"
    "3. **Check for Building Blocks:** For each intermediate:\n"
    "    - Identify if it is a building block. If it is, wrap it "
    "in <bb> and </bb> tags and save it for later use.\n"
    "    - If it is not a building block, apply additional reaction "
    "templates to further decompose it into building blocks.\n\n"
    "4. **Document Reactions:** For each reaction documented in the "
    "output, wrap the reaction template in <rxn> and </rxn> tags.\n\n"
    "5. **Repeat the Process:** Continue this process until all "
    "intermediates are decomposed into building blocks, and document "
    "each step clearly in a structured JSON format.\n\n"
)


class SynLlamaInput(BaseModel):
    """Input for retrosynthetic pathway prediction."""

    product_smiles: str = Field(
        description="SMILES string of the target molecule to decompose. "
        "Must be a valid, canonicalized SMILES string."
    )
    num_pathways: int = Field(
        default=1,
        ge=1,
        le=20,
        description="Number of diverse candidate pathways to generate.",
    )
    temperature: float = Field(
        default=1.5,
        description="High temperature (1.5) for diversity sampling.",
    )
    top_p: float = Field(
        default=0.9,
        description="Top-p for nucleus sampling (0.9 for diversity).",
    )


def synllama_retrosynthesis(params: SynLlamaInput) -> dict:
    """Generate a retrosynthetic pathway for a target molecule using SynLlama.

    SynLlama is a 1B-parameter LLM fine-tuned on 2M reactions across 91
    reaction templates. It decomposes a target molecule into purchasable
    building blocks via validated retrosynthetic disconnections.

    Call this tool when you have a candidate SMILES string and need to
    determine how to actually synthesize it. The model returns a structured
    pathway containing reaction templates (SMARTS wrapped in <rxn> tags)
    and building blocks (SMILES wrapped in <bb> tags).

    For diversity sampling, this tool is called multiple times with high
    temperature (T=1.5) and top-P (0.9) to generate different candidate
    pathways. After collecting N diverse pathways, filter downstream for
    computational validity.

    Args:
        params: SynLlamaInput with target SMILES and sampling config.

    Returns:
        Dict with 'product' (str), 'pathway' (parsed JSON or raw text),
        and 'parse_error' (bool).
    """
    prompt = (
        f"{SYNLLAMA_SYSTEM_PROMPT}"
        f"### Input:\n"
        f"Provide a synthetic pathway for this SMILES string: "
        f"{params.product_smiles}\n\n"
        f"### Response:\n"
    )

    # Generate multiple diverse pathways using high temperature sampling.
    # Each call may produce a different retrosynthetic disconnection.
    pathways = []
    for _ in range(params.num_pathways):
        output = _client.completions.create(
            model=SYNLLAMA_MODEL,
            prompt=prompt,
            max_tokens=256,
            temperature=params.temperature,
            top_p=params.top_p,
            # Stop tokens prevent the model from generating a new instruction/input pair
            stop=["### Input:", "### Instruction:"],
        )
        raw_text = output.choices[0].text

        # SynLlama outputs JSON, but sometimes wraps it in markdown code fences.
        # Strip those before parsing. On parse failure, preserve raw text for debugging.
        try:
            clean = raw_text.strip().strip("```json").strip("```").strip()
            result = json.loads(clean)
            pathways.append({"pathway": result, "parse_error": False})
        except json.JSONDecodeError:
            pathways.append({"raw_output": raw_text, "parse_error": True})

    return {
        "product": params.product_smiles,
        "pathways": pathways,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3.  LinkLlama  —  fragment linker design
# ═══════════════════════════════════════════════════════════════════════════
#
# Model : THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50  (1B params)
# Paper : https://www.biorxiv.org/content/10.64898/2026.04.15.718690v1
# Repo  : https://github.com/THGLab/LinkLlama
#
# Fine-tuned Llama-3.2-1B-Instruct to propose linker molecules between
# two molecular fragments given geometric constraints (distance, angle)
# and optional property constraints.  Returns JSON with linker SMILES
# and reasonability reasoning.
# ═══════════════════════════════════════════════════════════════════════════

LINKLLAMA_MODEL = os.getenv(
    "LINKLLAMA_MODEL", "THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50"
)

LINKLLAMA_SYSTEM = (
    "You are an expert medicinal chemist specializing in linker design. "
    "Your task is to design linkers to connect molecular fragments and "
    "assess chemical reasonability. Output your answer in JSON format."
)


class LinkLlamaInput(BaseModel):
    """Input for fragment linker design."""

    fragment1_smiles: str = Field(
        description="SMILES of the first fragment (with dummy atom [*] at attachment point)."
    )
    fragment2_smiles: str = Field(
        description="SMILES of the second fragment (with dummy atom [*] at attachment point)."
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
        default=None,
        description="Rotatable bonds constraint, e.g. '>= 2'.",
    )
    heavy_atoms_range: str | None = Field(
        default=None,
        description="Heavy atom count constraint, e.g. '>= 5'.",
    )
    hbd_range: str | None = Field(
        default=None,
        description="H-bond donors constraint for full molecule, e.g. '<= 5'.",
    )
    hba_range: str | None = Field(
        default=None,
        description="H-bond acceptors constraint for full molecule, e.g. '<= 10'.",
    )
    mw_range: str | None = Field(
        default=None,
        description="Molecular weight constraint for full molecule, e.g. '<= 500'.",
    )
    logp_range: str | None = Field(
        default=None,
        description="LogP constraint for full molecule, e.g. '<= 5'.",
    )
    tpsa_range: str | None = Field(
        default=None,
        description="TPSA constraint for full molecule, e.g. '<= 140'.",
    )
    reasonability: Literal["reasonable", "unreasonable"] = Field(
        default="reasonable",
        description="Whether to target chemically reasonable linkers.",
    )
    num_samples: int = Field(default=10, ge=1, le=100)
    temperature: float = Field(default=1.4)
    top_p: float = Field(default=0.99)


def _build_linkllama_prompt(params: LinkLlamaInput) -> str:
    """Construct the LinkLlama prompt from fragments, geometry, and property constraints.

    Follows the training prompt format from LinkLlama's sft_corpus.py:
    fragment info → linker properties → molecule properties → reasonability.
    """
    fragment_info = (
        f"Fragment 1 (SMILES: {params.fragment1_smiles}) and "
        f"Fragment 2 (SMILES: {params.fragment2_smiles}). "
        f"The distance between the attachment points is "
        f"{params.distance_angstrom:.2f} Angstroms, "
        f"and the angle between them is {params.angle_degrees:.2f} degrees."
    )

    linker_props: list[str] = []
    if params.linker_type:
        linker_props.append(f"Linker type: {params.linker_type}")
    if params.rotb_range:
        linker_props.append(f"Rotatable bonds: {params.rotb_range}")
    if params.heavy_atoms_range:
        linker_props.append(f"Heavy atoms: {params.heavy_atoms_range}")

    mol_props: list[str] = []
    if params.hbd_range:
        mol_props.append(f"H-bond donors {params.hbd_range}")
    if params.hba_range:
        mol_props.append(f"H-bond acceptors {params.hba_range}")
    if params.mw_range:
        mol_props.append(f"Molecular weight {params.mw_range}")
    if params.logp_range:
        mol_props.append(f"LogP {params.logp_range}")
    if params.tpsa_range:
        mol_props.append(f"TPSA {params.tpsa_range}")

    sections = [fragment_info]
    if linker_props:
        sections.append("Linker properties: " + "; ".join(linker_props) + ".")
    if mol_props:
        sections.append(
            "Desired molecule properties: " + "; ".join(mol_props) + "."
        )
    sections.append(f"Reasonability: {params.reasonability}.")

    return "\n".join(sections)


def linkllama_generate(params: LinkLlamaInput) -> dict:
    """Design linker molecules between two fragments using LinkLlama.

    LinkLlama is a 1B-parameter LLM (fine-tuned Llama-3.2-1B-Instruct)
    that proposes chemically reasonable linkers connecting two molecular
    fragments given geometric constraints (distance in Angstroms, angle
    in degrees) and optional property constraints (MW, LogP, TPSA, HBD,
    HBA, rotatable bonds, heavy atoms, linker topology).

    Use this tool when you have two molecular fragments from a
    structure-based drug design workflow and need to propose linker
    molecules that connect them with appropriate geometry and
    drug-like properties.

    Each generated sample is a JSON with 'linker' (SMILES) and
    'reasoning' (chemical rationale). Samples that fail JSON parsing
    are returned with a parse_error flag.

    Args:
        params: LinkLlamaInput with fragments, geometry, and constraints.

    Returns:
        Dict with 'fragments', 'geometry', and 'samples' (list of results).
    """
    user_prompt = _build_linkllama_prompt(params)
    samples = []

    for _ in range(params.num_samples):
        # LinkLlama uses the same Llama-3.2 chat template as SmileyLlama
        response = _client.completions.create(
            model=LINKLLAMA_MODEL,
            prompt=(
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{LINKLLAMA_SYSTEM}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            ),
            max_tokens=512,
            temperature=params.temperature,
            top_p=params.top_p,
            stop=["<|eot_id|>"],
        )
        raw = response.choices[0].text.strip()

        # LinkLlama returns JSON: {"linker": "<SMILES>", "reasoning": "<text>"}
        # Strip markdown fences and parse; preserve raw output on failure
        try:
            clean = raw.strip("```json").strip("```").strip()
            parsed = json.loads(clean)
            samples.append({
                "linker": parsed.get("linker", ""),
                "reasoning": parsed.get("reasoning", ""),
                "parse_error": False,
            })
        except json.JSONDecodeError:
            samples.append({"raw_output": raw, "parse_error": True})

    return {
        "fragments": [params.fragment1_smiles, params.fragment2_smiles],
        "geometry": {
            "distance_angstrom": params.distance_angstrom,
            "angle_degrees": params.angle_degrees,
        },
        "samples": samples,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Register all three as pydantic-ai agent tools
# ═══════════════════════════════════════════════════════════════════════════


def register_llm_tools(agent: Agent) -> None:
    """Register SmileyLlama, SynLlama, and LinkLlama as tools on a pydantic-ai agent.

    Each tool is registered via @agent.tool_plain (no RunContext deps needed
    since the vLLM client is a module-level singleton). The tool functions
    are thin wrappers that validate input via Pydantic then delegate to the
    underlying *_generate / *_retrosynthesis functions above.
    """

    @agent.tool_plain
    def generate_molecules(
        mw_range: str | None = None,
        logp_range: str | None = None,
        hbd_range: str | None = None,
        hba_range: str | None = None,
        rotatable_bonds: str | None = None,
        fsp3: str | None = None,
        macrocycle: bool | None = None,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[dict]:
        """Generate novel drug-like molecules with SmileyLlama.

        SmileyLlama is an 8B-parameter LLM fine-tuned on ~2M ChEMBL
        molecules. Given pharmaceutical property constraints it generates
        valid SMILES strings of novel drug-like molecules.

        Args:
            mw_range: Molecular weight range, e.g. '<= 500'.
            logp_range: LogP range, e.g. '<= 5'.
            hbd_range: H-bond donor count, e.g. '<= 5'.
            hba_range: H-bond acceptor count, e.g. '<= 10'.
            rotatable_bonds: Rotatable bonds, e.g. '<= 10'.
            fsp3: Fraction sp3 carbons, e.g. '>= 0.25'.
            macrocycle: Whether to include a macrocycle.
            num_samples: Number of molecules (1-50).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        params = SmileyLlamaInput(
            mw_range=mw_range,
            logp_range=logp_range,
            hbd_range=hbd_range,
            hba_range=hba_range,
            rotatable_bonds=rotatable_bonds,
            fsp3=fsp3,
            macrocycle=macrocycle,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
        )
        return smileyllama_generate(params)

    @agent.tool_plain
    def retrosynthesis(
        product_smiles: str,
        num_pathways: int = 1,
        temperature: float = 1.5,
        top_p: float = 0.9,
    ) -> dict:
        """Generate a retrosynthetic pathway for a target molecule using SynLlama.

        SynLlama is a 1B-parameter LLM fine-tuned on 2M reactions across
        91 reaction templates. It decomposes a target molecule into
        purchasable building blocks via validated retrosynthetic
        disconnections.

        Call this tool when you have a candidate SMILES string and need to
        determine how to actually synthesize it. For diversity, set
        num_pathways > 1.

        Args:
            product_smiles: SMILES of the target molecule.
            num_pathways: Number of diverse pathways to generate (1-20).
            temperature: Sampling temperature (1.5 for diversity).
            top_p: Nucleus sampling (0.9 for diversity).
        """
        params = SynLlamaInput(
            product_smiles=product_smiles,
            num_pathways=num_pathways,
            temperature=temperature,
            top_p=top_p,
        )
        return synllama_retrosynthesis(params)

    @agent.tool_plain
    def design_linker(
        fragment1_smiles: str,
        fragment2_smiles: str,
        distance_angstrom: float,
        angle_degrees: float,
        linker_type: str | None = None,
        rotb_range: str | None = None,
        heavy_atoms_range: str | None = None,
        hbd_range: str | None = None,
        hba_range: str | None = None,
        mw_range: str | None = None,
        logp_range: str | None = None,
        tpsa_range: str | None = None,
        reasonability: str = "reasonable",
        num_samples: int = 10,
        temperature: float = 1.4,
        top_p: float = 0.99,
    ) -> dict:
        """Design linker molecules between two fragments using LinkLlama.

        LinkLlama is a 1B-parameter LLM that proposes chemically
        reasonable linkers connecting two molecular fragments given
        geometric constraints (distance, angle) and optional property
        constraints.

        Use this when you have two fragments from structure-based drug
        design and need linker proposals with appropriate geometry.

        Args:
            fragment1_smiles: SMILES of fragment 1 (with [*] attachment).
            fragment2_smiles: SMILES of fragment 2 (with [*] attachment).
            distance_angstrom: Distance between attachment points (Å).
            angle_degrees: Angle between attachment points (degrees).
            linker_type: 'chain', 'branched', or 'ring-containing'.
            rotb_range: Rotatable bonds constraint.
            heavy_atoms_range: Heavy atom count constraint.
            hbd_range: H-bond donors constraint.
            hba_range: H-bond acceptors constraint.
            mw_range: Molecular weight constraint.
            logp_range: LogP constraint.
            tpsa_range: TPSA constraint.
            reasonability: 'reasonable' or 'unreasonable'.
            num_samples: Number of linkers to generate (1-100).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        params = LinkLlamaInput(
            fragment1_smiles=fragment1_smiles,
            fragment2_smiles=fragment2_smiles,
            distance_angstrom=distance_angstrom,
            angle_degrees=angle_degrees,
            linker_type=linker_type,
            rotb_range=rotb_range,
            heavy_atoms_range=heavy_atoms_range,
            hbd_range=hbd_range,
            hba_range=hba_range,
            mw_range=mw_range,
            logp_range=logp_range,
            tpsa_range=tpsa_range,
            reasonability=reasonability,
            num_samples=num_samples,
            temperature=temperature,
            top_p=top_p,
        )
        return linkllama_generate(params)
