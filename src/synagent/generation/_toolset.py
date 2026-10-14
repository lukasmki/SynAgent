"""Toolset for the three fine-tuned LLMs (SmileyLlama, SynLlama, LinkLlama).

Each model is served by its own vLLM server exposing an OpenAI-compatible endpoint.
"""

from __future__ import annotations

from openai import AsyncOpenAI
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT

from synagent.generation._models import (
    LinkLlamaInput,
    SmileyLlamaInput,
    SynLlamaInput,
)

# ---------------------------------------------------------------------------
# System prompts (matching each model's training format)
# ---------------------------------------------------------------------------
# Verbatim from the model card — no trailing period, which is how it was trained.
SMILEYLLAMA_SYSTEM = (
    "You love and excel at generating SMILES strings of drug-like molecules"
)

SYNLLAMA_SYSTEM = (
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

LINKLLAMA_SYSTEM = (
    "You are an expert medicinal chemist specializing in linker design. "
    "Your task is to design linkers to connect molecular fragments and "
    "assess chemical reasonability. Output your answer in JSON format."
)


class GenerationToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing SmileyLlama, SynLlama, and LinkLlama as agent tools."""

    include_return_schema = True

    def __init__(
        self,
        *,
        api_key: str,
        smileyllama_url: str,
        smileyllama_model: str,
        synllama_url: str,
        synllama_model: str,
        linkllama_url: str,
        linkllama_model: str,
    ):
        super().__init__()
        # vLLM clients — one per model, since each model has its own server
        self._smileyllama_client = AsyncOpenAI(
            base_url=smileyllama_url, api_key=api_key
        )
        self._synllama_client = AsyncOpenAI(base_url=synllama_url, api_key=api_key)
        self._linkllama_client = AsyncOpenAI(base_url=linkllama_url, api_key=api_key)
        self._smileyllama_model = smileyllama_model
        self._synllama_model = synllama_model
        self._linkllama_model = linkllama_model
        self.add_function(self.generate_molecules, name="generate_molecules")
        self.add_function(self.retrosynthesis, name="retrosynthesis")
        self.add_function(self.design_linker, name="design_linker")

    async def _complete(
        self, client: AsyncOpenAI, model: str, prompt: str, n: int, **kwargs
    ) -> list[str]:
        """Draw ``n`` independent samples for one prompt in a single request."""
        response = await client.completions.create(
            model=model, prompt=prompt, n=n, **kwargs
        )
        return [choice.text for choice in response.choices]

    # ═══════════════════════════════════════════════════════════════════════
    # SmileyLlama — de novo molecule generation
    # ═══════════════════════════════════════════════════════════════════════

    async def generate_molecules(self, request: SmileyLlamaInput) -> list[dict]:
        """Generate novel drug-like molecules satisfying property constraints.

        Args:
            request (SmileyLlamaInput): Property constraints and sampling settings.

        Returns:
            list[dict]: One entry per sample with ``smiles`` (the generated
            SMILES, not guaranteed valid) and ``raw_output`` (the full text).
        """
        outputs = await self._complete(
            self._smileyllama_client,
            self._smileyllama_model,
            f"### Instruction:\n{SMILEYLLAMA_SYSTEM}\n\n"
            f"### Input:\n{request.build_prompt()}\n\n"
            f"### Response:\n",
            request.num_samples,
            max_tokens=1024,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["### Instruction:", "### Input:"],
        )
        results = []
        for text in outputs:
            raw = text.strip()
            # Model may output extra text after SMILES; take first token only
            results.append({"smiles": raw.split()[0] if raw else "", "raw_output": raw})
        return results

    # ═══════════════════════════════════════════════════════════════════════
    # SynLlama — retrosynthetic pathway prediction
    # ═══════════════════════════════════════════════════════════════════════

    async def retrosynthesis(self, request: SynLlamaInput) -> dict:
        """Decompose a target molecule into building blocks via reaction templates.

        For diverse disconnections, set num_pathways > 1 and keep temperature high.

        Args:
            request (SynLlamaInput): Target molecule and sampling settings.

        Returns:
            dict: ``target`` (the input SMILES) and ``pathways``, one entry per
            sample. Each entry has only ``raw_output``, the unparsed model text.
            It is expected to be a JSON pathway (reaction steps with SMARTS
            templates, reactant and product SMILES, and building blocks), but
            it is not validated and may be truncated or malformed.
        """
        outputs = await self._complete(
            self._synllama_client,
            self._synllama_model,
            f"{SYNLLAMA_SYSTEM}### Input:\n{request.build_prompt()}\n\n### Response:\n",
            request.num_pathways,
            max_tokens=2048,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["### Input:", "### Instruction:"],
        )
        pathways = []
        for raw_text in outputs:
            pathways.append({"raw_output": raw_text})
        return {"target": request.target_smiles, "pathways": pathways}

    # ═══════════════════════════════════════════════════════════════════════
    # LinkLlama — fragment linker design
    # ═══════════════════════════════════════════════════════════════════════

    async def design_linker(self, request: LinkLlamaInput) -> dict:
        """Propose linkers connecting two fragments under geometric constraints.

        Args:
            request (LinkLlamaInput): Fragments, attachment geometry, optional
                property constraints, and sampling settings.

        Returns:
            dict: ``fragments`` (the two input SMILES), ``geometry`` (the input
            ``distance_angstrom`` and ``angle_degrees``) and ``samples``, one
            entry per sample. Each entry has only ``raw_output``, the unparsed
            model text. It is expected to be JSON with a ``linker`` SMILES and
            a ``reasoning`` string, but it is not validated and may be
            truncated or malformed.
        """
        outputs = await self._complete(
            self._linkllama_client,
            self._linkllama_model,
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{LINKLLAMA_SYSTEM}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{request.build_prompt()}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n",
            request.num_samples,
            max_tokens=1024,
            temperature=request.temperature,
            top_p=request.top_p,
            stop=["<|eot_id|>"],
        )
        samples = []
        for text in outputs:
            # LinkLlama returns {"linker": "<SMILES>", "reasoning": "<text>"}
            samples.append({"raw_output": text.strip()})

        return {
            "fragments": [request.fragment1_smiles, request.fragment2_smiles],
            "geometry": {
                "distance_angstrom": request.distance_angstrom,
                "angle_degrees": request.angle_degrees,
            },
            "samples": samples,
        }
