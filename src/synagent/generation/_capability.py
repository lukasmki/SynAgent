"""Capability definition for the generation toolset.

Follows the pydantic-ai v2 AbstractCapability pattern: the agent discovers
this capability by its id and description, lazily loads the toolset, and
uses the instructions to guide tool selection.
"""

import os
from dataclasses import dataclass, field

import httpx
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.generation._toolset import GenerationToolset


def _vllm_url(port: int) -> str:
    """VLLM_BASE_URL with its port replaced, for the servers in vllm/config.yaml."""
    base = httpx.URL(os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"))
    return str(base.copy_with(port=port))


@dataclass
class Generation(AbstractCapability[AgentDepsT]):
    id: str = field(default="generation", kw_only=True)
    description: str = field(
        default=(
            "Use for generating novel drug-like molecules, proposing "
            "retrosynthetic pathways for a target, or designing linkers "
            "between two fragments."
        ),
        kw_only=True,
    )
    defer_loading: bool = field(default=True, kw_only=True)

    api_key: str = "EMPTY"
    "vLLM API key. Must be non-empty, or the OpenAI client falls back to OPENAI_API_KEY"

    smileyllama_url: str = field(default_factory=lambda: _vllm_url(8001))
    "Base URL of the vLLM server serving SmileyLlama"

    smileyllama_model: str = "kysun63/smileyllama-1b-reproduced"
    "Served model name for SmileyLlama"

    synllama_url: str = field(default_factory=lambda: _vllm_url(8002))
    "Base URL of the vLLM server serving SynLlama"

    synllama_model: str = "kysun63/synllama-1b-reproduced"
    "Served model name for SynLlama"

    linkllama_url: str = field(default_factory=lambda: _vllm_url(8003))
    "Base URL of the vLLM server serving LinkLlama"

    linkllama_model: str = "kysun63/linkllama-1b-reproduced"
    "Served model name for LinkLlama"

    def get_instructions(self) -> str:
        return (
            "These tools sample from generative models. Treat every result as a "
            "candidate, not a verified answer:\n"
            "- SMILES may be invalid and property constraints are not guaranteed "
            "to hold. Pathways may contain invalid reaction steps.\n"
            "- Validate SMILES and reaction steps with the synthesis-validation "
            "capability before reporting results or building on them.\n"
            "- Entries with parse_error=True carry no usable result. Discard them "
            "or sample again; do not hand-repair raw_output.\n\n"
            "Usage:\n"
            "- Request several samples in one call (num_samples / num_pathways) "
            "and filter, rather than calling repeatedly for one sample.\n"
            "- The sampling defaults are tuned per tool. Change temperature or "
            "top_p only to trade diversity against conservatism.\n"
            "- Write constraints as a comparison or range, e.g. '<= 500' or "
            "'300-500'.\n"
            "- design_linker needs a [*] dummy atom at each fragment's attachment "
            "point, plus the distance and angle between the attachment points. "
            "If the user has not given the geometry, ask rather than guess.\n\n"
            "Use only the tools the task needs; not every request calls for "
            "generation, retrosynthesis, and linker design together."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return GenerationToolset(
            api_key=self.api_key,
            smileyllama_url=self.smileyllama_url,
            smileyllama_model=self.smileyllama_model,
            synllama_url=self.synllama_url,
            synllama_model=self.synllama_model,
            linkllama_url=self.linkllama_url,
            linkllama_model=self.linkllama_model,
        )
