"""Capability definition for the model_calls toolset.

Follows the pydantic-ai v2 AbstractCapability pattern: the agent discovers
this capability by its id and description, lazily loads the toolset, and
uses the instructions to guide tool selection.
"""

from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.model_calls._toolset import ModelCallsToolset


@dataclass
class ModelCalls(AbstractCapability[AgentDepsT]):
    id = "model-calls"
    description = (
        "Use for molecule generation (SmileyLlama), retrosynthetic pathway "
        "prediction (SynLlama), fragment linker design (LinkLlama), Enamine "
        "REAL database search, and composite fragment-linking workflows."
    )
    defer_loading = True

    def get_instructions(self) -> str:
        return (
            "You have access to three fine-tuned chemistry LLMs and Enamine search:\n\n"
            "1. **generate_molecules** (SmileyLlama, 8B) — Generate novel drug-like "
            "SMILES with property constraints (MW, LogP, HBD, HBA, rotatable bonds, Fsp3, macrocycle).\n\n"
            "2. **retrosynthesis** (SynLlama, 1B) — Decompose a target SMILES into "
            "retrosynthetic pathways with reaction templates (SMARTS) and building blocks.\n\n"
            "3. **design_linker** (LinkLlama, 1B) — Propose linker molecules between "
            "two fragments given geometry (distance, angle) and property constraints.\n\n"
            "4. **search_enamine_similarity** / **search_enamine_substructure** — Find "
            "purchasable molecules in the Enamine REAL database by similarity or substructure.\n\n"
            "5. **find_and_link_fragments** — Composite workflow: search Enamine for "
            "purchasable fragment analogs, then run LinkLlama to design linkers between them.\n\n"
            "Standard pipeline: generate_molecules → retrosynthesis → validation → "
            "Enamine search + design_linker → pricing.\n"
            "Challenge the user's tool choices before executing — not every task needs every step."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return ModelCallsToolset()
