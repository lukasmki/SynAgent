from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset

from synagent.scoring._toolset import ScoringToolset


@dataclass
class Scoring(AbstractCapability[None]):
    id = "scoring"
    description = "Use for scoring or ranking synthesis paths, reactions, or molecules."
    defer_loading = True

    def get_instructions(self) -> str:
        return "Validate all SMILES strings and then each reaction step."

    def get_toolset(self) -> AgentToolset[None]:
        return ScoringToolset()
