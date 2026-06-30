from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.scoring._toolset import ScoringToolset


@dataclass
class Scoring(AbstractCapability[AgentDepsT]):
    id = "scoring"
    description = "Use for scoring or ranking synthesis paths, reactions, or molecules."
    defer_loading = True

    def get_instructions(self) -> str:
        return (
            "The provided tools compute properties that can be used for ranking."
            "Composite scores should be calculated through code."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return ScoringToolset()
