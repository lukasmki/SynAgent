from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.analogues._toolset import AnalogueSearchToolset


@dataclass
class AnalogueSearch(AbstractCapability[AgentDepsT]):
    id: str = field(default="analogue-search", kw_only=True)
    description: str = field(
        default="Use for finding reaction templates and building blocks.",
        kw_only=True,
    )
    defer_loading: bool = field(default=True, kw_only=True)

    def get_instructions(self) -> str:
        return "Use the analogue searching capability to find building blocks that fit reaction templates."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return AnalogueSearchToolset()
