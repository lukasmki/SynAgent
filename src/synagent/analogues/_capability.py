from pydantic_ai import AgentToolset
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from synagent.analogues._toolset import AnalogueSearchToolset


class AnalogueSearch(AbstractCapability[AgentDepsT]):
    id = "analogue-search"
    description = "Use for finding reaction templates and building blocks."
    defer_loading = True

    def get_instructions(self) -> str:
        return "Use the analogue searching capability to find building blocks that fit reaction templates."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return AnalogueSearchToolset()
