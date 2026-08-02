from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.retrosynthesis._toolset import RetrosynthesisToolset


@dataclass
class Retrosynthesis(AbstractCapability[AgentDepsT]):
    id: str | None = "retrosynthesis"
    description: str | None = "Design new synthetic routes to a target molecule using retrosynthetic analysis."
    defer_loading: bool = False

    def get_instructions(self) -> str:
        return (
            "When the user asks to redesign a route, find an alternative synthesis path, or run "
            "retrosynthetic analysis: call retro_search() with target_smiles='from_report' — "
            "the tool reads the target molecule from the last ValidationReport automatically. "
            "Each returned route lists the steps in forward (synthesis) order with templates "
            "and building blocks. Report the shortest route first."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return RetrosynthesisToolset()
