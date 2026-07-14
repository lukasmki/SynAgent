from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.corrector._toolset import CorrectorToolset


@dataclass
class Corrector(AbstractCapability[AgentDepsT]):
    id = "corrector"
    description = "Diagnose and correct invalid retrosynthetic routes, design new routes, and look up literature."
    defer_loading = True

    def get_instructions(self) -> str:
        return (
            "When the user sends a synthesis route, call correct_route immediately "
            "with the entire message content as-is. Do not analyse or paraphrase it first. "
            "Then report exactly what the tool returned — status, fixed steps, and any errors — verbatim. "
            "If correct_route returns all_resolved: False or chain_consistent: False, call design_route next. "
            "Never write a report without calling a tool first."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return CorrectorToolset()
