from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.validation._toolset import SynthesisValidationToolset


@dataclass
class SynthesisValidation(AbstractCapability[AgentDepsT]):
    id = "synthesis-validation"
    description = "Use for synthesis path validation."
    defer_loading = True

    def get_instructions(self) -> str:
        return (
            "Validate all SMILES strings and then each reaction step. "
            "When finished, create a validation report."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return SynthesisValidationToolset()
