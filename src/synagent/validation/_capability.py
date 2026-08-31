from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.validation._toolset import SynthesisValidationToolset


@dataclass
class SynthesisValidation(AbstractCapability[AgentDepsT]):
    id: str = field(default="synthesis-validation", kw_only=True)
    description: str = field(default="Use for synthesis path validation.", kw_only=True)
    defer_loading: bool = field(default=True, kw_only=True)

    def get_instructions(self) -> str:
        return (
            "Validate all SMILES strings and then each reaction step. "
            "When finished, create a validation report."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return SynthesisValidationToolset()
