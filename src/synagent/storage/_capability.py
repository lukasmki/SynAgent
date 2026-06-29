from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.storage._toolset import _DEFAULT_PATH, StorageToolset


@dataclass
class Storage(AbstractCapability[AgentDepsT]):
    id = "storage"
    description = "Use for persisting and retrieving synthesis records."
    defer_loading = True

    path: Path = field(default=_DEFAULT_PATH)

    def get_instructions(self) -> str:
        return "Use storage tools to save and retrieve synthesis records as needed."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return StorageToolset(path=self.path)
