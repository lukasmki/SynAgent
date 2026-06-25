from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.toolsets import AgentToolset

from synagent.chemspace._toolset import ChemspaceToolset


@dataclass
class Chemspace(AbstractCapability[None]):
    id = "chemspace"
    description = "Search Chemspace for building block price and availability."
    defer_loading = True

    api_key: str | None = None
    "Chemspace API key"

    def get_instructions(self) -> str:
        return (
            "Search Chemspace for building block availability and pricing. "
            "Use search_exact first; fall back to search_similarity if no results are found."
        )

    def get_toolset(self) -> AgentToolset[None]:
        return ChemspaceToolset(self.api_key)
