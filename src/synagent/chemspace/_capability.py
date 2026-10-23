from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.chemspace._toolset import ChemspaceToolset


@dataclass
class Chemspace(AbstractCapability[AgentDepsT]):
    id: str = field(default="chemspace", kw_only=True)
    description: str = field(
        default="Search Chemspace for building block price and availability.",
        kw_only=True,
    )
    defer_loading: bool = field(default=True, kw_only=True)

    api_key: str | None = None
    "Chemspace API key"

    def get_instructions(self) -> str:
        return (
            "Only use the Chemspace tools if the user requested pricing information. "
            "Search Chemspace for building block availability and pricing. "
            "Use search_exact first; fall back to search_similarity if no results are found. "
            "If building blocks are not found in-stock, check if they can be made on-demand."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return ChemspaceToolset(self.api_key)
