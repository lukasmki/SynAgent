import os
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.chemspace._toolset import ChemspaceToolset

UNAVAILABLE = "Unavailable: CHEMSPACE_API_KEY is not set."


@dataclass
class Chemspace(AbstractCapability[AgentDepsT]):
    id: str = field(default="chemspace", kw_only=True)
    description: str = field(
        default="Search Chemspace for building block price and availability.",
        kw_only=True,
    )
    defer_loading: bool = field(default=True, kw_only=True)

    api_key: str | None = None
    "Chemspace API key; defaults to CHEMSPACE_API_KEY. Without one, there are no tools."

    def _key(self) -> str | None:
        return self.api_key or os.getenv("CHEMSPACE_API_KEY")

    def get_description(self) -> str:
        return self.description if self._key() else UNAVAILABLE

    def get_instructions(self) -> str:
        if not self._key():
            return (
                "Chemspace pricing and availability lookups are unavailable because "
                "no CHEMSPACE_API_KEY is set. If the user asks for pricing, say so."
            )
        return (
            "Only use the Chemspace tools if the user requested pricing information. "
            "Search Chemspace for building block availability and pricing. "
            "Use search_exact first; fall back to search_similarity if no results are found. "
            "If building blocks are not found in-stock, check if they can be made on-demand."
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        key = self._key()
        return ChemspaceToolset(key) if key else None
