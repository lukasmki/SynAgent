from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT

from synagent.chemspace.chemspace import (
    ChemspaceAPI,
    ChemspaceRequest,
    ChemspaceResponse,
)


class ChemspaceToolset(FunctionToolset[AgentDepsT]):
    """Toolset for Chemspace building block search."""

    include_return_schema = False

    def __init__(self, api_key: str | None = None):
        super().__init__()
        self.chemspace = ChemspaceAPI(api_key)
        self.add_function(self.search_exact, name="search_exact")
        self.add_function(self.search_similarity, name="search_similarity")
        self.add_function(self.search_substructure, name="search_substructure")

    async def search_exact(self, search: ChemspaceRequest) -> ChemspaceResponse:
        """Exact search by SMILES string

        Args:
            search (ChemspaceRequest): Search parameters

        Returns:
            ChemspaceResponse: Search results
        """
        return await self.chemspace.search("exact", search)

    async def search_similarity(self, search: ChemspaceRequest) -> ChemspaceResponse:
        """Similarity search by SMILES structure

        Args:
            search (ChemspaceRequest): Search parameters

        Returns:
            ChemspaceResponse: Search results
        """
        return await self.chemspace.search("sim", search)

    async def search_substructure(self, search: ChemspaceRequest) -> ChemspaceResponse:
        """Search by SMILES sub-structure

        Args:
            search (ChemspaceRequest): Search parameters

        Returns:
            ChemspaceResponse: Search results
        """
        return await self.chemspace.search("sub", search)
