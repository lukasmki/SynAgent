from typing import Any

from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT


class ScoringToolset(FunctionToolset[AgentDepsT]):
    """Toolset for scoring synthesis paths, reactions, and molecules."""

    def __init__(self):
        super().__init__()
        self.include_return_schema = True
        self.metadata = {"code_mode": True}
        self.add_function(self.score_molecules, name="score_molecules")
        self.add_function(self.score_reactions, name="score_reactions")
        self.add_function(self.score_paths, name="score_paths")

    async def score_molecules(self, smiles: list[str]) -> dict[str, float]:
        """Scores the molecules

        Args:
            smiles (list[str]): Input SMILES

        Returns:
            dict[str, float]: {smiles: score}
        """
        return {}

    async def score_reactions(self, reaction_smarts: list[str]) -> dict[str, float]:
        """Scores the reactions

        Args:
            reaction_smarts (list[str]): Input SMARTS

        Returns:
            dict[str, float]: {reaction_smarts: score}
        """
        return {}

    async def score_paths(
        self, path: list[dict[str, Any]]
    ) -> list[dict[str, dict[str, float]]]:
        """Scores the synthesis paths

        Args:
            path (list[SynLlamaFormat]): Input path(s)

        Returns:
            list[float]: scores
        """
        return []
