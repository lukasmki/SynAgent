from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT

from synagent.models import SynLlamaFormat


class ScoringToolset(FunctionToolset[AgentDepsT]):
    """Toolset for scoring synthesis paths, reactions, and molecules."""

    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.score_molecules, name="score_molecules")
        self.add_function(self.score_reactions, name="score_reactions")
        self.add_function(self.score_path, name="score_path")

    def score_molecules(self, smiles: list[str]) -> dict[str, float]:
        return {}

    def score_reactions(self, reaction_smarts: list[str]) -> dict[str, float]:
        return {}

    def score_path(self, path: SynLlamaFormat) -> float:
        return 0.0
