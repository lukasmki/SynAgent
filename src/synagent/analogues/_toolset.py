from pathlib import Path

from FPSim2.FPSim2 import FPSim2Engine
from FPSim2.FPSim2Reactions import ReactionEngine
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT


class AnalogueSearchToolset(FunctionToolset[AgentDepsT]):
    """Toolset for building block and reaction template search."""

    include_return_schema = True

    def __init__(
        self,
        moldb: Path | None = None,
        rxndb: Path | None = None,
        in_memory: bool = True,
        n_workers: int = 4,
    ):
        super().__init__()
        root = Path(__file__).parent
        self.moldb = moldb or root / "data/building_blocks.h5"
        self.rxndb = rxndb or root / "data/reactions.h5"
        self.mol_engine = FPSim2Engine(str(self.moldb), in_memory_fps=in_memory)
        self.rxn_engine = ReactionEngine(str(self.rxndb), in_memory_fps=in_memory)
        self.n_workers = n_workers

        self.add_function(self.search_building_blocks, name="search_building_blocks")
        self.add_function(self.search_templates, name="search_templates")
        self.add_function(
            self.search_building_blocks_by_template,
            name="search_building_blocks_by_template",
        )

    async def search_building_blocks(
        self,
        smiles: list[str],
        threshold: float = 0.7,
        max_results: int = 100,
    ) -> dict[str, list[str]]:
        """Finds similar building blocks. May contain false positives.

        Args:
            smiles (list[str]): List of SMILES to search
            threshold (float): Cosine similarity threshold
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, list[str]]: {query: search results}
        """
        result = {}
        for smi in smiles:
            hits = self.mol_engine.similarity(
                smi,
                threshold,
                metric="cosine",
                n_workers=self.n_workers,
                mol_format="smiles",
            )
            blocks = self.mol_engine.get_strings(hits)
            result[smi] = blocks[:max_results]
        return result

    async def search_templates(
        self,
        reaction_templates: list[str],
        threshold: float = 0.7,
        max_results: int = 100,
    ) -> dict[str, list[str]]:
        """Finds similar reaction templates. May contain false positives.

        Args:
            reaction_templates (list[str]): List of reaction templates to search
            threshold (float): Cosine similarity threshold
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, list[str]]: {query: search results}
        """
        result = {}
        for temp in reaction_templates:
            hits = self.rxn_engine.similarity(
                temp,
                threshold,
                metric="cosine",
                n_workers=self.n_workers,
                mol_format=None,
            )
            rxns = self.mol_engine.get_strings(hits)
            result[temp] = rxns[:max_results]
        return result

    async def search_building_blocks_by_template(
        self,
        reaction_template: str,
        max_results: int = 100,
    ) -> dict[str, list[str]]:
        """Finds building blocks that fit the reaction templat substructures. May contain false positives.

        Args:
            reaction_template (str): Reaction template to search
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, str]: {query: search results}
        """
        rside, pside = reaction_template.split(">>")
        reactants, products = rside.split("."), pside.split(".")
        result = {}
        for query in reactants + products:
            hits = self.mol_engine.substructure(
                query,
                n_workers=self.n_workers,
                mol_format="smarts",
            )
            rxns = self.mol_engine.get_strings(hits)
            result[query] = rxns[:max_results]
        return result
