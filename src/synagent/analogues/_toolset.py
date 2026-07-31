from pathlib import Path

from FPSim2.FPSim2 import FPSim2Engine
from FPSim2.FPSim2Reactions import ReactionEngine
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem
from rdkit.Chem import rdChemReactions


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
            if not smi or Chem.MolFromSmiles(smi) is None:
                result[smi] = [f"ERROR: invalid SMILES '{smi}' — cannot search"]
                continue
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
            try:
                rxn = rdChemReactions.ReactionFromSmarts(temp)
                if rxn is None:
                    result[temp] = [f"ERROR: invalid reaction SMARTS '{temp}'"]
                    continue
            except Exception as e:
                result[temp] = [f"ERROR: {e}"]
                continue
            try:
                hits = self.rxn_engine.similarity(
                    temp,
                    threshold,
                    metric="cosine",
                    n_workers=self.n_workers,
                    mol_format=None,
                )
                rxns = self.rxn_engine.get_strings(hits)
                result[temp] = rxns[:max_results]
            except Exception as e:
                result[temp] = [f"ERROR: fingerprint failed for '{temp}': {e}"]
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
        reaction_template = reaction_template.replace("\\u003e", ">")
        if ">>" not in reaction_template:
            return {"error": f"Invalid reaction template — missing '>>': {reaction_template}"}
        rside, pside = reaction_template.split(">>", 1)
        reactants, products = rside.split("."), pside.split(".")
        result = {}
        for query in reactants + products:
            if not query.strip():
                continue
            mol = Chem.MolFromSmarts(query)
            if mol is None:
                result[query] = [f"ERROR: invalid SMARTS fragment '{query}'"]
                continue
            hits = self.mol_engine.substructure(
                query,
                n_workers=self.n_workers,
                mol_format="smarts",
            )
            result[query] = self.mol_engine.get_strings(hits)[:max_results]
        return result
