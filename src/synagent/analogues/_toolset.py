from pathlib import Path
from typing import Literal

from FPSim2.FPSim2 import FPSim2Engine
from FPSim2.FPSim2Reactions import ReactionEngine
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_MOLDB = DATA_DIR / "building_blocks.h5"
DEFAULT_RXNDB = DATA_DIR / "reactions.h5"


def _canonical(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


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
        self.moldb = moldb or DEFAULT_MOLDB
        self.rxndb = rxndb or DEFAULT_RXNDB
        self.mol_engine = FPSim2Engine(str(self.moldb), in_memory_fps=in_memory)
        self.rxn_engine = ReactionEngine(str(self.rxndb), in_memory_fps=in_memory)
        self.n_workers = n_workers
        self.templates = set(self.rxn_engine.get_strings(self.rxn_engine.fps[:, 0]))

        self.add_function(self.search_building_blocks, name="search_building_blocks")
        self.add_function(self.search_templates, name="search_templates")
        self.add_function(
            self.search_building_blocks_by_template,
            name="search_building_blocks_by_template",
        )
        self.add_function(self.check_building_blocks, name="check_building_blocks")
        self.add_function(self.check_templates, name="check_templates")

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
            rxns = self.rxn_engine.get_strings(hits)
            result[temp] = rxns[:max_results]
        return result

    async def check_building_blocks(self, smiles: list[str]) -> dict[str, bool]:
        """Checks whether each molecule is in the building block library (exact match).

        Args:
            smiles (list[str]): Plain SMILES strings without tags. Pass every
                building block of a route in a single call.
                Example: ["CC(=O)O", "CN"]

        Returns:
            dict[str, bool]: {smiles: in_library}. Invalid SMILES are False.
        """
        result = {}
        for smi in smiles:
            canon = _canonical(smi)
            if canon is None:
                result[smi] = False
                continue
            # Identical molecules have identical fingerprints, so exact matches
            # are among the near-1.0 hits; compare those by canonical SMILES.
            hits = self.mol_engine.similarity(
                smi,
                0.99,
                metric="cosine",
                n_workers=self.n_workers,
                mol_format="smiles",
            )
            candidates = self.mol_engine.get_strings(hits)
            result[smi] = any(_canonical(c) == canon for c in candidates)
        return result

    async def check_templates(
        self, reaction_templates: list[str]
    ) -> dict[str, Literal["in_library", "reversed", "not_in_library"]]:
        """Checks whether each reaction template is in the template library.

        Args:
            reaction_templates (list[str]): Bare reaction SMARTS strings, without
                <rxn> tags. Pass every template of a route in a single call.
                Example: ["[OH][$([CX4]),c:1]>>[Cl][#6:1]"]

        Returns:
            dict[str, str]: {template: status}. "reversed" means the library
                holds the template with reactants and products swapped.
        """
        result = {}
        for temp in reaction_templates:
            temp = temp.strip()
            if temp in self.templates:
                result[temp] = "in_library"
            elif ">>".join(temp.split(">>")[::-1]) in self.templates:
                result[temp] = "reversed"
            else:
                result[temp] = "not_in_library"
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
