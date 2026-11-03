from pathlib import Path
from typing import Literal

from FPSim2.FPSim2 import FPSim2Engine
from FPSim2.FPSim2Reactions import ReactionEngine
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synagent.validation.rdkit_logging import capture_rdkit_log

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
    ) -> dict[str, list[str] | str]:
        """Finds similar building blocks. May contain false positives.

        Args:
            smiles (list[str]): List of SMILES to search
            threshold (float): Cosine similarity threshold
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, list[str] | str]: {query: search results | error}
        """
        result = {}
        for smi in smiles:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmiles(smi)
            if mol is None:
                result[smi] = log.annotate("Invalid SMILES.")
                continue
            hits = self.mol_engine.similarity(
                mol, threshold, metric="cosine", n_workers=self.n_workers
            )
            blocks = self.mol_engine.get_strings(hits)
            result[smi] = blocks[:max_results]
        return result

    async def search_templates(
        self,
        reaction_templates: list[str],
        threshold: float = 0.7,
        max_results: int = 100,
    ) -> dict[str, list[str] | str]:
        """Finds similar reaction templates. May contain false positives.

        Args:
            reaction_templates (list[str]): List of reaction templates to search
            threshold (float): Cosine similarity threshold
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, list[str] | str]: {query: search results | error}
        """
        result = {}
        for temp in reaction_templates:
            with capture_rdkit_log() as log:
                try:
                    rxn = rdChemReactions.ReactionFromSmarts(temp)
                except ValueError as e:
                    result[temp] = log.annotate(str(e))
                    continue
            hits = self.rxn_engine.similarity(
                rxn, threshold, metric="cosine", n_workers=self.n_workers
            )
            rxns = self.rxn_engine.get_strings(hits)
            result[temp] = rxns[:max_results]
        return result

    async def check_building_blocks(self, smiles: list[str]) -> dict[str, bool | str]:
        """Checks whether each molecule is in the building block library (exact match).

        Args:
            smiles (list[str]): Plain SMILES strings without tags. Pass every
                building block of a route in a single call.
                Example: ["CC(=O)O", "CN"]

        Returns:
            dict[str, bool | str]: {smiles: in_library | error}
        """
        result = {}
        for smi in smiles:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmiles(smi)
            if mol is None:
                result[smi] = log.annotate("Invalid SMILES.")
                continue
            canon = Chem.MolToSmiles(mol)
            # Identical molecules have identical fingerprints, so exact matches
            # are among the near-1.0 hits; compare those by canonical SMILES.
            hits = self.mol_engine.similarity(
                mol, 0.99, metric="cosine", n_workers=self.n_workers
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
    ) -> dict[str, list[str] | str]:
        """Finds building blocks that fit the reaction template substructures. May contain false positives.

        Args:
            reaction_template (str): Reaction template to search
            max_results (int): Maximum number of returned results per query

        Returns:
            dict[str, list[str] | str]: {query: search results | error}
        """
        rside, sep, pside = reaction_template.partition(">>")
        if not sep:
            return {
                reaction_template: "Expected a template of the form reactants>>products."
            }
        reactants, products = rside.split("."), pside.split(".")
        result = {}
        for query in reactants + products:
            with capture_rdkit_log() as log:
                mol = Chem.MolFromSmarts(query)
            if mol is None:
                result[query] = log.annotate("Invalid SMARTS.")
                continue
            hits = self.mol_engine.substructure(mol, n_workers=self.n_workers)
            rxns = self.mol_engine.get_strings(hits)
            result[query] = rxns[:max_results]
        return result
