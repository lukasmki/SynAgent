"""Toolset for the three fine-tuned LLMs (SmileyLlama, SynLlama, LinkLlama),
Enamine REAL database search, and the composite fragment linker workflow.

All LLM calls go through a vLLM server exposing an OpenAI-compatible endpoint.
Enamine search uses the REST API with local RDKit fingerprint fallback.
"""

from __future__ import annotations

import json
import os
from itertools import product as cartesian
from pathlib import Path

import httpx
from openai import OpenAI
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from synagent.model_calls._models import (
    EnamineHit,
    LinkerProposal,
)

# ---------------------------------------------------------------------------
# vLLM client — shared across all LLM tool calls
# ---------------------------------------------------------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
_client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

# Model IDs — override via env vars if serving under different names
SMILEYLLAMA_MODEL = os.getenv("SMILEYLLAMA_MODEL", "THGLab/Llama-3.1-8B-SmileyLlama-1.1")
SYNLLAMA_MODEL = os.getenv("SYNLLAMA_MODEL", "SynLlama-1B")
LINKLLAMA_MODEL = os.getenv("LINKLLAMA_MODEL", "THGLab/Llama-3.2-1B-Instruct-LinkLlama-Cap50")

# Enamine config
ENAMINE_API_KEY = os.getenv("ENAMINE_API_KEY", "")
ENAMINE_BASE_URL = os.getenv("ENAMINE_BASE_URL", "https://api.enamine.net/api/v1")
LOCAL_FRAGMENTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "enamine_fragments.csv"

# ---------------------------------------------------------------------------
# System prompts (matching each model's training format)
# ---------------------------------------------------------------------------
# Verbatim from the model card — no trailing period, which is how it was trained.
SMILEYLLAMA_SYSTEM = "You love and excel at generating SMILES strings of drug-like molecules"

SYNLLAMA_SYSTEM = (
    "### Instruction:\n"
    "You are an expert synthetic organic chemist. Your task is to "
    "design a synthesis pathway for a given target molecule using "
    "common and reliable reaction templates and building blocks. "
    "Follow these instructions:\n\n"
    "1. **Input the SMILES String:** Read in the SMILES string of "
    "the target molecule and identify common reaction templates "
    "that can be applied.\n\n"
    "2. **Decompose the Target Molecule:** Use the identified "
    "reaction templates to decompose the target molecule into "
    "different intermediates.\n\n"
    "3. **Check for Building Blocks:** For each intermediate:\n"
    "    - Identify if it is a building block. If it is, wrap it "
    "in <bb> and </bb> tags and save it for later use.\n"
    "    - If it is not a building block, apply additional reaction "
    "templates to further decompose it into building blocks.\n\n"
    "4. **Document Reactions:** For each reaction documented in the "
    "output, wrap the reaction template in <rxn> and </rxn> tags.\n\n"
    "5. **Repeat the Process:** Continue this process until all "
    "intermediates are decomposed into building blocks, and document "
    "each step clearly in a structured JSON format.\n\n"
)

LINKLLAMA_SYSTEM = (
    "You are an expert medicinal chemist specializing in linker design. "
    "Your task is to design linkers to connect molecular fragments and "
    "assess chemical reasonability. Output your answer in JSON format."
)


class ModelCallsToolset(FunctionToolset[AgentDepsT]):
    """Toolset providing SmileyLlama, SynLlama, LinkLlama, Enamine search,
    and the composite fragment linker workflow as agent tools.
    """

    include_return_schema = True

    def __init__(self):
        super().__init__()
        # LLM tools
        self.add_function(self.generate_molecules, name="generate_molecules")
        self.add_function(self.retrosynthesis, name="retrosynthesis")
        self.add_function(self.design_linker, name="design_linker")
        # Enamine search tools
        self.add_function(self.search_enamine_similarity, name="search_enamine_similarity")
        self.add_function(self.search_enamine_substructure, name="search_enamine_substructure")
        # Composite workflow
        self.add_function(self.find_and_link_fragments, name="find_and_link_fragments")

    # ═══════════════════════════════════════════════════════════════════════
    # SmileyLlama — de novo molecule generation (8B params)
    # ═══════════════════════════════════════════════════════════════════════

    async def generate_molecules(
        self,
        mw_range: str | None = None,
        logp_range: str | None = None,
        hbd_range: str | None = None,
        hba_range: str | None = None,
        rotatable_bonds: str | None = None,
        fsp3: str | None = None,
        macrocycle: bool | None = None,
        num_samples: int = 1,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> list[dict]:
        """Generate novel drug-like molecules using SmileyLlama.

        SmileyLlama is an 8B-parameter LLM fine-tuned on ~2M ChEMBL molecules.
        Given pharmaceutical property constraints it generates valid SMILES
        strings of novel drug-like molecules.

        Args:
            mw_range: Molecular weight range, e.g. '<= 500'.
            logp_range: LogP (lipophilicity) range, e.g. '<= 5'.
            hbd_range: Hydrogen-bond donor count, e.g. '<= 5'.
            hba_range: Hydrogen-bond acceptor count, e.g. '<= 10'.
            rotatable_bonds: Rotatable bonds range, e.g. '<= 10'.
            fsp3: Fraction of sp3 carbons, e.g. '>= 0.25'.
            macrocycle: Whether to include a macrocycle.
            num_samples: Number of molecules to generate (1-50).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        # Build the prompt from property constraints
        prompt = self._build_smileyllama_prompt(
            mw_range, logp_range, hbd_range, hba_range,
            rotatable_bonds, fsp3, macrocycle,
        )
        results = []
        for _ in range(num_samples):
            # Alpaca-style instruction format, per the SmileyLlama model card.
            #
            # This previously wrapped the prompt in the Llama-3.1 chat template
            # (<|begin_of_text|><|start_header_id|>...). That is the format the
            # official 8B THGLab/Llama-3.1-8B-SmileyLlama-1.1 expects, but the
            # 1B checkpoints are trained on the alpaca layout below. Sending
            # chat tokens to them yields degraded, often non-SMILES output.
            response = _client.completions.create(
                model=SMILEYLLAMA_MODEL,
                prompt=(
                    f"### Instruction:\n{SMILEYLLAMA_SYSTEM}\n\n"
                    f"### Input:\n{prompt}\n\n"
                    f"### Response:\n"
                ),
                max_tokens=256,
                temperature=temperature,
                top_p=top_p,
                stop=["### Instruction:", "### Input:"],
            )
            raw = response.choices[0].text.strip()
            # Model may output extra text after SMILES; take first token only
            results.append({"smiles": raw.split()[0] if raw else "", "raw_output": raw})
        return results

    @staticmethod
    def _build_smileyllama_prompt(
        mw_range, logp_range, hbd_range, hba_range,
        rotatable_bonds, fsp3, macrocycle,
    ) -> str:
        """Construct the user prompt matching SmileyLlama's training format.

        Property phrasing is "<comparison> <name>" — e.g. "<= 500 Molecular
        weight" — with the comparison FIRST. This is the order the model card
        documents and the order the model was trained on; the reverse
        ("molecular weight <= 500", which this used to emit) is off-distribution
        and measurably weakens constraint adherence.

        The trailing colon after the property list is also part of the trained
        format, not a typo.
        """
        props: list[str] = []
        if mw_range:
            props.append(f"{mw_range} Molecular weight")
        if logp_range:
            props.append(f"{logp_range} logP")
        if hbd_range:
            props.append(f"{hbd_range} H-bond donors")
        if hba_range:
            props.append(f"{hba_range} H-bond acceptors")
        if rotatable_bonds:
            props.append(f"{rotatable_bonds} Rotatable bonds")
        if fsp3:
            props.append(f"{fsp3} Fraction sp3")
        if macrocycle is True:
            props.append("a macrocycle")
        elif macrocycle is False:
            props.append("no macrocycles")
        if props:
            return (
                "Output a SMILES string for a drug like molecule with the "
                f"following properties: {', '.join(props)}:"
            )
        return "Output a SMILES string for a drug like molecule:"

    # ═══════════════════════════════════════════════════════════════════════
    # SynLlama — retrosynthetic pathway prediction (1B params)
    # ═══════════════════════════════════════════════════════════════════════

    async def retrosynthesis(
        self,
        product_smiles: str,
        num_pathways: int = 1,
        temperature: float = 1.5,
        top_p: float = 0.9,
    ) -> dict:
        """Generate retrosynthetic pathways for a target molecule using SynLlama.

        SynLlama is a 1B-parameter LLM fine-tuned on 2M reactions across 91
        reaction templates. It decomposes a target molecule into purchasable
        building blocks via validated retrosynthetic disconnections.

        For diversity, set num_pathways > 1. High temperature (1.5) produces
        different retrosynthetic disconnections each call.

        Args:
            product_smiles: SMILES of the target molecule.
            num_pathways: Number of diverse pathways to generate (1-20).
            temperature: Sampling temperature (1.5 for diversity).
            top_p: Nucleus sampling threshold (0.9 for diversity).
        """
        prompt = (
            f"{SYNLLAMA_SYSTEM}"
            f"### Input:\n"
            f"Provide a synthetic pathway for this SMILES string: "
            f"{product_smiles}\n\n"
            f"### Response:\n"
        )
        pathways = []
        for _ in range(num_pathways):
            output = _client.completions.create(
                model=SYNLLAMA_MODEL,
                # A full route is a JSON object with one entry per reaction
                # step, each carrying a SMARTS template plus reactant and
                # product SMILES. 256 tokens truncates that mid-object, so the
                # JSON never closes and every response lands in the
                # parse_error branch. Measured: a two-step aspirin route runs
                # past 500 tokens.
                prompt=prompt,
                max_tokens=1024,
                temperature=temperature,
                top_p=top_p,
                stop=["### Input:", "### Instruction:"],
            )
            raw_text = output.choices[0].text
            # Strip markdown code fences before JSON parsing
            try:
                clean = raw_text.strip().strip("```json").strip("```").strip()
                result = json.loads(clean)
                pathways.append({"pathway": result, "parse_error": False})
            except json.JSONDecodeError:
                pathways.append({"raw_output": raw_text, "parse_error": True})
        return {"product": product_smiles, "pathways": pathways}

    # ═══════════════════════════════════════════════════════════════════════
    # LinkLlama — fragment linker design (1B params)
    # ═══════════════════════════════════════════════════════════════════════

    async def design_linker(
        self,
        fragment1_smiles: str,
        fragment2_smiles: str,
        distance_angstrom: float,
        angle_degrees: float,
        linker_type: str | None = None,
        rotb_range: str | None = None,
        heavy_atoms_range: str | None = None,
        hbd_range: str | None = None,
        hba_range: str | None = None,
        mw_range: str | None = None,
        logp_range: str | None = None,
        tpsa_range: str | None = None,
        reasonability: str = "reasonable",
        num_samples: int = 10,
        temperature: float = 1.4,
        top_p: float = 0.99,
    ) -> dict:
        """Design linker molecules between two fragments using LinkLlama.

        LinkLlama is a 1B-parameter LLM that proposes chemically reasonable
        linkers connecting two molecular fragments given geometric constraints
        (distance in Angstroms, angle in degrees) and optional property constraints.

        Args:
            fragment1_smiles: SMILES of fragment 1 (with [*] attachment point).
            fragment2_smiles: SMILES of fragment 2 (with [*] attachment point).
            distance_angstrom: Distance between attachment points (Angstroms).
            angle_degrees: Angle between attachment points (degrees).
            linker_type: 'chain', 'branched', or 'ring-containing'.
            rotb_range: Rotatable bonds constraint.
            heavy_atoms_range: Heavy atom count constraint.
            hbd_range: H-bond donors constraint.
            hba_range: H-bond acceptors constraint.
            mw_range: Molecular weight constraint.
            logp_range: LogP constraint.
            tpsa_range: TPSA constraint.
            reasonability: 'reasonable' or 'unreasonable'.
            num_samples: Number of linkers to generate (1-100).
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
        """
        user_prompt = self._build_linkllama_prompt(
            fragment1_smiles, fragment2_smiles, distance_angstrom, angle_degrees,
            linker_type, rotb_range, heavy_atoms_range,
            hbd_range, hba_range, mw_range, logp_range, tpsa_range,
            reasonability,
        )
        samples = []
        for _ in range(num_samples):
            response = _client.completions.create(
                model=LINKLLAMA_MODEL,
                prompt=(
                    f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                    f"{LINKLLAMA_SYSTEM}<|eot_id|>"
                    f"<|start_header_id|>user<|end_header_id|>\n\n"
                    f"{user_prompt}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                max_tokens=512,
                temperature=temperature,
                top_p=top_p,
                stop=["<|eot_id|>"],
            )
            raw = response.choices[0].text.strip()
            # LinkLlama returns {"linker": "<SMILES>", "reasoning": "<text>"}
            try:
                clean = raw.strip("```json").strip("```").strip()
                parsed = json.loads(clean)
                samples.append({
                    "linker": parsed.get("linker", ""),
                    "reasoning": parsed.get("reasoning", ""),
                    "parse_error": False,
                })
            except json.JSONDecodeError:
                samples.append({"raw_output": raw, "parse_error": True})
        return {
            "fragments": [fragment1_smiles, fragment2_smiles],
            "geometry": {"distance_angstrom": distance_angstrom, "angle_degrees": angle_degrees},
            "samples": samples,
        }

    @staticmethod
    def _build_linkllama_prompt(
        frag1, frag2, distance, angle,
        linker_type, rotb_range, heavy_atoms_range,
        hbd_range, hba_range, mw_range, logp_range, tpsa_range,
        reasonability,
    ) -> str:
        """Construct the LinkLlama prompt from fragments, geometry, and constraints."""
        fragment_info = (
            f"Fragment 1 (SMILES: {frag1}) and Fragment 2 (SMILES: {frag2}). "
            f"The distance between the attachment points is {distance:.2f} Angstroms, "
            f"and the angle between them is {angle:.2f} degrees."
        )
        linker_props: list[str] = []
        if linker_type:
            linker_props.append(f"Linker type: {linker_type}")
        if rotb_range:
            linker_props.append(f"Rotatable bonds: {rotb_range}")
        if heavy_atoms_range:
            linker_props.append(f"Heavy atoms: {heavy_atoms_range}")
        mol_props: list[str] = []
        if hbd_range:
            mol_props.append(f"H-bond donors {hbd_range}")
        if hba_range:
            mol_props.append(f"H-bond acceptors {hba_range}")
        if mw_range:
            mol_props.append(f"Molecular weight {mw_range}")
        if logp_range:
            mol_props.append(f"LogP {logp_range}")
        if tpsa_range:
            mol_props.append(f"TPSA {tpsa_range}")
        sections = [fragment_info]
        if linker_props:
            sections.append("Linker properties: " + "; ".join(linker_props) + ".")
        if mol_props:
            sections.append("Desired molecule properties: " + "; ".join(mol_props) + ".")
        sections.append(f"Reasonability: {reasonability}.")
        return "\n".join(sections)

    # ═══════════════════════════════════════════════════════════════════════
    # Enamine REAL database search (API + local RDKit fallback)
    # ═══════════════════════════════════════════════════════════════════════

    async def search_enamine_similarity(
        self,
        smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
    ) -> dict:
        """Search Enamine REAL database for molecules similar to the query SMILES.

        Uses Tanimoto similarity on Morgan fingerprints. Returns purchasable
        molecules ranked by similarity score, with Enamine catalog IDs.

        Tries the Enamine REST API first; falls back to local RDKit fingerprint
        search against a cached fragment CSV if the API is unavailable.

        Args:
            smiles: Query SMILES string.
            similarity_threshold: Minimum Tanimoto score (0.0-1.0, default 0.7).
            max_results: Maximum results to return (1-100, default 10).
        """
        return await self._enamine_search(smiles, "similarity", similarity_threshold, max_results)

    async def search_enamine_substructure(
        self,
        smiles: str,
        max_results: int = 10,
    ) -> dict:
        """Search Enamine REAL database for molecules containing the query substructure.

        Returns purchasable molecules that contain the query SMILES as a
        substructure. Requires the Enamine API (no local fallback for substructure).

        Args:
            smiles: Query SMILES substructure.
            max_results: Maximum results to return (1-100, default 10).
        """
        return await self._enamine_search(smiles, "substructure", max_results=max_results)

    async def _enamine_search(
        self, smiles: str, search_type: str,
        similarity_threshold: float = 0.7, max_results: int = 10,
    ) -> dict:
        """Internal: try Enamine API, fall back to local RDKit."""
        results: list[dict] = []

        # Tier 1: Enamine REST API
        if ENAMINE_API_KEY:
            try:
                endpoint = f"{ENAMINE_BASE_URL}/search/{search_type}"
                headers = {"Authorization": f"Bearer {ENAMINE_API_KEY}"}
                payload = {"smiles": smiles, "max_results": max_results}
                if search_type == "similarity":
                    payload["threshold"] = similarity_threshold
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(endpoint, headers=headers, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                for hit in data.get("results", data.get("data", [])):
                    results.append({
                        "smiles": hit.get("smiles", ""),
                        "enamine_id": hit.get("id", hit.get("catalogId", "")),
                        "tanimoto_score": hit.get("similarity", hit.get("score", None)),
                        "availability": hit.get("availability", "unknown"),
                        "price_info": hit.get("price", None),
                        "source": "enamine_api",
                    })
                return {"query": smiles, "search_type": search_type, "results": results}
            except (httpx.HTTPError, httpx.HTTPStatusError):
                pass

        # Tier 2: Local RDKit fingerprint fallback (similarity only)
        if search_type == "similarity":
            results = self._local_similarity_search(smiles, similarity_threshold, max_results)
        else:
            results = [{"error": "Substructure search requires the Enamine API.", "source": "local_cache"}]
        return {"query": smiles, "search_type": search_type, "results": results}

    @staticmethod
    def _local_similarity_search(query_smiles: str, threshold: float, max_results: int) -> list[dict]:
        """Local RDKit Morgan fingerprint similarity against cached Enamine fragments."""
        if not LOCAL_FRAGMENTS_PATH.exists():
            return [{"error": f"Local fragment cache not found at {LOCAL_FRAGMENTS_PATH}.", "source": "local_cache"}]

        query_mol = Chem.MolFromSmiles(query_smiles)
        if query_mol is None:
            return [{"error": f"Invalid query SMILES: {query_smiles}", "source": "local_cache"}]

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

        import polars as pl
        df = pl.read_csv(LOCAL_FRAGMENTS_PATH)
        smiles_col = next((c for c in df.columns if c.lower() in ("smiles", "smi", "molecule")), df.columns[0])
        id_col = next((c for c in df.columns if "id" in c.lower()), None)

        scored: list[tuple[float, str, str]] = []
        for row in df.iter_rows(named=True):
            smi = row[smiles_col]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            score = DataStructs.TanimotoSimilarity(query_fp, fp)
            if score >= threshold:
                scored.append((score, smi, str(row[id_col]) if id_col else ""))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"smiles": smi, "enamine_id": eid, "tanimoto_score": round(sc, 4),
             "availability": "unknown", "price_info": None, "source": "local_cache"}
            for sc, smi, eid in scored[:max_results]
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # Composite workflow: Enamine → LinkLlama
    # ═══════════════════════════════════════════════════════════════════════

    async def find_and_link_fragments(
        self,
        fragment1_smiles: str,
        fragment2_smiles: str,
        distance_angstrom: float,
        angle_degrees: float,
        similarity_threshold: float = 0.6,
        max_enamine_results: int = 5,
        num_linker_samples: int = 10,
    ) -> dict:
        """Find purchasable fragment analogs via Enamine, then design linkers with LinkLlama.

        This composite tool chains Enamine similarity search with LinkLlama:
        1. Searches Enamine for purchasable molecules similar to each fragment
        2. Takes top matches and runs LinkLlama on the best fragment pairs
        3. Returns ranked linker proposals with purchasability metadata

        Args:
            fragment1_smiles: SMILES of fragment 1 (with [*] attachment point).
            fragment2_smiles: SMILES of fragment 2 (with [*] attachment point).
            distance_angstrom: Distance between attachment points (Angstroms).
            angle_degrees: Angle between attachment points (degrees).
            similarity_threshold: Min Tanimoto for Enamine search (default 0.6).
            max_enamine_results: Max Enamine hits per fragment (default 5).
            num_linker_samples: Total linker samples across all pairs (default 10).
        """
        # Search Enamine for both fragments
        enamine_frag1 = await self._enamine_search(fragment1_smiles, "similarity", similarity_threshold, max_enamine_results)
        enamine_frag2 = await self._enamine_search(fragment2_smiles, "similarity", similarity_threshold, max_enamine_results)

        frag1_hits = [r for r in enamine_frag1.get("results", []) if "error" not in r]
        frag2_hits = [r for r in enamine_frag2.get("results", []) if "error" not in r]

        # Fall back to original fragments if no purchasable analogs found
        if not frag1_hits:
            frag1_hits = [{"smiles": fragment1_smiles, "tanimoto_score": 1.0, "source": "original"}]
        if not frag2_hits:
            frag2_hits = [{"smiles": fragment2_smiles, "tanimoto_score": 1.0, "source": "original"}]

        all_proposals: list[dict] = []
        seen_linkers: set[str] = set()

        # Generate linkers for top-3 × top-3 fragment combinations
        pairs = list(cartesian(frag1_hits[:3], frag2_hits[:3]))
        samples_per_pair = max(1, num_linker_samples // len(pairs))

        for f1, f2 in pairs:
            result = await self.design_linker(
                fragment1_smiles=f1["smiles"],
                fragment2_smiles=f2["smiles"],
                distance_angstrom=distance_angstrom,
                angle_degrees=angle_degrees,
                num_samples=samples_per_pair,
            )
            for sample in result.get("samples", []):
                linker_smi = sample.get("linker", "")
                if sample.get("parse_error") or not linker_smi or linker_smi in seen_linkers:
                    continue
                seen_linkers.add(linker_smi)
                all_proposals.append({
                    "linker_smiles": linker_smi,
                    "reasoning": sample.get("reasoning", ""),
                    "fragment1": f1["smiles"],
                    "fragment2": f2["smiles"],
                    "fragment1_tanimoto": f1.get("tanimoto_score"),
                    "fragment2_tanimoto": f2.get("tanimoto_score"),
                    "fragment1_enamine_id": f1.get("enamine_id", ""),
                    "fragment2_enamine_id": f2.get("enamine_id", ""),
                    "purchasable": f1.get("source") != "original" and f2.get("source") != "original",
                })

        # Rank: purchasable pairs first, then by combined Tanimoto score
        all_proposals.sort(
            key=lambda p: (p["purchasable"], (p["fragment1_tanimoto"] or 0) + (p["fragment2_tanimoto"] or 0)),
            reverse=True,
        )

        return {
            "purchasable_fragments": {
                "fragment1_query": fragment1_smiles, "fragment1_hits": frag1_hits,
                "fragment2_query": fragment2_smiles, "fragment2_hits": frag2_hits,
            },
            "linker_proposals": all_proposals,
            "summary": {
                "total_linkers": len(all_proposals),
                "purchasable_pairs_used": len(pairs),
                "fully_purchasable_linkers": sum(1 for p in all_proposals if p["purchasable"]),
            },
        }
