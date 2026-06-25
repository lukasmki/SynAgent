"""
Enamine REAL database search tool.

Supports similarity and substructure search via the Enamine REST API,
with a local RDKit fingerprint fallback when the API is unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Enamine REST API credentials — set ENAMINE_API_KEY in your .env file.
# If the key is empty, all searches fall back to local RDKit fingerprint matching.
ENAMINE_API_KEY = os.getenv("ENAMINE_API_KEY", "")
ENAMINE_BASE_URL = os.getenv(
    "ENAMINE_BASE_URL", "https://api.enamine.net/api/v1"
)

# Path to a local CSV cache of Enamine fragments for offline fallback.
# Expected columns: a SMILES column (auto-detected) and optionally an ID column.
LOCAL_FRAGMENTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "enamine_fragments.csv"


class EnamineSearchInput(BaseModel):
    smiles: str = Field(description="Query SMILES string.")
    search_type: Literal["similarity", "substructure"] = Field(
        default="similarity",
        description="Type of search to perform.",
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum Tanimoto similarity score (similarity search only).",
    )
    max_results: int = Field(default=10, ge=1, le=100)


# ---------------------------------------------------------------------------
# Enamine REST API
# ---------------------------------------------------------------------------

async def _enamine_api_search(
    smiles: str,
    search_type: str,
    similarity_threshold: float,
    max_results: int,
) -> list[dict]:
    endpoint = f"{ENAMINE_BASE_URL}/search/{search_type}"
    headers = {"Authorization": f"Bearer {ENAMINE_API_KEY}"}
    payload = {
        "smiles": smiles,
        "max_results": max_results,
    }
    if search_type == "similarity":
        payload["threshold"] = similarity_threshold

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    results = []
    for hit in data.get("results", data.get("data", [])):
        results.append({
            "smiles": hit.get("smiles", ""),
            "enamine_id": hit.get("id", hit.get("catalogId", "")),
            "tanimoto_score": hit.get("similarity", hit.get("score", None)),
            "availability": hit.get("availability", "unknown"),
            "price_info": hit.get("price", None),
            "source": "enamine_api",
        })
    return results


# ---------------------------------------------------------------------------
# Local RDKit fallback
# ---------------------------------------------------------------------------

def _local_similarity_search(
    query_smiles: str,
    threshold: float,
    max_results: int,
) -> list[dict]:
    if not LOCAL_FRAGMENTS_PATH.exists():
        return [{
            "error": f"Local fragment cache not found at {LOCAL_FRAGMENTS_PATH}. "
            "Provide data/enamine_fragments.csv or set ENAMINE_API_KEY.",
            "source": "local_cache",
        }]

    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return [{"error": f"Invalid query SMILES: {query_smiles}", "source": "local_cache"}]

    # Morgan fingerprint with radius=2 and 2048 bits — standard for drug-like similarity
    query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

    # Lazy import polars since this path is only hit when the API is unavailable
    import polars as pl
    df = pl.read_csv(LOCAL_FRAGMENTS_PATH)
    smiles_col = next(
        (c for c in df.columns if c.lower() in ("smiles", "smi", "molecule")),
        df.columns[0],
    )
    id_col = next(
        (c for c in df.columns if "id" in c.lower()),
        None,
    )

    scored: list[tuple[float, str, str]] = []
    for row in df.iter_rows(named=True):
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        score = DataStructs.TanimotoSimilarity(query_fp, fp)
        if score >= threshold:
            eid = row[id_col] if id_col else ""
            scored.append((score, smi, str(eid)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "smiles": smi,
            "enamine_id": eid,
            "tanimoto_score": round(score, 4),
            "availability": "unknown",
            "price_info": None,
            "source": "local_cache",
        }
        for score, smi, eid in scored[:max_results]
    ]


# ---------------------------------------------------------------------------
# Public API — tries Enamine REST, falls back to local RDKit
# ---------------------------------------------------------------------------

async def enamine_search(
    smiles: str,
    search_type: str = "similarity",
    similarity_threshold: float = 0.7,
    max_results: int = 10,
) -> dict:
    """Search Enamine REAL database for molecules similar to the query.

    Tries the Enamine REST API first. If the API is unavailable or the key
    is not set, falls back to local RDKit Morgan fingerprint similarity
    against a cached fragment file.

    Args:
        smiles: Query SMILES string.
        search_type: 'similarity' or 'substructure'.
        similarity_threshold: Minimum Tanimoto score (similarity only).
        max_results: Maximum number of results.

    Returns:
        Dict with 'query', 'search_type', and 'results' list.
    """
    # Two-tier search strategy:
    # 1. Try the Enamine REST API (fast, comprehensive, requires API key)
    # 2. Fall back to local RDKit fingerprint search against cached CSV
    results: list[dict] = []

    if ENAMINE_API_KEY:
        try:
            results = await _enamine_api_search(
                smiles, search_type, similarity_threshold, max_results
            )
            return {"query": smiles, "search_type": search_type, "results": results}
        except (httpx.HTTPError, httpx.HTTPStatusError):
            pass

    if search_type == "similarity":
        results = _local_similarity_search(smiles, similarity_threshold, max_results)
    else:
        results = [{"error": "Substructure search requires the Enamine API.", "source": "local_cache"}]

    return {"query": smiles, "search_type": search_type, "results": results}


# ---------------------------------------------------------------------------
# pydantic-ai tool registration
# ---------------------------------------------------------------------------

def register_enamine_tools(agent: Agent) -> None:
    """Register Enamine search tools on a pydantic-ai agent."""

    @agent.tool_plain
    async def search_enamine_similarity(
        smiles: str,
        similarity_threshold: float = 0.7,
        max_results: int = 10,
    ) -> dict:
        """Search Enamine REAL database for molecules similar to the query SMILES.

        Uses Tanimoto similarity on Morgan fingerprints. Returns purchasable
        molecules ranked by similarity score, with Enamine catalog IDs and
        availability info.

        Use this when you need to find commercially available molecules
        that are structurally similar to a target or fragment. Especially
        useful before calling LinkLlama to find purchasable fragment
        alternatives for linker design.

        Args:
            smiles: Query SMILES string.
            similarity_threshold: Minimum Tanimoto score (0.0-1.0, default 0.7).
            max_results: Maximum results to return (1-100, default 10).
        """
        return await enamine_search(smiles, "similarity", similarity_threshold, max_results)

    @agent.tool_plain
    async def search_enamine_substructure(
        smiles: str,
        max_results: int = 10,
    ) -> dict:
        """Search Enamine REAL database for molecules containing the query as a substructure.

        Returns purchasable molecules that contain the query SMILES as a
        substructure, with Enamine catalog IDs and availability info.

        Use this when you need to find commercially available molecules
        that contain a specific scaffold or functional group.

        Args:
            smiles: Query SMILES substructure.
            max_results: Maximum results to return (1-100, default 10).
        """
        return await enamine_search(smiles, "substructure", max_results=max_results)
