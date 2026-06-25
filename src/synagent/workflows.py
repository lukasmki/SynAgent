"""
Composite workflows that chain multiple tools together.

Currently implements the Enamine → LinkLlama fragment linker pipeline:
find purchasable fragment analogs, then design linkers between them.
"""

from __future__ import annotations

from itertools import product as cartesian

from pydantic_ai import Agent

from .enaminetool import enamine_search
from .llm_tools import LinkLlamaInput, linkllama_generate


async def fragment_linker_workflow(
    fragment1_smiles: str,
    fragment2_smiles: str,
    distance_angstrom: float,
    angle_degrees: float,
    similarity_threshold: float = 0.6,
    max_enamine_results: int = 5,
    num_linker_samples: int = 10,
    temperature: float = 1.4,
    top_p: float = 0.99,
) -> dict:
    """Find purchasable fragment analogs via Enamine, then design linkers with LinkLlama.

    This composite workflow:
    1. Searches Enamine REAL database for purchasable molecules similar
       to each input fragment (by Tanimoto similarity).
    2. Takes the top matches for each fragment.
    3. Runs LinkLlama on the best purchasable fragment pairs to propose
       linker molecules with appropriate geometry.
    4. Returns ranked linker proposals with purchasability metadata.

    Args:
        fragment1_smiles: SMILES of fragment 1 (with [*] attachment point).
        fragment2_smiles: SMILES of fragment 2 (with [*] attachment point).
        distance_angstrom: Distance between attachment points in Angstroms.
        angle_degrees: Angle between attachment points in degrees.
        similarity_threshold: Minimum Tanimoto score for Enamine hits.
        max_enamine_results: Max purchasable analogs per fragment.
        num_linker_samples: Linker samples per fragment pair.
        temperature: LinkLlama sampling temperature.
        top_p: LinkLlama nucleus sampling threshold.

    Returns:
        Dict with purchasable_fragments, linker_proposals, and summary.
    """
    enamine_frag1 = await enamine_search(
        fragment1_smiles, "similarity", similarity_threshold, max_enamine_results
    )
    enamine_frag2 = await enamine_search(
        fragment2_smiles, "similarity", similarity_threshold, max_enamine_results
    )

    frag1_hits = [r for r in enamine_frag1.get("results", []) if "error" not in r]
    frag2_hits = [r for r in enamine_frag2.get("results", []) if "error" not in r]

    if not frag1_hits:
        frag1_hits = [{"smiles": fragment1_smiles, "tanimoto_score": 1.0, "source": "original"}]
    if not frag2_hits:
        frag2_hits = [{"smiles": fragment2_smiles, "tanimoto_score": 1.0, "source": "original"}]

    all_proposals: list[dict] = []
    seen_linkers: set[str] = set()

    pairs = list(cartesian(frag1_hits[:3], frag2_hits[:3]))

    for f1, f2 in pairs:
        params = LinkLlamaInput(
            fragment1_smiles=f1["smiles"],
            fragment2_smiles=f2["smiles"],
            distance_angstrom=distance_angstrom,
            angle_degrees=angle_degrees,
            num_samples=max(1, num_linker_samples // len(pairs)),
            temperature=temperature,
            top_p=top_p,
        )
        result = linkllama_generate(params)

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

    all_proposals.sort(
        key=lambda p: (
            p["purchasable"],
            (p["fragment1_tanimoto"] or 0) + (p["fragment2_tanimoto"] or 0),
        ),
        reverse=True,
    )

    return {
        "purchasable_fragments": {
            "fragment1_query": fragment1_smiles,
            "fragment1_hits": frag1_hits,
            "fragment2_query": fragment2_smiles,
            "fragment2_hits": frag2_hits,
        },
        "linker_proposals": all_proposals,
        "summary": {
            "total_linkers": len(all_proposals),
            "purchasable_pairs_used": len(pairs),
            "fully_purchasable_linkers": sum(1 for p in all_proposals if p["purchasable"]),
        },
    }


def register_workflow_tools(agent: Agent) -> None:
    """Register composite workflow tools on a pydantic-ai agent."""

    @agent.tool_plain
    async def find_and_link_fragments(
        fragment1_smiles: str,
        fragment2_smiles: str,
        distance_angstrom: float,
        angle_degrees: float,
        similarity_threshold: float = 0.6,
        max_enamine_results: int = 5,
        num_linker_samples: int = 10,
    ) -> dict:
        """Find purchasable fragment analogs and design linkers between them.

        This is a composite tool that chains Enamine similarity search
        with LinkLlama linker generation. It:
        1. Searches Enamine for purchasable molecules similar to each fragment
        2. Runs LinkLlama on the best purchasable fragment pairs
        3. Returns ranked linker proposals with purchasability metadata

        Use this tool when you have two molecular fragments and want to
        find commercially available analogs and propose linkers in one step.

        Args:
            fragment1_smiles: SMILES of fragment 1 (with [*] attachment point).
            fragment2_smiles: SMILES of fragment 2 (with [*] attachment point).
            distance_angstrom: Distance between attachment points (Angstroms).
            angle_degrees: Angle between attachment points (degrees).
            similarity_threshold: Min Tanimoto for Enamine search (default 0.6).
            max_enamine_results: Max Enamine hits per fragment (default 5).
            num_linker_samples: Total linker samples across all pairs (default 10).
        """
        return await fragment_linker_workflow(
            fragment1_smiles=fragment1_smiles,
            fragment2_smiles=fragment2_smiles,
            distance_angstrom=distance_angstrom,
            angle_degrees=angle_degrees,
            similarity_threshold=similarity_threshold,
            max_enamine_results=max_enamine_results,
            num_linker_samples=num_linker_samples,
        )
