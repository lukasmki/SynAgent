"""Tests for composite Enamine → LinkLlama workflow."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from conftest import make_completion


@pytest.mark.asyncio
async def test_fragment_linker_workflow(monkeypatch):
    enamine_results = {
        "query": "test",
        "search_type": "similarity",
        "results": [
            {"smiles": "[*]c1ccc(F)cc1", "enamine_id": "EN100", "tanimoto_score": 0.9,
             "availability": "in stock", "price_info": "$20", "source": "enamine_api"},
        ],
    }
    mock_enamine = AsyncMock(return_value=enamine_results)
    monkeypatch.setattr("synagent.workflows.enamine_search", mock_enamine)

    linkllama_result = {
        "fragments": ["[*]c1ccc(F)cc1", "[*]c1ccc(F)cc1"],
        "geometry": {"distance_angstrom": 4.5, "angle_degrees": 120.0},
        "samples": [
            {"linker": "CCC", "reasoning": "short chain", "parse_error": False},
        ],
    }
    monkeypatch.setattr("synagent.workflows.linkllama_generate", lambda params: linkllama_result)

    from synagent.workflows import fragment_linker_workflow

    result = await fragment_linker_workflow(
        fragment1_smiles="[*]c1ccccc1",
        fragment2_smiles="[*]C1CCCCC1",
        distance_angstrom=4.5,
        angle_degrees=120.0,
        max_enamine_results=3,
        num_linker_samples=5,
    )

    assert "purchasable_fragments" in result
    assert "linker_proposals" in result
    assert "summary" in result
    assert result["summary"]["total_linkers"] >= 1
    assert result["linker_proposals"][0]["linker_smiles"] == "CCC"


@pytest.mark.asyncio
async def test_workflow_no_enamine_hits_uses_originals(monkeypatch):
    empty_results = {"query": "test", "search_type": "similarity", "results": []}
    mock_enamine = AsyncMock(return_value=empty_results)
    monkeypatch.setattr("synagent.workflows.enamine_search", mock_enamine)

    linkllama_result = {
        "fragments": ["[*]c1ccccc1", "[*]C1CCCCC1"],
        "geometry": {"distance_angstrom": 5.0, "angle_degrees": 110.0},
        "samples": [
            {"linker": "CCCC", "reasoning": "fallback", "parse_error": False},
        ],
    }
    monkeypatch.setattr("synagent.workflows.linkllama_generate", lambda params: linkllama_result)

    from synagent.workflows import fragment_linker_workflow

    result = await fragment_linker_workflow(
        fragment1_smiles="[*]c1ccccc1",
        fragment2_smiles="[*]C1CCCCC1",
        distance_angstrom=5.0,
        angle_degrees=110.0,
    )

    assert result["summary"]["total_linkers"] >= 1
    assert result["linker_proposals"][0]["purchasable"] is False
