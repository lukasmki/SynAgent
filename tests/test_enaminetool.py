"""Tests for Enamine search tool (API + local fallback)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from synagent.enaminetool import enamine_search


@pytest.mark.asyncio
async def test_similarity_api_success(mock_httpx, monkeypatch):
    monkeypatch.setattr("synagent.enaminetool.ENAMINE_API_KEY", "test-key")
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = lambda: None
    mock_resp.json.return_value = {
        "results": [
            {"smiles": "CCO", "id": "EN001", "similarity": 0.95, "availability": "in stock", "price": "$10"},
            {"smiles": "CCCO", "id": "EN002", "similarity": 0.82, "availability": "in stock", "price": "$15"},
        ]
    }
    mock_httpx.return_value = mock_resp

    result = await enamine_search("CCO", "similarity", 0.7, 10)
    assert result["query"] == "CCO"
    assert result["search_type"] == "similarity"
    assert len(result["results"]) == 2
    assert result["results"][0]["enamine_id"] == "EN001"
    assert result["results"][0]["tanimoto_score"] == 0.95
    assert result["results"][0]["source"] == "enamine_api"


@pytest.mark.asyncio
async def test_api_failure_falls_back_to_local(mock_httpx, monkeypatch):
    monkeypatch.setattr("synagent.enaminetool.ENAMINE_API_KEY", "test-key")
    mock_httpx.side_effect = httpx.HTTPError("connection failed")

    result = await enamine_search("CCO", "similarity", 0.7, 10)
    for r in result["results"]:
        assert r["source"] == "local_cache"


@pytest.mark.asyncio
async def test_no_api_key_uses_local(monkeypatch):
    monkeypatch.setattr("synagent.enaminetool.ENAMINE_API_KEY", "")
    result = await enamine_search("CCO", "similarity", 0.7, 10)
    for r in result["results"]:
        assert r["source"] == "local_cache"


@pytest.mark.asyncio
async def test_substructure_without_api_returns_error(monkeypatch):
    monkeypatch.setattr("synagent.enaminetool.ENAMINE_API_KEY", "")
    result = await enamine_search("c1ccccc1", "substructure", max_results=5)
    assert any("error" in r for r in result["results"])
