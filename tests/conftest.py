"""Shared fixtures for SynAgent tests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mock_openai_client(monkeypatch):
    """Patch the vLLM OpenAI client in llm_tools."""
    client = MagicMock()
    monkeypatch.setattr("synagent.llm_tools._client", client)
    return client


def make_completion(text: str):
    """Build a fake OpenAI Completion response."""
    choice = SimpleNamespace(text=text)
    return SimpleNamespace(choices=[choice])


@pytest.fixture
def mock_httpx(monkeypatch):
    """Return a helper that patches httpx.AsyncClient for Enamine tests."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()
    mock_post = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = mock_post

    monkeypatch.setattr("synagent.enaminetool.httpx.AsyncClient", lambda **kw: mock_client)
    return mock_post
