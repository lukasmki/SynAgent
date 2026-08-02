"""Shared fixtures for SynAgent tests.

Adapted from the mock patterns on the SynLlama-SmileyLlama-linkLlama-added-in
branch. The patch targets moved when those tools were ported into the
capability architecture: `synagent.llm_tools._client` is now
`synagent.model_calls._toolset._client`.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# vLLM / chemistry model mocking
# ---------------------------------------------------------------------------


def make_completion(text: str):
    """Build a fake OpenAI Completion response.

    The toolset reads `response.choices[0].text`, so that is all this needs
    to model.
    """
    return SimpleNamespace(choices=[SimpleNamespace(text=text)])


@pytest.fixture
def mock_vllm(monkeypatch):
    """Patch the shared vLLM client used by all three chemistry models.

    Returns the MagicMock so a test can set
    `mock_vllm.completions.create.return_value = make_completion(...)`
    or supply a side_effect list for multi-call sequences.
    """
    client = MagicMock()
    monkeypatch.setattr("synagent.model_calls._toolset._client", client)
    return client


@pytest.fixture
def mock_enamine(monkeypatch):
    """Patch httpx.AsyncClient inside the model_calls toolset."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_post = AsyncMock()
    mock_client.post = mock_post

    monkeypatch.setattr(
        "synagent.model_calls._toolset.httpx.AsyncClient",
        lambda **kw: mock_client,
    )
    return mock_post


# ---------------------------------------------------------------------------
# Live-server detection
# ---------------------------------------------------------------------------


def _port_open(host: str, port: int, timeout: float = 0.35) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _hostport(url: str, default_port: int) -> tuple[str, int]:
    rest = url.split("://", 1)[-1].split("/", 1)[0]
    if ":" in rest:
        host, _, port = rest.partition(":")
        return host, int(port)
    return rest, default_port


def vllm_available() -> bool:
    host, port = _hostport(os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"), 8000)
    return _port_open(host, port)


def ollama_available() -> bool:
    host, port = _hostport(os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"), 11434)
    return _port_open(host, port)


def anthropic_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


requires_vllm = pytest.mark.skipif(
    not vllm_available(),
    reason="vLLM server not reachable — the three chemistry models are unavailable.",
)

requires_orchestrator = pytest.mark.skipif(
    not (ollama_available() or anthropic_available()),
    reason="No orchestrator backend: Ollama unreachable and ANTHROPIC_API_KEY unset.",
)


@pytest.fixture
def orchestrator() -> tuple[str, str]:
    """Pick an available orchestrator as (model_name, provider).

    Prefers a local Ollama; falls back to Claude when ANTHROPIC_API_KEY is set.
    """
    if ollama_available():
        return os.getenv("SYNAGENT_MODEL", "qwen3.5:9b"), "ollama"
    if anthropic_available():
        return os.getenv("SYNAGENT_MODEL", "claude-sonnet-5"), "anthropic"
    pytest.skip("no orchestrator backend available")


# ---------------------------------------------------------------------------
# Corrector gate fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gate_ctx():
    """Build a minimal RunContext-like object carrying user messages.

    Corrector.prepare_tools only reads ctx.messages, looking for ModelRequest
    parts of type UserPromptPart, so a SimpleNamespace is enough and avoids
    constructing a full RunContext.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    def _make(*user_messages: str):
        messages = [
            ModelRequest(parts=[UserPromptPart(content=m)]) for m in user_messages
        ]
        return SimpleNamespace(messages=messages)

    return _make


@pytest.fixture
def tool_defs():
    """A representative tool list spanning corrector and non-corrector tools."""
    from pydantic_ai.tools import ToolDefinition

    names = [
        # corrector-owned (gated)
        "fix_step",
        "fix_building_blocks",
        "apply_fixes",
        "search_step_building_blocks",
        "fix_smarts",
        "extract_template_from_reaction",
        "fix_template",
        "fix_smiles",
        # other capabilities (must always pass through)
        "validate_route",
        "retro_search",
        "generate_molecules",
        "retrosynthesis",
    ]
    return [
        ToolDefinition(name=n, description=n, parameters_json_schema={"type": "object"})
        for n in names
    ]


# ---------------------------------------------------------------------------
# anyio
# ---------------------------------------------------------------------------


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_pathway() -> dict:
    """A minimal SynLlama-shaped pathway used across pipeline tests."""
    return {
        "target": "CC(=O)Nc1ccc(O)cc1",
        "steps": [
            {
                "step": 1,
                "template": "[C:1](=[O:2])[OH]>>[C:1](=[O:2])Cl",
                "reactants": ["CC(=O)O", "Nc1ccc(O)cc1"],
                "product": "CC(=O)Nc1ccc(O)cc1",
            }
        ],
        "building_blocks": ["CC(=O)O", "Nc1ccc(O)cc1"],
    }
