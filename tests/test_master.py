"""Tests for master agent configuration and prompt."""

from __future__ import annotations


def test_master_prompt_is_disagreeable():
    from synagent.agents.master import MASTER_PROMPT

    prompt_lower = MASTER_PROMPT.lower()
    assert "disagreeable" in prompt_lower or "challenge" in prompt_lower
    assert "why" in prompt_lower
    assert "alternative" in prompt_lower


def test_master_agent_has_llm_tools():
    from synagent.agents.master import agent

    tool_names = {t.name for t in agent._function_tools}
    assert "generate_molecules" in tool_names, f"Missing generate_molecules. Found: {tool_names}"
    assert "retrosynthesis" in tool_names, f"Missing retrosynthesis. Found: {tool_names}"
    assert "design_linker" in tool_names, f"Missing design_linker. Found: {tool_names}"


def test_master_agent_has_enamine_tools():
    from synagent.agents.master import agent

    tool_names = {t.name for t in agent._function_tools}
    assert "search_enamine_similarity" in tool_names
    assert "search_enamine_substructure" in tool_names


def test_master_agent_has_workflow_tools():
    from synagent.agents.master import agent

    tool_names = {t.name for t in agent._function_tools}
    assert "find_and_link_fragments" in tool_names


def test_master_agent_has_subagent_tools():
    from synagent.agents.master import agent

    tool_names = {t.name for t in agent._function_tools}
    assert "call_validation_agent" in tool_names
    assert "call_chemspace_agent" in tool_names
    assert "call_optimization_agent" in tool_names
    assert "full_route_evaluation" in tool_names
