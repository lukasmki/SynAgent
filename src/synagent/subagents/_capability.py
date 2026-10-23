"""Spec-loadable wrapper around the pydantic-ai-harness SubAgents capability.

The harness capability holds live Agent instances, so it opts out of agent
specs. This subclass builds each sub-agent from its own spec file instead, so
the roster can be declared in YAML:

    - SubAgents:
        agents:
          - validator.yaml
          - path: retriever.yaml
            timeout_seconds: 300
"""

from dataclasses import dataclass
from typing import Any, NotRequired, TypedDict

from pydantic import ConfigDict, TypeAdapter, with_config
from pydantic_ai.tools import AgentDepsT
from pydantic_ai_harness import subagents as harness


@with_config(ConfigDict(extra="forbid"))
class SubAgentFile(TypedDict):
    """A sub-agent spec file plus its per-delegate run controls."""

    path: str
    "Spec file path, relative to the spec that references it"

    description: NotRequired[str]
    "Overrides the description from the sub-agent's spec"

    timeout_seconds: NotRequired[float]
    "Wall-clock limit for one delegation"

    max_calls: NotRequired[int]
    "Maximum delegations to this sub-agent per parent run"

    on_failure: NotRequired[str]
    "Message returned to the parent instead of raising when a delegation fails"


_entries = TypeAdapter(list[str | SubAgentFile])


@dataclass
class SubAgents(harness.SubAgents[AgentDepsT]):
    @classmethod
    def get_serialization_name(cls) -> str | None:
        return "SubAgents"

    @classmethod
    def from_spec(
        cls,
        agents: list[str | SubAgentFile],
        max_depth: int | None = None,
        forward_usage: bool | None = None,
        contain_errors: bool | None = None,
        include_self: bool | None = None,
    ) -> "SubAgents[Any]":
        # Deferred: synagent.spec imports this module to register the capability.
        from synagent.spec import load_agent

        sub_agents: list[harness.SubAgent[Any]] = []
        for entry in _entries.validate_python(agents):
            options = {"path": entry} if isinstance(entry, str) else dict(entry)
            agent = load_agent(options.pop("path"))
            sub_agents.append(harness.SubAgent(agent, **options))

        # Unset options keep the harness defaults.
        overrides = {
            "max_depth": max_depth,
            "forward_usage": forward_usage,
            "contain_errors": contain_errors,
            "include_self": include_self,
        }
        return cls(
            agents=sub_agents,
            **{key: value for key, value in overrides.items() if value is not None},
        )
