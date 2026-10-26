from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from synagent.spec import DEFAULT_SPEC, load_agent
from synagent.tracing import build_tracing_capabilities


def get_agent(
    model: Model[Any] | KnownModelName | str | None = None,
    spec: Path | str = DEFAULT_SPEC,
    *,
    trace: Path | None = None,
) -> Agent[None, str]:
    """Build SynAgent from `spec`, optionally overriding the spec's model.

    With `trace`, the agent and every sub-agent its spec loads record to that file,
    so sub-agent trajectories are traced as their own runs, linked to the parent by
    run id and the delegating tool call.
    """
    return load_agent(spec, model=model, capabilities=build_tracing_capabilities(trace))
