from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model

from synagent.spec import DEFAULT_SPEC, load_agent


def get_agent(
    model: Model[Any] | KnownModelName | str | None = None,
    spec: Path | str = DEFAULT_SPEC,
) -> Agent[None, str]:
    """Build SynAgent from `spec`, optionally overriding the spec's model."""
    return load_agent(spec, model=model)
