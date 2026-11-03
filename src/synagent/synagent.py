from pathlib import Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.usage import UsageLimits

from synagent.spec import DEFAULT_SPEC, load_agent, resolve_spec
from synagent.tracing import build_tracing_capabilities

# Route generation and validation take far more than pydantic-ai's default of
# 50 model requests per run. Sub-agents share the parent's usage, so this also
# bounds the requests they make.
USAGE_LIMITS = UsageLimits(request_limit=1000)


def get_agent(
    model: Model[Any] | KnownModelName | str | None = None,
    spec: Path | str = DEFAULT_SPEC,
    *,
    trace: Path | None = None,
) -> Agent[None, str]:
    """Build SynAgent from `spec`, optionally overriding the spec's model.

    `spec` is a spec file path or the name of a bundled spec, such as `"nogpu"`.

    With `trace`, the agent and every sub-agent its spec loads record to that file,
    so sub-agent trajectories are traced as their own runs, linked to the parent by
    run id and the delegating tool call.
    """
    return load_agent(
        resolve_spec(spec),
        model=model,
        capabilities=build_tracing_capabilities(trace),
    )
