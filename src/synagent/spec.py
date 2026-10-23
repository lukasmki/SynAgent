"""Build SynAgent agents from pydantic-ai AgentSpec files (YAML or JSON)."""

import json
from collections.abc import Sequence
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, AgentSpec
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model

from synagent.analogues import AnalogueSearch
from synagent.chemspace import Chemspace
from synagent.generation import Generation
from synagent.scoring import Scoring
from synagent.storage import Storage
from synagent.subagents import SubAgents
from synagent.validation import SynthesisValidation

SPEC_DIR = Path(__file__).parent / "specs"
DEFAULT_SPEC = SPEC_DIR / "synagent.yaml"
SCHEMA_PATH = SPEC_DIR / "synagent_schema.json"

CAPABILITY_TYPES = (
    AnalogueSearch,
    Chemspace,
    Generation,
    Scoring,
    Storage,
    SubAgents,
    SynthesisValidation,
)
"Capabilities that specs can reference by class name, beyond pydantic-ai's built-ins"

_loading: ContextVar[tuple[Path, ...]] = ContextVar("_loading", default=())
"Spec files currently being loaded, outermost first"

_shared: ContextVar[tuple[AgentCapability[Any], ...]] = ContextVar(
    "_shared", default=()
)
"Capabilities added to every agent built while the outermost spec loads"


def resolve_spec_path(path: Path | str) -> Path:
    """Resolve `path` against the spec file being loaded, or the working directory."""
    path = Path(path)
    if not path.is_absolute():
        stack = _loading.get()
        path = (stack[-1].parent if stack else Path.cwd()) / path
    return path.resolve()


def load_agent(
    spec: Path | str = DEFAULT_SPEC,
    model: Model[Any] | KnownModelName | str | None = None,
    capabilities: Sequence[AgentCapability[Any]] | None = None,
) -> Agent[None, str]:
    """Build an agent from a spec file.

    `model` overrides the spec's `model`. Sub-agent specs that leave `model`
    unset run on whichever model the parent run uses.

    `capabilities` are added to this agent and to every sub-agent its spec
    loads. When left unset, a nested load inherits the outer load's.
    """
    path = resolve_spec_path(spec)
    stack = _loading.get()
    if path in stack:
        chain = " -> ".join(p.name for p in (*stack, path))
        raise UserError(f"Agent specs reference each other in a cycle: {chain}")

    shared = _shared.get() if capabilities is None else tuple(capabilities)
    token = _loading.set((*stack, path))
    shared_token = _shared.set(shared)
    try:
        return Agent.from_file(
            path,
            custom_capability_types=CAPABILITY_TYPES,
            model=model,
            capabilities=list(shared),
        )
    finally:
        _shared.reset(shared_token)
        _loading.reset(token)


def write_schema(path: Path | str = SCHEMA_PATH) -> Path:
    """Write the JSON schema for SynAgent specs, for editor validation."""
    path = Path(path)
    schema = AgentSpec.model_json_schema_with_capabilities(CAPABILITY_TYPES)
    path.write_text(json.dumps(schema, indent=2) + "\n")
    return path
