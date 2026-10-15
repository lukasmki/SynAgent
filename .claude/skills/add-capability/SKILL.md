---
name: add-capability
description: Scaffold a new SynAgent capability — creates the 4-file package and registers it in the agent spec. Pass the snake_case name and optional one-line description as args.
---

Scaffold a new SynAgent capability package and register it with the agent. The capability pattern is a 4-file package plus two registration edits.

## Inputs

Resolve these two values before writing any files:

- **`name`** — snake_case package name (e.g. `retrosynthesis`). Take from skill args if provided; otherwise ask the user.
- **`description`** — one-line description shown to the LLM (e.g. `"Score and rank retrosynthetic pathways"`). Take from skill args if provided; otherwise ask.
- **`ClassName`** — derive from `name` by converting to PascalCase (e.g. `retrosynthesis` → `Retrosynthesis`, `my_tool` → `MyTool`).

## Step 1 — Create the package directory

```
src/synagent/<name>/
```

Use `mkdir -p` or the Write tool (writing any file inside the directory creates it implicitly).

## Step 2 — Write `_models.py`

Minimal stub — extend with real I/O schemas as the capability is built out.

```python
from pydantic import BaseModel


class <ClassName>Result(BaseModel):
    """Replace with real output schema."""
    pass
```

## Step 3 — Write `_toolset.py`

```python
from pydantic_ai import FunctionToolset
from pydantic_ai.tools import AgentDepsT


class <ClassName>Toolset(FunctionToolset[AgentDepsT]):
    include_return_schema = True

    def __init__(self):
        super().__init__()
        self.add_function(self.placeholder, name="placeholder")

    async def placeholder(self, input: str) -> str:
        """Placeholder tool — replace with real implementation.

        Args:
            input (str): Input string.

        Returns:
            str: Result.
        """
        return input
```

Replace `placeholder` with the real tool name(s) and implementation. Each tool:
- Must be an `async def` method
- Must have a docstring — it becomes the tool description shown to the LLM
- Must be registered in `__init__` via `self.add_function(self.<method>, name="<tool_name>")`

## Step 4 — Write `_capability.py`

```python
from dataclasses import dataclass, field

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.<name>._toolset import <ClassName>Toolset


@dataclass
class <ClassName>(AbstractCapability[AgentDepsT]):
    id: str = field(default="<name>", kw_only=True)
    description: str = field(default="<description>", kw_only=True)
    defer_loading: bool = field(default=True, kw_only=True)

    def get_instructions(self) -> str:
        return "Instructions injected into the agent system prompt. Describe when and how to use these tools."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return <ClassName>Toolset()
```

Update `get_instructions()` with instructions specific to the capability.

`id`, `description`, and `defer_loading` are inherited **dataclass fields** of
`AbstractCapability`, not plain class constants. They must be written as annotated
fields with `kw_only=True` (matching the base class's `KW_ONLY` marker). Writing them
as bare class attributes (`id = "<name>"`) silently fails: the generated `__init__`
overwrites each instance attribute with the base class default (`None`, `None`,
`False`), so the capability loses its id/description and is never deferred.

Any additional fields (e.g. `api_key: str | None = None`) go after these three and
stay positional-or-keyword.

## Step 5 — Write `__init__.py`

```python
from synagent.<name>._capability import <ClassName>
from synagent.<name>._toolset import <ClassName>Toolset

__all__ = ["<ClassName>", "<ClassName>Toolset"]
```

## Step 6 — Register in the agent spec

The agent is built from YAML specs in `src/synagent/specs/` (pydantic-ai AgentSpec).
A capability can be listed in a spec only once its class is registered.

1. Edit `src/synagent/spec.py`: import the class alongside the existing capability
   imports and add it to `CAPABILITY_TYPES`:
   ```python
   from synagent.<name> import <ClassName>

   CAPABILITY_TYPES = (
       ...,
       <ClassName>,   # ← add here
   )
   ```

2. List it under `capabilities:` in `src/synagent/specs/synagent.yaml`, and in any
   sub-agent spec (`validator.yaml`, `retriever.yaml`) that should have it. Specs refer
   to capabilities by class name. Pass dataclass fields as a mapping:
   ```yaml
   capabilities:
     - <ClassName>                      # no arguments
     - <ClassName>: {api_key: "..."}    # with field values
   ```
   Fields must be JSON/YAML-representable (str, int, float, bool, lists, dicts). If the
   capability needs something richer, override `from_spec` (see
   `src/synagent/subagents/_capability.py`).

3. Regenerate the editor schema so YAML autocompletion knows the new capability:
   ```bash
   uv run synagent schema
   ```

## Step 7 — Verify

Start the server in the background, wait for it to be ready, then stop it:

```bash
uv run synagent serve &>/tmp/synagent.log &
SERVER_PID=$!
for i in {1..20}; do
  curl -sf http://localhost:8000/api/health >/dev/null && break
  sleep 0.5
done
curl -sf http://localhost:8000/api/health && echo "OK"
kill $SERVER_PID
```

A `{"ok":true}` response confirms the new capability loaded without import or instantiation errors. If the server fails to start, check `/tmp/synagent.log` for the traceback.

## Gotchas

- **`id` must be unique** — duplicate `id` values across capabilities cause silent conflicts. Use the snake_case `name` as the id.
- **`defer_loading = True` is required** — omitting it causes the capability to load eagerly before the agent is fully constructed, which can break initialization.
- **All tool methods must be `async`** — synchronous methods registered via `add_function` will raise at runtime.
- **Docstrings are the tool description** — the LLM sees nothing else about a tool, so make them precise and include arg/return descriptions.
- **`__init__.py` must re-export `ClassName`** — `spec.py` imports from the package, not from the submodule directly.
