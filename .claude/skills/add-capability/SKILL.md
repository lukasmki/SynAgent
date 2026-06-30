---
name: add-capability
description: Scaffold a new SynAgent capability — creates the 4-file package and registers it in __main__.py. Pass the snake_case name and optional one-line description as args.
---

Scaffold a new SynAgent capability package and register it with the agent. The capability pattern is a 4-file package plus one registration edit.

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
from dataclasses import dataclass

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from synagent.<name>._toolset import <ClassName>Toolset


@dataclass
class <ClassName>(AbstractCapability[AgentDepsT]):
    id = "<name>"
    description = "<description>"
    defer_loading = True

    def get_instructions(self) -> str:
        return "Instructions injected into the agent system prompt. Describe when and how to use these tools."

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        return <ClassName>Toolset()
```

Update `get_instructions()` with instructions specific to the capability.

## Step 5 — Write `__init__.py`

```python
from synagent.<name>._capability import <ClassName>
from synagent.<name>._toolset import <ClassName>Toolset

__all__ = ["<ClassName>", "<ClassName>Toolset"]
```

## Step 6 — Register in `synagent.py`

Edit `src/synagent/synagent.py`:

1. Add import alongside the existing capability imports at the top:
   ```python
   from synagent.<name> import <ClassName>
   ```

2. Add `<ClassName>()` to the `capabilities=[...]` list inside `get_agent()`:
   ```python
   capabilities=[
       AnalogueSearch(),
       Chemspace(),
       SynthesisValidation(),
       Scoring(),
       Storage(),
       SubAgents(...),
       <ClassName>(),   # ← add here
   ],
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
- **`__init__.py` must re-export `ClassName`** — `__main__.py` imports from the package, not from the submodule directly.
