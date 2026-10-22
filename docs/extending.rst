Extending SynAgent
==================

Adding a capability
-------------------

Each capability is a package under ``src/synagent/`` with four files:

``_models.py``
   Pydantic models for tool inputs and outputs.
``_toolset.py``
   A ``FunctionToolset`` subclass. Each tool is an ``async`` method registered
   in ``__init__`` with ``self.add_function(self.method, name="tool_name")``.
   The docstring is the only description of the tool the model sees, so
   document every argument and the return value.
``_capability.py``
   An ``AbstractCapability`` dataclass with an ``id``, a ``description``, usage
   instructions from ``get_instructions()``, and the toolset from
   ``get_toolset()``.
``__init__.py``
   Re-exports the capability and toolset classes.

A minimal capability:

.. code-block:: python

   from dataclasses import dataclass, field

   from pydantic_ai.capabilities import AbstractCapability
   from pydantic_ai.tools import AgentDepsT
   from pydantic_ai.toolsets import AgentToolset

   from synagent.example._toolset import ExampleToolset


   @dataclass
   class Example(AbstractCapability[AgentDepsT]):
       id: str = field(default="example", kw_only=True)
       description: str = field(default="Use for ...", kw_only=True)
       defer_loading: bool = field(default=True, kw_only=True)

       api_url: str = "http://localhost:9000"
       "Fields after the three above can be set from a spec"

       def get_instructions(self) -> str:
           return "When and how to use these tools."

       def get_toolset(self) -> AgentToolset[AgentDepsT]:
           return ExampleToolset(self.api_url)

``id``, ``description`` and ``defer_loading`` must be declared as dataclass
fields with ``kw_only=True``, as above. Written as plain class attributes, they
are overwritten by the base class defaults when the dataclass is instantiated,
and the capability loses its id and is never deferred. Ids must be unique
across capabilities.

To make the capability available in specs:

1. Import it in ``src/synagent/spec.py`` and add it to ``CAPABILITY_TYPES``.
2. List it under ``capabilities:`` in the specs that should use it.
3. Run ``synagent schema`` to update the editor schema.

Fields set from a spec must be YAML-representable (strings, numbers, booleans,
lists and mappings). For anything richer, override the ``from_spec`` class
method; ``synagent.subagents.SubAgents`` does this to build sub-agents from
spec paths.

Claude Code users can scaffold all of this with the ``/add-capability`` skill
in ``.claude/skills/``.

To check that it loads, start the server and query the health endpoint:

.. code-block:: sh

   synagent serve &
   curl http://localhost:8000/api/health

Adding a workflow
-----------------

Workflows are functions in ``src/synagent/workflows.py`` registered with the
``@workflow`` decorator. A workflow takes the agent and the ``REQUEST`` string
from the command line, and returns a Pydantic model, which ``synagent run``
prints as JSON:

.. code-block:: python

   @workflow("summary")
   def summary(agent: Agent[None, str], request: str) -> RouteSummary:
       result = agent.run_sync(
           user_prompt=request,
           instructions="Summarize the route.",
           output_type=RouteSummary,
       )
       return result.output

The name defaults to the function name. Registering a name twice raises an
error. Workflows are registered when ``synagent.workflows`` is imported, so a
workflow defined in another module must be imported from there.

Code style
----------

The project is formatted and linted with ruff. CI runs both on pushes and pull
requests to ``main``, and commits any formatting fixes it makes. To run them
locally:

.. code-block:: sh

   ruff format
   ruff check
