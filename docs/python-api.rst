Python and notebooks
====================

``import synagent as sa`` gives a small interface for notebooks and scripts:
build the agent from a spec, chat with it, inspect the tools it called, and run
the :doc:`workflows <workflows>` on routes.

.. code-block:: python

   import synagent as sa

   agent = sa.SynAgent(spec="nogpu")
   chat = agent.chat()
   reply = await chat.ask("Is CC(=O)Oc1ccccc1C(=O)O a building block?")

   route = sa.Route.from_steps(
       [
           (
               "[OH:1][c:2].[Cl:3][C:4]=[O:5]>>[O:1]([c:2])[C:4]=[O:5]",
               ["Oc1ccccc1C(=O)O", "CC(=O)Cl"],
               "CC(=O)Oc1ccccc1C(=O)O",
           ),
       ]
   )
   report = await agent.validate(route)
   report.passed

``notebook-examples/notebook.ipynb`` walks through all of it. Jupyter
is an optional extra; install it and open the notebook:

.. code-block:: sh

   # with uv
   uv sync --extra jupyter

   # or with pip
   pip install ".[jupyter]"

   jupyter lab notebook-examples/notebook.ipynb

The agent
---------

:class:`~synagent.api.SynAgent` takes keyword arguments only:

``spec``
   A bundled spec name or a spec file path. ``sa.specs()`` lists the bundled
   names: ``synagent`` (the default, which needs the :doc:`model servers
   <model-servers>`), ``nogpu``, ``validator`` and ``retriever``.

``model``
   Overrides the spec's model, for example ``"google:gemini-3-flash-preview"``.

``trace``
   ``True`` writes a JSONL :ref:`trace <tracing>` to the default location, and a
   path writes it there. The file is at ``agent.trace_path``.

``dotenv``
   Load API keys from the nearest ``.env`` above the working directory.
   On by default.

The agent's display shows its spec, model and capabilities. The pydantic-ai
``Agent`` it wraps is ``agent.agent``.

Chat and replies
----------------

``agent.chat()`` starts a :class:`~synagent.api.Chat` that keeps its own
history, so each ``await chat.ask(...)`` sees the turns before it.
``chat.reset()`` clears the history, like ``/clear`` in the REPL. For a single
question with no history, use ``await agent.ask(...)``.

With ``show_tools=True``, the default, tool calls, thinking and the answer are
printed while the agent works, as in ``synagent cli``. With
``show_tools=False``, nothing is printed until the reply is displayed.

Both return a :class:`~synagent.api.Reply`:

``reply.text``
   The answer. A reply displays as Markdown, unless it was already printed
   while streaming; ``reply.show()`` displays it again.

``reply.tool_calls``
   One :class:`~synagent.api.ToolCall` per call, with ``name``, ``args`` and
   ``result``. It displays as a table.

``reply.usage``
   Requests, tool calls, tokens and cost.

``reply.result``
   The underlying pydantic-ai ``AgentRunResult``.

Routes and reports
------------------

``sa.Route`` is :class:`~synagent.models.SynLlamaFormat`.
``Route.from_steps`` builds one from ``(template, reactants, product)`` steps in
forward order. It numbers the steps and takes the building blocks to be the
reactants that no earlier step produces. ``Route.from_json`` reads the JSON that
``synagent run generation`` prints. A route draws each step with RDKit.

``sa.Report`` is :class:`~synagent.validation.ValidationReport`. Beyond its
fields it has ``report.passed`` (every building block valid and every step
passed, by the per-step results as well as the summary flags), ``report.failed_steps`` and ``report.to_polars()``, which gives one row
per step. It displays as a summary line and a table of steps.

Workflows
---------

The workflows are agent methods:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Method
     - Returns
     - Needs
   * - ``await agent.validate(route)``
     - ``Report``
     - ``SynthesisValidation`` or ``SubAgents``
   * - ``await agent.correct(report)``
     - ``Report``
     - ``SynthesisValidation`` or ``SubAgents``
   * - ``await agent.generate(request)``
     - ``Route``
     - ``Generation`` (SynLlama)
   * - ``await agent.pipeline(request, max_iter=4)``
     - ``Report``
     - ``Generation`` (SynLlama), and ``SynthesisValidation`` or ``SubAgents``

When the spec doesn't load the capability a method needs, the method raises a
``UserError`` naming it before any model request.

Scripts
-------

Every method that runs the agent is async. Outside a notebook, use its blocking
twin, which has a ``_sync`` suffix: ``ask_sync``, ``validate_sync``,
``correct_sync``, ``generate_sync``, ``pipeline_sync``, and ``Chat.ask_sync``.

.. code-block:: python

   import synagent as sa

   agent = sa.SynAgent(spec="nogpu")
   route = sa.Route.from_json(open("route.json").read())
   print(agent.validate_sync(route).passed)

Inside a running event loop, as in a notebook, the ``_sync`` methods raise an
error that points to the async method instead.

Compared with the lower-level API
---------------------------------

The same tour can be taken with the building blocks this interface is made of:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Task
     - Lower level
     - ``synagent`` interface
   * - Load ``.env``
     - ``load_dotenv()``
     - done when the agent is built
   * - Build
     - ``get_agent(model, SPEC_DIR / "synagent-nogpu.yaml")``
     - ``sa.SynAgent(spec="nogpu", model=...)``
   * - Chat
     - ``agent.run(prompt, message_history=result.all_messages())``
     - ``chat = agent.chat()``, then ``await chat.ask(prompt)``
   * - Tool calls
     - walk ``result.new_messages()`` for tool call and return parts
     - ``reply.tool_calls``
   * - Workflows
     - ``await validation(agent, route)``
     - ``await agent.validate(route)``
   * - Routes
     - ``SynLlamaFormat.model_validate({...})`` with numbered steps and the
       building blocks listed separately
     - ``sa.Route.from_steps([...])``
   * - Tracing
     - ``get_agent(..., trace=resolve_trace_path(None))``
     - ``sa.SynAgent(..., trace=True)``
