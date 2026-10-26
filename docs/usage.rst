Usage
=====

All commands are subcommands of ``synagent``. Every command that builds an
agent accepts these options:

``--spec PATH``
   The agent spec to load. Defaults to the bundled
   ``src/synagent/specs/synagent.yaml``. See :doc:`configuration`.

``--model NAME``
   Run on this model instead of the one in the spec, for example
   ``--model google:gemini-3-flash-preview``.

``--logfire``
   Send traces of agent runs and tool calls to
   `Logfire <https://logfire.pydantic.dev/>`_. Logfire must already be
   authenticated (``logfire auth``) or have ``LOGFIRE_TOKEN`` set.

``--trace``
   Write a full JSONL trace of the session to
   ``.synagent/traces/YYYY-MM-DD/<HHMMSS>-<pid>.jsonl``. See
   :ref:`tracing`. It works with or without ``--logfire``.

``--trace-path PATH``
   Write the trace to ``PATH`` instead, and turn tracing on. A path ending in
   ``.jsonl`` is used as the file; any other path is treated as a directory,
   and a timestamped file is created inside it.

Web chat: ``synagent serve``
----------------------------

.. code-block:: sh

   synagent serve --host localhost --port 8000

This serves pydantic-ai's chat UI at ``http://localhost:8000``. The UI page is
fetched from a CDN on the first request, so the server needs network access the
first time it is opened.

The same server exposes an HTTP API:

``GET /api/health``
   Returns ``{"ok": true}`` once the agent has loaded.

``POST /api/chat``
   Runs one chat turn and streams the reply as server-sent events, in the
   `Vercel AI SDK stream format <https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol>`_.
   The top-level ``trigger`` field is required.

.. code-block:: sh

   curl -N -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{
       "trigger": "submit-message",
       "id": "req-1",
       "messages": [
         {"id": "msg-1", "role": "user",
          "parts": [{"type": "text", "text": "Validate this SMILES: CCO"}]}
       ]
     }'

The stream contains ``start``, ``text-delta``, tool call and ``finish`` events.
Errors from the model provider, such as a missing API key, appear in the server
log rather than in the response.

Terminal REPL: ``synagent cli``
-------------------------------

.. code-block:: sh

   synagent cli

An interactive chat in the terminal. Each tool call and its result are printed
as they happen, followed by the model's thinking (when the model returns it) and
its reply. Running ``synagent`` with no subcommand also starts the REPL, with
the default spec and model.

REPL commands:

=========== ===============================================
``/help``   Show the commands.
``/clear``  Start a new conversation.
``exit``    Quit. ``quit``, ``/exit`` and Ctrl-C also work.
=========== ===============================================

Workflows: ``synagent run``
---------------------------

.. code-block:: sh

   synagent run NAME REQUEST

Runs a predefined workflow once and prints its result as JSON on stdout, which
makes it suitable for scripts and batch runs. For example:

.. code-block:: sh

   synagent run pipeline "Propose a synthesis for CC(=O)Nc1ccc(O)cc1" > route.json

The available workflows and their inputs are listed in :doc:`workflows`.

.. _tracing:

Traces: ``synagent trace export``
---------------------------------

With ``--trace`` or ``--trace-path``, ``serve``, ``cli`` and ``run`` record every
model request and response, tool call and agent run to a JSONL file:

.. code-block:: sh

   synagent run pipeline "Propose a synthesis for CC(=O)Nc1ccc(O)cc1" --trace
   synagent cli --trace-path runs/session.jsonl

Each line is one record, and the ``record`` field gives its type:

``model_call``
   One model request and its response: the messages sent, the instructions
   and tool definitions in effect at that step, and the reply, including
   thinking and tool calls.

``tool_result``
   One tool execution, with its arguments and its result or error.

``run``
   One finished agent run, with its status and counts of steps, tool calls
   and retries.

``label``
   Ground truth added after a run. The ``pipeline`` workflow labels each run
   with whether its final route was ``valid``.

Every record carries the ``run_id`` of the run it belongs to and the ``agent``
that made it, which is the ``name`` from the agent's spec. Sub-agent runs are
traced as runs of their own, and their records also hold the
``parent_run_id`` and ``parent_tool_call_id`` of the delegation that started
them. Records from concurrent runs, such as several chats on one server, can
interleave in one file, and the ``run_id`` separates them again.

Traces include full prompts and results. ``.synagent/`` is in ``.gitignore``,
so they are not committed by accident.

``synagent trace export`` converts traces into a supervised fine-tuning dataset:

.. code-block:: sh

   synagent trace export '.synagent/traces/**/*.jsonl' -o dataset.jsonl

Quote the pattern so that it reaches ``synagent`` instead of being expanded by
the shell. Several files or patterns can be given.

Each ``model_call`` becomes one sample in the chat format that TRL, axolotl,
Unsloth and OpenAI fine-tuning read: a ``messages`` list that ends with the
recorded reply as the assistant turn, and a ``tools`` list of the tools the
model could call. The instructions become the system message. Model calls that
failed are skipped, and identical samples are written once.

Options:

``-o``, ``--out PATH``
   The dataset file to write. Required.

``--agent NAME``
   Keep only the calls made by this agent, for example ``synagent`` for the
   main agent or ``validator`` for a sub-agent.

``--only-successful``
   Keep only runs that finished without an error and were not labelled
   invalid. Only the ``pipeline`` workflow writes labels, so for other runs
   this checks for errors only.

``--min-steps N``
   Drop runs with fewer than ``N`` model requests.

``--include-thinking``
   Keep the model's thinking as ``reasoning_content`` on each assistant turn.
   It is left out by default.

For example, to keep only the main agent's calls from multi-step pipeline runs
that produced a valid route:

.. code-block:: sh

   synagent trace export '.synagent/traces/**/*.jsonl' -o dataset.jsonl \
       --only-successful --agent synagent --min-steps 2

Spec schema: ``synagent schema``
--------------------------------

.. code-block:: sh

   synagent schema [PATH]

Writes a JSON schema for agent spec files, by default to
``src/synagent/specs/synagent_schema.json``. The bundled specs reference this
file in their first line, so editors with the YAML language server validate and
autocomplete them. Rerun it after adding or changing a capability.
