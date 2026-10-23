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

Spec schema: ``synagent schema``
--------------------------------

.. code-block:: sh

   synagent schema [PATH]

Writes a JSON schema for agent spec files, by default to
``src/synagent/specs/synagent_schema.json``. The bundled specs reference this
file in their first line, so editors with the YAML language server validate and
autocomplete them. Rerun it after adding or changing a capability.
