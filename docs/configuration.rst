Configuration
=============

SynAgent is built from
`pydantic-ai agent specs <https://pydantic.dev/docs/ai/core-concepts/agent-spec/>`_: YAML (or
JSON) files that set an agent's model, instructions and capabilities. The
bundled specs are in ``src/synagent/specs/``:

``synagent.yaml``
   The main agent. Used unless ``--spec`` is given.
``synagent-nogpu.yaml``
   The main agent without the ``Generation`` tools, on a hosted model. See
   :ref:`no-gpu`.
``validator.yaml``
   A sub-agent with only the validation tools.
``retriever.yaml``
   A sub-agent with the analogue search and Chemspace tools.

To change the configuration, copy these files somewhere, edit them, and pass
the copy with ``--spec``.

The main spec
-------------

An abridged version of ``synagent.yaml``:

.. code-block:: yaml

   # yaml-language-server: $schema=synagent_schema.json
   model: vllm:google/gemma-4-26B-A4B-it
   instructions: |-
     You are SynAgent, a synthesis planning assistant.
     ...
   capabilities:
     - AnalogueSearch
     - Generation
     - SynthesisValidation

``model``
   A pydantic-ai model name, ``<provider>:<model>``. The ``vllm`` provider
   sends requests to ``VLLM_BASE_URL``.

``instructions``
   The system prompt. The bundled prompt sets out how the agent handles a
   retrosynthesis request: generate several pathways, check that each route is
   internally consistent, check building blocks and templates against the
   libraries, validate every step, correct failed steps (at most three attempts
   each), and validate the assembled route once more at the end.

``capabilities``
   The tool groups available to the agent, by class name. Each capability also
   adds its own usage instructions to the prompt. See :doc:`capabilities` for
   the full list.

Capability options
------------------

A capability with no options is written as its bare name. To set options, write
it as a mapping from the name to its fields:

.. code-block:: yaml

   capabilities:
     - AnalogueSearch
     - Generation:
         synllama_url: "http://gpu-node:8002/v1"
     - Chemspace: {api_key: "..."}
     - Storage: {path: "runs/storage.json"}

Each capability's fields are listed in :doc:`capabilities`. Unknown fields are
rejected when the spec loads.

.. _model-override:

Choosing a model
----------------

``--model`` replaces the spec's ``model`` for the main agent:

.. code-block:: sh

   synagent cli --model anthropic:claude-sonnet-5-5
   synagent cli --model google:gemini-3-flash-preview

Hosted providers need their API key in ``.env``. The ``Generation`` tools still
call the SynLlama-family vLLM servers, so without those servers the agent can
validate and search, but not generate.

.. _no-gpu:

Running without GPUs
--------------------

``synagent-nogpu.yaml`` needs no vLLM servers. It runs on
``anthropic:claude-opus-5-5``, so ``.env`` needs ``ANTHROPIC_API_KEY``, and it
leaves out ``Generation``. Its capabilities are ``AnalogueSearch``,
``SynthesisValidation`` and ``Chemspace``.

.. code-block:: sh

   synagent cli --spec src/synagent/specs/synagent-nogpu.yaml

Since there is no SynLlama, the agent writes the disconnections itself. For a
retrosynthesis request, its instructions have it:

1. validate the target and check whether it is already a building block;
2. propose 2-3 disconnections, each written as a forward template and the
   precursor SMILES;
3. replace each template that is not in the library with the closest library
   template from ``search_templates`` that still forms the product;
4. check the precursors against the building block library, swap in a library
   analogue only if the step still forms the same product, and disconnect any
   remaining precursor again (at most 5 steps per branch);
5. assemble, validate and correct the routes as the main spec does.

Routes the user supplies are checked and corrected the same way. Requests for
molecule generation or linker design are answered with a note that those tools
are not available in this configuration.

``Chemspace`` raises an error when the spec loads if ``CHEMSPACE_API_KEY`` is
not set. Without a key, remove it from ``capabilities``. Use ``--model`` to run
on another hosted model.

Sub-agents
----------

The ``SubAgents`` capability lets the main agent delegate a task to another
agent that has its own instructions and tools. Each entry is the path of
another spec file, relative to the spec that lists it:

.. code-block:: yaml

   capabilities:
     - SubAgents:
         agents:
           - validator.yaml
           - path: retriever.yaml
             timeout_seconds: 300
             max_calls: 5

Per-entry options:

==================== ===========================================================
``path``             Spec file of the sub-agent (required).
``description``      Replaces the ``description`` from the sub-agent's spec. The
                     main agent reads it to decide when to delegate.
``timeout_seconds``  Time limit for one delegation.
``max_calls``        Maximum delegations to this sub-agent per run.
``on_failure``       Message returned to the main agent instead of an error when
                     a delegation fails.
==================== ===========================================================

Options on ``SubAgents`` itself, all optional: ``max_depth``,
``forward_usage``, ``contain_errors`` and ``include_self``. These are passed to
the ``SubAgents`` capability of
`pydantic-ai-harness <https://github.com/pydantic/pydantic-ai-harness>`_; unset
options keep its defaults.

A sub-agent spec should set ``name`` and ``description``. If it leaves out
``model``, the sub-agent runs on the main agent's model, including one set with
``--model``. Specs that refer to each other in a cycle are rejected when they
load.

Editor support
--------------

Run ``synagent schema`` to regenerate ``synagent_schema.json``. Each bundled
spec starts with ``# yaml-language-server: $schema=synagent_schema.json``, which
gives validation and autocompletion of capability names and fields in editors
that use the YAML language server. Keep that line, adjusted to the schema's
path, in copied specs.
