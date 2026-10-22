SynAgent
========

SynAgent is an agentic assistant for retrosynthesis planning and synthetic
pathway reconstruction. It pairs a general-purpose LLM agent with
`SynLlama <https://github.com/THGLab/SynLlama>`_ and related fine-tuned models,
and checks every proposed route against RDKit and local building block and
reaction template libraries before presenting it.

The agent is built with `pydantic-ai <https://ai.pydantic.dev/>`_. Its model,
instructions, tools and sub-agents are declared in YAML
:doc:`agent specs <configuration>`, and its tools are grouped into
:doc:`capabilities <capabilities>` that it loads on demand.

What it can do
--------------

- Propose retrosynthetic pathways for a target molecule with SynLlama.
- Validate each step of a route: SMILES and reaction SMARTS parsing, and
  running each reaction template to confirm it forms the expected product.
- Check building blocks and templates against the bundled libraries, and find
  similar alternatives when a step fails.
- Generate drug-like molecules under property constraints (SmileyLlama) and
  design linkers between two fragments (LinkLlama).
- Look up building block price and availability on Chemspace.

Ways to run it
--------------

- ``synagent serve``: a web chat UI and HTTP API.
- ``synagent cli``: an interactive terminal REPL that shows tool calls as they run.
- ``synagent run``: a fixed, non-interactive :doc:`workflow <workflows>` that
  prints a JSON result.

.. figure:: ../assets/synagent-flowchart.png
   :alt: Three ways SynAgent combines SynLlama with LLM agents
   :width: 90%
   :align: center

   (a) SynLlama proposes a route and an agent refines and validates it
   (the ``generation`` and ``validation`` workflows). (b) A co-planning agent
   calls SynLlama as a tool while chatting (``serve`` and ``cli``).
   (c) The validation and optimizer agents loop until the route passes
   (the ``pipeline`` workflow).

See :doc:`installation` to set it up, then :doc:`usage`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   usage
   configuration
   model-servers

.. toctree::
   :maxdepth: 2
   :caption: Reference

   capabilities
   workflows
   api

.. toctree::
   :maxdepth: 2
   :caption: Development

   extending
