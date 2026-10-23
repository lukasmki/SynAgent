Installation
============

Requirements
------------

- Python 3.13 or newer.
- `uv <https://docs.astral.sh/uv/>`_ (recommended) or ``pip``.
- Access to the model servers. The default configuration expects four
  vLLM servers, described in :doc:`model-servers`. You can also point the agent
  at a hosted model; see :ref:`model-override`.

Install the package
-------------------

.. code-block:: sh

   git clone https://github.com/lukasmki/SynAgent.git
   cd SynAgent

   # with uv
   uv sync

   # or with pip
   python3 -m venv .venv
   source .venv/bin/activate
   pip install .

Activate the environment if it isn't already, and check that the command line
tool is installed:

.. code-block:: sh

   source .venv/bin/activate
   synagent --help

The rest of this documentation assumes the environment is active. With uv, you
can instead prefix each command with ``uv run``.

``fpsim2`` is installed from a fork
(`lukasmki/FPSim2 <https://github.com/lukasmki/FPSim2>`_) that adds the
reaction template search used by the analogue search tools, so installation
needs network access to GitHub.

Configure the environment
-------------------------

SynAgent reads settings from a ``.env`` file in the working directory when it
starts. Copy the sample and fill it in:

.. code-block:: sh

   cp .env.sample .env

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Purpose
   * - ``VLLM_BASE_URL``
     - Endpoint of the agent model, for example ``http://<gpu-node>:8000/v1``.
       The generation tools reuse this host and swap in ports 8001-8003 for
       SmileyLlama, SynLlama and LinkLlama.
   * - ``CHEMSPACE_API_KEY``
     - Needed only if the ``Chemspace`` capability is enabled.
   * - ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ``GOOGLE_API_KEY``
     - Needed only when you run the agent on a model from that provider
       (``--model openai:...``, ``--model anthropic:...``, ``--model google:...``).

Build this documentation
------------------------

The documentation tools are in the ``dev`` dependency group. ``uv sync``
installs it by default; with pip (25.1 or newer), run
``pip install --group dev``.

.. code-block:: sh

   make -C docs html
   open docs/_build/html/index.html
