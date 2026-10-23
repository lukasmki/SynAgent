Model servers
=============

The default configuration runs four models, each on its own
`vLLM <https://docs.vllm.ai/>`_ server with an OpenAI-compatible API:

=========== ===================================== ====== =====================================
Server      Model                                 Port   Used by
=========== ===================================== ====== =====================================
gemma       ``google/gemma-4-26B-A4B-it``         8000   The agent itself
smileyllama ``kysun63/smileyllama-1b-reproduced`` 8001   ``generate_molecules``
synllama    ``kysun63/synllama-1b-reproduced``    8002   ``retrosynthesis``
linkllama   ``kysun63/linkllama-1b-reproduced``   8003   ``design_linker``
=========== ===================================== ====== =====================================

The launch scripts in ``vllm/`` start all four on a single node with four
40 GB A100 GPUs: Gemma on GPUs 1-3 with pipeline parallelism, and the three
1B models sharing GPU 0. The reasoning behind this layout and benchmark numbers
are in the comments at the top of ``vllm/config.yaml``.

Starting the servers
--------------------

``vllm/`` is a separate uv project, so vLLM's dependencies stay out of the
SynAgent environment. On the GPU node:

.. code-block:: sh

   cd vllm
   uv sync
   ./start.sh                     # all four servers
   ./start.sh gemma synllama      # only these
   ./start.sh --dry-run           # print the plan without starting anything

On a Slurm cluster, submit it as a batch job instead; ``start.sh`` carries its
own ``#SBATCH`` header (one node, four GPUs, four hours):

.. code-block:: sh

   sbatch vllm/start.sh
   sbatch -q debug -t 30 vllm/start.sh

``start.sh`` writes a ``vllm serve --config`` file and a log for each server to
``vllm/logs/``, and keeps model weights and compile caches in ``vllm/.cache/``.
Servers that share a GPU are started one after another, because vLLM sizes its
memory budget from what is free when it starts. If a server exits during
startup, the end of its log is printed and the others are stopped. Ctrl-C or
SIGTERM stops all of them.

Connecting SynAgent
-------------------

Set ``VLLM_BASE_URL`` in ``.env`` to the Gemma server:

.. code-block:: sh

   VLLM_BASE_URL='http://<gpu-node>:8000/v1'

The ``Generation`` capability derives the other three endpoints from it by
replacing the port with 8001, 8002 and 8003. If the servers run elsewhere, set
the URLs on the capability in the spec:

.. code-block:: yaml

   capabilities:
     - Generation:
         smileyllama_url: "http://host-a:9001/v1"
         synllama_url: "http://host-a:9002/v1"
         linkllama_url: "http://host-b:9003/v1"

Changing the layout
-------------------

Edit ``vllm/config.yaml``. Each entry under ``servers`` has ``gpus`` (the GPU
indices it may use) and a ``vllm`` block of ``vllm serve`` options, merged over
the shared ``defaults``. If you change a port, update ``VLLM_BASE_URL`` or the
``Generation`` URLs to match.

The Gemma server is started with ``language-model-only: true``, which skips the
vision encoder. Remove that line if the agent needs image input.
