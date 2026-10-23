Workflows
=========

A workflow runs the agent with fixed instructions and a structured output type,
without a chat. Run one from the command line with:

.. code-block:: sh

   synagent run NAME REQUEST [--spec PATH] [--model NAME]

The result is printed to stdout as JSON. Workflows use the same agent and
capabilities as chat, so the spec must include the capabilities a workflow
relies on.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Name
     - ``REQUEST``
     - Output
   * - ``generation``
     - A request in plain text, such as a target SMILES and any constraints.
     - One route in SynLlama format
       (:class:`~synagent.models.SynLlamaFormat`).
   * - ``validation``
     - A route in SynLlama format, as JSON.
     - A :class:`~synagent.validation.ValidationReport` for that route.
   * - ``correction``
     - A ``ValidationReport``, as JSON.
     - A new ``ValidationReport`` for the corrected route.
   * - ``pipeline``
     - A request in plain text.
     - The final ``ValidationReport``.

``pipeline`` chains the other three: it generates a route, validates it, and
repeats correction until every building block is valid and every step passes,
or after 4 correction rounds. Check ``all_building_blocks_valid`` and
``all_reactions_passed`` in its output to see whether it converged.

The steps can also be run one at a time:

.. code-block:: sh

   synagent run generation "Propose a synthesis for CC(=O)Nc1ccc(O)cc1" > route.json
   synagent run validation "$(cat route.json)" > report.json
   synagent run correction "$(cat report.json)" > corrected.json

Route format
------------

``generation`` returns, and ``validation`` takes, a route in SynLlama format:

.. code-block:: json

   {
     "reactions": [
       {
         "reaction_number": 1,
         "reaction_template": "[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]",
         "reactants": ["CC(=O)O", "CN"],
         "product": "CNC(C)=O"
       }
     ],
     "building_blocks": ["CC(=O)O", "CN"]
   }

Adding a workflow
-----------------

See :ref:`extending:Adding a workflow`.
