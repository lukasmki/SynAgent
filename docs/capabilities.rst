Capabilities
============

A capability is a group of related tools plus instructions on how to use them.
Specs list capabilities by class name (see :doc:`configuration`). All SynAgent
capabilities are deferred: the agent sees each one's id and description, and
the tools are loaded only when the agent decides it needs them.

The default ``synagent.yaml`` enables ``AnalogueSearch``, ``Generation`` and
``SynthesisValidation``. The others are listed in the spec but commented out.

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Class
     - Id
     - Purpose
   * - ``AnalogueSearch``
     - ``analogue-search``
     - Search and check the local building block and reaction template libraries.
   * - ``Generation``
     - ``generation``
     - Sample from SmileyLlama, SynLlama and LinkLlama.
   * - ``SynthesisValidation``
     - ``synthesis-validation``
     - Check a route step by step with RDKit.
   * - ``Chemspace``
     - ``chemspace``
     - Building block price and availability from the Chemspace API.
   * - ``Storage``
     - ``storage``
     - Save and look up records in a local JSON file.
   * - ``Scoring``
     - ``scoring``
     - Placeholder for scoring molecules, reactions and routes.
   * - ``SubAgents``
     - (from the harness)
     - Delegate to other agents defined by spec files. See
       :ref:`configuration:Sub-agents`.

SMILES and reaction SMARTS passed to any tool must be bare strings. The
``<bb>`` and ``<rxn>`` tags in SynLlama output have to be removed first.

AnalogueSearch
--------------

Searches two libraries bundled with the package in
``src/synagent/analogues/data/``:

- ``building_blocks.h5``: a building block catalog.
- ``reactions.h5``: reaction templates.

Both are `FPSim2 <https://github.com/lukasmki/FPSim2>`_ databases of RDKit
pattern fingerprints, loaded into memory the first time the capability is used.
Similarity searches use cosine similarity on these fingerprints, so their hits
can include false positives.

No fields.

======================================= ==========================================================
Tool                                    Description
======================================= ==========================================================
``search_building_blocks``              Building blocks similar to each query SMILES.
                                        ``threshold`` (default 0.7), ``max_results`` (default 100).
``search_templates``                    Library templates similar to each query template. Same
                                        options.
``search_building_blocks_by_template``  Building blocks matching each reactant and product pattern
                                        of a template, by substructure search.
``check_building_blocks``               Whether each SMILES is in the library, by exact match of
                                        canonical SMILES.
``check_templates``                     For each template: ``in_library``, ``reversed`` (the
                                        library has it with the two sides swapped) or
                                        ``not_in_library``. Templates are compared as strings.
======================================= ==========================================================

To rebuild the libraries, see ``python data/create_fp_dbs.py``.
It writes ``reactions.h5`` and ``building_blocks.h5`` to ``data/`` (skipping any
that already exist); copy them into ``src/synagent/analogues/data/``.

Generation
----------

Calls the three fine-tuned 1B models on their vLLM servers (see
:doc:`model-servers`). Each call can draw several samples at once. Results are
unparsed model output and are not validated, so the agent is instructed to
validate them before using them.

========================== ====================================================
Tool                       Description
========================== ====================================================
``generate_molecules``     SmileyLlama. Drug-like SMILES under optional
                           constraints on molecular weight, logP, H-bond donors
                           and acceptors, rotatable bonds, fraction sp3 and
                           macrocycles. Up to 50 samples; ``temperature`` 0.8.
``retrosynthesis``         SynLlama. Candidate pathways for a target SMILES as
                           raw JSON text: reaction steps with SMARTS templates,
                           reactants, products and building blocks. Up to 20
                           pathways; ``temperature`` 1.5 for diversity.
``design_linker``          LinkLlama. Linkers joining two fragments, each with a
                           ``[*]`` dummy atom at the attachment point, given the
                           distance (Å) and angle between the attachment points.
                           Optional topology and property constraints. Up to 100
                           samples.
========================== ====================================================

Constraints are written as a comparison or range, such as ``"<= 500"`` or
``"300-500"``.

Fields:

====================== ======================================= ==========================================
Field                  Default                                 Description
====================== ======================================= ==========================================
``smileyllama_url``    ``VLLM_BASE_URL`` with port 8001        SmileyLlama server.
``smileyllama_model``  ``kysun63/smileyllama-1b-reproduced``   Served model name.
``synllama_url``       ``VLLM_BASE_URL`` with port 8002        SynLlama server.
``synllama_model``     ``kysun63/synllama-1b-reproduced``      Served model name.
``linkllama_url``      ``VLLM_BASE_URL`` with port 8003        LinkLlama server.
``linkllama_model``    ``kysun63/linkllama-1b-reproduced``     Served model name.
``api_key``            ``"EMPTY"``                             Key sent to the vLLM servers. Must not be
                                                               empty, or the OpenAI client falls back to
                                                               ``OPENAI_API_KEY``.
====================== ======================================= ==========================================

SynthesisValidation
-------------------

Checks a route with RDKit and returns a structured report. Its instructions
tell the agent to check all molecules and templates in one call each, run every
step even after an earlier one fails, and retry a failing step once with the
template reversed or the reactants reordered.

============================ ==============================================================
Tool                         Description
============================ ==============================================================
``validate_smiles``          ``True`` for each SMILES that parses, otherwise the RDKit error.
``validate_reaction_smarts`` ``True`` for each reaction SMARTS that parses, otherwise the
                             error.
``validate_products``        Runs one template on its reactants, in the order of the
                             template's reactant patterns, and checks whether the expected
                             product forms. With no expected product, lists the products.
``reverse_reaction``         Swaps the reactant and product sides of each template.
``create_report``            Assembles the final :class:`~synagent.validation.ValidationReport`.
============================ ==============================================================

Each step in the report has a ``status`` of ``passed`` or ``failed``. A failed
step has one of these ``failure_mode`` values:

``invalid_template``
   The template could not be parsed.
``invalid_reactants``
   A reactant SMILES is invalid.
``no_products``
   The template did not match the reactants.
``product_mismatch``
   Products formed, but not the expected one.

The report also has ``all_building_blocks_valid`` and ``all_reactions_passed``
flags, which the ``pipeline`` :doc:`workflow <workflows>` uses to decide whether
to keep correcting.

No fields.

Chemspace
---------

Searches the `Chemspace <https://chem-space.com/>`_ catalog for price, vendor and
lead time. Its instructions limit it to requests that ask for pricing.

=================================== ====================================
Tool                                Search type
=================================== ====================================
``chemspace_search_exact``          Exact structure match.
``chemspace_search_similarity``     Similar structures.
``chemspace_search_substructure``   Structures containing the query.
=================================== ====================================

Each search takes a SMILES and optionally a ship-to country (default ``US``), a
page size and page number, and product categories: in-stock building blocks
(``CSSB``), in-stock screening compounds (``CSSS``), make-on-demand building
blocks (``CSMB``), make-on-demand screening compounds (``CSMS``) and custom
synthesis (``CSCS``). The default is ``CSSB`` and ``CSMB``.

Fields:

``api_key``
   Chemspace API key. Defaults to ``CHEMSPACE_API_KEY``. The access token is
   cached in the system temp directory as ``.chemspace_token_cache`` and
   refreshed when it expires.

Storage
-------

Saves records (an id and an arbitrary JSON object) to a local JSON file.

================ ==========================================================
Tool             Description
================ ==========================================================
``save_record``  Saves a record, replacing any record with the same id.
``get_record``   Looks up a record by id, by key/value filters on its data,
                 or both. Returns the first match.
``list_records`` Returns every record.
================ ==========================================================

Fields:

``path``
   The JSON file. Defaults to ``.synagent/storage.json`` in the working
   directory.

Scoring
-------

Not implemented yet. ``score_molecules``, ``score_reactions`` and
``score_paths`` are defined but return empty results.
