API reference
=============

Generated from the source docstrings.

Building agents
---------------

.. automodule:: synagent.spec
   :members: load_agent, write_schema, resolve_spec_path, CAPABILITY_TYPES

.. autofunction:: synagent.synagent.get_agent

Workflows
---------

.. automodule:: synagent.workflows
   :members: workflow, get_workflow, generation, validation, correction, pipeline
   :undoc-members:

Data models
-----------

.. autoclass:: synagent.models.SynLlamaFormat
   :no-show-inheritance:

.. autoclass:: synagent.models.SynLlamaReaction
   :no-show-inheritance:

.. autoclass:: synagent.validation.ValidationReport
   :no-show-inheritance:

.. autoclass:: synagent.validation._models.ReactionResult
   :no-show-inheritance:

.. autoclass:: synagent.validation._models.BuildingBlockResult
   :no-show-inheritance:

Capabilities
------------

.. autoclass:: synagent.analogues.AnalogueSearch
   :members: get_instructions

.. autoclass:: synagent.generation.Generation
   :members: get_instructions

.. autoclass:: synagent.validation.SynthesisValidation
   :members: get_instructions

.. autoclass:: synagent.chemspace.Chemspace
   :members: get_instructions

.. autoclass:: synagent.storage.Storage(path='.synagent/storage.json', *, id='storage', description=..., defer_loading=True)
   :members: get_instructions

.. autoclass:: synagent.scoring.Scoring
   :members: get_instructions

.. autoclass:: synagent.subagents.SubAgents
   :members: from_spec

Toolsets
--------

The tool docstrings below are what the agent sees as each tool's description.

.. autoclass:: synagent.analogues.AnalogueSearchToolset
   :members: search_building_blocks, search_templates, search_building_blocks_by_template, check_building_blocks, check_templates

.. autoclass:: synagent.generation.GenerationToolset
   :members: generate_molecules, retrosynthesis, design_linker

.. The create_report docstring is laid out for the model, not for reST, so it is
   left out here; see the SynthesisValidation section of capabilities.rst.

.. autoclass:: synagent.validation.SynthesisValidationToolset
   :members: validate_smiles, validate_reaction_smarts, validate_products, reverse_reaction

.. autoclass:: synagent.chemspace.ChemspaceToolset
   :members: search_exact, search_similarity, search_substructure

.. autoclass:: synagent.storage.StorageToolset
   :members: save_record, get_record, list_records

.. autoclass:: synagent.scoring.ScoringToolset
   :members: score_molecules, score_reactions, score_paths
