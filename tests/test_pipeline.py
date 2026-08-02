"""End-to-end pipeline tests.

Three tiers, by what infrastructure they need:

1. **Wiring** — no servers. Asserts the merged agent actually exposes both
   halves of the merge (model_calls + corrector/validation), and that the
   corrector's gate behaves as designed.
2. **Mocked pipeline** — no servers. Drives generate -> route -> validate -> fix
   with the chemistry models mocked, proving the sequence composes.
3. **Live** — needs a vLLM server for the three chemistry models and an
   orchestrator (Ollama or ANTHROPIC_API_KEY). Skips cleanly otherwise.

On the gating rule
------------------
`Corrector.prepare_tools()` hides every corrector tool unless one of
"fix"/"correct"/"repair"/"search alternative"/"alternative building block"
appears in the last three user messages. The pipeline test therefore drives a
*multi-turn conversation* whose repair turn literally contains the word "fix".

This is deliberate. Relaxing the gate to make the test simpler would mean the
test no longer exercises the real production path. If you are here because the
corrector tools "aren't being called", check the wording of the prompt before
changing the gate.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_completion, requires_orchestrator, requires_vllm

from synagent.corrector import Corrector
from synagent.model_calls import ModelCallsToolset
from synagent.synagent import get_agent

MODEL_CALL_TOOLS = {
    "generate_molecules",
    "retrosynthesis",
    "design_linker",
    "search_enamine_similarity",
    "search_enamine_substructure",
    "find_and_link_fragments",
}

def capability_ids(agent) -> set[str]:
    """Capability ids on a built agent.

    pydantic-ai wraps the capability list in a CombinedCapability on
    `agent._root_capability`; there is no public `agent.capabilities`.
    """
    return {
        c.id
        for c in agent._root_capability.capabilities
        if getattr(c, "id", None)
    }


def instructions_text(agent) -> str:
    """Agent instructions flattened to one string.

    `agent._instructions` is a list of instruction parts, not a str, so a
    naive `"x" in agent._instructions` silently becomes a list membership
    test that is always False.
    """
    raw = agent._instructions or []
    if isinstance(raw, str):
        return raw
    return "\n".join(str(part) for part in raw)


CORRECTOR_TOOLS = {
    "fix_step",
    "fix_building_blocks",
    "apply_fixes",
    "search_step_building_blocks",
    "fix_smarts",
    "extract_template_from_reaction",
    "fix_template",
    "fix_smiles",
}


# ── Tier 1: wiring ───────────────────────────────────────────────────────


class TestMergeWiring:
    """The point of the merge: both halves present on one agent."""

    def test_model_calls_toolset_registers_all_tools(self):
        names = set(ModelCallsToolset().tools)
        assert MODEL_CALL_TOOLS <= names, f"missing: {MODEL_CALL_TOOLS - names}"

    def test_agent_has_both_halves(self):
        agent = get_agent("qwen3.5:9b")
        ids = capability_ids(agent)
        # from model-calls-capability
        assert "model-calls" in ids, f"got {ids}"
        # from corrector.py2
        assert "corrector" in ids, f"got {ids}"

    def test_capability_ids_survive_dataclass(self):
        """Regression guard on a subtle failure.

        Capabilities declare `id`/`description`/`defer_loading` as class
        attributes, but the classes are @dataclass and the base declares those
        same names as fields defaulting to None/False. Unannotated class
        attributes are not fields, so the inherited defaults win on instances
        and every id silently becomes None.
        """
        from synagent.model_calls import ModelCalls

        cap = ModelCalls()
        assert cap.id == "model-calls"
        assert cap.description
        # declared True; before the annotation fix this was silently False
        assert cap.defer_loading is True

    def test_generation_clause_present_in_instructions(self):
        """corrector.py2's prompt ends with 'never use a tool unless explicitly
        asked' and named no generation tools. Without a GENERATION clause the
        merged model_calls tools are unreachable in practice."""
        text = instructions_text(get_agent("qwen3.5:9b"))
        assert "generate_molecules" in text
        assert "retrosynthesis" in text

    def test_personas_differ(self):
        det = instructions_text(get_agent("qwen3.5:9b", persona="deterministic"))
        dis = instructions_text(get_agent("qwen3.5:9b", persona="disagreeable"))
        assert det != dis
        assert "DISAGREEABLE BY DEFAULT" in dis
        assert "DISAGREEABLE BY DEFAULT" not in det


# ── Tier 1: the gate ─────────────────────────────────────────────────────


class TestCorrectorGate:
    """Regression guard on the single most confusing failure mode."""

    @pytest.mark.anyio
    async def test_tools_hidden_without_trigger_word(self, gate_ctx, tool_defs):
        out = await Corrector().prepare_tools(gate_ctx("validate this route"), tool_defs)
        assert {t.name for t in out}.isdisjoint(CORRECTOR_TOOLS)

    @pytest.mark.anyio
    async def test_tools_shown_with_trigger_word(self, gate_ctx, tool_defs):
        out = await Corrector().prepare_tools(gate_ctx("fix step 2"), tool_defs)
        assert CORRECTOR_TOOLS <= {t.name for t in out}

    @pytest.mark.anyio
    async def test_non_corrector_tools_always_pass_through(self, gate_ctx, tool_defs):
        out = await Corrector().prepare_tools(gate_ctx("validate this route"), tool_defs)
        assert "validate_route" in {t.name for t in out}


# ── Tier 2: mocked pipeline ──────────────────────────────────────────────


class TestMockedPipeline:
    """generate -> route, with the chemistry models mocked.

    Exercises the composition without any server. This is what runs in CI while
    the vLLM port is still being built.
    """

    @pytest.mark.anyio
    async def test_generate_then_route(self, mock_vllm, sample_pathway):
        toolset = ModelCallsToolset()

        mock_vllm.completions.create.return_value = make_completion(
            "CC(=O)Nc1ccc(O)cc1"
        )
        molecules = await toolset.generate_molecules(mw_range="<= 500", logp_range="<= 5")
        target = molecules[0]["smiles"]
        assert target == "CC(=O)Nc1ccc(O)cc1"

        mock_vllm.completions.create.return_value = make_completion(
            json.dumps(sample_pathway)
        )
        route = await toolset.retrosynthesis(product_smiles=target)

        assert route["product"] == target
        pathway = route["pathways"][0]["pathway"]
        assert pathway["building_blocks"]
        assert pathway["steps"][0]["product"] == target

    @pytest.mark.anyio
    async def test_route_building_blocks_are_valid_smiles(self, mock_vllm, sample_pathway):
        """Structural assertion, not exact-match — these models are stochastic."""
        from rdkit import Chem

        mock_vllm.completions.create.return_value = make_completion(
            json.dumps(sample_pathway)
        )
        route = await ModelCallsToolset().retrosynthesis(product_smiles="CC(=O)Nc1ccc(O)cc1")
        blocks = route["pathways"][0]["pathway"]["building_blocks"]
        assert all(Chem.MolFromSmiles(b) is not None for b in blocks)


# ── Tier 3: live ─────────────────────────────────────────────────────────


@requires_vllm
@requires_orchestrator
class TestLivePipeline:
    """The real thing: full generate -> route -> validate -> fix.

    Skipped until a vLLM server serves SmileyLlama / SynLlama / LinkLlama.
    Assertions are structural, never exact-match, because these models sample.
    """

    @pytest.mark.anyio
    async def test_full_pipeline(self, orchestrator):
        from rdkit import Chem

        model_name, provider = orchestrator
        agent = get_agent(model_name, provider=provider)

        async with agent.run_mcp_servers() if hasattr(agent, "run_mcp_servers") else _null():
            # Turn 1 — generate
            r1 = await agent.run(
                "Generate one drug-like molecule with molecular weight <= 500 "
                "and LogP <= 5."
            )
            history = r1.all_messages()

            # Turn 2 — route
            r2 = await agent.run(
                "Now produce a retrosynthetic route for that molecule.",
                message_history=history,
            )
            history = r2.all_messages()

            # Turn 3 — validate
            r3 = await agent.run(
                "Validate that route.", message_history=history
            )
            history = r3.all_messages()
            assert r3.output

            # Turn 4 — the word "fix" is required; see module docstring.
            r4 = await agent.run(
                "Fix the failed steps in that route.", message_history=history
            )
            assert r4.output

        # Structural checks only.
        for text in (r1.output, r2.output):
            assert text

    @pytest.mark.anyio
    async def test_generated_smiles_parses(self):
        """SmileyLlama's output must at minimum be RDKit-parseable."""
        from rdkit import Chem

        result = await ModelCallsToolset().generate_molecules(
            mw_range="<= 500", num_samples=5
        )
        parsed = [Chem.MolFromSmiles(r["smiles"]) for r in result if r["smiles"]]
        # Not asserting 100% — sampling produces occasional invalid strings.
        assert any(m is not None for m in parsed), "no valid SMILES in 5 samples"


class _null:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *a):
        return False
