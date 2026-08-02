"""Unit tests for the model_calls toolset — no servers required.

Ported from tests/test_llm_tools.py on the SynLlama-SmileyLlama-linkLlama-added-in
branch and retargeted at the capability architecture. The prompt builders are
now staticmethods on ModelCallsToolset rather than module functions.
"""

from __future__ import annotations

import json

import pytest
from conftest import make_completion

from synagent.model_calls import ModelCallsToolset

build_smiley = ModelCallsToolset._build_smileyllama_prompt


# ── SmileyLlama prompt construction ──────────────────────────────────────


class TestSmileyLlamaPrompt:
    def test_no_constraints(self):
        prompt = build_smiley(None, None, None, None, None, None, None)
        assert prompt == "Output a SMILES string for a drug like molecule:"

    def test_mw_and_logp(self):
        prompt = build_smiley("<= 500", "<= 5", None, None, None, None, None)
        assert "molecular weight <= 500" in prompt
        assert "LogP <= 5" in prompt

    def test_macrocycle_true(self):
        prompt = build_smiley(None, None, None, None, None, None, True)
        assert "containing a macrocycle" in prompt

    def test_macrocycle_false(self):
        prompt = build_smiley(None, None, None, None, None, None, False)
        assert "without macrocycles" in prompt

    def test_all_constraints(self):
        prompt = build_smiley("300-500", "1-3", "<= 5", "<= 10", "<= 10", ">= 0.25", None)
        for kw in (
            "molecular weight",
            "LogP",
            "hydrogen-bond donors",
            "hydrogen-bond acceptors",
            "rotatable bonds",
            "Fsp3",
        ):
            assert kw in prompt


# ── SmileyLlama generation ───────────────────────────────────────────────


class TestGenerateMolecules:
    @pytest.mark.anyio
    async def test_valid_smiles(self, mock_vllm):
        mock_vllm.completions.create.return_value = make_completion("CCO")
        result = await ModelCallsToolset().generate_molecules()
        assert result == [{"smiles": "CCO", "raw_output": "CCO"}]

    @pytest.mark.anyio
    async def test_trailing_text_is_trimmed(self, mock_vllm):
        # The model often appends commentary; only the first token is the SMILES.
        mock_vllm.completions.create.return_value = make_completion(
            "CCO  this molecule is ethanol"
        )
        result = await ModelCallsToolset().generate_molecules()
        assert result[0]["smiles"] == "CCO"
        assert "ethanol" in result[0]["raw_output"]

    @pytest.mark.anyio
    async def test_empty_output(self, mock_vllm):
        mock_vllm.completions.create.return_value = make_completion("")
        result = await ModelCallsToolset().generate_molecules()
        assert result[0]["smiles"] == ""

    @pytest.mark.anyio
    async def test_num_samples_drives_call_count(self, mock_vllm):
        mock_vllm.completions.create.return_value = make_completion("CCO")
        await ModelCallsToolset().generate_molecules(num_samples=3)
        assert mock_vllm.completions.create.call_count == 3


# ── SynLlama retrosynthesis ──────────────────────────────────────────────


class TestRetrosynthesis:
    @pytest.mark.anyio
    async def test_parses_plain_json(self, mock_vllm, sample_pathway):
        mock_vllm.completions.create.return_value = make_completion(
            json.dumps(sample_pathway)
        )
        result = await ModelCallsToolset().retrosynthesis(product_smiles="CCO")
        assert result["product"] == "CCO"
        assert result["pathways"][0]["parse_error"] is False
        assert result["pathways"][0]["pathway"]["steps"][0]["step"] == 1

    @pytest.mark.anyio
    async def test_parses_fenced_json(self, mock_vllm, sample_pathway):
        mock_vllm.completions.create.return_value = make_completion(
            f"```json\n{json.dumps(sample_pathway)}\n```"
        )
        result = await ModelCallsToolset().retrosynthesis(product_smiles="CCO")
        assert result["pathways"][0]["parse_error"] is False

    @pytest.mark.anyio
    async def test_unparseable_output_is_preserved(self, mock_vllm):
        mock_vllm.completions.create.return_value = make_completion("not json at all")
        result = await ModelCallsToolset().retrosynthesis(product_smiles="CCO")
        pathway = result["pathways"][0]
        assert pathway["parse_error"] is True
        assert pathway["raw_output"] == "not json at all"

    @pytest.mark.anyio
    async def test_num_pathways_drives_call_count(self, mock_vllm, sample_pathway):
        mock_vllm.completions.create.return_value = make_completion(
            json.dumps(sample_pathway)
        )
        await ModelCallsToolset().retrosynthesis(product_smiles="CCO", num_pathways=4)
        assert mock_vllm.completions.create.call_count == 4

    @pytest.mark.xfail(
        reason=(
            "Known fragility: the fence stripper uses str.strip(chars), not a "
            "substring strip, so it eats leading/trailing ` j s o n characters. "
            "A payload whose JSON is followed by a bare 'json' token, or fenced "
            "without a newline, can be mangled. Documented, not yet fixed."
        ),
        strict=False,
    )
    @pytest.mark.anyio
    async def test_fence_without_newline(self, mock_vllm, sample_pathway):
        mock_vllm.completions.create.return_value = make_completion(
            f"```json{json.dumps(sample_pathway)}```"
        )
        result = await ModelCallsToolset().retrosynthesis(product_smiles="CCO")
        assert result["pathways"][0]["parse_error"] is False


# ── LinkLlama ────────────────────────────────────────────────────────────


class TestDesignLinker:
    @pytest.mark.anyio
    async def test_returns_geometry_and_samples(self, mock_vllm):
        mock_vllm.completions.create.return_value = make_completion(
            json.dumps({"linker": "CCC", "reasoning": "short flexible chain"})
        )
        result = await ModelCallsToolset().design_linker(
            fragment1_smiles="[*]c1ccccc1",
            fragment2_smiles="[*]C(=O)O",
            distance_angstrom=5.0,
            angle_degrees=120.0,
            num_samples=1,
        )
        assert result["fragments"] == ["[*]c1ccccc1", "[*]C(=O)O"]
        assert result["geometry"]["distance_angstrom"] == 5.0
        assert len(result["samples"]) == 1
