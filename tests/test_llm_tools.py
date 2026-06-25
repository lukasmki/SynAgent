"""Tests for SmileyLlama, SynLlama, and LinkLlama tool functions."""

from __future__ import annotations

import json

from conftest import make_completion

from synagent.llm_tools import (
    LinkLlamaInput,
    SmileyLlamaInput,
    SynLlamaInput,
    _build_linkllama_prompt,
    _build_smileyllama_prompt,
    linkllama_generate,
    smileyllama_generate,
    synllama_retrosynthesis,
)


# ── SmileyLlama prompt construction ──────────────────────────────────────

class TestSmileyLlamaPrompt:
    def test_no_constraints(self):
        params = SmileyLlamaInput()
        prompt = _build_smileyllama_prompt(params)
        assert prompt == "Output a SMILES string for a drug like molecule:"

    def test_mw_and_logp(self):
        params = SmileyLlamaInput(mw_range="<= 500", logp_range="<= 5")
        prompt = _build_smileyllama_prompt(params)
        assert "molecular weight <= 500" in prompt
        assert "LogP <= 5" in prompt

    def test_macrocycle_true(self):
        params = SmileyLlamaInput(macrocycle=True)
        prompt = _build_smileyllama_prompt(params)
        assert "containing a macrocycle" in prompt

    def test_macrocycle_false(self):
        params = SmileyLlamaInput(macrocycle=False)
        prompt = _build_smileyllama_prompt(params)
        assert "without macrocycles" in prompt

    def test_all_constraints(self):
        params = SmileyLlamaInput(
            mw_range="300-500", logp_range="1-3", hbd_range="<= 5",
            hba_range="<= 10", rotatable_bonds="<= 10", fsp3=">= 0.25",
        )
        prompt = _build_smileyllama_prompt(params)
        for kw in ["molecular weight", "LogP", "hydrogen-bond donors",
                    "hydrogen-bond acceptors", "rotatable bonds", "Fsp3"]:
            assert kw in prompt


# ── SmileyLlama generation ───────────────────────────────────────────────

class TestSmileyLlamaGenerate:
    def test_valid_smiles(self, mock_openai_client):
        mock_openai_client.completions.create.return_value = make_completion("CCO")
        params = SmileyLlamaInput(num_samples=1)
        results = smileyllama_generate(params)
        assert len(results) == 1
        assert results[0]["smiles"] == "CCO"
        assert results[0]["raw_output"] == "CCO"

    def test_empty_response(self, mock_openai_client):
        mock_openai_client.completions.create.return_value = make_completion("")
        results = smileyllama_generate(SmileyLlamaInput(num_samples=1))
        assert results[0]["smiles"] == ""

    def test_multiple_samples(self, mock_openai_client):
        mock_openai_client.completions.create.return_value = make_completion("c1ccccc1")
        results = smileyllama_generate(SmileyLlamaInput(num_samples=3))
        assert len(results) == 3
        assert all(r["smiles"] == "c1ccccc1" for r in results)


# ── SynLlama ─────────────────────────────────────────────────────────────

class TestSynLlama:
    def test_valid_json_pathway(self, mock_openai_client):
        pathway = {
            "reactions": [{"reaction_number": 1, "template": "[C:1]>>[C:1]O", "reactants": ["CCO"]}],
            "building_blocks": ["CCO", "CC"]
        }
        mock_openai_client.completions.create.return_value = make_completion(json.dumps(pathway))
        params = SynLlamaInput(product_smiles="CCCO", num_pathways=1)
        result = synllama_retrosynthesis(params)
        assert result["product"] == "CCCO"
        assert len(result["pathways"]) == 1
        assert result["pathways"][0]["parse_error"] is False
        assert result["pathways"][0]["pathway"]["building_blocks"] == ["CCO", "CC"]

    def test_invalid_json_fallback(self, mock_openai_client):
        mock_openai_client.completions.create.return_value = make_completion("not valid json {{{")
        result = synllama_retrosynthesis(SynLlamaInput(product_smiles="CCO", num_pathways=1))
        assert result["pathways"][0]["parse_error"] is True
        assert "raw_output" in result["pathways"][0]

    def test_multiple_pathways(self, mock_openai_client):
        pathway = json.dumps({"reactions": [], "building_blocks": ["C"]})
        mock_openai_client.completions.create.return_value = make_completion(pathway)
        result = synllama_retrosynthesis(SynLlamaInput(product_smiles="C", num_pathways=5))
        assert len(result["pathways"]) == 5


# ── LinkLlama ────────────────────────────────────────────────────────────

class TestLinkLlamaPrompt:
    def test_basic_prompt(self):
        params = LinkLlamaInput(
            fragment1_smiles="[*]c1ccccc1",
            fragment2_smiles="[*]C1CCCCC1",
            distance_angstrom=4.5,
            angle_degrees=120.0,
        )
        prompt = _build_linkllama_prompt(params)
        assert "[*]c1ccccc1" in prompt
        assert "4.50 Angstroms" in prompt
        assert "120.00 degrees" in prompt

    def test_with_constraints(self):
        params = LinkLlamaInput(
            fragment1_smiles="[*]c1ccccc1",
            fragment2_smiles="[*]C1CCCCC1",
            distance_angstrom=4.5,
            angle_degrees=120.0,
            linker_type="chain",
            mw_range="<= 500",
            logp_range="<= 5",
        )
        prompt = _build_linkllama_prompt(params)
        assert "chain" in prompt
        assert "Molecular weight <= 500" in prompt
        assert "LogP <= 5" in prompt


class TestLinkLlamaGenerate:
    def test_valid_json(self, mock_openai_client):
        resp = json.dumps({"linker": "CCC", "reasoning": "short chain linker"})
        mock_openai_client.completions.create.return_value = make_completion(resp)
        params = LinkLlamaInput(
            fragment1_smiles="[*]c1ccccc1",
            fragment2_smiles="[*]C1CCCCC1",
            distance_angstrom=4.5,
            angle_degrees=120.0,
            num_samples=1,
        )
        result = linkllama_generate(params)
        assert len(result["samples"]) == 1
        assert result["samples"][0]["linker"] == "CCC"
        assert result["samples"][0]["parse_error"] is False

    def test_parse_error(self, mock_openai_client):
        mock_openai_client.completions.create.return_value = make_completion("broken json")
        params = LinkLlamaInput(
            fragment1_smiles="[*]c1ccccc1",
            fragment2_smiles="[*]C1CCCCC1",
            distance_angstrom=4.5,
            angle_degrees=120.0,
            num_samples=1,
        )
        result = linkllama_generate(params)
        assert result["samples"][0]["parse_error"] is True
        assert "raw_output" in result["samples"][0]
