import asyncio

from synagent.validation._toolset import SynthesisValidationToolset, _match_product

EXPECTED = "CC(=O)Nc1ccc(O)cc1"
APPROVED_ANALOG = "CC(=O)Nc1ccc(N)cc1"  # 4096-bit Morgan/Tanimoto = 0.6154
LOW_SIMILARITY = "CCO"


def test_product_match_prefers_exact_canonical_match():
    matched, match_type, product, similarity = _match_product(
        EXPECTED, [APPROVED_ANALOG, EXPECTED]
    )

    assert matched is True
    assert match_type == "exact"
    assert product == EXPECTED
    assert similarity == 1.0


def test_product_match_approves_analog_above_point_six():
    matched, match_type, product, similarity = _match_product(
        EXPECTED, [APPROVED_ANALOG]
    )

    assert matched is True
    assert match_type == "analog"
    assert product == APPROVED_ANALOG
    assert similarity == 0.6154


def test_product_match_rejects_low_similarity_product():
    matched, match_type, product, similarity = _match_product(
        EXPECTED, [LOW_SIMILARITY]
    )

    assert matched is False
    assert match_type is None
    assert product == LOW_SIMILARITY
    assert similarity < 0.6


def test_product_match_can_preserve_original_exact_only_metric():
    matched, match_type, product, similarity = _match_product(
        EXPECTED, [APPROVED_ANALOG], analog_threshold=None
    )

    assert matched is False
    assert match_type is None
    assert product is None
    assert similarity is None


def test_validate_products_tool_uses_analog_matcher():
    result = asyncio.run(
        SynthesisValidationToolset().validate_products(
            reaction_smarts="[c:1][O:2]>>[c:1][N:2]",
            reactant_smiles=[EXPECTED],
            expected_product=EXPECTED,
        )
    )

    assert result[0] is True
    assert "approved product analog" in result[1]
    assert "Morgan/Tanimoto=0.6154" in result[1]


def test_validate_products_tool_preserves_exact_only_mode():
    result = asyncio.run(
        SynthesisValidationToolset().validate_products(
            reaction_smarts="[c:1][O:2]>>[c:1][N:2]",
            reactant_smiles=[EXPECTED],
            expected_product=EXPECTED,
            analog_similarity_threshold=None,
        )
    )

    assert result[0] is False
    assert "approved analog" in result[1]
