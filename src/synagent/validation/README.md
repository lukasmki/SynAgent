# Synthesis Path Validation

The `synthesis-validation` capability gives a Pydantic AI agent the tools to verify retrosynthesis pathways using RDKit. It checks SMILES strings for structural validity, verifies reaction SMARTS patterns, and simulates reactions to confirm that expected products are formed. In the SynAgent pipeline the master agent calls validation first — before cost analysis or building-block search — so that downstream agents only see chemically consistent routes.

## Capability class

```python
@dataclass
class SynthesisValidation(AbstractCapability[None]):
    id = "validation"
    description = "Use for synthesis path validation."
    defer_loading = True
```

| Attribute | Value | Notes |
| --- | --- | --- |
| `id` | `"synthesis-validation"` | Identifier used when the master agent selects capabilities |
| `defer_loading` | `True` | Toolset is instantiated only when the capability is first used |
| `get_instructions()` | `"Validate all SMILES strings and then each reaction step."` | Injected into the agent's system prompt when the capability is active |
| `get_toolset()` | `ValidationToolset()` | Returns the four RDKit-backed tools described below |

### Attaching to an agent

```python
from pydantic_ai import Agent
from synagent.validation import SynthesisValidation

agent = Agent(capabilities=[SynthesisValidation()])
```

## Tools

| Tool | Input | Output |
| --- | --- | --- |
| `validate_smiles` | `list[str]` | `dict[str, bool]` |
| `validate_reaction_smarts` | `list[str]` | `dict[str, bool \| str]` |
| `validate_products` | `reaction_smarts`, `reactant_smiles`, `expected_product` | `tuple[bool, str]` |
| `reverse_reaction` | `list[str]` | `list[str]` |

### `validate_smiles`

Checks whether each SMILES string can be parsed into a valid RDKit molecule.

```python
validate_smiles(smiles: list[str]) -> dict[str, bool]
```

Returns `{smiles: is_valid}` for every input string. Uses `Chem.MolFromSmiles` — `None` return means invalid.

### `validate_reaction_smarts`

Checks whether each reaction SMARTS string is parseable and initializable.

```python
validate_reaction_smarts(reaction_smarts: list[str]) -> dict[str, bool | str]
```

Returns `True` on success. If `ReactionFromSmarts` or `rxn.Initialize()` raises `ValueError`, the value is the error message string rather than `False` — preserving the original error for debugging.

### `validate_products`

Simulates a reaction and checks whether the expected product is produced.

```python
validate_products(
    reaction_smarts: str,
    reactant_smiles: list[str],
    expected_product: str | None,
) -> tuple[bool, str]
```

Returns `(success, message)`. Failure modes in order:

| Failure | Message |
| --- | --- |
| Invalid reaction SMARTS | `` "`reaction_smarts` could not be parsed." `` |
| Invalid reactant SMILES | `` "`reactant_smiles` contains invalid SMILES strings." `` |
| Reaction produced no products | `"Reaction produced no products."` |
| Product mismatch | `"Reaction did not produce expected product, instead got {products}"` |

When `expected_product` is `None` the reaction is still run and the actual products are returned in the message.

### `reverse_reaction`

Flips reactants and products in a list of reaction SMARTS strings.

```python
reverse_reaction(reaction_smarts: list[str]) -> list[str]
```

Splits on `>>` and reverses the two sides. Useful when converting a forward reaction template into its retrosynthetic equivalent.

## Related types

Defined in `src/synagent/models.py`:

```python
type Smiles = Annotated[str, "Molecule SMILES string"]
type ReactionSmarts = Annotated[str, "Reaction SMARTS string"]
```

The standalone `validate.py` module produces a `SynLlamaReport` that aggregates validation results across an entire pathway:

| Field | Type | Description |
| --- | --- | --- |
| `reactions` | `list[ReactionResult]` | Per-step results with status, failure mode, and actual products |
| `building_blocks` | `list[BuildingBlockResult]` | SMILES validity for each starting material |
| `all_reactions_passed` | `bool` | `True` only if every reaction step produced the expected product |
| `all_building_blocks_valid` | `bool` | `True` only if every building block passed SMILES validation |
| `target_molecule` | `Smiles` | The final product of the synthesis route |

`ReactionResult.failure_mode` is one of `"invalid_reactants"`, `"no_products"`, `"product_mismatch"`, `"invalid_template"`, or `None` (success) — a structured enum for downstream agents to branch on without parsing free text.
