# Chemspace Capability

The `chemspace` capability gives a Pydantic AI agent tools to search the [Chemspace](https://chem-space.com/) API for building block availability and pricing. It requires `ChemspaceDeps` as the agent's dependency type so that the token manager can be injected at runtime.

## Capability class

```python
@dataclass
class ChemspaceCapability(AbstractCapability[ChemspaceDeps]):
    id = "chemspace"
    description = "Search Chemspace for building block price and availability."
    defer_loading = True
```

| Attribute | Value | Notes |
| --- | --- | --- |
| `id` | `"chemspace"` | Used when the master agent selects capabilities |
| `defer_loading` | `True` | Toolset is instantiated only when the capability is first loaded |
| `get_instructions()` | See source | Injected into the agent's system prompt; advises exact search first, similarity as fallback |
| `get_toolset()` | `ChemspaceToolset()` | Wraps the three search functions from `chemspacetool.py` |

### Attaching to an agent

The agent must declare `deps_type=ChemspaceDeps` and receive a `ChemspaceDeps` instance at run time:

```python
from pydantic_ai import Agent
from synagent.chemspace import ChemspaceCapability
from synagent.chemspacetool import ChemspaceDeps
from synagent.tokenmanager import ChemspaceTokenManager

agent = Agent(deps_type=ChemspaceDeps, capabilities=[ChemspaceCapability()])

mgr = ChemspaceTokenManager()  # reads CHEMSPACE_API_KEY from env
result = await agent.run("Find exact matches for CCO", deps=ChemspaceDeps(mgr=mgr))
```

## Tools

All three tools share the same optional parameters.

| Tool | Search type | Chemspace endpoint |
| --- | --- | --- |
| `search_exact` | Exact structure match | `POST /v4/search/exact` |
| `search_substructure` | Substructure match | `POST /v4/search/sub` |
| `search_similarity` | Similarity match | `POST /v4/search/sim` |

### Common parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `smiles` | `str` | required | SMILES string to search for |
| `ship_to_country` | `str` | `"US"` | Two-letter ISO country code |
| `count` | `int` | `10` | Max results per page |
| `page` | `int` | `1` | Page number |
| `categories` | `list[ProductCategory]` | `["CSSB", "CSMB"]` | Product categories to search |

`ProductCategory` values: `"CSSB"` (screening blocks, small), `"CSSS"` (screening blocks, standard), `"CSMB"` (make-on-demand blocks, small), `"CSMS"` (make-on-demand blocks, standard), `"CSCS"` (custom synthesis).

All three functions return the raw JSON response dict from the Chemspace API.

### Authentication

Tools access the token via `ctx.deps.mgr` (`ChemspaceTokenManager`). The manager caches the OAuth token in `/tmp/.token_cache` and auto-refreshes before expiry. If the API returns `401`, the token is force-refreshed once and the request is retried automatically.

## Related modules

| File | Purpose |
| --- | --- |
| `src/synagent/chemspacetool.py` | Source functions registered by the toolset; also defines `ChemspaceDeps` and `ChemspaceSearchInput` |
| `src/synagent/tokenmanager.py` | `ChemspaceTokenManager` — OAuth lifecycle and token cache |
