import httpx
from typing import Literal, List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from .tokenmanager import ChemspaceTokenManager


ProductCategory = Literal["CSSB", "CSSS", "CSMB", "CSMS", "CSCS"]


class ChemspaceSearchInput(BaseModel):
    smiles: str = Field(description="SMILES string to search for")
    ship_to_country: str = Field(
        default="US",
        description="Two-letter country ISO code for shipping, e.g. US, DE, FR",
        min_length=2,
        max_length=2,
    )
    count: int = Field(default=10, ge=1, description="Maximum number of results on a page")
    page: int = Field(default=1, ge=1, description="Page number")
    categories: List[ProductCategory] = Field(
        default_factory=lambda: ["CSSB", "CSMB"],
        description="Product categories to search",
        min_length=1,
    )


class ChemspaceDeps(BaseModel):
    mgr: ChemspaceTokenManager

    class Config:
        arbitrary_types_allowed = True


async def _chemspace_search(
    ctx: RunContext[ChemspaceDeps],
    endpoint: str,
    inp: ChemspaceSearchInput,
) -> dict:
    async def do_request(access_token: str):
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(
                url=f"https://api.chem-space.com/v4/search/{endpoint}",
                headers={
                    "Accept": "application/json; version=4.1",
                    "Authorization": f"Bearer {access_token}",
                },
                params={
                    "shipToCountry": inp.ship_to_country,
                    "count": inp.count,
                    "page": inp.page,
                    "categories": ",".join(inp.categories),
                },
                files={"SMILES": (None, inp.smiles)},
            )

    access_token = await ctx.deps.mgr.get_token()
    response = await do_request(access_token)

    # If ChemSpace rejects the access token, force-refresh once and retry.
    if response.status_code == 401:
        await ctx.deps.mgr.refresh_token()
        access_token = await ctx.deps.mgr.get_token()
        response = await do_request(access_token)

    response.raise_for_status()
    return response.json()


async def search_exact(
    ctx: RunContext[ChemspaceDeps],
    smiles: str,
    ship_to_country: str = "US",
    count: int = 10,
    page: int = 1,
    categories: Optional[List[ProductCategory]] = None,
) -> dict:
    """Exact search by SMILES."""
    inp = ChemspaceSearchInput(
        smiles=smiles,
        ship_to_country=ship_to_country,
        count=count,
        page=page,
        categories=categories or ["CSSB", "CSMB"],
    )
    return await _chemspace_search(ctx, "exact", inp)


async def search_substructure(
    ctx: RunContext[ChemspaceDeps],
    smiles: str,
    ship_to_country: str = "US",
    count: int = 10,
    page: int = 1,
    categories: Optional[List[ProductCategory]] = None,
) -> dict:
    """Substructure search by SMILES."""
    inp = ChemspaceSearchInput(
        smiles=smiles,
        ship_to_country=ship_to_country,
        count=count,
        page=page,
        categories=categories or ["CSSB", "CSMB"],
    )
    return await _chemspace_search(ctx, "sub", inp)


async def search_similarity(
    ctx: RunContext[ChemspaceDeps],
    smiles: str,
    ship_to_country: str = "US",
    count: int = 10,
    page: int = 1,
    categories: Optional[List[ProductCategory]] = None,
) -> dict:
    """Similarity search by SMILES."""
    inp = ChemspaceSearchInput(
        smiles=smiles,
        ship_to_country=ship_to_country,
        count=count,
        page=page,
        categories=categories or ["CSSB", "CSMB"],
    )
    return await _chemspace_search(ctx, "sim", inp)