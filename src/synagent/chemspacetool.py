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


async def _chemspace_search_direct(
    deps: ChemspaceDeps,
    endpoint: str,
    inp: ChemspaceSearchInput,
) -> dict:
    """Same request _chemspace_search makes, but takes ChemspaceDeps directly instead of
    a RunContext — for calling from orchestration code (e.g. master.py) outside of an
    agent tool call, where there's no RunContext to construct."""
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

    access_token = await deps.mgr.get_token()
    response = await do_request(access_token)

    if response.status_code == 401:
        await deps.mgr.refresh_token()
        access_token = await deps.mgr.get_token()
        response = await do_request(access_token)

    response.raise_for_status()
    return response.json()


async def check_building_block_available(deps: ChemspaceDeps, smiles: str) -> dict:
    """Checks whether a SMILES is a real, purchasable building block — an exact-match
    ChemSpace search. Returns {"smiles": str, "available": bool, "vendor_count": int,
    "error": str | None}. Never raises: a network/API failure is reported as
    available=False with the error message, since "couldn't confirm" must never be
    silently treated as "confirmed available"."""
    inp = ChemspaceSearchInput(smiles=smiles, count=1)
    try:
        result = await _chemspace_search_direct(deps, "exact", inp)
        items = result.get("items", []) if isinstance(result, dict) else []
        return {
            "smiles": smiles,
            "available": len(items) > 0,
            "vendor_count": len(items),
            "error": None,
        }
    except Exception as e:
        return {"smiles": smiles, "available": False, "vendor_count": 0, "error": str(e)}


def _cheapest_offer(item: dict) -> dict | None:
    """Finds the lowest real USD price across an item's vendor offers/pack sizes. Skips
    offers with no listed price (priceUsd null) rather than treating them as free."""
    best = None
    for offer in item.get("offers", []):
        for price in offer.get("prices", []):
            usd = price.get("priceUsd")
            if usd is None:
                continue
            if best is None or usd < best["price_usd"]:
                best = {
                    "price_usd": usd,
                    "pack_size": f"{price.get('pack')}{price.get('uom', '')}",
                    "vendor": offer.get("vendorName"),
                    "lead_time_days": offer.get("leadTimeDays"),
                    "purity": offer.get("purity"),
                }
    return best


async def check_building_block_economics(deps: ChemspaceDeps, smiles: str) -> dict:
    """Like check_building_block_available, but also surfaces real pricing/lead-time
    data when present — a structurally "terminal" molecule in a search isn't useful if
    nobody actually lists a price for it, or if the only listed price is for a
    multi-week-lead-time specialty reagent. Never raises and never invents a price: a
    vendor offer with no listed price is skipped, not treated as free, and a failed
    lookup is reported as unconfirmed rather than unavailable-with-a-fabricated-price.

    Returns: {
        "smiles": str,
        "available": bool,
        "vendor_count": int,
        "cheapest": {"price_usd": float, "pack_size": str, "vendor": str,
                     "lead_time_days": int, "purity": float} | None,
        "error": str | None,
    }
    """
    inp = ChemspaceSearchInput(smiles=smiles, count=1)
    try:
        result = await _chemspace_search_direct(deps, "exact", inp)
        items = result.get("items", []) if isinstance(result, dict) else []
        cheapest = _cheapest_offer(items[0]) if items else None
        return {
            "smiles": smiles,
            "available": len(items) > 0,
            "vendor_count": len(items),
            "cheapest": cheapest,
            "error": None,
        }
    except Exception as e:
        return {
            "smiles": smiles, "available": False, "vendor_count": 0,
            "cheapest": None, "error": str(e),
        }


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