import json
import os
import pathlib
import tempfile
import time
from typing import Literal

import httpx
from httpx._urls import URL
from pydantic import BaseModel, ConfigDict, Field


class ChemspaceRequest(BaseModel):
    smiles: str = Field(description="SMILES string to search for")
    shipToCountry: str = Field(
        default="US",
        description="Two-letter country ISO code for shipping, e.g. US, DE, FR",
        min_length=2,
        max_length=2,
    )
    count: int = Field(
        default=10, ge=1, description="Maximum number of results on a page"
    )
    page: int = Field(default=1, ge=1, description="Page number")
    categories: list[Literal["CSSB", "CSSS", "CSMB", "CSMS", "CSCS"]] = Field(
        default=["CSSB", "CSMB"],
        description=(
            "Product categories to search. "
            "CSSB: in-stock building blocks, CSSS: in-stock screening compounds, "
            "CSMB: make-on-demand building blocks, CSMS: make-on-demand screening compounds, "
            "CSCS: custom synthesis request."
        ),
        min_length=1,
    )


class ChemspacePrice(BaseModel):
    pack: float
    uom: str
    priceUsd: float | None
    priceEur: float | None


class ChemspaceOffer(BaseModel):
    vendorName: str
    vendorCode: str
    leadTimeDays: int | None
    purity: int
    prices: list[ChemspacePrice]


class ChemspaceProperties(BaseModel):
    mw: float
    hac: int
    logp: float
    rotb: int
    hba: int
    hbd: int
    ringCount: int
    fsp3: float
    tpsa: float


class ChemspaceItem(BaseModel):
    csId: str
    link: str
    smiles: str | None
    matchType: str
    query: str
    queryIndex: int
    molFormula: str | None
    cas: str | None
    mfcd: str | None
    properties: ChemspaceProperties
    offerCount: int
    offers: list[ChemspaceOffer]


class ChemspaceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    count: int
    searchType: str
    filtered: int
    all_: int = Field(alias="all")
    items: list[ChemspaceItem]


class ChemspaceAPI:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.chem-space.com/",
        version: str = "4.1",
    ):
        self.api_key = api_key or os.getenv("CHEMSPACE_API_KEY")
        self.base_url: URL = httpx.URL(base_url)
        self.auth_url: URL = self.base_url.join("/auth/token")
        self.version = "4.1"
        self.major_version: str = version.split(".")[0]
        self.get_search_url = lambda endpoint: self.base_url.join(
            f"/v{self.major_version}/search/{endpoint}"
        )

        if not self.api_key:
            raise ValueError(
                "CHEMSPACE_API_KEY is missing. Set it in your environment or pass api_key explicitly."
            )

        self.access_token: str = ""
        self.expires_at: float = 0

        self.token_cache = (
            pathlib.Path(tempfile.gettempdir()) / ".chemspace_token_cache"
        ).resolve()
        if self.token_cache.exists():
            with open(self.token_cache) as fp:
                data = json.load(fp)
            self.access_token = data["access_token"]
            self.expires_at = float(data["expires_at"])

    async def refresh_token(self):
        async with httpx.AsyncClient() as client:
            r = await client.get(
                self.auth_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": f"application/json; version={self.version}",
                },
                timeout=10,
            )

        r.raise_for_status()
        data = r.json()

        self.access_token = data["access_token"]
        self.expires_at = time.time() + data["expires_in"] - 30

        with open(self.token_cache, "w") as fp:
            json.dump(
                {"access_token": self.access_token, "expires_at": self.expires_at},
                fp,
            )

    async def get_token(self) -> str:
        if time.time() >= self.expires_at:
            await self.refresh_token()
        return self.access_token

    async def post(
        self, endpoint: str, token: str, req: ChemspaceRequest
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(
                url=self.get_search_url(endpoint),
                headers={
                    "Accept": "application/json; version=4.1",
                    "Authorization": f"Bearer {token}",
                },
                params={
                    "shipToCountry": req.shipToCountry,
                    "count": req.count,
                    "page": req.page,
                    "categories": ",".join(req.categories),
                },
                files={"SMILES": (None, req.smiles)},
            )

    async def search(self, endpoint: str, req: ChemspaceRequest) -> ChemspaceResponse:
        token = await self.get_token()
        response = await self.post(endpoint, token, req)
        if response.status_code == 401:
            await self.refresh_token()
            return await self.search(endpoint, req)
        response.raise_for_status()
        return ChemspaceResponse.model_validate(response.json())
