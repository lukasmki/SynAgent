from typing import Literal

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
