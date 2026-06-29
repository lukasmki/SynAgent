from typing import Any

from pydantic import BaseModel, Field


class StorageRecord(BaseModel):
    id: str = Field(description="Unique identifier for the record.")
    data: dict[str, Any] = Field(description="Arbitrary payload to persist.")


class StorageQuery(BaseModel):
    id: str | None = Field(default=None, description="Exact record ID to look up.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional key/value filters applied to record data.",
    )
