"""Pydantic request schemas for the NeuroDB HTTP API.

Bounds here are the first line of defence for an internet-facing deployment:
they cap memory/CPU blow-ups (huge vectors, giant batches, unbounded text or
metadata) before any request reaches the store.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Request bounds (see module docstring).
MAX_DIMENSION = 65536
BULK_MAX = 1000  # items per write batch
TEXT_MAX = 32_768  # characters per text field
META_MAX_BYTES = 16_384  # serialized metadata per item
ID_MAX = 512  # characters per id


def _check_metadata_size(value: dict[str, Any]) -> dict[str, Any]:
    if len(json.dumps(value, default=str)) > META_MAX_BYTES:
        raise ValueError(f"metadata exceeds {META_MAX_BYTES} bytes when serialized")
    return value


class CreateMemoryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, examples=["records"])
    dimension: int = Field(..., gt=0, le=MAX_DIMENSION, examples=[256])
    beta: float = Field(8.0, gt=0, description="Inverse temperature for Hopfield recall.")
    fields: list[str] | None = Field(
        None,
        max_length=MAX_DIMENSION,
        description="Optional per-dimension field names (len must equal dimension).",
    )


class PatternItem(BaseModel):
    id: str | None = Field(
        None, max_length=ID_MAX, description="Stable id; generated when omitted."
    )
    vector: list[float] = Field(..., min_length=1, max_length=MAX_DIMENSION)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("vector")
    @classmethod
    def _finite_vector(cls, v: list[float]) -> list[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("vector must contain only finite numbers")
        return v

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_metadata_size(v)


class WriteRequest(BaseModel):
    items: list[PatternItem] = Field(..., min_length=1, max_length=BULK_MAX)


class CompleteRequest(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=MAX_DIMENSION)
    beta: float | None = Field(None, gt=0)
    mask: list[int] | None = Field(
        None,
        min_length=1,
        max_length=MAX_DIMENSION,
        description="Indices of *known* dimensions; the rest are completed.",
    )
    steps: int = Field(1, ge=1, le=64)
    top_k: int = Field(5, ge=0, le=1000)


class SearchRequest(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=MAX_DIMENSION)
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False
    metric: Literal["cosine", "dot", "euclidean"] = "cosine"


class AnomalyRequest(BaseModel):
    query: list[float] = Field(..., min_length=1, max_length=MAX_DIMENSION)
    beta: float | None = Field(None, gt=0)
    top_k: int = Field(5, ge=0, le=1000)


class TextItem(BaseModel):
    id: str | None = Field(None, max_length=ID_MAX)
    text: str = Field(..., min_length=1, max_length=TEXT_MAX)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _check_metadata_size(v)


class TextWriteRequest(BaseModel):
    items: list[TextItem] = Field(..., min_length=1, max_length=BULK_MAX)


class TextSearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=TEXT_MAX)
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False


class EmbedRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=TEXT_MAX)
