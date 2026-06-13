"""Pydantic request/response schemas for the NeuroDB HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, examples=["documents"])
    dimension: int = Field(..., gt=0, le=65536, examples=[256])
    metric: str = Field("cosine", examples=["cosine", "dot", "euclidean"])


class VectorItem(BaseModel):
    id: str | None = Field(None, description="Stable id; generated when omitted.")
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    items: list[VectorItem]


class SearchRequest(BaseModel):
    vector: list[float]
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False


class TextItem(BaseModel):
    id: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextUpsertRequest(BaseModel):
    items: list[TextItem]


class TextSearchRequest(BaseModel):
    text: str
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False


class EmbedRequest(BaseModel):
    text: str
