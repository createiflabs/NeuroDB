"""Pydantic request schemas for the NeuroDB HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateMemoryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128, examples=["records"])
    dimension: int = Field(..., gt=0, le=65536, examples=[256])
    beta: float = Field(8.0, gt=0, description="Inverse temperature for Hopfield recall.")
    fields: list[str] | None = Field(
        None, description="Optional per-dimension field names (len must equal dimension)."
    )


class PatternItem(BaseModel):
    id: str | None = Field(None, description="Stable id; generated when omitted.")
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class WriteRequest(BaseModel):
    items: list[PatternItem]


class CompleteRequest(BaseModel):
    query: list[float]
    beta: float | None = Field(None, gt=0)
    mask: list[int] | None = Field(
        None, description="Indices of *known* dimensions; the rest are completed."
    )
    steps: int = Field(1, ge=1, le=64)
    top_k: int = Field(5, ge=0, le=1000)


class SearchRequest(BaseModel):
    query: list[float]
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False


class AnomalyRequest(BaseModel):
    query: list[float]
    beta: float | None = Field(None, gt=0)
    top_k: int = Field(5, ge=0, le=1000)


class TextItem(BaseModel):
    id: str | None = None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextWriteRequest(BaseModel):
    items: list[TextItem]


class TextSearchRequest(BaseModel):
    text: str
    k: int = Field(10, gt=0, le=1000)
    filter: dict[str, Any] | None = None
    include_vectors: bool = False


class EmbedRequest(BaseModel):
    text: str
