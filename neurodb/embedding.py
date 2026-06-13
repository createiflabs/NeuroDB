"""A tiny, dependency-free text embedder.

This is a deterministic *hashing* embedder (a "hashing trick" bag of words plus
character n-grams). It is **not** a neural model and will never match a real
transformer embedding, but it requires no downloads, runs anywhere and is good
enough to make semantic-ish search demos work out of the box.

For production workloads, generate embeddings with your model of choice (OpenAI,
Cohere, sentence-transformers, ...) and push the raw vectors to the ``/vectors``
endpoints instead of the ``/texts`` convenience endpoints.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Whole-word matches are the strongest signal; character n-grams add fuzzy
# overlap (e.g. "dog" ~ "dogs") but should not dominate exact-word matches.
_WORD_WEIGHT = 3.0
_NGRAM_WEIGHT = 1.0


def _features(text: str, ngram: int = 3) -> list[tuple[str, float]]:
    """Yield ``(token, weight)`` features: whole words plus padded char n-grams."""

    text = text.lower()
    words = _TOKEN_RE.findall(text)
    features: list[tuple[str, float]] = [(word, _WORD_WEIGHT) for word in words]
    for word in words:
        padded = f"#{word}#"
        if len(padded) >= ngram:
            for i in range(len(padded) - ngram + 1):
                features.append((padded[i : i + ngram], _NGRAM_WEIGHT))
    return features


def _hash(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def embed_text(text: str, dim: int = 256) -> np.ndarray:
    """Embed a single string into an L2-normalised float32 vector of length ``dim``."""

    vec = np.zeros(dim, dtype=np.float32)
    for token, weight in _features(text):
        h = _hash(token)
        idx = h % dim
        sign = 1.0 if (h >> 1) & 1 else -1.0
        vec[idx] += sign * weight
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec


def embed_texts(texts: list[str], dim: int = 256) -> np.ndarray:
    """Embed a batch of strings into a ``(len(texts), dim)`` float32 matrix."""

    if not texts:
        return np.zeros((0, dim), dtype=np.float32)
    return np.vstack([embed_text(t, dim) for t in texts]).astype(np.float32)
