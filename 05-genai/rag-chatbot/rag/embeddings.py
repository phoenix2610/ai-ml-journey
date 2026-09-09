"""Embedding chunks, with a cache that makes re-indexing free.

Two things this module exists to get right.

**Task types.** Gemini embeds documents and queries into deliberately different
regions of the space, and you have to tell it which you are doing:
`RETRIEVAL_DOCUMENT` when indexing, `RETRIEVAL_QUERY` when searching. Using one
for both is a silent failure -- nothing errors, retrieval just gets worse.
`embed_chunks` and `embed_query` make it impossible to pick the wrong one.

**Caching.** Embedding is the expensive part of ingestion and the text rarely
changes. The cache is keyed on a hash of the exact text plus the model name, so
editing one document re-embeds one chunk, and switching models invalidates
everything automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.llm import EMBED_DIMENSIONS, GeminiClient


def cache_key(text: str, model: str, dimensions: int) -> str:
    digest = hashlib.sha256(f"{model}:{dimensions}:{text}".encode()).hexdigest()
    return digest[:32]


class EmbeddingCache:
    """A JSON file of key -> vector. Small, portable, and good enough."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    def _load(self) -> None:
        if not self.path or not self.path.exists():
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._data = {}          # a corrupt cache is just a cold cache

    def get(self, key: str) -> list[float] | None:
        value = self._data.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def put(self, key: str, vector: list[float]) -> None:
        self._data[key] = vector

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data), encoding="utf-8")
        tmp.replace(self.path)

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        """Always truthy.

        Without this, `__len__` supplies truthiness and an *empty* cache is
        falsy -- so `if cache:` silently skips storing anything, and a fresh
        cache can never warm up. A cache object exists or it does not; its
        contents are irrelevant to that question.
        """
        return True


def embed_chunks(
    chunks: list[Chunk],
    client: GeminiClient,
    *,
    cache: EmbeddingCache | None = None,
    dimensions: int = EMBED_DIMENSIONS,
    batch_size: int = 100,
) -> np.ndarray:
    """Embed chunks as documents. Returns an (n, dimensions) float32 array.

    Chunks are embedded with their heading path prepended (`with_context`), so
    a deeply nested passage still carries the words that identify its topic.
    """
    if not chunks:
        return np.zeros((0, dimensions), dtype=np.float32)

    texts = [chunk.with_context() for chunk in chunks]
    vectors: list[list[float] | None] = [None] * len(texts)
    pending: list[int] = []

    for i, text in enumerate(texts):
        hit = (
            cache.get(cache_key(text, client.embed_model, dimensions))
            if cache is not None
            else None
        )
        if hit is not None:
            vectors[i] = hit
        else:
            pending.append(i)

    if pending:
        fresh = client.embed(
            [texts[i] for i in pending],
            task_type="RETRIEVAL_DOCUMENT",
            dimensions=dimensions,
            batch_size=batch_size,
        )
        for i, vector in zip(pending, fresh):
            vectors[i] = vector
            if cache is not None:
                cache.put(cache_key(texts[i], client.embed_model, dimensions), vector)
        if cache is not None:
            cache.save()

    return np.asarray(vectors, dtype=np.float32)


def embed_query(
    question: str,
    client: GeminiClient,
    *,
    dimensions: int = EMBED_DIMENSIONS,
) -> np.ndarray:
    """Embed a question as a *query*, never as a document."""
    return np.asarray(client.embed_query(question, dimensions=dimensions), dtype=np.float32)


def normalise(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, so cosine similarity is a plain dot product.

    Normalising once at index time turns every later search into a single
    matrix multiply instead of n divisions.
    """
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


__all__ = ["embed_chunks", "embed_query", "EmbeddingCache", "cache_key", "normalise"]
