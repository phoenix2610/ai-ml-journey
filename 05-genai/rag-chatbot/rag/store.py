"""The index: dense vectors, keyword search, and a hybrid of the two.

No vector database. At this scale a normalised NumPy matrix and one matrix
multiply is faster than the network round-trip to a service, and it keeps the
whole retrieval path readable.

**Why hybrid.** Dense embeddings capture meaning but are bad at rare literal
tokens -- error codes, product names, `gemini-embedding-2`. Keyword search is
the opposite. Neither alone is reliable, so results are fused with Reciprocal
Rank Fusion, which combines *ranks* rather than scores and therefore needs no
calibration between two incomparable scoring scales.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.embeddings import normalise

WORD = re.compile(r"[a-z0-9_]+")

# Common enough to appear everywhere, so they carry no retrieval signal.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have how in is it its of on or that "
    "the to was were what when where which who why with you your this these".split()
)

RRF_K = 60          # standard damping constant from the RRF paper


def tokenise(text: str) -> list[str]:
    return [w for w in WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0
    dense_rank: int | None = None
    keyword_rank: int | None = None

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def citation(self) -> str:
        return self.chunk.citation


class VectorStore:
    """Chunks plus their embeddings, searchable three ways."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalise(np.asarray(vectors, dtype=np.float32))
        self._build_keyword_index()

    def __len__(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------- keyword

    def _build_keyword_index(self) -> None:
        """A compact BM25-style index over the chunk texts."""
        self._tokens = [tokenise(c.with_context()) for c in self.chunks]
        self._lengths = np.array([len(t) or 1 for t in self._tokens], dtype=np.float32)
        self._avg_length = float(self._lengths.mean()) if len(self._lengths) else 1.0

        self._term_frequency: list[Counter] = [Counter(t) for t in self._tokens]
        document_frequency: Counter = Counter()
        for tokens in self._tokens:
            document_frequency.update(set(tokens))

        n = max(1, len(self.chunks))
        self._idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def keyword_search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenise(query)
        if not terms:
            return []

        k1, b = 1.5, 0.75
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        for term in terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i, frequency in enumerate(self._term_frequency):
                f = frequency.get(term, 0)
                if not f:
                    continue
                denominator = f + k1 * (1 - b + b * self._lengths[i] / self._avg_length)
                scores[i] += idf * (f * (k1 + 1)) / denominator

        return self._top(scores, k, min_score=1e-9)

    # --------------------------------------------------------------- dense

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[Hit]:
        """Cosine similarity. Vectors are pre-normalised, so this is a dot product."""
        if not len(self.chunks):
            return []
        scores = self.vectors @ normalise(np.asarray(query_vector, dtype=np.float32))
        return self._top(scores, k)

    def _top(self, scores: np.ndarray, k: int, min_score: float | None = None) -> list[Hit]:
        """Top-k by score.

        No positive-score filter by default: cosine similarity is legitimately
        negative for a poor-but-best match, and silently dropping those makes
        `k` mean "at most k", which callers do not expect. Filtering is opt-in
        via `min_score`.
        """
        k = min(k, len(self.chunks))
        if k <= 0:
            return []
        order = np.argpartition(-scores, k - 1)[:k]
        order = order[np.argsort(-scores[order])]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
            if min_score is None or scores[i] >= min_score
        ]

    # -------------------------------------------------------------- hybrid

    def hybrid_search(
        self, query: str, query_vector: np.ndarray, k: int = 5, *, pool: int = 20
    ) -> list[Hit]:
        """Fuse dense and keyword rankings with Reciprocal Rank Fusion.

        RRF combines ranks, not scores: cosine similarity and BM25 live on
        incomparable scales, and any attempt to weight them directly needs
        tuning that does not transfer between corpora.

            score = sum over rankers of 1 / (K + rank)
        """
        dense = self.search(query_vector, k=pool)
        keyword = self.keyword_search(query, k=pool)

        fused: dict[str, dict] = {}
        for hits, label in ((dense, "dense_rank"), (keyword, "keyword_rank")):
            for hit in hits:
                entry = fused.setdefault(
                    hit.chunk.chunk_id,
                    {"chunk": hit.chunk, "score": 0.0, "dense_rank": None, "keyword_rank": None},
                )
                entry["score"] += 1.0 / (RRF_K + hit.rank)
                entry[label] = hit.rank

        ranked = sorted(fused.values(), key=lambda e: -e["score"])[:k]
        return [
            Hit(
                chunk=entry["chunk"],
                score=entry["score"],
                rank=rank,
                dense_rank=entry["dense_rank"],
                keyword_rank=entry["keyword_rank"],
            )
            for rank, entry in enumerate(ranked, start=1)
        ]

    # ----------------------------------------------------------- persistence

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.json").write_text(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                        "headings": list(c.headings), "index": c.index,
                        "title": c.title, "source": c.source,
                    }
                    for c in self.chunks
                ]
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "VectorStore":
        directory = Path(directory)
        vectors_path = directory / "vectors.npy"
        chunks_path = directory / "chunks.json"

        if not vectors_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"no index at {directory} -- run ingestion first")

        raw = json.loads(chunks_path.read_text(encoding="utf-8"))
        chunks = [
            Chunk(
                chunk_id=r["chunk_id"], doc_id=r["doc_id"], text=r["text"],
                headings=tuple(r["headings"]), index=r["index"],
                title=r.get("title", ""), source=r.get("source", ""),
            )
            for r in raw
        ]
        return cls(chunks, np.load(vectors_path))


__all__ = ["VectorStore", "Hit", "tokenise", "STOPWORDS", "RRF_K"]
