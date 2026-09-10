"""End to end: documents in, cited answers out.

One object owning ingestion, retrieval and generation, so the Streamlit app,
the CLI and the evaluation harness all exercise the same path. When the eval
harness says retrieval recall is 0.85, that is the number the app gets too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from rag.chunking import Chunk, Document, chunk_documents, load_markdown
from rag.embeddings import EmbeddingCache, embed_chunks, embed_query
from rag.generate import Answer, answer_question, verify_citations
from rag.llm import EMBED_DIMENSIONS, GeminiClient
from rag.store import Hit, VectorStore

DEFAULT_INDEX = Path("index")
DEFAULT_CACHE = Path(".cache/embeddings.json")


@dataclass
class RagConfig:
    top_k: int = 5
    max_tokens_per_chunk: int = 320
    overlap_tokens: int = 60
    dimensions: int = EMBED_DIMENSIONS
    retrieval: str = "hybrid"          # 'hybrid' | 'dense' | 'keyword'
    temperature: float = 0.1


@dataclass
class RagPipeline:
    client: GeminiClient
    config: RagConfig = field(default_factory=RagConfig)
    store: VectorStore | None = None
    cache: EmbeddingCache | None = None

    # ------------------------------------------------------------- ingestion

    def ingest(self, documents: list[Document]) -> VectorStore:
        chunks = chunk_documents(
            documents,
            max_tokens=self.config.max_tokens_per_chunk,
            overlap_tokens=self.config.overlap_tokens,
        )
        if not chunks:
            raise ValueError("no chunks produced -- are the documents empty?")

        vectors = embed_chunks(
            chunks, self.client, cache=self.cache, dimensions=self.config.dimensions
        )
        self.store = VectorStore(chunks, vectors)
        return self.store

    def ingest_directory(self, directory: Path | str, pattern: str = "**/*.md") -> VectorStore:
        directory = Path(directory)
        paths = sorted(p for p in directory.glob(pattern) if p.is_file())
        if not paths:
            raise FileNotFoundError(f"no files matching {pattern!r} under {directory}")
        return self.ingest([load_markdown(p) for p in paths])

    # ------------------------------------------------------------- retrieval

    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        if self.store is None:
            raise RuntimeError("nothing indexed yet -- call ingest() or load() first")

        k = k or self.config.top_k
        mode = self.config.retrieval

        if mode == "keyword":
            return self.store.keyword_search(question, k=k)

        vector = embed_query(question, self.client, dimensions=self.config.dimensions)
        if mode == "dense":
            return self.store.search(vector, k=k)
        return self.store.hybrid_search(question, vector, k=k)

    # ------------------------------------------------------------ generation

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k=k)
        answer = answer_question(
            question, hits, self.client, temperature=self.config.temperature
        )
        answer.problems = verify_citations(answer)
        return answer

    # ----------------------------------------------------------- persistence

    def save(self, directory: Path | str = DEFAULT_INDEX) -> Path:
        if self.store is None:
            raise RuntimeError("nothing to save -- call ingest() first")
        return self.store.save(directory)

    def load(self, directory: Path | str = DEFAULT_INDEX) -> VectorStore:
        self.store = VectorStore.load(directory)
        return self.store

    @property
    def chunk_count(self) -> int:
        return len(self.store) if self.store else 0


def build_pipeline(
    directory: Path | str | None = None,
    *,
    client: GeminiClient | None = None,
    config: RagConfig | None = None,
    cache_path: Path | str | None = DEFAULT_CACHE,
    index_path: Path | str = DEFAULT_INDEX,
) -> RagPipeline:
    """Load an existing index, or build one from a directory of markdown."""
    pipeline = RagPipeline(
        client=client or GeminiClient(),
        config=config or RagConfig(),
        cache=EmbeddingCache(cache_path),
    )

    index_path = Path(index_path)
    if (index_path / "chunks.json").exists():
        pipeline.load(index_path)
    elif directory is not None:
        pipeline.ingest_directory(directory)
        pipeline.save(index_path)
    else:
        raise FileNotFoundError(
            f"no index at {index_path} and no source directory given"
        )

    return pipeline


__all__ = [
    "RagPipeline", "RagConfig", "build_pipeline",
    "Document", "Chunk", "Answer", "Hit",
    "DEFAULT_INDEX", "DEFAULT_CACHE",
]
