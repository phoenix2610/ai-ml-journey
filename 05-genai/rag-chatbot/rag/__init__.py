"""Retrieval-augmented generation with citations, over the Gemini API."""

from rag.chunking import Chunk, Document, chunk_document, chunk_documents
from rag.llm import GeminiClient, LLMError

__all__ = ["Document", "Chunk", "chunk_document", "chunk_documents", "GeminiClient", "LLMError"]
__version__ = "0.1.0"
