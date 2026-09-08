"""Splitting documents into retrievable pieces.

Chunking is the decision that most determines whether a RAG system works, and
it gets the least attention. Two failure modes bracket the choice:

* **Chunks too large** -- the embedding averages several topics, matches
  everything vaguely and nothing precisely, and the model gets a wall of
  mostly-irrelevant text.
* **Chunks too small** -- a sentence loses the context that made it meaningful.
  "It costs £45 per contact" is unretrievable once separated from what "it" is.

The approach here is **structure first, size second**. Markdown documents
already carry an author's own idea of where topics begin: headings. So split on
headings, and only fall back to size-based splitting inside a section that is
still too big. Each chunk carries its heading path, which is both useful
context for the model and a citation the user can verify.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
# Prose sentence boundary: '.', '!' or '?' followed by whitespace and a capital.
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

DEFAULT_CHUNK_TOKENS = 320
DEFAULT_OVERLAP_TOKENS = 60


def estimate_tokens(text: str) -> int:
    """Approximate token count without pulling in a tokeniser.

    Roughly 4 characters per token for English prose. This only has to be good
    enough to keep chunks inside a budget, and being slightly conservative is
    the safe direction to be wrong in.
    """
    return max(1, len(text) // 4)


@dataclass
class Document:
    doc_id: str
    text: str
    title: str = ""
    source: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    headings: tuple[str, ...] = ()
    index: int = 0
    title: str = ""
    source: str = ""

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    @property
    def heading_path(self) -> str:
        return " > ".join(self.headings)

    @property
    def citation(self) -> str:
        """What gets shown to a user so they can check the answer."""
        parts = [self.title or self.doc_id]
        if self.headings:
            parts.append(self.heading_path)
        return " — ".join(parts)

    def with_context(self) -> str:
        """Text as embedded and as given to the model.

        The heading path is prepended so a chunk reading "It costs £45 per
        contact" still embeds near "retention campaign economics". Without this,
        deep sections retrieve badly because the words that identify the topic
        appear only in an ancestor heading.
        """
        header = self.heading_path
        return f"{header}\n\n{self.text}" if header else self.text


@dataclass
class Section:
    headings: tuple[str, ...]
    text: str


def split_sections(text: str) -> list[Section]:
    """Split markdown on headings, tracking the full heading path."""
    matches = list(HEADING.finditer(text))

    if not matches:
        stripped = text.strip()
        return [Section((), stripped)] if stripped else []

    sections: list[Section] = []
    path: list[str] = []

    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(Section((), preamble))

    for i, match in enumerate(matches):
        level = len(match.group(1))
        heading = match.group(2).strip()

        # Truncate the path to this heading's depth, then push it.
        path = path[: level - 1]
        while len(path) < level - 1:
            path.append("")
        path.append(heading)

        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        if body:
            sections.append(Section(tuple(p for p in path if p), body))

    return sections


def split_by_size(
    text: str,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[str]:
    """Split text on sentence boundaries, packing up to a token budget.

    Overlap repeats the tail of the previous chunk so a fact spanning a
    boundary is retrievable from either side.
    """
    if estimate_tokens(text) <= max_tokens:
        return [text] if text.strip() else []

    sentences = [s.strip() for s in SENTENCE.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = estimate_tokens(sentence)

        # A single sentence over budget cannot be split further on sentence
        # boundaries; emit it whole rather than dropping it.
        if sentence_tokens > max_tokens and not current:
            chunks.append(sentence)
            continue

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current))
            current, current_tokens = _overlap_tail(current, overlap_tokens)

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: list[str], overlap_tokens: int) -> tuple[list[str], int]:
    """Take whole sentences from the end until the overlap budget is spent."""
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        tokens = estimate_tokens(sentence)
        if total + tokens > overlap_tokens:
            break
        tail.insert(0, sentence)
        total += tokens
    return tail, total


def chunk_document(
    document: Document,
    *,
    max_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    min_tokens: int = 12,
) -> list[Chunk]:
    """Structure first, size second."""
    chunks: list[Chunk] = []

    for section in split_sections(document.text):
        for piece in split_by_size(
            section.text, max_tokens=max_tokens, overlap_tokens=overlap_tokens
        ):
            # Drop fragments too small to carry meaning -- a stray "See below."
            # embeds near everything and helps nothing.
            if estimate_tokens(piece) < min_tokens:
                continue
            index = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}#{index}",
                    doc_id=document.doc_id,
                    text=piece,
                    headings=section.headings,
                    index=index,
                    title=document.title or document.doc_id,
                    source=document.source,
                )
            )

    return chunks


def chunk_documents(documents: list[Document], **kwargs) -> list[Chunk]:
    return [chunk for document in documents for chunk in chunk_document(document, **kwargs)]


def load_markdown(path, doc_id: str | None = None) -> Document:
    """Read a markdown file, taking the title from its first H1 if present."""
    from pathlib import Path

    path = Path(path)
    text = path.read_text(encoding="utf-8")

    title = ""
    first = HEADING.search(text)
    if first and len(first.group(1)) == 1:
        title = first.group(2).strip()

    return Document(
        doc_id=doc_id or path.stem,
        text=text,
        title=title or path.stem,
        source=str(path),
    )


__all__ = [
    "Document", "Chunk", "Section",
    "chunk_document", "chunk_documents", "split_sections", "split_by_size",
    "estimate_tokens", "load_markdown",
    "DEFAULT_CHUNK_TOKENS", "DEFAULT_OVERLAP_TOKENS",
]
