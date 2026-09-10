"""Answering from retrieved context, with citations that can be checked.

The whole value of RAG is that an answer is *checkable*. That requires three
things, and skipping any one of them gives you a chatbot that sounds grounded
without being grounded:

1. **The model must be told to refuse.** If the context does not contain the
   answer, "I don't know" is the correct output. Without an explicit
   instruction, a model will helpfully fill the gap from its parameters, and
   that is precisely the failure RAG exists to prevent.
2. **Citations must be structural, not decorative.** The model emits `[1]`,
   `[2]` markers which are parsed and resolved back to real chunks. A citation
   pointing at source 7 when only 5 were supplied is a detectable error, and
   `verify_citations` detects it.
3. **The context must be labelled.** Sources are numbered in the prompt so the
   markers have something unambiguous to refer to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag.llm import GeminiClient, LLMError
from rag.store import Hit

CITATION = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You answer questions using only the numbered sources provided.

Rules, in order of importance:
1. Use ONLY information present in the sources. Never use prior knowledge.
2. If the sources do not contain the answer, reply exactly: I don't know based \
on the provided sources.
3. Cite every factual claim with the source number in square brackets, like [1] \
or [2][3]. Cite the specific source the claim came from.
4. Quote exact figures rather than rounding or rephrasing them.
5. Be concise. Two or three sentences unless the question needs more.

Never invent a source number that was not provided."""

REFUSAL = "I don't know based on the provided sources."


def format_context(hits: list[Hit]) -> str:
    """Number the sources so citation markers have a stable referent."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.chunk.citation}\n{hit.chunk.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    if not hits:
        return (
            "No sources were retrieved.\n\n"
            f"Question: {question}\n\n"
            "Reply exactly: I don't know based on the provided sources."
        )
    return f"Sources:\n\n{format_context(hits)}\n\nQuestion: {question}\n\nAnswer:"


@dataclass
class Citation:
    number: int
    hit: Hit | None

    @property
    def valid(self) -> bool:
        return self.hit is not None

    @property
    def label(self) -> str:
        return self.hit.chunk.citation if self.hit else f"unknown source [{self.number}]"


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    usage_tokens: int = 0
    # Filled in by the pipeline via verify_citations(); empty means well-formed.
    problems: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return REFUSAL.lower().rstrip(".") in self.text.lower()

    @property
    def cited_numbers(self) -> set[int]:
        return {c.number for c in self.citations}

    @property
    def has_invalid_citation(self) -> bool:
        return any(not c.valid for c in self.citations)

    @property
    def sources_used(self) -> list[Hit]:
        """Only the sources the answer actually cited, in citation order."""
        seen, out = set(), []
        for citation in self.citations:
            if citation.valid and citation.number not in seen:
                seen.add(citation.number)
                out.append(citation.hit)
        return out

    def formatted(self) -> str:
        """Answer plus a resolved source list, for a terminal or the UI."""
        if not self.sources_used:
            return self.text

        lines = [self.text, "", "Sources:"]
        seen: set[int] = set()
        for citation in self.citations:
            if citation.valid and citation.number not in seen:
                seen.add(citation.number)
                lines.append(f"  [{citation.number}] {citation.label}")
        return "\n".join(lines)


def parse_citations(text: str, hits: list[Hit]) -> list[Citation]:
    """Resolve `[n]` markers back to the sources that were supplied.

    A marker outside the supplied range resolves to `None`, which is how a
    hallucinated citation becomes a detectable fact rather than a plausible
    footnote.
    """
    citations: list[Citation] = []
    for match in CITATION.finditer(text):
        number = int(match.group(1))
        hit = hits[number - 1] if 1 <= number <= len(hits) else None
        citations.append(Citation(number=number, hit=hit))
    return citations


def verify_citations(answer: "Answer") -> list[str]:
    """Problems worth surfacing. Empty list means the answer looks well-formed."""
    problems: list[str] = []

    if answer.refused:
        return problems

    invalid = sorted({c.number for c in answer.citations if not c.valid})
    if invalid:
        problems.append(
            f"cites source(s) {invalid} that were never supplied "
            f"({len(answer.hits)} were)"
        )

    if answer.hits and not answer.citations:
        problems.append("makes claims but cites nothing")

    return problems


def answer_question(
    question: str,
    hits: list[Hit],
    client: GeminiClient,
    *,
    temperature: float = 0.1,
    max_output_tokens: int = 2048,
) -> Answer:
    """Generate a grounded answer.

    Temperature is low by default: this is an extraction task, not a creative
    one, and sampling variance here shows up as invented detail.
    """
    if not hits:
        return Answer(question=question, text=REFUSAL, hits=[], citations=[])

    before = client.usage.total_tokens
    try:
        text = client.generate(
            build_prompt(question, hits),
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except LLMError as exc:
        raise LLMError(f"could not generate an answer: {exc}") from None

    return Answer(
        question=question,
        text=text,
        hits=hits,
        citations=parse_citations(text, hits),
        usage_tokens=client.usage.total_tokens - before,
    )


__all__ = [
    "answer_question", "Answer", "Citation",
    "build_prompt", "format_context", "parse_citations", "verify_citations",
    "SYSTEM_PROMPT", "REFUSAL", "CITATION",
]
