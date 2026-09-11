"""Measuring whether the RAG system actually works.

A demo that answers three questions convincingly tells you nothing. These are
the numbers that do, and they separate two failure modes that look identical
from the outside:

* **Retrieval failed** -- the answer is wrong because the right chunk never
  made it into the context. Measured by `recall@k` and `MRR` against
  hand-labelled expected sources.
* **Generation failed** -- the right chunk *was* in the context and the model
  still got it wrong. Measured by whether required facts appear in the answer
  and whether citations resolve.

Conflating them wastes time: you cannot fix a retrieval problem by editing the
prompt, and recall@k tells you which one you have.

Also measured: **refusal behaviour**. A system that never says "I don't know"
is not grounded, it is confident. `eval/questions.yaml` includes questions the
corpus deliberately cannot answer, and the system is expected to decline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag.generate import verify_citations
from rag.pipeline import RagPipeline

DEFAULT_QUESTIONS = Path(__file__).resolve().parent.parent / "eval" / "questions.yaml"


@dataclass
class EvalCase:
    question: str
    expect_docs: list[str] = field(default_factory=list)
    expect_facts: list[str] = field(default_factory=list)
    should_refuse: bool = False
    note: str = ""


@dataclass
class CaseResult:
    case: EvalCase
    retrieved_docs: list[str]
    answer_text: str
    hit_rank: int | None
    facts_found: list[str]
    facts_missing: list[str]
    refused: bool
    citation_problems: list[str]

    @property
    def retrieval_ok(self) -> bool:
        """Did any expected document make it into the context at all?"""
        return not self.case.expect_docs or self.hit_rank is not None

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.hit_rank if self.hit_rank else 0.0

    @property
    def facts_ok(self) -> bool:
        return not self.facts_missing

    @property
    def refusal_ok(self) -> bool:
        return self.refused == self.case.should_refuse

    @property
    def passed(self) -> bool:
        return self.retrieval_ok and self.facts_ok and self.refusal_ok and not self.citation_problems

    @property
    def diagnosis(self) -> str:
        """Which stage broke -- the whole point of separating the metrics."""
        if self.passed:
            return "ok"
        if not self.refusal_ok:
            return "should have refused" if self.case.should_refuse else "refused wrongly"
        if not self.retrieval_ok:
            return "RETRIEVAL failed -- right source never reached the context"
        if not self.facts_ok:
            return f"GENERATION failed -- context had it, answer missed {self.facts_missing}"
        return f"citations: {'; '.join(self.citation_problems)}"


@dataclass
class EvalReport:
    results: list[CaseResult]

    @property
    def n(self) -> int:
        return len(self.results)

    def _rate(self, predicate) -> float:
        return sum(1 for r in self.results if predicate(r)) / self.n if self.n else 0.0

    @property
    def recall_at_k(self) -> float:
        graded = [r for r in self.results if r.case.expect_docs]
        return sum(1 for r in graded if r.retrieval_ok) / len(graded) if graded else 0.0

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank -- rewards putting the right source first."""
        graded = [r for r in self.results if r.case.expect_docs]
        return sum(r.reciprocal_rank for r in graded) / len(graded) if graded else 0.0

    @property
    def faithfulness(self) -> float:
        graded = [r for r in self.results if r.case.expect_facts]
        return sum(1 for r in graded if r.facts_ok) / len(graded) if graded else 0.0

    @property
    def refusal_accuracy(self) -> float:
        return self._rate(lambda r: r.refusal_ok)

    @property
    def citation_validity(self) -> float:
        return self._rate(lambda r: not r.citation_problems)

    @property
    def pass_rate(self) -> float:
        return self._rate(lambda r: r.passed)

    def __str__(self) -> str:
        return (
            f"\n  {self.n} cases\n\n"
            f"    recall@k            {self.recall_at_k:>6.1%}   right source reached the context\n"
            f"    MRR                 {self.mrr:>6.3f}   how highly it ranked\n"
            f"    faithfulness        {self.faithfulness:>6.1%}   required facts present\n"
            f"    refusal accuracy    {self.refusal_accuracy:>6.1%}   declined when it should\n"
            f"    citation validity   {self.citation_validity:>6.1%}   no invented sources\n"
            f"    overall pass        {self.pass_rate:>6.1%}\n"
        )

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results if not r.passed]


def load_cases(path: Path | str = DEFAULT_QUESTIONS) -> list[EvalCase]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no eval set at {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [
        EvalCase(
            question=entry["question"],
            expect_docs=entry.get("expect_docs", []) or [],
            expect_facts=[str(f) for f in (entry.get("expect_facts", []) or [])],
            should_refuse=bool(entry.get("should_refuse", False)),
            note=entry.get("note", ""),
        )
        for entry in raw.get("cases", [])
    ]


def _normalise(text: str) -> str:
    """Loose matching so '38.8 percent' matches '38.8%'."""
    return (
        text.lower()
        .replace(",", "")
        .replace("percent", "%")
        .replace(" %", "%")
        .replace("£", "")
        .replace("$", "")
    )


def run_case(pipeline: RagPipeline, case: EvalCase, k: int | None = None) -> CaseResult:
    answer = pipeline.ask(case.question, k=k)
    retrieved = [hit.chunk.doc_id for hit in answer.hits]

    hit_rank = None
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in case.expect_docs:
            hit_rank = rank
            break

    normalised = _normalise(answer.text)
    found = [f for f in case.expect_facts if _normalise(f) in normalised]
    missing = [f for f in case.expect_facts if _normalise(f) not in normalised]

    # Facts are only a fair test of *generation* when retrieval succeeded.
    if case.expect_docs and hit_rank is None:
        found, missing = [], missing or list(case.expect_facts)

    return CaseResult(
        case=case,
        retrieved_docs=retrieved,
        answer_text=answer.text,
        hit_rank=hit_rank,
        facts_found=found,
        facts_missing=[] if answer.refused and case.should_refuse else missing,
        refused=answer.refused,
        citation_problems=verify_citations(answer),
    )


def evaluate(
    pipeline: RagPipeline, cases: list[EvalCase] | None = None, *, k: int | None = None
) -> EvalReport:
    cases = cases if cases is not None else load_cases()
    return EvalReport([run_case(pipeline, case, k=k) for case in cases])


__all__ = [
    "evaluate", "run_case", "load_cases",
    "EvalCase", "CaseResult", "EvalReport", "DEFAULT_QUESTIONS",
]
