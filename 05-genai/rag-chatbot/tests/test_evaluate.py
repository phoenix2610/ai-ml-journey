import re

import pytest

from rag.evaluate import (
    DEFAULT_QUESTIONS,
    EvalCase,
    EvalReport,
    evaluate,
    load_cases,
    run_case,
)
from rag.generate import REFUSAL
from rag.llm import GeminiClient
from rag.pipeline import RagConfig, RagPipeline

from conftest import CORPUS, DIMENSIONS, FakeTransport


def build(reply):
    """A pipeline whose model answers with `reply` (string or prompt->string)."""
    transport = FakeTransport(reply=reply)
    client = GeminiClient(transport=transport, sleep=lambda _s: None)
    pipeline = RagPipeline(
        client=client,
        config=RagConfig(dimensions=DIMENSIONS, max_tokens_per_chunk=90, overlap_tokens=15),
    )
    pipeline.ingest(CORPUS)
    return pipeline


def echo_context(prompt: str) -> str:
    """Answer by quoting the retrieved context -- a 'perfect' generator.

    Isolates retrieval: if a fact is in the context this answer contains it, so
    any failure must be a retrieval failure.
    """
    body = prompt.split("Question:")[0]
    return f"{body.strip()[:1500]} [1]"


# --------------------------------------------------------------- the eval set


def test_shipped_question_set_loads():
    cases = load_cases(DEFAULT_QUESTIONS)
    assert len(cases) >= 10


def test_question_set_has_unanswerable_cases():
    """Without these, refusal accuracy is untested and meaningless."""
    assert sum(1 for c in load_cases(DEFAULT_QUESTIONS) if c.should_refuse) >= 3


def test_graded_cases_declare_expected_documents():
    for case in load_cases(DEFAULT_QUESTIONS):
        assert case.should_refuse or case.expect_docs


def test_missing_eval_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no eval set"):
        load_cases(tmp_path / "absent.yaml")


def test_cases_can_be_loaded_from_a_custom_file(tmp_path):
    path = tmp_path / "q.yaml"
    path.write_text("cases:\n  - question: hi\n    expect_docs: [fraud]\n")
    cases = load_cases(path)
    assert cases[0].question == "hi" and cases[0].expect_docs == ["fraud"]


# ------------------------------------------------------------------ scoring


def test_perfect_generator_scores_well_on_retrieval():
    pipeline = build(echo_context)
    cases = [
        EvalCase(question="What was the cost-optimal threshold?", expect_docs=["fraud"]),
        EvalCase(question="What percentage of the book was contacted?", expect_docs=["churn"]),
    ]
    report = evaluate(pipeline, cases)
    assert report.recall_at_k == 1.0
    assert report.mrr > 0


def test_recall_is_zero_when_the_wrong_document_is_expected():
    pipeline = build(echo_context)
    case = EvalCase(question="fraud threshold precision recall", expect_docs=["nonexistent"])
    assert evaluate(pipeline, [case]).recall_at_k == 0.0


def test_mrr_rewards_a_higher_rank():
    pipeline = build(echo_context)
    result = run_case(pipeline, EvalCase(question="fraud threshold", expect_docs=["fraud"]))
    assert result.hit_rank is not None
    assert result.reciprocal_rank == pytest.approx(1 / result.hit_rank)


def test_facts_are_matched_loosely():
    """'38.8 percent' in the corpus must satisfy an expected fact of '38.8%'."""
    pipeline = build(echo_context)
    case = EvalCase(
        question="What percentage of the book did the campaign contact?",
        expect_docs=["churn"],
        expect_facts=["38.8%"],
    )
    assert evaluate(pipeline, [case]).faithfulness == 1.0


def test_a_generation_failure_is_diagnosed_separately():
    """Retrieval succeeds, the model ignores it -- must not read as retrieval failure."""
    pipeline = build("I have no idea, but probably 0.99 [1].")
    case = EvalCase(
        question="What was the cost-optimal threshold?",
        expect_docs=["fraud"],
        expect_facts=["0.1141"],
    )
    result = evaluate(pipeline, [case]).results[0]
    assert result.retrieval_ok is True
    assert result.facts_ok is False
    assert result.diagnosis.startswith("GENERATION failed")


def test_a_retrieval_failure_is_diagnosed_separately():
    pipeline = build(echo_context)
    case = EvalCase(question="anything", expect_docs=["not-in-corpus"], expect_facts=["x"])
    result = evaluate(pipeline, [case]).results[0]
    assert result.retrieval_ok is False
    assert "RETRIEVAL failed" in result.diagnosis


# ----------------------------------------------------------------- refusals


def test_correct_refusal_passes():
    pipeline = build(REFUSAL)
    result = evaluate(pipeline, [EvalCase(question="capital of France?", should_refuse=True)]).results[0]
    assert result.refused and result.refusal_ok and result.passed


def test_failing_to_refuse_is_caught():
    pipeline = build("The capital of France is Paris [1].")
    report = evaluate(pipeline, [EvalCase(question="capital of France?", should_refuse=True)])
    assert report.refusal_accuracy == 0.0
    assert "should have refused" in report.results[0].diagnosis


def test_refusing_when_it_should_not_is_caught():
    pipeline = build(REFUSAL)
    case = EvalCase(question="fraud threshold", expect_docs=["fraud"], expect_facts=["0.1141"])
    report = evaluate(pipeline, [case])
    assert report.refusal_accuracy == 0.0
    assert "refused wrongly" in report.results[0].diagnosis


# ---------------------------------------------------------------- citations


def test_invented_citation_is_penalised():
    pipeline = build("A claim citing nothing real [99].")
    report = evaluate(pipeline, [EvalCase(question="fraud threshold", expect_docs=["fraud"])])
    assert report.citation_validity == 0.0


def test_valid_citation_passes_verification():
    pipeline = build("A grounded claim [1].")
    report = evaluate(pipeline, [EvalCase(question="fraud threshold", expect_docs=["fraud"])])
    assert report.citation_validity == 1.0


# ------------------------------------------------------------------ report


def test_report_renders_every_metric():
    pipeline = build(echo_context)
    text = str(evaluate(pipeline, [EvalCase(question="fraud", expect_docs=["fraud"])]))
    for label in ("recall@k", "MRR", "faithfulness", "refusal accuracy", "citation validity"):
        assert label in text


def test_failures_lists_only_failing_cases():
    pipeline = build("wrong answer with no citation")
    cases = [
        EvalCase(question="fraud threshold", expect_docs=["fraud"], expect_facts=["0.1141"]),
        EvalCase(question="churn cutoff", expect_docs=["churn"], expect_facts=["0.177"]),
    ]
    report = evaluate(pipeline, cases)
    assert len(report.failures()) == 2
    assert report.pass_rate == 0.0


def test_empty_report_does_not_divide_by_zero():
    report = EvalReport([])
    assert report.pass_rate == 0.0 and report.recall_at_k == 0.0 and report.mrr == 0.0


def test_full_shipped_eval_runs_end_to_end():
    """Mechanics check: every shipped case executes and produces a diagnosis."""
    pipeline = build(echo_context)
    report = evaluate(pipeline, load_cases(DEFAULT_QUESTIONS))
    assert report.n >= 10
    assert all(isinstance(r.diagnosis, str) for r in report.results)
