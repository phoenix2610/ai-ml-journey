import pytest

from rag.chunking import Document
from rag.generate import (
    REFUSAL,
    Answer,
    answer_question,
    build_prompt,
    format_context,
    parse_citations,
    verify_citations,
)
from rag.llm import GeminiClient, LLMError
from rag.pipeline import RagConfig, RagPipeline

from conftest import CORPUS, DIMENSIONS, FakeTransport


@pytest.fixture
def config():
    return RagConfig(dimensions=DIMENSIONS, max_tokens_per_chunk=90, overlap_tokens=15)


@pytest.fixture
def pipeline(client, config):
    p = RagPipeline(client=client, config=config)
    p.ingest(CORPUS)
    return p


# -------------------------------------------------------------------- prompt


def test_context_numbers_the_sources(store, client, config):
    p = RagPipeline(client=client, config=config, store=store)
    hits = p.retrieve("fraud threshold", k=3)
    context = format_context(hits)
    assert "[1]" in context and "[2]" in context and "[3]" in context


def test_prompt_contains_the_question(store, client, config):
    p = RagPipeline(client=client, config=config, store=store)
    prompt = build_prompt("what was the cutoff?", p.retrieve("cutoff", k=2))
    assert "what was the cutoff?" in prompt


def test_prompt_with_no_hits_demands_a_refusal():
    assert "I don't know" in build_prompt("anything", [])


# ----------------------------------------------------------------- citations


def test_citations_resolve_to_sources(store, client, config):
    p = RagPipeline(client=client, config=config, store=store)
    hits = p.retrieve("fraud", k=3)
    citations = parse_citations("Claim one [1] and claim two [3].", hits)
    assert [c.number for c in citations] == [1, 3]
    assert all(c.valid for c in citations)


def test_out_of_range_citation_is_invalid(store, client, config):
    p = RagPipeline(client=client, config=config, store=store)
    hits = p.retrieve("fraud", k=2)
    citations = parse_citations("Fabricated [7].", hits)
    assert citations[0].valid is False


def test_multiple_markers_are_all_parsed(store, client, config):
    p = RagPipeline(client=client, config=config, store=store)
    hits = p.retrieve("fraud", k=3)
    assert len(parse_citations("A [1][2] and B [3].", hits)) == 3


def test_text_with_no_citations_parses_to_nothing():
    assert parse_citations("No markers at all.", []) == []


# -------------------------------------------------------------- verification


def _answer(text, hits):
    return Answer(question="q", text=text, hits=hits, citations=parse_citations(text, hits))


def test_well_formed_answer_has_no_problems(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=3)
    assert verify_citations(_answer("Grounded claim [1].", hits)) == []


def test_hallucinated_citation_is_caught(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=2)
    problems = verify_citations(_answer("Invented [9].", hits))
    assert problems and "never supplied" in problems[0]


def test_uncited_claim_is_caught(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=3)
    problems = verify_citations(_answer("A confident claim with no marker.", hits))
    assert problems and "cites nothing" in problems[0]


def test_a_refusal_is_not_a_problem(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=3)
    assert verify_citations(_answer(REFUSAL, hits)) == []


# ------------------------------------------------------------------ generate


def test_answer_carries_its_hits(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=3)
    answer = answer_question("what is the threshold?", hits, client)
    assert len(answer.hits) == 3
    assert answer.text


def test_no_hits_means_an_automatic_refusal(client):
    answer = answer_question("anything", [], client)
    assert answer.refused
    assert answer.text == REFUSAL


def test_refusal_needs_no_api_call(client, transport):
    answer_question("anything", [], client)
    assert transport.generate_calls == []


def test_system_prompt_forbids_outside_knowledge(store, client, config, transport):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=2)
    answer_question("q", hits, client)
    system = transport.generate_calls[0]["systemInstruction"]["parts"][0]["text"]
    assert "ONLY" in system
    assert "I don't know" in system


def test_temperature_is_low_by_default(store, client, config, transport):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=2)
    answer_question("q", hits, client)
    assert transport.generate_calls[0]["generationConfig"]["temperature"] <= 0.2


def test_sources_used_lists_only_cited_ones(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=4)
    answer = _answer("Only this one [2].", hits)
    assert len(answer.sources_used) == 1
    assert answer.sources_used[0] is hits[1]


def test_formatted_output_includes_a_source_list(store, client, config):
    hits = RagPipeline(client=client, config=config, store=store).retrieve("fraud", k=3)
    text = _answer("Claim [1].", hits).formatted()
    assert "Sources:" in text and "[1]" in text


def test_generation_failure_is_wrapped(store):
    # Retrieve with keyword search so the failure can only come from generation.
    hits = store.keyword_search("fraud precision", k=2)
    failing = GeminiClient(transport=FakeTransport(fail_times=99), sleep=lambda _s: None)
    with pytest.raises(LLMError, match="could not generate"):
        answer_question("q", hits, failing)


def test_transient_failures_are_retried(store):
    """503 is retryable, so two failures then success must still produce an answer."""
    hits = store.keyword_search("fraud precision", k=2)
    flaky = GeminiClient(transport=FakeTransport(fail_times=2), sleep=lambda _s: None)
    assert answer_question("q", hits, flaky).text


# ------------------------------------------------------------------ pipeline


def test_ingest_builds_a_store(pipeline):
    assert pipeline.chunk_count > 0


def test_ingesting_nothing_raises(client, config):
    with pytest.raises(ValueError, match="no chunks"):
        RagPipeline(client=client, config=config).ingest([Document(doc_id="x", text="")])


def test_retrieve_before_ingest_raises(client, config):
    with pytest.raises(RuntimeError, match="nothing indexed"):
        RagPipeline(client=client, config=config).retrieve("q")


def test_ask_returns_a_grounded_answer(pipeline):
    answer = pipeline.ask("what was the cost optimal threshold?")
    assert answer.hits
    assert answer.text


def test_ask_attaches_verification(pipeline):
    assert isinstance(pipeline.ask("what is the churn cutoff?").problems, list)


@pytest.mark.parametrize("mode", ["hybrid", "dense", "keyword"])
def test_every_retrieval_mode_works(client, config, mode):
    config.retrieval = mode
    p = RagPipeline(client=client, config=config)
    p.ingest(CORPUS)
    assert p.retrieve("fraud precision recall", k=3)


def test_keyword_mode_makes_no_embedding_call_at_query_time(client, config, transport):
    config.retrieval = "keyword"
    p = RagPipeline(client=client, config=config)
    p.ingest(CORPUS)
    before = len(transport.embed_calls)
    p.retrieve("precision", k=2)
    assert len(transport.embed_calls) == before


def test_save_and_load_round_trip(pipeline, client, config, tmp_path):
    pipeline.save(tmp_path / "idx")
    fresh = RagPipeline(client=client, config=config)
    fresh.load(tmp_path / "idx")
    assert fresh.chunk_count == pipeline.chunk_count
    assert fresh.retrieve("fraud", k=2)


def test_ingest_directory(client, config, tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n" + "content about fraud detection " * 20)
    (tmp_path / "b.md").write_text("# B\n\n" + "content about churn retention " * 20)
    p = RagPipeline(client=client, config=config)
    p.ingest_directory(tmp_path)
    assert p.chunk_count > 0


def test_ingest_empty_directory_raises(client, config, tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matching"):
        RagPipeline(client=client, config=config).ingest_directory(tmp_path)
