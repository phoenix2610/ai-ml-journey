import pytest

from rag.chunking import (
    Chunk,
    Document,
    chunk_document,
    chunk_documents,
    estimate_tokens,
    load_markdown,
    split_by_size,
    split_sections,
)

MARKDOWN = """\
# Retention Campaign

Some preamble about the whole document.

## Economics

Contacting a customer costs money. The offer is the expensive part.

### Contact cost

It costs 45 per contact, including the incentive actually given.

## Results

The campaign contacted 38.8 percent of the book.
"""


# ------------------------------------------------------------------- tokens


def test_estimate_tokens_grows_with_length():
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)


def test_estimate_tokens_is_never_zero():
    assert estimate_tokens("") >= 1


# ----------------------------------------------------------------- sections


def test_plain_text_is_one_section():
    sections = split_sections("just some prose with no headings")
    assert len(sections) == 1
    assert sections[0].headings == ()


def test_empty_text_yields_nothing():
    assert split_sections("   ") == []


def test_headings_become_sections():
    sections = split_sections(MARKDOWN)
    paths = [s.headings for s in sections]
    assert ("Retention Campaign",) in paths
    assert ("Retention Campaign", "Economics") in paths


def test_heading_path_is_nested():
    sections = split_sections(MARKDOWN)
    deep = [s for s in sections if len(s.headings) == 3]
    assert deep[0].headings == ("Retention Campaign", "Economics", "Contact cost")


def test_path_truncates_when_depth_decreases():
    """'## Results' must drop back to depth 2, not stay under 'Contact cost'."""
    sections = split_sections(MARKDOWN)
    results = [s for s in sections if s.headings and s.headings[-1] == "Results"][0]
    assert results.headings == ("Retention Campaign", "Results")


def test_preamble_before_any_heading_is_kept():
    sections = split_sections("intro text\n\n# Title\n\nbody")
    assert sections[0].headings == ()
    assert "intro text" in sections[0].text


def test_heading_with_no_body_is_skipped():
    sections = split_sections("# Empty\n\n## Real\n\ncontent here")
    assert all(s.text.strip() for s in sections)


# --------------------------------------------------------------- size split


def test_short_text_is_not_split():
    assert split_by_size("Short sentence.", max_tokens=100) == ["Short sentence."]


def test_long_text_is_split():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    assert len(split_by_size(text, max_tokens=60, overlap_tokens=0)) > 1


def test_chunks_respect_the_budget():
    text = " ".join(f"Sentence {i} has some words in it." for i in range(200))
    for chunk in split_by_size(text, max_tokens=60, overlap_tokens=10):
        assert estimate_tokens(chunk) <= 60 * 1.6      # sentence granularity


def test_overlap_repeats_content():
    text = " ".join(f"Sentence number {i} here." for i in range(60))
    with_overlap = split_by_size(text, max_tokens=50, overlap_tokens=25)
    without = split_by_size(text, max_tokens=50, overlap_tokens=0)
    assert len(with_overlap) >= len(without)


def test_a_single_oversized_sentence_is_not_dropped():
    giant = "word " * 800
    chunks = split_by_size(giant.strip(), max_tokens=50)
    assert chunks
    assert "".join(chunks).strip()


def test_no_content_is_lost_without_overlap():
    text = " ".join(f"Fact {i} is true." for i in range(40))
    rejoined = " ".join(split_by_size(text, max_tokens=40, overlap_tokens=0))
    assert "Fact 0" in rejoined and "Fact 39" in rejoined


# ------------------------------------------------------------------ chunks


@pytest.fixture
def doc():
    return Document(doc_id="retention", text=MARKDOWN, title="Retention Campaign")


def test_chunking_produces_chunks(doc):
    assert len(chunk_document(doc)) > 1


def test_chunk_ids_are_unique(doc):
    chunks = chunk_document(doc)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_include_the_doc(doc):
    assert all(c.chunk_id.startswith("retention#") for c in chunk_document(doc))


def test_chunks_carry_their_heading_path(doc):
    chunks = chunk_document(doc)
    deep = [c for c in chunks if "Contact cost" in c.headings][0]
    assert deep.heading_path == "Retention Campaign > Economics > Contact cost"


def test_with_context_prepends_the_heading_path(doc):
    """Why it matters: '45 per contact' alone has no topic words to match on."""
    chunk = [c for c in chunk_document(doc) if "45 per contact" in c.text][0]
    assert chunk.with_context().startswith("Retention Campaign > Economics")
    assert "45 per contact" in chunk.with_context()


def test_with_context_is_plain_text_when_there_are_no_headings():
    chunk = Chunk(chunk_id="x#0", doc_id="x", text="body")
    assert chunk.with_context() == "body"


def test_citation_is_human_readable(doc):
    chunk = [c for c in chunk_document(doc) if c.headings][0]
    assert "Retention Campaign" in chunk.citation


def test_tiny_fragments_are_dropped():
    document = Document(doc_id="d", text="# A\n\nSee below.\n\n## B\n\n" + "real content " * 40)
    assert all(c.tokens >= 12 for c in chunk_document(document))


def test_indices_are_sequential(doc):
    assert [c.index for c in chunk_document(doc)] == list(range(len(chunk_document(doc))))


def test_chunk_documents_handles_many(doc):
    other = Document(doc_id="other", text="# Other\n\n" + "content " * 60)
    chunks = chunk_documents([doc, other])
    assert {c.doc_id for c in chunks} == {"retention", "other"}


def test_empty_document_yields_no_chunks():
    assert chunk_document(Document(doc_id="empty", text="")) == []


# ------------------------------------------------------------------ loading


def test_load_markdown_reads_the_title(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("# The Real Title\n\nbody text here", encoding="utf-8")
    document = load_markdown(path)
    assert document.title == "The Real Title"
    assert document.doc_id == "guide"


def test_load_markdown_falls_back_to_the_filename(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("no heading here", encoding="utf-8")
    assert load_markdown(path).title == "notes"
