# RAG Chatbot

Grounded question-answering over your own documents, with **citations you can
check** and an **evaluation harness that proves it works**.

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-key-here
export RAG_DOCS=./docs
streamlit run app.py        # http://localhost:8501
pytest -q                   # 111 tests, no network, no key needed
```

See [DEPLOY.md](./DEPLOY.md) for Docker and Streamlit Cloud.

## Measured, not demoed

A chatbot that answers three questions convincingly proves nothing. The golden
set in [`eval/questions.yaml`](eval/questions.yaml) is 12 hand-labelled cases,
run against the **live Gemini API**:

```
  12 cases

    recall@k            100.0%   right source reached the context
    MRR                  1.000   how highly it ranked
    faithfulness        100.0%   required facts present
    refusal accuracy    100.0%   declined when it should
    citation validity   100.0%   no invented sources
    overall pass        100.0%
```

Three of those cases are **unanswerable on purpose**. The hardest asks for "the
fraud model's F1 score on the validation set" — the corpus discusses fraud
metrics and thresholds at length but never states an F1 or a validation split.
Everything about the question looks answerable. It correctly returned:

```
I don't know based on the provided sources.
```

A system that never refuses is not grounded, it is confident.

## Retrieval and generation are measured separately

This is the metric design decision that matters. Both failures look identical
from outside — a wrong answer — but they need opposite fixes, and you cannot
fix a retrieval problem by editing the prompt.

```
RETRIEVAL failed -- right source never reached the context
GENERATION failed -- context had it, answer missed ['0.1141']
```

`recall@k` and `MRR` grade retrieval; fact presence grades generation. Facts are
only counted as a generation failure when retrieval actually succeeded, so the
two numbers stay independent.

## Chunking: structure first, size second

The decision that most determines whether RAG works, and the one that gets the
least attention. Two failure modes bracket it:

- **Too large** — the embedding averages several topics, matches everything
  vaguely and nothing precisely.
- **Too small** — *"It costs £45 per contact"* is unretrievable once separated
  from what "it" is.

Markdown already carries the author's own view of where topics begin, so
splitting follows headings first and falls back to sentence-boundary packing
only inside an oversized section.

Each chunk keeps its **heading path**, and embeds with it prepended:

```
Customer Churn > Campaign economics

Contacting a customer costs 45 pounds including the retention incentive...
```

Without that, a deep section retrieves badly — the words identifying the topic
appear only in an ancestor heading. It doubles as the citation shown to the user.

## Hybrid retrieval, and why

Dense embeddings capture meaning but are weak on rare literal tokens — error
codes, model names, `0.1141`. Keyword search is the opposite. Neither is
reliable alone, so results are fused with **Reciprocal Rank Fusion**:

```
score = Σ 1 / (60 + rank)
```

RRF combines *ranks*, not scores. Cosine similarity and BM25 live on
incomparable scales, and any direct weighting needs tuning that does not
transfer between corpora. There is a test asserting hybrid recovers an exact
numeric literal that dense search alone smears over.

## Citations are structural

The model emits `[1]`, `[2]`; those markers are parsed and resolved back to real
chunks. A marker outside the supplied range resolves to `None`:

```python
verify_citations(answer)
# ["cites source(s) [9] that were never supplied (3 were)"]
```

So a hallucinated citation is a **detectable fact**, not a plausible footnote.
The UI marks which sources the answer actually cited, and warns when
verification fails.

## Task types are not optional

Gemini places documents and queries in deliberately different regions of the
embedding space:

```python
embed_chunks(...)   # taskType=RETRIEVAL_DOCUMENT
embed_query(...)    # taskType=RETRIEVAL_QUERY
```

Using one for both is a **silent** failure — nothing errors, retrieval just
quietly gets worse. Two separate functions make it impossible to pick wrong,
and two tests pin which type each one sends.

## Two bugs worth keeping

**An empty cache was falsy.** `EmbeddingCache` defined `__len__` but not
`__bool__`, so Python fell back to length for truthiness and `if cache:` was
`False` for a *fresh* cache — it silently skipped every store and save, and
could never warm up. Now `__bool__` returns `True` and every call site uses
`is not None`.

**Rate limits found the hard way.** A 12-question eval run died at HTTP 429:
the free tier allows ~5 generation calls/minute. The fix was to honour the
server's own `retryDelay` hint instead of guessing with exponential backoff,
plus optional client-side pacing (`RAG_MIN_INTERVAL`). See
[DEPLOY.md](./DEPLOY.md#rate-limits--read-this-before-demoing).

## Tests run offline

111 tests, no network, no API key. `Transport` is a protocol, so the suite runs
against a fake with deterministic pseudo-embeddings that cluster by topic. An
LLM test suite that hits the real API is slow, costs money, and is
non-deterministic — which means in practice nobody runs it.

## Layout

```
rag/
├── llm.py         Gemini REST client, retries, rate-limit handling
├── chunking.py    structure-aware splitting; heading paths
├── embeddings.py  document/query task types; the embedding cache
├── store.py       dense + BM25 + RRF fusion; persistence
├── generate.py    grounded prompting, citation parsing and verification
├── pipeline.py    ingest -> retrieve -> answer
└── evaluate.py    recall@k, MRR, faithfulness, refusal accuracy
app.py             Streamlit chat with a source panel
eval/questions.yaml the golden set
```
