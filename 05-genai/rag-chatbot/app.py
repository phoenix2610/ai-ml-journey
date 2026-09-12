#!/usr/bin/env python3
"""Streamlit chat UI with a source panel.

The source panel is the point. A RAG answer is only worth more than a plain
chatbot answer if the user can check it, so every response shows the exact
chunks it was built from, which retriever found them, and any verification
problems. Hiding the sources gives you a chatbot that is merely *probably*
grounded.

    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag.evaluate import evaluate, load_cases
from rag.llm import GeminiClient, LLMError, load_dotenv, load_key
from rag.pipeline import DEFAULT_CACHE, DEFAULT_INDEX, RagConfig, RagPipeline
from rag.embeddings import EmbeddingCache

DOCS_DIR = Path(os.environ.get("RAG_DOCS", "docs"))

st.set_page_config(page_title="Grounded Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Building the index...")
def get_pipeline(retrieval: str, top_k: int) -> RagPipeline:
    """One pipeline per configuration, cached across reruns.

    Streamlit re-executes this script top to bottom on every interaction, so
    without caching each keystroke would re-embed the entire corpus.
    """
    load_dotenv()
    client = GeminiClient(
        api_key=load_key(),
        # Free tier is ~5 generate calls/minute; pacing beats a wall of 429s.
        min_generate_interval=float(os.environ.get("RAG_MIN_INTERVAL", "0")),
    )
    pipeline = RagPipeline(
        client=client,
        config=RagConfig(retrieval=retrieval, top_k=top_k),
        cache=EmbeddingCache(DEFAULT_CACHE),
    )

    if (DEFAULT_INDEX / "chunks.json").exists():
        pipeline.load(DEFAULT_INDEX)
    else:
        if not DOCS_DIR.exists():
            st.error(f"No index and no documents at `{DOCS_DIR}`. Set RAG_DOCS.")
            st.stop()
        pipeline.ingest_directory(DOCS_DIR)
        pipeline.save(DEFAULT_INDEX)

    return pipeline


def render_sources(answer) -> None:
    cited = {c.number for c in answer.citations if c.valid}

    for i, hit in enumerate(answer.hits, start=1):
        was_cited = i in cited
        marker = "✅" if was_cited else "○"
        label = f"{marker} [{i}] {hit.chunk.citation}"

        with st.expander(label, expanded=was_cited):
            ranks = []
            if hit.dense_rank is not None:
                ranks.append(f"dense #{hit.dense_rank}")
            if hit.keyword_rank is not None:
                ranks.append(f"keyword #{hit.keyword_rank}")
            st.caption(
                f"score {hit.score:.4f}"
                + (f" · found by {' + '.join(ranks)}" if ranks else "")
                + f" · {hit.chunk.tokens} tokens"
            )
            st.text(hit.chunk.text)


def main() -> None:
    st.title("📚 Grounded Q&A")
    st.caption(
        "Answers come only from the indexed documents, with citations you can check. "
        "If the sources do not cover it, the model says so."
    )

    with st.sidebar:
        st.header("Retrieval")
        retrieval = st.radio(
            "Strategy",
            ["hybrid", "dense", "keyword"],
            help=(
                "Hybrid fuses dense and keyword ranks. Dense alone is weak on rare "
                "literal tokens like '0.1141'; keyword alone misses paraphrases."
            ),
        )
        top_k = st.slider("Sources retrieved", 1, 10, 5)

        pipeline = get_pipeline(retrieval, top_k)
        st.metric("Chunks indexed", pipeline.chunk_count)

        if st.button("Run evaluation", use_container_width=True):
            with st.spinner("Evaluating against the golden set..."):
                try:
                    report = evaluate(pipeline, load_cases())
                except (LLMError, FileNotFoundError) as exc:
                    st.error(str(exc))
                else:
                    st.metric("Recall@k", f"{report.recall_at_k:.0%}")
                    st.metric("Faithfulness", f"{report.faithfulness:.0%}")
                    st.metric("Refusal accuracy", f"{report.refusal_accuracy:.0%}")
                    st.metric("Overall pass", f"{report.pass_rate:.0%}")

        st.divider()
        usage = pipeline.client.usage
        st.caption(f"{usage.calls} API calls · {usage.total_tokens:,} tokens this session")

    if "history" not in st.session_state:
        st.session_state.history = []

    for entry in st.session_state.history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])

    question = st.chat_input("Ask something about the documents...")
    if not question:
        return

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and answering..."):
            try:
                answer = pipeline.ask(question)
            except LLMError as exc:
                st.error(str(exc))
                return

        st.markdown(answer.text)

        if answer.problems:
            for problem in answer.problems:
                st.warning(f"Citation check: {problem}")

        if answer.refused:
            st.info("The sources did not cover this, so the model declined rather than guessing.")

        if answer.hits:
            st.divider()
            st.caption("Sources — ✅ marks the ones the answer cited")
            render_sources(answer)

    st.session_state.history.append({"role": "assistant", "content": answer.text})


if __name__ == "__main__":
    main()
