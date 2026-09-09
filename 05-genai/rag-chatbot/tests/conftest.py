"""Shared fixtures: a fake Gemini that never touches the network.

Embeddings are deterministic hashes of the text with a little topical
structure, so retrieval tests are reproducible and free.
"""

import hashlib

import numpy as np
import pytest

from rag.chunking import Document, chunk_documents
from rag.llm import GeminiClient

DIMENSIONS = 64

# Words that pull a vector toward a topic, so semantically related texts land
# near each other without needing a real model.
TOPICS = {
    "retention": 0, "churn": 0, "customer": 0, "campaign": 0, "contact": 0,
    "fraud": 1, "threshold": 1, "precision": 1, "recall": 1, "alert": 1,
    "house": 2, "price": 2, "regression": 2, "area": 2, "neighbourhood": 2,
}


def fake_vector(text: str, dimensions: int = DIMENSIONS) -> list[float]:
    """Deterministic pseudo-embedding with topical clustering."""
    digest = hashlib.sha256(text.lower().encode()).digest()
    base = np.frombuffer(digest * ((dimensions // len(digest)) + 1), dtype=np.uint8)
    vector = base[:dimensions].astype(np.float32) / 255.0 - 0.5
    vector *= 0.35                       # keep the noise small

    lowered = text.lower()
    for word, topic in TOPICS.items():
        if word in lowered:
            vector[topic * 8 : topic * 8 + 8] += 1.0

    norm = float(np.linalg.norm(vector)) or 1.0
    return (vector / norm).tolist()


class FakeTransport:
    """Answers both endpoints from the URL, and records every call."""

    def __init__(self, reply="A grounded answer [1].", fail_times: int = 0):
        # `reply` may be a string, or a callable taking the prompt -- the latter
        # lets a test simulate refusals, wrong answers, or bad citations.
        self.reply = reply
        self.fail_times = fail_times
        self.calls: list[tuple[str, dict]] = []

    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls.append((url, payload))

        if self.fail_times > 0:
            from rag.llm import HTTPError

            self.fail_times -= 1
            raise HTTPError(503, "service unavailable")

        if "batchEmbedContents" in url:
            requests = payload["requests"]
            dimensions = requests[0].get("outputDimensionality", DIMENSIONS)
            return {
                "embeddings": [
                    {"values": fake_vector(r["content"]["parts"][0]["text"], dimensions)}
                    for r in requests
                ]
            }

        prompt = payload["contents"][0]["parts"][0]["text"]
        text = self.reply(prompt) if callable(self.reply) else self.reply
        return {
            "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 25},
        }

    @property
    def embed_calls(self) -> list[dict]:
        return [p for u, p in self.calls if "batchEmbedContents" in u]

    @property
    def generate_calls(self) -> list[dict]:
        return [p for u, p in self.calls if "generateContent" in u]


@pytest.fixture
def transport():
    return FakeTransport()


@pytest.fixture
def client(transport):
    return GeminiClient(transport=transport, sleep=lambda _s: None)


CORPUS = [
    Document(
        doc_id="churn",
        title="Customer Churn",
        text=(
            "# Customer Churn\n\n"
            "## Campaign economics\n\n"
            "Contacting a customer costs 45 pounds including the retention incentive. "
            "The campaign contacted 38.8 percent of the book and the cutoff probability was 0.177. "
            "Ranking by expected value beat ranking by probability alone.\n\n"
            "## Calibration\n\n"
            "Logistic regression won on Brier score. Calibration turned out to be unnecessary "
            "because the model was already well calibrated with a gap of 0.020.\n"
        ),
    ),
    Document(
        doc_id="fraud",
        title="Fraud Detection",
        text=(
            "# Fraud Detection\n\n"
            "## Metrics\n\n"
            "Average precision was 0.3516 for the supervised model and 0.0777 for the "
            "isolation forest. ROC-AUC was flattering at 0.8017 and 0.7585 respectively. "
            "The fraud rate is 0.3 percent so accuracy is meaningless.\n\n"
            "## Threshold\n\n"
            "The cost optimal threshold was 0.1141 with precision 39.5 percent and "
            "recall 33.3 percent. The default of 0.5 gave 100 percent precision and 13.3 recall.\n"
        ),
    ),
    Document(
        doc_id="house",
        title="House Prices",
        text=(
            "# House Prices\n\n"
            "## Model\n\n"
            "Gradient boosting reached a mean absolute error of 13772 and an R squared of 0.934. "
            "The area of the house was the strongest feature followed by neighbourhood.\n\n"
            "## Weakness\n\n"
            "The model underprices the top decile by 13126 on average which is regression "
            "to the mean at the tails.\n"
        ),
    ),
]


@pytest.fixture
def chunks():
    return chunk_documents(CORPUS, max_tokens=90, overlap_tokens=15)


@pytest.fixture
def store(chunks, client):
    from rag.embeddings import embed_chunks
    from rag.store import VectorStore

    return VectorStore(chunks, embed_chunks(chunks, client, dimensions=DIMENSIONS))
