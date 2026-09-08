"""Gemini client: embeddings and generation, behind an injectable transport.

Same discipline as the weather app. `Transport` is a protocol, not a hard
dependency on `requests`, so the entire test suite runs offline against a fake
and never spends a token or needs a key. That is not a testing nicety -- an LLM
test suite that hits the real API is slow, costs money, and is
non-deterministic, which means in practice nobody runs it.

Only the REST API is used, so there is no SDK to keep in step. Two endpoints:

    POST /v1beta/models/{model}:generateContent
    POST /v1beta/models/{model}:batchEmbedContents
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

CHAT_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-2"
EMBED_DIMENSIONS = 768

# Transient. Anything else is a bug in the request and retrying just repeats it.
RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """Any failure to get a usable response, phrased for a human."""


class MissingKey(LLMError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def add(self, prompt: int, output: int) -> None:
        self.prompt_tokens += prompt
        self.output_tokens += output
        self.calls += 1


class Transport(Protocol):
    """The minimum the client needs from an HTTP library."""

    def post_json(self, url: str, payload: dict, timeout: float) -> dict: ...


class RequestsTransport:
    """Real network access. Imported lazily so tests never need `requests`."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()

    def post_json(self, url: str, payload: dict, timeout: float) -> dict:
        import requests

        try:
            response = self._session.post(url, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            raise LLMError(f"could not reach Gemini: {exc}") from None

        if response.status_code >= 400:
            try:
                body = response.json()
            except json.JSONDecodeError:
                body = {}
            raise HTTPError(response.status_code, _error_message(response), body)
        try:
            return response.json()
        except json.JSONDecodeError:
            raise LLMError("Gemini returned a non-JSON response") from None


RETRY_DELAY = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


class HTTPError(LLMError):
    def __init__(self, status: int, message: str, body: dict | None = None) -> None:
        self.status = status
        self.body = body or {}
        super().__init__(f"Gemini HTTP {status}: {message}")

    @property
    def retry_after(self) -> float | None:
        """How long the API asked us to wait, if it said.

        A 429 from Gemini carries the exact delay, both in `error.details` as a
        RetryInfo and in the message text. Honouring it beats exponential
        backoff, which either waits too long or hammers the limit again.
        """
        for detail in self.body.get("error", {}).get("details", []) or []:
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    pass
        match = RETRY_DELAY.search(str(self))
        return float(match.group(1)) if match else None


def _error_message(response) -> str:
    try:
        return response.json().get("error", {}).get("message", response.text[:200])
    except Exception:
        return response.text[:200]


KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY")


def load_key(explicit: str | None = None) -> str:
    """Resolve the API key from an argument or the environment.

    Matched **case-insensitively**, because environment variables on Linux are
    case-sensitive and `Gemini_API_KEY` is an easy and completely invisible way
    to lose an afternoon -- `os.environ["GEMINI_API_KEY"]` simply reports that
    nothing is set.
    """
    if explicit:
        return explicit

    wanted = {name.upper() for name in KEY_NAMES}
    for name, value in os.environ.items():
        if name.strip().upper() in wanted and value.strip():
            return value.strip()

    raise MissingKey(
        "no API key found. Set GEMINI_API_KEY in the environment or a .env file:\n"
        "  GEMINI_API_KEY=your-key-here\n"
        "Both the name's case and spaces around '=' are tolerated by "
        "rag.llm.load_dotenv, but not by most other .env readers."
    )


def load_dotenv(path: str | os.PathLike = ".env") -> dict[str, str]:
    """Minimal .env reader, tolerant of the formatting people actually write.

    Strips whitespace around both name and value, ignores comments and blank
    lines, and unquotes values. Deliberately does not overwrite variables that
    are already set in the environment.
    """
    loaded: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except OSError:
        return loaded

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if not name:
            continue
        loaded[name] = value
        os.environ.setdefault(name, value)
    return loaded


@dataclass
class GeminiClient:
    api_key: str | None = None
    transport: Transport | None = None
    chat_model: str = CHAT_MODEL
    embed_model: str = EMBED_MODEL
    timeout: float = 60.0
    retries: int = 3
    # Free tier allows ~5 generate calls/minute. Pacing client-side turns a
    # burst of 429s into a slower run that actually completes.
    min_generate_interval: float = 0.0
    sleep: Any = time.sleep
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        # A fake transport means no key is needed -- that is what makes the
        # test suite runnable with nothing configured.
        if self.transport is None:
            self.api_key = load_key(self.api_key)
            self.transport = RequestsTransport()
        else:
            self.api_key = self.api_key or "test-key"

    # ------------------------------------------------------------- plumbing

    def _url(self, model: str, method: str) -> str:
        return f"{BASE_URL}/models/{model}:{method}?key={self.api_key}"

    def _post(self, url: str, payload: dict) -> dict:
        last: Exception | None = None

        for attempt in range(self.retries + 1):
            try:
                return self.transport.post_json(url, payload, self.timeout)
            except HTTPError as exc:
                last = exc
                if exc.status not in RETRY_STATUS:
                    raise
                wait = exc.retry_after
            except LLMError as exc:
                last = exc
                wait = None

            if attempt < self.retries:
                # The server's own hint beats a guess; fall back to backoff.
                self.sleep(
                    wait + 0.5 if wait is not None
                    else min(2**attempt, 20) + random.uniform(0, 0.4)
                )

        raise LLMError(f"gave up after {self.retries + 1} attempts: {last}")

    # ----------------------------------------------------------- generation

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 1024,
    ) -> str:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        self._pace()
        data = self._post(self._url(self.chat_model, "generateContent"), payload)
        return self._extract_text(data)

    _last_generate: float = 0.0

    def _pace(self) -> None:
        if self.min_generate_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_generate
        if self._last_generate and elapsed < self.min_generate_interval:
            self.sleep(self.min_generate_interval - elapsed)
        self._last_generate = time.monotonic()

    def _extract_text(self, data: dict) -> str:
        meta = data.get("usageMetadata") or {}
        self.usage.add(
            int(meta.get("promptTokenCount", 0)), int(meta.get("candidatesTokenCount", 0))
        )

        candidates = data.get("candidates") or []
        if not candidates:
            # A safety block returns no candidates at all, with the reason here.
            blocked = (data.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(
                f"Gemini returned no candidates (blocked: {blocked})" if blocked
                else "Gemini returned no candidates"
            )

        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts).strip()

        if not text:
            reason = candidates[0].get("finishReason", "unknown")
            raise LLMError(f"Gemini returned empty text (finishReason: {reason})")
        return text

    # ----------------------------------------------------------- embeddings

    def embed(
        self,
        texts: list[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        dimensions: int = EMBED_DIMENSIONS,
        batch_size: int = 100,
    ) -> list[list[float]]:
        """Embed a list of texts.

        `task_type` matters and is easy to get wrong: documents must be embedded
        as RETRIEVAL_DOCUMENT and queries as RETRIEVAL_QUERY. Using one for both
        measurably degrades retrieval, because the model places them in
        deliberately different regions of the space.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.embed_model}",
                        "content": {"parts": [{"text": text}]},
                        "taskType": task_type,
                        "outputDimensionality": dimensions,
                    }
                    for text in chunk
                ]
            }
            data = self._post(self._url(self.embed_model, "batchEmbedContents"), payload)
            batch = data.get("embeddings")
            if batch is None or len(batch) != len(chunk):
                raise LLMError(
                    f"expected {len(chunk)} embeddings, got "
                    f"{0 if batch is None else len(batch)}"
                )
            vectors.extend([e["values"] for e in batch])

        return vectors

    def embed_query(self, text: str, *, dimensions: int = EMBED_DIMENSIONS) -> list[float]:
        return self.embed([text], task_type="RETRIEVAL_QUERY", dimensions=dimensions)[0]


__all__ = [
    "GeminiClient", "Transport", "RequestsTransport", "Usage",
    "LLMError", "HTTPError", "MissingKey",
    "load_key", "load_dotenv",
    "CHAT_MODEL", "EMBED_MODEL", "EMBED_DIMENSIONS", "BASE_URL",
]
