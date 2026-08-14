"""HTTP fetching: retries, per-host rate limiting, and an on-disk cache.

The transport is injected rather than hard-coded to ``requests``. That is what
lets the whole test suite run offline against a fake, and it means swapping in
``httpx`` later touches one class instead of the entire crawler.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

DEFAULT_UA = "journey-scraper/0.1 (+https://github.com/; educational project)"

# Worth retrying: transient server problems and explicit rate limiting.
RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class Response:
    url: str
    status: int
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    from_cache: bool = False

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class Transport(Protocol):
    """The minimum a fetcher needs from an HTTP library."""

    def get(self, url: str, headers: dict[str, str], timeout: float) -> Response: ...


class RequestsTransport:
    """Real network access. Imported lazily so tests never need `requests`."""

    def __init__(self) -> None:
        import requests

        self._session = requests.Session()

    def get(self, url: str, headers: dict[str, str], timeout: float) -> Response:
        r = self._session.get(url, headers=headers, timeout=timeout)
        return Response(
            url=r.url, status=r.status_code, text=r.text, headers=dict(r.headers)
        )


class RateLimiter:
    """Enforce a minimum gap between requests, tracked per host.

    Being polite to one host should not slow down a crawl that spans twenty,
    so the clock is kept per-netloc rather than globally.
    """

    def __init__(self, delay: float = 1.0, *, jitter: float = 0.3, sleep=time.sleep) -> None:
        self.delay = delay
        self.jitter = jitter
        self._sleep = sleep
        self._last: dict[str, float] = {}

    def wait(self, url: str, *, now=time.monotonic) -> float:
        host = urlparse(url).netloc
        current = now()
        last = self._last.get(host)

        slept = 0.0
        if last is not None:
            # Jitter stops a crawl from hitting a host on an exact metronome.
            target = self.delay + random.uniform(0, self.jitter)
            remaining = target - (current - last)
            if remaining > 0:
                self._sleep(remaining)
                slept = remaining
                current = now()

        self._last[host] = current
        return slept


class Cache:
    """Content-addressed response cache. Re-running a crawl costs nothing."""

    def __init__(self, directory: Path | str | None) -> None:
        self.directory = Path(directory) if directory else None
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, url: str) -> Path:
        assert self.directory is not None
        return self.directory / f"{hashlib.sha256(url.encode()).hexdigest()[:20]}.json"

    def get(self, url: str) -> Response | None:
        if not self.directory:
            return None
        path = self._path(url)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return Response(**raw, from_cache=True)

    def put(self, response: Response) -> None:
        if not self.directory or not response.ok:
            return
        self._path(response.url).write_text(
            json.dumps(
                {
                    "url": response.url,
                    "status": response.status,
                    "text": response.text,
                    "headers": response.headers,
                }
            ),
            encoding="utf-8",
        )


class FetchError(RuntimeError):
    """A URL that could not be retrieved after every retry was spent."""


class Fetcher:
    def __init__(
        self,
        transport: Transport | None = None,
        *,
        user_agent: str = DEFAULT_UA,
        delay: float = 1.0,
        timeout: float = 15.0,
        retries: int = 3,
        cache_dir: Path | str | None = None,
        sleep=time.sleep,
    ) -> None:
        self.transport = transport or RequestsTransport()
        self.user_agent = user_agent
        self.timeout = timeout
        self.retries = retries
        self.limiter = RateLimiter(delay, sleep=sleep)
        self.cache = Cache(cache_dir)
        self._sleep = sleep
        self.stats = {"fetched": 0, "cached": 0, "retried": 0, "failed": 0}

    def get(self, url: str) -> Response:
        cached = self.cache.get(url)
        if cached is not None:
            self.stats["cached"] += 1
            return cached

        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        last_error = "unknown error"

        for attempt in range(self.retries + 1):
            self.limiter.wait(url)
            try:
                response = self.transport.get(url, headers, self.timeout)
            except Exception as exc:  # transport-specific; normalised here
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.ok:
                    self.stats["fetched"] += 1
                    self.cache.put(response)
                    return response
                last_error = f"HTTP {response.status}"
                if response.status not in RETRY_STATUS:
                    break  # 404 will still be 404 next time

            if attempt < self.retries:
                self.stats["retried"] += 1
                self._sleep(self._backoff(attempt))

        self.stats["failed"] += 1
        raise FetchError(f"{url}: {last_error}")

    @staticmethod
    def _backoff(attempt: int) -> float:
        """Exponential backoff, capped, with jitter to avoid retry stampedes."""
        return min(2**attempt, 30) + random.uniform(0, 0.5)


__all__ = ["Fetcher", "Response", "FetchError", "RateLimiter", "Cache", "Transport", "DEFAULT_UA"]
