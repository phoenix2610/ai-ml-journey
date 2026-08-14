import pytest

from scraper.fetcher import Cache, FetchError, Fetcher, RateLimiter, Response


class FakeTransport:
    """Replays a scripted list of responses (or exceptions) in order."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []

    def get(self, url, headers, timeout):
        self.calls.append((url, headers))
        item = self.script.pop(0) if self.script else Response(url, 200, "ok")
        if isinstance(item, Exception):
            raise item
        return Response(url, item.status, item.text, item.headers)


def r(status=200, text="ok"):
    return Response("http://x", status, text)


@pytest.fixture
def no_sleep():
    return lambda _s: None


def test_successful_fetch(no_sleep):
    f = Fetcher(FakeTransport(r(200, "hello")), sleep=no_sleep)
    assert f.get("http://example.com").text == "hello"
    assert f.stats["fetched"] == 1


def test_user_agent_is_sent(no_sleep):
    t = FakeTransport(r())
    Fetcher(t, user_agent="mybot/1.0", sleep=no_sleep).get("http://example.com")
    assert t.calls[0][1]["User-Agent"] == "mybot/1.0"


def test_retries_on_500_then_succeeds(no_sleep):
    f = Fetcher(FakeTransport(r(500), r(503), r(200, "finally")), retries=3, sleep=no_sleep)
    assert f.get("http://example.com").text == "finally"
    assert f.stats["retried"] == 2


def test_gives_up_after_the_retry_budget(no_sleep):
    f = Fetcher(FakeTransport(r(500), r(500), r(500), r(500)), retries=2, sleep=no_sleep)
    with pytest.raises(FetchError, match="HTTP 500"):
        f.get("http://example.com")
    assert f.stats["failed"] == 1


def test_404_is_not_retried(no_sleep):
    t = FakeTransport(r(404), r(200))
    f = Fetcher(t, retries=3, sleep=no_sleep)
    with pytest.raises(FetchError, match="404"):
        f.get("http://example.com")
    assert len(t.calls) == 1


def test_429_is_retried(no_sleep):
    f = Fetcher(FakeTransport(r(429), r(200, "ok")), retries=2, sleep=no_sleep)
    assert f.get("http://example.com").ok


def test_transport_exception_is_retried_then_wrapped(no_sleep):
    f = Fetcher(FakeTransport(TimeoutError("slow"), TimeoutError("slow")), retries=1, sleep=no_sleep)
    with pytest.raises(FetchError, match="TimeoutError"):
        f.get("http://example.com")


def test_backoff_grows_and_is_capped():
    assert Fetcher._backoff(0) < Fetcher._backoff(3)
    assert Fetcher._backoff(20) <= 30.5


# ------------------------------------------------------------------ rate limit


def test_first_request_to_a_host_does_not_wait():
    slept = []
    limiter = RateLimiter(1.0, sleep=slept.append)
    assert limiter.wait("http://a.com", now=lambda: 0.0) == 0.0
    assert slept == []


def test_second_request_to_the_same_host_waits():
    limiter = RateLimiter(1.0, jitter=0.0, sleep=lambda _s: None)
    clock = iter([0.0, 0.2, 0.2])
    limiter.wait("http://a.com", now=lambda: next(clock))
    assert limiter.wait("http://a.com", now=lambda: next(clock)) == pytest.approx(0.8)


def test_different_hosts_do_not_block_each_other():
    limiter = RateLimiter(1.0, jitter=0.0, sleep=lambda _s: None)
    limiter.wait("http://a.com", now=lambda: 0.0)
    assert limiter.wait("http://b.com", now=lambda: 0.1) == 0.0


# ----------------------------------------------------------------------- cache


def test_cache_round_trip(tmp_path):
    cache = Cache(tmp_path)
    cache.put(Response("http://x", 200, "body", {"a": "b"}))
    hit = cache.get("http://x")
    assert hit.text == "body"
    assert hit.from_cache is True


def test_cache_miss_returns_none(tmp_path):
    assert Cache(tmp_path).get("http://never-seen") is None


def test_errors_are_not_cached(tmp_path):
    cache = Cache(tmp_path)
    cache.put(Response("http://x", 500, "boom"))
    assert cache.get("http://x") is None


def test_disabled_cache_is_a_no_op():
    cache = Cache(None)
    cache.put(Response("http://x", 200, "body"))
    assert cache.get("http://x") is None


def test_fetcher_serves_from_cache_without_touching_the_network(tmp_path, no_sleep):
    t = FakeTransport(r(200, "once"))
    f = Fetcher(t, cache_dir=tmp_path, sleep=no_sleep)
    f.get("http://example.com")
    second = f.get("http://example.com")
    assert second.from_cache is True
    assert len(t.calls) == 1
    assert f.stats["cached"] == 1


def test_corrupt_cache_entry_is_treated_as_a_miss(tmp_path):
    cache = Cache(tmp_path)
    cache.put(Response("http://x", 200, "body"))
    next(tmp_path.glob("*.json")).write_text("{not json")
    assert cache.get("http://x") is None
